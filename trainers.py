import logging
import os
import random
from collections import defaultdict
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

import wandb

from .data import build_dataset
from .losses import build_loss, edl_ce_loss_balanced, softplus_evidence
from .models import TextBERT
from .utils import (
    build_optimizer,
    compute_metrics,
    display_cm,
    flatten_config,
    fmt_metric,
    get_class_counts,
    get_perm,
    get_remix_y,
    log_uncertainty_kde,
    save_metrics_to_csv,
    t_sne_vis,
)


def setup_distributed(use_distributed: bool):
    """
    Setup compatible with three cases:

    1. DDP with torchrun:
       torchrun --nproc_per_node=N main.py

    2. DDP with Slurm direct srun:
       #SBATCH --ntasks-per-node=N
       #SBATCH --gres=gpu:N
       srun python main.py

    3. Non-distributed mode:
       #SBATCH --ntasks-per-node=1
       #SBATCH --gres=gpu:N
       srun python main.py
       -> possible DataParallel later if torch.cuda.device_count() > 1

    Important safety rule:
    If use_distributed=False but Slurm/torchrun launched multiple processes,
    we raise an error to avoid duplicated training.
    """

    # ------------------------------------------------------------
    # Detect whether we are in a multi-process launch
    # ------------------------------------------------------------
    torchrun_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    slurm_world_size = int(os.environ.get("SLURM_NTASKS", "1"))

    launched_with_torchrun = (
        "WORLD_SIZE" in os.environ
        and "RANK" in os.environ
        and "LOCAL_RANK" in os.environ
        and torchrun_world_size > 1
    )

    launched_with_slurm_ddp = (
        "SLURM_NTASKS" in os.environ
        and "SLURM_PROCID" in os.environ
        and "SLURM_LOCALID" in os.environ
        and slurm_world_size > 1
    )

    multi_process_launch = launched_with_torchrun or launched_with_slurm_ddp

    # ------------------------------------------------------------
    # Case A: user does NOT want DDP
    # ------------------------------------------------------------
    if not use_distributed:
        if multi_process_launch:
            raise RuntimeError(
                "train.distributed=false, but multiple processes were launched. "
                "This would duplicate the same training job on each Slurm task/process.\n"
                f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
                f"RANK={os.environ.get('RANK')}, "
                f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
                f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}, "
                f"SLURM_PROCID={os.environ.get('SLURM_PROCID')}, "
                f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}.\n"
                "Fix: either set train.distributed=true for DDP, or launch with "
                "#SBATCH --ntasks-per-node=1 if you want DataParallel/non-distributed mode."
            )

        # Non-distributed mode. DataParallel may be activated later in Trainer
        # if torch.cuda.device_count() > 1.
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        return False, 0, 0, 1

    # ------------------------------------------------------------
    # Case B: user wants DDP, but process group already initialized
    # ------------------------------------------------------------
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        local_rank = int(
            os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size

    # ------------------------------------------------------------
    # Case C: DDP via torchrun
    # ------------------------------------------------------------
    if launched_with_torchrun:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

    # ------------------------------------------------------------
    # Case D: DDP via Slurm direct srun
    # ------------------------------------------------------------
    elif launched_with_slurm_ddp:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])

        # Normalize env vars so the rest of the code can use torchrun-like names.
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

    # ------------------------------------------------------------
    # Case E: train.distributed=true but launch is not distributed
    # ------------------------------------------------------------
    else:
        raise RuntimeError(
            "train.distributed=true, but no distributed launch was detected. "
            "For Slurm DDP, use #SBATCH --ntasks-per-node=N and srun python main.py. "
            "For torchrun DDP, use torchrun --nproc_per_node=N main.py.\n"
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
            f"RANK={os.environ.get('RANK')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
            f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}, "
            f"SLURM_PROCID={os.environ.get('SLURM_PROCID')}, "
            f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}."
        )

    # ------------------------------------------------------------
    # Initialize DDP
    # ------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("DDP with NCCL requires CUDA, but CUDA is not available.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
        device_id=device,
    )

    return True, rank, local_rank, world_size


MODEL_NAME = "bert-large-uncased"

dsdir = os.getenv("DSDIR")
if dsdir is None:
    raise EnvironmentError("DSDIR doesn't exist")

root_path = os.path.join(dsdir, "HuggingFace_Models")
model_path = os.path.join(root_path, MODEL_NAME)

if not os.path.isdir(model_path):
    raise FileNotFoundError(f"The model doesn't exist in this folder : {model_path!r}")


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.use_unc = self.cfg.train.uncertainty

        self.distributed, self.rank, self.local_rank, self.world_size = (
            setup_distributed(use_distributed=bool(self.cfg.train.distributed))
        )

        if torch.cuda.is_available():
            if self.distributed:
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        random.seed(self.cfg.train.seed)
        np.random.seed(self.cfg.train.seed)
        torch.manual_seed(self.cfg.train.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.train.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.base_path = os.path.join(
            self.cfg.output.model_dir,
            self.cfg.dataset.name,
            f"IR_{self.cfg.dataset.ir}",
            self.cfg.experiment,
            f"seed_{self.cfg.train.seed}",
        )
        os.makedirs(self.base_path, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        self.model_save_path = os.path.join(self.base_path, "best_model.pt")
        self.log_file = os.path.join(self.base_path, "log_metrics.csv")

        self.cm_path = os.path.join(self.base_path, "confusion_matrices")
        os.makedirs(self.cm_path, exist_ok=True)

        self.kde_path = os.path.join(self.base_path, "kde_uncertainty")
        os.makedirs(self.kde_path, exist_ok=True)

        self.tsne_path = os.path.join(self.base_path, "tsne_plots")
        os.makedirs(self.tsne_path, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, add_prefix_space=True
        )
        self.logger.info(f"Tokenizer loaded : {self.tokenizer.__class__.__name__}")

        self.train_loader, self.val_loader, self.test_loader = build_dataset(
            cfg, self.tokenizer, self.logger
        )

        self.n_classes = cfg.dataset.num_cls
        self.n_pc = get_class_counts(self.train_loader, self.n_classes)

        torch.cuda.empty_cache()

        self.model = self._build_model()
        self.model = self.model.to(self.device)

        if self.distributed:
            self.logger.info(
                f"[DDP] rank={self.rank}, local_rank={self.local_rank}, "
                f"world_size={self.world_size}, device={self.device}"
            )

            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        elif getattr(self.cfg.train, "distributed", False):
            raise RuntimeError(
                "train.distributed=true but DDP was not initialized. "
                f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}, "
                f"SLURM_PROCID={os.environ.get('SLURM_PROCID')}, "
                f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}, "
                f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
                f"RANK={os.environ.get('RANK')}, "
                f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
            )
        elif torch.cuda.device_count() > 1:
            self.logger.info(f"[DataParallel] Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        else:
            self.logger.info(f"[Single GPU/CPU] device={self.device}")

        if self.is_main_process():
            print("=" * 80)
            print(f"Rank: {getattr(self, 'rank', 'NA')}")
            print(f"Local rank: {getattr(self, 'local_rank', 'NA')}")
            print(f"World size: {getattr(self, 'world_size', 'NA')}")
            print(f"Device: {self.device}")
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            print(f"Model class: {self.model.__class__.__name__}")
            print(f"Is DDP: {isinstance(self.model, DDP)}")
            print(f"Is DataParallel: {isinstance(self.model, nn.DataParallel)}")
            print(f"Train sampler: {self.train_loader.sampler.__class__.__name__}")
            print("=" * 80, flush=True)
            self.logger.info(f"Model loaded : {self.model.__class__.__name__}")
            self.logger.info(f"Training-set label distribution : {self.n_pc}")
            self.logger.info(
                f"Validation-set label distribution : {get_class_counts(self.val_loader, self.n_classes)}"
            )
            self.logger.info(
                f"Test-set label distribution : {get_class_counts(self.test_loader, self.n_classes)}"
            )

        self.criterion = build_loss(cfg, self.n_pc, self.device)
        self.test_criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = build_optimizer(
            cfg, self.model, self.train_loader
        )

        self.best_val_f1 = 0
        self.previous_beliefs = None
        self.switch_epoch = self.cfg.train.switch_epoch

    # ------------------------------------------------------------------
    def _build_model(self):
        return TextBERT(
            pretrained_model=model_path,
            num_class=self.n_classes,
            fine_tune=self.cfg.train.finetune,
            dropout=self.cfg.model.dropout,
            freeze=self.cfg.train.freeze,
            use_unc=self.use_unc,
        )

    def is_distributed(self):
        return dist.is_available() and dist.is_initialized()

    def is_main_process(self):
        return (not self.is_distributed()) or dist.get_rank() == 0

    def get_model_without_wrapper(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def ddp_sum(self, value):
        """
        Somme une valeur scalaire sur tous les processus DDP.
        En mode non distribué, retourne simplement la valeur.
        """
        if not self.is_distributed():
            return value

        tensor = torch.tensor(float(value), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def get_model_to_save(self):
        """
        Retourne le vrai modèle, sans wrapper DataParallel ou DDP.
        """
        return self.model.module if hasattr(self.model, "module") else self.model

    def _step(self, batch, epoch):
        y1 = batch.pop("label")  # [B]
        mu_prior = float(getattr(self.cfg.loss, "mu_prior", 0.0))

        def _get_beta_vec(dtype):
            nonlocal mu_prior
            if hasattr(self.model, "module") and hasattr(self.model.module, "beta_raw"):
                return (softplus_evidence(self.model.module.beta_raw)).to(
                    dtype=dtype
                )  # [C]
            if hasattr(self.model, "beta_raw"):
                return (softplus_evidence(self.model.beta_raw)).to(dtype=dtype)

            mu_prior = 0.0
            return torch.ones(self.n_classes, device=self.device, dtype=dtype)

        if self.cfg.mix.method == "none":
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs, _ = self.model(**batch)

            if self.use_unc:
                logits = outputs.float()
                evidences = softplus_evidence(logits)  # [B,K] >=0
                beta_vec = _get_beta_vec(evidences.dtype)  # [C]
                alphas = evidences + beta_vec.unsqueeze(0)  # [B,C]
                S = torch.sum(alphas, dim=1, keepdim=True)  # [B,1]
                probs = (alphas / S).clamp(min=1e-10)  # [B,K]
                preds = torch.argmax(probs, dim=1)  # [B]

                focal_flag = getattr(self.cfg.loss, "type", "CE") == "FL"
                """gamma = getattr(self.cfg.loss, "fl_gamma", 2.0) if focal_flag else 0.0"""

                loss, ce_loss, kl_div, _Lp = edl_ce_loss_balanced(
                    alphas,
                    y1,
                    epoch,
                    self.cfg.loss.annealing_step,
                    self.n_classes,
                    self.n_pc,
                    beta=beta_vec,
                    mu=mu_prior,
                    focal=focal_flag,
                    reduction="mean",  # balanced (1/C)*sum_i L_i/N_{y_i}
                    device=self.device,
                )
            else:
                loss = self.criterion(outputs.float(), y1)
                loss = loss.mean()
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                ce_loss = kl_div = None

            match = preds.eq(y1).sum()  # a revoir cette .sum()
        elif self.cfg.mix.method == "remix" or self.cfg.mix.method == "mixup":
            x1, att1 = batch["input_ids"], batch["attention_mask"]
            index = get_perm(x1)
            x2, y2, att2 = x1[index], y1[index], att1[index]

            lam_x = np.random.beta(self.cfg.mix.alpha, self.cfg.mix.alpha)
            lam_y = lam_x
            if self.cfg.mix.method == "remix":
                lam_y = get_remix_y(
                    y1,
                    y2,
                    lam_x,
                    self.n_pc,
                    self.cfg.mix.k_majority,
                    self.cfg.mix.tau,
                    self.device,
                )

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model.module.forward_mix_encoder(
                    x1, att1, x2, att2, lam_x
                )

            if self.use_unc:
                logits = outputs.float()
                evidences = softplus_evidence(logits)
                beta_vec = _get_beta_vec(evidences.dtype)  # [C]
                alphas = evidences + beta_vec.unsqueeze(0)  # [B,C]
                S = torch.sum(alphas, dim=1, keepdim=True)
                probs = (alphas / S).clamp(min=1e-10)
                preds = torch.argmax(probs, dim=1)

                focal_flag = getattr(self.cfg.loss, "type", "CE") == "FL"
                """gamma = getattr(self.cfg.loss, "fl_gamma", 2.0) if focal_flag else 0.0

                self.logger.info(
                    f"The focal loss is activated for mixup or remix: {focal_flag}"
                )"""

                loss1, ce_loss1, kl_div1, lp1 = edl_ce_loss_balanced(
                    alphas,
                    y1,
                    epoch,
                    self.cfg.loss.annealing_step,
                    self.n_classes,
                    self.n_pc,
                    beta=beta_vec,
                    mu=0.0,
                    focal=focal_flag,
                    reduction="none",
                    device=self.device,
                )
                loss2, ce_loss2, kl_div2, _ = edl_ce_loss_balanced(
                    alphas,
                    y2,
                    epoch,
                    self.cfg.loss.annealing_step,
                    self.n_classes,
                    self.n_pc,
                    beta=beta_vec,
                    mu=0.0,
                    focal=focal_flag,
                    reduction="none",
                    device=self.device,
                )
                # class pooling weights (soft-label compatible)
                '''spc = torch.as_tensor(
                    self.n_pc, device=self.device, dtype=torch.float32
                )  # [C]
                w1 = 1.0 / spc[y1].clamp_min(1e-12)  # [B]
                w2 = 1.0 / spc[y2].clamp_min(1e-12)  # [B]

                loss_vec = w1 * lam_y * loss1 + w2 * (1.0 - lam_y) * loss2
                ce_vec = w1 * lam_y * ce_loss1 + w2 * (1.0 - lam_y) * ce_loss2
                kl_vec = w1 * lam_y * kl_div1 + w2 * (1.0 - lam_y) * kl_div2

                C = float(self.n_classes)
                ce_loss = (ce_vec).sum() / C
                kl_div = (kl_vec).sum() / C
                loss = ((loss_vec).sum() / C) + mu_prior * lp1'''
                spc = torch.as_tensor(
                    self.n_pc, device=self.device, dtype=loss1.dtype
                ).clamp_min(1e-12)  # [C]

                N = spc.sum()
                C = float(self.n_classes)

                # Normalized class-balanced weights: N / (C * N_c)
                w_all = N / (C * spc)  # [C]

                w1 = w_all[y1]  # [B]
                w2 = w_all[y2]  # [B]

                # Ensure lam_y is a tensor on the correct device
                if not torch.is_tensor(lam_y):
                    lam_y_t = torch.full_like(loss1, float(lam_y))
                else:
                    lam_y_t = lam_y.to(device=self.device, dtype=loss1.dtype)

                loss_vec = lam_y_t * w1 * loss1 + (1.0 - lam_y_t) * w2 * loss2
                ce_vec = lam_y_t * w1 * ce_loss1 + (1.0 - lam_y_t) * w2 * ce_loss2
                kl_vec = lam_y_t * w1 * kl_div1 + (1.0 - lam_y_t) * w2 * kl_div2

                loss = loss_vec.mean() + mu_prior * lp1
                ce_loss = ce_vec.mean()
                kl_div = kl_vec.mean()
            else:
                print("criterion :", self.criterion)
                loss1, loss2 = (
                    self.criterion(outputs.float(), y1),
                    self.criterion(outputs.float(), y2),
                )
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                loss = (lam_y * loss1 + (1 - lam_y) * loss2).mean()

            match = (
                lam_y * preds.eq(y1).float() + (1 - lam_y) * preds.eq(y2).float()
            ).sum()
        else:
            raise ValueError("The method name is not correct")

        return (loss, match, ce_loss, kl_div) if self.use_unc else (loss, match)

    def train_one_epoch(self, epoch):
        # Important en DDP : change le shuffle à chaque époque
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        self.model.train()

        tr_loss, tr_ce_loss, tr_kl_div = 0.0, 0.0, 0.0
        n_samples, correct, n_steps = 0, 0, 0
        steps_in_epoch = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, 1):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            y = batch["label"]
            bs = y.size(0)

            if self.use_unc:
                loss, match, ce_loss, kl_div = self._step(batch, epoch)

                tr_loss += float(loss.item())
                tr_ce_loss += float(ce_loss.item())
                tr_kl_div += float(kl_div.item())
                correct += float(match.item())
                n_steps += 1
            else:
                loss, match = self._step(batch, epoch)

                tr_loss += float(loss.item()) * bs
                correct += float(match.item())

            n_samples += bs

            # --- backward / opti ----
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # ---- logging / validation ----
            if (step % self.cfg.train.eval_interval == 0) or (step == steps_in_epoch):
                global_step = step + epoch * steps_in_epoch

                # ------------------------------------------------------------
                # Synchronisation des métriques train entre tous les processus
                # ------------------------------------------------------------
                if self.is_distributed():
                    global_tr_loss = self.ddp_sum(tr_loss)
                    global_tr_ce_loss = self.ddp_sum(tr_ce_loss)
                    global_tr_kl_div = self.ddp_sum(tr_kl_div)
                    global_n_samples = self.ddp_sum(n_samples)
                    global_correct = self.ddp_sum(correct)
                    global_n_steps = self.ddp_sum(n_steps)
                else:
                    global_tr_loss = tr_loss
                    global_tr_ce_loss = tr_ce_loss
                    global_tr_kl_div = tr_kl_div
                    global_n_samples = n_samples
                    global_correct = correct
                    global_n_steps = n_steps

                acc = global_correct / max(1, global_n_samples)

                if self.use_unc:
                    avg_loss = global_tr_loss / max(1, global_n_steps)
                    avg_ce_loss = global_tr_ce_loss / max(1, global_n_steps)
                    avg_kl_div = global_tr_kl_div / max(1, global_n_steps)
                else:
                    avg_loss = global_tr_loss / max(1, global_n_samples)

                # ------------------------------------------------------------
                # Validation sur TOUS les ranks
                # ------------------------------------------------------------
                val_metrics = self.evaluate(
                    self.val_loader,
                    epoch,
                    step,
                    test=False,
                )

                # Important pour Mix-Belief / curriculum :
                # tous les ranks doivent garder previous_beliefs
                if (
                    self.cfg.train.curriculum
                    and self.use_unc
                    and epoch >= self.switch_epoch - 1
                ):
                    self.previous_beliefs = val_metrics["per_class"]

                # ------------------------------------------------------------
                # Logs, wandb et sauvegarde uniquement sur rank 0
                # ------------------------------------------------------------
                if self.is_main_process():
                    self.logger.info(
                        f"[epoch {epoch} step {step}] "
                        f"Train loss:{avg_loss:.4f} | "
                        f"Train acc:{acc:.4f} | "
                        f"LR: {current_lr:.8f}"
                    )

                    self.logger.info(
                        f"[epoch {epoch} step {step}] "
                        f"Loss={val_metrics['loss']:.4f} | "
                        f"ECE={val_metrics['ece']:.4f} | "
                        f"F1={val_metrics['f1']:.4f} | "
                        f"GM={val_metrics['gm']:.4f} | "
                        f"ACC={val_metrics['acc']:.4f}"
                    )

                    if (
                        self.cfg.train.curriculum
                        and self.use_unc
                        and epoch >= self.switch_epoch - 1
                    ):
                        self.logger.info(
                            f"Saving belief vectors at epoch {epoch} for Mix-Belief..."
                        )

                    log_dict = {
                        "train/loss": avg_loss,
                        "train/accuracy": acc,
                        "val/loss": val_metrics["loss"],
                        "val/acc": val_metrics.get("acc", 0.0),
                        "val/balanced_acc": val_metrics.get("balanced_acc", 0.0),
                        "val/gm": val_metrics.get("gm", 0.0),
                        "val/macro_f1": val_metrics.get("macro_f1", 0.0),
                        "val/macro_precision": val_metrics.get("macro_prec", 0.0),
                        "val/macro_recall": val_metrics.get("macro_rec", 0.0),
                        "val/weighted_f1": val_metrics.get("weighted_f1", 0.0),
                        "val/tail_f1": val_metrics.get("tail_f1", 0.0),
                        "val/tail_recall": val_metrics.get("tail_recall", 0.0),
                        "val/head_tail_f1_gap": val_metrics.get(
                            "head_tail_f1_gap", 0.0
                        ),
                        "val/ece": val_metrics.get("ece", 0.0),
                        "val/nll": val_metrics.get("nll", 0.0),
                        "val/brier": val_metrics.get("brier", 0.0),
                        "val/mean_u": val_metrics.get("mean_u", 0.0),
                        "val/mean_u_correct": val_metrics.get("mean_u_correct", 0.0),
                        "val/mean_u_wrong": val_metrics.get("mean_u_wrong", 0.0),
                        "val/auroc_error": val_metrics.get("auroc_error", 0.0),
                        "val/aupr_error": val_metrics.get("aupr_error", 0.0),
                        "val/step": global_step,
                        "val/epoch": epoch,
                        "train/lr": current_lr,
                    }

                    if self.use_unc:
                        log_dict["beta/beta_sum"] = val_metrics.get("beta_sum", 0)
                        log_dict["beta/beta_mean"] = val_metrics.get("beta_mean", 0)
                        log_dict["beta/beta_std"] = val_metrics.get("beta_std", 0)

                        log_dict["train/ce_loss"] = avg_ce_loss
                        log_dict["train/kl_div"] = avg_kl_div
                        log_dict["val/ce_loss"] = val_metrics.get("ce_loss", 0)
                        log_dict["val/kl_div"] = val_metrics.get("kl_div", 0)

                        per_class_extra = val_metrics.get("per_class_extra", {})
                        for c, stats in per_class_extra.items():
                            log_dict[f"beta/cls_{c}"] = val_metrics.get(
                                f"beta_{c}", 0.0
                            )

                    wandb.log(log_dict)

                    save_metrics_to_csv(
                        self.log_file,
                        "val",
                        val_metrics,
                        epoch,
                        step,
                        self.cfg.train.seed,
                    )

                    if val_metrics["f1"] > self.best_val_f1:
                        model_to_save = self.get_model_to_save()

                        torch.save(
                            model_to_save.state_dict(),
                            self.model_save_path,
                        )

                        print(
                            f"############### Where the model is saved : "
                            f"{self.model_save_path} ##############"
                        )

                        self.best_val_f1 = val_metrics["f1"]
                # reset counters train interval sur tous les ranks
                tr_loss, tr_ce_loss, tr_kl_div = 0.0, 0.0, 0.0
                n_samples, correct, n_steps = 0, 0, 0

                self.model.train()

    # Modified
    def evaluate(
        self,
        loader,
        epoch=None,
        step=None,
        test=False,
    ):
        # En DDP, on évalue avec le modèle non wrappé.
        # Important si evaluate() est appelée seulement sur rank 0.
        model_for_eval = self.get_model_without_wrapper()
        model_for_eval.eval()

        beta_vec = None

        if self.use_unc:
            if hasattr(model_for_eval, "beta_raw"):
                beta_raw = model_for_eval.beta_raw
            else:
                raise ValueError("beta_raw is not learnable !")

            beta_vec = softplus_evidence(beta_raw).detach() + 1e-6  # [K]
            beta_vec = beta_vec.to(self.device)

        sum_loss, sum_ce, sum_kl = 0.0, 0.0, 0.0
        sum_loss = 0.0
        n_samples = 0
        n_steps = 0

        all_e_true = []
        all_sum_e = []
        all_sum_S = []

        all_preds, all_labels = [], []
        all_uncertainties, all_beliefs, all_probs = [], [], []
        all_embeddings = []

        all_e_pred = []
        all_e_gap = []

        class_beliefs = defaultdict(list)
        class_uncertainties = defaultdict(list)

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
                y = batch.pop("label")

                logits, embeddings = model_for_eval(**batch)  # [B,K]

                if self.use_unc:
                    evidences = softplus_evidence(logits)  # [B,K] >=0
                    alphas = evidences + beta_vec.unsqueeze(0)  # [B,K]
                    S = torch.sum(alphas, dim=1, keepdim=True)  # [B,1]
                    probs = (alphas / S).clamp(min=1e-10)  # [B,K]
                    preds = torch.argmax(probs, dim=1)  # [B]

                    sum_e = evidences.sum(dim=1)  # [B]
                    all_sum_e.extend(sum_e.detach().cpu().tolist())

                    all_sum_S.extend(S.squeeze(-1).detach().cpu().tolist())  # [B]

                    loss, ce_loss, kl_div, Lp = edl_ce_loss_balanced(
                        alphas,
                        y,
                        epoch,
                        self.cfg.loss.annealing_step,
                        self.n_classes,
                        self.n_pc,
                        beta=beta_vec,
                        mu=self.cfg.loss.mu_prior,
                        focal=False,
                        reduction="mean",
                        device=self.device,
                    )

                    # accumulate weighted sums
                    sum_loss += float(loss.item())
                    sum_ce += float(ce_loss.item())
                    sum_kl += float(kl_div.item())
                    n_steps += 1

                    beta_sum = float(beta_vec.sum().item())
                    u = (beta_sum / S).squeeze(-1)  # [B]
                    belief = evidences / S  # [B,K]

                    bs = y.size(0)

                    # all_uncertainties.extend(u.cpu().squeeze().tolist())
                    all_uncertainties.extend(u.cpu().tolist())  # list of floats
                    all_beliefs.extend(belief.cpu().tolist())  # list of [k]
                    all_probs.append(probs.detach().cpu())  # [B,K]

                    e_true = evidences.gather(1, y.unsqueeze(1)).squeeze(1)  # [B]
                    all_e_true.extend(e_true.detach().cpu().tolist())
                    e_pred = evidences.gather(1, preds.unsqueeze(1)).squeeze(1)  # [B]
                    all_e_pred.extend(e_pred.detach().cpu().tolist())
                    # Gap d'évidence
                    e_gap = e_pred - e_true  # [B]
                    all_e_gap.extend(e_gap.detach().cpu().tolist())

                    for i in range(bs):
                        a = int(y[i].item())

                        class_beliefs[a].append(belief[i].cpu().numpy())
                        class_uncertainties[a].append(float(u[i].item()))
                else:
                    loss = self.test_criterion(logits, y)
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    bs = y.size(0)
                    n_samples += bs
                    sum_loss += loss.item() * bs
                    all_probs.append(probs.detach().cpu())

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())
                if test:
                    all_embeddings.append(embeddings.detach().cpu())

        all_probs = torch.cat(all_probs, dim=0)  # shape [N, K]

        metrics = compute_metrics(all_labels, all_preds, all_probs, self.n_classes)
        if not self.use_unc:
            metrics["loss"] = sum_loss / max(1, n_samples)

        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        matches = all_preds_np == all_labels_np  # shape: [N] boolean array

        if self.use_unc:
            beta_cpu = beta_vec.detach().cpu().numpy()
            metrics["beta_sum"] = float(beta_cpu.sum())
            metrics["beta_mean"] = float(beta_cpu.mean())
            metrics["beta_std"] = float(beta_cpu.std())
            for k in range(self.n_classes):
                metrics[f"beta_{k}"] = float(beta_cpu[k])

            metrics["loss"] = sum_loss / max(1, n_steps)
            u_np = np.asarray(all_uncertainties, dtype=np.float32)  # [N]
            metrics["mean_u"] = float(u_np.mean()) if u_np.size > 0 else 0.0
            metrics["ce_loss"] = sum_ce / max(1, n_steps)
            metrics["kl_div"] = sum_kl / max(1, n_steps)

            all_e_true_np = np.asarray(all_e_true, dtype=np.float32)  # [N]
            metrics["mean_ev_true"] = (
                float(all_e_true_np.mean()) if all_e_true_np.size > 0 else 0.0
            )

            succ = matches
            fail = ~matches

            per_class_metrics = {}
            for c in range(self.n_classes):
                class_b = np.array(
                    class_beliefs[c], dtype=np.float32
                )  # shape: (N_c, C)
                class_u = np.array(
                    class_uncertainties[c], dtype=np.float32
                )  # shape: (N_c,)
                if len(class_b) > 0:
                    per_class_metrics[c] = {
                        "mean_belief": class_b.mean(
                            axis=0
                        ).tolist(),  # liste de masses moyennes par classe
                        "mean_uncertainty": float(class_u.mean()),
                    }
                else:
                    per_class_metrics[c] = {
                        "mean_belief": [0.0] * self.n_classes,
                        "mean_uncertainty": 0.0,
                    }

            metrics["per_class"] = per_class_metrics
            if self.is_main_process():
                for c in range(self.n_classes):
                    m = metrics["per_class"][c]
                    belief_vec_str = ", ".join([f"{v:.4f}" for v in m["mean_belief"]])
                    self.logger.info(
                        f"Val Class {c} Mean‑u:{m['mean_uncertainty']:.4f} | Belief‑vec:[{belief_vec_str}]"
                    )
        if test and self.is_main_process():
            base_log = (
                f"Test acc={fmt_metric(metrics, 'acc')}, "
                f"Test balanced_acc={fmt_metric(metrics, 'balanced_acc')}, "
                f"Test gm={fmt_metric(metrics, 'gm')}, "
                f"Test macro_f1={fmt_metric(metrics, 'macro_f1')}, "
                f"Test macro_precision={fmt_metric(metrics, 'macro_prec')}, "
                f"Test macro_recall={fmt_metric(metrics, 'macro_rec')}, "
                f" Test weighted_f1={fmt_metric(metrics, 'weighted_f1')}, "
                f"Test tail_f1={fmt_metric(metrics, 'tail_f1')}, "
                f"Test tail_recall={fmt_metric(metrics, 'tail_recall')}, "
                f"Test head_tail_f1_gap={fmt_metric(metrics, 'head_tail_f1_gap')}, "
                f"Test ece={fmt_metric(metrics, 'ece')}, "
                f"Test nll={fmt_metric(metrics, 'nll')}, "
                f"Test brier={fmt_metric(metrics, 'brier')}, "
                f"Test mean_u={fmt_metric(metrics, 'mean_u')}, "
                f"Test mean_u_correct={fmt_metric(metrics, 'mean_u_correct')}, "
                f"Test mean_u_wrong={fmt_metric(metrics, 'mean_u_wrong')}, "
                f"Test auroc_error={fmt_metric(metrics, 'auroc_error')}, "
                f"Test aupr_error={fmt_metric(metrics, 'aupr_error')}, "
                f"Test loss={fmt_metric(metrics, 'loss')}"
            )

            self.logger.info(base_log)

        if self.is_main_process():
            display_cm(
                self.cm_path,
                wandb,
                metrics["cm"],
                self.cfg.experiment,
                self.cfg.train.seed,
                epoch,
                step,
                data="Test" if test else "Val",
            )

            if self.use_unc:
                log_uncertainty_kde(
                    u_np[succ],
                    u_np[fail],
                    wandb,
                    step=step,
                    set="Test" if test else "Val",
                    save_dir=self.kde_path,
                )

            # t-SNE uniquement pour le test
            if test and len(all_embeddings) > 0:
                all_embeddings = torch.cat(all_embeddings, dim=0)
                all_embeddings_np = all_embeddings.squeeze().cpu().numpy()

                t_sne_vis(
                    all_embeddings_np,
                    all_labels_np,
                    self.cfg.train.seed,
                    self.cfg.experiment,
                    self.tsne_path,
                )
        return metrics

    def run(self):
        if self.is_main_process():
            wandb.init(
                project="uncertainty-v5",
                config=flatten_config(self.cfg),
                name=self.cfg.experiment + f"_run{self.cfg.train.seed}",
                reinit=True,
            )

        for e in range(self.cfg.train.epochs):
            self.train_one_epoch(e)

        if self.is_distributed():
            dist.barrier()

        model_to_load = self.get_model_to_save()
        model_to_load.load_state_dict(
            torch.load(self.model_save_path, map_location=self.device)
        )

        test_metrics = self.evaluate(self.test_loader, test=True)

        if self.is_main_process():
            self.logger.info("Training complete!")
            print("Best Validation f1-score macro: ", self.best_val_f1)

            save_metrics_to_csv(
                self.log_file,
                "test",
                test_metrics,
                seed=self.cfg.train.seed,
            )

            wandb.finish()

        return test_metrics
