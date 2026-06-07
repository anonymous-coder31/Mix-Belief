import csv
import math
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import torch
import torch.nn as nn
from imblearn.metrics import geometric_mean_score
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils import resample
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import CalibrationError

from .models import TextBERT


def add_weight_noise_textbert(model, sigma=0.02, last_k_encoder_layers=4):
    """
    Ajoute un bruit gaussien léger sur :
      - la tête de classification
      - les last_k_encoder_layers dernières couches de l'encoder BERT

    model : peut être TextBERT ou DataParallel(TextBERT)
    sigma : intensité du bruit (en proportion de l'écart type des poids)
    last_k_encoder_layers : nombre de couches encoder à perturber à partir du haut
    """
    # Déballer DataParallel si nécessaire
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        base_model = model.module
    else:
        base_model = model

    # On s'assure qu'on est bien sur ton TextBERT
    if not isinstance(base_model, TextBERT):
        raise TypeError(f"Expected TextBERT, got {type(base_model)}")

    bert = base_model.bert

    with torch.no_grad():
        # 1) Bruit sur la tête de classification
        for name, p in base_model.classifier.named_parameters():
            if not p.requires_grad:
                continue
            if p.data.numel() <= 1:
                continue
            std = p.data.std().clamp(min=1e-8)
            noise = torch.randn_like(p) * (sigma * std)
            p.add_(noise)

        # 2) Bruit sur les dernières couches de l'encoder BERT
        n_layers = bert.config.num_hidden_layers  # 24 pour bert-large-uncased
        start = max(0, n_layers - last_k_encoder_layers)

        for layer_idx in range(start, n_layers):
            layer = bert.encoder.layer[layer_idx]
            for name, p in layer.named_parameters():
                # Respecte ton freeze : on ne touche pas aux params requires_grad=False
                if not p.requires_grad:
                    continue
                if p.data.numel() <= 1:
                    continue
                std = p.data.std().clamp(min=1e-8)
                noise = torch.randn_like(p) * (sigma * std)
                p.add_(noise)

    print(
        f"[add_weight_noise_textbert] Added noise (sigma={sigma}) on classifier "
        f"+ encoder layers {start}..{n_layers - 1}"
    )


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_counts(loader, num_cls):
    # Initialize a counter
    class_counts = Counter()

    # Iterate through the data loader
    for batch in loader:
        labels = batch.pop("label")
        # Update the counter with labels
        class_counts.update(labels.tolist())

    counts_array = [class_counts[i] for i in range(num_cls)]

    return counts_array


"""def build_optimizer(cfg, model, loader):
    base_lr = cfg.train.base_lr
    weight_decay = cfg.train.weight_decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=weight_decay
    )

    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    num_training_steps = len(loader) * cfg.train.epochs
    num_warmup_steps = int(num_training_steps * cfg.train.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler"""


def build_optimizer(cfg, model, loader):
    base_lr = cfg.train.base_lr
    weight_decay = cfg.train.weight_decay

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.train.epochs
    min_scale = float(getattr(cfg.train, "min_lr_scale", 0.0))

    def linear_warmup(step, warmup_steps):
        return step / float(warmup_steps) if step < warmup_steps else 1.0

    def cosine_decay(step, total_phase_steps, warmup_steps):
        if total_phase_steps <= warmup_steps:
            return 1.0
        t = (step - warmup_steps) / float(total_phase_steps - warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    if cfg.train.curriculum:
        switch_step = steps_per_epoch * int(cfg.train.switch_epoch)

        phaseA_steps = max(1, switch_step)
        warmupA = max(1, int(phaseA_steps * float(cfg.train.warmup_ratio)))

        phaseB_steps = max(1, total_steps - switch_step)
        warmupB = max(
            1,
            int(
                phaseB_steps
                * float(getattr(cfg.train, "warmup_ratio2", cfg.train.warmup_ratio))
            ),
        )

        mix_scale = float(getattr(cfg.train, "mix_lr_scale", 0.3))

        def lr_lambda(global_step):
            if global_step < switch_step:
                w = linear_warmup(global_step, warmupA)
                d = cosine_decay(global_step, phaseA_steps, warmupA)
                scale = w * d
                return max(scale, min_scale)

            step_in_phase_b = global_step - switch_step
            w = linear_warmup(step_in_phase_b, warmupB)
            d = cosine_decay(step_in_phase_b, phaseB_steps, warmupB)
            scale = mix_scale * w * d
            return max(scale, mix_scale * min_scale)

    else:
        warmup_steps = max(1, int(total_steps * float(cfg.train.warmup_ratio)))

        def lr_lambda(global_step):
            w = linear_warmup(global_step, warmup_steps)
            d = cosine_decay(global_step, total_steps, warmup_steps)
            scale = w * d
            return max(scale, min_scale)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def get_perm(x):
    """get random permutation"""
    batch_size = x.size()[0]
    device = x.device
    index = torch.randperm(batch_size).to(device)
    return index


def fmt_metric(metrics, key, default=None):
    value = metrics.get(key, default)

    if value is None:
        return "NA"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def compute_multiclass_brier_score(y_true, probs, n_classes):
    """
    Multiclass Brier score:
    mean over samples of sum_c (p_c - y_c)^2
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    y_onehot = np.eye(n_classes)[y_true]
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

    return float(brier)


def get_head_medium_tail_classes(class_counts):
    """
    class_counts: dict or pandas Series
    Example:
        {0: 3000, 1: 2000, 2: 500, 3: 100}

    Returns:
        head_classes, medium_classes, tail_classes
    """

    if hasattr(class_counts, "to_dict"):
        class_counts = class_counts.to_dict()

    sorted_classes = sorted(
        class_counts.keys(), key=lambda c: class_counts[c], reverse=True
    )
    n_classes = len(sorted_classes)

    if n_classes <= 3:
        head_classes = [sorted_classes[0]]
        tail_classes = [sorted_classes[-1]]
        medium_classes = sorted_classes[1:-1]
    else:
        n_head = max(1, int(round(0.30 * n_classes)))
        n_tail = max(1, int(round(0.30 * n_classes)))

        head_classes = sorted_classes[:n_head]
        tail_classes = sorted_classes[-n_tail:]
        medium_classes = sorted_classes[n_head:-n_tail]

    return head_classes, medium_classes, tail_classes


def safe_mean(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    if len(values) == 0:
        return None
    return float(np.mean(values))


def compute_group_metrics(per_class_values, classes):
    """
    per_class_values: array, one value per class
    classes: list of class indices
    """
    if classes is None or len(classes) == 0:
        return None

    values = [per_class_values[c] for c in classes]
    return safe_mean(values)


def compute_metrics(
    y_true,
    y_pred,
    probs,
    n_classes,
    uncertainty=None,
    class_counts_for_groups=None,
    ece_bins=10,
):
    """
    y_true: list/array of true labels
    y_pred: list/array of predicted labels
    probs: numpy array or torch tensor of shape [N, C]
    uncertainty: optional array of uncertainty scores, e.g. EDL u = C / S
                 If None, we use 1 - max probability as proxy uncertainty.
    class_counts_for_groups: class distribution in training set, used to define head/medium/tail.
                             Example: train_df["label"].value_counts()
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
        probs_tensor = probs.detach().cpu().float()
    else:
        probs_np = np.asarray(probs)
        probs_tensor = torch.tensor(probs_np, dtype=torch.float32)

    # Avoid log(0)
    eps = 1e-12
    probs_np = np.clip(probs_np, eps, 1.0)
    probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)

    probs_tensor = torch.tensor(probs_np, dtype=torch.float32)

    y_true_tensor = torch.tensor(y_true, dtype=torch.long)

    # Basic classification metrics
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_prec, per_class_rec, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
        )
    )

    gm = geometric_mean_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Calibration metrics
    ece_metric = CalibrationError(
        task="multiclass", num_classes=n_classes, n_bins=ece_bins
    )
    ece_value = float(ece_metric(probs_tensor, y_true_tensor).detach().cpu().item())

    # NLL from probabilities
    true_class_probs = probs_np[np.arange(len(y_true)), y_true]
    nll = float(-np.mean(np.log(true_class_probs + eps)))

    # Brier score
    brier = compute_multiclass_brier_score(y_true, probs_np, n_classes)

    # Uncertainty
    if uncertainty is None:
        # Proxy uncertainty if EDL uncertainty is not available
        uncertainty = 1.0 - np.max(probs_np, axis=1)
    else:
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.detach().cpu().numpy()
        else:
            uncertainty = np.asarray(uncertainty)

    mean_u = float(np.mean(uncertainty))

    correct_mask = y_true == y_pred
    wrong_mask = y_true != y_pred

    mean_u_correct = safe_mean(list(uncertainty[correct_mask]))
    mean_u_wrong = safe_mean(list(uncertainty[wrong_mask]))

    # Error detection metrics
    # error_label = 1 means prediction is wrong
    error_label = wrong_mask.astype(int)

    if len(np.unique(error_label)) == 2:
        auroc_error = float(roc_auc_score(error_label, uncertainty))
        aupr_error = float(average_precision_score(error_label, uncertainty))
    else:
        auroc_error = None
        aupr_error = None

    # Head / medium / tail metrics
    head_recall = None
    medium_recall = None
    tail_recall = None

    head_f1 = None
    medium_f1 = None
    tail_f1 = None
    head_tail_f1_gap = None
    head_tail_recall_gap = None

    if class_counts_for_groups is not None:
        head_classes, medium_classes, tail_classes = get_head_medium_tail_classes(
            class_counts_for_groups
        )

        head_recall = compute_group_metrics(per_class_rec, head_classes)
        medium_recall = compute_group_metrics(per_class_rec, medium_classes)
        tail_recall = compute_group_metrics(per_class_rec, tail_classes)

        head_f1 = compute_group_metrics(per_class_f1, head_classes)
        medium_f1 = compute_group_metrics(per_class_f1, medium_classes)
        tail_f1 = compute_group_metrics(per_class_f1, tail_classes)

        if head_f1 is not None and tail_f1 is not None:
            head_tail_f1_gap = float(head_f1 - tail_f1)

        if head_recall is not None and tail_recall is not None:
            head_tail_recall_gap = float(head_recall - tail_recall)

    return {
        # Basic classification
        "acc": float(acc),
        "balanced_acc": float(balanced_acc),
        "macro_prec": float(macro_prec),
        "macro_rec": float(macro_rec),
        "f1": float(macro_f1),  # keep your old key
        "macro_f1": float(macro_f1),
        "weighted_prec": float(weighted_prec),
        "weighted_rec": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "gm": float(gm),
        # Calibration
        "ece": float(ece_value),
        "nll": float(nll),
        "brier": float(brier),
        # Uncertainty
        "mean_u": float(mean_u),
        "mean_u_correct": mean_u_correct,
        "mean_u_wrong": mean_u_wrong,
        "auroc_error": auroc_error,
        "aupr_error": aupr_error,
        # Imbalance-specific
        "head_recall": head_recall,
        "medium_recall": medium_recall,
        "tail_recall": tail_recall,
        "head_f1": head_f1,
        "medium_f1": medium_f1,
        "tail_f1": tail_f1,
        "head_tail_f1_gap": head_tail_f1_gap,
        "head_tail_recall_gap": head_tail_recall_gap,
        # Detailed objects
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "cm": cm,
    }


def display_cm(
    save_dir, wandb, cm, run_name, seed=None, epoch=None, step=None, data="Val"
):
    num_classes = cm.shape[0]

    # Ajustement automatique de la taille de la figure
    fig_size = max(6, int(num_classes * 0.6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(f"{data} CM — {run_name}")
    plt.tight_layout()

    if data == "Val":
        filename = f"{run_name}_epoch{epoch:02d}_step{step:04d}_cm.png"
    elif data == "Test":
        filename = f"{run_name}_Test_cm.png"

    cm_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(cm_dir, exist_ok=True)
    out_path = os.path.join(cm_dir, filename)
    fig.savefig(out_path, dpi=150)
    wandb.log({f"{data}/cm": wandb.Image(fig)}, step=step)

    plt.close(fig)

    return out_path


def log_uncertainty_kde(
    u_correct,
    u_wrong,
    wandb=None,
    step=None,
    set="Val",
    save_dir=None,
    filename=None,
):
    data = pd.DataFrame(
        {
            "u": np.concatenate([u_correct, u_wrong]),
            "type": (["Correct"] * len(u_correct)) + (["Error"] * len(u_wrong)),
        }
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.kdeplot(
        data=data,
        x="u",
        hue="type",
        common_norm=True,
        fill=True,
        alpha=0.4,
        linewidth=2,
        ax=ax,
    )

    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title(f"{set} - Distribution of uncertainty")

    fig.tight_layout()

    # Save locally
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            if step is None:
                filename = f"{set.lower()}_uncertainty_kde.png"
            else:
                filename = f"{set.lower()}_uncertainty_kde_step_{step}.png"

        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Log to Weights & Biases
    if wandb is not None:
        wandb.log(
            {f"{set}/uncertainty_kde": wandb.Image(fig)},
            step=step,
        )

    plt.close(fig)


# TODO revise this function to integrate uncertainty information #Done
def save_metrics_to_csv(
    path: str,
    mode: str,
    metrics: dict,
    epoch=None,
    step=None,
    seed=None,
):
    if mode not in ["val", "test"]:
        raise ValueError("Incorrect mode. Choose 'val' or 'test'.")

    base_header = ["mode"]

    if mode == "val":
        base_header += ["seed", "epoch", "step"]
    else:
        base_header += ["seed"]

    metric_keys = [
        # Loss
        "loss",
        # Standard classification metrics
        "acc",
        "balanced_acc",
        "prec",
        "rec",
        "f1",
        "macro_prec",
        "macro_rec",
        "macro_f1",
        "weighted_prec",
        "weighted_rec",
        "weighted_f1",
        "gm",
        # Calibration metrics
        "ece",
        "nll",
        "brier",
        # Uncertainty metrics
        "mean_u",
        "mean_uncertainty",
        "mean_u_correct",
        "mean_u_wrong",
        "auroc_error",
        "aupr_error",
        # Imbalance-specific global metrics
        "head_recall",
        "medium_recall",
        "tail_recall",
        "head_f1",
        "medium_f1",
        "tail_f1",
        "head_tail_f1_gap",
        "head_tail_recall_gap",
    ]

    header = base_header + metric_keys

    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(header)

        row = [mode]

        if mode == "val":
            row += [seed, epoch, step]
        else:
            row += [seed]

        for key in metric_keys:
            row.append(metrics.get(key, ""))

        writer.writerow(row)


def create_sample(df, sr, split_seed):
    n_samples = int(len(df) * sr)
    return resample(
        df,
        n_samples=n_samples,
        replace=False,
        stratify=df["label"],
        random_state=split_seed,
    )


"""def maybe_eda(df, cfg):
    if cfg.data.eda:
        return apply_eda(df, cfg)
    return df"""


def create_step_imbalance(df, ir, split_seed, label_col="label"):
    """
    Create synthetic step imbalance.

    Head classes keep n_max samples.
    Tail classes keep n_max / ir samples.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing label_col.

    ir : float
        Target imbalance ratio.

    split_seed : int
        Random seed.

    label_col : str
        Label column name.

    Returns
    -------
    pandas.DataFrame
        Imbalanced dataframe.
    """

    class_counts = df[label_col].value_counts()
    classes_sorted = class_counts.sort_values(ascending=False).index.tolist()

    n_classes = len(classes_sorted)
    n_max = class_counts.max()

    print("Original distribution:")
    print(class_counts)
    print(f"Target IR: {ir}")
    print(f"Number of classes: {n_classes}")
    print(f"n_max: {n_max}")

    if ir <= 1:
        print("IR <= 1: original dataset preserved, shuffle only.")
        return df.sample(frac=1, random_state=split_seed).reset_index(drop=True)

    # First half = head classes, second half = tail classes
    n_head_classes = n_classes // 2

    resampled_dfs = []

    for idx, cls in enumerate(classes_sorted):
        df_cls = df[df[label_col] == cls]

        if idx < n_head_classes:
            desired_n = n_max
        else:
            desired_n = int(round(n_max / ir))

        desired_n = max(1, desired_n)
        desired_n = min(len(df_cls), desired_n)

        sampled = resample(
            df_cls, replace=False, n_samples=desired_n, random_state=split_seed
        )

        resampled_dfs.append(sampled)

    df_imbalanced = (
        pd.concat(resampled_dfs)
        .sample(frac=1, random_state=split_seed)
        .reset_index(drop=True)
    )

    counts_after = df_imbalanced[label_col].value_counts()
    ir_actual = counts_after.max() / counts_after.min()

    print("\nNew distribution:")
    print(counts_after)
    print(f"Actual IR: {ir_actual:.2f}")

    return df_imbalanced


def create_exponential_imbalance(df, ir, split_seed, label_col="label"):
    """
    Create synthetic long-tailed imbalance using the standard exponential decay formula.

    n_i = n_max * ir^(-i / (C - 1))

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing a label column.

    ir : float
        Target imbalance ratio: max_class_count / min_class_count.

    split_seed : int
        Random seed.

    label_col : str
        Name of the label column.

    Returns
    -------
    pandas.DataFrame
        Imbalanced dataframe.
    """

    class_counts = df[label_col].value_counts()

    # Sort classes by original frequency, descending
    classes_sorted = class_counts.sort_values(ascending=False).index.tolist()

    n_classes = len(classes_sorted)
    n_max = class_counts.max()

    print("Original distribution:")
    print(class_counts)
    print(f"Target IR: {ir}")
    print(f"Number of classes: {n_classes}")
    print(f"Majority class preserved: {classes_sorted[0]} with {n_max} samples")

    if ir <= 1:
        print("IR <= 1: original dataset preserved, shuffle only.")
        return df.sample(frac=1, random_state=split_seed).reset_index(drop=True)

    resampled_dfs = []

    for i, cls in enumerate(classes_sorted):
        df_cls = df[df[label_col] == cls]

        desired_n = int(round(n_max * (ir ** (-i / (n_classes - 1)))))
        desired_n = max(1, desired_n)
        desired_n = min(len(df_cls), desired_n)

        sampled = resample(
            df_cls, replace=False, n_samples=desired_n, random_state=split_seed
        )

        resampled_dfs.append(sampled)

    df_imbalanced = (
        pd.concat(resampled_dfs)
        .sample(frac=1, random_state=split_seed)
        .reset_index(drop=True)
    )

    counts_after = df_imbalanced[label_col].value_counts()
    ir_actual = counts_after.max() / counts_after.min()

    print("\nNew distribution:")
    print(counts_after)
    print(f"Actual IR: {ir_actual:.2f}")

    return df_imbalanced


def create_synthetic_imbalance(
    df, ir, split_seed, imbalance_type="long_tail", label_col="label"
):
    if imbalance_type == "long_tail":
        return create_exponential_imbalance(
            df=df, ir=ir, split_seed=split_seed, label_col=label_col
        )

    elif imbalance_type == "step":
        return create_step_imbalance(
            df=df, ir=ir, split_seed=split_seed, label_col=label_col
        )

    else:
        raise ValueError(
            f"Unknown imbalance_type={imbalance_type}. "
            "Choose from: 'long_tail' or 'step'."
        )


def create_power_law_imbalance_ratio(df, ir, split_seed):
    class_counts = df["label"].value_counts()
    classes_sorted = class_counts.sort_values(
        ascending=False
    ).index.tolist()  # tri décroissant
    n_classes = len(classes_sorted)
    n_max = class_counts.max()

    print("Original Distribution :\n", class_counts)
    print(f"Classe majoritaire conservée : {classes_sorted[0]} avec {n_max} exemples")
    print(f"Nombre de classes : {n_classes}, IR cible : {ir}")

    if ir <= 1:
        print("IR <= 1 : dataset original conservé (shuffle only)")
        return df.sample(frac=1, random_state=split_seed).reset_index(drop=True)

    # Formule log-tailed sur classes triées par fréquence décroissante
    alpha = np.log(ir) / np.log(n_classes)
    print(f"Alpha (forme du déséquilibre) = {alpha:.4f}")

    ranks = np.arange(1, n_classes + 1)  # rangs croissants
    desired_sizes = n_max / (ranks**alpha)
    class_to_desired = dict(zip(classes_sorted, desired_sizes))

    resampled_dfs = []
    for cls in classes_sorted:
        df_cls = df[df["label"] == cls]
        desired = int(round(min(len(df_cls), class_to_desired[cls])))

        df_res = resample(
            df_cls, replace=False, n_samples=desired, random_state=split_seed
        )
        resampled_dfs.append(df_res)

    df_imbalanced = (
        pd.concat(resampled_dfs)
        .sample(frac=1, random_state=split_seed)
        .reset_index(drop=True)
    )

    counts_after = df_imbalanced["label"].value_counts()
    print("New distribution of classes :\n", counts_after)
    ir_actual = counts_after.max() / counts_after.min()
    print(f"IR obtenu : {ir_actual:.2f}")

    return df_imbalanced


def get_remix_y(y1, y2, lam_x, samples_per_class, K, tau, device):
    """
    Returns lambda_y: tensor
    *Args*
    k: hyper parameter of k-majority
    tau: hyper parameter
    where in original paper they suggested to use k = 3, and tau = 0.5
    Here, lambda_y is defined in the original paper of remix, where there
    are three cases of lambda_y as the following:
    (a). lambda_y = 0
    (b). lambda_y = 1
    (c). lambda_y = lambda_x
    """

    cls_num_list = torch.tensor(samples_per_class)

    # check list stored pairs of embeddings index where one mixup with the other
    check = []
    for i in range(len(y1)):
        check.append([cls_num_list[y1[i]].item(), cls_num_list[y2[i]].item()])
    check = torch.tensor(check)
    lam_y = []

    for i in range(check.size()[0]):
        # temp1 = n_i; temp2 = n_j
        temp1 = check[i][0]
        temp2 = check[i][1]

        if (temp1 / temp2) >= K and lam_x < tau:
            lam_y.append(0)
        elif (temp1 / temp2) <= (1 / K) and (1 - lam_x) < tau:
            lam_y.append(1)
        else:
            lam_y.append(lam_x)

    lam_y = torch.tensor(lam_y).to(device)

    return lam_y


def flatten_config(cfg, parent_key="", sep="/"):
    """Transforme une config imbriquée en un dict plat compatible wandb."""
    items = {}
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict) or hasattr(v, "items"):
            items.update(flatten_config(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def t_sne_vis(embeddings, labels, seed, experiment, save_path):
    tsne = TSNE(n_components=3, random_state=seed)
    embeddings_3d = tsne.fit_transform(embeddings)

    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        color=labels,
        title=f"3D t-SNE Visualization of {experiment}",
    )

    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
            xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray"),
            yaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray"),
            zaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray"),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    )

    # Sauvegarde du fichier interactif HTML en 3D
    filename = os.path.join(save_path, f"tsne_seed_{seed}.html")
    pio.write_html(fig, file=filename, auto_open=False)

    # Sauvegarde PNG statique en 2D
    fig_2d = px.scatter(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        color=labels,
        title=f"2D t-SNE Visualization of {experiment}",
    )

    fig_2d.update_traces(marker=dict(size=8, opacity=0.85))
    fig_2d.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    fig_2d.write_image(os.path.join(save_path, f"tsne_seed_{seed}.png"))

    return fig
