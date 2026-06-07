import numpy as np
import torch
from torch.amp import autocast

from .losses import edl_ce_loss_balanced, softplus_evidence
from .trainers import Trainer
from .utils import get_perm, get_remix_y


class MixBeliefTrainer2(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.prev_class_beliefs = {}
        self.prev_class_uncertainty = {}

        self.p_mix_by_class = {}
        self.rivals_by_class = {}
        self.rival_probs_by_class = {}

        self.mixup_alpha = float(self.cfg.mix.alpha)
        self.original_mix_method = cfg.mix.method

    def extract_confused_pairs_from_mean_belief(self):
        """
        Selective Pairwise Mix-Belief.

        For each class, this function estimates:
        - whether examples from this class should be mixed with a rival class;
        - which rival classes are plausible partners.

        Important:
        p_mix_by_class is the probability of inter-class guided mixup.
        If inter-class mixup is not selected or no rival is available in the batch,
        the algorithm falls back to within-class mixup.
        """

        EPS = 1e-8

        p_mix_by_class = {}
        rivals_by_class = {}
        rival_probs_by_class = {}
        confused_pairs = set()

        for cls, belief_vec in self.prev_class_beliefs.items():
            cls = int(cls)

            p_mix_by_class[cls] = 0.0
            rivals_by_class[cls] = []
            rival_probs_by_class[cls] = []

            b = np.asarray(belief_vec, dtype=np.float64)
            C = len(b)

            if cls < 0 or cls >= C:
                continue

            b = np.clip(b, 0.0, None)

            b_true = float(b[cls])

            off_diag = b.copy()
            off_diag[cls] = 0.0

            off_mass = float(off_diag.sum())

            if off_mass <= EPS:
                continue

            # Total belief mass assigned outside + inside the true class.
            # If mean_belief is evidence-based, this should be close to 1 - uncertainty.
            belief_mass = float(b_true + off_mass)
            belief_reliability = float(np.clip(belief_mass, 0.0, 1.0))

            # How much of the available belief is assigned outside the true class?
            confusion = off_mass / (b_true + off_mass + EPS)

            # Distribution over rival classes
            rival_probs = off_diag / (off_mass + EPS)

            # If one or a few rivals dominate, concentration is high.
            # If confusion is diffuse, concentration is low.
            concentration = float(rival_probs.max())

            # Probability of inter-class guided mixup.
            # High only when:
            #   1) the class is confused,
            #   2) the confusion points to identifiable rival(s),
            #   3) the belief signal is reliable enough.
            p_mix = confusion * np.sqrt(concentration) * np.sqrt(belief_reliability)
            p_mix = float(np.clip(p_mix, 0.0, 1.0))

            if C > 1:
                uniform_level = 1.0 / float(C - 1)
            else:
                uniform_level = 1.0

            candidate_rivals = np.where(rival_probs >= uniform_level)[0].tolist()

            if len(candidate_rivals) == 0:
                candidate_rivals = [int(np.argmax(rival_probs))]

            candidate_rivals = [
                int(r)
                for r in candidate_rivals
                if int(r) != cls and float(rival_probs[int(r)]) > EPS
            ]

            if len(candidate_rivals) == 0:
                continue

            candidate_probs = np.array(
                [rival_probs[r] for r in candidate_rivals],
                dtype=np.float64,
            )

            candidate_probs = candidate_probs / (candidate_probs.sum() + EPS)

            p_mix_by_class[cls] = p_mix
            rivals_by_class[cls] = candidate_rivals
            rival_probs_by_class[cls] = candidate_probs.tolist()

            for r in candidate_rivals:
                confused_pairs.add((cls, int(r)))

        self.p_mix_by_class = p_mix_by_class
        self.rivals_by_class = rivals_by_class
        self.rival_probs_by_class = rival_probs_by_class

        return confused_pairs

    # ADDED
    def train_one_epoch(self, epoch):
        if epoch < self.switch_epoch:
            self.cfg.mix.method = "none"
            if getattr(self, "rank", 0) == 0:
                self.logger.info(f"Epoch {epoch}: Warmup (no mixup)")
        else:
            self.cfg.mix.method = self.original_mix_method
            self.cfg.mix.alpha = self.mixup_alpha
            if getattr(self, "rank", 0) == 0:
                self.logger.info(
                    f"Epoch {epoch}: Mix method = {self.cfg.mix.method} | alpha = {self.mixup_alpha}"
                )

        super().train_one_epoch(epoch)

        if self.use_unc and (self.previous_beliefs is not None):
            # mean beliefs per class
            self.prev_class_beliefs = {
                int(cls): np.array(
                    self.previous_beliefs[cls]["mean_belief"], dtype=np.float32
                )
                for cls in self.previous_beliefs.keys()
            }
            # mean uncertainty per class
            self.prev_class_uncertainty = {
                int(cls): float(self.previous_beliefs[cls]["mean_uncertainty"])
                for cls in self.previous_beliefs.keys()
            }

            confused_pairs = self.extract_confused_pairs_from_mean_belief()

            if getattr(self, "rank", 0) == 0 and epoch == self.switch_epoch:
                self.logger.info(
                    f"[Ratio-MixBelief] confused_pairs={confused_pairs} | "
                    f"p_mix_by_class={self.p_mix_by_class} | "
                    f"rivals_by_class={self.rivals_by_class}"
                )

    def _mixbelief_indices_and_lambdas(self, y1, alpha, generator=None):
        """
        Selective Mix-Belief with within-class fallback.

        Pairing strategy:
        1. Guided inter-class mixup if the class is confused and a rival is present.
        2. Within-class mixup if no valid rival is available or if the class should not be mixed.
        3. Self-pairing only if no same-class partner exists in the batch.

        Important:
        - One scalar lambda is sampled per batch.
        - The scalar is duplicated to keep the same interpolation strength
            for all examples in the batch.
        """

        if generator is None:
            generator = torch.Generator(device="cpu")

        y_np = y1.detach().cpu().numpy()
        B = len(y_np)

        beta_dist = torch.distributions.Beta(alpha, alpha)

        # ------------------------------------------------------------
        # One scalar lambda for the whole batch
        # ------------------------------------------------------------
        lam_scalar = float(beta_dist.sample().item())

        # Anchor-preserving lambda for inter-class mixup.
        # This keeps y1 dominant.
        lam_scalar = max(lam_scalar, 1.0 - lam_scalar)

        lam_vec = torch.full(
            (B,),
            lam_scalar,
            dtype=torch.float32,
            device=self.device,
        )

        index = []

        for i, cls in enumerate(y_np):
            cls = int(cls)

            p_mix = float(self.p_mix_by_class.get(cls, 0.0))
            rivals = self.rivals_by_class.get(cls, [])
            rival_probs = self.rival_probs_by_class.get(cls, [])

            do_guided_inter = (
                len(rivals) > 0 and torch.rand((), generator=generator).item() < p_mix
            )

            j = None

            # ------------------------------------------------------------
            # 1. Try guided inter-class pairing
            # ------------------------------------------------------------
            if do_guided_inter:
                available_rivals = []
                available_probs = []

                for r, p in zip(rivals, rival_probs):
                    idxs = np.where(y_np == int(r))[0]
                    idxs = idxs[idxs != i]

                    if len(idxs) > 0:
                        available_rivals.append((int(r), idxs))
                        available_probs.append(float(p))

                if len(available_rivals) > 0:
                    available_probs = np.asarray(available_probs, dtype=np.float64)

                    prob_sum = available_probs.sum()

                    if (not np.isfinite(prob_sum)) or prob_sum <= 0.0:
                        rival_position = int(
                            torch.randint(
                                len(available_rivals),
                                (1,),
                                generator=generator,
                            ).item()
                        )
                    else:
                        probs_t = torch.as_tensor(
                            available_probs,
                            dtype=torch.float32,
                            device="cpu",
                        )

                        probs_t = probs_t.clamp_min(0.0)
                        probs_t = probs_t / probs_t.sum().clamp_min(1e-12)

                        rival_position = int(
                            torch.multinomial(
                                probs_t,
                                num_samples=1,
                                replacement=True,
                                generator=generator,
                            ).item()
                        )

                    _, idxs = available_rivals[rival_position]

                    j = int(
                        idxs[
                            torch.randint(
                                len(idxs),
                                (1,),
                                generator=generator,
                            ).item()
                        ]
                    )

            # ------------------------------------------------------------
            # 2. Fallback: within-class mixup
            # ------------------------------------------------------------
            if j is None:
                same_class_idxs = np.where(y_np == cls)[0]
                same_class_idxs = same_class_idxs[same_class_idxs != i]

                if len(same_class_idxs) > 0:
                    j = int(
                        same_class_idxs[
                            torch.randint(
                                len(same_class_idxs),
                                (1,),
                                generator=generator,
                            ).item()
                        ]
                    )
                else:
                    # ----------------------------------------------------
                    # 3. Last fallback: self-pairing
                    # ----------------------------------------------------
                    j = i

            """# ------------------------------------------------------------
            # 2. Ablation: remove the within-class fallback
            # ------------------------------------------------------------
            if j is None:
                # Prefer an arbitrary example from another class.
                # Unlike the full method, this forces cross-class interpolation
                # even when no reliable rival has been identified.
                different_class_idxs = np.where(y_np != cls)[0]

                if len(different_class_idxs) > 0:
                    j = int(
                        different_class_idxs[
                            torch.randint(
                                len(different_class_idxs),
                                (1,),
                                generator=generator,
                            ).item()
                        ]
                    )
                else:
                    # If the batch contains only one class, select any non-self
                    # example to preserve interpolation whenever possible.
                    non_self_idxs = np.arange(B)
                    non_self_idxs = non_self_idxs[non_self_idxs != i]

                    if len(non_self_idxs) > 0:
                        j = int(
                            non_self_idxs[
                                torch.randint(
                                    len(non_self_idxs),
                                    (1,),
                                    generator=generator,
                                ).item()
                            ]
                        )
                    else:
                        # Only possible when the batch contains one example.
                        j = i"""

            index.append(j)

        indices_t = torch.tensor(index, device=self.device, dtype=torch.long)

        return indices_t, lam_vec

    def _step(self, batch, epoch):
        if self.cfg.mix.method == "none":
            return super()._step(batch, epoch)

        y1 = batch.pop("label")  # [B]
        x1, att1 = batch["input_ids"], batch["attention_mask"]  # [B,*], [B,*]

        method = self.cfg.mix.method.lower()
        alpha = float(self.mixup_alpha)
        bs = y1.size(0)

        if method in ("mixup", "remix"):
            index = get_perm(x1)
            x2, y2, att2 = x1[index], y1[index], att1[index]

            beta_dist = torch.distributions.Beta(alpha, alpha)
            lam_scalar = float(beta_dist.sample().item())

            lam_vec = torch.full(
                (bs,),
                lam_scalar,
                dtype=torch.float32,
                device=self.device,
            )

            if method == "mixup":
                lam_y = lam_vec
            else:
                lam_y = get_remix_y(
                    y1,
                    y2,
                    lam_scalar,
                    self.n_pc,
                    self.cfg.mix.k_majority,
                    self.cfg.mix.tau,
                    self.device,
                )

        elif method == "mix-belief":
            index, lam_vec = self._mixbelief_indices_and_lambdas(y1, alpha)

            x2, y2, att2 = x1[index], y1[index], att1[index]

            # Same lambda for input and loss
            lam_y = lam_vec

        else:
            raise ValueError(f"Unknown mix method: {self.cfg.mix.method}")

        # lambda for hidden-state interpolation
        lam_x = lam_vec.view(-1, 1)

        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model.module.forward_mix_encoder(x1, att1, x2, att2, lam_x)

        logits = outputs.float()
        evidences = softplus_evidence(logits)

        mu_prior = float(getattr(self.cfg.loss, "mu_prior", 0.0))

        beta_vec = softplus_evidence(self.model.module.beta_raw)  # [C], > 0
        beta_vec = beta_vec.to(device=self.device, dtype=evidences.dtype)

        alphas = evidences + beta_vec.unsqueeze(0)  # [B, C]

        S = torch.sum(alphas, dim=1, keepdim=True)
        probs = (alphas / S).clamp(min=1e-10)
        preds = torch.argmax(probs, dim=1)

        focal_flag = getattr(self.cfg.loss, "type", "CE") == "FL"

        # Per-sample EDL loss for y1
        loss1, ce_loss1, kl_div1, lp1 = edl_ce_loss_balanced(
            alphas,
            y1,
            epoch,
            self.cfg.loss.annealing_step,
            self.n_classes,
            self.n_pc,
            beta=beta_vec,
            mu=0.0,  # no effect with reduction="none", kept explicit
            focal=focal_flag,
            reduction="none",
            device=self.device,
        )

        # Per-sample EDL loss for y2
        loss2, ce_loss2, kl_div2, _ = edl_ce_loss_balanced(
            alphas,
            y2,
            epoch,
            self.cfg.loss.annealing_step,
            self.n_classes,
            self.n_pc,
            beta=beta_vec,
            mu=0.0,  # no effect with reduction="none", kept explicit
            focal=focal_flag,
            reduction="none",
            device=self.device,
        )

        lamb = lam_y.to(device=self.device, dtype=loss1.dtype)  # [B]

        # ------------------------------------------------------------------
        # Corrected class-balanced weighting for mini-batch training
        # w_y = N / (C * N_y)
        # ------------------------------------------------------------------
        spc = torch.as_tensor(
            self.n_pc,
            device=self.device,
            dtype=loss1.dtype,
        ).clamp_min(1e-12)  # [C]

        N = spc.sum()
        C = float(self.n_classes)

        class_weights = N / (C * spc)  # [C]

        w1 = class_weights[y1]  # [B]
        w2 = class_weights[y2]  # [B]

        # Mixup-compatible class-balanced EDL loss
        loss_vec = lamb * w1 * loss1 + (1.0 - lamb) * w2 * loss2
        ce_vec = lamb * w1 * ce_loss1 + (1.0 - lamb) * w2 * ce_loss2
        kl_vec = lamb * w1 * kl_div1 + (1.0 - lamb) * w2 * kl_div2

        # Mini-batch mean with normalized class-balanced weights
        loss = loss_vec.mean() + mu_prior * lp1
        ce_loss = ce_vec.mean()
        kl_div = kl_vec.mean()

        match = (
            lamb * (preds == y1).float() + (1.0 - lamb) * (preds == y2).float()
        ).sum()

        if getattr(self, "rank", 0) == 0 and epoch == self.switch_epoch:
            self.logger.info(f"lambda = {float(lam_vec[0].detach().cpu()):.4f}")

        return loss, match, ce_loss, kl_div
