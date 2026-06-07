import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, reduction="mean"):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)

        assert s > 0
        self.s = s
        self.weight = weight
        self.reduction = reduction  # "mean" par défaut

    def forward(self, x, target, reduction=None):
        # reduction: None -> utilise self.reduction
        red = self.reduction if reduction is None else reduction

        # bool (uint8 est déprécié)
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)

        index_float = index.float()
        batch_m = torch.matmul(
            self.m_list[None, :].to(x.device), index_float.transpose(0, 1)
        )
        batch_m = batch_m.view((-1, 1))

        x_m = x - batch_m
        output = torch.where(index, x_m, x)

        return F.cross_entropy(
            self.s * output, target, weight=self.weight, reduction=red
        )


class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, weight, gamma=2.0, reduction="mean"):
        super().__init__()
        self.samples_per_cls = np.array(samples_per_cls, dtype=np.float32)
        self.weight = weight
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        logits: [B, C]
        labels: [B]
        returns:
            - [B] si reduction='none'
            - scalaire sinon
        """
        B, C = logits.shape

        labels_one_hot = F.one_hot(labels, C).float()  # [B, C]

        # poids par échantillon à partir de la classe vraie
        alpha = self.weight[labels]  # [B]
        alpha = alpha.unsqueeze(1).expand(-1, C)  # [B, C]

        bce = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels_one_hot,
            reduction="none",
        )  # [B, C]

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(
                -self.gamma * labels_one_hot * logits
                - self.gamma * torch.log1p(torch.exp(-logits))
            )  # [B, C]

        loss = alpha * modulator * bce  # [B, C]
        loss = loss.sum(dim=1)  # [B]

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def build_loss(cfg, samples_per_class, device):
    if cfg.loss.type == "CE":
        return nn.CrossEntropyLoss(reduction="none")

    if cfg.loss.type == "FL":
        effective_num = 1.0 - np.power(cfg.loss.fl_beta, samples_per_class)
        _weight = (1.0 - cfg.loss.fl_beta) / np.array(effective_num)
        _weight = _weight / np.sum(_weight) * len(samples_per_class)
        weights = torch.tensor(_weight).float().to(device)

        return CBFocalLoss(
            samples_per_cls=samples_per_class,
            weight=weights,
            gamma=cfg.loss.fl_gamma,
            reduction="none",
        )
    if cfg.loss.type == "LDAM":
        effective_num = 1.0 - np.power(0.9999, samples_per_class)
        _weight = (1.0 - 0.9999) / np.array(effective_num)
        _weight = _weight / np.sum(_weight) * len(samples_per_class)
        weights = torch.tensor(_weight).float().to(device)

        return LDAMLoss(
            cls_num_list=samples_per_class,
            weight=weights,
            reduction="none",
        )

    raise ValueError


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, max=10))


def softplus_evidence(y):
    return F.softplus(y) + 1e-4


def kl_divergence(alpha, num_classes, device):
    alpha = alpha.to(device=device, dtype=torch.float32)
    B, K = alpha.shape
    if num_classes is not None:
        assert K == num_classes, f"{K=} != {num_classes=}"
    alpha0 = alpha.sum(dim=1)
    lnB = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(alpha0)  # [B]
    lnB_uni = -torch.lgamma(
        torch.tensor(K, dtype=alpha.dtype, device=device)
    )  # scalaire

    term = (
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha0.unsqueeze(-1)))
    ).sum(dim=1)  # [B]

    kl = lnB_uni - lnB + term  # [B]
    return kl.unsqueeze(1)


def balanced_kl_divergence(alpha, num_classes, device, beta):
    """
    KL( Dir(alpha) || Dir(beta) )
    alpha: [B,K]
    beta : None ou [K] uniquement (prior global par classe)
    return: [B]
    """
    alpha = alpha.to(device=device, dtype=torch.float32)
    B, K = alpha.shape
    if num_classes is not None:
        assert K == num_classes, f"{K=} != {num_classes=}"

    beta = beta.to(device=device, dtype=alpha.dtype)
    assert beta.dim() == 1 and beta.numel() == K, f"beta must be [K], got {beta.shape}"

    beta_b = beta.unsqueeze(0).expand(B, -1)  # [B,K]
    alpha0 = alpha.sum(dim=1)
    beta0 = beta.sum(dim=0)  # [B]

    # lnB(x) = sum lgamma(x_k) - lgamma(sum x_k)
    lnB_alpha = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(alpha0)  # [B]
    lnB_beta = torch.lgamma(beta).sum(dim=0) - torch.lgamma(beta0)  # [B]

    # KL(Dir(a)||Dir(b)) = lnB(b)-lnB(a) + sum_k (a_k-b_k) * (psi(a_k)-psi(a0))
    term = (
        (alpha - beta_b) * (torch.digamma(alpha) - torch.digamma(alpha0.unsqueeze(-1)))
    ).sum(dim=1)  # [B]
    kl = (lnB_beta - lnB_alpha) + term  # [B]
    return kl


def kl_cat(p, q):
    """
    KL( Cat(p) || Cat(q) ) avec p,q des distributions (sommant à 1).
    """
    return (p * (p.log() - q.log())).sum()


def edl_ce_loss_balanced(
    alphas,
    labels,
    epoch,
    annealing_step,
    num_cls,
    samples_per_class,
    beta,
    mu: float = 1.0,
    focal=False,
    gamma_focal=2.0,
    reduction: str = "mean",
    device=None,
):
    """
    Balanced-EDL with mini-batch normalized class-balanced weighting.

    Main idea:
      - The theoretical class pooling loss is:
            (1/C) * sum_c (1/N_c) * sum_{i:y_i=c} L_i

      - In mini-batch training, we use the equivalent normalized weight:
            w_y = N / (C * N_y)

        and compute:
            mean_i [ w_{y_i} * L_i ]

      This preserves the class-balanced objective while keeping the loss scale
      comparable to a standard mini-batch loss.

    Important:
      - alphas must be the posterior alpha':
            alpha' = evidence + beta
      - beta must already be positive, e.g. beta = softplus(beta_raw).
    """

    # ---------------------- Input preparation
    alphas = alphas.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device).long()

    B, K = alphas.shape
    assert K == num_cls, f"{K=} != {num_cls=}"

    beta_vec = beta.to(device=device, dtype=alphas.dtype)

    if beta_vec.dim() != 1 or beta_vec.numel() != K:
        raise ValueError(f"beta must be shape [K], got {beta_vec.shape}")

    # Avoid numerical issues if beta is extremely small
    beta_vec = beta_vec.clamp_min(1e-8)

    # ----------------------Evidential CE term
    S = alphas.sum(dim=1, keepdim=True)  # [B, 1]

    one_hot = F.one_hot(labels, num_classes=K).to(
        device=device,
        dtype=alphas.dtype,
    )  # [B, K]

    # E_q[-log p_y] = psi(S) - psi(alpha_y)
    A = torch.sum(
        one_hot * (torch.digamma(S) - torch.digamma(alphas)),
        dim=1,
    )  # [B]

    if focal:
        probs = (alphas / S).clamp_min(1e-12)  # [B, K]
        A = add_focal(
            probs,
            one_hot,
            gamma_focal,
            A,
            samples_per_class,
            num_cls,
            device,
        )  # [B]

    # ----------------------Annealing coefficient
    if epoch is None or annealing_step is None or annealing_step == 0:
        annealing_coef = torch.tensor(
            1.0,
            dtype=alphas.dtype,
            device=device,
        )
    else:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=alphas.dtype, device=device),
            torch.tensor(
                float(epoch) / float(annealing_step),
                dtype=alphas.dtype,
                device=device,
            ),
        )

    # ----------------------KL regularization: KL(Dir(kl_alpha) || Dir(beta))
    beta_b = beta_vec.unsqueeze(0)  # [1, K]

    # Remove evidence from the true class, but keep beta for the true class
    kl_alpha = alphas * (1.0 - one_hot) + beta_b * one_hot  # [B, K]

    kl = balanced_kl_divergence(
        kl_alpha,
        num_classes=K,
        device=device,
        beta=beta_vec,
    )  # [B]

    kl_div = annealing_coef * kl  # [B]

    loss_vec = A + kl_div  # [B]

    # ----------------------Prior penalty: KL(Cat(beta) || Cat(eta))
    spc = torch.as_tensor(
        samples_per_class,
        device=device,
        dtype=alphas.dtype,
    ).clamp_min(1e-8)  # [K]

    if spc.numel() != K:
        raise ValueError(f"samples_per_class must have shape [K], got {spc.shape}")

    N = spc.sum()  # total number of training samples

    # eta_c = N / N_c, then normalized as a categorical distribution
    eta = N / spc  # [K]

    p_beta = beta_vec / beta_vec.sum()
    q_eta = eta / eta.sum()

    p_beta = p_beta.clamp_min(1e-12)
    q_eta = q_eta.clamp_min(1e-12)

    p_beta = p_beta / p_beta.sum()
    q_eta = q_eta / q_eta.sum()

    Lp = kl_cat(p_beta, q_eta)

    # ----------------------Reduction
    if reduction == "none":
        # Return unweighted per-sample losses.
        # This is useful for mixup/remix, where the weighting is handled manually.
        return loss_vec, A, kl_div, Lp

    # Normalized class-balanced weights:
    # w_y = N / (C * N_y)
    C = float(K)
    class_weights = N / (C * spc)  # [K]
    sample_weights = class_weights[labels]  # [B]

    if reduction == "mean":
        L = (loss_vec * sample_weights).mean() + mu * Lp
        A_out = (A * sample_weights).mean()
        KL_out = (kl_div * sample_weights).mean()

        return L, A_out, KL_out, Lp

    if reduction == "sum":
        # Less commonly used for training.
        # This keeps the normalized class-balanced weights, but sums over the batch.
        L = (loss_vec * sample_weights).sum() + mu * Lp
        A_out = (A * sample_weights).sum()
        KL_out = (kl_div * sample_weights).sum()

        return L, A_out, KL_out, Lp

    raise ValueError(f"Unknown reduction: {reduction}")


# VERIFIED (dynamic annealing_coef is better than fixed annealing_step)
def edl_ce_loss(
    alpha,
    labels,
    epoch,
    annealing_step,
    num_cls,
    samples_per_class,
    focal: bool = False,
    gamma_focal=2.0,
    reduction: str = "mean",
    device=None,
):
    """reduction=="mean" return scalar mean(batch), "sum" return a scalar sum(batch), "none" return tensors [batch]"""

    alpha = alpha.to(device=device, dtype=torch.float32)
    labels = labels.to(device).long()
    S = torch.sum(alpha, dim=1, keepdim=True)  # [B,1]
    one_hot = F.one_hot(labels, num_classes=num_cls).to(
        device=device, dtype=alpha.dtype
    )  # [B,K]

    A = torch.sum(one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)  # [B]
    if focal:
        probs = (alpha / S).clamp_min(
            1e-12
        )  # [B,K]                         # predicted class probabilities
        A = add_focal(
            probs, one_hot, gamma_focal, A, samples_per_class, num_cls, device
        )  # [B]

    if epoch is None or annealing_step is None:
        annealing_coef = torch.tensor(1.0, dtype=torch.float32)
    else:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            torch.tensor(epoch / annealing_step, dtype=torch.float32, device=device),
        )
        # annealing_coef = annealing_step
    # KL regularization
    kl_alpha = (alpha - 1.0) * (1.0 - one_hot) + 1.0  # [B,K]
    kl = kl_divergence(kl_alpha, num_cls, device)  # [B]
    kl_div = annealing_coef * kl  # [B]

    loss = A + kl_div  # [B]
    # loss = A

    if reduction == "mean":
        return torch.mean(loss), torch.mean(A), torch.mean(kl_div)
    elif reduction == "sum":
        return torch.sum(loss), torch.sum(A), torch.sum(kl_div)
    else:
        return loss, A, kl_div


# VERIFIED
def edl_mse_loss(
    alpha, labels, epoch, annealing_step, num_cls, reduction=False, device="cuda:0"
):
    """
    EDL-MSE loss with KL regularization.
    """
    S = torch.sum(alpha, dim=1, keepdim=True)  # total evidence + num_classes
    probs = alpha / S  # predicted class probabilities

    one_hot = F.one_hot(labels, num_classes=num_cls).float().to(device)

    # Squared error between predicted probs and one-hot labels
    A = torch.sum((one_hot - probs) ** 2, dim=1, keepdim=True)  # shape [B, 1]
    B = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )  # could be simplified by (probs*(1-probs)/(s+1)).sum(dim=-1)
    mse = A + B

    # Annealing coefficient
    if epoch is None:
        annealing_coef = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            torch.tensor(epoch / annealing_step, dtype=torch.float32, device=device),
        )

    # KL regularization
    kl_alpha = (alpha - 1) * (1 - one_hot) + 1
    kl = kl_divergence(kl_alpha, num_cls, device)
    kl_term = annealing_coef * kl

    # Final loss
    loss = mse + kl_term

    if reduction:
        return torch.mean(loss), torch.mean(mse), torch.mean(kl_term)
    else:
        return loss.squeeze(), mse.squeeze(), kl_term.squeeze()


def add_focal(probs, one_hot, gamma_focal, loss, samples_per_class, num_cls, device):
    # focal loss
    p_y = (probs * one_hot).sum(-1)  # [B] proba de la vraie classe
    focal_factor = (1.0 - p_y) ** gamma_focal  # [B]
    focal_loss = focal_factor * loss  # [B]

    effective_num = 1.0 - np.power(0.9999, samples_per_class)
    _weight = (1.0 - 0.9999) / np.array(effective_num)
    _weight = _weight / np.sum(_weight) * num_cls
    class_weights = torch.tensor(_weight).float().to(device)  # [K]

    # weighted loss
    weights = (one_hot * class_weights.unsqueeze(0)).sum(-1)  # [B]

    weighted_loss = weights * focal_loss
    return weighted_loss


# TODO Isolate the bloc focal and weighted and add to the previous two functions MSE et CE with digamma
def edl_mse_focal_loss(
    alpha,
    labels,
    epoch,
    annealing_step,
    num_cls,
    gamma_focal,
    samples_per_class,
    reduction=False,
    device="cuda:0",
):
    """
    EDL-MSE loss with KL regularization.
    """
    S = torch.sum(alpha, dim=1, keepdim=True)  # total evidence + num_classes
    probs = alpha / S  # predicted class probabilities

    one_hot = F.one_hot(labels, num_classes=num_cls).float().to(device)

    # Squared error between predicted probs and one-hot labels
    A = torch.sum((one_hot - probs) ** 2, dim=1)  # shape [B]
    B = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1
    )  # could be simplified by (probs*(1-probs)/(s+1)).sum(dim=-1)
    mse = A + B

    # focal loss
    p_y = (probs * one_hot).sum(-1)  # [B] proba de la vraie classe
    focal_factor = (1.0 - p_y) ** gamma_focal  # [B]
    focal_mse = focal_factor * mse  # [B]

    effective_num = 1.0 - np.power(0.9999, samples_per_class)
    _weight = (1.0 - 0.9999) / np.array(effective_num)
    _weight = _weight / np.sum(_weight) * num_cls
    class_weights = torch.tensor(_weight).float().to(device)  # [K]
    print("weights size before", len(class_weights))

    # weighted loss
    weights = (one_hot * class_weights.unsqueeze(0)).sum(-1)  # [B]
    print("weights size after", len(weights))

    weighted_loss = weights * focal_mse

    # Annealing coefficient
    if epoch is None:
        annealing_coef = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            torch.tensor(epoch / annealing_step, dtype=torch.float32, device=device),
        )

    # KL regularization
    kl_alpha = (alpha - 1) * (1 - one_hot) + 1
    kl = kl_divergence(kl_alpha, num_cls, device)
    kl_term = annealing_coef * kl

    # Final loss
    loss = weighted_loss + kl_term

    if reduction:
        return torch.mean(loss), torch.mean(weighted_loss), torch.mean(kl_term)
    else:
        return loss.squeeze(), weighted_loss.squeeze(), kl_term.squeeze()
