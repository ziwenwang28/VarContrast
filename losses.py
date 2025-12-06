from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarConLoss(nn.Module):
    """
    VarCon: Variational Supervised Contrastive Learning

    Loss definition (per sample, averaged over batch):
        L_VarCon = D_KL(q_φ(r'|z) || p_θ(r'|z)) - log p_θ(r|z)

    where
        p_θ(r|z)        = exp(z^T w_r / τ₁) / Σ_{r'} exp(z^T w_{r'} / τ₁)   (Eq. 8)
        q_one-hot(r'|z) = 1 if r' = r else 0                                (Eq. 10)
        q_exp(r'|z)     = 1 + [exp(1/τ₂) - 1] · q_one-hot(r'|z)             (Eq. 11)
        q_φ(r'|z)       = q_exp(r'|z) / Σ_{r'} q_exp(r'|z)                  (Eq. 12)
        τ₂              = (τ₁ - ε) + 2ε · p_θ(r|z)                          (Eq. 13)

    Notation:
        z       : L2-normalized embedding from encoder
        w_r     : L2-normalized centroid (class reference vector) for class r
        τ₁      : fixed temperature scaling the logits
        τ₂      : confidence-adaptive temperature for target distribution
        ε       : temperature adaptation range parameter
        r       : ground-truth class index
        r'      : dummy class index for summation

    Args (init):
        num_classes:    total number of classes (not strictly needed for VarCon itself,
                        since we use batch-wise unique labels, kept for interface)
        feat_dim:       feature dimension (for information only)
        temperature:    τ₁, temperature used in p_θ (softmax logits scaling)
        epsilon:        ε, adaptive temperature strength
        normalize:      whether to L2-normalize centroids (features assumed already normalized)
    """
    def __init__(
        self,
        num_classes: int = 1000,
        feat_dim: int = 128,
        temperature: float = 0.1,
        epsilon: float = 0.02,
        normalize: bool = True,
    ):
        super().__init__()
        self.tau1 = temperature
        self.normalize = normalize
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.epsilon = epsilon

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: [B, feat_dim]  encoder output features
                      (assumed to be L2-normalized, e.g., from VarConResNet)
            labels:   [B]            integer class labels

        Returns:
            dict with:
                'total_loss': batch-averaged VarCon loss (scalar)
                'kl_div':     KL(q||p) term (detached scalar)
                'nll':        -log p_θ(r|z) term (detached scalar)
                'avg_tau2':   average τ₂ over batch
                'avg_confidence': average p_θ(r|z) over batch
                'epsilon':    current epsilon value (Tensor)
                'all_tau2s':  τ₂ for all samples (Tensor [B])
        """
        device = features.device
        B = features.shape[0]

        # 1) No re-normalization of features:
        #    If using VarConResNet, features are already L2-normalized.
        # if self.normalize:
        #     features = F.normalize(features, p=2, dim=1)

        # 2) Compute centroid for each class in batch (vectorized, no for loop)
        unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)
        C_batch = len(unique_labels)  # number of classes in current batch

        # one_hot: [B, C_batch]
        one_hot = torch.zeros(B, C_batch, device=device)
        one_hot.scatter_(1, inverse_indices.unsqueeze(1), 1.0)

        # number of samples per class: [C_batch]
        class_counts = one_hot.sum(dim=0).clamp(min=1)

        # centroids: [C_batch, feat_dim] = [C_batch, B] @ [B, feat_dim]
        centroids = one_hot.T @ features / class_counts.unsqueeze(1)

        # Normalize centroids (corresponding to w_r = \bar z_r / ||\bar z_r||)
        if self.normalize:
            centroids = F.normalize(centroids, p=2, dim=1)

        # Detach centroids to prevent gradients from flowing back to features
        centroids = centroids.detach()

        # 3) logits: z^T w_r / τ₁ (Eq. 8)
        #    features:  [B, feat_dim]
        #    centroids: [C_batch, feat_dim]
        logits = features @ centroids.T / self.tau1  # [B, C_batch]

        # 4) p_θ(r|z) (softmax over classes, Eq. 8)
        p_theta = F.softmax(logits, dim=1)  # [B, C_batch]
        # Confidence p_θ(r|z) for true class of each sample: [B]
        confidences = p_theta.gather(1, inverse_indices.unsqueeze(1)).squeeze(1)

        # 5) τ₂ (Eq. 13)
        eps = self.epsilon
        tau2 = (self.tau1 - eps) + 2.0 * eps * confidences  # [B]
        tau2 = torch.clamp(tau2, min=1e-6)

        # 6) q_φ (Eq. 10, 11, 12), reusing one_hot
        # q_exp: exp(1/τ₂) for true class, 1 for other classes
        exp_inv_tau2 = torch.exp(1.0 / tau2)               # [B]
        q_exp = 1.0 + (exp_inv_tau2.unsqueeze(1) - 1.0) * one_hot  # [B, C_batch]
        q_phi = q_exp / q_exp.sum(dim=1, keepdim=True)     # normalize to probability

        # 7) Loss (Eq. 7)
        log_p_theta = F.log_softmax(logits, dim=1)         # [B, C_batch]

        # KL(q || p) over batch
        kl_div = F.kl_div(
            log_p_theta,  # log p
            q_phi,        # q
            reduction='batchmean',
            log_target=False,
        )

        # NLL: -log p_θ(r|z)
        nll = -log_p_theta.gather(1, inverse_indices.unsqueeze(1)).mean()

        loss = kl_div + nll

        return {
            'total_loss': loss,
            'kl_div': kl_div.detach(),
            'nll': nll.detach(),
            'avg_tau2': tau2.mean().detach(),
            'avg_confidence': confidences.mean().detach(),
            'epsilon': torch.tensor(eps) if not torch.is_tensor(eps) else eps.detach(),
            'all_tau2s': tau2.detach(),
        }
