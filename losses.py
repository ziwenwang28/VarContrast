from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarConLoss(nn.Module):
    """
    VarCon: Variational Supervised Contrastive Learning

    Loss definition (per batch):
        L_VarCon = D_KL(q_φ(r'|z) || p_θ(r'|z)) - log p_θ(r|z)

    where
        p_θ(r|z)   = softmax(z^T w_r / τ₁)                   (Eq. 8)
        q_one-hot = 1 if r' = r else 0                      (Eq. 10)
        q_exp(r') = 1 + [exp(1/τ₂) - 1] * q_one-hot(r')     (Eq. 11)
        q_φ(r')   = q_exp(r') / Σ_k q_exp(k)                (Eq. 12)
        τ₂        = (τ₁ - ε) + 2ε * p_θ(r|z)                (Eq. 13)

    Args (init):
        num_classes:    total number of classes (not strictly needed for VarCon itself,
                        since we use batch-wise unique labels, kept for interface)
        feat_dim:       feature dimension (for information only)
        temperature:    τ₁, temperature used in p_θ (softmax logits scaling)
        epsilon:        ε, adaptive temperature strength (fixed if learn_epsilon=False)
        learn_epsilon:  whether to learn ε as nn.Parameter
        epsilon_upper_bound: clamp upper bound when learn_epsilon=True
        normalize:      whether to L2-normalize centroids (features assumed already normalized)
    """
    def __init__(
        self,
        num_classes: int = 1000,
        feat_dim: int = 128,
        temperature: float = 0.1,
        epsilon: float = 0.02,
        learn_epsilon: bool = False,
        epsilon_upper_bound: float = 0.08,
        normalize: bool = True,
    ):
        super().__init__()
        self.tau1 = temperature
        self.normalize = normalize
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.epsilon_upper_bound = epsilon_upper_bound
        self.learn_epsilon = learn_epsilon

        if learn_epsilon:
            self._epsilon = nn.Parameter(torch.tensor(float(epsilon)))
        else:
            self.register_buffer('_epsilon', torch.tensor(float(epsilon)))

    @property
    def epsilon(self) -> torch.Tensor:
        """Return epsilon, clamped to [1e-6, epsilon_upper_bound] if learnable."""
        if self.learn_epsilon:
            return torch.clamp(self._epsilon, min=1e-6, max=self.epsilon_upper_bound)
        return self._epsilon

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: [B, feat_dim]  encoder output features
                      (在当前实现中假设已经是 L2-normalized 的，比如来自 SupConResNet)
            labels:   [B]            integer class labels

        Returns:
            dict with:
                'total_loss': scalar VarCon loss
                'kl_div':     KL(q||p) term (detached scalar)
                'nll':        -log p_θ(r|z) term (detached scalar)
                'avg_tau2':   average τ₂ over batch
                'avg_confidence': average p_θ(r|z) over batch
                'epsilon':    current epsilon value (Tensor)
                'all_tau2s':  τ₂ for all samples (Tensor [B])
        """
        device = features.device
        B = features.shape[0]

        # 1) 不再对 features 做二次归一化：
        #    如果使用 SupConResNet，则 features 已经是 L2-normalized 的。
        # if self.normalize:
        #     features = F.normalize(features, p=2, dim=1)

        # 2) 计算 batch 内每个类的 centroid (向量化，无 for 循环)
        unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)
        C_batch = len(unique_labels)  # 当前 batch 中出现的类别数

        # one_hot: [B, C_batch]
        one_hot = torch.zeros(B, C_batch, device=device)
        one_hot.scatter_(1, inverse_indices.unsqueeze(1), 1.0)

        # 每个类的样本数: [C_batch]
        class_counts = one_hot.sum(dim=0).clamp(min=1)

        # centroids: [C_batch, feat_dim] = [C_batch, B] @ [B, feat_dim]
        centroids = one_hot.T @ features / class_counts.unsqueeze(1)

        # 归一化 centroids（对应 w_r = \bar z_r / ||\bar z_r||）
        if self.normalize:
            centroids = F.normalize(centroids, p=2, dim=1)

        # 不让梯度从 centroids 反传回 features（与论文实现一致）
        centroids = centroids.detach()

        # 3) logits: z^T w_r / τ₁ (Eq. 8)
        #    features:  [B, feat_dim]
        #    centroids: [C_batch, feat_dim]
        logits = features @ centroids.T / self.tau1  # [B, C_batch]

        # 4) p_θ(r|z) (softmax over classes, Eq. 8)
        p_theta = F.softmax(logits, dim=1)  # [B, C_batch]
        # 当前样本对应真实类的置信度 p_θ(r|z): [B]
        confidences = p_theta.gather(1, inverse_indices.unsqueeze(1)).squeeze(1)

        # 5) τ₂ (Eq. 13)
        eps = self.epsilon
        tau2 = (self.tau1 - eps) + 2.0 * eps * confidences  # [B]
        tau2 = torch.clamp(tau2, min=1e-6)

        # 6) q_φ (Eq. 10, 11, 12)，复用 one_hot
        # q_exp: 对真类是 exp(1/τ₂)，其余类是 1
        exp_inv_tau2 = torch.exp(1.0 / tau2)               # [B]
        q_exp = 1.0 + (exp_inv_tau2.unsqueeze(1) - 1.0) * one_hot  # [B, C_batch]
        q_phi = q_exp / q_exp.sum(dim=1, keepdim=True)     # 归一化到概率

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
            'epsilon': eps.detach() if torch.is_tensor(eps) else torch.tensor(eps),
            'all_tau2s': tau2.detach(),
        }
