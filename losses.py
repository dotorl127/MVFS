"""
DreamID 논문 Loss 구현
- Reconstruction Loss (L2, weight=10)
- ID Loss (ArcFace cosine, weight=1)
- Diffusion Loss (noise prediction, weight=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    L_rec = ||Ã - A2||²  × 10
    생성 이미지와 GT(A2) 간 pixel-level L2 loss
    """
    def __init__(self, weight=10.0):
        super().__init__()
        self.weight = weight

    def forward(self, generated, gt):
        """
        Args:
            generated: [B, C, H, W] 생성된 이미지
            gt: [B, C, H, W] GT 이미지 A2
        """
        return self.weight * F.mse_loss(generated, gt)


class IDLoss(nn.Module):
    """
    L_id = 1 - cos(e_ID_A1, e_ID_Ã)  × 1
    생성 이미지와 source A1 간 identity cosine similarity loss
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, generated_embedding, source_embedding):
        """
        Args:
            generated_embedding: [B, 512] 생성 이미지 ArcFace embedding
            source_embedding: [B, 512] source A1 ArcFace embedding
        """
        cos_sim = F.cosine_similarity(generated_embedding, source_embedding, dim=-1)
        return self.weight * (1 - cos_sim).mean()


class DiffusionLoss(nn.Module):
    """
    L_DM = E[||ε - ε_θ(z_t, c, t)||²]  × 1
    SD-Turbo noise prediction loss (t=999 고정)
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, noise_pred, noise_target):
        """
        Args:
            noise_pred: [B, 4, H, W] 예측된 노이즈
            noise_target: [B, 4, H, W] 실제 노이즈
        """
        return self.weight * F.mse_loss(noise_pred, noise_target)


class TotalLoss(nn.Module):
    """
    L = λ_id * L_id + λ_DM * L_DM + λ_rec * L_rec
    """
    def __init__(self, lambda_rec=10.0, lambda_id=1.0, lambda_dm=1.0):
        super().__init__()
        self.rec_loss = ReconstructionLoss(weight=lambda_rec)
        self.id_loss = IDLoss(weight=lambda_id)
        self.diff_loss = DiffusionLoss(weight=lambda_dm)

    def forward(self, generated, gt, generated_embedding, source_embedding,
                noise_pred, noise_target):
        """
        Args:
            generated: 생성 이미지
            gt: GT 이미지 A2
            generated_embedding: 생성 이미지 ArcFace embedding
            source_embedding: source A1 ArcFace embedding
            noise_pred: 예측 노이즈
            noise_target: 실제 노이즈
        """
        l_rec = self.rec_loss(generated, gt)
        l_id = self.id_loss(generated_embedding, source_embedding)
        l_dm = self.diff_loss(noise_pred, noise_target)

        total = (l_rec
                 + l_id
                 + l_dm)

        return {
            "total": total,
            "rec": l_rec,
            "id": l_id,
            "dm": l_dm
        }
