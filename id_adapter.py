"""
ID Adapter: semantic-level ID feature → SwapNet cross-attention
ArcFace embedding → Linear → LN → SwapNet KV matrix에 주입
"""

import torch
import torch.nn as nn


class IDAdapter(nn.Module):
    """
    DreamID 논문 구조:
    A1 → ID Encoder(freeze) → 512d embedding
    → Linear → LN → SwapNet cross-attention KV에 추가
    """
    def __init__(self, id_embed_dim=512, cross_attention_dim=1024, num_tokens=4):
        """
        Args:
            id_embed_dim: ArcFace embedding 차원 (512)
            cross_attention_dim: SwapNet UNet cross-attention 차원
            num_tokens: projection할 토큰 수
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.id_embed_dim = id_embed_dim

        # [B, 512] → [B, num_tokens * cross_attention_dim]
        self.proj = nn.Linear(id_embed_dim, num_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            id_embedding: [B, 512] ArcFace embedding
        Returns:
            [B, num_tokens, cross_attention_dim]
        """
        # [B, 512] → [B, num_tokens * cross_attention_dim]
        x = self.proj(id_embedding)
        # [B, num_tokens * cross_attention_dim] → [B, num_tokens, cross_attention_dim]
        B = id_embedding.shape[0]
        cross_dim = x.shape[-1] // self.num_tokens
        x = x.view(B, self.num_tokens, cross_dim)
        x = self.norm(x)
        return x