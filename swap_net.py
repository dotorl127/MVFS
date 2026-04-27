"""
DreamID 논문 구조:

FaceNet:
  - SwapNet과 동일한 UNet 구조 (전체, encoder+decoder)
  - ref A1 latent 입력 (4채널)
  - 각 attn1 레이어의 hidden_states를 hook으로 수집
  - 기본 attention processor 사용 (FaceNetAttentionProcessor 아님)

SwapNet:
  - SD-Turbo UNet (전체)
  - 입력: [noisy(4ch) | ε(B')(4ch)] concat → 8채널
  - FaceNetAttentionProcessor 등록
  - 각 attn1 레이어에서:
      FaceNet feature(h) + SwapNet feature(h) → concat(2h)
      → self-attention
      → FaceNet feature(h)만 output으로 전달 (논문: pixel-level ID Features)
  - 각 attn2 레이어에서:
      text + id_features → KV concat → cross-attention
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from copy import deepcopy
from typing import Optional, List

try:
    import xformers.ops as xops
    USE_XFORMERS = True
except ImportError:
    USE_XFORMERS = False


class FaceNetAttentionProcessor:
    """
    SwapNet의 각 attention layer에 등록되는 processor

    attn1 (self-attention):
        FaceNet feature(h) + SwapNet feature(h) → concat(2h)
        → self-attention
        → FaceNet feature 부분(h)만 output (pixel-level ID Features)

    attn2 (cross-attention):
        text + id_features → KV concat → cross-attention
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        facenet_features: Optional[List[torch.Tensor]] = None,
        id_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        is_cross = encoder_hidden_states is not None

        if is_cross:
            # ── attn2: text + id_features KV concat ──
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            if id_features is not None:
                id_key = attn.to_k(id_features)
                id_value = attn.to_v(id_features)
                key = torch.cat([key, id_key], dim=1)
                value = torch.cat([value, id_value], dim=1)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            if USE_XFORMERS:
                hidden_states = xops.memory_efficient_attention(query, key, value)
            else:
                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                hidden_states = torch.bmm(attention_probs, value)

            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        else:
            # ── attn1: FaceNet feature concat → self-attention ──
            facenet_feat = None
            if (
                facenet_features is not None
                and self.layer_idx < len(facenet_features)
            ):
                f = facenet_features[self.layer_idx]
                if f is not None and f.shape == hidden_states.shape:
                    facenet_feat = f

            if facenet_feat is not None:
                # [FaceNet(h) | SwapNet(h)] → 2h
                concat_states = torch.cat([facenet_feat, hidden_states], dim=1)

                # query: FaceNet feature (h)
                # key/value: concat (2h)
                query = attn.to_q(facenet_feat)
                key = attn.to_k(concat_states)
                value = attn.to_v(concat_states)

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                if USE_XFORMERS:
                    hidden_states = xops.memory_efficient_attention(query, key, value)
                else:
                    attention_probs = attn.get_attention_scores(query, key, attention_mask)
                    hidden_states = torch.bmm(attention_probs, value)

                hidden_states = attn.batch_to_head_dim(hidden_states)

            else:
                # FaceNet feature 없을 때 일반 self-attention
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                if USE_XFORMERS:
                    hidden_states = xops.memory_efficient_attention(query, key, value)
                else:
                    attention_probs = attn.get_attention_scores(query, key, attention_mask)
                    hidden_states = torch.bmm(attention_probs, value)

                hidden_states = attn.batch_to_head_dim(hidden_states)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_facenet_processors(unet: UNet2DConditionModel):
    """SwapNet UNet의 모든 Attention layer에 FaceNetAttentionProcessor 등록"""
    idx = 0
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(FaceNetAttentionProcessor(layer_idx=idx))
            idx += 1


class FaceNet(nn.Module):
    """
    SwapNet과 동일한 UNet 구조
    ref A1 → attn1 레이어별 hidden_states 수집 → SwapNet에 주입
    기본 attention processor 사용 (FaceNetAttentionProcessor 아님)
    """

    def __init__(self, swapnet: "SwapNet"):
        super().__init__()
        self.unet = deepcopy(swapnet.unet)
        self._reset_input_channels()
        # FaceNet은 기본 processor 사용 (FaceNetAttentionProcessor 제거)
        self.unet.set_attn_processor(AttnProcessor2_0())

    def _reset_input_channels(self):
        """8채널 conv_in → 4채널 (ref latent만 입력)"""
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
        new_conv.weight.data = old_conv.weight.data[:, :4, :, :].clone()
        new_conv.bias.data = old_conv.bias.data.clone()
        self.unet.conv_in = new_conv

    def forward(
        self,
        ref_latent: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        attn1 레이어의 hidden_states를 hook으로 수집
        Returns: List[Tensor] — attn1 레이어 순서대로
        """
        features = []
        hooks = []

        def make_hook():
            def hook(module, args, output):
                if args:
                    features.append(args[0])
            return hook

        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and "attn1" in name:
                hooks.append(module.register_forward_hook(make_hook()))

        self.unet(
            ref_latent,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

        for h in hooks:
            h.remove()

        return features


class SwapNet(nn.Module):
    """
    SD-Turbo UNet 기반 face swap 생성 주체
    입력: [noisy(4ch) | ε(B')(4ch)] → 8채널
    """

    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet
        self._expand_input_channels()
        register_facenet_processors(self.unet)

    def build_facenet(self) -> FaceNet:
        return FaceNet(self)

    def _expand_input_channels(self):
        """conv_in: 4채널 → 8채널, 추가 4채널 zero init"""
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            8,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
        new_conv.weight.data[:, :4, :, :] = old_conv.weight.data.clone()
        new_conv.weight.data[:, 4:, :, :] = 0.0
        new_conv.bias.data = old_conv.bias.data.clone()
        self.unet.conv_in = new_conv

    def forward(
        self,
        noisy_latent: torch.Tensor,
        target_latent: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        facenet_features: Optional[List[torch.Tensor]] = None,
        id_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.cat([noisy_latent, target_latent], dim=1)  # [B, 8, H, W]

        return self.unet(
            x,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={
                "facenet_features": facenet_features,
                "id_features": id_features,
            },
        ).sample