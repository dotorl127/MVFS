"""
SwapNet: SD-Turbo UNet 기반 face swap 생성 주체
FaceNet: SwapNet 가중치 복사본, ref pixel-level ID feature 추출 후 SwapNet self-attention에 주입

DreamID 논문 구조:
  - FaceNet feature (h) + SwapNet feature (h) → concat (2h) → self-attention → 앞쪽 h만 SwapNet으로 전달
  - ID Adapter feature → SwapNet cross-attention KV에 추가
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from copy import deepcopy
from typing import Optional, List


# ─────────────────────────────────────────────
# FaceNet feature를 self-attention에 주입하는 custom processor
# ─────────────────────────────────────────────
class FaceNetAttentionProcessor:
    """
    DreamID FaceNet feature injection:
      self-attn: concat(facenet_feat, swapnet_feat) → self-attention → 앞쪽 절반만 사용
      cross-attn: text + id_features → KV concat
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

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            # ── Cross-attention: text + id_features KV concat ──
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

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        else:
            # ── Self-attention: FaceNet feature concat ──
            if (
                facenet_features is not None
                and self.layer_idx < len(facenet_features)
                and facenet_features[self.layer_idx] is not None
            ):
                facenet_feat = facenet_features[self.layer_idx]
                if facenet_feat.shape == hidden_states.shape:
                    # [B, S, C] + [B, S, C] → [B, 2S, C]
                    concat_states = torch.cat([hidden_states, facenet_feat], dim=1)
                else:
                    concat_states = hidden_states
            else:
                concat_states = hidden_states

            # query는 swapnet hidden_states만, key/value는 concat
            query = attn.to_q(hidden_states)
            key = attn.to_k(concat_states)
            value = attn.to_v(concat_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_facenet_processors(unet: UNet2DConditionModel):
    """UNet의 모든 Attention layer에 FaceNetAttentionProcessor 등록"""
    idx = 0
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(FaceNetAttentionProcessor(layer_idx=idx))
            idx += 1


# ─────────────────────────────────────────────
# FaceNet: ref 이미지 → pixel-level feature 추출
# ─────────────────────────────────────────────
class FaceNet(nn.Module):
    """
    SwapNet UNet 가중치 복사본 (ReferenceNet 역할)
    ref A1 → self-attention layer별 hidden_states 수집
    """

    def __init__(self, swapnet_unet: UNet2DConditionModel):
        super().__init__()
        self.unet = deepcopy(swapnet_unet)
        self._reset_input_channels()

    def _reset_input_channels(self):
        """FaceNet은 ref latent 4채널만 받음 (8채널 → 4채널)"""
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
        # 8채널 중 앞 4채널만 복사
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
        forward hook으로 각 self-attention의 input hidden_states 수집
        Returns:
            features: List[Tensor] 각 self-attention layer의 hidden_states
        """
        features = []
        hooks = []

        def make_hook():
            def hook(module, args, kwargs_hook, output):
                # self-attention (attn1)의 input hidden_states 저장
                if args:
                    features.append(args[0].clone())
            return hook

        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and "attn1" in name:
                hooks.append(
                    module.register_forward_hook(make_hook(), with_kwargs=True)
                )

        with torch.no_grad():
            self.unet(
                ref_latent,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        for h in hooks:
            h.remove()

        return features


# ─────────────────────────────────────────────
# SwapNet: face swap 생성 주체
# ─────────────────────────────────────────────
class SwapNet(nn.Module):
    """
    SD-Turbo UNet 기반 SwapNet
    입력: [noisy(4ch) | ε(B')(4ch)] concat → 8채널
    """

    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet
        self._expand_input_channels()
        register_facenet_processors(self.unet)

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
        """
        Args:
            noisy_latent:          [B, 4, H, W] 순수 가우시안 노이즈 (t=999)
            target_latent:         [B, 4, H, W] ε(B') VAE encoded
            timestep:              [B] 999 고정
            encoder_hidden_states: [B, seq, dim] text embedding
            facenet_features:      List[Tensor] FaceNet self-attn hidden_states
            id_features:           [B, num_tokens, dim] ID Adapter semantic features
        Returns:
            noise_pred: [B, 4, H, W]
        """
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