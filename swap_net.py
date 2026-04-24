"""
SwapNet: SD-Turbo UNet 기반 face swap 생성 주체
FaceNet: SwapNet 가중치 복사본, ref pixel-level ID feature 추출
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from copy import deepcopy


class SwapNet(nn.Module):
    """
    SD-Turbo UNet 기반 SwapNet
    입력 채널 4 → 8로 확장 (noisy latent 4 + target latent 4)
    """
    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet
        self._expand_input_channels()

    def _expand_input_channels(self):
        """
        UNet 첫 conv를 4채널 → 8채널로 확장
        추가 4채널은 zero init (SD-Turbo pretrained 능력 보존)
        """
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            8,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        # 기존 가중치 복사
        new_conv.weight.data[:, :4, :, :] = old_conv.weight.data
        # 추가 4채널 zero init
        new_conv.weight.data[:, 4:, :, :] = 0
        new_conv.bias.data = old_conv.bias.data.clone()
        self.unet.conv_in = new_conv

    def forward(self, noisy_latent, target_latent, timestep, encoder_hidden_states, facenet_features=None):
        """
        Args:
            noisy_latent: [B, 4, H, W] - t=999 노이즈
            target_latent: [B, 4, H, W] - ε(B') VAE encoded
            timestep: diffusion timestep (999)
            encoder_hidden_states: text embedding "cinematic full head portrait"
            facenet_features: FaceNet에서 추출한 pixel-level features (self-attention용)
        """
        # concat noisy + target → 8채널
        x = torch.cat([noisy_latent, target_latent], dim=1)

        return self.unet(
            x,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={"facenet_features": facenet_features}
        ).sample


class FaceNet(nn.Module):
    """
    SwapNet 가중치 복사본 (ReferenceNet)
    ref A1 이미지 → pixel-level ID feature 추출
    SwapNet self-attention에 주입
    """
    def __init__(self, swapnet: SwapNet):
        super().__init__()
        # SwapNet UNet encoder 부분만 복사
        self.unet = deepcopy(swapnet.unet)
        # FaceNet은 encoder half만 사용
        # 입력 채널을 다시 4채널로 (ref 이미지만 받음)
        self._reset_input_channels()

    def _reset_input_channels(self):
        """FaceNet은 ref 이미지 1장만 받으므로 4채널로 리셋"""
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        # 원래 4채널 가중치 복사
        new_conv.weight.data = old_conv.weight.data[:, :4, :, :].clone()
        new_conv.bias.data = old_conv.bias.data.clone()
        self.unet.conv_in = new_conv

    def forward(self, ref_latent, timestep, encoder_hidden_states):
        """
        Args:
            ref_latent: [B, 4, H, W] - ε(A1) VAE encoded ref 이미지
            timestep: diffusion timestep
            encoder_hidden_states: text embedding
        Returns:
            pixel-level ID features (각 UNet block의 intermediate features)
        """
        # UNet encoder 통과하며 intermediate features 수집
        features = []

        # down blocks
        down_block_res_samples = (ref_latent,)
        for downsample_block in self.unet.down_blocks:
            ref_latent, res_samples = downsample_block(
                hidden_states=ref_latent,
                temb=self.unet.get_time_embed(sample=ref_latent, timestep=timestep),
                encoder_hidden_states=encoder_hidden_states,
            )
            down_block_res_samples += res_samples
            features.append(ref_latent)

        # mid block
        ref_latent = self.unet.mid_block(
            ref_latent,
            self.unet.get_time_embed(sample=ref_latent, timestep=timestep),
            encoder_hidden_states=encoder_hidden_states,
        )
        features.append(ref_latent)

        return features
