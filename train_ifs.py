"""
1단계: IFS 베이스 학습
SD-Turbo 기반 SwapNet + FaceNet + ID Adapter 학습

실행:
python training/train_ifs.py \
    --data_dir /path/to/dataset \
    --output_dir ./checkpoints \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-5 \
    --max_steps 70000 \
    --save_steps 1000
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

# bitsandbytes 8-bit Adam (3060 12GB 메모리 절약)
try:
    import bitsandbytes as bnb
    USE_8BIT_ADAM = True
except ImportError:
    USE_8BIT_ADAM = False
    print("bitsandbytes not found, using standard AdamW")

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.models import AutoencoderKL
import open_clip
import insightface
from insightface.app import FaceAnalysis

sys.path.append(str(Path(__file__).parent.parent))
from models.swap_net import SwapNet, FaceNet
from models.id_adapter import IDAdapter
from utils.losses import TotalLoss
from data.dataset_image import get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=70000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def setup_models(args, device):
    """모델 초기화"""
    logger.info("Loading SD-Turbo...")

    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae"
    ).to(device)
    vae.requires_grad_(False)

    # SD-Turbo UNet → SwapNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet"
    )
    swapnet = SwapNet(unet).to(device)
    swapnet.unet.enable_gradient_checkpointing()

    # FaceNet (SwapNet 가중치 복사)
    facenet = FaceNet(swapnet).to(device)
    facenet.unet.enable_gradient_checkpointing()

    # ID Adapter
    id_adapter = IDAdapter(
        id_embed_dim=512,
        cross_attention_dim=unet.config.cross_attention_dim,
        num_tokens=4
    ).to(device)

    # Text Encoder (OpenCLIP, freeze)
    logger.info("Loading OpenCLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained='laion2b_s32b_b79k'
    )
    clip_model = clip_model.to(device)
    clip_model.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    # ID Encoder (InsightFace ArcFace, freeze)
    logger.info("Loading InsightFace...")
    face_app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0, det_size=(512, 512))

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    return {
        "vae": vae,
        "swapnet": swapnet,
        "facenet": facenet,
        "id_adapter": id_adapter,
        "clip_model": clip_model,
        "tokenizer": tokenizer,
        "face_app": face_app,
        "noise_scheduler": noise_scheduler
    }


def get_text_embedding(clip_model, tokenizer, device, prompt="cinematic full head portrait"):
    """고정 텍스트 임베딩 캐싱"""
    tokens = tokenizer([prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    return text_features


def get_id_embedding(face_app, images_np):
    """
    ArcFace ID embedding 추출
    images_np: numpy array [B, H, W, C] uint8
    """
    embeddings = []
    for img in images_np:
        faces = face_app.get(img)
        if faces:
            embeddings.append(torch.tensor(faces[0].embedding))
        else:
            # 얼굴 감지 실패 시 zero embedding
            embeddings.append(torch.zeros(512))
    return torch.stack(embeddings)


def encode_image(vae, image_tensor):
    """이미지 → VAE latent"""
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent


def decode_latent(vae, latent):
    """VAE latent → 이미지"""
    latent = latent / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latent).sample
    return image


def setup_optimizer(models, lr, use_8bit=True):
    """학습 가능한 파라미터 설정 및 optimizer 초기화"""
    trainable_params = []

    # SwapNet, FaceNet, ID Adapter 학습
    # ID Encoder, VAE, CLIP은 freeze
    for name, param in models["swapnet"].named_parameters():
        param.requires_grad_(True)
        trainable_params.append(param)

    for name, param in models["facenet"].named_parameters():
        param.requires_grad_(True)
        trainable_params.append(param)

    for name, param in models["id_adapter"].named_parameters():
        param.requires_grad_(True)
        trainable_params.append(param)

    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    if use_8bit and USE_8BIT_ADAM:
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )
        logger.info("Using 8-bit AdamW")
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )

    return optimizer


@torch.no_grad()
def run_validation(models, vae, text_embedding, batch, device, save_dir, step):
    """
    학습 중 추론 결과 시각화
    A1(ref) | B'(가짜타겟) | Generated | A2(GT) 순으로 저장
    """
    models["swapnet"].eval()
    models["facenet"].eval()
    models["id_adapter"].eval()

    a1 = batch["a1"][:1].to(device)
    a2 = batch["a2"][:1].to(device)
    b_prime = batch["b_prime"][:1].to(device)

    # VAE encode
    a1_latent = encode_image(vae, a1)
    a2_latent = encode_image(vae, a2)
    b_latent = encode_image(vae, b_prime)

    # Noisy latent (t=999)
    noise = torch.randn_like(a2_latent)
    timesteps = torch.full((1,), 999, dtype=torch.long, device=device)
    noisy_latent = models["noise_scheduler"].add_noise(a2_latent, noise, timesteps)

    # ID embedding
    a1_np = ((a1.cpu().permute(0,2,3,1).numpy()+1)*127.5).astype(np.uint8)
    a1_id_embed = get_id_embedding(models["face_app"], a1_np).to(device)
    id_features = models["id_adapter"](a1_id_embed)

    # FaceNet features
    facenet_features = models["facenet"](
        a1_latent, timesteps,
        text_embedding.expand(1, -1, -1)
    )

    # SwapNet forward
    noise_pred = models["swapnet"](
        noisy_latent, b_latent, timesteps,
        encoder_hidden_states=text_embedding.expand(1, -1, -1),
        facenet_features=facenet_features
    )

    # x0 prediction → decode
    alpha_prod_t = models["noise_scheduler"].alphas_cumprod[999].to(device)
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (noisy_latent - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
    generated = decode_latent(vae, pred_x0).clamp(-1, 1)

    # [A1 | B' | Generated | A2] 나란히 저장
    grid = torch.cat([a1, b_prime, generated, a2], dim=0)
    os.makedirs(save_dir, exist_ok=True)
    save_image(grid * 0.5 + 0.5, os.path.join(save_dir, f"val_{step:07d}.png"), nrow=4)

    models["swapnet"].train()
    models["facenet"].train()
    models["id_adapter"].train()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 로드
    models = setup_models(args, device)

    # 텍스트 임베딩 캐싱 (고정 프롬프트)
    text_embedding = get_text_embedding(
        models["clip_model"],
        models["tokenizer"],
        device
    )
    logger.info("Text embedding cached")

    # Optimizer
    optimizer = setup_optimizer(models, args.lr)

    # Mixed precision
    scaler = GradScaler() if args.mixed_precision else None

    # 데이터로더
    dataloader = get_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    # Loss
    criterion = TotalLoss(lambda_rec=10.0, lambda_id=1.0, lambda_dm=1.0)

    # Resume
    global_step = 0
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        models["swapnet"].load_state_dict(checkpoint["swapnet"])
        models["facenet"].load_state_dict(checkpoint["facenet"])
        models["id_adapter"].load_state_dict(checkpoint["id_adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
        logger.info(f"Resumed from step {global_step}")

    # 학습 루프
    models["swapnet"].train()
    models["facenet"].train()
    models["id_adapter"].train()

    optimizer.zero_grad()
    loss_accum = {"total": 0, "rec": 0, "id": 0, "dm": 0}

    pbar = tqdm(total=args.max_steps, initial=global_step)

    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            a1 = batch["a1"].to(device)      # [B, 3, H, W] ref
            a2 = batch["a2"].to(device)      # [B, 3, H, W] GT
            b_prime = batch["b_prime"].to(device)  # [B, 3, H, W] 가짜 타겟

            with autocast(enabled=args.mixed_precision):

                # 1. VAE encode
                a1_latent = encode_image(models["vae"], a1)      # ε(A1)
                a2_latent = encode_image(models["vae"], a2)      # ε(A2) GT
                b_latent = encode_image(models["vae"], b_prime)   # ε(B')

                # 2. Noisy latent 생성 (t=999 고정)
                noise = torch.randn_like(a2_latent)
                timesteps = torch.full(
                    (a1.shape[0],), 999,
                    dtype=torch.long, device=device
                )
                noisy_latent = models["noise_scheduler"].add_noise(
                    a2_latent, noise, timesteps
                )

                # 3. ID embedding (A1)
                a1_np = ((a1.cpu().permute(0, 2, 3, 1).numpy() + 1) * 127.5).astype(np.uint8)
                a1_id_embed = get_id_embedding(models["face_app"], a1_np).to(device)
                id_features = models["id_adapter"](a1_id_embed)

                # 4. FaceNet: A1 pixel-level features
                facenet_features = models["facenet"](
                    a1_latent,
                    timesteps,
                    text_embedding.expand(a1.shape[0], -1, -1)
                )

                # 5. SwapNet forward
                noise_pred = models["swapnet"](
                    noisy_latent,
                    b_latent,
                    timesteps,
                    encoder_hidden_states=text_embedding.expand(a1.shape[0], -1, -1),
                    facenet_features=facenet_features
                )

                # 6. 1-step denoising → clean image
                # SD-Turbo 1-step: t=999에서 바로 x0 예측
                alpha_prod_t = models["noise_scheduler"].alphas_cumprod[999].to(device)
                beta_prod_t = 1 - alpha_prod_t
                pred_x0 = (noisy_latent - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

                # 7. VAE decode → generated image
                generated = decode_latent(models["vae"], pred_x0)

                # 8. Generated image ID embedding
                gen_np = ((generated.cpu().detach().permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                gen_id_embed = get_id_embedding(models["face_app"], gen_np).to(device)

                # 9. Loss 계산
                losses = criterion(
                    generated=generated,
                    gt=a2,
                    generated_embedding=gen_id_embed,
                    source_embedding=a1_id_embed,
                    noise_pred=noise_pred,
                    noise_target=noise
                )

                loss = losses["total"] / args.grad_accum

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Loss 누적
            for k in loss_accum:
                loss_accum[k] += losses[k].item() / args.grad_accum

            # Gradient accumulation
            if (global_step + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(models["swapnet"].parameters()) +
                        list(models["facenet"].parameters()) +
                        list(models["id_adapter"].parameters()),
                        1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(models["swapnet"].parameters()) +
                        list(models["facenet"].parameters()) +
                        list(models["id_adapter"].parameters()),
                        1.0
                    )
                    optimizer.step()

                optimizer.zero_grad()

            global_step += 1
            pbar.update(1)

            # 로깅
            if global_step % args.log_steps == 0:
                log_str = f"Step {global_step} | "
                log_str += " | ".join([f"{k}: {v:.4f}" for k, v in loss_accum.items()])
                pbar.set_description(log_str)
                loss_accum = {k: 0 for k in loss_accum}

            # 체크포인트 저장
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "swapnet": models["swapnet"].state_dict(),
                    "facenet": models["facenet"].state_dict(),
                    "id_adapter": models["id_adapter"].state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                logger.info(f"Saved checkpoint: {save_path}")

                run_validation(
                    models=models,
                    vae=models["vae"],
                    text_embedding=text_embedding,
                    batch=batch,  # 현재 배치 재사용
                    device=device,
                    save_dir=os.path.join(args.output_dir, "validation"),
                    step=global_step
                )

    pbar.close()
    logger.info("Training complete")


if __name__ == "__main__":
    args = parse_args()
    train(args)
