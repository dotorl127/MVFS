"""
2단계: 특정 인물 X LoRA 파인튜닝
베이스 체크포인트 위에 SwapNet + ID Adapter LoRA 학습
FaceNet은 freeze 유지

실행:
python train_lora.py \
    --data_dir ../finetune_dataset \
    --output_dir ./lora_checkpoints \
    --base_checkpoint ./checkpoints/checkpoint_XXXXX.pt \
    --lora_rank 32 \
    --lr 5e-6 \
    --max_steps 2000 \
    --save_steps 500
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.checkpoint import checkpoint
import numpy as np
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    USE_8BIT_ADAM = True
except ImportError:
    USE_8BIT_ADAM = False
    print("bitsandbytes not found, using standard AdamW")

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import open_clip
from insightface.app import FaceAnalysis

sys.path.append(str(Path(__file__).parent.parent))
from swap_net import SwapNet
from id_adapter import IDAdapter
from losses import TotalLoss
from dataset_image import get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LoRA 레이어
# ─────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Linear 레이어에 LoRA 적용"""
    def __init__(self, original: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)


def inject_lora(model: nn.Module, rank: int, alpha: float = 1.0, target_modules=None):
    """
    모델의 Linear 레이어에 LoRA 주입
    target_modules: 적용할 레이어 이름 키워드 (None이면 모든 Linear)
    """
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]

    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model if not parent_name else dict(model.named_modules())[parent_name]
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                lora_params.extend(list(lora_layer.lora_A.parameters()))
                lora_params.extend(list(lora_layer.lora_B.parameters()))

    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """LoRA 가중치만 저장"""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.weight.data
            lora_state[f"{name}.lora_B"] = module.lora_B.weight.data
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str, device):
    """LoRA 가중치 로드"""
    lora_state = torch.load(path, map_location=device)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in lora_state:
                module.lora_A.weight.data = lora_state[f"{name}.lora_A"].to(device)
                module.lora_B.weight.data = lora_state[f"{name}.lora_B"].to(device)


# ─────────────────────────────────────────────
# 공통 유틸 (train_ifs.py와 동일)
# ─────────────────────────────────────────────
def get_text_embedding(clip_model, tokenizer, device, prompt="cinematic full head portrait"):
    tokens = tokenizer([prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    return text_features


def get_id_embedding(face_app, images_np):
    embeddings = []
    for img in images_np:
        faces = face_app.get(img)
        if faces:
            embeddings.append(torch.tensor(faces[0].embedding))
        else:
            embeddings.append(torch.zeros(512))
    return torch.stack(embeddings)


def encode_image(vae, image_tensor):
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent


def decode_latent(vae, latent, no_grad=False):
    latent = latent / vae.config.scaling_factor
    if no_grad:
        with torch.no_grad():
            image = vae.decode(latent).sample
    else:
        image = checkpoint(vae.decode, latent, use_reentrant=False).sample
    return image


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def run_validation(models, vae, text_embedding, batch, device, save_dir, step):
    models["swapnet"].eval()
    models["facenet"].eval()
    models["id_adapter"].eval()

    a1 = batch["a1"][:1].to(device)
    a2 = batch["a2"][:1].to(device)
    b_prime = batch["b_prime"][:1].to(device)
    B = a1.shape[0]

    a1_latent = encode_image(vae, a1)
    b_latent  = encode_image(vae, b_prime)
    noisy_latent = torch.randn_like(b_latent)
    timesteps = torch.full((B,), 999, dtype=torch.long, device=device)

    a1_np = ((a1.cpu().permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    a1_id_embed = get_id_embedding(models["face_app"], a1_np).to(device)
    id_features = models["id_adapter"](a1_id_embed)

    with torch.no_grad():
        facenet_features = models["facenet"](
            a1_latent, timesteps,
            text_embedding.expand(B, -1, -1),
        )
    facenet_features = [f.detach() for f in facenet_features]

    noise_pred = models["swapnet"](
        noisy_latent, b_latent, timesteps,
        encoder_hidden_states=text_embedding.expand(B, -1, -1),
        facenet_features=facenet_features,
        id_features=id_features,
    )

    pred_x0 = models["noise_scheduler"].step(noise_pred, 999, noisy_latent).pred_original_sample
    generated = decode_latent(vae, pred_x0, no_grad=True).clamp(-1, 1)

    grid = torch.cat([a1, b_prime, generated, a2], dim=0)
    os.makedirs(save_dir, exist_ok=True)
    save_image(grid * 0.5 + 0.5, os.path.join(save_dir, f"val_{step:07d}.png"), nrow=4)

    models["swapnet"].train()
    models["facenet"].train()
    models["id_adapter"].train()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora_checkpoints")
    parser.add_argument("--base_checkpoint", type=str, required=True,
                        help="베이스 학습 체크포인트 경로 (train_ifs.py 결과)")
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--preview_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--resume_lora", type=str, default=None,
                        help="이어서 학습할 LoRA 체크포인트 경로")
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 모델 초기화 (train_ifs와 동일) ──
    logger.info("Loading SD-Turbo...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    swapnet = SwapNet(unet).to(device)
    swapnet.unet.enable_gradient_checkpointing()

    facenet = swapnet.build_facenet().to(device)
    facenet.unet.enable_gradient_checkpointing()
    for param in facenet.parameters():
        param.requires_grad_(False)

    id_adapter = IDAdapter(
        id_embed_dim=512,
        cross_attention_dim=unet.config.cross_attention_dim,
        num_tokens=4
    ).to(device)

    # ── 베이스 체크포인트 로드 ──
    logger.info(f"Loading base checkpoint: {args.base_checkpoint}")
    base_ckpt = torch.load(args.base_checkpoint, map_location=device)
    swapnet.load_state_dict(base_ckpt["swapnet"])
    facenet.load_state_dict(base_ckpt["facenet"])
    id_adapter.load_state_dict(base_ckpt["id_adapter"])
    logger.info("Base checkpoint loaded")

    # ── SwapNet에 LoRA 주입 ──
    logger.info(f"Injecting LoRA (rank={args.lora_rank})...")
    swapnet_lora_params = inject_lora(swapnet, rank=args.lora_rank, alpha=args.lora_alpha)

    # SwapNet base 가중치 freeze (LoRA만 학습)
    for name, param in swapnet.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)

    # ID Adapter는 전체 학습 (파라미터 작음)
    for param in id_adapter.parameters():
        param.requires_grad_(True)

    # ── 학습 파라미터 ──
    trainable_params = swapnet_lora_params + list(id_adapter.parameters())
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M "
                f"(LoRA: {sum(p.numel() for p in swapnet_lora_params) / 1e6:.2f}M)")

    # ── OpenCLIP ──
    logger.info("Loading OpenCLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k'
    )
    clip_model = clip_model.to(device)
    clip_model.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    # ── InsightFace ──
    logger.info("Loading InsightFace...")
    face_app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0, det_size=(512, 512))

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    models = {
        "vae": vae,
        "swapnet": swapnet,
        "facenet": facenet,
        "id_adapter": id_adapter,
        "clip_model": clip_model,
        "tokenizer": tokenizer,
        "face_app": face_app,
        "noise_scheduler": noise_scheduler
    }

    # ── 텍스트 임베딩 캐싱 ──
    text_embedding = get_text_embedding(clip_model, tokenizer, device)
    logger.info("Text embedding cached")

    # ── Optimizer ──
    if USE_8BIT_ADAM:
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
        logger.info("Using 8-bit AdamW")
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    use_amp = args.mixed_precision and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Resume LoRA ──
    global_step = 0
    if args.resume_lora:
        logger.info(f"Resuming LoRA from: {args.resume_lora}")
        lora_ckpt = torch.load(args.resume_lora, map_location=device)
        load_lora_weights(swapnet, args.resume_lora, device)
        if "id_adapter" in lora_ckpt:
            id_adapter.load_state_dict(lora_ckpt["id_adapter"])
        if "optimizer" in lora_ckpt:
            optimizer.load_state_dict(lora_ckpt["optimizer"])
        if "step" in lora_ckpt:
            global_step = lora_ckpt["step"]
        logger.info(f"Resumed from step {global_step}")

    # ── 데이터로더 ──
    dataloader = get_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=True,   # 파인튜닝 시 도메인 aug 적용
        num_workers=args.num_workers
    )

    criterion = TotalLoss(lambda_rec=10.0, lambda_id=1.0, lambda_dm=1.0)

    swapnet.train()
    facenet.eval()   # FaceNet은 항상 eval
    id_adapter.train()

    optimizer.zero_grad()
    loss_accum = {"total": 0, "rec": 0, "id": 0, "dm": 0}
    pbar = tqdm(total=args.max_steps, initial=global_step)

    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            a1 = batch["a1"].to(device)
            a2 = batch["a2"].to(device)
            b_prime = batch["b_prime"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                B = a1.shape[0]

                a1_latent = encode_image(vae, a1)
                a2_latent = encode_image(vae, a2)
                b_latent  = encode_image(vae, b_prime)

                noise = torch.randn_like(b_latent)
                noisy_latent = noise
                timesteps = torch.full((B,), 999, dtype=torch.long, device=device)

                # ID embedding
                if batch["a1_embedding"].abs().sum() > 0:
                    a1_id_embed = batch["a1_embedding"].to(device)
                else:
                    a1_np = ((a1.cpu().permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    a1_id_embed = get_id_embedding(face_app, a1_np).to(device)
                id_features = id_adapter(a1_id_embed)

                # FaceNet (freeze + no_grad)
                with torch.no_grad():
                    facenet_features = facenet(
                        a1_latent, timesteps,
                        text_embedding.expand(B, -1, -1),
                    )
                facenet_features = [f.detach() for f in facenet_features]

                # SwapNet forward (LoRA 적용된 상태)
                noise_pred = swapnet(
                    noisy_latent, b_latent, timesteps,
                    encoder_hidden_states=text_embedding.expand(B, -1, -1),
                    facenet_features=facenet_features,
                    id_features=id_features,
                )

                pred_x0 = noise_scheduler.step(noise_pred, 999, noisy_latent).pred_original_sample
                generated = decode_latent(vae, pred_x0, no_grad=False)

                gen_np = ((generated.detach().cpu().permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                gen_id_embed = get_id_embedding(face_app, gen_np).to(device)

                pred_type = noise_scheduler.config.prediction_type
                if pred_type == "v_prediction":
                    noise_target = noise_scheduler.get_velocity(a2_latent, noise, timesteps)
                elif pred_type == "epsilon":
                    noise_target = noise
                else:
                    noise_target = a2_latent

                losses = criterion(
                    generated=generated, gt=a2,
                    generated_embedding=gen_id_embed,
                    source_embedding=a1_id_embed,
                    noise_pred=noise_pred,
                    noise_target=noise_target,
                )

                loss = losses["total"] / args.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            for k in loss_accum:
                loss_accum[k] += losses[k].item() / args.grad_accum

            if (global_step + 1) % args.grad_accum == 0:
                lora_all_params = swapnet_lora_params + list(id_adapter.parameters())
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(lora_all_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            pbar.update(1)

            if global_step % args.log_steps == 0:
                log_str = f"Step {global_step} | "
                log_str += " | ".join([f"{k}: {v:.4f}" for k, v in loss_accum.items()])
                pbar.set_description(log_str)
                loss_accum = {k: 0 for k in loss_accum}

            if global_step % args.save_steps == 0:
                # LoRA 가중치만 저장
                save_path = os.path.join(args.output_dir, f"lora_{global_step}.pt")
                lora_state = {}
                for name, module in swapnet.named_modules():
                    if isinstance(module, LoRALinear):
                        lora_state[f"{name}.lora_A"] = module.lora_A.weight.data
                        lora_state[f"{name}.lora_B"] = module.lora_B.weight.data
                torch.save({
                    "step": global_step,
                    "lora": lora_state,
                    "id_adapter": id_adapter.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                }, save_path)
                logger.info(f"Saved LoRA: {save_path}")

            if global_step % args.preview_steps == 0:
                run_validation(
                    models=models, vae=vae,
                    text_embedding=text_embedding,
                    batch=batch, device=device,
                    save_dir=os.path.join(args.output_dir, "validation"),
                    step=global_step
                )

    pbar.close()
    logger.info("LoRA fine-tuning complete")


if __name__ == "__main__":
    args = parse_args()
    train(args)
