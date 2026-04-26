"""
이미지 Triplet 데이터셋

디렉토리 구조:
dataset/
  identity_001/
    A1/          ← 대표 이미지 1장
      ref.jpg
    A2/          ← 멀티앵글 GT 이미지들
      angle_front.jpg
      angle_left.jpg
      angle_right.jpg
      ...
    B_/          ← A2 이미지별 가짜 타겟 (FaceDancer 스왑 결과)
      angle_front/
        swap_001.jpg   ← 타인 001을 소스로 angle_front에 스왑
        swap_002.jpg
        ...
      angle_left/
        swap_001.jpg
        ...
  identity_002/
    ...
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TripletDataset(Dataset):
    """
    (A1, B', A2) triplet 데이터셋

    A1: 대표 이미지 1장 (ref)
    B': 가짜 타겟 (타인을 소스로 A2에 스왑한 결과)
    A2: GT 이미지 (멀티앵글)
    """
    def __init__(self, root_dir, image_size=512, augment=False):
        """
        Args:
            root_dir: 데이터셋 루트 디렉토리
            image_size: 출력 이미지 크기
            augment: X LoRA 파인튜닝 시 도메인 aug 적용 여부
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        self.samples = []

        self._build_samples()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if augment:
            self.aug_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def _build_samples(self):
        """
        각 identity 디렉토리를 순회하며 (A1, B', A2) triplet 수집
        """
        for identity_dir in sorted(self.root_dir.iterdir()):
            if not identity_dir.is_dir():
                continue

            a1_dir = identity_dir / "A1"
            a2_dir = identity_dir / "A2"
            b_dir = identity_dir / "B_"

            if not (a1_dir.exists() and a2_dir.exists() and b_dir.exists()):
                continue

            # A1: 대표 이미지 1장
            a1_images = list(a1_dir.glob("*.jpg")) + list(a1_dir.glob("*.png"))
            if not a1_images:
                continue
            a1_path = a1_images[0]

            # A2 이미지별로 B_ 매핑
            for a2_img in sorted(a2_dir.iterdir()):
                if a2_img.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                    continue

                # 해당 A2에 대응하는 B_ 디렉토리
                b_subdir = b_dir / a2_img.stem #/ 'trash'
                if not b_subdir.exists():
                    continue

                b_images = list(b_subdir.glob("*.jpg")) + list(b_subdir.glob("*.png"))
                if not b_images:
                    continue

                # 각 B' 이미지에 대해 triplet 생성
                for b_img in b_images:
                    self.samples.append({
                        "a1": a1_path,
                        "a2": a2_img,
                        "b_prime": b_img,
                        "identity": identity_dir.name
                    })

        print(f"Total triplets: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        a1 = Image.open(sample["a1"]).convert("RGB")
        a2 = Image.open(sample["a2"]).convert("RGB")
        b_prime = Image.open(sample["b_prime"]).convert("RGB")

        # A1 (ref): augment 적용 시 도메인 aug
        if self.augment:
            a1 = self.aug_transform(a1)
        else:
            a1 = self.transform(a1)

        # A2 (GT): 항상 기본 transform
        a2 = self.transform(a2)

        # B' (가짜 타겟): 기본 transform
        b_prime = self.transform(b_prime)

        return {
            "a1": a1,           # ref 이미지
            "a2": a2,           # GT 이미지
            "b_prime": b_prime, # 가짜 타겟
            "identity": sample["identity"],
            "a1_embedding": self._load_a1_embedding(sample["a1"]),  # 캐싱된 ID embedding
        }

    def _load_a1_embedding(self, a1_path: Path) -> torch.Tensor:
        """
        A1 이미지와 같은 경로의 .npy 파일에서 ArcFace embedding 로드
        없으면 zero tensor 반환 (학습 루프에서 fallback 처리)
        """
        npy_path = a1_path.with_suffix('.npy')
        if npy_path.exists():
            return torch.tensor(np.load(npy_path), dtype=torch.float32)
        return torch.zeros(512, dtype=torch.float32)  # fallback


def get_dataloader(root_dir, batch_size=1, image_size=512,
                   augment=False, num_workers=4, shuffle=True):
    dataset = TripletDataset(root_dir, image_size=image_size, augment=augment)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )