"""
A1 이미지의 ArcFace embedding을 미리 계산해서 .npy로 저장
학습 전 한 번만 실행하면 됨

실행:
python precompute_embeddings.py --data_dir ../mfvs_dataset
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--det_size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.data_dir)

    app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    identity_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    success, failed, skipped = 0, 0, 0

    for identity_dir in tqdm(identity_dirs, desc="Precomputing embeddings"):
        a1_dir = identity_dir / "A1"
        if not a1_dir.exists():
            continue

        a1_images = list(a1_dir.glob("*.jpg")) + list(a1_dir.glob("*.png"))
        if not a1_images:
            continue

        for a1_path in a1_images:
            npy_path = a1_path.with_suffix('.npy')

            # 이미 존재하면 스킵
            if npy_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(a1_path))
            if img is None:
                failed += 1
                continue

            faces = app.get(img)
            if not faces:
                # 패딩 추가 후 재시도
                img_padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
                faces = app.get(img_padded)

            if faces:
                emb = faces[0].normed_embedding.astype(np.float32)  # [512]
                np.save(str(npy_path), emb)
                success += 1
            else:
                failed += 1
                print(f"[WARN] 얼굴 감지 실패: {a1_path}")

    print(f"\n완료: 성공 {success} / 실패 {failed} / 스킵(기존) {skipped}")
    print(f"실패한 항목은 학습 시 InsightFace fallback으로 처리됩니다.")


if __name__ == "__main__":
    main()
