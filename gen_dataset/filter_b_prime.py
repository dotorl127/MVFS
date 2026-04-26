"""
기존 B_ 이미지 중 A2와 너무 비슷한 것을 trash 디렉토리로 이동

실행:
python filter_b_prime.py --data_dir ../mfvs_dataset --threshold 0.7
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
    parser.add_argument("--data_dir", type=str, default="../mfvs_dataset")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="B'와 A2 유사도가 이 값 이상이면 trash로 이동 (기본 0.7)")
    parser.add_argument("--det_size", type=int, default=512)
    return parser.parse_args()


def get_embedding(app, img):
    faces = app.get(cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT))
    if faces:
        return faces[0].normed_embedding
    return None


def main():
    args = parse_args()
    root_dir = Path(args.data_dir)

    app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    total_checked = 0
    total_moved = 0

    for identity_dir in tqdm(sorted(root_dir.iterdir()), desc="Filtering"):
        if not identity_dir.is_dir():
            continue

        a2_dir = identity_dir / "A2"
        b_dir = identity_dir / "B_"

        if not a2_dir.exists() or not b_dir.exists():
            continue

        # A2 embedding 캐싱
        a2_embeddings = {}
        for a2_img in a2_dir.iterdir():
            if a2_img.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue
            img = cv2.imread(str(a2_img))
            if img is None:
                continue
            emb = get_embedding(app, img)
            if emb is not None:
                a2_embeddings[a2_img.stem] = emb

        if not a2_embeddings:
            continue

        # B_ 서브디렉토리 순회
        for b_subdir in b_dir.iterdir():
            if not b_subdir.is_dir() or b_subdir.name == "trash":
                continue

            a2_stem = b_subdir.name
            if a2_stem not in a2_embeddings:
                continue
            a2_emb = a2_embeddings[a2_stem]

            for b_img in list(b_subdir.iterdir()):
                if b_img.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                    continue

                total_checked += 1
                img = cv2.imread(str(b_img))
                if img is None:
                    continue

                b_emb = get_embedding(app, img)
                if b_emb is None:
                    continue

                sim = float(np.dot(a2_emb, b_emb))

                if sim >= args.threshold:
                    total_moved += 1

                    trash_dir = b_subdir / "trash"
                    trash_dir.mkdir(exist_ok=True)
                    b_img.rename(trash_dir / b_img.name)

    print(f"\n완료: 전체 {total_checked}장 중 {total_moved}장 trash로 이동")


if __name__ == "__main__":
    main()
