import os
import json
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# ==========================================
# FaceDancer 의존성 임포트
# ==========================================
import tensorflow as tf
from tensorflow.keras.models import load_model
from networks.layers import AdaIN, AdaptiveAttention
from retinaface.models import *
from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)

'''
Locate this file into facedancer root directory
'''

try:
    from tensorflow_addons.layers import InstanceNormalization
except ImportError:
    print("tensorflow-addons가 설치되지 않았습니다.")

# ==========================================
# 경로 설정
# ==========================================
FINETUNE_DIR = "../finetune_dataset"         # 파인튜닝 인물 X (배경)
BASE_DIR = "../mfvs_dataset"                 # 범용 데이터셋 (얼굴 소스)

# metadata: 파인튜닝 인물 X의 pose + 범용 인물들의 pose 모두 포함
FINETUNE_META_PATH = "../finetune_dataset/metadata.csv"  # X의 메타데이터
BASE_META_PATH = "../mfvs_dataset/metadata.csv"          # 범용 메타데이터
MATCH_MAP_PATH = "../finetune_dataset/match_map.json"    # finetune_matching.py 결과

MODEL_PATH = "./model_zoo/FaceDancer_config_c_HQ.h5"
AF_MODEL_PATH = "./arcface_model/ArcFace-Res50.h5"
R_MODEL_PATH = "./retinaface/RetinaFace-Res50.h5"

# ── 데이터 로드 ──
ft_df = pd.read_csv(FINETUNE_META_PATH)    # X의 메타데이터
base_df = pd.read_csv(BASE_META_PATH)      # 범용 메타데이터

with open(MATCH_MAP_PATH, 'r') as f:
    match_map = json.load(f)


def find_best_angle_source(source_pose, face_id):
    """
    범용 데이터셋에서 face_id의 A2 중
    source_pose와 가장 유사한 각도 이미지 반환
    """
    candidates = base_df[(base_df['id'] == face_id) & (base_df['folder_type'] == 'A2')]
    if candidates.empty:
        return None

    candidates = candidates.copy()
    candidates['pose_dist'] = (
        (candidates['pitch'] - source_pose['pitch']) ** 2 +
        (candidates['yaw'] - source_pose['yaw']) ** 2
    ) ** 0.5

    return candidates.sort_values('pose_dist').iloc[0]


def run_finetune_swapping():
    print("🚀 FaceDancer 모델 로드 중...")
    custom_objects = {
        'InstanceNormalization': InstanceNormalization,
        'AdaIN': AdaIN,
        'AdaptiveAttention': AdaptiveAttention
    }
    FaceDancer = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    RetinaFace = load_model(R_MODEL_PATH, compile=False,
                            custom_objects={"FPN": FPN, "SSH": SSH,
                                            "BboxHead": BboxHead,
                                            "LandmarkHead": LandmarkHead,
                                            "ClassHead": ClassHead})
    ArcFace = load_model(AF_MODEL_PATH, compile=False)
    print("✅ 모델 로드 완료!")

    blend_mask_base = np.zeros(shape=(256, 256, 1))
    blend_mask_base[77:240, 32:224] = 1
    blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

    # 파인튜닝 인물 X 순회 (finetune_dataset의 모든 인물)
    for ft_id, matches in tqdm(match_map.items(), desc="Finetune Swapping"):

        # X의 A2 이미지 목록 (배경/GT)
        ft_a2_list = ft_df[(ft_df['id'] == ft_id) & (ft_df['folder_type'] == 'A2')]
        if ft_a2_list.empty:
            continue

        all_face_ids = matches['hard'] + matches['semi_hard'] + matches['easy']

        for _, ft_a2_meta in ft_a2_list.iterrows():
            ft_a2_filename = ft_a2_meta['file_name']
            ft_a2_stem = os.path.splitext(ft_a2_filename)[0]

            # 저장 경로: finetune_dataset/X/B_/A2이미지명/
            save_dir = os.path.join(FINETUNE_DIR, ft_id, 'B_', ft_a2_stem)
            os.makedirs(save_dir, exist_ok=True)

            # X의 A2 이미지 (배경)
            background_img_path = os.path.join(FINETUNE_DIR, ft_id, 'A2', ft_a2_filename)
            source_pose = {'pitch': ft_a2_meta['pitch'], 'yaw': ft_a2_meta['yaw']}

            for face_id in all_face_ids:
                # 범용 데이터셋에서 가장 유사한 각도의 이미지 선택
                best_face = find_best_angle_source(source_pose, face_id)
                if best_face is None:
                    continue

                # 범용 인물의 A2 이미지 (얼굴 소스)
                face_img_path = os.path.join(BASE_DIR, face_id, 'A2', best_face['file_name'])

                save_name = f"{face_id}_swapped.jpg"
                save_path = os.path.join(save_dir, save_name)

                # Resume: 이미 생성된 파일 스킵
                if os.path.exists(save_path):
                    continue

                try:
                    # 배경: X의 A2
                    target = cv2.imread(background_img_path)
                    if target is None:
                        continue
                    target = np.array(target)

                    # 얼굴: 범용 인물의 A2
                    source = cv2.imread(face_img_path)
                    if source is None:
                        continue
                    source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
                    source = np.array(source)

                    # ArcFace embedding 추출 (얼굴 소스)
                    source_h, source_w, _ = source.shape
                    source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0]
                    source_lm = get_lm(source_a, source_w, source_h)
                    source_aligned = norm_crop(source, source_lm, image_size=112, shrink_factor=1.0)
                    source_z = ArcFace.predict(np.expand_dims(source_aligned / 255.0, axis=0))

                    # 배경 이미지에서 얼굴 감지 및 스왑
                    im = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
                    im_h, im_w, _ = im.shape
                    im_shape = (im_w, im_h)

                    detection_scale = (im_w // 640) if (im_w > 640) else 1
                    faces = RetinaFace(np.expand_dims(
                        cv2.resize(im, (im_w // detection_scale, im_h // detection_scale)), axis=0
                    )).numpy()
                    total_img = im / 255.0

                    for annotation in faces:
                        lm_align = get_lm(annotation, im_w, im_h)
                        M, _ = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
                        im_aligned = cv2.warpAffine(im, M, (256, 256), borderValue=0.0)

                        face_swap = FaceDancer.predict(
                            [np.expand_dims((im_aligned - 127.5) / 127.5, axis=0), source_z]
                        )
                        face_swap = (face_swap[0] + 1) / 2

                        transformed_lmk = transform_landmark_points(M, lm_align)
                        iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
                        iim_aligned = cv2.warpAffine(face_swap, iM, im_shape, borderValue=0.0)

                        blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
                        blend_mask = np.expand_dims(blend_mask, axis=-1)
                        total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))

                    total_img = np.clip(total_img * 255, 0, 255).astype('uint8')
                    total_img = cv2.cvtColor(total_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, total_img)

                except Exception as e:
                    print(f"\n[Error] {face_img_path} -> {background_img_path}: {e}")


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    run_finetune_swapping()
