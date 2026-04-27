import os
import json
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================================
# FaceDancer 의존성 임포트
# ==========================================
import tensorflow as tf
from tensorflow.keras.models import load_model
from networks.layers import AdaIN, AdaptiveAttention
from utils.swap_func import run_inference

'''
Locate this file into facedaner root directory
'''

try:
    from tensorflow_addons.layers import InstanceNormalization
except ImportError:
    print("tensorflow-addons가 설치되지 않았습니다. (pip install tensorflow-addons)")

# ==========================================

# 1. 경로 설정
METADATA_PATH = "../dataset/metadata.csv"
MATCH_MAP_PATH = "../dataset/match_map.json"
ROOT_DIR = "../dataset/5"
# FaceDancer 모델 가중치 경로 (허깅페이스에서 다운받은 h5 파일)
MODEL_PATH = "./model_zoo/FaceDancer_config_c_HQ.h5"
AF_MODEL_PATH = "./arcface_model/ArcFace-Res50.h5"
MAX_POSE_DIST = 20.0  # 포즈 거리 최대 허용값

df = pd.read_csv(METADATA_PATH)
with open(MATCH_MAP_PATH, 'r') as f:
    match_map = json.load(f)


def find_best_angle_target_a2(source_pose, target_id):
    target_a2_candidates = df[(df['id'] == target_id) & (df['folder_type'] == 'A2')]
    if target_a2_candidates.empty:
        return None

    target_a2_candidates = target_a2_candidates.copy()
    target_a2_candidates['pose_dist'] = (
        (target_a2_candidates['pitch'] - source_pose['pitch']) ** 2 +
        (target_a2_candidates['yaw'] - source_pose['yaw']) ** 2
    ) ** 0.5

    # 가장 가까운 각도 선택
    best = target_a2_candidates.sort_values('pose_dist').iloc[0]

    # threshold 초과 시 None 반환 → 해당 쌍 스킵
    if best['pose_dist'] > MAX_POSE_DIST:
        return None

    return best


def run_multi_angle_swapping():
    # 2. FaceDancer 모델 로드 (루프 밖에서 1회만 실행하여 오버헤드 방지)
    print("🚀 FaceDancer 모델을 로드하는 중입니다...")
    custom_objects = {
        'InstanceNormalization': InstanceNormalization,
        'AdaIN': AdaIN,
        'AdaptiveAttention': AdaptiveAttention
    }
    # compile=False로 로드하여 추론 시 메모리 사용량을 최소화합니다.
    FaceDancer = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    ArcFace = load_model(AF_MODEL_PATH, compile=False)
    print("✅ 모델 로드 완료!")

    for source_id, matches in tqdm(match_map.items(), desc="Swapping with FaceDancer"):

        source_a2_list = df[(df['id'] == source_id) & (df['folder_type'] == 'A2')]
        if source_a2_list.empty: continue

        for _, s_a2_meta in source_a2_list.iterrows():
            s_a2_filename = s_a2_meta['file_name']
            s_a2_stem = os.path.splitext(s_a2_filename)[0]

            save_dir = os.path.join(ROOT_DIR, source_id, 'B_', s_a2_stem)
            os.makedirs(save_dir, exist_ok=True)

            # 배경(Ground Truth 포즈)이 될 소스 A2
            background_img_path = os.path.join(ROOT_DIR, source_id, 'A2', s_a2_filename)
            source_pose = {'pitch': s_a2_meta['pitch'], 'yaw': s_a2_meta['yaw']}

            all_target_ids = matches['hard'] + matches['semi_hard'] + matches['easy']

            for t_id in all_target_ids:
                best_target_a2 = find_best_angle_target_a2(source_pose, t_id)
                if best_target_a2 is None: continue

                # 얼굴(Identity)로 사용할 타겟 A2
                face_img_path = os.path.join(ROOT_DIR, t_id, 'A2', best_target_a2['file_name'])

                save_name = f"{t_id}_swapped.jpg"
                save_path = os.path.join(save_dir, save_name)

                # 이미 작업된 파일이면 스킵 (중단 후 재개(Resume) 기능)
                if os.path.exists(save_path): continue

                # ==========================================
                # 3. FaceDancer 엔진 실행
                # ==========================================
                try:
                    # run_inference 함수 호출 (소스 얼굴 이미지, 타겟 배경 이미지, 모델 객체 전달)
                    target = np.asarray(Image.open(background_img_path).resize((256, 256)))
                    source = np.asarray(Image.open(face_img_path).resize((112, 112)))
                    source_z = ArcFace(np.expand_dims(source / 255.0, axis=0))

                    face_swap = FaceDancer([np.expand_dims((target - 127.5) / 127.5, axis=0), source_z]).numpy()
                    face_swap = (face_swap[0] + 1) / 2
                    face_swap = np.clip(face_swap * 255, 0, 255).astype('uint8')

                    # 반환된 이미지가 정상이라면 저장
                    if face_swap is not None:
                        # run_inference가 RGB를 반환할 경우 cv2.cvtColor(face_swap, cv2.COLOR_RGB2BGR) 필요
                        face_swap = cv2.cvtColor(face_swap, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, face_swap)
                except Exception as e:
                    print(f"\n[Error] {face_img_path} -> {background_img_path} 합성 실패: {e}")


if __name__ == "__main__":
    # GPU VRAM 동적 할당 (OOM 에러 방지용)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    run_multi_angle_swapping()