import os
import shutil
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

# 1. 환경 설정 및 모델 초기화
app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'landmark_3d_68', 'recognition'],
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

# 경로 설정
ROOT_DIR = "./vgg"
LIGHTING_THRESHOLD = 0.98  # 조명 유사도 임계값 (0.95~0.98 추천)


def get_face_data_with_padding(img_path):
    """이미지 로드, 패딩 추가 및 얼굴 데이터 추출"""
    raw_img = cv2.imread(img_path)
    if raw_img is None: return None

    h, w = raw_img.shape[:2]
    pad_h, pad_w = int(h * 0.25), int(w * 0.25)
    img = cv2.copyMakeBorder(raw_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    faces = app.get(img)
    if not faces: return None

    # 가장 큰 얼굴 선택
    face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]

    return {
        'emb': face.normed_embedding,
        'pitch': face.pose[1],
        'yaw': face.pose[0],
        'landmark': face.landmark_2d_106
    }


def get_brightness_hist(img_path):
    """이미지의 밝기 히스토그램 추출 및 정규화"""
    img = cv2.imread(img_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def run_integrated_pipeline():
    id_dirs = [d for d in sorted(os.listdir(ROOT_DIR)) if os.path.isdir(os.path.join(ROOT_DIR, d))]

    for id_name in tqdm(id_dirs, desc="Processing Pipeline"):
        id_path = os.path.join(ROOT_DIR, id_name)

        # 결과 폴더 생성
        a1_path = os.path.join(id_path, 'A1')
        a2_path = os.path.join(id_path, 'A2')
        trash_path = os.path.join(id_path, 'A_trash')
        for f in [a1_path, a2_path, trash_path]:
            os.makedirs(f, exist_ok=True)

        # ---------------------------------------------------------
        # STEP 1: Classifying (Centroid Based)
        # ---------------------------------------------------------
        img_list = [f for f in os.listdir(id_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        valid_faces = []

        for fname in img_list:
            fpath = os.path.join(id_path, fname)
            data = get_face_data_with_padding(fpath)
            if data:
                data['file_name'] = fname
                valid_faces.append(data)
            else:
                shutil.move(fpath, os.path.join(trash_path, fname))

        if not valid_faces: continue

        # 평균 임베딩 계산 및 L2 정규화
        all_embs = np.array([f['emb'] for f in valid_faces])
        mean_emb = np.mean(all_embs, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        # 유사도 검사 및 1차 분류
        final_candidates = []
        for f in valid_faces:
            sim = np.dot(mean_emb, f['emb'])
            fpath = os.path.join(id_path, f['file_name'])

            if sim < 0.4:  # 타인 혹은 극심한 노이즈(워터마크 등)
                shutil.move(fpath, os.path.join(trash_path, f['file_name']))
            else:
                f['similarity'] = sim
                final_candidates.append(f)

        if not final_candidates: continue

        # A1 선정 (가장 정면이면서 아이덴티티가 확실한 것)
        final_candidates.sort(
            key=lambda x: x['similarity'] - (abs(x['pitch']) + abs(x['yaw'])) * 0.01,
            reverse=True
        )

        a1_node = final_candidates[0]
        shutil.move(os.path.join(id_path, a1_node['file_name']), os.path.join(a1_path, a1_node['file_name']))

        # 나머지는 일단 A2로 이동
        for f in final_candidates[1:]:
            shutil.move(os.path.join(id_path, f['file_name']), os.path.join(a2_path, f['file_name']))

        # ---------------------------------------------------------
        # STEP 2: Filtering (Lighting Redundancy)
        # ---------------------------------------------------------
        a2_files = [f for f in os.listdir(a2_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(a2_files) <= 1: continue

        kept_hists = []
        # 파일명 순으로 처리하여 일관성 유지
        for fname in sorted(a2_files):
            fpath = os.path.join(a2_path, fname)
            curr_hist = get_brightness_hist(fpath)
            if curr_hist is None: continue

            is_redundant = False
            for _, existing_hist in kept_hists:
                similarity = cv2.compareHist(curr_hist, existing_hist, cv2.HISTCMP_CORREL)
                if similarity > LIGHTING_THRESHOLD:
                    is_redundant = True
                    break

            if is_redundant:
                shutil.move(fpath, os.path.join(trash_path, fname))
            else:
                kept_hists.append((fname, curr_hist))


if __name__ == "__main__":
    run_integrated_pipeline()
    print("\n✨ 통합 분류 및 조명 필터링 작업이 완료되었습니다!")