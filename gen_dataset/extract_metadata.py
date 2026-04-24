import os
import pandas as pd
import cv2
import tqdm
import numpy as np
from insightface.app import FaceAnalysis

# 1. 모델 설정 (Buffalo_L 기반 탐지, 랜드마크, 인식 모듈 로드)
app = FaceAnalysis(
    name='buffalo_l', 
    allowed_modules=['detection', 'landmark_3d_68', 'recognition'],
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

ROOT_DIR = "../../mfvs_dataset"
OUTPUT_CSV = "metadata.csv"

def extract_final_metadata():
    id_dirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    final_data_list = []

    for id_name in tqdm.tqdm(id_dirs, desc="Final Extraction"):
        id_path = os.path.join(ROOT_DIR, id_name)
        
        # 정제된 A1(Identity)과 A2(Pose) 폴더만 순회
        for folder_type in ['A1', 'A2']:
            target_folder = os.path.join(id_path, folder_type)
            if not os.path.exists(target_folder):
                continue

            img_files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_name in img_files:
                img_path = os.path.join(target_folder, img_name)
                raw_img = cv2.imread(img_path)
                if raw_img is None: continue

                # [검출률 향상] 상하좌우 25% 패딩 추가 (검은색 여백)
                h, w = raw_img.shape[:2]
                pad_h, pad_w = int(h * 0.25), int(w * 0.25)
                padded_img = cv2.copyMakeBorder(
                    raw_img, pad_h, pad_h, pad_w, pad_w, 
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

                # 패딩된 이미지에서 얼굴 탐지
                faces = app.get(padded_img)
                
                if not faces:
                    # 패딩을 줘도 못 찾는 경우 (너무 저화질이거나 얼굴이 아닐 확률 높음)
                    continue

                # 가장 큰 얼굴 기준 (여러 명일 경우 대비)
                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]

                final_data_list.append({
                    'id': id_name,
                    'file_name': img_name,
                    'folder_type': folder_type,  # 'A1' 또는 'A2'
                    'pitch': round(float(face.pose[1]), 2),
                    'yaw': round(float(face.pose[0]), 2),
                    'roll': round(float(face.pose[2]), 2)
                })

    # 최종 CSV 저장
    df = pd.DataFrame(final_data_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✨ 추출 완료! 총 {len(df)}장의 학습용 메타데이터가 {OUTPUT_CSV}에 저장되었습니다.")

if __name__ == "__main__":
    extract_final_metadata()