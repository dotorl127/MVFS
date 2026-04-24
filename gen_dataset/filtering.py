import shutil
from pathlib import Path
from deepface import DeepFace

root = Path('./vgg')

for img_dir in root.iterdir():
    if not img_dir.is_dir():
        continue

    # 1. 이미지 파일 하나 가져오기
    img_list = list(img_dir.glob('*.jpg'))
    if not img_list:
        continue

    img_path = str(img_list[0])

    try:
        # 2. DeepFace 분석 (enforce_detection=False 설정 권장)
        # 분석 실패 시 에러 방지를 위해 False로 두는 것이 좋습니다.
        results = DeepFace.analyze(img_path, actions=['gender', 'race', 'age'], enforce_detection=False)
        attr = results[0]  # 첫 번째 얼굴 결과

        # 3. 필터링 조건 (성별은 보통 'Man', 'Woman'으로 나옴)
        # 주의: DeepFace 결과에서 'dominant_gender'와 'dominant_race' 키를 쓰는 것이 더 정확합니다.
        gender = attr['dominant_gender']
        race = attr['dominant_race']
        age = attr['age']

        # 조건: 여자가 아니거나, 인종이 black인 경우 삭제
        if gender != 'Woman' or race == 'black':
            print(f"Deleting {img_dir}: {gender}, {race}")
            shutil.rmtree(img_dir)  # 폴더 통째로 삭제

    except Exception as e:
        print(f"Error processing {img_dir}: {e}")