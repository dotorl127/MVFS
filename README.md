# DreamID Custom - IFS Base Training

## Pretrained 모델 다운로드

### 1. SD-Turbo (SwapNet/FaceNet base)
```bash
# HuggingFace에서 자동 다운로드 (학습 시 자동)
# 또는 수동:
git lfs install
git clone https://huggingface.co/stabilityai/sd-turbo
```

### 2. InsightFace buffalo_l (얼굴 감지 + ArcFace)
```bash
# 자동 다운로드 (첫 실행 시)
python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)"
# 또는 수동: ~/.insightface/models/buffalo_l/
```

### 3. OpenCLIP ViT-H-14 (텍스트 인코더)
```bash
# 자동 다운로드 (첫 실행 시)
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')"
```

### 4. Glint36k (ID Encoder - 선택, InsightFace buffalo_l 대체 가능)
```
https://github.com/deepinsight/insightface/tree/master/model_zoo
ms1mv3_arcface_r100_fp16 또는 glint360k_cosface_r100_fp16
```

---

## 데이터셋 구조
```
dataset/
  identity_001/
    A1/
      ref.jpg              ← 대표 이미지 1장
    A2/
      front.jpg            ← 정면
      left_45.jpg          ← 좌측 45도
      right_45.jpg         ← 우측 45도
      ...                  ← 멀티앵글 GT
    B_/
      front/               ← A2의 front에 대한 가짜 타겟들
        swap_001.jpg
        swap_002.jpg
        ...
      left_45/
        swap_001.jpg
        ...
  identity_002/
    ...
```

---

## 설치
```bash
pip install -r requirements.txt
```

## 1단계 학습 실행
```bash
python training/train_ifs.py \
    --data_dir /path/to/dataset \
    --output_dir ./checkpoints \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-5 \
    --max_steps 70000 \
    --save_steps 1000 \
    --mixed_precision
```

## RTX 3060 12GB 최적화 설정
```bash
# 8-bit Adam 자동 적용 (bitsandbytes 설치 시)
# gradient checkpointing은 diffusers에서 활성화 가능:
# swapnet.unet.enable_gradient_checkpointing()
# facenet.unet.enable_gradient_checkpointing()

python training/train_ifs.py \
    --data_dir /path/to/dataset \
    --output_dir ./checkpoints \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-5 \
    --max_steps 10000 \
    --save_steps 500 \
    --mixed_precision
```

## Loss 모니터링
```
Step 100 | total: 12.3456 | rec: 10.1234 | id: 0.9876 | dm: 1.2345
```
- rec loss가 빠르게 낮아지면 정상
- id loss가 0.3 이하로 내려오면 identity 학습 시작
- dm loss는 상대적으로 천천히 감소

## 학습 검증
- 매 save_steps마다 체크포인트 저장
- 초반 100-200 steps에서 rec loss가 내려가는지 확인
- 내려가지 않으면 lr 조정 또는 데이터 확인 필요
