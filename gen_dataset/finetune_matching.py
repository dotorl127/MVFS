import os, json, random, numpy as np, cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ==========================================
# 파인튜닝용 매칭
# finetune_dataset의 특정 인물 X와
# mfvs_dataset의 범용 인물들을 매칭
# ==========================================

MAX_SIM = 0.5  # 유사도 0.5 미만인 쌍만 사용

FINETUNE_DIR = "../../finetune_dataset"   # 파인튜닝 인물 X
BASE_DIR = "../../mfvs_dataset"           # 범용 데이터셋

app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition'],
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))


def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT))
    if faces:
        return faces[0].normed_embedding
    return None


# ── 1. 파인튜닝 인물 X embedding 추출 ──
ft_ids, ft_embs = [], []
for id_n in tqdm(sorted(os.listdir(FINETUNE_DIR)), desc="Finetune Embedding"):
    a1_p = os.path.join(FINETUNE_DIR, id_n, 'A1')
    if not os.path.exists(a1_p):
        continue
    img_n = next((f for f in os.listdir(a1_p) if f.lower().endswith(('.jpg', '.png'))), None)
    if not img_n:
        continue
    emb = get_embedding(os.path.join(a1_p, img_n))
    if emb is not None:
        ft_ids.append(id_n)
        ft_embs.append(emb)

print(f"파인튜닝 인물: {len(ft_ids)}명")

# ── 2. 범용 데이터셋 embedding 추출 ──
base_ids, base_embs = [], []
for id_n in tqdm(sorted(os.listdir(BASE_DIR)), desc="Base Embedding"):
    a1_p = os.path.join(BASE_DIR, id_n, 'A1')
    if not os.path.exists(a1_p):
        continue
    img_n = next((f for f in os.listdir(a1_p) if f.lower().endswith(('.jpg', '.png'))), None)
    if not img_n:
        continue
    emb = get_embedding(os.path.join(a1_p, img_n))
    if emb is not None:
        base_ids.append(id_n)
        base_embs.append(emb)

print(f"범용 인물: {len(base_ids)}명")

# ── 3. X ↔ 범용 유사도 계산 ──
ft_embs_np = np.array(ft_embs)
base_embs_np = np.array(base_embs)
sim_matrix = np.dot(ft_embs_np, base_embs_np.T)  # [ft_count, base_count]

# ── 4. 매칭 ──
match_map = {}
skipped = []

for i, ft_id in enumerate(ft_ids):
    sims = sim_matrix[i]  # ft_id vs 모든 base_id
    # MAX_SIM 미만인 범용 인물만
    valid_idxs = [idx for idx in np.argsort(sims)[::-1] if sims[idx] < MAX_SIM]

    if not valid_idxs:
        skipped.append(ft_id)
        continue

    # semi_hard: 0.1 ~ 0.2
    semi = [base_ids[idx] for idx in valid_idxs if 0.1 <= sims[idx] < 0.2][:10]

    # easy: 0.1 미만
    easy_cand = [base_ids[idx] for idx in valid_idxs if sims[idx] < 0.1]
    easy = random.sample(easy_cand, 10) if len(easy_cand) > 10 else easy_cand

    match_map[ft_id] = {"hard": [], "semi_hard": semi, "easy": easy}

if skipped:
    print(f"\n[WARNING] {len(skipped)}명 매칭 실패: {skipped}")

print(f"\n매칭 완료: {len(match_map)}명")
with open(os.path.join(FINETUNE_DIR, "match_map.json"), "w") as f:
    json.dump(match_map, f, indent=4)
