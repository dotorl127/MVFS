import os, json, random, numpy as np, cv2, pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis

# FaceDancer 스왑 품질 보장을 위해 유사도 0.5 이상인 쌍만 사용
MIN_SIM = 0.5

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider']);
app.prepare(ctx_id=0, det_size=(640, 640))
ROOT_DIR = "../../mfvs_dataset"
id_list, embs = [], []

for id_n in tqdm(sorted(os.listdir(ROOT_DIR)), desc="Embedding Bank"):
    a1_p = os.path.join(ROOT_DIR, id_n, 'A1')
    img_n = next((f for f in os.listdir(a1_p) if f.lower().endswith(('.jpg', '.png'))), None) if os.path.exists(
        a1_p) else None
    if img_n:
        img = cv2.imread(os.path.join(a1_p, img_n))
        f = app.get(cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT))
        if f: id_list.append(id_n); embs.append(f[0].normed_embedding)

sim_matrix = np.dot(embs, np.array(embs).T)
match_map = {}
skipped = []

for i, s_id in enumerate(id_list):
    sims = sim_matrix[i]
    # MIN_SIM 이상인 후보만 (자기 자신 제외)
    valid_idxs = [idx for idx in np.argsort(sims)[::-1]
                  if id_list[idx] != s_id and sims[idx] >= MIN_SIM]

    if not valid_idxs:
        skipped.append(s_id)
        continue

    # hard: 0.6 이상 (가장 유사)
    hard = [id_list[idx] for idx in valid_idxs if sims[idx] >= 0.6][:5]

    # semi_hard: 0.5~0.6
    semi = [id_list[idx] for idx in valid_idxs if 0.5 <= sims[idx] < 0.6][:10]

    # easy: valid 중 가장 비유사한 것 (0.5에 가까운 것)
    easy_cand = [id_list[idx] for idx in reversed(valid_idxs)]
    easy = random.sample(easy_cand, 10) if len(easy_cand) > 10 else easy_cand

    match_map[s_id] = {"hard": hard, "semi_hard": semi, "easy": easy}

if skipped:
    print(f"\n[WARNING] {len(skipped)}명은 유사도 {MIN_SIM} 이상 매칭 없어 제외: {skipped}")

print(f"\n매칭 완료: {len(match_map)}명 / 전체 {len(id_list)}명")
with open("../../mfvs_dataset/match_map.json", "w") as f: json.dump(match_map, f, indent=4)