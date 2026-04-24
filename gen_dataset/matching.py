import os, json, random, numpy as np, cv2, pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis

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
for i, s_id in enumerate(id_list):
    sims = sim_matrix[i]
    sorted_idxs = [idx for idx in np.argsort(sims)[::-1] if id_list[idx] != s_id]

    hard = [id_list[idx] for idx in sorted_idxs if sims[idx] >= 0.6][:5]
    if len(hard) < 5: hard = [id_list[idx] for idx in sorted_idxs[:5]]

    semi = [id_list[idx] for idx in sorted_idxs if 0.4 <= sims[idx] < 0.6][:10]
    if len(semi) < 10: semi = [id_list[idx] for idx in sorted_idxs[5:15]]

    easy_cand = [id_list[idx] for idx in sorted_idxs if sims[idx] < 0.4]
    easy = random.sample(easy_cand, 10) if len(easy_cand) > 10 else easy_cand

    match_map[s_id] = {"hard": hard, "semi_hard": semi, "easy": easy}

with open("../../mfvs_dataset/match_map.json", "w") as f: json.dump(match_map, f, indent=4)