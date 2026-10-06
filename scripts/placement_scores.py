# -*- coding: utf-8 -*-
r"""DENEY 3c ADIM 1 — kare-basi VERIFIER SKORLARINI diske yazar.

NEDEN: `placement_fsm_endtoend.py` kare basina yalnizca IKILI (0/1) kaydediyordu.
Olay-seviyesi verifier tasarimlarini (A/B/C) denemek icin SUREKLI skor gerekiyor.
Bu script pahali kismi (dedektor + CLIP + verifier) BIR KEZ kosar ve diziyi kaydeder;
sonrasinda butun tasarim varyantlari saniyeler icinde offline denenebilir.

IKI KUME:
  (1) 12 KONUM klibi  -> sigara NEREDEYSE SUREKLI var (GT varlik .97-1.00)
                         => yalnizca MALIYETI (olay parcalanmasi) olcebilir.
  (2) S02 FP klipleri -> sigara HIC YOK, temporal sonrasi yanlis olaylar kaliyor
                         => FAYDAYI (yanlis olay silme) olcebilir.
  Ikisi birlikte olmadan maliyet-fayda tablosu KURULAMAZ. Tek basina (1) sadece
  kotu haber, tek basina (2) sadece iyi haber verir.

Cikti: runs_v3_eval/angle_experiment/_scores/<stem>.npz
       raw(uint8) | fused(float32, kare-basi MAX) | dconf(float32, kare-basi MAX)
       + _scores/_meta.json (esik, dislanan kisiler, FSM parametreleri)
"""
import os
os.environ["PYTHONUTF8"] = "1"
import sys
import glob
import json
import numpy as np
import cv2
import torch
import timm
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from ultralytics import YOLO

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
sys.path.insert(0, os.path.join(ROOT, "tools"))

RAWDIR = os.path.join(ROOT, "Raw_Farklı_Acılar")
GTV = os.path.join(ROOT, "Ground_Truth", "people_videos", "kisi_02")
OUT = os.path.join(ROOT, "runs_v3_eval", "angle_experiment", "_scores")
MODEL = os.path.join(ROOT, "models", "yolo11m_full_dataset_finetune_e38_best.pt")
CROPS = os.path.join(ROOT, "crops_sweep", "W96")
FEAT_CACHE = os.path.join(ROOT, "runs_v3_eval", "angle_experiment", "_clip_feats_crops.npz")
CACHE = os.path.join(ROOT, "runs_v3_eval", "_batchA_cache.npz")

# Konum kliplerindeki icenler + (S02 klipleri icin) S02 deneginin kendisi de dislanir.
EXCLUDE_PLACEMENT = {"kisi_01", "kisi_04", "kisi_05"}
EXCLUDE_S02 = {"kisi_02"}
CLIP_NAME = "vit_base_patch16_clip_224.openai"
IMGSZ, OVERLAP, MERGE_IOU = 960, 0.2, 0.5
CONF, CROP_PX, RETENTION = 0.30, 96, 0.96
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCEN = [("bos durucak", "SN"), ("false positive 2dk", "SF"), ("<smoker2> sigara", "MF")]


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def nms(boxes, scores, thr=MERGE_IOU):
    if len(boxes) == 0:
        return []
    idx = list(np.argsort(scores)[::-1])
    keep = []
    while idx:
        i = idx[0]
        keep.append(i)
        idx = [j for j in idx[1:] if iou(boxes[i], boxes[j]) < thr]
    return keep


def detect_tiled(model, fr, conf):
    H, W = fr.shape[:2]
    th, tw = int(H * (0.5 + OVERLAP / 2)), int(W * (0.5 + OVERLAP / 2))
    allb, alls = [], []
    for (x1, y1, x2, y2) in [(0, 0, W, H)] + [(ox, oy, ox + tw, oy + th)
                                              for oy in (0, H - th) for ox in (0, W - tw)]:
        r = model.predict(fr[y1:y2, x1:x2], imgsz=IMGSZ, conf=conf, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                allb.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1])
                alls.append(float(s))
    allb = np.array(allb) if allb else np.zeros((0, 4))
    alls = np.array(alls) if alls else np.zeros(0)
    k = nms(allb, alls)
    return (allb[k] if len(k) else np.zeros((0, 4))), (alls[k] if len(k) else np.zeros(0))


def crop96(fr, b):
    H, W = fr.shape[:2]
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    h = CROP_PX // 2
    x1, y1 = int(max(0, cx - h)), int(max(0, cy - h))
    x2, y2 = int(min(W, cx + h)), int(min(H, cy + h))
    c = fr[y1:y2, x1:x2]
    return c if c.size else np.zeros((CROP_PX, CROP_PX, 3), np.uint8)


class Clip:
    def __init__(self):
        self.m = timm.create_model(CLIP_NAME, pretrained=True, num_classes=0).eval().to(DEV)
        cfg = timm.data.resolve_model_data_config(self.m)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(DEV)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(DEV)
        self.sz = cfg["input_size"][-1]

    def feats(self, imgs):
        if not len(imgs):
            return np.zeros((0, 768), np.float32)
        out = []
        for i in range(0, len(imgs), 64):
            buf = [cv2.resize(im, (self.sz, self.sz)) for im in imgs[i:i + 64]]
            with torch.no_grad():
                x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().to(DEV) / 255.0
                out.append(self.m((x - self.mean) / self.std).cpu().numpy())
        F = np.concatenate(out, 0)
        return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


def crop_index():
    y, g = [], []
    for person in sorted(os.listdir(CROPS)):
        for cls in ("cig", "fp"):
            for _ in sorted(glob.glob(os.path.join(CROPS, person, cls, "*.jpg"))):
                y.append(0 if cls == "cig" else 1)
                g.append(person)
    return np.array(y), np.array(g)


def build_verifier(exclude, tag):
    """leave-N-out model + AYNI N-disli havuzda LOPO ile ESLESMIS OOF esigi."""
    y, g = crop_index()
    d = np.load(CACHE, allow_pickle=True)
    assert (y == d["y"]).all() and (g == d["g"]).all(), "cache hizalamasi BOZUK"
    F_all = np.load(FEAT_CACHE)["F"]
    keep = np.array([p not in exclude for p in g])
    F, yv, gv, dcv = F_all[keep], y[keep], g[keep], d["dconf"][keep]
    people = sorted(set(gv))
    assert not (set(exclude) & set(people)), "SIZINTI!"
    oof = np.full(len(yv), np.nan)
    for pr in tqdm(people, desc=f"LOPO[{tag}]", ncols=80):
        te = gv == pr
        oof[te] = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                                random_state=0).fit(F[~te], yv[~te]).predict_proba(F[te])[:, 1]
    fused = dcv * (1.0 - oof)
    thr = float(np.quantile(fused[yv == 0], 1.0 - RETENTION))
    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                        random_state=0).fit(F, yv)
    print(f"[verifier:{tag}] {len(people)} kisi | esik={thr:.5f} | "
          f"OOF korunma={float((fused[yv==0]>=thr).mean()):.4f}", flush=True)
    return clf, thr


def stem_of(path):
    b = os.path.basename(path)
    ang = "A" + b.split(".")[0]
    low = b.lower()
    for key, c in SCEN:
        if key.lower() in low:
            return f"{ang}_{c}"
    return f"{ang}_XX"


def score_video(model, clip, clf, vp, stem):
    dst = os.path.join(OUT, f"{stem}.npz")
    if os.path.exists(dst):
        print(f"  {stem}: zaten var, atlandi", flush=True)
        return
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw = np.zeros(n, np.uint8)
    fused = np.zeros(n, np.float32)
    dconf = np.zeros(n, np.float32)
    i = 0
    pbar = tqdm(total=n, desc=stem, ncols=88)
    while True:
        ok, fr = cap.read()
        if not ok or i >= n:
            break
        b, s = detect_tiled(model, fr, CONF)
        if len(b):
            raw[i] = 1
            dconf[i] = float(s.max())
            pfp = clf.predict_proba(clip.feats([crop96(fr, bb) for bb in b]))[:, 1]
            fused[i] = float(np.max(s * (1.0 - pfp)))
        i += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    np.savez_compressed(dst, raw=raw, fused=fused, dconf=dconf, fps=fps, n=n)
    print(f"  [OK] {stem}: n={n} fps={fps:.1f} kare-var %{100*raw.mean():.1f}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = YOLO(MODEL)
    clip = Clip()

    # ---- (1) 12 konum klibi: leave-3-out verifier ----
    clf_p, thr_p = build_verifier(EXCLUDE_PLACEMENT, "konum")
    for vp in sorted(glob.glob(os.path.join(RAWDIR, "*.mp4"))):
        score_video(model, clip, clf_p, vp, stem_of(vp))

    # ---- (2) S02 klipleri: leave-S02-out verifier (FAYDA olcumu) ----
    clf_a, thr_a = build_verifier(EXCLUDE_S02, "s02")
    s02 = [(os.path.join(GTV, "gun1", "false_positive", "<subject02_fp_clip>.mp4"), "S02_FP"),
           (os.path.join(GTV, "gun1", "positive", "<subject02_pos_clip>.mp4"), "S02_POS")]
    for vp, stem in s02:
        if os.path.exists(vp):
            score_video(model, clip, clf_a, vp, stem)
        else:
            print(f"  [ATLANDI] bulunamadi: {vp}", flush=True)

    with open(os.path.join(OUT, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(conf=CONF, retention_rule=RETENTION,
                       thr_placement=thr_p, thr_ali=thr_a,
                       exclude_placement=sorted(EXCLUDE_PLACEMENT),
                       exclude_ali=sorted(EXCLUDE_S02),
                       note=("konum klipleri MALIYETI (parcalanma) olcer - sigara neredeyse surekli; "
                             "S02 FP klibi FAYDAYI (yanlis olay silme) olcer - sigara hic yok")),
                  f, indent=2, ensure_ascii=False)
    print("\n[OK] skorlar -> _scores/")


if __name__ == "__main__":
    main()
