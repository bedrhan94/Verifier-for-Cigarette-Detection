# -*- coding: utf-8 -*-
"""B3: classic-feature baselines vs deep CLIP verifier (same W96 crops, same LOPO).
Color-hist (HSV), HOG, color+HOG, raw-pixel -> LogisticRegression LOPO AUC. Justifies the deep verifier.
fig24 + classic_baseline.csv"""
import os
os.environ["PYTHONUTF8"] = "1"
import glob
import numpy as np
import cv2
from tqdm import tqdm
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
SRC = ROOT + r"\crops_sweep\W96"
EV = ROOT + r"\runs_v3_eval"
FIG = EV + r"\figures"
P2 = ROOT + r"\PAPERS\PAPER2_verifier"


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def color_hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    h = cv2.normalize(h, h).flatten()
    return h


def hog_feat(bgr):
    g = cv2.cvtColor(cv2.resize(bgr, (96, 96)), cv2.COLOR_BGR2GRAY)
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)


def raw_pixel(bgr):
    return cv2.resize(bgr, (32, 32)).astype(np.float32).flatten() / 255.0


def main():
    items = []
    for person in sorted(os.listdir(SRC)):
        for cls in ("cig", "fp"):
            for p in glob.glob(os.path.join(SRC, person, cls, "*.jpg")):
                items.append((p, person, 0 if cls == "cig" else 1))
    g = np.array([i[1] for i in items]); y = np.array([i[2] for i in items])
    print(f"[B3] crops={len(items)} people={len(set(g))}", flush=True)

    CH, HG, RP = [], [], []
    for p, _, _ in tqdm(items, desc="classic feat"):
        im = imread_u(p)
        if im is None:
            im = np.zeros((224, 224, 3), np.uint8)
        CH.append(color_hist(im)); HG.append(hog_feat(im)); RP.append(raw_pixel(im))
    CH = np.array(CH); HG = np.array(HG); RP = np.array(RP)
    CHG = np.concatenate([CH, HG], axis=1)
    feats = {"Color-hist (HSV)": CH, "HOG": HG, "Color+HOG": CHG, "Raw-pixel 32x32": RP}

    rows = ["feature,lopo_auc_mean,lopo_auc_std"]
    results = {}
    for name, F in feats.items():
        per = []
        for tr, te in LeaveOneGroupOut().split(F, y, g):
            if len(set(y[te])) < 2:
                continue
            clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, C=1.0))
            clf.fit(F[tr], y[tr])
            per.append(roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]))
        m, s = float(np.mean(per)), float(np.std(per))
        results[name] = (m, s)
        rows.append(f"{name},{m:.4f},{s:.4f}")
        print(f"[B3] {name:20s} LOPO AUC = {m:.3f} +/- {s:.3f}", flush=True)
    # add deep reference
    rows.append("CLIP-ViT-B (deep, ref),0.9017,0.0496")
    results["CLIP-ViT-B (deep)"] = (0.9017, 0.0496)
    for p in (EV + r"\classic_baseline.csv", P2 + r"\tables\classic_baseline.csv"):
        open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")

    # figure
    names = list(results.keys()); means = [results[n][0] for n in names]; stds = [results[n][1] for n in names]
    colors = ["#888888"] * (len(names) - 1) + ["#c0392b"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(range(len(names)), means, xerr=stds, color=colors, capsize=3)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("LOPO ROC-AUC"); ax.set_xlim(0.4, 1.0); ax.axvline(0.5, color="k", ls=":", alpha=.5)
    ax.set_title("B3: classic hand-crafted features vs deep CLIP verifier")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f"{m:.3f}", (m + s + 0.005, i), va="center", fontsize=8)
    ax.invert_yaxis(); plt.tight_layout()
    for d in (FIG, P2 + r"\figures"):
        plt.savefig(d + r"\fig24_classic_baseline.png", dpi=300)
    plt.close()
    print("=== B3 DONE: fig24 + classic_baseline.csv ===", flush=True)


if __name__ == "__main__":
    main()
