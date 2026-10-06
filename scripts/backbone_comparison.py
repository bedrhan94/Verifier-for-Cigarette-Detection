# -*- coding: utf-8 -*-
"""IZOLE: Backbone kiyasini TEK protokolde birlestir (fig13 ile fig19 arasindaki DINOv2 tutarsizligi).

SORUN: backbone_ablation.csv (fig13) DINOv2 = 0.8445 diyor, verifier_auc_ci.csv (fig19) ayni
6740 crop / 17 kisi uzerinde per-person 0.8847 diyor. Ayni havuz, farkli sayi -> hakem bunu sorar.
HIPOTEZ: fark, DINOv2'nin timm varsayilan girdi cozunurlugunden (518px) geliyor; fig13 muhtemelen
224px kullanmisti. Bu script HER IKISINI de olcup farki KANITLAR.

CIKTI (yeni klasor, HICBIR mevcut dosyayi EZMEZ):
  runs_v3_eval/backbone_unified/backbone_unified.csv
  runs_v3_eval/backbone_unified/fig13b_backbone_unified.png
Protokol batchA_p2.py ile birebir ayni: W96 croplar, LeaveOneGroupOut(kisi), MLP(256) head.
"""
import os, glob, sys
os.environ["PYTHONUTF8"] = "1"
import numpy as np
import cv2
import torch
import timm
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
SRC = ROOT + r"\crops_sweep\W96"
OUT = ROOT + r"\runs_v3_eval\backbone_unified"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)

# (etiket, timm adi, zorlanan girdi boyutu veya None=timm varsayilani)
CONFIGS = [
    ("CLIP-ViT-B",        "vit_base_patch16_clip_224.openai",        None),
    ("DINOv2-ViT-S@224",  "vit_small_patch14_dinov2.lvd142m",        224),
    ("DINOv2-ViT-S@518",  "vit_small_patch14_dinov2.lvd142m",        None),
    ("ImageNet-ViT-S",    "vit_small_patch16_224.augreg_in21k_ft_in1k", None),
    ("ResNet50",          "resnet50.a1_in1k",                        None),
]


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def load_items():
    items = []
    for person in sorted(os.listdir(SRC)):
        for cls in ("cig", "fp"):
            for p in glob.glob(os.path.join(SRC, person, cls, "*.jpg")):
                items.append((p, person, 0 if cls == "cig" else 1))
    return items


def extract_feats(paths, timm_name, force_sz=None):
    kw = {}
    if force_sz is not None:
        kw["img_size"] = force_sz
    m = timm.create_model(timm_name, pretrained=True, num_classes=0, **kw).eval().to(DEV)
    cfg = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(DEV)
    insz = force_sz if force_sz is not None else cfg["input_size"][-1]
    F, buf = [], []
    for i, p in enumerate(tqdm(paths, desc=f"{timm_name.split('.')[0][:20]}@{insz}", ncols=90)):
        im = imread_u(p)
        im = cv2.resize(im, (insz, insz)) if im is not None else np.zeros((insz, insz, 3), np.uint8)
        buf.append(im)
        if len(buf) == 64 or i == len(paths) - 1:
            with torch.no_grad():
                x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().to(DEV) / 255.0
                F.append(m((x - mean) / std).cpu().numpy())
            buf = []
    del m
    torch.cuda.empty_cache()
    F = np.concatenate(F, 0)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8), insz


def lopo_oof(F, y, g):
    prob = np.full(len(y), np.nan)
    per_auc = {}
    for tr, te in LeaveOneGroupOut().split(F, y, g):
        if len(set(y[te])) < 2:
            continue
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                            early_stopping=True, random_state=0).fit(F[tr], y[tr])
        prob[te] = clf.predict_proba(F[te])[:, 1]
        per_auc[g[te][0]] = roc_auc_score(y[te], prob[te])
    return prob, per_auc


def boot_ci(y, s, n_boot=2000):
    n = len(y); a = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    os.makedirs(OUT, exist_ok=True)
    items = load_items()
    paths = [i[0] for i in items]
    y = np.array([i[2] for i in items])
    g = np.array([i[1] for i in items])
    print(f"[load] crops={len(items)}  cig={(y == 0).sum()}  fp={(y == 1).sum()}  kisi={len(set(g))}", flush=True)

    rows = ["label,timm_model,input_px,pooled_auc,ci95_low,ci95_high,perperson_auc_mean,perperson_auc_std,n_person"]
    res = []
    for k, (label, tname, fsz) in enumerate(CONFIGS, 1):
        print(f"\n[{k}/{len(CONFIGS)}] {label}", flush=True)
        F, insz = extract_feats(paths, tname, fsz)
        prob, per_auc = lopo_oof(F, y, g)
        valid = ~np.isnan(prob)
        pooled = roc_auc_score(y[valid], prob[valid])
        lo, hi = boot_ci(y[valid], prob[valid])
        pa = np.array(list(per_auc.values()))
        rows.append(f"{label},{tname},{insz},{pooled:.4f},{lo:.4f},{hi:.4f},"
                    f"{pa.mean():.4f},{pa.std():.4f},{len(pa)}")
        res.append((label, pooled, lo, hi, pa.mean(), pa.std()))
        print(f"  -> pooled={pooled:.4f} CI[{lo:.4f},{hi:.4f}]  per-person={pa.mean():.4f}+-{pa.std():.4f}", flush=True)

    with open(os.path.join(OUT, "backbone_unified.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    # ---- figur: pooled (CI) vs per-person, tek protokol ----
    labels = [r[0] for r in res]
    yy = np.arange(len(res))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    for i, (lab, pooled, lo, hi, pm, ps) in enumerate(res):
        yv = yy[i]
        ax.plot([lo, hi], [yv + 0.13, yv + 0.13], color="#2B7BBA", lw=2.2, solid_capstyle="round")
        ax.plot(pooled, yv + 0.13, "o", color="#1F3A5F", ms=7, zorder=3)
        ax.errorbar(pm, yv - 0.13, xerr=ps, fmt="s", color="#B74B00", ms=6, capsize=3, lw=1.6)
        ax.text(hi + 0.004, yv + 0.13, f"{pooled:.3f}", va="center", fontsize=8.5, color="#1F3A5F")
        ax.text(pm + ps + 0.004, yv - 0.13, f"{pm:.3f}", va="center", fontsize=8.5, color="#B74B00")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("ROC-AUC  (aynı 6740 crop, 17 kişi, LOPO, aynı MLP head)", fontsize=10)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.plot([], [], "o-", color="#1F3A5F", label="pooled (detection-level) + %95 bootstrap CI")
    ax.plot([], [], "s", color="#B74B00", label="per-person ortalama ± sd (n=17)")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    ax.set_title("Backbone karşılaştırması — TEK protokol\n"
                 "DINOv2 @224 vs @518: fig13/fig19 farkının kaynağı", fontsize=11, weight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0.70, 0.97)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig13b_backbone_unified.png"), dpi=160)
    plt.close()
    print(f"\n=== BITTI -> {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
