# -*- coding: utf-8 -*-
"""BATCH A (Paper 2) - reviewer-completeness on EXISTING crops/features.
 A1 verifier vs detector-conf baseline | A8 score-level fusion (same retention-vs-rejection axes)
 A2 bootstrap 95% CI + DeLong (CLIP/DINOv2/ResNet) + per-person Wilcoxon (unit-of-analysis honest)
 A3 PR-AUC / Average Precision | A4 t-SNE | A5 temperature-scaling calibration + reliability/ECE
Join: W96 crops (96px verifier features) <- crops_verifier manifest (detector conf) by (person,cls,frame,det).
Figs fig18-21 + CSVs. GPU feature-extract ~2-3 dk, tqdm."""
import os
os.environ["PYTHONUTF8"] = "1"
import glob
import re
import numpy as np
import cv2
import torch
import timm
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.manifold import TSNE
from scipy.stats import wilcoxon
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
SRC = ROOT + r"\crops_sweep\W96"
MANI = ROOT + r"\crops_verifier\manifest_verifier.csv"
EV = ROOT + r"\runs_v3_eval"
FIG = EV + r"\figures"
P2 = ROOT + r"\PAPERS\PAPER2_verifier"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)

BACKBONES = {
    "CLIP-ViT-B": "vit_base_patch16_clip_224.openai",
    "DINOv2-ViT-S": "vit_small_patch14_dinov2.lvd142m",
    "ResNet50": "resnet50.a1_in1k",
}


def imread_u(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def parse_fd(fname):
    m = re.search(r"_f?(\d{6})_(\d+)\.jpg$", fname)  # W96: _000000_0 ; manifest: _f000000_0
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


# ---------- load detector conf manifest -> (person,cls,frame,det)->conf ----------
def load_conf():
    d = {}
    with open(MANI, encoding="utf-8") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split(",")
            if len(p) < 4:
                continue
            path, person, cls, conf = p[0], p[1], p[2], p[-1]
            fr, det = parse_fd(os.path.basename(path))
            if fr is not None:
                d[(person, cls, fr, det)] = float(conf)
    return d


# ---------- load W96 crops + join conf ----------
def load_items():
    conf_map = load_conf()
    items, miss = [], 0
    for person in sorted(os.listdir(SRC)):
        for cls in ("cig", "fp"):
            for p in glob.glob(os.path.join(SRC, person, cls, "*.jpg")):
                fr, det = parse_fd(os.path.basename(p))
                c = conf_map.get((person, cls, fr, det))
                if c is None:
                    miss += 1
                    continue
                items.append((p, person, 0 if cls == "cig" else 1, c))
    print(f"[load] crops={len(items)} conf-join-miss={miss} people={len(set(i[1] for i in items))}", flush=True)
    return items


# ---------- feature extraction for one backbone ----------
def extract_feats(paths, timm_name):
    m = timm.create_model(timm_name, pretrained=True, num_classes=0).eval().to(DEV)
    cfg = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(DEV)
    insz = cfg["input_size"][-1]
    F, buf = [], []
    for i, p in enumerate(tqdm(paths, desc=f"feat {timm_name.split('.')[0][:18]}")):
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
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


# ---------- LOPO OOF probabilities ----------
def lopo_oof(F, y, g):
    prob = np.full(len(y), np.nan)
    per_auc = {}
    for tr, te in LeaveOneGroupOut().split(F, y, g):
        if len(set(y[te])) < 2:
            continue
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True, random_state=0).fit(F[tr], y[tr])
        prob[te] = clf.predict_proba(F[te])[:, 1]
        per_auc[g[te][0]] = roc_auc_score(y[te], prob[te])
    return prob, per_auc


# ================= DeLong (fast, Sun & Xu 2014) =================
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]; k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(pos[r, :]); ty[r, :] = _midrank(neg[r, :]); tz[r, :] = _midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_p(y_true, s1, s2):
    """two-sided p for AUC(s1)==AUC(s2); y_true 1=positive(fp)."""
    order = np.argsort(-y_true, kind="mergesort")  # positives first
    yb = y_true[order]; m = int(yb.sum())
    preds = np.vstack([s1[order], s2[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    from scipy.stats import norm
    if var <= 0:
        return aucs, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs, 2 * norm.sf(abs(z))


def boot_ci(y, s, n_boot=2000):
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], s[b]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def op_curve(keep_score, y):
    """keep detection if keep_score>=t. retention=frac cig kept, rejection=frac fp dropped."""
    thrs = np.linspace(keep_score.min(), keep_score.max(), 200)
    cig = keep_score[y == 0]; fp = keep_score[y == 1]
    ret = np.array([np.mean(cig >= t) for t in thrs])
    rej = np.array([np.mean(fp < t) for t in thrs])
    return ret, rej


def rej_at(ret, rej, target):
    ok = [(r, j) for r, j in zip(ret, rej) if r >= target]
    return max(ok, key=lambda z: z[1])[1] if ok else 0.0


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def main():
    items = load_items()
    paths = [i[0] for i in items]
    g = np.array([i[1] for i in items]); y = np.array([i[2] for i in items]); dconf = np.array([i[3] for i in items])

    feats = {name: extract_feats(paths, tn) for name, tn in BACKBONES.items()}
    oof = {}; per_auc = {}
    for name, F in feats.items():
        p, pa = lopo_oof(F, y, g)
        oof[name] = p; per_auc[name] = pa
        print(f"[lopo] {name}: mean per-person AUC={np.mean(list(pa.values())):.3f}", flush=True)

    valid = ~np.isnan(oof["CLIP-ViT-B"])
    yv = y[valid]; dcv = dconf[valid]

    # ============ A1 + A8: operating curves (det-conf vs verifier vs fused) ============
    ver_keep = 1.0 - oof["CLIP-ViT-B"][valid]      # P(cig) from verifier
    det_keep = dcv                                  # detector confidence = P(cig)
    # fused: product of cig-probabilities (both oriented as "is cigarette")
    fused_keep = det_keep * ver_keep
    curves = {"Detector-conf (A1 baseline)": op_curve(det_keep, yv),
              "Verifier (CLIP-ViT-B)": op_curve(ver_keep, yv),
              "Fused (conf x verifier, A8)": op_curve(fused_keep, yv)}
    rows_op = ["method,rej@95,rej@90,rej@99"]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    colors = {"Detector-conf (A1 baseline)": "#888888", "Verifier (CLIP-ViT-B)": "#2b7bba", "Fused (conf x verifier, A8)": "#c0392b"}
    styles = {"Detector-conf (A1 baseline)": "--", "Verifier (CLIP-ViT-B)": "-", "Fused (conf x verifier, A8)": "-"}
    for name, (ret, rej) in curves.items():
        ax.plot(ret * 100, rej * 100, styles[name], color=colors[name], lw=2, label=name)
        rows_op.append(f"{name},{rej_at(ret,rej,0.95):.4f},{rej_at(ret,rej,0.90):.4f},{rej_at(ret,rej,0.99):.4f}")
        print(f"[A1/A8] {name}: rej@95={rej_at(ret,rej,0.95)*100:.1f}% rej@90={rej_at(ret,rej,0.90)*100:.1f}%", flush=True)
    ax.set_xlabel("Cigarette retention (%)"); ax.set_ylabel("Look-alike rejection (%)")
    ax.set_title("Verifier and fusion vs. raising detector confidence\n(same retention-rejection axes)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.invert_xaxis()
    ax.axvline(95, color="green", ls=":", alpha=.5)
    plt.tight_layout()
    for d in (FIG, P2 + r"\figures"):
        plt.savefig(d + r"\fig18_verifier_vs_detconf.png", dpi=300)
    plt.close()
    open(EV + r"\verifier_vs_detconf.csv", "w", encoding="utf-8").write("\n".join(rows_op) + "\n")
    np.savez(EV + r"\_batchA_cache.npz", clip_prob=oof["CLIP-ViT-B"], dinov2_prob=oof["DINOv2-ViT-S"],
             resnet_prob=oof["ResNet50"], dconf=dconf, y=y, g=g)

    # ---- extraction-floor asymmetry (cig>=0.5, fp>=0.3 gives det-conf a free-lunch on low-conf fp) ----
    cig_p = np.percentile(dcv[yv == 0], [5, 25, 50]); fp_p = np.percentile(dcv[yv == 1], [50, 75, 95])
    print(f"[A1 floors] cig-conf p5/25/50={np.round(cig_p,3)} | fp-conf p50/75/95={np.round(fp_p,3)} | fp<0.5 frac={np.mean(dcv[yv==1]<0.5):.3f}", flush=True)

    # ============ A1b: MATCHED-FLOOR decisive test (both cig & fp with det_conf>=0.5) ============
    # removes the free-lunch; simulates 'detections that already passed a 0.5 detector threshold'
    # = the persistent high-confidence look-alike regime that is the paper's actual problem
    mf = dcv >= 0.5
    ymf = yv[mf]
    n_hi_fp = int(np.sum(ymf == 1))
    print(f"[A1b] matched-floor>=0.5: cig={int(np.sum(ymf==0))} high-conf-fp={n_hi_fp}", flush=True)
    curves_mf = {"Detector-conf (A1 baseline)": op_curve(det_keep[mf], ymf),
                 "Verifier (CLIP-ViT-B)": op_curve(ver_keep[mf], ymf),
                 "Fused (conf x verifier, A8)": op_curve(fused_keep[mf], ymf)}
    rows_mf = ["method,rej@95,rej@90,rej@99"]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for name, (ret, rej) in curves_mf.items():
        ax.plot(ret * 100, rej * 100, styles[name], color=colors[name], lw=2, label=name)
        rows_mf.append(f"{name},{rej_at(ret,rej,0.95):.4f},{rej_at(ret,rej,0.90):.4f},{rej_at(ret,rej,0.99):.4f}")
        print(f"[A1b] {name}: rej@95={rej_at(ret,rej,0.95)*100:.1f}% rej@90={rej_at(ret,rej,0.90)*100:.1f}%", flush=True)
    ax.set_xlabel("Cigarette retention (%)"); ax.set_ylabel("High-conf look-alike rejection (%)")
    ax.set_title("Matched floor (conf >= 0.5): the persistent high-confidence FP regime\n(no extraction free lunch; the paper's actual problem case)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.invert_xaxis(); ax.axvline(95, color="green", ls=":", alpha=.5)
    plt.tight_layout()
    for d in (FIG, P2 + r"\figures"):
        plt.savefig(d + r"\fig18b_matched_floor.png", dpi=300)
    plt.close()
    open(EV + r"\verifier_vs_detconf_matched.csv", "w", encoding="utf-8").write("\n".join(rows_mf) + "\n")

    # ============ A2: bootstrap CI + DeLong (detection-level) + per-person Wilcoxon ============
    rows_a2 = ["backbone,pooled_auc,ci95_low,ci95_high,perperson_auc_mean,perperson_auc_std"]
    pooled = {}
    for name in BACKBONES:
        s = oof[name][valid]
        pa = np.array(list(per_auc[name].values()))
        pooled[name] = roc_auc_score(yv, s)
        lo, hi = boot_ci(yv, s)
        rows_a2.append(f"{name},{pooled[name]:.4f},{lo:.4f},{hi:.4f},{pa.mean():.4f},{pa.std():.4f}")
        print(f"[A2] {name}: pooled AUC={pooled[name]:.3f} CI[{lo:.3f},{hi:.3f}] perperson={pa.mean():.3f}±{pa.std():.3f}", flush=True)
    # DeLong pooled (CLIP vs others) -- note independence caveat
    rows_del = ["comparison,level,delong_p_detlevel,wilcoxon_p_perperson"]
    persons = sorted(per_auc["CLIP-ViT-B"].keys())
    clip_pp = np.array([per_auc["CLIP-ViT-B"][p] for p in persons])
    for other in ("DINOv2-ViT-S", "ResNet50"):
        _, pdel = delong_p(yv, oof["CLIP-ViT-B"][valid], oof[other][valid])
        oth_pp = np.array([per_auc[other][p] for p in persons])
        try:
            _, pw = wilcoxon(clip_pp, oth_pp)
        except Exception:
            pw = float("nan")
        rows_del.append(f"CLIP-ViT-B_vs_{other},detlevel_n{valid.sum()}_pp_n{len(persons)},{pdel:.5f},{pw:.5f}")
        print(f"[A2] CLIP vs {other}: DeLong(det-level) p={pdel:.4f} | Wilcoxon(per-person) p={pw:.4f}", flush=True)
    open(EV + r"\verifier_auc_ci.csv", "w", encoding="utf-8").write("\n".join(rows_a2) + "\n")
    open(EV + r"\verifier_significance.csv", "w", encoding="utf-8").write("\n".join(rows_del) + "\n")

    # forest plot of AUC CIs
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(BACKBONES.keys())
    for i, name in enumerate(names):
        s = oof[name][valid]; lo, hi = boot_ci(yv, s); a = roc_auc_score(yv, s)
        ax.errorbar(a, i, xerr=[[a - lo], [hi - a]], fmt="o", color="#2b7bba", capsize=4)
        ax.annotate(f"{a:.3f} [{lo:.3f},{hi:.3f}]", (a, i), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_xlabel("Pooled ROC-AUC (95% bootstrap CI)"); ax.set_title("Verifier AUC with 95% bootstrap CIs (detection level)")
    ax.grid(alpha=.3, axis="x")
    plt.tight_layout()
    for d in (FIG, P2 + r"\figures"):
        plt.savefig(d + r"\fig19_auc_ci.png", dpi=300)
    plt.close()

    # ============ A3: PR-AUC / Average Precision ============
    rows_pr = ["backbone,roc_auc,pr_auc_ap"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name in BACKBONES:
        s = oof[name][valid]
        ap = average_precision_score(yv, s)
        rows_pr.append(f"{name},{pooled[name]:.4f},{ap:.4f}")
        prec, rec, _ = precision_recall_curve(yv, s)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
        print(f"[A3] {name}: PR-AUC/AP={ap:.3f} (positive=look-alike)", flush=True)
    ax.axhline(yv.mean(), color="gray", ls="--", lw=1, label=f"prevalence={yv.mean():.3f}")
    ax.set_xlabel("Recall (look-alike)"); ax.set_ylabel("Precision (look-alike)")
    ax.set_title("Precision-recall, positive class = look-alike (3:1 imbalance)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    for d in (FIG, P2 + r"\figures"):
        plt.savefig(d + r"\fig19b_pr_curve.png", dpi=300)
    plt.close()
    open(EV + r"\verifier_pr_auc.csv", "w", encoding="utf-8").write("\n".join(rows_pr) + "\n")

    # ============ A4: t-SNE of CLIP features ============
    F = feats["CLIP-ViT-B"]
    sub = rng.choice(len(F), min(3000, len(F)), replace=False)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(F[sub])
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ys = y[sub]
    ax.scatter(emb[ys == 0, 0], emb[ys == 0, 1], s=6, alpha=.5, c="#2b7bba", label="cigarette")
    ax.scatter(emb[ys == 1, 0], emb[ys == 1, 1], s=6, alpha=.5, c="#c0392b", label="look-alike")
    ax.set_title("CLIP-ViT-B feature space (t-SNE)"); ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    for d in (FIG, P2 + r"\figures"):
        plt.savefig(d + r"\fig20_tsne.png", dpi=300)
    plt.close()
    print(f"[A4] t-SNE on {len(sub)} points saved", flush=True)

    # ============ A5: temperature scaling calibration + reliability/ECE ============
    p = np.clip(oof["CLIP-ViT-B"][valid], 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))

    def nll(T):
        q = 1 / (1 + np.exp(-z / T))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q))
    Topt = minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded").x
    p_cal = 1 / (1 + np.exp(-z / Topt))
    ece_before = ece(yv, p); ece_after = ece(yv, p_cal)
    print(f"[A5] temperature T={Topt:.3f} | ECE {ece_before:.4f} -> {ece_after:.4f}", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, (pp, tag, ee) in zip(ax, [(p, "raw", ece_before), (p_cal, f"T={Topt:.2f}", ece_after)]):
        edges = np.linspace(0, 1, 11); mids = (edges[:-1] + edges[1:]) / 2; acc = []
        for i in range(10):
            mm = (pp >= edges[i]) & (pp <= edges[i + 1])
            acc.append(yv[mm].mean() if mm.sum() else np.nan)
        a.plot([0, 1], [0, 1], "k--", alpha=.5)
        a.bar(mids, acc, width=0.09, color="#2b7bba", edgecolor="k", alpha=.8)
        a.set_xlabel("predicted P(look-alike)"); a.set_ylabel("empirical fraction")
        a.set_title(f"Reliability ({tag}), ECE={ee:.3f}"); a.grid(alpha=.3)
    plt.suptitle("A5: verifier calibration via temperature scaling")
    plt.tight_layout()
    for d in (FIG, P2 + r"\figures"):
        plt.savefig(d + r"\fig21_calibration.png", dpi=300)
    plt.close()
    open(EV + r"\verifier_calibration.csv", "w", encoding="utf-8").write(
        f"temperature,ece_before,ece_after\n{Topt:.4f},{ece_before:.4f},{ece_after:.4f}\n")

    print("\n=== BATCH A (P2) DONE: fig18,19,19b,20,21 + 6 CSVs ===", flush=True)


if __name__ == "__main__":
    main()
