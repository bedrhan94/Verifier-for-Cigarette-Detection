# -*- coding: utf-8 -*-
"""PAPER 2 — Chefer transformer-attribution XAI TAM SURUM.
Feasibility (izole) Chefer'in localize ettigini gostermisti; bu script makale-kalitesinde uretir:
 (A) nitel figur: cig + kolay/zor look-alike ornekleri, Grad-CAM vs Chefer yan yana
 (B) NICEL: N crop uzerinde 'localization concentration' (en sicak %10 patch'teki relevans kutlesi)
     dagilimi -> Chefer vs Grad-CAM, ortalama±std + tablo (rastgele taban ~0.10)
Model: verifier/gradcam_model.pt (DINOv2-ViT-S, 2 sinif) = Grad-CAM'in basarisiz oldugu AYNI model.
Cikti: PAPERS/PAPER2_verifier/{figures,tables} + runs_v3_eval kopyasi."""
import os
os.environ["PYTHONUTF8"] = "1"
import csv
import random
import numpy as np
import cv2
import torch
import timm
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(0); torch.manual_seed(0); np.random.seed(0)
ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
MAN = ROOT + r"\crops_verifier\manifest_verifier.csv"
CKPT = ROOT + r"\verifier\gradcam_model.pt"
P2 = ROOT + r"\PAPERS\PAPER2_verifier"
EV = ROOT + r"\runs_v3_eval"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
N_PER = 600           # TAM SURUM: nicel metrik icin SINIF BASINA crop (600 cig + 600 fp)
N_SHOW = 3            # nitel figurde her siniftan kac ornek
HARD_CONF = 0.50      # look-alike zorluk esigi: detektor conf >= bu => ZOR (israrli yuksek-conf FP)


def imread_u(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def prep(img):
    x = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEV)
    return (t - MEAN) / STD


def concentration(m):
    """Localization metrigi: relevans kutlesinin en sicak %10 patch'teki orani (rastgele~0.10)."""
    v = m.flatten(); v = np.maximum(v, 0); s = v.sum()
    if s <= 0:
        return 0.0
    v = v / s
    k = max(1, int(0.10 * len(v)))
    return float(np.sort(v)[::-1][:k].sum())


def main():
    rows = list(csv.DictReader(open(MAN, encoding="utf-8")))
    cig = [r for r in rows if r["cls"] != "fp"]
    fp = [r for r in rows if r["cls"] == "fp"]
    random.shuffle(cig); random.shuffle(fp)
    print(f"crop havuzu: cig={len(cig)} fp={len(fp)} | device={DEV}", flush=True)

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False,
                              num_classes=2, img_size=224, dynamic_img_size=True).to(DEV)
    model.load_state_dict(torch.load(CKPT, map_location=DEV))
    model.eval()

    attn_maps = []

    def fwd_hook(mod, inp, out):
        a = inp[0]; a.retain_grad(); attn_maps.append(a)
    for blk in model.blocks:
        blk.attn.fused_attn = False
        blk.attn.attn_drop.register_forward_hook(fwd_hook)
    feats = {}
    model.blocks[-1].register_forward_hook(lambda m, i, o: feats.__setitem__("t", o))

    def run(img, cls_idx):
        """Ayni ileri/geri gecisten hem Grad-CAM hem Chefer relevance."""
        attn_maps.clear()
        model.zero_grad()
        out = model(prep(img))
        t = feats["t"]; t.retain_grad()
        out[0, cls_idx].backward()
        T = attn_maps[0].shape[-1]; npatch = 256; pre = T - npatch; g = 16
        # Grad-CAM (son block patch tokenlari)
        act = t[0, pre:pre + npatch]; gr = t.grad[0, pre:pre + npatch]
        cam = (gr.mean(0, keepdim=True) * act).sum(-1).reshape(g, g).detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        # Chefer: layer-wise (grad ⊙ attn)+ rollout
        R = torch.eye(T, device=DEV)
        for a in attn_maps:
            c = (a.grad * a).clamp(min=0).mean(1)[0]
            c = c + torch.eye(T, device=DEV)
            c = c / c.sum(-1, keepdim=True)
            R = c @ R
        rel = R[0, pre:pre + npatch].reshape(g, g).detach().cpu().numpy()
        rel = np.maximum(rel, 0)
        # Attention-rollout (Abnar & Zuidema 2020) — ADIL BASELINE: gradyansiz, saf attention.
        # Grad-CAM ViT'te dejenere (post-ReLU sifir) oldugu icin kiyasi buna dayandiriyoruz.
        Rr = torch.eye(T, device=DEV)
        for a in attn_maps:
            ar = a.mean(1)[0].detach()
            ar = ar + torch.eye(T, device=DEV)
            ar = ar / ar.sum(-1, keepdim=True)
            Rr = ar @ Rr
        roll = Rr[0, pre:pre + npatch].reshape(g, g).detach().cpu().numpy()
        roll = np.maximum(roll, 0)
        return cam, roll, rel

    # ---------- (B) NICEL (TAM SURUM: N_PER/sinif + look-alike zorluk kirilimi) ----------
    n_fp = min(len(fp), N_PER)
    sample = [(r, 0) for r in cig[:N_PER]] + [(r, 1) for r in fp[:n_fp]]
    cc_gc, cc_ro, cc_ch, s_cls, s_cnf = [], [], [], [], []
    for r, cls in tqdm(sample, desc="nicel"):
        img = imread_u(r["path"])
        if img is None:
            continue
        cam, roll, rel = run(img, cls)
        cc_gc.append(concentration(cam)); cc_ro.append(concentration(roll)); cc_ch.append(concentration(rel))
        s_cls.append(cls)
        try:
            s_cnf.append(float(r.get("conf", "nan")))
        except (TypeError, ValueError):
            s_cnf.append(float("nan"))
    cc_gc = np.array(cc_gc); cc_ro = np.array(cc_ro); cc_ch = np.array(cc_ch)
    s_cls = np.array(s_cls); s_cnf = np.array(s_cnf)
    print(f"\n[NICEL n={len(cc_ch)}] Grad-CAM {cc_gc.mean():.3f}±{cc_gc.std():.3f} | "
          f"rollout {cc_ro.mean():.3f}±{cc_ro.std():.3f} | "
          f"Chefer {cc_ch.mean():.3f}±{cc_ch.std():.3f} (rastgele~0.10)", flush=True)
    # look-alike ZORLUK kirilimi: detektor conf >= HARD_CONF => ZOR (israrli, esik yukseltmenin silemedigi)
    fpm = s_cls == 1
    hard = fpm & (s_cnf >= HARD_CONF); easy = fpm & (s_cnf < HARD_CONF)
    ch_hard, ch_easy = cc_ch[hard], cc_ch[easy]
    ro_hard = cc_ro[hard]; gc_hard = cc_gc[hard]
    print(f"[LOOK-ALIKE ZORLUK] Chefer ZOR(conf>={HARD_CONF}) n={int(hard.sum())} "
          f"{ch_hard.mean():.3f}±{ch_hard.std():.3f} | KOLAY n={int(easy.sum())} "
          f"{ch_easy.mean():.3f}±{ch_easy.std():.3f}  (rollout ZOR {ro_hard.mean():.3f})", flush=True)

    os.makedirs(P2 + r"\figures", exist_ok=True); os.makedirs(P2 + r"\tables", exist_ok=True)
    with open(P2 + r"\tables\chefer_xai_localization.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_crops", "concentration_mean", "concentration_std", "random_baseline", "note"])
        w.writerow(["GradCAM", len(cc_gc), f"{cc_gc.mean():.4f}", f"{cc_gc.std():.4f}", "0.10",
                    "degenerate on ViT (post-ReLU all-zero) - reported honestly, NOT used as headline baseline"])
        w.writerow(["AttentionRollout", len(cc_ro), f"{cc_ro.mean():.4f}", f"{cc_ro.std():.4f}", "0.10",
                    "fair non-degenerate baseline (Abnar 2020) - gradient-free"])
        w.writerow(["Chefer", len(cc_ch), f"{cc_ch.mean():.4f}", f"{cc_ch.std():.4f}", "0.10",
                    "grad-weighted attention relevance (Chefer 2021)"])
        w.writerow(["Chefer_lookalike_HARD", int(hard.sum()), f"{ch_hard.mean():.4f}", f"{ch_hard.std():.4f}", "0.10",
                    f"persistent HIGH-conf look-alikes (det conf>={HARD_CONF}) - Chefer still localizes where threshold-raising fails"])
        w.writerow(["Chefer_lookalike_EASY", int(easy.sum()), f"{ch_easy.mean():.4f}", f"{ch_easy.std():.4f}", "0.10",
                    f"low-conf look-alikes (det conf<{HARD_CONF})"])

    # ---------- (A)+(B) FIGUR ----------
    def overlay(img, m):
        base = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        mm = m / (m.max() + 1e-8)
        mm = cv2.GaussianBlur(cv2.resize(mm, (224, 224)), (0, 0), 6)
        mm = mm / (mm.max() + 1e-8)
        hm = cv2.applyColorMap((mm * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32) / 255.0
        a = (mm ** 1.5)[..., None]
        return np.clip((1 - .7 * a) * base + .7 * a * hm, 0, 1)

    def _cf(r):
        try:
            return float(r.get("conf", 0))
        except (TypeError, ValueError):
            return 0.0
    fp_hard = sorted(fp, key=lambda r: -_cf(r))  # en ZOR (yuksek-conf) look-alike'lar
    show = [("cigarette", r, 0) for r in cig[:N_SHOW]] + [("hard look-alike", r, 1) for r in fp_hard[:N_SHOW]]
    nrow = len(show)
    fig = plt.figure(figsize=(14.5, 2.5 * nrow + 2.8))
    gs = fig.add_gridspec(nrow + 1, 4, height_ratios=[1] * nrow + [1.15], hspace=.25, wspace=.06)
    for i, (name, r, cls) in enumerate(show):
        img = imread_u(r["path"])
        cam, roll, rel = run(img, cls)
        for j, (im, ttl) in enumerate([(cv2.resize(img, (224, 224)) / 255.0, name),
                                       (overlay(img, cam), "Grad-CAM (degenerate)"),
                                       (overlay(img, roll), "Attention-rollout (diffuse)"),
                                       (overlay(img, rel), "Chefer (localized)")]):
            ax = fig.add_subplot(gs[i, j]); ax.imshow(np.clip(im, 0, 1)); ax.axis("off")
            if i == 0:
                ax.set_title(ttl, fontsize=11, fontweight="bold")
            if j == 0:
                ax.text(-.06, .5, name, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9, color="#333")
    # alt panel: nicel dagilim
    axq = fig.add_subplot(gs[nrow, :])
    bp = axq.boxplot([cc_gc, cc_ro, cc_ch], vert=False, patch_artist=True, widths=.55)
    axq.set_yticks([1, 2, 3]); axq.set_yticklabels(["Grad-CAM", "Attention-rollout", "Chefer"])
    for patch, c in zip(bp["boxes"], ["#999999", "#c0392b", "#2b7bba"]):
        patch.set_facecolor(c); patch.set_alpha(.75)
    axq.axvline(0.10, color="#666", ls="--", lw=1.2)
    axq.text(0.105, 2.62, "random floor 0.10", fontsize=8, color="#666", va="center")
    axq.set_xlabel("Localization concentration: share of relevance mass in the hottest 10% of patches (higher is more localized)")
    axq.set_title(f"Quantitative (n={len(cc_ch)} crops): Chefer {cc_ch.mean():.2f}±{cc_ch.std():.2f}  >  "
                  f"rollout {cc_ro.mean():.2f}±{cc_ro.std():.2f}  >  Grad-CAM {cc_gc.mean():.2f} "
                  f"(degenerate on this ViT)", fontsize=10)
    axq.text(0.5, -0.42, f"On persistent HARD look-alikes (det conf≥{HARD_CONF}, n={int(hard.sum())}): Chefer {ch_hard.mean():.2f}; "
             f"easy (n={int(easy.sum())}): {ch_easy.mean():.2f}. Localization holds on exactly the hard false positives that raising the threshold does not remove.",
             transform=axq.transAxes, ha="center", va="top", fontsize=8.5, color="#2b7bba", style="italic")
    axq.grid(alpha=.3, axis="x")
    fig.suptitle("Transformer attribution localizes the cigarette where generic methods do not\n"
                 "Fair baseline: gradient-free attention rollout (diffuse); Grad-CAM degenerates on this ViT",
                 fontsize=12, y=.995)
    plt.savefig(P2 + r"\figures\fig37_chefer_xai.png", dpi=300, bbox_inches="tight")
    plt.savefig(EV + r"\figures\fig37_chefer_xai.png", dpi=300, bbox_inches="tight")
    plt.close()
    # ---------- SLAYT SURUMU (genis 16:9 dostu: 2 ornek satiri + yandan boxplot) ----------
    show_s = [("cigarette", cig[0], 0), ("hard look-alike", fp_hard[0], 1)]
    figs = plt.figure(figsize=(16, 6.2))
    gss = figs.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.35], hspace=.12, wspace=.06)
    for i, (name, r, cls) in enumerate(show_s):
        img = imread_u(r["path"])
        cam, roll, rel = run(img, cls)
        for j, (im, ttl) in enumerate([(cv2.resize(img, (224, 224)) / 255.0, name),
                                       (overlay(img, cam), "Grad-CAM\n(dejenere)"),
                                       (overlay(img, roll), "Attention-rollout\n(dağınık)"),
                                       (overlay(img, rel), "Chefer\n(localize)")]):
            ax = figs.add_subplot(gss[i, j]); ax.imshow(np.clip(im, 0, 1)); ax.axis("off")
            if i == 0:
                ax.set_title(ttl, fontsize=13, fontweight="bold")
    axs = figs.add_subplot(gss[:, 4])
    bps = axs.boxplot([cc_gc, cc_ro, cc_ch], vert=False, patch_artist=True, widths=.5)
    axs.set_yticks([1, 2, 3]); axs.set_yticklabels(["Grad-CAM", "rollout", "Chefer"], fontsize=12)
    for patch, c in zip(bps["boxes"], ["#999999", "#c0392b", "#2b7bba"]):
        patch.set_facecolor(c); patch.set_alpha(.8)
    axs.axvline(0.10, color="#666", ls="--", lw=1.2)
    axs.text(0.11, 2.60, "random 0.10", fontsize=9, color="#666", va="center")
    axs.set_xlabel("Localization concentration (↑ localize)", fontsize=11)
    axs.set_title(f"n={len(cc_ch)} crops\nChefer {cc_ch.mean():.2f} > rollout {cc_ro.mean():.2f} > GC {cc_gc.mean():.2f}",
                  fontsize=12)
    axs.grid(alpha=.3, axis="x")
    figs.suptitle("Grad-CAM and attention rollout are diffuse; Chefer attribution is localized", fontsize=15, y=.99)
    plt.savefig(P2 + r"\figures\fig37s_chefer_slide.png", dpi=300, bbox_inches="tight")
    plt.close()

    import shutil
    shutil.copy(P2 + r"\tables\chefer_xai_localization.csv", EV + r"\chefer_xai_localization.csv")
    print("SAVED -> fig37_chefer_xai.png + chefer_xai_localization.csv (PAPER2 + runs_v3_eval)", flush=True)
    print("=== PAPER2 Chefer TAM SURUM BITTI ===", flush=True)
    print("\n*** UYARI: figurler YUZ icerir ve BLUR'LU DEGIL. Bu scripti her kosturduktan SONRA", flush=True)
    print("*** MUTLAKA calistir: python tools/paper_analysis/blur_faces_in_figs.py", flush=True)
    print("*** (aksi halde nitel panelde tanimlanabilir yuz makaleye/sunuma sizar).", flush=True)


if __name__ == "__main__":
    main()
