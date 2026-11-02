# -*- coding: utf-8 -*-
r"""E41 — DOGRULAYICININ HIC GORULMEMIS KISIDE DIS-GECERLILIGI.

E35 (DENEY 3) leave-3-out'tu: o 3 kisi 17'lik crop havuzunda VARDI, crop'lari cikarildi.
E41'de kisi HICBIR havuzda yok -> temiz dis-ornek. Bu yuzden E41 "E35'ten GUCLU" diye
cerceveleniyor; ama bu iddiayi kurabilmek icin ikisinin AYNI dogrulayicidan ve AYNI
esikten gecmesi SART.

TASARIM KARARLARI (ON-KAYITLI, goruntuye BAKILMADAN sabit):
  * Dedektor KILITLI, yeniden egitim YOK.
  * Dogrulayici = placement_verifier_eval.py'nin 14-kisilik modeli; esik o modelin
    ESLESMIS OOF'undan %96 korunma kuralyla turetilir -> beklenen thr = 0.14042.
    ⚠️ 3 kisinin dislanmasi BURADA SIZINTI YUZUNDEN DEGIL; E35 ile KIYASLANABILIRLIK
       icin devralindi. (Nobody klibindeki kisi zaten 17'nin hicbirinde yok.)
  * REPRODUKSIYON KAPISI: thr ve OOF korunmasi E35'in degerlerini yeniden uretmezse
    script DURUR (assert) — aksi halde E41 sayilari E26/E35 ile kiyaslanamaz.

KOLLAR (AYRI DIZINLERDEN okunur — kolay gozden kacar):
  * KORUNMA (retention): POS klipleri. Goruntu+GT -> CVAT_UPLOAD/EX nobody_pos
  * RED (rejection):     FP klibi.    Goruntu -> CVAT_UPLOAD/nobody_aksam (NB_AKS_FP_*),
                         GT tasarim geregi BOS (etiketlenmedi; bkz. OKUBENI).

Cikti: runs_v3_eval/angle_experiment/e41_nobody_{boxes.csv,summary.json}
"""
import os
os.environ["PYTHONUTF8"] = "1"
import io
import sys
import csv
import json
import glob
import collections

import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import placement_verifier_eval as P          # ayni kod: Clip / crop_list / crop96 / sabitler

ROOT = P.ROOT
UP, EX, OUT = P.UP, P.EX, P.OUT
W_DEF, H_DEF = P.W_DEF, P.H_DEF
DEPLOY_CONF, IOU_MAIN = P.DEPLOY_CONF, P.IOU_MAIN
# --- oturum secimi: aksam (varsayilan) / sabah ---
import argparse as _ap
_a = _ap.ArgumentParser()
_a.add_argument("--session", choices=["aksam", "sabah"], default="aksam")
_a.add_argument("--tag", default=None)
_ARGS, _ = _a.parse_known_args()
_SES = _ARGS.session
PRED = os.path.join(OUT, "_pred_boxes_nobody.csv" if _SES == "aksam"
                    else "_pred_boxes_nobody_sabah.csv")
RETENTION = 0.96
THR_EXPECTED = 0.14042                       # E35'te olculen; kapi bunu bekler
TOL = 5e-4

POS_TASK = "nobody_pos" if _SES == "aksam" else "nobody_sabah_pos"
FULL_TASK = "nobody_aksam" if _SES == "aksam" else "nobody_sabah"
FP_PREFIX = "NB_AKS_FP_" if _SES == "aksam" else "NB_SAB_FP_"
TAG = _ARGS.tag if _ARGS.tag is not None else ("" if _SES == "aksam" else "_sabah")
E26_MEAN, E26_SD = 0.958, 0.047              # per-fold korunma dagilimi (konumlandirma icin)


def clip_feats_cached():
    """CLIP ozellikleri — placement_verifier_eval ile AYNI cache."""
    paths, y_all, g_all = P.crop_list()
    d = np.load(P.CACHE, allow_pickle=True)
    assert len(paths) == len(d["y"]), f"crop sayisi uyusmuyor: {len(paths)} vs {len(d['y'])}"
    assert (y_all == d["y"]).all() and (g_all == d["g"]).all(), "cache hizalamasi BOZUK"
    if os.path.exists(P.FEAT_CACHE):
        F = np.load(P.FEAT_CACHE)["F"]
        assert F.shape[0] == len(paths)
        print(f"[clip] cache'ten: {F.shape}")
    else:
        clip = P.Clip()
        F = clip.feats([P.imread_u(p) for p in tqdm(paths, desc="crop oku", ncols=88)], "clip crops")
        np.savez_compressed(P.FEAT_CACHE, F=F)
    return F, y_all, g_all, d["dconf"]


def derive_threshold(F_all, y_all, g_all, dconf_all):
    """E35 ile BIREBIR AYNI esik turetmesi + REPRODUKSIYON KAPISI."""
    keep = np.array([p not in P.EXCLUDE for p in g_all])
    F, yv, gv, dcv = F_all[keep], y_all[keep], g_all[keep], dconf_all[keep]
    people = sorted(set(gv))
    assert not (P.EXCLUDE & set(people)), "SIZINTI: dislanan kisi havuzda!"
    print(f"[havuz] {len(people)} kisi, {len(yv)} crop "
          f"(cig={int((yv==0).sum())} fp={int((yv==1).sum())}) | dislanan: {sorted(P.EXCLUDE)}")
    oof = np.full(len(yv), np.nan)
    for pr in tqdm(people, desc="14-ici LOPO", ncols=88):
        te = gv == pr
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                            random_state=0).fit(F[~te], yv[~te])
        oof[te] = clf.predict_proba(F[te])[:, 1]
    assert not np.isnan(oof).any()
    fused = dcv * (1.0 - oof)
    thr = float(np.quantile(fused[yv == 0], 1.0 - RETENTION))
    ret = float((fused[yv == 0] >= thr).mean())
    rej = float((fused[yv == 1] < thr).mean())
    print(f"\n{'='*78}\nREPRODUKSIYON KAPISI\n{'='*78}")
    print(f"  thr        = {thr:.5f}   (E35: {THR_EXPECTED:.5f})")
    print(f"  OOF korunma= {ret:.4f}   (E35: 0.9600)")
    print(f"  OOF red    = {rej:.4f}")
    assert abs(thr - THR_EXPECTED) < TOL, (
        f"KAPI GECILMEDI: thr {thr:.5f} != {THR_EXPECTED:.5f}. "
        "E41 sayilari E26/E35 ile KIYASLANAMAZ -> yorumlama.")
    print("  ✅ KAPI GECILDI — esik E35 ile ayni, kiyas gecerli\n")
    clf_final = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                              random_state=0).fit(F, yv)
    return thr, clf_final


def load_gt_pos():
    """POS klipleri: tum kareler bos baslar, label dosyasi varsa dolar."""
    gt = {}
    for p in sorted(glob.glob(os.path.join(UP, POS_TASK, "images", "train", "*.jpg"))):
        gt[os.path.basename(p)] = []
    n = 0
    for f in glob.glob(os.path.join(EX, POS_TASK, "labels", "**", "*.txt"), recursive=True):
        k = os.path.splitext(os.path.basename(f))[0] + ".jpg"
        if k in gt:
            n += 1
            for ln in open(f, encoding="utf-8"):
                q = ln.split()
                if len(q) >= 5:
                    gt[k].append(P.yolo_to_xyxy(*map(float, q[1:5]), W_DEF, H_DEF))
    print(f"[GT POS] {len(gt)} kare | etiket dosyasi {n} | "
          f"kutusuz {sum(1 for v in gt.values() if not v)} | kutu {sum(len(v) for v in gt.values())}")
    return gt


def fp_frames():
    """FP klibi kareleri — AYRI dizinden; GT tasarim geregi BOS."""
    fr = sorted(os.path.basename(p) for p in
                glob.glob(os.path.join(UP, FULL_TASK, "images", "train", FP_PREFIX + "*.jpg")))
    print(f"[FP klibi] {len(fr)} kare (GT bos — gercek sigara yok)")
    assert fr, f"FP kareleri bulunamadi! ({FULL_TASK}/images/train/{FP_PREFIX}*)"
    return fr


def main():
    F_all, y_all, g_all, dconf_all = clip_feats_cached()
    thr, clf_final = derive_threshold(F_all, y_all, g_all, dconf_all)

    gt = load_gt_pos()
    fps = fp_frames()
    for k in fps:
        gt[k] = []                                   # FP klibi: bos GT

    pred = collections.defaultdict(list)
    for r in csv.DictReader(open(PRED, encoding="utf-8-sig")):
        pred[r["image"]].append(((float(r["x1"]), float(r["y1"]),
                                  float(r["x2"]), float(r["y2"])), float(r["conf"])))

    # --- kutulari sinifla: TP / FP_nearmiss / FP_zero (E35 ile ayni) ---
    rows = []
    img_dir = {os.path.basename(p): p for t in (POS_TASK, FULL_TASK)
               for p in glob.glob(os.path.join(UP, t, "images", "train", "*.jpg"))}
    for im in tqdm(sorted(gt), desc="kutu sinifla", ncols=88):
        pb = sorted([(b, s) for b, s in pred.get(im, []) if s >= DEPLOY_CONF], key=lambda x: -x[1])
        if not pb:
            continue
        fr = P.imread_u(img_dir[im])
        g = list(gt.get(im, []))
        used = set()
        for b, s in pb:
            best, bi = 0.0, -1
            for j, gg in enumerate(g):
                if j in used:
                    continue
                v = P.iou_xyxy(b, gg) if hasattr(P, "iou_xyxy") else _iou(b, gg)
                if v > best:
                    best, bi = v, j
            if best >= IOU_MAIN and bi >= 0:
                kind = "TP"; used.add(bi)
            else:
                any_ov = max([(P.iou_xyxy(b, gg) if hasattr(P, "iou_xyxy") else _iou(b, gg))
                              for gg in g], default=0.0)
                kind = "FP_nearmiss" if any_ov > 0 else "FP_zero"
            clip_grp = ("FP" if im.startswith(FP_PREFIX) else
                        ("POS1" if "POS1" in im else "POS2"))
            rows.append(dict(image=im, clip=clip_grp, kind=kind, det_conf=s,
                             crop=P.crop96(fr, b)))

    Fb = P.Clip().feats([r.pop("crop") for r in rows], "clip kutular")
    p_fp = clf_final.predict_proba(Fb)[:, 1]
    for r, pf in zip(rows, p_fp):
        r["p_fp"] = float(pf)
        r["fused"] = r["det_conf"] * (1.0 - pf)
        r["accepted"] = int(r["fused"] >= thr)

    # --- rapor ---
    def rate(sel, want_accept):
        n = sum(1 for r in rows if sel(r))
        if not n:
            return float("nan"), 0
        k = sum(1 for r in rows if sel(r) and (r["accepted"] == 1) == want_accept)
        return k / n, n

    print(f"\n{'='*86}\nE41 [{_SES.upper()}] — HIC GORULMEMIS KISI "
          f"(conf {DEPLOY_CONF}, IoU {IOU_MAIN}, thr {thr:.5f})\n{'='*86}")
    print(f"{'kol':22s} {'korunma/red':>12s} {'n':>6s}")
    ret_pos, n_tp = rate(lambda r: r["kind"] == "TP" and r["clip"] != "FP", True)
    print(f"{'KORUNMA (POS, TP)':22s} {ret_pos:12.3f} {n_tp:6d}")
    rej_fp, n_fp = rate(lambda r: r["clip"] == "FP" and r["kind"] == "FP_zero", False)
    print(f"{'RED (FP klibi)':22s} {rej_fp:12.3f} {n_fp:6d}")
    rej_pos, n_pz = rate(lambda r: r["clip"] != "FP" and r["kind"] == "FP_zero", False)
    print(f"{'RED (POS kliplerinde)':22s} {rej_pos:12.3f} {n_pz:6d}")
    rejn, n_nm = rate(lambda r: r["kind"] == "FP_nearmiss", False)
    print(f"{'RED (yakin-isabet)':22s} {rejn:12.3f} {n_nm:6d}")

    z = (ret_pos - E26_MEAN) / E26_SD if n_tp else float("nan")
    lo, hi = E26_MEAN - 2 * E26_SD, E26_MEAN + 2 * E26_SD
    print(f"\nE26 per-fold korunma dagilimina konum: {E26_MEAN:.3f} ± {E26_SD:.3f} "
          f"(±2sd: {lo:.3f}–{hi:.3f})")
    print(f"  E41 korunma {ret_pos:.3f}  ->  z = {z:+.2f}  "
          f"{'ICINDE' if lo <= ret_pos <= hi else 'DISINDA'}")
    print("\n⚠️ TEK KISI = ILLUSTRATIF, benchmark DEGIL (on-kayitli kavat).")
    print("⚠️ FP klibinde look-alike nesne ETIKETLENMEDI -> 'look-alike reddi' degil,")
    print("   'yanlis ateslerin reddi' diye raporlanir (E35'teki ayni kavat).")

    with open(os.path.join(OUT, "e41_nobody_boxes" + TAG + ".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "clip", "kind", "det_conf", "p_fp",
                                           "fused", "accepted"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "e41_nobody_summary" + TAG + ".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(threshold=thr, retention_target=RETENTION,
                       retention_pos=None if np.isnan(ret_pos) else round(ret_pos, 4), n_TP=n_tp,
                       reject_fp_clip=None if np.isnan(rej_fp) else round(rej_fp, 4), n_FP_clip=n_fp,
                       reject_pos_clip=None if np.isnan(rej_pos) else round(rej_pos, 4), n_FP_pos=n_pz,
                       reject_nearmiss=None if np.isnan(rejn) else round(rejn, 4), n_nearmiss=n_nm,
                       e26_mean=E26_MEAN, e26_sd=E26_SD,
                       z_vs_e26=None if np.isnan(z) else round(z, 3),
                       caveat="tek kisi = illustratif; look-alike etiketlenmedi"),
                  fh, indent=2, ensure_ascii=False)
    print("\n[OK] e41_nobody_boxes.csv / e41_nobody_summary.json")


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


if __name__ == "__main__":
    main()
