# -*- coding: utf-8 -*-
r"""DENEY 3 — Verifier'in KONUM cekimlerindeki (DENEY 2) gercek katkisi.

SORU: dedektorun urettigi kutulari, verifier dogru ayiriyor mu?
      TP kutular (gercek sigara, GT ile eslesen) KORUNMALI,
      FP boxes (above all the held look-alike object) must be REJECTED.

=========================== DURUSTLUK TASARIMI ===========================
1. SIZINTI ENGELI — LEAVE-3-OUT. The three smokers in the clips are in the verifier's
   17 kisilik egitim kumesinde VAR. O yuzden verifier bu UCU DISLAYARAK, 14 kisiyle egitilir.
   The look-alike holder is NOT among the 17 at all ⇒ FP tarafi bastan temiz.
   ⇒ Bu goruntulerdeki HICBIR kisiyi verifier gormemistir (tam held-out).
   Crop'lar 2026-07-05'te uretildi, klipler 2026-07-22'de cekildi ⇒ zaman olarak da ayrik.

2. ESIK, 14-KISILIK MODELE UYUMLU TURETILIR (kritik).
   Mevcut `_batchA_cache.npz` 17-kisilik LOPO'nun OOF ciktisidir; oradan alinan esik 14-kisilik
   modelin P(cig) olceginde BASKA bir korunma oranina denk gelir. Bu yuzden 14 kisi ICINDE
   yeniden LOPO kosulup **eslesmis OOF cache** uretilir ve esik ORADAN alinir.
   (Eski cache'ten 3 satiri silmek EDEGER DEGILDIR: o OOF tahminleri, 3'unu GORMUS modellerden gelir.)
   Esik goruntuye bakilarak AYARLANMAZ; sabit %96 sigara-korunmasi kuralindan gelir.

3. FP'LER HOMOJEN DEGIL — AYRISTIRILIR:
     - SN senaryosu: sahnede look-alike YOK ⇒ bunlar arka plan/spurious atesler (baska mekanizma)
     - SF+MF: the held look-alike ⇒ verifier'in ASIL hedefi
     - "yakin-isabet" (0 < IoU < 0.30): gercek sigaranin uzerinde ama kotu lokalize ⇒ bunu
       kabul etmek tartismali bir hata; ayri raporlanir.

4. Bu bir TESPIT-SEVIYESI ayirt etme olcumudur; Paper 2'nin OLAY-seviyesi "esit korunmada 3x"
   basligiyla AYNI metrik DEGILDIR. Ikisi karistirilmayacak.
=========================================================================

Kosma: python tools/paper_analysis/placement_verifier_eval.py [--retention 0.96]
"""
import os
os.environ["PYTHONUTF8"] = "1"
import sys
import csv
import glob
import json
import argparse
import numpy as np
import cv2
import torch
import timm
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_video_metrics import iou_xyxy   # noqa: E402

CROPS = os.path.join(ROOT, "crops_sweep", "W96")
CACHE = os.path.join(ROOT, "runs_v3_eval", "_batchA_cache.npz")
UP = os.path.join(ROOT, "CVAT_UPLOAD")
EX = os.path.join(ROOT, "CVAT_EXPORT")
OUT = os.path.join(ROOT, "runs_v3_eval", "angle_experiment")
FEAT_CACHE = os.path.join(OUT, "_clip_feats_crops.npz")

# OTURUM HARITASI — AKSAM tekrari (E42) ayni dort konum, ayni uc senaryo ve AYNI kisilerle
# cekildi: the same three smokers, with the same look-alike holder (havuzda HIC yok).
# Dolayisiyla leave-3-out tasarimi ve 14-kisilik eslesmis esik DEGISMEDEN gecerlidir;
# gunduz/aksam farki SADECE isik rejiminden gelir. Esik akşama bakilarak YENIDEN AYARLANMAZ.
SESSIONS = {
    "gunduz": dict(tasks=[f"aci_A{i}" for i in (1, 2, 3, 4)], suffix="", clock="13:30"),
    "aksam":  dict(tasks=[f"aksam_A{i}" for i in (1, 2, 3, 4)], suffix="_aksam", clock="21:30"),
}
# REPRODUKSIYON KAPISI — donmus gunduz esigi (placement_verifier_summary.json, 2026-07-23).
# Boru hatti (CLIP ozellikleri + 14-ici LOPO + sklearn) bu sayiyi yeniden uretemiyorsa
# akşam sayilari gunduzle KIYASLANAMAZ; script cikti yazmadan ONCE assert ile durur.
THR_FROZEN = 0.14042356655120855
THR_TOL = 1e-9
POS = {"A1": "K1", "A2": "K2", "A3": "K3", "A4": "K4"}
EXCLUDE = {"kisi_01", "kisi_04", "kisi_05"}   # kliplerdeki icenler
CLIP_NAME = "vit_base_patch16_clip_224.openai"
CROP_PX = 96
DEPLOY_CONF = 0.30
IOU_MAIN = 0.30
W_DEF, H_DEF = 1920, 1080
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


class Clip:
    def __init__(self):
        self.m = timm.create_model(CLIP_NAME, pretrained=True, num_classes=0).eval().to(DEV)
        cfg = timm.data.resolve_model_data_config(self.m)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(DEV)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(DEV)
        self.sz = cfg["input_size"][-1]

    def feats(self, imgs, desc="clip"):
        if not len(imgs):
            return np.zeros((0, 768), np.float32)
        out = []
        for i in tqdm(range(0, len(imgs), 64), desc=desc, ncols=88):
            buf = [cv2.resize(im, (self.sz, self.sz)) for im in imgs[i:i + 64]]
            with torch.no_grad():
                x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().to(DEV) / 255.0
                out.append(self.m((x - self.mean) / self.std).cpu().numpy())
        F = np.concatenate(out, 0)
        return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


def crop96(fr, b):
    H, W = fr.shape[:2]
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    h = CROP_PX // 2
    x1, y1 = int(max(0, cx - h)), int(max(0, cy - h))
    x2, y2 = int(min(W, cx + h)), int(min(H, cy + h))
    c = fr[y1:y2, x1:x2]
    return c if c.size else np.zeros((CROP_PX, CROP_PX, 3), np.uint8)


def crop_list():
    """Cache ile AYNI sirada (person -> cig,fp -> glob)."""
    paths, y, g = [], [], []
    for person in sorted(os.listdir(CROPS)):
        for cls in ("cig", "fp"):
            for p in sorted(glob.glob(os.path.join(CROPS, person, cls, "*.jpg"))):
                paths.append(p)
                y.append(0 if cls == "cig" else 1)
                g.append(person)
    return paths, np.array(y), np.array(g)


def yolo_to_xyxy(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def load_gt(TASKS):
    gt = {}
    for t in TASKS:
        for p in sorted(glob.glob(os.path.join(UP, t, "images", "train", "*.jpg"))):
            gt[os.path.basename(p)] = []
    for t in TASKS:
        for f in glob.glob(os.path.join(EX, t, "labels", "**", "*.txt"), recursive=True):
            key = os.path.splitext(os.path.basename(f))[0] + ".jpg"
            if key in gt:
                for line in open(f, encoding="utf-8"):
                    q = line.split()
                    if len(q) >= 5:
                        gt[key].append(yolo_to_xyxy(*map(float, q[1:5]), W_DEF, H_DEF))
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retention", type=float, default=0.96,
                    help="hedef sigara korunmasi; esik BUNDAN turetilir, goruntuye bakilarak DEGIL")
    ap.add_argument("--session", choices=sorted(SESSIONS), default="gunduz",
                    help="gunduz = Temmuz 13:30 cekimi (varsayilan), aksam = E42 21:30 tekrari")
    args = ap.parse_args()
    S = SESSIONS[args.session]
    TASKS, SUF = S["tasks"], S["suffix"]
    print(f"[oturum] {args.session} ({S['clock']}) | gorevler: {TASKS[0]}..{TASKS[-1]} | "
          f"cikti eki: '{SUF or '(yok)'}'")

    # ---------- 1) crop listesi + cache hizalama dogrulamasi ----------
    paths, y_all, g_all = crop_list()
    d = np.load(CACHE, allow_pickle=True)
    assert len(paths) == len(d["y"]), f"crop sayisi uyusmuyor: {len(paths)} vs {len(d['y'])}"
    assert (y_all == d["y"]).all(), "y hizalamasi BOZUK"
    assert (g_all == d["g"]).all(), "person hizalamasi BOZUK"
    dconf_all = d["dconf"]
    print(f"[hizalama] OK — {len(paths)} crop, cache ile birebir ayni sirada")

    # ---------- 2) CLIP ozellikleri (bir kez, cache'li) ----------
    clip = Clip()
    if os.path.exists(FEAT_CACHE):
        F_all = np.load(FEAT_CACHE)["F"]
        assert F_all.shape[0] == len(paths)
        print(f"[clip] cache'ten okundu: {F_all.shape}")
    else:
        imgs = [imread_u(p) for p in tqdm(paths, desc="crop oku", ncols=88)]
        assert all(im is not None for im in imgs), "okunamayan crop var"
        F_all = clip.feats(imgs, "clip crops")
        np.savez_compressed(FEAT_CACHE, F=F_all)
        print(f"[clip] hesaplandi ve cache'lendi: {F_all.shape}")

    # ---------- 3) 14 kisi: ESLESMIS OOF -> esik ----------
    keep = np.array([p not in EXCLUDE for p in g_all])
    F, yv, gv, dcv = F_all[keep], y_all[keep], g_all[keep], dconf_all[keep]
    people = sorted(set(gv))
    print(f"[leave-3-out] disarida: {sorted(EXCLUDE)} | egitim havuzu: {len(people)} kisi, "
          f"{len(yv)} crop (cig={int((yv==0).sum())} fp={int((yv==1).sum())})")
    assert not (EXCLUDE & set(people)), "SIZINTI: dislanan kisi havuzda!"

    oof = np.full(len(yv), np.nan)
    for pr in tqdm(people, desc="14-ici LOPO", ncols=88):
        te = gv == pr
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                            random_state=0).fit(F[~te], yv[~te])
        oof[te] = clf.predict_proba(F[te])[:, 1]          # P(fp)
    assert not np.isnan(oof).any()

    fused_oof = dcv * (1.0 - oof)                          # det_conf x P(cig)
    thr = float(np.quantile(fused_oof[yv == 0], 1.0 - args.retention))
    ret_check = float((fused_oof[yv == 0] >= thr).mean())
    rej_oof = float((fused_oof[yv == 1] < thr).mean())
    print(f"[esik] 14-kisilik ESLESMIS OOF'tan: thr={thr:.5f} "
          f"| OOF sigara korunmasi={ret_check:.4f} (hedef {args.retention}) "
          f"| OOF look-alike reddi={rej_oof:.4f}")

    # --- REPRODUKSIYON KAPISI (cikti YAZILMADAN once) ---
    # Esik, oturumdan BAGIMSIZ olarak ayni 14 kisilik havuzdan turetilir; dolayisiyla
    # gunduz icin dondurulmus degerle BIREBIR cikmali. Cikmiyorsa CLIP/sklearn/crop
    # tarafinda bir sey kaymistir ve gun-aksam karsilastirmasi gecersizdir.
    if abs(args.retention - 0.96) < 1e-12:
        assert abs(thr - THR_FROZEN) < THR_TOL, (
            f"REPRODUKSIYON KAPISI DUSTU: thr={thr!r} != donmus {THR_FROZEN!r} "
            f"(fark {abs(thr - THR_FROZEN):.3e}). Esik oturumdan bagimsiz turetildigi icin "
            f"bu fark gun/aksam karsilastirmasini gecersiz kilar — once sebebini bul.")
        print(f"[KAPI] OK — esik donmus degerle birebir ayni ({THR_FROZEN})")
    else:
        print(f"[KAPI] ATLANDI — retention {args.retention} != 0.96, donmus esik gecerli degil")

    # ---------- 4) 14 kisinin TAMAMIYLA nihai model ----------
    clf_final = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                              random_state=0).fit(F, yv)
    print(f"[model] 14 kisiyle egitildi (train acc={clf_final.score(F, yv):.4f})")

    # ---------- 5) klip kutularini sinifla ----------
    gt = load_gt(TASKS)
    preds = {}
    meta = {}            # image -> (konum, senaryo); dosya adini PARCALAMIYORUZ
    with open(os.path.join(OUT, f"_pred_boxes{SUF}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["conf"]) < DEPLOY_CONF:
                continue
            # Gunduz adi 'A1_SN_f..', aksam adi 'AK_A1_SN_f..' — split("_")[0] aksamda 'AK'
            # verip ayristirmayi bozuyordu. Konum/senaryo artik CSV kolonlarindan okunuyor
            # (o kolonlari zaten tahmin scripti --name-offset ile dogru yazmis).
            meta[r["image"]] = (r["angle"], r["scenario"])
            preds.setdefault(r["image"], []).append(
                ((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])), float(r["conf"])))

    rows = []
    img_dirs = {os.path.basename(p): p
                for t in TASKS
                for p in glob.glob(os.path.join(UP, t, "images", "train", "*.jpg"))}
    for im in tqdm(sorted(preds), desc="kutu sinifla", ncols=88):
        fr = imread_u(img_dirs[im])
        g = list(gt.get(im, []))
        used = set()
        for b, s in sorted(preds[im], key=lambda x: -x[1]):
            best, bi = 0.0, -1
            for j, gg in enumerate(g):
                if j in used:
                    continue
                v = iou_xyxy(b, gg)
                if v > best:
                    best, bi = v, j
            if best >= IOU_MAIN and bi >= 0:
                kind, used = "TP", used | {bi}
            else:
                # yakin-isabet mi (herhangi bir GT ile ortusuyor mu), yoksa tamamen bos mu?
                any_ov = max([iou_xyxy(b, gg) for gg in g], default=0.0)
                kind = "FP_nearmiss" if any_ov > 0 else "FP_zero"
            rows.append(dict(image=im, angle=meta[im][0], scenario=meta[im][1],
                             kind=kind, det_conf=s, crop=crop96(fr, b)))

    F_box = clip.feats([r.pop("crop") for r in rows], "clip kutular")
    p_fp = clf_final.predict_proba(F_box)[:, 1]
    for r, pf in zip(rows, p_fp):
        r["p_fp"] = float(pf)
        r["fused"] = r["det_conf"] * (1.0 - float(pf))
        r["accepted"] = int(r["fused"] >= thr)

    # ---------- 6) raporla ----------
    def rate(sel, want_accept):
        n = sum(1 for r in rows if sel(r))
        if n == 0:
            return float("nan"), 0
        k = sum(1 for r in rows if sel(r) and (r["accepted"] == 1) == want_accept)
        return k / n, n

    print(f"\n{'='*82}\nVERIFIER — TESPIT SEVIYESI (conf {DEPLOY_CONF}, IoU {IOU_MAIN}, "
          f"esik %{args.retention*100:.0f} korunma kuralindan)\n{'='*82}")
    print(f"{'senaryo':8s} {'sigara KORUNMA':>16s} {'n_TP':>6s} | "
          f"{'look-alike RED':>15s} {'n_FP0':>6s} | {'yakin-isabet RED':>17s} {'n_FPn':>6s}")
    summary = {}
    for sc in ["SN", "SF", "MF", "ALL"]:
        f_sc = (lambda r: True) if sc == "ALL" else (lambda r, s=sc: r["scenario"] == s)
        ret, ntp = rate(lambda r: f_sc(r) and r["kind"] == "TP", True)
        rej, nfp = rate(lambda r: f_sc(r) and r["kind"] == "FP_zero", False)
        rejn, nfn = rate(lambda r: f_sc(r) and r["kind"] == "FP_nearmiss", False)
        print(f"{sc:8s} {ret:16.3f} {ntp:6d} | {rej:15.3f} {nfp:6d} | {rejn:17.3f} {nfn:6d}")
        summary[sc] = dict(retention=None if np.isnan(ret) else round(ret, 4), n_TP=ntp,
                           reject_FP_zero=None if np.isnan(rej) else round(rej, 4), n_FP_zero=nfp,
                           reject_FP_nearmiss=None if np.isnan(rejn) else round(rejn, 4),
                           n_FP_nearmiss=nfn)

    print("\nNOT: SN'de sahnede look-alike YOK -> oradaki FP'ler arka-plan/spurious atestir,")
    print("     SF/MF'teki look-alike FP'leriyle AYNI mekanizma degildir; ayri okunmalidir.")

    print(f"\n{'-'*82}\nKONUM BAZINDA (tum senaryolar)")
    print(f"{'konum':8s} {'korunma':>9s} {'n_TP':>6s} {'FP0 red':>9s} {'n_FP0':>6s}")
    for a in ["A1", "A2", "A3", "A4"]:
        ret, ntp = rate(lambda r, x=a: r["angle"] == x and r["kind"] == "TP", True)
        rej, nfp = rate(lambda r, x=a: r["angle"] == x and r["kind"] == "FP_zero", False)
        print(f"{POS[a]:8s} {ret:9.3f} {ntp:6d} {rej:9.3f} {nfp:6d}")
        summary[POS[a]] = dict(retention=round(ret, 4), n_TP=ntp,
                               reject_FP_zero=round(rej, 4), n_FP_zero=nfp)

    # ---------- 7) ONCE / SONRA: operasyonel kazanc ----------
    n_gt = {}
    with open(os.path.join(OUT, f"placement_stratified{SUF}.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["konum"] == "TOPLAM":
                n_gt[r["senaryo"]] = int(r["n_gt"])
            if r["senaryo"] == "TOPLAM":
                n_gt[r["konum"]] = int(r["n_gt"])
    n_gt["ALL"] = sum(n_gt[s] for s in ("SN", "SF", "MF"))

    def before_after(key, sel):
        R = [r for r in rows if sel(r)]
        TP = [r for r in R if r["kind"] == "TP"]
        FP = [r for r in R if r["kind"] != "TP"]
        tp0, fp0 = len(TP), len(FP)
        tp1 = sum(1 for r in TP if r["accepted"])
        fp1 = sum(1 for r in FP if r["accepted"])
        G = n_gt[key]
        P0, R0 = tp0 / max(tp0 + fp0, 1), tp0 / G
        P1, R1 = (tp1 / (tp1 + fp1) if tp1 + fp1 else float("nan")), tp1 / G
        F0 = 2 * P0 * R0 / (P0 + R0)
        F1 = 2 * P1 * R1 / (P1 + R1)
        print(f"{key:5s} | ONCE  P={P0:.3f} R={R0:.3f} F1={F0:.3f} (TP{tp0} FP{fp0})"
              f"  ->  SONRA P={P1:.3f} R={R1:.3f} F1={F1:.3f} (TP{tp1} FP{fp1})"
              f"  | dP={P1-P0:+.3f} dR={R1-R0:+.3f} dF1={F1-F0:+.3f}")
        return dict(key=key, n_gt=G, TP_before=tp0, FP_before=fp0, TP_after=tp1, FP_after=fp1,
                    P_before=round(P0, 4), R_before=round(R0, 4), F1_before=round(F0, 4),
                    P_after=round(P1, 4), R_after=round(R1, 4), F1_after=round(F1, 4),
                    dP=round(P1 - P0, 4), dR=round(R1 - R0, 4), dF1=round(F1 - F0, 4))

    print(f"\n{'='*82}\nOPERASYONEL KAZANC — verifier ONCESI / SONRASI\n{'='*82}")
    ba = []
    print("--- senaryo ---")
    for sc in ["SN", "SF", "MF", "ALL"]:
        ba.append(before_after(sc, (lambda r: True) if sc == "ALL"
                               else (lambda r, s=sc: r["scenario"] == s)))
    print("--- konum ---")
    for a in ["A1", "A2", "A3", "A4"]:
        ba.append(before_after(POS[a], lambda r, x=a: r["angle"] == x))
    with open(os.path.join(OUT, f"placement_verifier_gain{SUF}.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ba[0]))
        w.writeheader()
        w.writerows(ba)
    summary["before_after"] = ba

    with open(os.path.join(OUT, f"placement_verifier_boxes{SUF}.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "angle", "scenario", "kind", "det_conf",
                                          "p_fp", "fused", "accepted"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, f"placement_verifier_summary{SUF}.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(session=args.session, shoot_clock=S["clock"], tasks=TASKS,
                       retention_target=args.retention, threshold=thr,
                       threshold_frozen_gate=THR_FROZEN,
                       oof_retention=ret_check, oof_rejection=rej_oof,
                       excluded_people=sorted(EXCLUDE), train_people=people,
                       n_train_crops=int(len(yv)), deploy_conf=DEPLOY_CONF, iou=IOU_MAIN,
                       results=summary), f, indent=2, ensure_ascii=False)
    print(f"\n[OK] placement_verifier_boxes{SUF}.csv / placement_verifier_summary{SUF}.json")


if __name__ == "__main__":
    main()
