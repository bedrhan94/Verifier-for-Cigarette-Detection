# -*- coding: utf-8 -*-
r"""DENEY 3c ADIM 2 — verifier'i OLAY seviyesinde uygulamanin tasarim karsilastirmasi.

PROBLEM (olculdu, MASTER §4.20): verifier KARE seviyesinde uygulaninca gercek sigara
epizodunun kapsanan suresi 12 klibin 10'unda DUSUYOR (ort. -0.058, hicbirinde artmiyor),
ve kalabalik MF sahnesinde olay SAYISI artiyor (epizod boluniyor).

=========================== ON-KAYIT: PARAMETRELER SONUCA BAKILMADAN SABITLENDI ==========
FRAME  (mevcut) : present = fused >= thr                       [yeni parametre: 0]
A  OLAY-KAPISI  : FSM ham sinyalde kosar -> aday olaylar; olayin tespit iceren
                  karelerindeki fused'in ORTALAMASI >= AYNI thr ise olay KALIR.
                  Olay ya butun kalir ya butun gider -> BOLUNME MATEMATIKSEL OLARAK IMKANSIZ.
                                                               [yeni parametre: 0]
A2 OLAY-KAPISI  : ayni ama COGUNLUK oyu (karelerin >=%50'si esigi gecerse kalir).
   (cogunluk)                                                  [yeni parametre: 1 (0.5 = dogal varsayilan)]
B  YUMUSATMA    : fused'a W=K_off+1=9 karelik medyan filtre, sonra esik.
                  W, FSM'in MEVCUT kacirma toleransina baglandi - serbest secilmedi.
                                                               [yeni parametre: 0 (turetilmis)]
C  HISTEREZIS   : ac=thr, kapa=thr/2 (sabit 2x oran, onceden ilan edildi).
                                                               [yeni parametre: 1 (2x orani)]
Hepsi sonuc ne cikarsa ciksin RAPORLANIR. Kazanan sonradan secilmez; tablo tam verilir.
=========================================================================================

IKI KUME — biri olmadan tablo kurulamaz:
  MALIYET: 12 konum klibi. Sigara neredeyse surekli (GT varlik .97-1.00) => temiz negatif
           donem YOK => yanlis olay olusamaz => yalniz KAPSAMA/BOLUNME maliyeti olculur.
  FAYDA  : S02_FP klibi. Sigara HIC YOK => temporal'i gecen HER olay yanlistir => silinen
           olay sayisi dogrudan faydadir. S02_POS ise fayda'nin bedelini (gercek olay
           kaybi) verir.

Kosma: python tools/paper_analysis/verifier_event_designs.py
"""
import os
os.environ["PYTHONUTF8"] = "1"
import sys
import csv
import json
import glob
import numpy as np

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from eval_video_metrics import build_events_from_binary   # noqa: E402

D = os.path.join(ROOT, "runs_v3_eval", "angle_experiment")
SC = os.path.join(D, "_scores")
CONFIRM_S, STOP_MISS, MIN_DUR_S = 2.0, 8, 0.8
MED_W = STOP_MISS + 1          # 9 — FSM'in kacirma toleransina bagli
HYST_RATIO = 0.5               # kapa esigi = thr * 0.5
POS = {"A1": "K1", "A2": "K2", "A3": "K3", "A4": "K4"}


def median_filter(x, w):
    if w <= 1:
        return x.copy()
    h = w // 2
    pad = np.pad(x, (h, h), mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(len(x))], np.float32)


def hysteresis(x, hi, lo):
    out = np.zeros(len(x), np.int8)
    on = False
    for i, v in enumerate(x):
        if not on and v >= hi:
            on = True
        elif on and v < lo:
            on = False
        out[i] = 1 if on else 0
    return out


def events(sig, fps):
    kon = max(1, int(round(CONFIRM_S * fps)))
    return build_events_from_binary(sig, fps, kon, STOP_MISS, MIN_DUR_S)


def ev_mask(ev, n):
    m = np.zeros(n, np.int8)
    for a, b in ev:
        m[a:b + 1] = 1
    return m


def designs(raw, fused, fps, thr):
    """Doner: {ad: (olay_listesi, OLAY MASKESI)}

    ⚠️ KAPSAMA HER TASARIM ICIN AYNI SEYI OLCER: onaylanmis OLAY icinde kalan kare orani
    (kare-basi ham varlik sinyali DEGIL). Ilk surumde RAW/FRAME/B/C icin kare sinyali,
    A/A2 icin olay maskesi kullaniliyordu -> elmayla armut. Duzeltildi.
    """
    n = len(raw)
    out = {}

    def pack(ev):
        return (ev, ev_mask(ev, n))

    out["RAW"] = pack(events(raw, fps))
    out["FRAME"] = pack(events((fused >= thr).astype(np.int8), fps))

    # A / A2 — olay kapisi (ham FSM olaylari uzerinde; olay ya butun kalir ya butun gider)
    base = events(raw, fps)
    for name, rule in (("A_mean", "mean"), ("A2_major", "major")):
        keep_ev = []
        for a, b in base:
            m = raw[a:b + 1] == 1
            if not m.any():
                continue
            f = fused[a:b + 1][m]
            ok = (f.mean() >= thr) if rule == "mean" else ((f >= thr).mean() >= 0.5)
            if ok:
                keep_ev.append((a, b))
        out[name] = pack(keep_ev)

    # B — medyan yumusatma
    out["B_median"] = pack(events((median_filter(fused, MED_W) >= thr).astype(np.int8), fps))

    # C — histerezis
    out["C_hyst"] = pack(events(hysteresis(fused, thr, thr * HYST_RATIO), fps))
    return out


def main():
    meta = json.load(open(os.path.join(SC, "_meta.json"), encoding="utf-8"))
    thr_p, thr_a = meta["thr_placement"], meta["thr_ali"]

    # ---------------- MALIYET: 12 konum klibi ----------------
    rows = []
    agg = {}
    for f in sorted(glob.glob(os.path.join(SC, "A?_??.npz"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        z = np.load(f)
        raw, fused, fps, n = z["raw"], z["fused"], float(z["fps"]), int(z["n"])
        res = designs(raw, fused, fps, thr_p)
        base_ev, base_m = res["RAW"]
        base_cov = base_m.mean()
        r = dict(stem=stem, konum=POS[stem.split("_")[0]], senaryo=stem.split("_")[1],
                 olay_RAW=len(base_ev), kapsama_RAW=round(float(base_cov), 4))
        for k, (ev, m) in res.items():
            if k == "RAW":
                continue
            r[f"olay_{k}"] = len(ev)
            r[f"kapsama_{k}"] = round(float(m.mean()), 4)
            a = agg.setdefault(k, dict(dev=[], dcov=[]))
            a["dev"].append(len(ev) - len(base_ev))
            a["dcov"].append(float(m.mean() - base_cov))
        rows.append(r)

    print("=" * 96)
    print("MALIYET — 12 konum klibi (sigara neredeyse surekli; yanlis olay OLUSAMAZ)")
    print("=" * 96)
    print(f"{'tasarim':10s} {'d_olay top':>11s} {'artan/12':>9s} {'d_kapsama ort':>14s} "
          f"{'dusen/12':>9s} {'en kotu':>9s}")
    summ = {}
    for k in ["FRAME", "A_mean", "A2_major", "B_median", "C_hyst"]:
        dv, dc = agg[k]["dev"], agg[k]["dcov"]
        summ[k] = dict(d_event_total=int(sum(dv)), n_event_up=int(sum(1 for x in dv if x > 0)),
                       d_cov_mean=round(float(np.mean(dc)), 4),
                       n_cov_down=int(sum(1 for x in dc if x < -1e-9)),
                       d_cov_worst=round(float(min(dc)), 4))
        s = summ[k]
        print(f"{k:10s} {s['d_event_total']:+11d} {s['n_event_up']:9d} "
              f"{s['d_cov_mean']:+14.4f} {s['n_cov_down']:9d} {s['d_cov_worst']:+9.4f}")

    # ---------------- FAYDA: S02 klipleri ----------------
    print()
    print("=" * 96)
    print("FAYDA / BEDEL - S02 klipleri (leave-S02-out verifier, ayri esik)")
    print("  S02_FP : sigara HIC YOK  -> her olay YANLIS  -> az olay = IYI")
    print("  S02_POS: gercek sigara   -> olay ve kapsama KORUNMALI")
    print("=" * 96)
    s02 = {}
    for stem in ["S02_FP", "S02_POS"]:
        p = os.path.join(SC, stem + ".npz")
        if not os.path.exists(p):
            print(f"  [YOK] {stem}")
            continue
        z = np.load(p)
        raw, fused, fps = z["raw"], z["fused"], float(z["fps"])
        res = designs(raw, fused, fps, thr_a)
        base_ev, base_m = res["RAW"]
        print(f"\n{stem}  (RAW: {len(base_ev)} olay, kapsama {base_m.mean():.3f})")
        print(f"  {'tasarim':10s} {'olay':>6s} {'d_olay':>7s} {'kapsama':>9s} {'d_kaps':>8s}")
        s02[stem] = dict(RAW=dict(events=len(base_ev), cov=round(float(base_m.mean()), 4)))
        for k in ["FRAME", "A_mean", "A2_major", "B_median", "C_hyst"]:
            ev, m = res[k]
            print(f"  {k:10s} {len(ev):6d} {len(ev)-len(base_ev):+7d} "
                  f"{m.mean():9.3f} {m.mean()-base_m.mean():+8.3f}")
            s02[stem][k] = dict(events=len(ev), d_events=len(ev) - len(base_ev),
                                cov=round(float(m.mean()), 4),
                                d_cov=round(float(m.mean() - base_m.mean()), 4))

    cols = list(rows[0])
    with open(os.path.join(D, "verifier_event_designs.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(D, "verifier_event_designs_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(thr_placement=thr_p, thr_ali=thr_a, median_window=MED_W,
                       hyst_ratio=HYST_RATIO, confirm_s=CONFIRM_S, stop_miss=STOP_MISS,
                       min_dur_s=MIN_DUR_S, cost_placement=summ, benefit_s02=s02,
                       note=("Parametreler sonuca bakilmadan sabitlendi; tum tasarimlar "
                             "sonuc ne olursa olsun raporlanir.")),
                  f, indent=2, ensure_ascii=False)
    print("\n[OK] verifier_event_designs.csv / verifier_event_designs_summary.json")


if __name__ == "__main__":
    main()
