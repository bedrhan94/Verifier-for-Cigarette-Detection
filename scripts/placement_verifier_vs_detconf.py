# -*- coding: utf-8 -*-
r"""E42 — VERIFIER mi, yoksa sadece DET_CONF mu? (esit-korunmada taban cizgi)

SORUN: fused = det_conf x (1 - p_fp). Eger akşamki yanlis-pozitiflerin det_conf'u
zaten dusukse, SF reddindeki 0.563 -> 0.710 artisinin bir kismi GORUNUSTEN DEGIL,
det_conf'tan geliyor olabilir. O zaman "verifier esigin basaramadigini basariyor"
cumlesi desteksiz kalir.

TEST (Paper 2 §6.3'teki A1 testinin bu goruntuye uygulanmis hali):
  Verifier TP'lerin %R'sini koruyor. Dedektor guvenini, AYNI %R'yi koruyacak sekilde
  yukselt (esit-korunma kurali) ve o esikte kac look-alike FP'nin elendigine bak.
  Iki red oranini YAN YANA koy.

  ⚠️ Paper 1'deki "conf 0.70'te akşam 184 FP" gozlemi BU TESTIN YERINE GECMEZ:
     orada korunma %11'e cokuyor (TP 44), verifier'in %89-93'uyle kiyaslanamaz.

Girdi : placement_verifier_boxes{,_aksam}.csv   (GPU gerekmez)
Cikti : placement_verifier_vs_detconf.csv
"""
import os
os.environ["PYTHONUTF8"] = "1"
import csv
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
OUT = os.path.join(ROOT, "runs_v3_eval", "angle_experiment")
POS = {"A1": "K1", "A2": "K2", "A3": "K3", "A4": "K4"}


def load(suf):
    rows = []
    with open(os.path.join(OUT, f"placement_verifier_boxes{suf}.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(dict(angle=r["angle"], scenario=r["scenario"], kind=r["kind"],
                             det_conf=float(r["det_conf"]), accepted=int(r["accepted"])))
    return rows


def analyse(rows, unit, sel):
    tp = [r for r in rows if sel(r) and r["kind"] == "TP"]
    fp = [r for r in rows if sel(r) and r["kind"] == "FP_zero"]
    if not tp or not fp:
        return None
    # --- verifier ---
    ret_v = sum(r["accepted"] for r in tp) / len(tp)
    rej_v = sum(1 - r["accepted"] for r in fp) / len(fp)
    # --- esit-korunmada det_conf esigi ---
    # verifier TP'lerin ret_v'sini koruyor; ayni orani koruyan conf kesimi:
    c = float(np.quantile([r["det_conf"] for r in tp], 1.0 - ret_v))
    ret_c = float(np.mean([r["det_conf"] >= c for r in tp]))      # kontrol: ~ret_v olmali
    rej_c = float(np.mean([r["det_conf"] < c for r in fp]))
    return dict(birim=unit, n_TP=len(tp), n_FP0=len(fp),
                korunma_verifier=round(ret_v, 4), red_verifier=round(rej_v, 4),
                esit_korunma_conf_esigi=round(c, 4), korunma_detconf=round(ret_c, 4),
                red_detconf=round(rej_c, 4), fark=round(rej_v - rej_c, 4))


out = []
for sess, suf, clock in [("gunduz", "", "13:30"), ("aksam", "_aksam", "21:30")]:
    rows = load(suf)
    units = [("SN", lambda r: r["scenario"] == "SN"), ("SF", lambda r: r["scenario"] == "SF"),
             ("MF", lambda r: r["scenario"] == "MF"), ("ALL", lambda r: True)]
    units += [(POS[a], (lambda r, x=a: r["angle"] == x)) for a in ["A1", "A2", "A3", "A4"]]
    for u, s in units:
        d = analyse(rows, u, s)
        if d:
            d.update(oturum=sess, saat=clock)
            out.append(d)

dst = os.path.join(OUT, "placement_verifier_vs_detconf.csv")
cols = ["oturum", "saat", "birim", "n_TP", "n_FP0", "korunma_verifier", "red_verifier",
        "esit_korunma_conf_esigi", "korunma_detconf", "red_detconf", "fark"]
with open(dst, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(out)

print("=" * 96)
print("ESIT-KORUNMADA: verifier mi, det_conf mu? (look-alike/spurious FP reddi)")
print("=" * 96)
print(f"{'oturum':8s} {'birim':6s} {'n_FP0':>6s} | {'korunma':>8s} | "
      f"{'VERIFIER red':>13s} | {'conf esigi':>11s} {'DET_CONF red':>13s} | {'fark':>7s}")
print("-" * 96)
last = None
for d in out:
    if last and d["oturum"] != last:
        print("-" * 96)
    last = d["oturum"]
    star = " ⭐" if d["fark"] >= 0.10 else ("  ⚠️" if d["fark"] <= 0 else "   ")
    print(f"{d['oturum']:8s} {d['birim']:6s} {d['n_FP0']:6d} | {d['korunma_verifier']:8.3f} | "
          f"{d['red_verifier']:13.3f} | {d['esit_korunma_conf_esigi']:11.3f} "
          f"{d['red_detconf']:13.3f} | {d['fark']:+7.3f}{star}")

print(f"\n[OK] {dst}")
print("Okuma: 'fark' > 0 ise verifier, esit korunmada det_conf esiginden DAHA COK")
print("       look-alike eliyor. <= 0 ise bu goruntude verifier'in katkisi det_conf ile aciklanir.")
