# -*- coding: utf-8 -*-
r"""E42 — verifier'in GUNDUZ (13:30) ve AKSAM (21:30) yerlesim cekimlerindeki davranisi.

placement_verifier_eval.py'nin iki oturum ciktisini yan yana koyar.

NEDEN ONEMLI: Paper 1'in dedektor tarafi, akşamda look-alike yanlis-pozitiflerinin
ESIKLE temizlenemedigini gosterdi (SF senaryosu, conf 0.70: gunduz 0 FP, akşam 184 FP,
ama TP de 44'e cokuyor). Buradaki soru onun devami: GORUNUS DOGRULAYICISI, esigin
basaramadigini basariyor mu?

⚠️ Esik iki oturumda da AYNI (0.14042, 14-kisilik eslesmis OOF'tan, %96 korunma kuralindan).
   Akşama bakilarak yeniden AYARLANMADI. Reprodüksiyon kapisi gunduzu byte-byte yeniden
   uretti, o yuzden iki oturum kiyaslanabilir.

Cikti: placement_verifier_day_evening.csv
"""
import os
os.environ["PYTHONUTF8"] = "1"
import csv
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
OUT = os.path.join(ROOT, "runs_v3_eval", "angle_experiment")

D = json.load(open(os.path.join(OUT, "placement_verifier_summary.json"), encoding="utf-8"))
A = json.load(open(os.path.join(OUT, "placement_verifier_summary_aksam.json"), encoding="utf-8"))

# iki oturum da AYNI esikten gecmis olmali, yoksa kiyas gecersiz
assert abs(D["threshold"] - A["threshold"]) < 1e-12, "esikler farkli — kiyas gecersiz!"
assert D["excluded_people"] == A["excluded_people"], "dislanan kisiler farkli!"
print(f"[dogrulama] iki oturum da thr={D['threshold']:.5f}, ayni 3 kisi dislanmis — kiyas gecerli")


def gain(summ):
    return {r["key"]: r for r in summ["results"]["before_after"]}


gd, ga = gain(D), gain(A)
rows = []
for k in ["SN", "SF", "MF", "ALL", "K1", "K2", "K3", "K4"]:
    d, a = gd[k], ga[k]
    sd = D["results"].get(k, {})
    sa = A["results"].get(k, {})
    rows.append(dict(
        birim=k,
        # look-alike / spurious FP havuzunun BUYUKLUGU (sifir-ortusmeli)
        nFP0_1330=sd.get("n_FP_zero"), nFP0_2130=sa.get("n_FP_zero"),
        # verifier bunlarin ne kadarini REDDEDIYOR
        red_1330=sd.get("reject_FP_zero"), red_2130=sa.get("reject_FP_zero"),
        # sigara korunmasi (maliyet)
        kor_1330=sd.get("retention"), kor_2130=sa.get("retention"),
        # operasyonel: once/sonra
        P_once_1330=d["P_before"], P_sonra_1330=d["P_after"], dF1_1330=d["dF1"],
        P_once_2130=a["P_before"], P_sonra_2130=a["P_after"], dF1_2130=a["dF1"],
        dP_1330=d["dP"], dP_2130=a["dP"], dR_1330=d["dR"], dR_2130=a["dR"],
    ))

dst = os.path.join(OUT, "placement_verifier_day_evening.csv")
with open(dst, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(f"\n{'='*94}")
print("VERIFIER — GUNDUZ (13:30) vs AKSAM (21:30), ayni model, ayni esik")
print(f"{'='*94}")
print(f"{'birim':6s} | {'sifir-ortusmeli FP':>19s} | {'verifier reddi':>15s} | "
      f"{'sigara korunmasi':>17s} | {'dF1':>15s}")
print(f"{'':6s} | {'13:30':>8s} {'21:30':>10s} | {'13:30':>6s} {'21:30':>8s} | "
      f"{'13:30':>7s} {'21:30':>9s} | {'13:30':>6s} {'21:30':>8s}")
print("-" * 94)
for r in rows:
    if r["birim"] == "ALL":
        print("-" * 94)
    print(f"{r['birim']:6s} | {r['nFP0_1330']:8d} {r['nFP0_2130']:10d} | "
          f"{r['red_1330']:6.3f} {r['red_2130']:8.3f} | "
          f"{r['kor_1330']:7.3f} {r['kor_2130']:9.3f} | "
          f"{r['dF1_1330']:+6.3f} {r['dF1_2130']:+8.3f}")

sf, al = rows[1], rows[3]
print(f"\n⭐ SF (look-alike sahnede): FP havuzu {sf['nFP0_1330']} -> {sf['nFP0_2130']} "
      f"({sf['nFP0_2130']/sf['nFP0_1330']:.1f} kat), verifier reddi "
      f"{sf['red_1330']:.3f} -> {sf['red_2130']:.3f}, dF1 {sf['dF1_1330']:+.3f} -> {sf['dF1_2130']:+.3f}")
print(f"⚠️ MALIYET: sigara korunmasi (ALL) {al['kor_1330']:.3f} -> {al['kor_2130']:.3f}, "
      f"recall degisimi {al['dR_1330']:+.3f} -> {al['dR_2130']:+.3f}")
print(f"⚠️ SN (sahnede look-alike YOK): dF1 {rows[0]['dF1_1330']:+.3f} -> {rows[0]['dF1_2130']:+.3f} "
      f"— akşamda NEGATIF; bunlar arka-plan atesleri, verifier'in hedefi degil")
print(f"\n[OK] {dst}")
