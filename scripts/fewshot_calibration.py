# -*- coding: utf-8 -*-
r"""#3 FEW-SHOT PER-DEPLOYMENT KALİBRASYON (İZOLE / cache'ten, GPU'suz).

E26 bulgusu "dağıtım-başı kalibrasyon TAVSİYE edilir" diye BİTİYORDU — asılan uç.
Bu analiz onu kapatır: her held-out kişi için, o kişinin KENDİ K etiketli sigara crop'undan
(K ∈ {0,5,10,20}) per-kişi eşik türet, gerçekleşen korumanın %96'ya TIRMANIŞINI göster.
K=0 = havuzlanmış eşik (transfer, E26 = %95.8±4.7). K arttıkça per-kişi hedefe kilitlenir.

DÜRÜSTLÜK: eşik held-out kişinin K crop'undan türetilir ama o K crop DEĞERLENDİRMEDEN dışlanır
(kalan crop'larda ölçülür) → K crop hem eşik-set hem test olmaz. K rastgele seçilir, R tekrarla
ortalanır (seçim gürültüsü). Havuzlanmış eşik hâlâ diğer 16 kişiden (LOPO-temiz).

Çıktı: runs_v3_eval/fewshot_calib/  (CSV + figür)
"""
import os
os.environ["PYTHONUTF8"] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
OUT = ROOT + r"\runs_v3_eval\fewshot_calib"
CACHE = ROOT + r"\runs_v3_eval\_batchA_cache.npz"
RETENTION = 0.96
K_GRID = [0, 5, 10, 20]
R_REP = 200          # seçim-gürültüsü için tekrar
NAVY = "#1F3A5F"; BLUE = "#2B7BBA"; GREEN = "#1E8A3E"; RED = "#C0392B"
rng = np.random.default_rng(0)


def realized(cig_eval, fp_eval, thr):
    ret = float((cig_eval >= thr).mean()) if len(cig_eval) else np.nan
    rej = float((fp_eval < thr).mean()) if len(fp_eval) else np.nan
    return ret, rej


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(CACHE, allow_pickle=True)
    y = d["y"]; g = d["g"]; dconf = d["dconf"]; clip = d["clip_prob"]
    v = ~np.isnan(clip)
    y, g, dconf, clip = y[v], g[v], dconf[v], clip[v]
    pcig = 1.0 - clip
    fused = dconf * pcig
    persons = sorted(set(g))

    rows = ["score,K,person,mean_realized_retention,mean_realized_rejection,n_cig_eval,n_fp_eval"]
    summary = {}
    for tag, score in [("VERIFIER-tek", pcig), ("FUSED", fused)]:
        summary[tag] = {}
        for K in K_GRID:
            rets, rejs = [], []
            for p in persons:
                te = g == p
                cig_p = score[te & (y == 0)]
                fp_p = score[te & (y == 1)]
                if K == 0:
                    # havuzlanmış eşik = DİĞER 16 kişiden (E26 ile aynı, LOPO-temiz)
                    cig_tr = score[(g != p) & (y == 0)]
                    thr = float(np.quantile(cig_tr, 1 - RETENTION))
                    ret, rej = realized(cig_p, fp_p, thr)
                    rets.append(ret); rejs.append(rej)
                    rows.append(f"{tag},{K},{p},{ret:.4f},{rej:.4f},{len(cig_p)},{len(fp_p)}")
                else:
                    if len(cig_p) <= K + 5:      # yeterli eval crop kalsın
                        continue
                    pr, pj = [], []
                    for _ in range(R_REP):
                        idx = rng.permutation(len(cig_p))
                        cal_i, ev_i = idx[:K], idx[K:]
                        thr = float(np.quantile(cig_p[cal_i], 1 - RETENTION))  # per-kişi K crop'tan
                        ret, rej = realized(cig_p[ev_i], fp_p, thr)            # kalan cig + tüm fp
                        pr.append(ret); pj.append(rej)
                    ret, rej = float(np.mean(pr)), float(np.nanmean(pj))
                    rets.append(ret); rejs.append(rej)
                    rows.append(f"{tag},{K},{p},{ret:.4f},{rej:.4f},{len(cig_p)-K},{len(fp_p)}")
            rr = np.array(rets); jj = np.array(rejs)
            summary[tag][K] = (np.nanmean(rr), np.nanstd(rr), np.nanmean(jj), np.nanstd(jj))
            print(f"[{tag}] K={K:2d} -> koruma %{100*np.nanmean(rr):.1f} ± {100*np.nanstd(rr):.1f} "
                  f"| red %{100*np.nanmean(jj):.1f} ± {100*np.nanstd(jj):.1f}")
    open(os.path.join(OUT, "fewshot_calibration.csv"), "w", encoding="utf-8").write("\n".join(rows) + "\n")

    # figür: K arttıkça koruma %96'ya kilitlenir + saçılım daralır
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, tag in zip((a1, a2), ("VERIFIER-tek", "FUSED")):
        ks = K_GRID
        rm = [summary[tag][k][0] * 100 for k in ks]
        rs = [summary[tag][k][1] * 100 for k in ks]
        ax.axhline(96, color=GREEN, ls="--", lw=1.4, label="hedef %96")
        ax.errorbar(ks, rm, yerr=rs, fmt="o-", color=BLUE, lw=2, ms=7, capsize=4,
                    label="gerçekleşen koruma (ort ± sd, 17 kişi)")
        for k, m, s in zip(ks, rm, rs):
            ax.annotate(f"%{m:.1f}\n±{s:.1f}", (k, m), textcoords="offset points",
                        xytext=(6, -18 if k == 0 else 8), fontsize=8, color=NAVY)
        ax.set_title(f"{tag}: few-shot per-kişi kalibrasyon", fontsize=10.5, weight="bold", color=NAVY)
        ax.set_xlabel("K = kişinin kendi etiketli sigara crop sayısı (K=0: havuzlanmış transfer)")
        ax.set_ylabel("gerçekleşen sigara-koruması (%)")
        ax.set_xticks(ks); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Few-shot dağıtım-başı kalibrasyon — DÜRÜST NEGATİF (E26'yı güçlendirir)\n"
                 "Naif few-shot yeniden-kalibrasyon HER K≤20'de havuzlanmıştan (K=0, %95.8) KÖTÜ: "
                 "%96-eşiği ≤20 örnekten güvenilir kestirilemiyor ⇒ havuzlanmış transfer en sağlam varsayılan",
                 fontsize=10.5, weight="bold", color=NAVY)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(OUT, "fig_fewshot_calibration.png"), dpi=160); plt.close()
    print(f"\n=== çıktı -> {OUT} ===")


if __name__ == "__main__":
    main()
