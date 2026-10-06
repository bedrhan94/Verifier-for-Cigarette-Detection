# -*- coding: utf-8 -*-
"""B5: end-to-end throughput (detector + verifier). Measures CLIP-ViT-B crop inference speed,
combines with measured detector wall-clock (fig15b: full-frame 12.3ms, tiled2x2 46.5ms).
Reports pipeline FPS vs #detections/frame. fig25 + end_to_end_fps.csv"""
import os
os.environ["PYTHONUTF8"] = "1"
import time
import numpy as np
import torch
import timm
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Elsevier: TrueType (Type-3 reddedilir)
import matplotlib.pyplot as plt

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
EV = ROOT + r"\runs_v3_eval"
FIG = EV + r"\figures"
P1 = ROOT + r"\PAPERS\PAPER1_detector_tiling_temporal"
P2 = ROOT + r"\PAPERS\PAPER2_verifier"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

DET_FULL_MS = 12.3    # measured fig15b (full-frame, imgsz960)
DET_TILED_MS = 46.5   # measured fig15b (tiled 2x2)


def time_verifier(batch, reps=30):
    m = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0).eval().to(DEV)
    x = torch.randn(batch, 3, 224, 224, device=DEV)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); m(x)
            if DEV == "cuda":
                torch.cuda.synchronize()
            t.append(time.perf_counter() - t0)
    del m
    torch.cuda.empty_cache()
    return float(np.median(t)) * 1e3  # ms for the whole batch


def main():
    print("[B5] measuring CLIP-ViT-B verifier inference...", flush=True)
    ver_single = time_verifier(1)
    ver_batch8 = time_verifier(8)
    per_crop_batched = ver_batch8 / 8
    print(f"[B5] verifier: single-crop={ver_single:.2f}ms | batch8={ver_batch8:.2f}ms ({per_crop_batched:.2f}ms/crop batched)", flush=True)

    Ns = [0, 1, 2, 3, 5]
    rows = ["n_detections,det_full_fps,det_full+ver_seq_fps,det_full+ver_batch_fps,det_tiled+ver_batch_fps"]
    fps_seq, fps_batch, fps_tiled = [], [], []
    for N in Ns:
        seq = 1000.0 / (DET_FULL_MS + N * ver_single)
        bat = 1000.0 / (DET_FULL_MS + (0 if N == 0 else max(ver_single, N * per_crop_batched)))
        til = 1000.0 / (DET_TILED_MS + (0 if N == 0 else max(ver_single, N * per_crop_batched)))
        fps_seq.append(seq); fps_batch.append(bat); fps_tiled.append(til)
        rows.append(f"{N},{1000.0/DET_FULL_MS:.1f},{seq:.1f},{bat:.1f},{til:.1f}")
        print(f"[B5] N={N}: det-only={1000.0/DET_FULL_MS:.0f} | +ver(seq)={seq:.0f} | +ver(batch)={bat:.0f} | tiled+ver={til:.0f} fps", flush=True)
    for p in (EV + r"\end_to_end_fps.csv", P1 + r"\tables\end_to_end_fps.csv", P2 + r"\tables\end_to_end_fps.csv"):
        open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(Ns, fps_seq, "-o", color="#888888", label="full-frame + verifier (sequential)")
    ax.plot(Ns, fps_batch, "-s", color="#2b7bba", label="full-frame + verifier (batched)")
    ax.plot(Ns, fps_tiled, "-^", color="#c0392b", label="tiled 2x2 + verifier (batched)")
    ax.axhline(25, color="green", ls="--", alpha=.6, label="25 fps (real-time)")
    ax.axhline(30, color="green", ls=":", alpha=.5, label="30 fps (typical camera rate)")
    ax.set_xlabel("detections per frame (verifier calls)"); ax.set_ylabel("end-to-end throughput (FPS)")
    # Not: baslikta "B5" gibi IC DENEY KODU kullanilmaz; makalede gorunur ve
    # calisma-notu izlenimi birakir (kullanici kurali).
    ax.set_title(f"Two-stage pipeline throughput (RTX 5060 Ti)\nverifier {ver_single:.1f} ms/crop sequential, {per_crop_batched:.1f} ms/crop batched")
    ax.set_xticks(Ns); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    for d in (FIG, P1 + r"\figures", P2 + r"\figures"):
        plt.savefig(d + r"\fig25_end_to_end_fps.png", dpi=300)
        plt.savefig(d + r"\fig25_end_to_end_fps.pdf", dpi=300)
    plt.close()
    open(EV + r"\_b5_verifier_ms.txt", "w").write(f"ver_single_ms={ver_single:.3f}\nver_percrop_batched_ms={per_crop_batched:.3f}\n")
    print("=== B5 DONE: fig25 + end_to_end_fps.csv ===", flush=True)


if __name__ == "__main__":
    main()
