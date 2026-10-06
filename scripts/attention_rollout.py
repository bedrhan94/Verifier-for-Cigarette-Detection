# -*- coding: utf-8 -*-
"""B4: ViT attention-rollout (Abnar & Zuidema 2020) on CLIP-ViT-B verifier -> replace diffuse Grad-CAM.
Hook attn_drop input (=softmax attn) with fused_attn disabled; rollout CLS->patch; overlay on example crops.
Honest: if also diffuse (register-token issue), report as negative + keep t-SNE/qualitative. fig28."""
import os
os.environ["PYTHONUTF8"] = "1"
import glob
import numpy as np
import cv2
import torch
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\bedrhan94\Desktop\yuztanıma 22.09\cigarette smokers.v6-finalmo2.yolov11"
SRC = ROOT + r"\crops_sweep\W96"
EV = ROOT + r"\runs_v3_eval"
FIG = EV + r"\figures"
P2 = ROOT + r"\PAPERS\PAPER2_verifier"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def imread_rgb(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def pick(cls, n=3):
    out = []
    for person in sorted(os.listdir(SRC)):
        g = glob.glob(os.path.join(SRC, person, cls, "*.jpg"))
        if g:
            out.append(g[len(g)//2])
        if len(out) >= n:
            break
    return out


def main():
    m = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0).eval().to(DEV)
    cfg = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(DEV); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(DEV)
    insz = cfg["input_size"][-1]; P = 16; grid = insz // P

    attns = []
    def hook(mod, inp, out):
        attns.append(inp[0].detach())
    for blk in m.blocks:
        blk.attn.fused_attn = False
        blk.attn.attn_drop.register_forward_hook(hook)

    def rollout(img_rgb):
        attns.clear()
        x = cv2.resize(img_rgb, (insz, insz)).astype(np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEV)
        with torch.no_grad():
            m((t - mean) / std)
        N = attns[0].shape[-1]
        result = torch.eye(N, device=DEV)
        for A in attns:
            A = A.mean(1)[0]                       # avg heads -> [N,N]
            A = A + torch.eye(N, device=DEV)
            A = A / A.sum(-1, keepdim=True)
            result = A @ result
        mask = result[0, 1:].reshape(grid, grid).cpu().numpy()   # CLS -> patches
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return cv2.resize(mask, (insz, insz))

    examples = [("cigarette", p) for p in pick("cig", 3)] + [("look-alike", p) for p in pick("fp", 3)]
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.4))
    stds_report = []
    for j, (lab, p) in enumerate(examples):
        im = imread_rgb(p)
        if im is None:
            continue
        disp = cv2.resize(im, (insz, insz))
        heat = rollout(im); stds_report.append(float(np.std(heat)))
        ax[0, j].imshow(disp); ax[0, j].set_title(lab, fontsize=9); ax[0, j].axis("off")
        ax[1, j].imshow(disp); ax[1, j].imshow(heat, cmap="jet", alpha=0.5); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("crop", fontsize=9); ax[1, 0].set_ylabel("attn-rollout", fontsize=9)
    conc = np.mean(stds_report)
    plt.suptitle(f"B4: CLIP-ViT-B attention rollout (Abnar 2020). heatmap std={conc:.3f} "
                 f"({'focused' if conc > 0.18 else 'diffuse -> honest negative, use t-SNE/qualitative'})")
    plt.tight_layout()
    for d in (FIG, P2 + r"\figures"):
        plt.savefig(d + r"\fig28_attention_rollout.png", dpi=140)
    plt.close()
    open(EV + r"\attention_rollout.csv", "w", encoding="utf-8").write(
        "metric,value\nmean_heatmap_std," + f"{conc:.4f}\n" + "verdict," + ("focused" if conc > 0.18 else "diffuse") + "\n")
    print(f"[B4] mean heatmap std={conc:.3f} -> {'FOCUSED' if conc > 0.18 else 'DIFFUSE (honest negative)'}", flush=True)
    print("=== B4 DONE: fig28 + attention_rollout.csv ===", flush=True)


if __name__ == "__main__":
    main()
