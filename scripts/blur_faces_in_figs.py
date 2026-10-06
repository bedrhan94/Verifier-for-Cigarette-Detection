# -*- coding: utf-8 -*-
r"""Figürlerdeki YÜZLERİ blur'la (KVKK + makale etik beyanı: yayımlanan niteliksel figürlerde yüz görünmez).
insightface det (CPU, açı-toleranslı) + düşük eşik -> tespit edilen her yüzü GENİŞLET + ağır blur + pixelate.
Orijinaller _faces_backup/'a yedeklenir. SONRA insan gözüyle KONTROL et (danışman)."""
import os, sys, shutil
os.environ["PYTHONUTF8"] = "1"
import cv2, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGS = [
    "PAPERS/PAPER2_verifier/figures/fig7_qualitative.png",
    "PAPERS/PAPER2_verifier/figures/fig28_attention_rollout.png",
    "PAPERS/PAPER2_verifier/figures/fig37_chefer_xai.png",
    "PAPERS/PAPER2_verifier/figures/fig37s_chefer_slide.png",
    "PAPERS/PAPER1_detector_tiling_temporal/figures/fig38_placement_setup.png",
]


def blur_box(img, x1, y1, x2, y2):
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    # pixelate + gauss (geri döndürülemez)
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // 12), max(1, h // 12)), interpolation=cv2.INTER_LINEAR)
    roi2 = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(9, (min(w, h) // 3) | 1)
    roi2 = cv2.GaussianBlur(roi2, (k, k), 0)
    img[y1:y2, x1:x2] = roi2


def main():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.30)  # düşük eşik -> daha çok yakala
    bak = os.path.join(ROOT, "PAPERS", "_faces_backup")
    os.makedirs(bak, exist_ok=True)
    for rel in FIGS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print("[yok]", rel); continue
        shutil.copy2(p, os.path.join(bak, os.path.basename(p)))
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        faces = app.get(img)
        n = 0
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            mx = int(0.25 * (x2 - x1)); my = int(0.30 * (y2 - y1))   # genişlet (saç/çene dahil)
            blur_box(img, x1 - mx, y1 - my, x2 + mx, y2 + my)
            n += 1
        ok, buf = cv2.imencode(".png", img)
        if ok:
            buf.tofile(p)
        print(f"{os.path.basename(p)}: {n} yüz blur'landı")


if __name__ == "__main__":
    main()
