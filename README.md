# Appearance Verification for Persistent False Positives

Analysis code for **"Appearance Verification for Persistent False Positives: A Second-Stage
Verifier for Cigarette Detection in Surveillance"** (under review).

Temporal stabilization removes most frame-level false positives in cigarette detection, but a
residual stream of **persistent, high-confidence** false alarms survives it: pens and markers held
near the mouth. These are temporally stable (the object really is there) and carry high detector
confidence, so neither temporal filtering nor raising the confidence threshold can remove them. The
paper asks whether a second, appearance-only verification stage is warranted, and tests that
question rather than assuming it.

| Result | Number |
|---|---|
| Verifier vs raised confidence, at 95% cigarette retention | rejects **51.4%** of look-alikes vs **17.3%** |
| Per-person AUC, frozen CLIP-ViT-B on 96 px crops, 17 held-out subjects | **0.90 ± 0.05** |
| Attribution: Chefer vs attention rollout (random floor 0.10) | **0.691** vs 0.358 |
| Two-stage pipeline, batched verification | about **33 FPS** untiled |

The second claim is a design conclusion rather than a performance one: **where** the stage is
inserted matters as much as what it computes. Applied per frame it fragments genuine events;
applied as a gate on whole events it delivers the same benefit at no measured cost.

---

## Method in one paragraph

A frozen single-class YOLO11m detector proposes boxes. Each surviving box is cropped at 96 px and
embedded with a **frozen CLIP-ViT-B** encoder; a small MLP head classifies cigarette vs look-alike.
The verifier score is fused with detector confidence as `fused = det_conf × (1 − P(fp))`, and the
operating threshold is set by a fixed retention rule rather than tuned on test footage. Evaluation
is leave-one-person-out over 17 subjects, on per-person same-day crops, after an initial pilot
scoring 0.96 was traced to scene leakage and discarded.

The upstream detector is the SHA-locked checkpoint released with the companion detector study:

```
SHA-256  17BF89F3BBDCC2B85315627485B1EE56CF4B67C826B114EE6EB3A36C583E3D14
```

It is available at
[Toward-Generalizable-Cigarette-Detection](https://github.com/bedrhan94/Toward-Generalizable-Cigarette-Detection)
and is not duplicated here. The verifier itself has no released weight file: the encoder is a
public pretrained CLIP checkpoint fetched by `timm`, and the MLP head is trained by the scripts
below.

---

## Setup

Python 3.10+ is required. PyTorch is deliberately **not pinned**; install the build matching your
CUDA first.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Scripts and the sections they produce

| Script | Section |
|---|---|
| `scripts/verifier_core_analysis.py` | Verifier vs raised detector confidence at matched retention, bootstrap AUC CIs, PR under imbalance, t-SNE, temperature calibration (§6.3–6.5) |
| `scripts/backbone_comparison.py` | CLIP-ViT-B vs DINOv2-ViT-S vs ImageNet-ViT-S vs ResNet50 under one protocol (§6.2) |
| `scripts/classic_feature_baseline.py` | Colour histogram, HOG and raw-pixel baselines (§6.3) |
| `scripts/attention_rollout.py` | Gradient-free attention-rollout attribution baseline (§6.7) |
| `scripts/chefer_attribution.py` | Chefer transformer attribution and localization concentration (§6.7) |
| `scripts/end_to_end_throughput.py` | Verifier latency and two-stage throughput, batched vs sequential (§6.8) |
| `scripts/fewshot_calibration.py` | Few-shot per-deployment calibration from each subject's own crops (§6.4) |
| `scripts/event_gate_designs.py` | Five pre-registered per-frame vs per-event designs (§6.10) |
| `scripts/placement_verifier_eval.py` | Verifier on the placement recordings, leave-three-out with a matched out-of-fold threshold (§6.9) |
| `scripts/placement_day_vs_evening.py` | Daylight vs artificial-light comparison of the same grid (§6.9) |
| `scripts/placement_verifier_vs_detconf.py` | Equal-retention control against a raised confidence threshold (§6.9) |
| `scripts/placement_scores.py` | Scores placement and held-out clips for the event-level analysis (§6.9–6.10) |
| `scripts/unseen_subject_verifier.py` | A subject absent from every training pool, both sessions (§6.9) |
| `scripts/blur_faces_in_figs.py` | Privacy utility: blurs faces in any rendered figure |

Any script that renders crops writes identifiable frames. **Run the blurring utility afterwards:**

```powershell
python scripts/blur_faces_in_figs.py
```

> **Paths and identifiers.** These scripts were written for one machine: several carry an absolute
> `ROOT` constant at the top, and subject directory names have been replaced with neutral
> identifiers (`kisi_01`, `kisi_02`, …) before publication. They are published as a record of how
> the reported numbers were computed; they will not run as-is without the in-house data, which is
> not released.

---

## Data

**Not included and not releasable.** Every crop, clip and frame used here comes from human-subject
recordings collected under informed consent and an ethics approval that does not permit
redistribution, and is governed by the Turkish Personal Data Protection Law (KVKK). No image, video
or frame of any participant appears in this repository, and subject identifiers have been
anonymized in the code.

De-identified derived data (labels, cached out-of-fold predictions and the per-frame detection
records used in the evaluations) is available from the corresponding author on reasonable request.

The public cigarette-detection benchmark behind the upstream detector is linked from the companion
repository.

---

## License

Released under **AGPL-3.0**, matching the license of the Ultralytics YOLO11 framework the upstream
detector is fine-tuned from.

---

## Citation

A citation entry will be added once the paper is published. Until then please cite it as under
review.
