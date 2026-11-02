# Second-Stage Appearance Verification for Cigarette Detection

Analysis code for a two-stage pipeline that removes a specific class of false
alarm in cigarette detection from fixed cameras: pens, markers and straws held
steadily at the mouth. These detections survive temporal filtering and a higher
confidence threshold, because the object really is there and the detector is
confident about it. A second, appearance-only stage is added behind the detector
to decide whether the object is a cigarette.

This repository holds the analysis scripts only. It contains no data, no model
weights and no human-subject material.

## Architecture

**Stage 1 — detector.** A frozen single-class YOLO11m detector proposes boxes.
It is never retrained or re-tuned by this code. The checkpoint is identified by

```
SHA-256  17BF89F3BBDCC2B85315627485B1EE56CF4B67C826B114EE6EB3A36C583E3D14
```

and lives in the companion detector repository,
[Toward-Generalizable-Cigarette-Detection](https://github.com/bedrhan94/Toward-Generalizable-Cigarette-Detection).
It is not duplicated here.

**Temporal stage.** An asymmetric hysteresis rule turns frame-level detections
into events: an event opens after a two-second confirmation window, closes after
eight consecutive negative frames, and is discarded below a minimum duration of
0.8 s.

**Stage 2 — verifier.** Each candidate box is cropped at 96 px in the source
image, embedded by a frozen CLIP-ViT-B/16 encoder, and classified by a
single-hidden-layer perceptron of 256 units. Detector confidence and verifier
output are fused as `det_conf × (1 − p_fp)`. The operating threshold is a fixed
retention quantile of out-of-fold scores and is never tuned on evaluation
footage. Evaluation is leave-one-person-out over 17 subjects.

The verifier has no released weight file: the encoder is a public pretrained
checkpoint fetched by `timm`, and the head is trained by the scripts.

## Verified results

| Quantity | Value | Condition |
|---|---|---|
| Hard-negative rejection at 95 % cigarette retention | 51.5 % | confidence-matched pool, detector confidence ≥ 0.50 |
| Same, raising the detector threshold instead | 17.3 % | same pool, same retention |
| Discrimination, per-subject mean | 0.900 verifier, 0.860 detector confidence | 17 held-out subjects, leave-one-person-out |
| Event gate on twelve cigarette-free recordings | 109 → 39 confirmed false events, 1.94 → about 0.69 per minute | causal gate at confirmation |
| Event gate on twelve placement recordings | 13 of 13 reference periods recalled | event counts there are pipeline output, not verified per object |
| End-to-end throughput, evaluated configuration | 15.1 – 15.6 FPS | 960 px with unconditional 2×2 tiling, measured on recorded video |
| Camera input rate | 30 FPS | the evaluated configuration is therefore not real time |
| Untiled, sequential / batched | 23.7 / 32.8 FPS | synthetic estimates, see below |
| Attribution concentration, Chefer / rollout | 0.691 / 0.358 | DINOv2-ViT-S/14 encoder, 1 200 crops |

Three qualifications that matter for reading these numbers.

**Synthetic versus measured throughput.** The 23.7 and 32.8 FPS figures carry a
measured encoder cost over an untiled detector pass and exclude video decoding,
crop extraction and host-to-device transfer. They are not interchangeable with
the 15.1 – 15.6 FPS measured end to end, and no accuracy result was produced
under them.

**The attribution reference.** A concentration of 0.10 is the value of a
perfectly uniform map, not what a random map scores. Because the hottest decile
is selected from the map itself, a randomly varying map scores higher: the
measured reference range is about 0.193 – 0.332. Chefer's 0.691 clears that whole
range; rollout's 0.358 sits just above its upper end. These maps are computed on
a DINOv2-ViT-S/14 encoder, so they are not a direct quantitative explainability
measurement of the CLIP verifier.

**The event gate is a trade, not a free gain.** Removing false events at the
event level costs more time spent inside a false alarm than filtering per frame
does, and the per-frame filter in turn fragments genuine episodes. Neither
placement dominates.

## Setup

Python 3.10 or newer. PyTorch is deliberately not pinned; install the build that
matches the local CUDA first.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

A CUDA device is expected for the feature-extraction and throughput scripts; the
rest run on CPU, more slowly.

## Running the analyses

Every script reads its inputs from one root directory, given by the environment
variable `VERIFIER_DATA_ROOT`. That root is the private data tree and is not part
of this repository.

```powershell
$env:VERIFIER_DATA_ROOT = "D:\path\to\private\data\root"
python scripts\verifier_core_analysis.py
```

Expected layout under that root:

| Path | Contents |
|---|---|
| `crops_sweep\W96\<subject>\<class>\*.jpg` | 96 px crops, one directory per subject, split into cigarette and look-alike |
| `crops_verifier\manifest_verifier.csv` | per-crop record: subject, class, source frame, detection index, detector confidence |
| `runs_v3_eval\` | per-recording detection records, cached scores and the evaluation outputs the later scripts read |
| `runs_v3_eval\_batchA_cache.npz` | cached features and out-of-fold predictions, written by the core script and reused by the others |

Outputs are written beside the inputs under `runs_v3_eval\`: figures as PNG and
PDF, tables as CSV, run metadata as JSON, and cached arrays as NPZ. Nothing is
written back into this repository.

Any script that renders crops produces identifiable frames. Run the blurring
utility afterwards:

```powershell
python scripts\blur_faces_in_figs.py
```

## Scripts

Grouped by what they evaluate.

| Group | Script | Produces |
|---|---|---|
| Crop-level evaluation | `verifier_core_analysis.py` | leave-one-person-out discrimination, bootstrap intervals, DeLong and per-subject tests, precision-recall under imbalance, t-SNE, temperature calibration |
| | `placement_verifier_vs_detconf.py` | the verifier against a raised detector threshold on the confidence-matched pool |
| | `backbone_comparison.py` | CLIP-ViT-B against DINOv2-ViT-S, ImageNet-ViT-S and ResNet50 under one protocol |
| | `classic_feature_baseline.py` | colour-histogram, HOG and raw-pixel baselines |
| | `fewshot_calibration.py` | few-shot per-deployment calibration from each subject's own crops |
| Footage and event level | `placement_scores.py` | scores the placement and held-out clips for the event-level analyses |
| | `placement_verifier_eval.py` | the camera-placement grid with a matched out-of-fold threshold |
| | `placement_day_vs_evening.py` | the same grid under daylight and artificial light |
| | `unseen_subject_verifier.py` | a participant absent from the training pool, both sessions |
| | `event_gate_designs.py` | five pre-registered per-frame and per-event designs and the causal-gate timing variants |
| Cost and interpretability | `end_to_end_throughput.py` | verifier latency and two-stage throughput, batched against sequential |
| | `attention_rollout.py` | gradient-free attention-rollout attribution |
| | `chefer_attribution.py` | Chefer transformer attribution and concentration |
| Privacy utility | `blur_faces_in_figs.py` | blurs faces in any rendered figure |

## Data: what is public and what is not

**Not in this repository and not releasable.** Every crop, clip and frame behind
these analyses comes from human-subject recordings collected under written
informed consent that does not permit redistribution, and the material is
governed by the Turkish Personal Data Protection Law (KVKK). No image, video or
frame of any participant appears here. Subject directory names in the code are
neutral identifiers.

**Available on request.** De-identified derived data, including cached
out-of-fold predictions, per-detection scores and per-recording event tables, is
available from the corresponding author on reasonable request.

**Public sources.** The public cigarette-detection datasets behind the upstream
detector are linked from the companion detector repository, under their own
licences and redistribution terms. No public dataset is mirrored here.

## Limits of reproducibility

These scripts are published so that the computation behind each reported number
can be read, not so that it can be re-executed from this repository alone.

- Nothing runs without the private data root described above. No substitute,
  synthetic or placeholder data is provided, and none should be invented to make
  a script run.
- Feature extraction depends on the pretrained encoder weights served by the
  model hub. Those weights can change, which shifts absolute values by a few
  points while preserving the ordering of methods. The cached NPZ is the stable
  reference.
- The throughput measurements depend on the specific GPU and driver stack and
  will not reproduce exactly on other hardware.
- Several scripts were written for a single machine and read a fixed directory
  layout under the data root.

## Licence

AGPL-3.0, matching the licence of the Ultralytics YOLO11 framework the upstream
detector is fine-tuned from. See `LICENSE`.
