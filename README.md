# A Preprocessing Framework for Video Machine Vision under Compression

A faithful, runnable implementation of the framework from *"A Preprocessing
Framework for Video Machine Vision under Compression"* (Zhao et al.) — **with the
paper's hand-crafted differentiable "virtual codec" replaced by
[CompressAII](https://github.com/InterDigitalInc/CompressAI)**.

A small neural **preprocessor** edits each video *before* compression so that a
frozen downstream vision model (action recognition / tracking) survives the
codec at far lower bitrate, while keeping the coded video close to the original.

```
 video ─▶ Preprocessor ─▶ Codec ─▶ reconstruction ─▶ frozen Analyzer ─▶ task
          (trained)       (frozen)                    (frozen)
             ▲                │                            │
             └──── gradients ─┴──── L = α(L_D + λ·L_R) + L_Acc ◀───────┘
```

Only the preprocessor is trained; gradients flow **through** the frozen codec
and analyzer back into it.

## Why CompressAI replaces the virtual codec

The paper hand-builds a differentiable "virtual codec" (intra/inter prediction →
transform → quantize → inverse, plus a **factorized-prior rate estimate à la
Ballé et al.**) purely to give the preprocessor a differentiable
rate + distortion signal during training.

CompressAI's `bmshj2018-factorized` model **is** that Ballé-et-al. factorized-prior
codec, already trained and battle-tested. So it provides the exact same
supervision, but better:

| | training | evaluation |
|---|---|---|
| **reconstruction** | differentiable `x_hat` (additive-noise quantization) | real range coder (`compress`/`decompress`) |
| **rate** | estimated bpp from entropy-model likelihoods | **actual** coded bpp |

→ gradients during training, honest bitrates when reporting results.

## The objective (paper Eq. 1)

```
L = α · (L_D + λ · L_R) + L_Acc          α = 10,  λ = 0.001
```
* `L_D` distortion — MSE(reconstruction, **source** video)
* `L_R` rate — estimated bits-per-pixel from the codec's entropy model
* `L_Acc` task loss — from the frozen analyzer: cross-entropy (action
  recognition) or the SiamFC balanced-logistic response loss (tracking)

## The preprocessor (paper Fig. 2)

Two-branch residual network:
* **temporal branch** — 3D convs spanning frames (inter-frame context),
* **spatial branch** — per-frame 2D convs (intra-frame context),
* **conditional attention** fuses the two streams,
* **residual** output `x + Δ` (last conv zero-initialised → starts as identity).

## Tasks

| task | analyzer | metric | status |
|------|----------|--------|--------|
| **Action recognition** (Kinetics-400) | frozen `r3d_18` (torchvision, Kinetics-400 weights) | top-1 accuracy | fully wired & runnable |
| **Object tracking** (GOT-10k) | frozen SiamFC (`resnet18` backbone); optional pytracking KYS/DiMP/ATOM/PrDiMP | success-plot AUC (+AO, SR) | fully wired & runnable |

Both tasks run end-to-end on ≤3 GB data. **Tracking** uses the GOT-10k **val**
split (ground-truth boxes on every frame): the preprocessor is trained with
SiamFC's differentiable balanced-logistic loss on the *compressed* clip, and
evaluation runs the tracker over each sequence to report the **real success AUC**
at every rate point. The default tracker is a self-contained SiamFC (frozen
ImageNet backbone, no extra install). For the paper's exact trackers
(KYS/DiMP/ATOM/PrDiMP), install [pytracking](https://github.com/visionml/pytracking)
(`scripts/install_pytracking.sh`) and set `task.tracker: pytracking:dimp:dimp50`
in the config — training still uses the differentiable SiamFC loss. See
`src/tasks/siamfc.py`, `src/tasks/tracking.py`, `src/tasks/pytracking_adapter.py`.

## Evaluation

`evaluate.py` traces **rate-accuracy curves** with the real coders and compares:

* `prep+compressai` — the proposed method,
* `compressai` — CompressAI alone (ablation: what the preprocessor adds),
* `h264` — bare H.264 (ffmpeg libx264), the anchor,
* `h265` — bare H.265 (ffmpeg libx265), the anchor,

then reports **BD-Rate** (Bjøntegaard) of `prep+compressai` against each anchor,
where the "quality" axis is the task metric (top-1 accuracy for action
recognition, success AUC for tracking) instead of PSNR. Negative BD-Rate =
fewer bits for the same accuracy = win.

Outputs: `results.json` (carries `task` + `metric`), `curves.csv`,
`rate_accuracy.png`.

## Layout

```
src/
  models/
    preprocessor.py     two-branch neural preprocessor (trained)
    codec.py            CompressAI wrapper  ← replaces the virtual codec
  tasks/
    base.py             frozen-analyzer interface + factory
    action_recognition.py   r3d_18 (Kinetics-400)
    siamfc.py           self-contained SiamFC tracker (differentiable loss + inference)
    tracking.py         SiamFC analyzer (GOT-10k)
    pytracking_adapter.py   optional KYS/DiMP/ATOM/PrDiMP via pytracking
  data/
    video_dataset.py    Kinetics clip reader ([B,C,T,H,W] in [0,1])
    prepare_3gb.py      balanced <=3 GB Kinetics index builder
    got10k.py           GOT-10k reader (clips + boxes, whole sequences)
    prepare_got10k.py   <=3 GB GOT-10k index builder (val split)
  codecs/standard.py    ffmpeg H.264 / H.265 anchors (real bpp)
  metrics/
    bd_rate.py          BD-Rate / BD-accuracy
    accuracy.py         top-k
    tracking_auc.py     success-plot AUC / AO / SR
  losses.py             α(L_D + λL_R) + L_Acc
  engine.py             train / eval loops (dispatched per task)
configs/                action_recognition.yaml, tracking.yaml
train.py  evaluate.py   CLI entry points
scripts/install_pytracking.sh   optional pytracking setup
kaggle/                 one-shot Kaggle driver + notebook
```

## Quickstart (local)

```bash
pip install -r requirements.txt          # needs ffmpeg (libx264+libx265) on PATH

# --- action recognition (Kinetics) ---
# 1) build a <=3GB balanced index from a Kinetics-style dir
python -m src.data.prepare_3gb --root /path/to/kinetics --out data/index/kinetics_3gb.json --cap-gb 3
# 2) train the preprocessor
python train.py --config configs/action_recognition.yaml
# 3) evaluate prep+CompressAI vs CompressAI / bare H.264 / H.265
python evaluate.py --config configs/action_recognition.yaml \
    --ckpt outputs/checkpoints/preprocessor.pth

# --- object tracking (GOT-10k val) ---
# 1) build a <=3GB GOT-10k index (whole sequences, train/val split)
python -m src.data.prepare_got10k --root /path/to/got10k/val --out data/index/got10k_3gb.json --cap-gb 3
# 2) train, then 3) evaluate real success AUC + BD-Rate
python train.py    --config configs/tracking.yaml
python evaluate.py --config configs/tracking.yaml --ckpt outputs/checkpoints/preprocessor.pth
```

## Quickstart (Kaggle)

See [`kaggle/README.md`](kaggle/README.md). In short:

```python
!git clone https://github.com/wagur1/preprocessing_final.git
%cd preprocessing_final
!pip install -q compressai
!python kaggle/run_kaggle.py --cap-gb 3 --epochs 3 --max-steps 300
```

Dataset: [`rohanmallick/kinetics-train-5per`](https://www.kaggle.com/datasets/rohanmallick/kinetics-train-5per)
(Kinetics-400, 5%). The 3 GB cap is enforced by *indexing* (round-robin across
classes), not copying.

## Notes & caveats

* The codec is applied **frame-wise** (the standard CompressAI setup); all
  temporal modelling lives in the preprocessor. This matches the paper, where
  temporal reasoning is the preprocessor's job. (The H.264/H.265 anchors do use
  inter-frame coding — the whole sequence is piped to ffmpeg as one video — so
  the comparison is honest against real temporal codecs.)
* Kinetics labels are mapped to the frozen analyzer's canonical 400-class
  ordering (`weights.meta['categories']`), so zero-shot top-1 is meaningful
  without any classifier fine-tuning.
* The SiamFC tracker's backbone is frozen and only ImageNet-pretrained (no
  tracker training), so *absolute* AUC is modest; because the same tracker is
  used across all pipelines, the *relative* BD-Rate is the meaningful number.
  Use pytracking for the paper's exact absolute numbers.
* CompressAI qualities are swept at eval and sampled per-batch at train, playing
  the role of the paper's random quantization factor.

## Reference

Zhao et al., *A Preprocessing Framework for Video Machine Vision under
Compression*. This repo replaces the paper's virtual codec with CompressAI;
it is an independent implementation, not the authors' code.
