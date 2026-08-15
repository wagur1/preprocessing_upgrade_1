# Preprocessing for Video Machine Vision — Upgrade 1 (train on the real x265 codec)

Fork của [`preprocessing_final`](https://github.com/wagur1/preprocessing_final) (bản dựng lại
Zhao et al., *"A Preprocessing Framework for Video Machine Vision under Compression"*,
với "virtual codec" của paper thay bằng CompressAI `bmshj2018-factorized`).

**Điểm mới của bản này:** preprocessor được **train với forward đi qua codec x265 THẬT**
(ffmpeg libx265), còn **gradient chảy qua proxy CompressAI khả vi** — vá thẳng điểm yếu
W3 (lệch proxy↔codec thật) của baseline. Kỹ thuật: **Straight-Through Estimator (STE) /
detached-difference**. Pipeline giữ nguyên, chỉ đổi vòng train.

```
 video ─▶ Preprocessor ─▶ Codec ─▶ reconstruction ─▶ frozen Analyzer ─▶ task
          (trained)       │                           (frozen)           │
             ▲            │                                              │
             │       train:  x265 THẬT (forward)  +  CompressAI proxy (gradient)
             │       eval :  range coder thật + anchor H.264/H.265
             └──── gradients ──── L = α(L_D + λ·L_R) + L_Acc ◀───────────┘
```

Chỉ preprocessor được train; codec và analyzer đông cứng.

## Cải tiến chính — STE: forward = x265 thật, backward = proxy

x265 không khả vi nên không thể backprop trực tiếp. Trick detached-difference:
giá trị forward lấy từ codec thật, đạo hàm mượn từ proxy.

```python
qp   = random.choice(qp_list)          # SAMPLE QP TRƯỚC pre()
q    = qp_to_quality[qp]               # map QP → CompressAI quality index
x_pre = pre(clips)

x_hat_prx, bpp_prx = codec(x_pre, q)                    # proxy khả vi
if real_step:                                           # mỗi real_codec_every_n_steps
    x_hat_real, bpp_real = real_codec(x_pre, qp)        # x265 thật, no_grad
    x_hat = x_hat_prx + (x_hat_real - x_hat_prx).detach()   # forward=thật, grad qua proxy
    bpp   = bpp_prx   + (bpp_real_t - bpp_prx ).detach()
else:
    x_hat, bpp = x_hat_prx, bpp_prx                     # fallback proxy-only
```

Bất biến: `x_hat == x_hat_real` (forward), `∂x_hat/∂x_pre = ∂x_hat_prx/∂x_pre` (backward).
Helper `_straight_through` ở `src/engine.py`; áp cho **cả hai** vòng `_train_classification`
và `_train_tracking`. Không có ffmpeg → tự cảnh báo và fallback proxy-only, không crash.

## Bảng cải thiện so với baseline (`preprocessing_final`)

| # | Phần | Baseline `final1` | Upgrade1 | Vá điểm yếu | Trạng thái |
|---|------|-------------------|----------|-------------|-----------|
| 1 | **Forward codec lúc train** | Chỉ proxy CompressAI khả vi | **x265 thật** qua STE (`_training_codec_forward`) | **W3** proxy↔real mismatch | ✅ đã code (Phase 1) |
| 2 | **Gradient** | Qua proxy | Qua proxy, giữ nguyên (STE) | — | ✅ đã code |
| 3 | **Thứ tự sample QP** | `pre()` chạy trước, `q` chọn sau → mù rate | **Sample QP TRƯỚC `pre()`**, map `qp→q` | tiền đề **W2** (FiLM là PR sau) | ✅ đã code |
| 4 | **QP↔quality mapping** | Không có | `qp_to_quality` đơn điệu, validate ở `_training_codec_setup` | — | ✅ đã code |
| 5 | **Lỗi cú pháp `_train_classification`** | Thiếu header vòng lặp → `IndentationError` | Khôi phục `for epoch / pbar / for clips,labels` | bug | ✅ đã sửa |
| 6 | **Guard ffmpeg + amortize** | — | `ffmpeg_available()` fallback; `real_codec_every_n_steps` | chi phí | ✅ đã code |
| 7 | **Log chênh bitrate** | — | log `bpp_proxy` vs `bpp_real` + `gap%` mỗi 200 step | tinh chỉnh map | ✅ đã code |
| 8 | **Proxy calibration (trainable)** | — | `L_rate_cal`, `opt_proxy` two-timescale | **W3** đầy đủ (A3) | ⏳ Phase 2, CHƯA code |
| 9 | **FiLM QP-conditioning vào preprocessor** | — | `pre(clips, qp)` | **W2** | ⏳ ngoài scope PR này |

Các phần **không đổi** so với baseline (data loaders, metrics BD-Rate/AUC/top-1, tasks,
preprocessor, eval end-to-end) được giữ nguyên — bản này chỉ *extend* vòng train.
Chi tiết kế hoạch: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) (cho Codex),
[`PAPER_PLAN.md`](PAPER_PLAN.md) (roadmap Q1, W1–W9, Tier A/B/C).

## Hàm loss (giữ nguyên paper Eq. 1)

```
L = α · (L_D + λ · L_R) + L_Acc          α = 10,  λ = 0.001
```
`L_D` = MSE(recon, **source**); `L_R` = bpp; `L_Acc` = cross-entropy (AR) hoặc
SiamFC balanced-logistic (tracking). Bản STE **không đổi** `losses.py` — chỉ đổi
`x_hat/bpp` đầu vào (giờ mang giá trị x265 thật).

## Config mới (dưới `train:` trong cả hai file `configs/*.yaml`)

```yaml
train:
  real_codec: true              # bật STE forward thật
  real_codec_name: h265         # h264 | h265
  real_codec_preset: medium
  qp_list: [22, 27, 32, 37, 42]
  qp_to_quality: {22: 8, 27: 5, 32: 3, 37: 2, 42: 1}   # đơn điệu: QP↑ ↔ quality↓
  real_codec_every_n_steps: 1   # >1 để amortize chi phí ffmpeg
  calibration: none             # none (Phase 1) | rate (Phase 2, CHƯA implement → raise nếu set)
```
`codec.qualities` đã mở rộng `[1,2,3,5,8]` để phủ mọi giá trị trong `qp_to_quality`.

## Tasks & Evaluation (giữ nguyên baseline)

| task | analyzer | metric |
|------|----------|--------|
| Action recognition (Kinetics-400) | frozen `r3d_18` | top-1 |
| Object tracking (GOT-10k val) | frozen SiamFC; optional pytracking DiMP/ATOM/KYS/PrDiMP | success AUC (+AO, SR) |

`evaluate.py` vẽ **rate-accuracy curve** với range coder thật, so các pipeline —
`prep+compressai` (đề xuất), `compressai` (ablation), `h264`, `h265` (anchor), và
**`prep+h264` / `prep+h265`** (preprocessor + codec thật) — rồi báo **BD-Rate** (âm =
tiết kiệm bit ở cùng accuracy). Hai nhóm số:
- `bd_vs_anchor`: `prep+compressai` vs từng anchor (tham khảo, khác codec).
- `bd_prep_gain`: **cùng codec, chỉ khác có/không preprocessor** — `prep+h265 vs h265`,
  `prep+h264 vs h264`, `prep+compressai vs compressai`. **Đây mới là số đúng luận điểm**
  (khớp codec bạn train-through qua STE).
Output: `results.json`, `curves.csv`, `rate_accuracy.png`.

## Chạy trên Kaggle (cụ thể)

Cần **GPU** + **Internet ON** (để `pip install` và tải trọng số). Kaggle image có sẵn
`ffmpeg` (libx264/libx265) → nhánh x265-thật lúc train hoạt động; nếu image nào thiếu,
code tự fallback proxy-only kèm cảnh báo (không crash).

```python
# 1) Kéo repo + cài deps
!git clone https://github.com/wagur1/preprocessing_upgrade_1.git
%cd preprocessing_upgrade_1
!pip install -q compressai            # torch/torchvision đã có sẵn trên Kaggle
!ffmpeg -version | head -1            # xác nhận có libx265

# 2A) Action recognition (Kinetics 5%) — one-shot: prepare index → train → eval
!python kaggle/run_kaggle.py \
    --config configs/action_recognition.yaml \
    --cap-gb 3 --epochs 3 --max-steps 300

# 2B) Object tracking (GOT-10k val)
!python kaggle/run_kaggle.py \
    --config configs/tracking.yaml \
    --cap-gb 3 --epochs 3 --max-steps 300 \
    --max-seqs 30 --max-frames 48
```

Dataset AR: [`rohanmallick/kinetics-train-5per`](https://www.kaggle.com/datasets/rohanmallick/kinetics-train-5per)
(add vào Input của notebook). Cap 3 GB bằng *indexing* round-robin, không copy.
`run_kaggle.py` tự: build index → `train.py` → `evaluate.py`, ghi vào `outputs/`.
Flags: `--epochs --max-steps --cap-gb --frame-size --batch-size --max-seqs --max-frames
--skip-prepare --skip-train --ckpt`.

**Lưu ý chi phí (quan trọng):** `real_codec_every_n_steps: 1` = MỖI step gọi 1 subprocess
ffmpeg/clip → chậm. Smoke test nên tăng lên `4` (mở `configs/*.yaml`, sửa key này) để
step khác dùng proxy-only. Theo dõi log `bpp_proxy vs bpp_real gap%` (mỗi 200 step): nếu
`gap` lớn kéo dài, chỉnh `qp_to_quality` cho khớp điểm rate hơn.

## Kiểm chứng (không cần GPU)

```bash
python -m compileall src/            # bắt lỗi cú pháp (đặc biệt _train_classification)
python tests/test_ste.py             # STE self-check, không framework
```
`tests/test_ste.py` assert: (a) `x_hat == x_hat_real` (forward = thật); (b) sau backward,
`x_pre.grad` = grad qua proxy, KHÔNG qua real; (c) `qp_to_quality` đơn điệu.

## Layout

```
src/
  models/preprocessor.py   two-branch residual preprocessor (trained)
  models/codec.py          CompressAI proxy (khả vi + range coder)
  codecs/standard.py       ffmpeg H.264/H.265 — anchor VÀ nhánh x265 lúc train (qp per-call)
  tasks/                   r3d_18 (AR), SiamFC + pytracking adapter (tracking)
  data/                    Kinetics + GOT-10k readers, index builders (≤3GB)
  metrics/                 BD-Rate / top-k / tracking AUC
  losses.py                α(L_D + λL_R) + L_Acc  (KHÔNG đổi)
  engine.py                train/eval + STE (_straight_through, _training_codec_*)
configs/                   action_recognition.yaml, tracking.yaml
tests/test_ste.py          STE self-check
train.py  evaluate.py      CLI
kaggle/                    one-shot driver + notebook
IMPLEMENTATION_PLAN.md     kế hoạch STE cho Codex   PAPER_PLAN.md  roadmap Q1
```

## Notes & caveats

* Codec áp **frame-wise** (setup CompressAI chuẩn); temporal modelling ở preprocessor.
  Anchor H.264/H.265 dùng inter-frame coding (cả sequence pipe vào ffmpeg) → so sánh sòng phẳng.
* **Phase 2 (calibration) CHƯA implement**: đặt `calibration: rate` sẽ raise. Phase 1
  (proxy đông cứng) là mặc định an toàn và đủ để vá W3 ở mức forward.
* AUC tuyệt đối của SiamFC khiêm tốn (backbone chỉ ImageNet, không train tracker); BD-Rate
  *tương đối* mới là số có nghĩa. Dùng pytracking cho số tuyệt đối của paper.

## Reference

Zhao et al., *A Preprocessing Framework for Video Machine Vision under Compression*
(arXiv:2512.15331). Repo này là independent implementation, thay virtual codec bằng
CompressAI và mở rộng vòng train sang x265 thật; không phải code của tác giả.



