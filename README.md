# Preprocessing for Video Machine Vision — Upgrade 1 (rate-adaptive preprocessor)

Fork của [`preprocessing_final`](https://github.com/wagur1/preprocessing_final) (bản dựng lại
Zhao et al., *"A Preprocessing Framework for Video Machine Vision under Compression"*,
với "virtual codec" của paper thay bằng CompressAI `bmshj2018-factorized`).

**Điểm mới của bản này — preprocessor thích ứng theo bitrate (rate-adaptive).**
Baseline train **mù rate**: `pre()` chạy xong mới chọn điểm nén, nên preprocessor học một
phép chỉnh *trung bình* cho cả dải rate (tệ nhất ở bitrate thấp — điểm yếu **W2**). Bản này
**điều kiện hoá preprocessor theo điểm nén mục tiêu** qua **FiLM** (Perez et al. 2018):
một scalar QP-chuẩn-hoá được bơm vào mọi residual block dưới dạng affine per-channel, để
**một model duy nhất** tự điều chỉnh theo từng bitrate thay vì học một trung bình mờ.

> **STE (bản trước) đã bị tắt mặc định.** Cách train-through-x265 bằng Straight-Through
> Estimator làm *forward = x265* nhưng *gradient = proxy CompressAI* — hai họ artifact khác
> nhau nên gradient lệch hướng, khiến kết quả **tệ hơn** cả proxy-only. Code STE vẫn còn
> (`real_codec: true` để bật lại) nhưng mặc định `false`; hướng đúng để vá W3 là **emulator
> x265 khả vi distilled** (xem [Roadmap](#roadmap)).

Chỉ preprocessor được train; codec và analyzer đông cứng.

## Pipeline chi tiết

### Train (chỉ preprocessor học)

```
                    qp ~ qp_list                     (1) sample điểm nén TRƯỚC pre()
                        │
          ┌─────────────┴─────────────┐
          │                           │
   c = [qp_norm]                q = qp_to_quality[qp]  (proxy quality index)
   (rate condition)                   │
          │                           │
  x ─▶ Preprocessor(x, c) ─▶ x_pre ─▶ CompressAI(q) ─▶ x_hat, bpp
       │  FiLM(c) trong mọi          (proxy khả vi,     │
       │  residual block            đông cứng)          │
       ▼                                                ▼
  temporal 3D-conv ┐                          frozen Analyzer ─▶ acc_loss
  spatial  2D-conv ┘─▶ attention ─▶ tail ─▶ x+res·Δ        │
          ▲                                               ▼
          └──── gradients ◀── L = α(L_D + λ·L_R) + L_Acc ◀─┘
```

Các bước một step train:
1. **Sample `qp`** từ `qp_list` *trước* khi gọi preprocessor (baseline làm ngược → mù rate).
2. **Dựng condition** `c = [qp_norm]`, `qp_norm = (qp − qp_lo)/(qp_hi − qp_lo) ∈ [0,1]`
   (1 = nén mạnh nhất). `qp_lo/qp_hi` = `model.qp_ref`.
3. **`x_pre = pre(x, c)`** — FiLM biến `c` thành `(γ,β)` per-channel, áp `(1+γ)·feat+β`
   trong từng residual block của **cả hai** nhánh temporal & spatial.
4. **`x_hat, bpp = codec(x_pre, q)`** — proxy CompressAI khả vi (`q = qp_to_quality[qp]`).
5. **`L = α(L_D + λ·L_R) + L_Acc`**, backward, chỉ cập nhật preprocessor.

FiLM và residual tail đều **zero-init** ⇒ lúc khởi tạo preprocessor là *identity*
(train ổn định), conditioning chỉ "bật dần" khi học.

### Eval (rate-accuracy curve, range coder thật)

Vì output preprocessor **phụ thuộc điểm nén**, không thể preprocess-một-lần rồi quét rate;
`evaluate.py` **dựng `c` và chạy `pre(x, c)` lại cho từng điểm rate**:

```
for mỗi điểm rate (q ∈ codec.qualities, hoặc qp ∈ eval.qp_list):
    c      = [level(điểm rate)]            # QP → qp_norm; quality index → level tương ứng
    x_pre  = pre(x, c)
    ─▶ prep+compressai : CompressAI range-coder(x_pre, q)
    ─▶ prep+h264/h265  : ffmpeg(x_pre, qp)
    (anchor: compressai / h264 / h265 chạy trực tiếp trên x, không preprocessor)
```

So sánh 6 pipeline → **BD-Rate** (âm = tiết kiệm bit ở cùng accuracy). Hai nhóm số:
- `bd_prep_gain` — **cùng codec, chỉ khác có/không preprocessor** (`prep+h265 vs h265`,
  `prep+h264 vs h264`, `prep+compressai vs compressai`): **đây là số đúng luận điểm.**
- `bd_vs_anchor` — `prep+compressai` vs từng anchor (tham khảo, khác codec).

Output: `results.json`, `curves.csv`, `rate_accuracy.png`, `qualitative.png`.

## Cơ chế rate-conditioning (FiLM)

```python
class FiLM(nn.Module):                       # src/models/preprocessor.py
    def forward(self, x, cond):              # x:[B,C,T,H,W]  cond:[B,cond_dim]
        gamma, beta = self.net(cond).chunk(2, dim=1)   # per-channel affine
        return x * (1 + gamma) + beta        # zero-init net ⇒ identity lúc đầu
```

- `cond_dim = 1` hiện mang **qp_norm**. Vector để sẵn chỗ nối thêm `log R_target` khi làm
  **rate control** (đạt một bitrate mục tiêu, không chỉ biết QP) — bước kế của roadmap.
- FiLM đặt trong **mỗi `_ResBlock3d`** của nhánh temporal và spatial; attention-fusion & tail
  giữ nguyên.

## Bảng cải thiện so với baseline (`preprocessing_final`)

| # | Phần | Baseline `final1` | Upgrade1 | Vá điểm yếu | Trạng thái |
|---|------|-------------------|----------|-------------|-----------|
| 1 | **Rate-conditioning (FiLM)** | Preprocessor mù rate `pre(x)` | `pre(x, c)`, FiLM `(1+γ)·feat+β` mỗi res-block | **W2** | ✅ mới, mặc định bật |
| 2 | **Sample QP trước `pre()`** | `pre()` trước, `q` chọn sau | Sample `qp` → dựng `c` → `pre(x, c)` | **W2** | ✅ đã code |
| 3 | **Eval theo từng điểm rate** | preprocess 1 lần, quét rate | `pre(x, c)` lại cho mỗi q/qp | đúng ngữ nghĩa rate-adaptive | ✅ đã code |
| 4 | **STE forward x265 thật** | — | `_straight_through` (còn code) | **W3** (hỏng: gradient lệch) | ⚠️ **deprecated, tắt mặc định** |
| 5 | **Emulator x265 khả vi distilled** | — | thay STE để vá W3 đúng cách | **W3** | ⏳ roadmap |
| 6 | **Two-residual + task-importance mask** | — | `Δ_task`/`Δ_smooth`, video mask `[B,1,T,H,W]` | **W5/W7** | ⏳ roadmap |
| 7 | **Held-out analyzer eval** | — | train 1 analyzer, eval analyzer khác (pytracking) | **W1** (generalization) | ⏳ roadmap (gần free) |

Data loaders, metrics (BD-Rate/AUC/top-1), tasks, eval end-to-end **giữ nguyên** baseline.
Roadmap Q1 đầy đủ (W1–W9, Tier A/B/C): [`PAPER_PLAN.md`](PAPER_PLAN.md);
kế hoạch STE cũ: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Hàm loss (giữ nguyên paper Eq. 1)

```
L = α · (L_D + λ · L_R) + L_Acc          α = 10,  λ = 0.001
```
`L_D` = MSE(recon, **source**); `L_R` = bpp; `L_Acc` = cross-entropy (AR) hoặc SiamFC
balanced-logistic (tracking). FiLM **không đổi** `losses.py`.

## Config mới

```yaml
model:                       # preprocessor
  cond_dim: 1                # độ rộng rate-condition (1 = qp_norm; đầu vào FiLM)
  qp_ref: [20, 51]           # dải QP ánh xạ về mức condition [0,1]

train:
  real_codec: false          # STE deprecated → mặc định proxy-only (coherent)
  qp_list: [22, 27, 32, 37, 42]
  qp_to_quality: {22: 5, 27: 3, 32: 2, 37: 1, 42: 1}   # đơn điệu: QP↑ ↔ quality↓
  # điều khiển train cho real run (không phải smoke test):
  cosine: true               # cosine LR decay trên toàn bộ step
  patience: 5                # early-stop: dừng sau N epoch val-loss không giảm (0 = tắt)
  min_delta: 1.0e-4          # mức giảm tối thiểu tính là "cải thiện"
  val_max_batches: 20        # cap số batch val mỗi epoch (giới hạn chi phí đo)
  resume: false              # train tiếp từ outputs/checkpoints/preprocessor_last.pth
```
`codec.qualities = [1,2,3,5,8]` phủ mọi giá trị trong `qp_to_quality`.

**Checkpoint & dừng.** `_fit` (dùng chung cho AR & tracking) mỗi epoch: đo **val-loss**
(proxy-only, QP giữa cố định — tín hiệu coherent, rẻ), lưu `preprocessor_last.pth` (để
`--resume`) và ghi `preprocessor.pth` = **best-val** (đây là ckpt `evaluate.py` nạp). Cosine
LR decay bật mặc định; **early-stop** dừng khi val-loss không giảm quá `patience` epoch —
nên không phải đoán số epoch, cứ đặt `epochs` dư rồi để nó tự dừng ở best. Nếu val split
rỗng (cap quá nhỏ) thì bỏ qua val: train đủ epoch, `last = best`.

## Chạy trên Kaggle

Cần **GPU** + **Internet ON**. Kaggle image có sẵn `ffmpeg` (libx264/libx265) cho anchor
H.264/H.265 lúc eval.

```python
!git clone https://github.com/wagur1/preprocessing_upgrade_1.git
%cd preprocessing_upgrade_1
!pip install -q compressai            # torch/torchvision đã có trên Kaggle

# Action recognition (Kinetics 5%) — one-shot: prepare index → train → eval
!python kaggle/run_kaggle.py --config configs/action_recognition.yaml \
    --cap-gb 6 --epochs 12          # early-stop tự dừng ở best-val; --resume để train tiếp

# Object tracking (GOT-10k val)
!python kaggle/run_kaggle.py --config configs/tracking.yaml \
    --cap-gb 6 --epochs 12 --max-seqs 120 --max-frames 90
```

Real run: bỏ `--max-steps`, đặt `--epochs` dư (early-stop lo phần dừng). Không đủ 1 session
≤12h thì tách: session train (`preprocessor_last.pth` lưu ở Kaggle output) → session sau
`--resume` train tiếp, hoặc `--skip-train --ckpt outputs/checkpoints/preprocessor.pth` để eval.

Dataset AR: [`rohanmallick/kinetics-train-5per`](https://www.kaggle.com/datasets/rohanmallick/kinetics-train-5per).
Cap dataset bằng *indexing* round-robin (không copy); `--cap-gb` có thể nới lên khi cần
đường cong/CI chắc hơn (3 GB chỉ là mức prototype).

## Kiểm chứng

```bash
python -m compileall src/ tests/    # bắt lỗi cú pháp
python tests/test_film.py           # FiLM: identity lúc init + output phụ thuộc cond (cần torch)
python tests/test_earlystop.py      # điều kiện dừng: patience/min_delta (import engine → cần torch)
python tests/test_ste.py            # STE helper self-check (STE vẫn còn code)
```

## Roadmap

Thứ tự triển khai tiếp (compute Kaggle-free; xem `PAPER_PLAN.md`):
1. ✅ **FiLM rate-conditioning** (bản này) — vá W2.
2. **Two-residual + task-importance mask** video `[B,1,T,H,W]` + `L_mask-temp` motion-comp — vá W5/W7.
3. **Emulator x265 khả vi distilled** (offline distill → freeze → train-through; recalibrate
   định kỳ) thay STE — vá W3 đúng cách.
4. **Held-out analyzer eval** qua pytracking (train SiamFC, eval DiMP/ATOM) — bằng chứng W1, gần free.
5. Same-codec BD-rate + 3 seed/bootstrap CI + runtime/FLOPs/param + VMAF/LPIPS.

## Layout

```
src/
  models/preprocessor.py   two-branch residual preprocessor + FiLM rate-conditioning (trained)
  models/codec.py          CompressAI proxy (khả vi + range coder)
  codecs/standard.py       ffmpeg H.264/H.265 anchor (+ nhánh x265 STE, tắt mặc định)
  tasks/                   r3d_18 (AR), SiamFC + pytracking adapter (tracking)
  data/                    Kinetics + GOT-10k readers, index builders
  metrics/                 BD-Rate / top-k / tracking AUC
  losses.py                α(L_D + λL_R) + L_Acc  (KHÔNG đổi)
  engine.py                train/eval + rate-cond helpers (_qp_norm/_quality_level/_rate_cond)
                           + _fit dùng chung: cosine LR, val-loss/epoch, early-stop, resume
configs/                   action_recognition.yaml, tracking.yaml
tests/                     test_film.py (FiLM), test_earlystop.py (điều kiện dừng), test_ste.py
train.py  evaluate.py      CLI          kaggle/  one-shot driver + notebook
PAPER_PLAN.md              roadmap Q1   IMPLEMENTATION_PLAN.md  kế hoạch STE (cũ)
```

## Notes & caveats

* Codec proxy áp **frame-wise**; temporal modelling nằm ở preprocessor. Anchor H.264/H.265
  dùng inter-frame coding (cả sequence pipe vào ffmpeg) → so sánh sòng phẳng.
* Eval rate-adaptive **tốn hơn** baseline: preprocessor chạy lại cho mỗi điểm rate.
* AUC tuyệt đối của SiamFC khiêm tốn (backbone ImageNet, không train tracker); **BD-Rate
  tương đối** mới là số có nghĩa. Dùng pytracking cho số tuyệt đối của paper.

## Reference

Zhao et al., *A Preprocessing Framework for Video Machine Vision under Compression*
(arXiv:2512.15331). Repo này là independent implementation (virtual codec → CompressAI,
thêm rate-adaptive preprocessing); không phải code của tác giả.
