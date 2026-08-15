# IMPLEMENTATION_PLAN — Train trên codec thật (x265) + proxy khả vi cho backprop

> Kế hoạch để **Codex implement**. Repo này (`upgrade1`) là fork của `final1`; pipeline giữ nguyên, chỉ mở rộng vòng train. Đường dẫn file tính từ gốc repo.

## Mục tiêu

Hiện preprocessor train bằng **proxy CompressAI khả vi** (`src/models/codec.py:122`) nhưng eval đo trên **x264/x265 thật** (`src/codecs/standard.py`). Lệch proxy↔real → task loss tối ưu sai codec. Sửa: **forward đi qua x265 thật**, **gradient chảy qua proxy** (x265 không khả vi). Giữ nguyên pipeline `video→Pre→Codec→recon→Analyzer→task`.

Đây là kỹ thuật **Straight-Through Estimator (STE) / detached-difference**.

## Hai phase

- **Phase 1 (bắt buộc):** STE với proxy CompressAI **đông cứng**. Forward = x265 thật, backward = gradient CompressAI. Không train lại proxy. Diff nhỏ, rủi ro thấp.
- **Phase 2 (tùy chọn):** proxy **trainable** + `L_proxy` calibration để proxy bám x265. Bật bằng config `train.calibration: rate`.

## Cơ chế (áp cho CẢ `_train_classification` và `_train_tracking` trong `src/engine.py`)

```python
qp = random.choice(qp_list)                 # SAMPLE QP TRƯỚC pre()
q_prx = qp_to_quality[qp]                    # map QP → CompressAI quality index
x_pre = pre(clips)                           # forward chữ ký GIỮ NGUYÊN: pre(x)

x_hat_prx, bpp_prx = codec(x_pre, q_prx)     # proxy khả vi

if real_step:                               # mỗi real_codec_every_n_steps
    x_hat_real, bpp_real = real_codec(x_pre, qp)   # StandardCodec, no_grad
    bpp_real_t = torch.as_tensor(bpp_real, device=x_pre.device, dtype=bpp_prx.dtype)
    x_hat = x_hat_prx + (x_hat_real - x_hat_prx).detach()   # forward=thật, grad qua proxy
    bpp   = bpp_prx   + (bpp_real_t - bpp_prx ).detach()
else:
    x_hat, bpp = x_hat_prx, bpp_prx

acc_loss, _ = analyzer.accuracy_loss(x_hat, target)
parts = rate_distortion_accuracy_loss(clips, x_hat, bpp, acc_loss, weights)  # GIỮ NGUYÊN
```

Bất biến: forward `x_hat == x_hat_real`; `∂x_hat/∂x_pre = ∂x_hat_prx/∂x_pre`.

## Files phải sửa

### 1. `src/engine.py` — thay đổi chính
- **SỬA LỖI TRƯỚC:** `_train_classification` (khoảng `:97-160`) **thiếu header vòng lặp** (`for epoch`, `pbar = tqdm(...)`, `for clips, labels in pbar:`); code từ `:135` bị mồ côi → `IndentationError` khi chạy. Khôi phục theo đúng mẫu `_train_tracking` (`:195-197`).
- Ở CẢ hai vòng (`:135-137` cls, `:201-203` trk): đổi thứ tự sang **sample `qp` TRƯỚC `pre()`**, map `qp→q_prx`, chạy proxy + real, tổ hợp detached-difference, feed `x_hat/bpp` tổ hợp vào `rate_distortion_accuracy_loss` (call site `:140`/`:206` giữ nguyên chữ ký).
- Khởi tạo `real_codec` một lần ngoài vòng epoch. Guard `ffmpeg_available()` (`src/codecs/standard.py:29`): thiếu ffmpeg → warn + tự tắt real (fallback proxy-only, KHÔNG crash).

### 2. `src/codecs/standard.py` — cho dùng trong training
- `compress_decompress` đã `@torch.no_grad()`, trả `(x_hat [B,C,T,H,W] on x.device, bpp: float)` — dùng nguyên cho nhánh real. Thêm cách truyền `qp` per-call: hoặc tạo `StandardCodec(codec=name, qp=qp, preset=...)` mỗi step (object nhẹ), hoặc thêm arg `qp` optional vào `compress_decompress`.
- KHÔNG đổi logic encode/decode/bpp. Lưu ý bpp (`:107`) chia `T*H*W` (luma) — khớp convention `bpp_prx` cho Phase 2.

### 3. `src/losses.py` — CHỈ Phase 2
- Thêm `proxy_calibration_loss(bpp_prx, bpp_real_t, x_hat_prx, x_hat_real) -> tensor = |log bpp_prx − log bpp_real| + MSE(x_hat_prx, x_hat_real)`. Phase 1 KHÔNG đụng file này.

### 4. `configs/action_recognition.yaml` và `configs/tracking.yaml` — thêm dưới `train:`
```yaml
train:
  real_codec: true
  real_codec_name: h265          # h264 | h265
  real_codec_preset: medium
  qp_list: [22, 27, 32, 37, 42]
  qp_to_quality: {22: 8, 27: 5, 32: 3, 37: 2, 42: 1}   # đơn điệu: QP cao ↔ quality thấp
  real_codec_every_n_steps: 1    # >1 để amortize chi phí ffmpeg
  calibration: none              # none (Phase 1) | rate (Phase 2)
```

## QP ↔ quality mapping
Proxy dùng CompressAI quality index (repo set `[1,3,5,8]`), real dùng ffmpeg `-qp`. Hai nhánh phải ở **điểm rate xấp xỉ nhau** → `qp_to_quality` đơn điệu. Log chênh `bpp_prx` vs `bpp_real` mỗi vài trăm step để tinh chỉnh. Map lệch nhẹ chỉ nhiễu gradient, KHÔNG sai forward (nhánh real mang giá trị đúng).

## Kiểm soát chi phí
- x265 trong loop = 1 subprocess ffmpeg/clip → chậm. `real_codec_every_n_steps > 1`: step real dùng STE, step khác proxy-only.
- KHÔNG cache cross-step (x_pre trôi mỗi step). Cache chỉ dùng cho subset calibration Phase 2.

## Phase 2 — proxy calibrated (chỉ làm sau khi Phase 1 ổn)
- `CompressAICodec(..., trainable=True)` (`src/models/codec.py:68`), optimizer RIÊNG `opt_proxy`.
- Two-timescale: mỗi real_step, sau khi update preprocessor, backward `L_proxy` chỉ update `opt_proxy` (detach `x_pre`). Cache subset real-encode.
- Guard: sai số `|bpp_prx−bpp_real|/bpp_real` > ~8% kéo dài → rớt về `calibration: none`. `L_rate_cal` chỉ có nghĩa khi proxy trainable.

## Ngoài scope PR này
- B1 FiLM QP-conditioning vào preprocessor (`pre(clips, qp)`) — `preprocessor.forward` hiện chỉ nhận `x` (`src/models/preprocessor.py:124`). Sample-QP-trước-pre() ở đây là *tiền đề*, wiring FiLM là PR riêng.
- A1/A2/B2, feature-domain distortion, và `src/metrics/` — giữ nguyên hoàn toàn.

## Verification
1. **Compile:** `python -m compileall src/` sạch (bắt lỗi cú pháp `_train_classification`).
2. **Self-check `tests/test_ste.py` (không framework):** `x_pre` requires_grad; proxy giả + real giả (VD `real=round(x)`); assert (a) `torch.allclose(x_hat, x_hat_real)`; (b) sau `x_hat.sum().backward()`, `x_pre.grad` = grad qua proxy, KHÔNG qua real. Assert `qp_to_quality` đơn điệu.
3. **Smoke train (Kaggle GPU+ffmpeg):** `train.py --config configs/tracking.yaml train.max_steps=20 train.real_codec_every_n_steps=1` — loss giảm, không crash, log chênh bpp.
4. **Eval:** `evaluate.py` như cũ; BD-Rate `prep+compressai` vs `h265` cải thiện so với baseline proxy-only (kỳ vọng, không phải claim đo).

## Thứ tự
1. Sửa cú pháp `_train_classification`.
2. Thêm config keys + đọc trong `engine.py`.
3. Viết detached-difference cho `_train_tracking` trước (vòng nguyên vẹn), rồi mirror sang `_train_classification`.
4. `tests/test_ste.py` + `compileall`.
5. (Tùy chọn) Phase 2 calibration.
