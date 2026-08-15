# Kế hoạch nâng cấp preprocessor VCM 

Baseline: Zhao et al., *"A Preprocessing Framework for Video Machine Vision under Compression"* (arXiv:2512.15331, 2025).
Repo hiện tại: `C:\Users\Wagur1\Desktop\final1` (dựng lại paper, virtual codec thay bằng CompressAI `bmshj2018-factorized`).

> **Lưu ý:** MỌI con số BD-rate dưới đây là **giả thuyết kỹ thuật / mục tiêu**, KHÔNG phải kết quả đo. Claim cuối phải dùng CI, nhiều seed, đường cong từ bitstream thật.

Nhắc về ba trục:
- **Đường rate:** giảm BPP ở cùng ACC (làm mượt vùng ít giá trị, residual mạch lạc → nén rẻ).
- **Đường accuracy:** tăng ACC ở cùng BPP (giữ feature quan trọng cho task qua nén).
- **BD-rate** tích hợp cả hai. Phần lớn preprocessing thắng theo đường accuracy, phụ thêm đường rate.

---

## 1. Điểm yếu của baseline

- **W1 — Một preprocessor cho mỗi backbone.** Paper: *"For each machine vision network, we individually trained a corresponding preprocessor."* Không generalize, không triển khai thực tế. **Điểm yếu lớn nhất.**
- **W2 — Mù bitrate.** `fq` lấy ngẫu nhiên [30,50]; preprocessor không nhận QP làm input. Trong repo (`engine.py:135-137`): `pre(clips)` chạy trước, `q` chọn sau và chỉ đưa cho codec → học một phép chỉnh trung bình trên cả dải rate, tệ nhất ở bitrate thấp.
- **W3 — Lệch proxy ↔ codec thật.** Train trên proxy khả vi, test trên x264/x265. Repo bạn NẶNG hơn: proxy là CompressAI (codec ảnh học được), xa x265 hơn cả virtual codec tự chế của paper → results real-codec có thể đang thua chính paper.
- **W4 — Temporal không có motion alignment.** Chỉ 3D conv trên trục thời gian; vật thể chuyển động bị trộn qua vị trí không gian, không bù chuyển động. Yếu khi motion nhanh/occlusion.
- **W5 — Chỉnh sửa đồng đều.** `out = x + res_scale·delta` áp lên toàn khung; không có map tập trung giữ vùng task-critical và làm mượt nền ít giá trị.
- **W6 — Distortion neo vào nguồn (hướng người).** `L_D = MSE(x_hat, x_source)` — ép output giống ảnh gốc theo mắt người, có thể xung đột với phép chỉnh tối ưu cho máy.
- **W7 — Không ràng buộc nhất quán thời gian.** Không có loss ổn định residual theo thời gian → nhấp nháy, tốn bit inter-coding.
- **W8 — Phạm vi đánh giá hẹp.** Chỉ QP {30,35,40,45,50}, preset 'medium', chỉ x264/x265.
- **W9 — Perceptual sơ sài.** Chỉ VMAF; thiếu PSNR/MS-SSIM/LPIPS/color.

---

## 2. Đóng góp novelty (xương sống bài — Tier A)

Ba đóng góp này là phần **novel** . Riêng FiLM/gated-fusion (Tier B) là kỹ thuật đã biết — chỉ là *enabler*, không phải claim chính.

### A1 — Universal analyzer-agnostic preprocessor (HEADLINE)
- **Vá:** W1. Đây là đóng góp lớn nhất, khác biệt rõ nhất với baseline.
- **Thêm vào đâu:** thêm loss `L_feature` = multi-teacher feature distillation. Train preprocessor với ≥2 backbone teacher đồng thời (VD cho AR: r3d_18 + một backbone khác; cho tracking: 2 họ tracker). Ép feature trung gian của ảnh-đã-nén giống feature của ảnh nguồn ở **nhiều** analyzer, thay vì fit riêng một cái.
- **Bằng chứng bắt buộc:** **held-out backbone** — giữ lại ≥1 họ model không dùng khi train, đo BD-rate trên nó. Đây là thí nghiệm bán được bài.
- **Kỳ vọng:** trên backbone đã train giữ được gain của baseline; trên **held-out** đạt BD-rate dương (giả thuyết mục tiêu ~+10~12%) — baseline không làm được vì mỗi backbone một preprocessor.

### A2 — Machine-driven task-importance attention
- **Vá:** W5. Lấy kiến trúc spatial-attention của **GFSalNet** nhưng **importance map được giám sát bởi analyzer (máy), KHÔNG phải saliency người.** Đây là điểm repurpose = novelty; nếu chỉ copy GFSalNet (human saliency) thì không novel.
- **Thêm vào đâu:** thêm nhánh estimator xuất map `M ∈ [B,1,H,W]`; residual thành `out = x + res_scale · M ⊙ delta`. `M` supervise bằng tín hiệu từ task (gradient/feature-attribution của analyzer, hoặc vùng object/boundary/motion).
- **Kỳ vọng:** giữ vùng task-critical, làm mượt nền ít giá trị → **BPP↓** ở cùng ACC. BD-rate mục tiêu +2~4.
- **Lưu ý repo:** gated-fusion `g·f_t+(1-g)·f_s` trong `_ConditionalAttention` ĐÃ tồn tại (trùng GFSalNet gated fusion) → không tính là mới. Chỉ spatial mask + channelwise/SE + multiscale mới là phần thêm.

### A3 — Real-codec-calibrated training
- **Vá:** W3 (và với repo này là vá regression proxy CompressAI ↔ x265).
- **Thêm vào đâu:** định kỳ encode một subset clip bằng x265 thật, thêm `L_rate_cal = |log bpp_proxy − log bpp_real|` để hiệu chỉnh proxy về bitrate thật. Cache output để đỡ tốn.
- **Kỳ vọng:** giảm mismatch proxy→real, ổn định gain khi đo trên codec thật. BD-rate mục tiêu +2~6.

---

## 3. Enabler kỹ thuật (Tier B — cần, nhưng không phải novelty chính)

### B1 — FiLM QP/rate conditioning
- **Vá:** W2. Hiện `engine.py` chọn `q` SAU khi `pre()` chạy → preprocessor mù rate.
- **Thêm vào đâu:** đưa QP/target-rate thành condition vector, inject bằng FiLM (`γ,β` theo QP) vào các block preprocessor. Đồng thời trong train phải sample QP TRƯỚC rồi mới đưa vào cả preprocessor lẫn codec.
- **Kỳ vọng:** một model phủ nhiều điểm rate, mạnh nhất ở bitrate thấp. BD-rate mục tiêu +3~8.

### B2 — Motion-aligned temporal branch
- **Vá:** W4. Thay/bổ sung 3D-conv bằng nhánh temporal có bù chuyển động: warp feature khung trước theo motion (codec MV ưu tiên vì rẻ & sát nén; hoặc optical flow nhẹ / deformable conv).
- **Kỳ vọng:** tốt cho tracking & motion nhanh/occlusion. Tracking AUC BD-rate +2~5.

---

## 4. Bổ trợ (Tier C)

- **SE / channelwise attention + multiscale fusion** (phần còn lại của GFSalNet chưa có trong repo): +0~2.
- **L_temporal** — nhất quán residual theo thời gian (motion-compensated) → giảm nhấp nháy, đỡ tốn bit inter: vá W7, +1~3.
- **Feature-domain distortion** — thay/bổ sung `L_D=MSE(x_hat,x_source)` bằng distortion trên feature analyzer (vá W6, tránh xung đột hướng-người vs hướng-máy).

---

## 5. Hàm loss tổng hợp (đề xuất)

```
L = λ_rate · L_rate      (real-codec-calibrated, gồm L_rate_cal)
  + λ_task · L_task       (AR: top-1 / tracking: response loss)
  + λ_feat · L_feature    (multi-teacher distillation — A1)
  + λ_temp · L_temporal   (nhất quán residual theo thời gian — C)
  + λ_perc · L_perceptual (MS-SSIM/LPIPS/VMAF-surrogate/color/edge)
  + λ_edit · L_edit        (L1 + TV, chặn chỉnh sửa phá hủy)
```
Từ loss baseline `L = α(L_D+λL_R)+L_Acc` → mở rộng: L_Acc→L_task, thêm L_feature (A1), L_rate_cal (A3), L_temporal (C). Cân trọng số bằng ablation.

---

## 6. Kế hoạch thí nghiệm

**Phase I — publishable core**
- Codec: x265/HEVC, QP {22,27,32,37,42}, ≥2 preset + GOP settings.
- Task: bắt đầu detection/tracking (đủ signal), giữ AR nếu compute cho phép.
- Data: đủ lớn — cân nhắc BDD100K + MOT17/20 hoặc GOT-10k/LaSOT (5%-Kinetics/3GB chỉ đủ prototype, KHÔNG đủ camera-ready).
- Teacher: ≥2 họ mỗi task, **hold out ≥1 họ**.
- So sánh: codec-only anchor, reproduce baseline, denoise/sharpen generic, fine-tune-on-compressed, ROI/QP-map.

**Phase II — mở rộng journal**
- Thêm segmentation, thêm AV1/VVC, cross-codec transfer, rate-control modes thật.
- Đo latency/energy/params/MACs/memory + tổng thời gian preprocess+encode.
- Robustness: dataset lạ, ánh sáng, motion nhanh, occlusion, scene cut, small object.

**Metrics:** BD-rate vs {mAP, HOTA/MOTA/AUC, mIoU, top-1}; bitrate thật, VMAF, PSNR, MS-SSIM, LPIPS, color; runtime/throughput/size/peak-mem/energy.

---

## 7. Ablations bắt buộc

- spatial-only vs temporal-không-align vs **temporal-motion-aligned**.
- có/không: semantic map (A2), QP conditioning (B1), task conditioning, feature distillation (A1), temporal consistency (C), real-codec calibration (A3).
- single teacher vs teacher ensemble; task-specific vs universal.
- RGB vs YUV; causal vs non-causal; learned flow vs codec MV.
- độ nhạy residual-bound; trade-off complexity↔accuracy.

---

## 8. Thứ tự triển khai

1. **A1** (multi-teacher distillation + held-out) — làm trước, là headline.
2. **B1** (FiLM QP) — sửa rate-blindness, nền cho mọi thứ sau.
3. **A3** (real-codec calibration) — vá mismatch proxy↔x265 của repo.
4. **A2** (machine-driven attention).
5. **B2** (motion-aligned temporal).
6. **C/D** (SE/multiscale, L_temporal, feature-domain distortion, eval rộng + ablation + complexity).

---

## 9. Rủi ro & phạm vi (đọc trước khi commit)

- **Compute:** Kaggle + 5%-Kinetics + cap 3GB chỉ là harness prototype, **KHÔNG đủ cho camera-ready**. Cần scale data + GPU trước Phase I thật.
- **Novelty A1:** phải đối chiếu literature VCM/ICM 2023–2025 (universal/analyzer-agnostic preprocessing có thể đã có) TRƯỚC khi chốt là đóng góp chính. Nếu trùng → đẩy A3 (real-codec calibration) + generalization evidence làm mũi nhọn.
- **Universal model có thể mất performance task-specific** → dùng adapter nhẹ / MoE head trên shared backbone.
- **Semantic smoothing hại task lạ** → multi-teacher, uncertainty-aware map, residual bound bảo thủ.
- **Chi phí preprocess ăn mất lợi ích bitrate** → báo cáo end-to-end energy/latency, có bản lightweight (<5M params).
- **Claim bị chê incremental** → đặt generalization (held-out analyzer) + real-codec calibration làm trung tâm, KHÔNG phải "CNN to hơn".

> Nhắc lại: mọi con số BD-rate là **mục tiêu/giả thuyết**. Claim cuối phải dùng CI, nhiều seed, đường cong từ bitstream x265/AV1/VVC thật.
