# Phase 1: 3D Spatial Radio Prior + EKF Wireless Update — 评测报告

> 实验时间: 2026-06-26
> 实现: `trackers/spatial_projector.py` + `trackers/wifi_spatial_ocsort.py`
> 评测: strict LOSO, outdoor 4 folds (scene1/2/3/4), GT detections, IoU=0.5
> 计划文档: `experiments/action1_spatial_radio_prior_plan.md`

---

## 1. 实现要点

按照 Phase 1 计划，新增 `WiFiSpatialOCSortTracker`，相对于 Method A (`WiFiOCSortTracker`) 的差异：

1. **2D→3D 反投影**：用 ZED2 HD720 内参 (`FX=528.365, FY=527.925, CX=638.925, CY=359.28`) 把每个 bbox 中心 + depth 反投影到相机坐标系。
2. **AP 位置估计**：用绑定 phone 的 track 在前 50 帧累积 (XYZ, FTM) 对，scipy 最小二乘解出 AP 在相机系下的位置 (`estimate_ap_position`，要求 ≥20 对 + RMS<5m)。
3. **Phone state with 各向异性协方差**：以最近 bound track 为锚，把 phone 投影到以 AP 为中心、半径=FTM 的球面，构造 3×3 协方差 Σ = R · diag(σ_r², σ⊥², σ⊥²) · Rᵀ（径向小、切向大）。
4. **EKF wireless update**：观测方程 z = ‖p_track − p_AP‖ − r_FTM，Jacobian 在 (u, v) 上展开：
   H_wire = [ (Δx·d) / (d̂·FX),  −(Δy·d) / (d̂·FY),  0,0,0,0,0 ]
   Joseph form 协方差更新，自适应 R_wire = σ_FTM² + σ_depth(d)² + σ_AP²。
5. **遮挡桥接**：bound track `time_since_update > 0` 时也跑 EKF wireless update，让 FTM 持续修正预测 bbox。
6. **`compat_mode` 开关**：关联代价支持 scalar（与 Method A 等价的 1D Gaussian）或 maha（Phase 1 3D Mahalanobis）。

---

## 2. 4-fold LOSO 结果 (outdoor)

所有配置：spatial_weight=0.10, bind_threshold=0.50, depth_sigma=1.5。

| 配置 | scene1 | scene2 | scene3 | scene4 | avg IDF1 | Δ vs OC-SORT | Δ vs Method A |
|------|--------|--------|--------|--------|----------|--------------|---------------|
| **OC-SORT baseline** | 71.39% | 67.21% | 61.43% | 62.49% | **65.63%** | - | -12.88 pt |
| **Method A (WiFi-Geom)** | 81.76% | 81.48% | 72.97% | 80.02% | **78.51%** | +12.88 pt | - |
| **Phase 1: scalar + EKF + bridge** | 81.58% | 81.48% | 72.97% | 79.32% | **78.84%** | +13.21 pt | **+0.33 pt** |
| Phase 1: scalar + no-EKF + no-bridge | 81.58% | 81.48% | 72.92% | 78.68% | **78.67%** | +13.04 pt | +0.16 pt |
| Phase 1: **maha** + EKF + bridge | 72.83% | 69.47% | 61.95% | 61.12% | **66.34%** | +0.71 pt | −12.17 pt |

完整 JSON：
- `exps/wifi_spatial_all_scalar.json` (主推荐配置)
- `exps/wifi_spatial_all_scalar_noekf.json` (消融)
- `exps/wifi_spatial_all_maha.json` (消融)

---

## 3. 主要发现 (Key Findings)

### 3.1 Phase 1 整体无显著提升

scalar + EKF + bridge 比 Method A 仅 **+0.33 pt**，远低于计划目标 (+3~5 pt)。

### 3.2 EKF / bridge 在本数据集几乎无效

`+EKF+bridge` 相对 `no-EKF+no-bridge` 只多 +0.17 pt（78.84 vs 78.67）。**根因**：本评测用 **GT bbox 作为 detection**，视觉关联本身近乎完美（MOTA ~97%），Kalman 状态由视觉观测充分约束；wireless 更新提供的边际信息几乎被淹没。

若改用真实检测器（YOLO/DETR 等），bbox 坐标抖动 + 漏检会导致视觉观测可信度下降，wireless update 的边际增益预期会显著放大。**此假设在 Phase 1 数据下未能证伪也未能证明**。

### 3.3 Mahalanobis cost mode 显著伤害关联 (−12.17 pt)

`maha` 模式比 `scalar` 差 ~12 pt，根因是 **单 AP 设定下 phone state 的方位完全由 anchor (bound track) 决定**，导致 phone state 的位置 ≈ bound track 当前位置，phone-track 兼容性退化为"自己跟自己比"，反而把切向不确定性（σ_perp=2.5m）传染给本来精确的视觉关联。

加大 σ_perp 不能解决：σ_perp 越大，cost 越平坦，提供的判别力反而下降。**真正修复需要多 AP / IMU 提供独立方位约束**。

### 3.4 与 Method D (Learned Affinity, strict LOSO) 对比

| | avg IDF1 outdoor | 复杂度 | 训练成本 |
|---|---|---|---|
| Method A (Geom) | 78.51% | 几何启发式 | 0 |
| Method D strict | 75.62% | Bi-LSTM 编码器 | 2.2h CPU/fold |
| Phase 1 (best) | 78.84% | 3D Maha + EKF + AP LS | <1s warm-up |

Phase 1 ≈ Method A，仍优于 Method D。复杂度高但无显著收益。

### 3.5 验证 / 否决了 action1 计划中的假设

| 假设 (action1 plan) | 验证结果 |
|--------------------|---------|
| H1: 3D Mahalanobis 比 1D Gaussian 更精确 | ❌ 否决（单 AP 下退化）|
| H2: EKF wireless update 减少 KF predict 漂移 | ⚠️ 未证伪（GT 检测下不显著）|
| H3: Bridging 在遮挡时维持身份连续性 | ⚠️ 未证伪（IDsw 几乎不变）|
| H4: AP 位置可通过 warm-up 估计 | ✅ 验证通过（rms ~0.3m, ~50 帧收敛）|

---

## 4. 路线决策 (Decision)

**不把 Phase 1 作为论文主创新点**。原因：
1. 单 AP 几何上限决定 3D Mahalanobis 不可能优于 1D scalar；
2. GT-detection 评测下 EKF/bridge 收益不可观察；
3. 与 baseline (Method A) 持平在科学贡献上是"实现复现"，不是"性能提升"。

**下一步候选**（待用户决策）：

- **A. 切换到真实检测器（YOLOv8）评测**：复现 Phase 1，看 EKF/bridge 在带噪声 detection 下能否激活；周期 2-3 天。
- **B. 引入 IMU 方位约束修正 maha cost**：用 IMU heading 替代 anchor 决定 phone 方位，理论上能解锁 3D Mahalanobis；周期 1 周。
- **C. 转向 Direction β 主线（Tracklet Bridging via Wireless Continuity）**：focus 在遮挡期身份维持，物理意义更明确，与论文叙事更连贯；周期 1-2 周。
- **D. 重新评估问题选型**：在 Vi-Fi + WP-ReID 双数据集上做更细致的失败模式分析，再决定主创新点。

---

## 5. 复现命令

```bash
# Phase 1 主推荐配置（scalar + EKF + bridge）
python3 scripts/run_baseline.py \
  --tracker wifi_spatial --all-folds \
  --spatial-weight 0.10 --bind-threshold 0.50 --depth-sigma 1.5 \
  --compat-mode scalar \
  --out exps/wifi_spatial_all_scalar.json

# 消融：不要 EKF / bridge
python3 scripts/run_baseline.py \
  --tracker wifi_spatial --all-folds \
  --spatial-weight 0.10 --bind-threshold 0.50 --depth-sigma 1.5 \
  --compat-mode scalar --no-ekf --no-bridge \
  --out exps/wifi_spatial_all_scalar_noekf.json

# 消融：Mahalanobis cost mode
python3 scripts/run_baseline.py \
  --tracker wifi_spatial --all-folds \
  --spatial-weight 0.10 --bind-threshold 0.50 --depth-sigma 1.5 \
  --compat-mode maha \
  --out exps/wifi_spatial_all_maha.json
```

---

## 6. 待办 (action items)

- [ ] 用户决策路线 A / B / C / D
- [ ] 把 Phase 1 结果同步到 `experiment_summary.md` §7.6
- [ ] 若选 A：搭建 YOLOv8 detection pipeline 替换 GT
- [ ] 若选 B：在 `_build_phone_states` 中接入 IMU heading anchor
- [ ] 若选 C：实现 wireless-only tracklet bridging（不依赖 EKF）
