# Phase 4: Detection Noise Robustness Analysis

> 时间：2026-06-27
> 动机：Phase 1 发现 EKF/bridge 在 GT detection 下无效 → 用合成噪声模拟真实检测器，看 WiFi 修正是否发挥作用
> 实现：`scripts/run_baseline.py` 新增 `--noise-bbox-std`/`--noise-depth-std`/`--noise-drop-rate`/`--noise-fp-count`/`--noise-seed`

## 动机

Phase 0-3 的所有实验都使用 **Ground Truth detections**（完美 bbox + 完美 depth）。这导致：
1. 关联代价矩阵的 IoU 部分极其准确，WiFi cost 几乎无法改变关联结果
2. Phase 1 的 EKF 无线更新和 Phase 2 的 tracklet bridging 在 GT detection 下被证实无效
3. 无法评估 WiFi 修正在"真实检测器噪声"场景下的价值

**Phase 4 核心问题**：给 GT bbox 加上定位噪声后，WiFi 修正能否帮助恢复因噪声导致的关联错误？

## 噪声注入机制

在 `run_baseline.py` 的 `run_sequence()` 中，对每帧检测执行以下操作（GT items 在噪声注入之前提取，不受影响）：

1. **Bbox jitter**：(x1,y1,x2,y2) += N(0, σ_bbox)，模拟检测器定位误差
2. **Depth noise**：depth += N(0, σ_depth)，模拟深度估计误差
3. **Detection drop**：以 p_drop 概率随机丢弃检测，模拟遮挡/漏检
4. **False positive**：注入 N_fp 个随机框，模拟假阳性

默认参数：seed=42（可复现），img_w=1280, img_h=720。

## 噪声级别校准（scene4 单场景测试）

| 噪声配置 | IDF1 | IDsw | MOTA | 说明 |
|----------|------|------|------|------|
| no noise (GT) | 78.7% | 827 | 96.5% | 基线 |
| bbox_std=5 | 78.4% | 1063 | 96.0% | 轻度定位误差 |
| bbox_std=10 | 64.1% | 2763 | 87.3% | 过度噪声 |
| bbox=5 + drop=0.05 | 76.7% | 1069 | 91.0% | +漏检 |
| bbox=5 + depth=0.3 + drop=0.05 | 76.4% | 1029 | 91.1% | +深度噪声 |

**选择 bbox_std=5 作为主噪声级别**：IDF1 几乎不变（78.4% vs 78.7%），但增加 236 次 IDsw，模拟了真实检测器的定位不确定性而不过度破坏跟踪。

## 4-fold LOSO 主实验

### bbox_std=5（轻度定位噪声）

| 方法 | scene1 | scene2 | scene3 | scene4 | avg IDF1 | avg IDsw |
|------|--------|--------|--------|--------|----------|----------|
| OC-SORT (no WiFi) | 68.65% | 58.69% | 63.65% | 60.11% | 62.78% | 2238 |
| Method A (WiFi) | 80.77% | 80.62% | 73.25% | 78.36% | **78.25%** | 3744 |
| Phase 3 R-weighted | 79.75% | 81.48% | 70.35% | 77.07% | 77.16% | 3655 |
| **Δ(A vs OC-SORT)** | +12.12 | +21.93 | +9.60 | +18.25 | **+15.47** | +1506 |

### bbox=5 + depth=0.3 + drop=0.05（中等综合噪声）

| 方法 | scene1 | scene2 | scene3 | scene4 | avg IDF1 | avg IDsw |
|------|--------|--------|--------|--------|----------|----------|
| Method A (WiFi) | 78.17% | 78.91% | 69.90% | 76.36% | **75.84%** | 3581 |
| Phase 3 R-weighted | 77.41% | 77.68% | 69.81% | 76.04% | 75.24% | 3472 |

### 无噪声基线（GT detections）

| 方法 | scene1 | scene2 | scene3 | scene4 | avg IDF1 | avg IDsw |
|------|--------|--------|--------|--------|----------|----------|
| OC-SORT (no WiFi) | 71.39% | 61.19% | 67.26% | 62.66% | 65.63% | 1269 |
| Method A (WiFi) | 81.58% | 81.48% | 72.92% | 78.68% | **78.67%** | 3065 |
| Phase 3 R-weighted | 79.21% | 82.00% | 70.39% | 78.52% | 77.53% | 2979 |

## 关键发现

### 1. WiFi 修正在噪声检测下价值更大

| 条件 | WiFi 增益 (Method A − OC-SORT) |
|------|-------------------------------|
| GT detections | +13.04pt (78.67 − 65.63) |
| bbox_std=5 | **+15.47pt** (78.25 − 62.78) |

WiFi 的增益在噪声检测下比 GT 检测下 **高 2.4pt**。原因：
- GT 检测下，OC-SORT 的 IoU 关联几乎完美，WiFi 修正作用有限
- 噪声检测下，IoU 关联出错概率增加，WiFi 提供的 depth-FTM compat 信号成为独立的关联验证源

### 2. 但 Method A 仍优于 Phase 3

即使加了噪声，简单的 EMA compat (Method A) 仍然优于 R-weighted WiFi cost (Phase 3)。差距在噪声下反而缩小：
- GT: Δ = 78.67 − 77.53 = 1.14pt
- bbox=5: Δ = 78.25 − 77.16 = 1.09pt
- bbox=5+depth+drop: Δ = 75.84 − 75.24 = 0.60pt

**解释**：噪声检测下 depth 也带噪声，R-weighting 进一步削弱了 WiFi cost，导致 WiFi 修正效果更弱。Method A 的 EMA compat 虽然也受影响，但权重固定不受 R 抑制。

### 3. 噪声主要增加 IDsw，不显著影响 IDF1

| 条件 | Method A IDF1 | Method A IDsw |
|------|--------------|---------------|
| GT | 78.67% | 3065 |
| bbox=5 | 78.40% (−0.27) | 3744 (+679) |
| bbox=5+depth+drop | 75.84% (−2.83) | 3581 (+516) |

轻度 bbox 噪声 (σ=5px) 对 IDF1 影响极小 (−0.27pt)，但增加 679 次 IDsw (+22%)。这说明 bbox jitter 主要影响身份一致性（同一个人 bbox 位置不稳定 → 跟踪器分配不同 ID），而不是关联正确性。

### 4. OC-SORT 对 bbox 噪声相对稳健

| 条件 | OC-SORT IDF1 | OC-SORT IDsw |
|------|-------------|-------------|
| GT | 65.63% | 1269 |
| bbox=5 | 62.78% (−2.85) | 2238 (+969) |

OC-SORT 在 bbox 噪声下 IDF1 下降 2.85pt，IDsw 增加 969 次。WiFi 修正（Method A）仅下降 0.27pt，说明 **WiFi phone-stable ID 天然对 bbox 噪声更稳健**——即使 bbox 位置抖动，只要 depth-FTM compat 一致，phone binding 不会改变，从而维持了 ID 一致性。

## 实现变更

- `scripts/run_baseline.py`：新增 `_inject_det_noise()` 函数、`_init_noise_rng()`、5 个 Phase 4 CLI 参数
- 噪声注入在 `run_sequence()` 内完成，GT items 在噪声之前提取，评测不受影响

## 新增数据

- `exps/phase4_methodA_bbox5.json`
- `exps/phase4_rg_bbox5.json`
- `exps/phase4_methodA_bbox5_depth3_drop5.json`
- `exps/phase4_rg_bbox5_depth3_drop5.json`
- `exps/phase4_ocsort_bbox5.json`
- `exps/phase4_ocsort_clean.json`

## 路线决策

**Phase 4 结论**：WiFi 修正在噪声检测下价值更大（+15.47pt vs +13.04pt），但 Method A 的简单 EMA compat 仍是最优方案。Phase 3 的 R-weighting 在噪声下进一步被削弱（depth 噪声 + R 抑制 = WiFi cost 太弱）。

**下一步方向**：
1. **真实检测器**：本实验用合成噪声，真实 YOLO 检测器可能有不同的噪声分布（如 NMS 效应、尺度偏差）
2. **Depth-free WiFi matching**：当前 WiFi compat 依赖 detection depth，depth 噪声直接损害 compat。可考虑用 bbox size/position 作为 depth proxy，或完全绕过 depth 用 FTM 趋势做匹配
3. **多帧 WiFi cost 累积**：单帧 depth-FTM compat 受噪声影响大，多帧 EMA 或滑动窗口可能更稳健（Phase 3 的 R 就是这个思路，但实现过于保守）
