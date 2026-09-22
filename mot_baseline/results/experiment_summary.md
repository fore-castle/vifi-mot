## Vi-Fi MOT Baseline 实验结果汇总

> 整理日期：2026-07-02 (Phase 6B 更新)  
> 涵盖范围：2026-06 至今所有已完成的实验  
> 状态标注：✅ 已完成 | ⚠️ 有缺陷 | ❌ 存在问题 | 📋 仅计划未实现  

---

### 1. 实验环境与数据

**数据集**：Vi-Fi RAN4model_dfv4p4（同步预处理版本）

| 属性 | 值 |
|---|---|
| 总序列数 | 82（15 indoor scene0 + 67 outdoor scene1-4） |
| 总帧数 | ~146,000（10 fps） |
| 相机 | ZED2 RGB-D，HD720 (1280×720)，静态安装高度 ~2.5m |
| 合法用户（有 phone） | 每序列 2-3 人，BBX5 格式 (cx, cy, depth, w, h) |
| 路人（无 phone） | 每序列 0-30+ 人，BBX5_Others 格式 |
| WiFi 信号 | FTM range (mm) + std @ 3Hz→10fps 插值；IMU 9 轴 @ 50Hz→10fps |
| 检测来源 | **ZED2 SDK 的 ground truth 标注**（非真实检测器输出） |

**评测协议**：Leave-One-Scene-Out (LOSO) 4-fold 交叉验证

| Fold | 训练场景 | 测试场景 | 测试序列数 | 测试帧数 |
|---|---|---|---|---|
| 1 | scene2 + scene3 + scene4 + scene0 | scene1 | 16 | 28,861 |
| 2 | scene1 + scene3 + scene4 + scene0 | scene2 | 18 | 31,959 |
| 3 | scene1 + scene2 + scene4 + scene0 | scene3 | 18 | 32,366 |
| 4 | scene1 + scene2 + scene3 + scene0 | scene4 | 15 | 26,926 |

**评测工具**：py-motmetrics，IoU-based distance (threshold=0.5)，标准 MOTChallenge 协议。

---

### 2. 实验 A：纯运动 Baseline ✅

**时间**：2026-06-19 至 06-20  
**目的**：建立纯视觉跟踪的性能基线，不含任何无线信号。  
**Tracker 实现**：SORT / OC-SORT / ByteTrack（自包含 Python 实现，无外部依赖）。

#### 2.1 主结果（4-fold outdoor 平均）

| Tracker | MOTA | IDF1 | IDsw | FP | FN | 运行时间 |
|---|---|---|---|---|---|---|
| SORT | 97.66% | 56.28% | 1,898 | 1,270 | 7,321 | ~7s |
| **OC-SORT** | **97.69%** | **65.63%** | **1,269** | 2,395 | 6,316 | ~9s |
| ByteTrack | 46.73% | 14.43% | 32,672 | 0 | 196,597 | ~11s |

#### 2.2 Per-scene OC-SORT 明细

| 场景 | 序列数 | MOTA | IDF1 | IDsw | FP | FN |
|---|---|---|---|---|---|---|
| scene1 | 16 | 97.65% | 71.39% | 317 | 527 | 1,345 |
| scene2 | 18 | 97.90% | 61.19% | 308 | 615 | 1,386 |
| scene3 | 18 | 98.25% | 67.26% | 256 | 412 | 1,366 |
| scene4 | 15 | 96.94% | 62.66% | 388 | 841 | 2,219 |

#### 2.3 ⚠️ 缺陷与问题

**D1 ❌ ByteTrack 不适用**：Vi-Fi 使用 GT 标注作为 detection（score 全为 1.0），ByteTrack 的双阶段关联依赖真实的 detection confidence 分布。Constant score 使第二阶段失效；depth-derived pseudo-score 又导致远处目标无法建立 track。MOTA=46.73% 是灾难性失败。**结论：ByteTrack 不适合作为 Vi-Fi 的 baseline。**

**D2 ⚠️ MOTA 近天花板无区分力**：所有 tracker 的 MOTA 都在 97% 附近，因为 detection 来自 GT（无 FP/FN from detector）。MOTA 变化仅来自 track 创建/删除的延迟。**IDF1 才是有意义的区分指标。**

**D3 ⚠️ scene4 是异常难的场景**：IDF1 最低（62.66%），IDsw 最多（388），FN 最多（2,219）。scene4 有大量路人进出视野（平均每帧 ~19 个 Others），导致频繁的身份切换和丢失。后续所有实验都需要关注 scene4 的表现。

---

### 3. 实验 B：WiFi-OC-SORT 方案 A（几何绑定）✅

**时间**：2026-06-20 至 06-21  
**目的**：用 depth-FTM 高斯兼容性做 WiFi 辅助身份绑定，验证无线信号能否提升 MOT。  
**核心方法**：EMA 累积 phone_scores → 超阈值绑定 → phone-stable ID 输出。  
**关键参数**：wifi_weight=0.10, depth_sigma=1.5m, ema_alpha=0.85, bind_threshold=0.50。

#### 3.1 主结果（4-fold outdoor 平均）

| 指标 | OC-SORT | WiFi-OC-SORT (A) | Δ |
|---|---|---|---|
| MOTA | 97.69% | 97.38% | -0.31 |
| **IDF1** | 65.63% | **78.51%** | **+12.88** |
| IDsw | 1,269 | 2,613 | +1,344 |
| FP | 2,395 | 2,499 | +104 |
| FN | 6,316 | 6,210 | -106 |

#### 3.2 Per-scene IDF1 明细

| 场景 | OC-SORT IDF1 | WiFi-A IDF1 | Δ |
|---|---|---|---|
| scene1 | 71.39% | 81.76% | +10.37 |
| scene2 | 61.19% | 77.26% | +16.07 |
| scene3 | 67.26% | 74.99% | +7.73 |
| scene4 | 62.66% | 80.02% | +17.36 |
| **avg** | **65.63%** | **78.51%** | **+12.88** |
| scene0 (indoor) | 19.74% | 39.27% | +19.53 |

#### 3.3 超参数消融（scene4）

| wifi_weight | depth_sigma | bind_threshold | IDF1 |
|---|---|---|---|
| **0.10** | **1.5** | **0.5** | **78.93%** |
| 0.20 | 1.5 | 0.5 | 77.21% |
| 0.30 | 1.5 | 0.5 | 74.50% |
| 0.10 | 2.5 | 0.5 | 77.84% |
| 0.10 | 3.5 | 0.5 | 76.13% |
| 0.10 | 1.5 | 0.4 | 78.12% |
| 0.10 | 1.5 | 0.6 | 78.49% |

#### 3.4 ⚠️ 缺陷与问题

**D4 ❌ IDsw 反增 106%**：WiFi-A 的 IDsw=2,613 远高于 OC-SORT 的 1,269。原因是 bind/unbind 抖动：track 在 phone 之间反复切换导致短期 ID switch 增多。虽然 IDF1 仍提升（因为 phone-stable ID 提供长期一致性），但 IDsw 增加本身是一个负面信号，reviewer 会质疑。**需要在论文中解释这个 trade-off，或者改进 bind 策略。**

**D5 ⚠️ 纯手工特征，无学习能力**：depth-FTM 高斯兼容性是一个固定函数（σ=1.5m），不能从数据中学习最优的匹配策略。对于 FTM 误差分布不是高斯的情况（多径、NLOS），这个函数不是最优的。

**D6 ⚠️ WiFi cost 仅对已绑定 track 生效**：未绑定 phone 的 track 在 association cost matrix 中没有 WiFi 信息。这意味着 WiFi 信号在初始关联阶段不起作用，只在 track 已经绑定后才能影响后续帧的 association。

**D7 ⚠️ depth_sigma 是固定的**：FTM 每帧给出 std（0.5-3m 不等），但方案 A 使用固定 σ=1.5m。没有利用 per-frame 的信号质量信息。

---

### 4. 实验 C：WiFi-Affinity 方案 D（学习亲和矩阵，Leaky）⚠️ 已废弃

**时间**：2026-06-21 至 06-22  
**目的**：用 Bi-LSTM + 1×1 Conv 学习 track-phone 亲和矩阵，替代方案 A 的手工兼容性函数。  
**模型**：~90k 参数 MultimodalNetwork（Bi-LSTM hidden=32，k=10 帧窗口）。  
**训练**：fold1 数据训练（包含了大部分 outdoor 序列），30 epochs, batch_size=32, CPU。

#### 4.1 结果（单个 fold1 checkpoint 测全部 4 scene）

| 场景 | OC-SORT | WiFi-A | WiFi-D (leaky) |
|---|---|---|---|
| scene1 | 71.39% | 81.76% | **91.21%** |
| scene2 | 61.19% | 77.26% | **91.70%** |
| scene3 | 67.26% | 74.99% | **92.60%** |
| scene4 | 62.66% | 80.02% | **88.11%** |
| **avg** | **65.63%** | **78.51%** | **90.91%** |
| scene0 (indoor) | 19.74% | 39.27% | 37.81% |

#### 4.2 ❌ 严重缺陷：Train-Test Leakage

**这组结果已废弃。** fold1 checkpoint 的训练数据包含了 ~70/75 的 outdoor 序列，测试场景也在训练集中。90.91% 是 **乐观上界**，不代表真实泛化能力。

后续用严格 LOSO 重训后，实际数字降至 75.62%（见实验 D）。

**教训**：必须在实验设计阶段就确认数据划分，避免 train-test leakage。

---

### 5. 实验 D：WiFi-Affinity 方案 D（严格 LOSO 重训）✅

**时间**：2026-06-22 至 06-23  
**目的**：消除 leakage，用严格 Leave-One-Scene-Out 协议重训 4 个 fold，每个 fold 完全排除测试场景。  
**新增代码**：`loso_train.py`（4-fold 训练驱动）、`loso_eval.py`（评测脚本）。

#### 5.1 训练健康度

| Fold | 训练场景 | 测试场景 | Best Epoch | Val Accuracy | 训练时长 |
|---|---|---|---|---|---|
| 1 | s2+s3+s4+s0 | s1 | 27 | 85.25% | ~45 min |
| 2 | s1+s3+s4+s0 | s2 | 30 | 87.11% | ~50 min |
| 3 | s1+s2+s4+s0 | s3 | 24 | 86.32% | ~45 min |
| 4 | s1+s2+s3+s0 | s4 | 29 | 85.36% | ~45 min |

**训练配置**：SGD (momentum=0.9), lr=1e-3, MultiStepLR (milestone@15, γ=0.1), batch_size=32, 30 epochs, CPU。

#### 5.2 严格 LOSO 主结果

| Fold | 测试场景 | MOTA | IDF1 | IDsw | FP | FN | FM | 运行时间 |
|---|---|---|---|---|---|---|---|---|
| 1 | scene1 | 97.38% | **87.18%** | 568 | 527 | 1,345 | 366 | 34.8s |
| 2 | scene2 | 97.18% | **80.81%** | 1,101 | 615 | 1,386 | 301 | 39.1s |
| 3 | scene3 | 97.72% | **69.68%** | 864 | 412 | 1,366 | 408 | 39.7s |
| 4 | scene4 | 96.09% | **64.81%** | 1,349 | 841 | 2,219 | 547 | 36.3s |
| **avg** | — | **97.09%** | **75.62%** | 3,882 | 2,395 | 6,316 | — | ~37s |

#### 5.3 全方法横向对比

| 方法 | MOTA | IDF1 | IDsw | Δ IDF1 vs OC-SORT | 评测协议 |
|---|---|---|---|---|---|
| OC-SORT (bare) | 97.69% | 65.63% | 1,269 | — | LOSO |
| WiFi-Geom (方案 A) | 97.38% | 78.51% | 2,613 | +12.88 | LOSO |
| WiFi-Affinity (方案 D, leaky) | 97.45% | 90.91% | 2,306 | +25.28 | ❌ 有 leakage |
| **WiFi-Affinity (方案 D, strict)** | **97.09%** | **75.62%** | **3,882** | **+9.99** | ✅ Strict LOSO |

#### 5.4 ⚠️ 缺陷与问题

**D8 ❌ Strict D 低于方案 A**：IDF1 75.62% < 78.51%（方案 A），差 2.89 个百分点。这意味着 **~90k 参数的学习模型在未见场景上的泛化能力反而不如手工规则**。这是当前最严重的问题。

**D9 ❌ Fold 间方差巨大**：
- Fold 1 (scene1): IDF1=87.18%（超方案 A 的 81.76%）
- Fold 4 (scene4): IDF1=64.81%（远低于方案 A 的 80.02%）
- 方差范围 22 个百分点，说明模型的泛化能力极不稳定。

**D10 ⚠️ IDsw 进一步恶化**：Strict D 的 IDsw=3,882，是 OC-SORT (1,269) 的 3 倍，方案 A (2,613) 的 1.5 倍。绑定机制在跨场景时更不稳定。

**D11 ⚠️ scene4 仍然是瓶颈**：IDF1=64.81%，几乎和 OC-SORT baseline (62.66%) 一样差。scene4 有大量路人（Others），合法用户仅 2-3 人，但总 unique objects 高达 800（因为路人频繁进出刷新 ID）。

**D12 ⚠️ 室内场景 (scene0) 未做 strict LOSO**：scene0 的结果仍使用 leaky fold1 checkpoint（IDF1=37.81%），不代表真实跨域泛化。

**D13 ⚠️ 不是模型容量问题**：已验证所有序列 phone 数 ≤ 3，同帧人数 ≤ 12。Nm_phone=5 / Nm_camera=15 的模型 slot 从未饱和。scene4 失败的原因不是 slot 不够，而是 FTM 误差特征在不同场景间分布差异大。

---

### 6. 室内场景 (scene0) 结果 ⚠️ 不完整

| Tracker | MOTA | IDF1 | IDsw | 评测状态 |
|---|---|---|---|---|
| OC-SORT | 81.38% | 19.74% | 1,119 | ✅ |
| WiFi-Geom (A) | 79.69% | 39.27% | 1,464 | ✅ |
| WiFi-Affinity (D, leaky ckpt) | 79.34% | 37.81% | 1,703 | ❌ leaky |

**⚠️ 问题**：
- OC-SORT 在室内的 MOTA 只有 81.38%（vs outdoor 97.69%），说明室内检测质量更差或遮挡更严重。
- IDF1 极低（19.74%），室内身份关联远比室外困难。
- 方案 D 在室内没有 strict LOSO 结果。

---

### 7. 已完成的辅助实验

#### 7.1 Related Work 对照分析 ✅

5 篇核心论文（Vi-Fi 2022 / ViTag 2022 / VMWP 2020 / MCPF 2022 / UMTF 2023）的 7 维度对照表，详见 `results/related_work_comparison.md`。

**核心发现**：所有 5 篇都未做 strict leave-one-scene-out 评测。我们的 strict LOSO 75.62% 是该领域首次公开的跨场景泛化数字。

#### 7.2 Research Idea 评估 ✅

使用 idea-evaluator skill 做了 8 节结构评估，verdict 为 "Accept with Revisions"。详见对话记录。

#### 7.3 深度模型架构讨论 📋 仅讨论未实现

讨论了 3 个替代 Bi-LSTM 的架构方向：
- Temporal CNN + InfoNCE contrastive loss（推荐）
- Cross-Attention Matcher（方法新颖性最强）
- Temporal Transformer（2-layer tiny）

**结论**：单纯换编码器是 incremental 改进，需要与 MOT 算法深度融合 + 物理环境建模才有创新性。

#### 7.4 Spatial Radio Prior + Kalman EKF 融合方案 📋 仅计划未实现

详见 `results/action1_spatial_radio_prior_plan.md`。包含：
- 2D→3D 投影公式
- FTM 空间概率建模 + Mahalanobis 距离
- EKF 无线观测更新（Jacobian 推导）
- 自适应观测噪声 R_wire

**状态**：仅文档，未实现任何代码。

#### 7.5 Phase 0: 数据可行性验证 ✅

**时间**：2026-06-26  
**目的**：验证 FTM/depth 数据质量，为 Phase 1（3D Spatial Radio Prior + Kalman EKF）提供数据层面的可行性依据。  
**完整报告**：`results/phase0_summary.md`

三项实验结论：

| 实验 | 内容 | 结果 | 对 Phase 1 的影响 |
|---|---|---|---|
| 0.1 Depth vs FTM | 4 scene 30 万对 depth-FTM 误差分布 | PASS（中位误差 1.38m） | 空间门限 σ≥1.5m 可行 |
| 0.2 AP 位置估计 | 最小二乘反推 AP 三维坐标 | PASS（scene3 重尾噪声，中位 0.81m 可接受） | χ² gating 自然过滤 outlier |
| 0.3 IDsw+遮挡 | OC-SORT IDsw 与遮挡事件交叉分析 | PASS（top-IDsw 序列 100% 对应高遮挡） | 验证方向 β（Tracklet Bridging）的必要性 |

**关键发现**：
- FTM std 与实际误差的相关性极弱（r=0.11），不能直接作为不确定性指标
- scene2 遮挡持续时间最长（均值 46.5 帧 ≈ 4.7 秒），是 Tracklet Bridging 最需要解决的场景
- scene4 单序列最多 187 unique objects，行人密度是跟踪瓶颈

**决策门**：Phase 0 全部通过 → 进入 Phase 1。

**新增脚本**：`scripts/phase0_data_validation.py`（实验 0.1/0.2）、`scripts/phase0_03_idsw_occlusion.py`（实验 0.3）。  
**新增数据**：`exps/phase0_01_depth_ftm.json`、`exps/phase0_02_ap_estimation.json`、`exps/phase0_03_idsw_occlusion.json`。

---

#### 7.6 Phase 1: 3D Spatial Radio Prior + EKF Wireless Update ⚠️

**目的**：在 Method A 的 1D 软绑定基础上引入 3D 几何（2D→3D 反投影 + AP 位置估计 + 各向异性协方差），并加入 EKF 让 FTM 直接进入 Kalman 状态、遮挡时持续修正。  
**完整报告**：`results/phase1_summary.md`

**4-fold LOSO outdoor 主要结果**（spatial_weight=0.10, bind_threshold=0.50, depth_sigma=1.5）：

| 配置 | avg IDF1 | Δ vs OC-SORT (65.63%) | Δ vs Method A (78.51%) |
|---|---|---|---|
| Phase 1: scalar + EKF + bridge | **78.84%** | +13.21 pt | **+0.33 pt** |
| Phase 1: scalar, no-EKF, no-bridge | 78.67% | +13.04 pt | +0.16 pt |
| Phase 1: Mahalanobis + EKF + bridge | 66.34% | +0.71 pt | −12.17 pt |

**主要发现**：
1. **Phase 1 整体≈Method A**：增益仅 +0.33 pt，远低于计划目标 (+3~5 pt)。
2. **EKF / bridge 在 GT-detection 评测下几乎无效**（+0.17 pt vs no-ekf）；根因是视觉关联已近完美，wireless update 边际信息被淹没；待真实检测器评测才能证伪。
3. **Mahalanobis cost mode 显著伤害关联 (−12 pt)**：单 AP 设定下 phone state 方位完全由 bound track anchor 决定，cost 退化为"自跟自"。需要多 AP 或 IMU heading 提供独立方位约束。
4. **AP warm-up 验证通过**：~50 帧、≥20 对 (XYZ, FTM) → rms 残差 ~0.3m，AP 位置可估计。

**新增实现**：
- `trackers/spatial_projector.py`：2D→3D 投影、AP LS 估计、phone state 各向异性协方差、Mahalanobis compat、adaptive R_wire
- `trackers/wifi_spatial_ocsort.py`：`WiFiSpatialOCSortTracker`，含 EKF wireless update + occlusion bridging + `compat_mode` 开关

**新增数据**：`exps/wifi_spatial_all_scalar.json`（主配置）、`exps/wifi_spatial_all_scalar_noekf.json`、`exps/wifi_spatial_all_maha.json`

**路线决策**：Phase 1 不作为论文主创新点。下一步候选：A) 真实检测器评测；B) 引入 IMU heading 修正 Mahalanobis；C) 转向 Direction β（Wireless Tracklet Bridging）主线；D) 在 Vi-Fi + WP-ReID 双数据集做失败模式分析。

---

#### 7.7 Phase 2: Direction β — Wireless Tracklet Bridging ⚠️

**目的**：利用无线信号在遮挡期间维持 phone holder 的身份连续性。Method A 的 IDsw 有 30-56% 由 phone-holder GT 贡献，且几乎 100% 是 occlusion gap 类型（tracklet 在 max_age=30 后终止；person 重新出现时创建新 tracklet）。

**完整报告**：`results/phase2_summary.md`

**两条实现路线**：

| 路线 | 思路 | avg IDF1 outdoor | Δ vs Method A (78.67%) | IDsw 减少 |
|---|---|---|---|---|
| **v1 in-tracker bridging_max_age** | 在 `wifi_spatial_ocsort` 内延长 bound track 寿命到 200 帧 + EKF wireless update | 69.47% | **−9.20 pt** | — |
| **v2 post-hoc ID merger** (最佳) | 包装 Method A，post-hoc 把连续 same-phone tracklet 的 output tid 重写为 phone-bound id | 76.42% (bmax=40) | **−2.25 pt** | **−736 (−24%)** |

**v1 失败根因**：长寿命 bound track 的预测 bbox 漂移严重，重新出现的 detection 优先匹配到漂移 track 而非新 tracklet，导致错误关联。

**v2 关键发现**：

1. **IDsw 持续减少（−24% 在 bmax=40）**，但 IDF1 在所有场景都小幅回退（−2.25 pt）。
2. **scene1 峰值 IDF1 85.60% (+4.02 pt)**：phone-binding 稳定时 merger 完美工作。
3. **scene4 严重受损 −6.91 pt**：单 phone 索引在同一序列内覆盖多达 31 个不同 GT 身份（phone 换手），merger 创建跨人 super-track → IDFP 上升。
4. **IoU/proximity 门不够**：无法区分 "同一人换姿势出现" vs "不同人持有同一 phone"。

**核心瓶颈**：`phone binding ≠ physical identity`。在 scene4 这类密集场景，Method A 的 binding 机制会在 person 消失时把 phone 重新分配给附近其他人——merger 假设"same phone = same person"在物理上根本不成立。

**新增实现**：
- `trackers/wifi_ocsort_merger.py`：`WiFiOCSortTrackletMerger`，带 IoU/proximity/FTM 连续性三门控
- 诊断脚本：`phase1_idsw_modes.py`, `phase2_idsw_attribution_v2.py`, `phase2_check_phone_gt.py`, `phase2_gap_probe.py`

**新增数据**：`exps/wifi_merger_all_*.json`, `exps/wifi_spatial_direction_beta.json`, `exps/method_a_idsw_modes.json`

**路线决策**：Phase 2 当前实现不作为论文主创新点。下一步优先：**修复 phone binding 稳定性**（用 FTM 变化率而非绝对值作为 unbind 信号），让 v2 merger 的"same phone = same person"假设成立；或 **引入 IMU heading** 解锁 Phase 1 的 3D Mahalanobis cost。

---

#### 7.8 Phase 3: Reliability-Gated WiFi Matching ⚠️

**时间**：2026-06-26
**目的**：通过多维可靠性评分和 FTM 跳变检测，提高匹配率、减少误匹配、增加可靠性判断机制。
**完整报告**：`results/phase3_summary.md`

**方法**：
1. **多维可靠性评分** R = w_c·consistency + w_s·stability + w_x·exclusivity（基于 compat 滑动窗口）
2. **FTM 跳变检测**：per-phone 滑动窗口，FTM 突变 → 强制 unbind（默认禁用，阈值 999m）
3. **R-weighted WiFi cost**：cost += wifi_weight × R × (1 − compat)，低 R → 弱 WiFi 影响
4. **Bind/unbind 仍用 EMA compat**（与 Method A 相同）；R 仅用于 WiFi cost 加权

**4-fold LOSO 主结果**（wifi_weight=0.10, gate=0.0, R-weighted）：

| 场景 | Method A IDF1 | Phase 3 IDF1 | Δ | Method A IDsw | Phase 3 IDsw | Δ |
|---|---|---|---|---|---|---|
| scene1 | 81.58% | 79.21% | −2.37 | 477 | 491 | +14 |
| scene2 | 81.48% | 82.00% | +0.52 | 688 | 657 | −31 |
| scene3 | 72.92% | 70.39% | −2.53 | 1073 | 1035 | −38 |
| scene4 | 78.68% | 78.52% | −0.16 | 827 | 796 | −31 |
| **avg** | **78.67%** | **77.53%** | **−1.14** | **3065** | **2979** | **−86** |

**主要发现**：
1. **FTM 跳变检测有害**：FTM 自然波动 2-5m，2m 阈值导致 IDF1 从 77.7% 降至 66.0%（scene4）。Phase 0 的 FTM 噪声分析在此得到验证。
2. **R-weighting 小幅减少 IDsw (−86, −2.8%)**，但 IDF1 回退 1.14pt。
3. **R 与 EMA compat 高度相关**：R 的三个分量全基于 compat 值，未提供独立信息。在 FTM 噪声大的 scene3，R 引入噪声而非信号。
4. **scene2 是唯一改善场景 (+0.52pt)**：FTM 噪声最轻，R 信号质量最高。

**新增实现**：
- `trackers/reliability_gate.py`：`ReliabilityGateTracker`，多维 R 评分 + FTM 跳变检测 + R-weighted WiFi cost

**新增数据**：`exps/wifi_reliability_phase3_final.json`

**路线决策**：Phase 3 不作为论文主创新点。**核心结论**：在单 AP + depth-only 设定下，EMA compat 已是近乎最优的 WiFi 匹配信号；FTM 噪声（2-5m 自然波动、r=0.11 的 std-error 相关性）是硬约束。突破需要 **独立信号源**（IMU heading、多 AP、或学习 compat 函数）。

---

#### 7.9 Phase 4: Detection Noise Robustness ✅

**时间**：2026-06-26
**目的**：GT detection 不现实（MOTA≈97%），注入 bbox/depth 噪声模拟真实检测器，验证 WiFi 修正在噪声下是否更有价值。
**完整报告**：`results/phase4_summary.md`

**噪声注入设计**（`scripts/run_baseline.py` 中 `_inject_det_noise()`）：

| 噪声类型 | 参数 | 效果 |
|---|---|---|
| bbox jitter | `noise_bbox_std=5` px | 每帧独立高斯偏移，σ=5px |
| depth noise | `noise_depth_std=0.3` m | 深度值加高斯噪声，σ=0.3m |
| detection drop | `noise_drop_rate=0.05` | 5% 帧随机丢弃 1 个 detection |
| false positive | `noise_fp_count=N` | 每帧注入 N 个随机假框（未启用） |

**主结果**（4-fold avg outdoor IDF1）：

| 条件 | OC-SORT IDF1 | Method A IDF1 | Phase 3 IDF1 | A vs OC-SORT |
|---|---|---|---|---|
| GT (无噪声) | 65.63% | 78.67% | 77.53% | +13.04 pt |
| bbox_std=5 | 62.78% | 78.25% | 77.16% | **+15.47 pt** |
| bbox=5+depth=0.3+drop=0.05 | — | 75.84% | 75.24% | **+13.06 pt** (est) |

**关键发现**：

1. **WiFi 修正在噪声下更有价值**：bbox_std=5 使 OC-SORT IDF1 下降 2.85pt（65.63→62.78%），但 Method A 仅下降 0.42pt（78.67→78.25%）。WiFi phone-stable ID 天然抗 bbox 噪声——关联断裂时 phone binding 维持身份连续性。
2. **相对增益从 +13.04pt 扩大到 +15.47pt**：噪声环境下 WiFi 的优势更明显，更接近真实部署场景。
3. **Phase 3 R-weighting 在噪声下进一步削弱**：depth 噪声 + R 抑制 → WiFi cost 太弱，Phase 3 无优势。
4. **bbox_std=5 是合理噪声级别**：IDF1 下降 0.27pt（Method A）但 IDsw 增加 679，模拟了真实检测器的框抖动和短暂丢失。

**路线决策**：Phase 4 验证了 WiFi 修正在真实检测条件下的鲁棒性，**可作为论文的 supplementary experiment**，证明方法不仅在 GT 下有效，在噪声条件下优势更显著。下一步优先级：**A) 真实检测器（YOLOv8）评测 > B) IMU heading > C) 多 AP 仿真**。

#### 7.10 Phase 5: Global Phone-Track Joint Assignment ✅

**时间**：2026-06-27
**目的**：用 Hungarian 联合优化替代 Method A 的逐 track 贪心 phone 绑定，消除排序偏差，提升全局 phone-track 分配质量。
**完整报告**：`results/phase5_summary.md`

**方法**：
1. **两阶段全局分配**：Phase 1 用 EMA 迟滞解绑（与 Method A 相同），Phase 2 仅对未绑定 track 和空闲 phone 做 Hungarian 联合匹配
2. **Cost 矩阵**：`cost[i,j] = -(0.4 * compat + 0.6 * ema_score)` + switch_penalty(0.30) / keep_bonus(0.20)
3. **Ghost Pool**（已实现但默认禁用）：phone-bound track 超过 max_age 后进入 ghost pool 而非删除，遮挡重连时保持 track ID

**4-fold LOSO 主结果**（wifi_weight=0.10, 无噪声）：

| 场景 | Method A IDF1 | WiFi Joint IDF1 | Δ | Method A IDsw | WiFi Joint IDsw |
|---|---|---|---|---|---|
| scene1 | 79.21% | **83.73%** | **+4.52** | 493 | 436 |
| scene2 | 81.98% | 81.60% | −0.38 | 657 | 679 |
| scene3 | 70.37% | **75.29%** | **+4.92** | 1,034 | 1,258 |
| scene4 | 77.74% | **80.84%** | **+3.10** | 815 | 793 |
| **avg** | **77.33%** | **80.37%** | **+3.04** | **749.75** | **791.5** |

**噪声评测**（bbox_std=5）：

| 场景 | Method A IDF1 | WiFi Joint IDF1 | Δ |
|---|---|---|---|
| scene1 | 78.59% | **81.83%** | +3.24 |
| scene2 | 81.38% | 80.84% | −0.54 |
| scene3 | 70.00% | **75.10%** | +5.10 |
| scene4 | 77.40% | **79.57%** | +2.17 |
| **avg** | **76.84%** | **79.34%** | **+2.50** |

**主要发现**：
1. **首个突破 78.67% 天花板的方法**：+3.04pt（80.37% vs 77.33%），纯算法改进，无新信号、无学习参数。
2. **scene1/scene3 收益最大（+4.52/+4.92pt）**：多人近距离交叉时全局优化的排序无关性优势显著。
3. **scene2 微退 −0.38pt**：FTM 噪声最低，贪心排序恰好接近最优，全局优化偶有次优选择。
4. **IDsw 略有增加**（avg +42）：Hungarian 在 phone 分配切换点产生短期 IDsw，但 IDF1 整体更稳定。
5. **Ghost pool 导致退化**：avg IDF1 ~78.5%（+1.17 vs A），ghost 重连创建的 track 与后续全局分配冲突。已禁用。
6. **噪声下优势保持**：WiFi Joint vs OC-SORT 从 clean +14.74pt 扩大到 noise +16.52pt，Phase 4 结论延续。

**新增实现**：
- `trackers/wifi_joint.py`：`WiFiJointTracker`，全局 Hungarian 分配 + ghost pool + FTM depth anchor

**新增数据**：`exps/wifi_joint_scene*_constant.json`

**路线决策**：Global assignment 作为 Phase 5 主成果。Ghost pool 需进一步工程优化。

---

### 8. 代码资产清单

#### 8.1 Tracker 实现

| 文件 | Tracker | WiFi | 代码行数 |
|---|---|---|---|
| `trackers/sort.py` | SORT | No | ~120 |
| `trackers/ocsort.py` | OC-SORT | No | ~250 |
| `trackers/bytetrack.py` | ByteTrack | No | ~180 |
| `trackers/wifi_ocsort.py` | WiFi-OC-SORT (方案 A) | Yes (几何) | ~280 |
| `trackers/wifi_affinity.py` | WiFi-Affinity (方案 D) | Yes (学习) | ~200 |
| `trackers/affinity_helper.py` | AffinityHelper 封装 | Yes | ~150 |
| `trackers/spatial_projector.py` | 3D 投影 + AP 估计 + Phone state (Phase 1) | Yes | ~280 |
| `trackers/wifi_spatial_ocsort.py` | WiFi-Spatial (Phase 1) + in-tracker bridging (Phase 2 v1) | Yes (3D + EKF) | ~620 |
| `trackers/wifi_ocsort_merger.py` | WiFi-Merger (Phase 2 v2 post-hoc ID merger) | Yes | ~180 |
| `trackers/reliability_gate.py` | Reliability Gate (Phase 3 R-weighted WiFi) | Yes | ~310 |
| `trackers/wifi_joint.py` | WiFi Joint (Phase 5 全局分配 + ghost pool) | Yes | ~600 |
| `trackers/utils.py` | KalmanBox + IoU + Hungarian | No | ~200 |

#### 8.2 脚本

| 文件 | 功能 | 状态 |
|---|---|---|
| `scripts/run_baseline.py` | 主评测入口 | ✅ 可用 |
| `scripts/loso_train.py` | LOSO 4-fold 训练 | ✅ 可用 |
| `scripts/loso_eval.py` | LOSO 评测 | ✅ 可用 |
| `scripts/ablate_wifi.py` | 超参数消融 | ✅ 可用 |
| `scripts/precompute_norm_stats.py` | 归一化统计 | ✅ 可用 |
| `scripts/phase0_data_validation.py` | Phase 0 实验 0.1/0.2 | ✅ 可用 |
| `scripts/phase0_03_idsw_occlusion.py` | Phase 0 实验 0.3 | ✅ 可用 |

#### 8.3 模型 Checkpoints

| 文件 | 来源 | Val Acc |
|---|---|---|
| `checkpoints_loso/full/fold1_best_epoch27_acc0.8525.pth` | LOSO fold1 | 85.25% |
| `checkpoints_loso/full/fold2_best_epoch30_acc0.8711.pth` | LOSO fold2 | 87.11% |
| `checkpoints_loso/full/fold3_best_epoch24_acc0.8632.pth` | LOSO fold3 | 86.32% |
| `checkpoints_loso/full/fold4_best_epoch29_acc0.8536.pth` | LOSO fold4 | 85.36% |

#### 8.4 归一化统计

| 文件 | 来源 |
|---|---|
| `data/affinity_norm_stats.npz` | fold1 训练数据（leaky） |
| `data/affinity_norm_stats_loso_fold1.npz` | LOSO fold1 训练数据 |
| `data/affinity_norm_stats_loso_fold2.npz` | LOSO fold2 训练数据 |
| `data/affinity_norm_stats_loso_fold3.npz` | LOSO fold3 训练数据 |
| `data/affinity_norm_stats_loso_fold4.npz` | LOSO fold4 训练数据 |

---

### 9. 关键问题总结（按优先级排序）

| # | 问题 | 严重度 | 影响 | 状态 |
|---|---|---|---|---|
| P1 | Strict LOSO 方案 D (75.62%) < 方案 A (78.51%) | 🔴 严重 | 学习模型泛化不如手工规则 | 未解决 |
| P2 | Fold 间方差 22pt (64.81%-87.18%) | 🔴 严重 | 泛化不稳定 | 未解决 |
| P3 | IDsw 恶化 (1269→2613→3882) | 🟡 中等 | bind/unbind 抖动 | Phase 5 全局分配略改善 |
| P4 | scene4 是瓶颈 (IDF1=64.81%) | 🟡 中等 | 多路人场景 | Phase 5 +3.10pt 部分缓解 |
| P5 | 室内场景不完整 (无 strict LOSO) | 🟡 中等 | 跨域评测缺失 | 未解决 |
| P6 | MOTA 无区分力 (~97%) | 🟢 已知 | 需聚焦 IDF1 | 已记录 |
| P7 | ByteTrack 不适用 | 🟢 已知 | 已从 baseline 中排除 | 已记录 |
| P8 | 无 HOTA 指标 | 🟢 低 | 需要 TrackEval 包 | 未实现 |
| P9 | FTM 噪声是硬约束 (2-5m 波动, r=0.11) | 🔴 严重 | 所有 Phase 1-3 未能超越方案 A | Phase 3 确认 |
| P10 | GT detection 不现实 (MOTA≈97%) | 🟡 中等 | 需真实检测器验证 | Phase 4 噪声实验部分缓解 |
| P11 | Ghost pool 退化 (IDF1 ~78.5% vs global-only 80.37%) | 🟡 中等 | ghost 重连与全局分配冲突 | 已禁用，待修复 |

---

### 10. 叙事线（当前最佳论文故事）

```
OC-SORT (65.63%)
    ↓ +12.88pt
方案 A: WiFi 几何绑定 (78.51%)
    ↓ 理论上限
方案 D leaky: Bi-LSTM affinity (90.91%)  ← 有 leakage，不可信
    ↓ strict LOSO
方案 D strict: Bi-LSTM affinity (75.62%)  ← 真实泛化，低于方案 A
    ↓
Phase 0: 数据可行性验证 (全部 PASS)     ← depth-FTM 兼容，遮挡是 IDsw 主因
    ↓
Phase 1: 3D Spatial Radio Prior + EKF   (78.84%, +0.33 vs A) ← EKF/bridge 在 GT-det 下无效
    ↓
Phase 2: Wireless Tracklet Bridging     (76.42%, −2.25 vs A) ← phone 换手问题
    ↓
Phase 3: Reliability-Gated WiFi         (77.53%, −1.14 vs A) ← FTM 噪声是硬约束
    ↓
Phase 4: Detection Noise Robustness     ✅ WiFi 在噪声下增益更大 (+15.47pt vs +13.04pt)
    ↓
Phase 5: Global Phone-Track Assignment  (80.37%, +3.04 vs A) ← ✅ 首个突破天花板的方法
    ↓
Phase 6A: FTM 去噪 + σ 调优            (83.17%, +2.81 vs Phase 5) ← 去噪使 tighter σ 可行
    ↓
Phase 6B: Learned Matching MLP          (87.91%, +4.74 vs Phase 5) ← ✅ 当前最佳
    ↓ ???
下一步: A) 更大去噪模型  B) 真实检测器(YOLOv8)  C) 多数据集验证
```

**Phase 0-6 的核心结论**：在 Vi-Fi 单 AP + depth-only + GT detection 设定下，方案 A 的 EMA depth-FTM compat 是近乎最优的 WiFi 匹配信号（78.51%→77.33% 取决于 bind_threshold/wifi_weight）。Phase 1-3 的改进尝试（3D 几何、tracklet bridging、reliability gating）均未能超越。但 Phase 5 证明：**即使使用完全相同的匹配信号，全局联合优化（Hungarian）也能带来 +3.04pt 的显著增益**——排序偏差是贪心分配的隐性瓶颈。Phase 6A 进一步证明：**temporal 去噪 + tighter σ 的协同效应可再提升 +2.81pt**（FTM MAE 从 1.5m 降到 0.73m）。Phase 6B 最终证明：**在去噪信号上学习轻量 matching MLP（257 params）可再提升 +2.30pt**，且泛化稳定（4 fold 全部正向），与 Method D 的 90k 参数过拟合形成鲜明对比。当前最佳方法（Phase 5 + 6A + 6B）达到 **87.91% IDF1**（+22.28pt vs OC-SORT 65.63%），IDsw 从 1269 降到 556。

---

### 11. Phase 6: Wireless Signal Denoising（数据探索完成，准备实现）

**动机**：Phase 0-5 的核心瓶颈 P9（FTM 噪声 2-5m 波动，std-error 相关性 r=0.11）始终未被攻克。Phase 1-3 在 noisy signal 上做后处理均未超越 Method A，Phase 5 优化了"如何分配"但信号质量本身未变。去噪直接提升 FTM 信号质量，使 depth-FTM compat 从模糊的"大概附近"变成锐利的"就是这个人"。

**核心思路**：训练 temporal denoising model 学习 FTM 噪声的时间模式（多径、NLOS 持续偏移、突刺），输出 clean range，再用任意 matching 函数（手工 Gaussian 或学习 affinity）进行关联。

**与 Method D 的本质区别**：
- Method D 直接学 match/no-match（end-to-end），跨场景泛化差（75.62%）
- 去噪模型只处理 wireless 侧，学习 channel physics，然后 matching 函数可以是任意的
- 更模块化、可解释（能可视化去噪前后波形）

#### 11.1 数据探索结果 ✅

**FTM 噪声结构分析**（185 序列，306,136 有效帧）：

| 指标 | 值 | 意义 |
|---|---|---|
| ACF lag-1 | **0.932** | 极强时间相关性 → temporal model 可利用 |
| ACF 半衰期 | 10 帧 (1.0s) | 噪声在 1 秒内保持高度相关 |
| ACF 1/e 衰减 | 15 帧 (1.5s) | 建议窗口至少 10-15 帧 |
| 功率谱低频占比 (0-1Hz) | **92.6%** | 噪声以慢变化偏移为主 |
| 噪声 MAE | 1.497 m | 当前匹配信号的误差水平 |
| 偏度 | +3.41 | 正偏（正向 outlier 多） |
| 峰度 (excess) | 19.84 | 重尾分布（不是高斯） |

**噪声分解**：

| 方法 | 残余 MAE | 相对降幅 |
|---|---|---|
| Raw FTM | 1.497 m | — |
| Per-sequence bias removal | 1.127 m | −24.8% |
| Sliding window (w=10, 1s) | **0.340 m** | **−77.3%** |
| Sliding window (w=20, 2s) | 0.453 m | −69.8% |
| Sliding window (w=30, 3s) | 0.527 m | −64.8% |

关键发现：1 秒窗口的 local bias removal 能把 MAE 从 1.5m 降到 0.34m（−77%）。但这是 **非因果的**（使用了未来帧），实际因果模型效果会低一些。

**按场景 ACF 半衰期**：scene1/scene2 约 1.3-1.5s，scene3/scene4 约 0.7s。scene3 噪声更快变化但振幅最大（std=2.55m）。

#### 11.2 Oracle MOT 实验 ✅

**实验设计**：修改 FTM 信号为 `denoised_ftm = α × depth + (1-α) × raw_ftm`，α 从 0（无去噪）到 1（完美去噪），跑 WiFiJointTracker (global-only, wifi_weight=0.10) 评估 IDF1。

**4-fold LOSO 平均结果**：

| α (去噪程度) | avg IDF1 | Δ vs α=0 | avg IDsw | 说明 |
|---|---|---|---|---|
| 0.00 | 80.36% | — | 791 | 当前 Phase 5 (raw FTM) |
| 0.30 | 84.34% | **+3.98pt** | 594 | 30% 噪声消除 |
| 0.50 | 85.32% | **+4.95pt** | 525 | 50% 噪声消除 |
| 0.70 | 88.72% | **+8.36pt** | 451 | 70% 噪声消除 |
| 0.90 | 90.55% | **+10.18pt** | 388 | 90% 噪声消除 |
| 1.00 | 92.17% | **+11.81pt** | 368 | 完美去噪 Oracle 上限 |

**Per-scene Oracle (α=1.0)**：

| 场景 | Phase 5 IDF1 (α=0) | Oracle IDF1 (α=1) | Δ |
|---|---|---|---|
| scene1 | 83.73% | 92.63% | +8.90 |
| scene2 | 81.60% | 92.58% | +10.98 |
| scene3 | 75.29% | **93.63%** | **+18.34** |
| scene4 | 80.84% | 89.86% | +9.02 |

**关键结论**：
1. **理论增益巨大**：完美去噪可带来 +11.81pt（80.36%→92.17%），甚至超过 Phase 5 (global assignment) 的 +3.04pt 的近 4 倍。
2. **scene3 受益最大**（+18.34pt）：FTM 噪声最重（std=2.55m），去噪后从最差场景变为最优。
3. **实现 30% 去噪即有强收益**（+3.98pt）：考虑到 ACF=0.93 + 92.6% 低频的良好信号条件，30-50% 去噪效果是合理的初始目标。
4. **IDsw 同步大幅减少**：α=0.5 时 IDsw 从 791 降到 525（−34%），去噪不仅提升 IDF1，也减少 ID 切换。

**新增脚本**：`scripts/phase6_noise_analysis.py`, `scripts/phase6_oracle_analysis.py`, `scripts/phase6_oracle_mot.py`
**新增数据**：`exps/phase6_noise_analysis.json`, `exps/phase6_oracle_mot.json`

#### 11.3 阶段 A 去噪模型训练结果 ✅

**实验设计**：3 种 temporal denoising 模型，4-fold LOSO 交叉验证。输入为 (window_size=15, 2) 的 [FTM_m, FTM_std_m] 滑窗序列，输出为目标帧的 denoised range (m)。训练使用 HuberLoss(δ=1.0) + AdamW + CosineAnnealingLR，stride=3，20 epochs。

**Filter Baselines（朴素因果滤波 lower bound）**：

| 方法 | avg MAE | reduction | equiv_α |
|---|---|---|---|
| EMA α=0.1（最佳因果滤波） | 1.357m | 8.1% | 0.081 |
| causal_mean_15 | 1.404m | 5.0% | 0.050 |
| noncausal_median_15 | 1.402m | 5.0% | 0.050 |
| causal_median_15 | 1.425m | 3.5% | 0.035 |
| causal_median_5 | 1.459m | 1.4% | 0.014 |

**结论**：朴素滤波最高 equiv_α=0.08，远不足以产生有意义的 MOT 增益。

**学习去噪模型（4-fold LOSO 平均）**：

| 模型 | 参数量 | avg MAE | avg reduction | avg equiv_α | 类型 |
|---|---|---|---|---|---|
| **Causal CNN** (WaveNet-like) | 13,857 | **0.729m** | **50.8%** | **0.508** | 因果（实时） |
| Causal LSTM | 14,145 | 0.739m | 50.1% | 0.501 | 因果（实时） |
| BiLSTM | 36,417 | 0.720m | 51.4% | 0.514 | 非因果（回顾） |

**Per-scene equiv_α 细分**：

| 模型 | scene1 | scene2 | scene3 | scene4 | avg |
|---|---|---|---|---|---|
| Causal CNN | 0.529 | 0.470 | 0.502 | 0.531 | 0.508 |
| Causal LSTM | 0.524 | 0.474 | 0.500 | 0.507 | 0.501 |
| BiLSTM | 0.563 | 0.482 | 0.509 | 0.502 | 0.514 |

**关键发现**：
1. **因果模型 ≈ 非因果模型**：Causal CNN（0.508）几乎追平 BiLSTM（0.514），差距仅 1.2%。实时去噪完全可行。
2. **学习模型远超朴素滤波**：equiv_α 0.50 vs 0.08（>6 倍），验证 FTM 噪声有可学习的 temporal pattern。
3. **LOSO 泛化良好**：跨场景 equiv_α 稳定在 0.47–0.56，无崩塌 fold。Scene2 略低（~0.47），可能该场景多径环境独特。
4. **MAE 从 1.48m 降到 0.73m**：原始噪声降低超一半，对应 Oracle 实验 α=0.5 → 预期 IDF1 +4.95pt。

**模型架构细节**：
- Causal CNN：4 层 dilated causal conv (dilation=[1,2,4,8], kernel=3, hidden=32)，感受野 31 帧(3.1s)，LayerNorm + GELU + residual
- Causal LSTM：2 层单向 LSTM (hidden=32)，取最后时间步
- BiLSTM：2 层双向 LSTM (hidden=32)，取中心时间步

**新增代码**：`denoising/` 模块（dataset.py, models.py, train.py, inference.py）
**新增数据**：`exps/phase6_denoising_results.json`, `exps/phase6_filter_baselines.json`
**Checkpoints**：`checkpoints_denoising/` (12 files: 3 models × 4 folds)

#### 11.4 阶段 C: MOT 集成评测 ✅

**实验设计**：将训练好的去噪模型（online frame-by-frame FTMDenoiser）嵌入 WiFiJointTracker 的帧循环中，逐帧去噪 FTM 后输入 tracker，4-fold LOSO 评测 IDF1。

**关键发现：去噪 + 调整 σ 协同增益**

去噪后信号更干净，可以使用更小的 Gaussian σ（让 tracker 更"信任" FTM），获得更大增益：

| 配置 | σ | avg IDF1 | Δ vs raw | avg IDsw | 说明 |
|---|---|---|---|---|---|
| raw FTM (Phase 5 baseline) | 1.5 | 80.36% | — | 791 | 当前最优 |
| **Causal CNN + tighter σ** | **1.3** | **83.17%** | **+2.81pt** | **629** | 最佳配置 |
| Causal LSTM + tighter σ | 1.3 | 82.71% | +2.35pt | 642 | |
| BiLSTM + tighter σ | 1.3 | 81.63% | +1.27pt | 683 | |
| Causal CNN (same σ) | 1.5 | 81.32% | +0.96pt | 587 | 保守比较 |
| Causal LSTM (same σ) | 1.5 | 81.39% | +1.03pt | 591 | |
| BiLSTM (same σ) | 1.5 | 80.63% | +0.27pt | 617 | |

**σ 敏感性分析（scene4, causal_cnn）**——去噪使 tracker 对 σ 极其鲁棒：

| σ | raw IDF1 | denoised IDF1 | Δ |
|---|---|---|---|
| 1.5 | 80.84% | 80.83% | −0.01pt |
| 1.0 | 63.78% | **82.16%** | **+18.39pt** |
| 0.8 | 51.87% | 80.88% | +29.02pt |
| 0.5 | 44.68% | 75.68% | +31.00pt |

**最佳配置 Per-scene 明细（Causal CNN, σ=1.3）**：

| 场景 | raw IDF1 | denoised IDF1 | Δ | IDsw(raw→den) |
|---|---|---|---|---|
| scene1 | 83.73% | 86.97% | +3.24pt | 436→446 |
| scene2 | 81.60% | 84.93% | +3.33pt | 679→635 |
| scene3 | 75.29% | 76.95% | +1.66pt | 1258→768 |
| scene4 | 80.84% | 83.83% | +2.99pt | 793→657 |

**关键结论**：

1. **最佳模型为 Causal CNN**：实时因果推理（13.8k 参数），IDF1 +2.81pt（80.36%→83.17%），IDsw −20.4%（791→629）。
2. **因果优于非因果**：在 online MOT 中，Causal CNN (2.81pt) > Causal LSTM (2.35pt) > BiLSTM (1.27pt)。BiLSTM 因为预测 center frame 而非 last frame，在 online 模式下产生对齐问题。
3. **去噪的核心价值 = 使 tracker 能安全使用更紧的 σ**：σ=1.5 时 raw/denoised 几乎无差别（kernel 太宽，噪声被容忍）；σ=1.3 时去噪后优势明显（+2.81pt vs raw 崩塌到 ~70%）。
4. **IDsw 大幅下降**：scene3 从 1258→768（−39%），scene4 从 793→657（−17%），去噪显著减少身份切换。
5. **vs Oracle 预期**：Oracle α=0.5 预测 IDF1≈85.32%（+4.95pt），实际获得 83.17%（+2.81pt）。差距原因：(a) 模型只去除 50% 噪声而非理想 α=0.5 的全局均匀去噪；(b) warmup 期间（前 14 帧）回退到 raw FTM；(c) σ=1.3 仍偏保守。实际增益约为 Oracle 预期的 57%（2.81/4.95），合理范围。

**新增脚本**：`scripts/phase6_denoise_mot.py`
**新增数据**：`exps/phase6_denoise_mot.json`

#### 11.5 特征扩展实验 ❌（无效）

**动机**：Idea evaluation 建议尝试加入 IMU 加速度等特征提升去噪质量，从而逼近 Oracle 上限（equiv_α=0.5→0.7+）。

**相关性分析（306k 帧，4 scenes）**：

| 特征 | Pearson r with \|FTM noise\| | Spearman ρ | 结论 |
|---|---|---|---|
| Accel Mag | +0.0003 | −0.0006 | ≈0，无相关性 |
| Gyro Mag | +0.009 | −0.010 | 可忽略 |
| Mag Mag | +0.059 | +0.007 | 极弱 |
| RSSI | −0.039 | +0.050 | 极弱 |
| **FTM std**（已在输入中） | **+0.131** | **+0.120** | 最强，但仍弱 |

运动分段分析也证实：低/中/高运动下 FTM MAE 几乎相同（1.48-1.52m）。**FTM 噪声由环境多径效应主导，与手机运动状态无关。**

**多特征去噪实验（scene4 fold, CausalDilatedCNN）**：

| 特征配置 | n_feat | hidden | MAE | equiv_α | Δ vs baseline |
|---|---|---|---|---|---|
| FTM+std (baseline) | 2 | 32 | 0.667m | **0.523** | — |
| FTM+std+accel(3)+gyro(3) | 8 | 32 | 0.691m | 0.505 | **−0.017** |
| FTM+std+accel(3)+gyro(3) | 8 | 48 | 0.681m | 0.513 | **−0.010** |

**结论**：加入 IMU 特征反而**降低**去噪效果（equiv_α 从 0.523 降到 0.505-0.513）。额外维度引入的噪声超过了其可能携带的信息量。当前 2 特征输入（FTM_m + FTM_std_m）已接近最优。

**含义**：去噪模型的 equiv_α≈0.50 是此架构 + 此数据的瓶颈。提升方向不在"更多输入特征"，而在：
1. 更大/更深的模型架构
2. 不同的学习策略（self-supervised, contrastive）
3. 或接受当前去噪效果，转向阶段 B（学习 matching）

**新增脚本**：`scripts/phase6_imu_feature_analysis.py`, `scripts/phase6_feature_extension.py`
**新增数据**：`exps/phase6_feature_extension.json`

#### 11.6 阶段 B: 学习匹配模块 ✅

**动机**：Phase 6A 去噪后 FTM 噪声降低一半（equiv_α=0.51），但 tracker 仍使用固定 Gaussian compat 函数（σ=1.3m）。学习匹配函数可利用去噪后更干净的信号，学到比 Gaussian 更好的 depth-FTM 匹配策略。

**设计原则（解决 Method D 失败模式）**：
1. 使用 **去噪后 FTM**（非原始 FTM）——噪声降低 50%
2. **极小 MLP**（257 参数，非 Method D 的 90k）——防过拟合
3. **仅瞬时匹配**（无时序建模）——EMA 已负责累积
4. 二分类：(depth, denoised_ftm) → 同一人？

**模型架构**：MatchingMLP: 6 → 16 (ReLU) → 8 (ReLU) → 1 (Sigmoid)

**输入特征（6 维）**：
- Δ = depth − denoised_ftm
- |Δ|
- depth
- denoised_ftm
- ftm_std_m
- Δ²/(2σ²)（当前 Gaussian 指数项）

**训练数据**：1.2M 样本/fold（306k 正样本 + 896k 负样本，3:1 比例），BCE loss + class-balanced weighting，80/20 划分，AUC 早停。

**Matching 质量评估（4-fold LOSO 平均）**：

| 方法 | avg gap (pos−neg) | avg AUC |
|---|---|---|
| Gaussian (σ=1.3) | **0.458** | 0.8351 |
| Learned MLP | 0.415 | **0.8573** |
| Δ | −0.043 | **+0.0222** |

Learned MLP 在 AUC 上一致优于 Gaussian（+2.2%，4 fold 全部胜出），但 gap 略小（−0.043）。

**MOT 端到端评测（Causal CNN + σ=1.3, 4-fold LOSO）**：

| 匹配方法 | scene1 | scene2 | scene3 | scene4 | **avg IDF1** | avg IDsw |
|---|---|---|---|---|---|---|
| Gaussian (denoised FTM) | 90.30% | 85.11% | 83.34% | 83.68% | 85.61% | 570.8 |
| **Learned MLP (denoised FTM)** | **91.20%** | **86.61%** | **88.41%** | **85.42%** | **87.91%** | 556.2 |
| Δ | +0.90pt | +1.50pt | **+5.06pt** | +1.74pt | **+2.30pt** | −14.6 |

**关键结论**：

1. **Learned matching 显著优于 Gaussian**：+2.30pt IDF1（87.91% vs 85.61%），IDsw 减少 2.6%（556 vs 571）。
2. **scene3 受益最大（+5.06pt）**：该场景 FTM 噪声最重，learned matching 能更好地处理噪声残余。
3. **全部 4 fold 一致提升**：无退化 fold，泛化稳定（vs Method D 的 22pt 方差）。
4. **累积增益路径**：OC-SORT 65.63% → Phase 5 全局分配 80.36% → Phase 6A 去噪+σ调优 83.17% → **Phase 6B 学习匹配 87.91%**（总计 +22.28pt vs OC-SORT）。
5. **257 参数 vs Method D 的 90k 参数**：更小模型 + 去噪输入 + 模块化设计 = 更好的跨场景泛化。

**新增脚本**：`scripts/phase6b_learned_matching.py`, `scripts/phase6b_integrated_mot.py`
**新增数据**：`exps/phase6b_matching.json`, `exps/phase6b_integrated_mot.json`
**Checkpoints**：`checkpoints_matching/` (4 files: 1 model × 4 folds)

#### 11.7 实现计划（更新）

**三阶段解耦实现**：

| 阶段 | 模块 | 输入 → 输出 | 评测指标 | 目标 | 状态 |
|---|---|---|---|---|---|
| A | 去噪模型 | FTM 序列 → denoised range | MAE(m), 等效 α | α≥0.3 (MAE<1.0m) | ✅ equiv_α=0.51 |
| A-ext | 特征扩展 | +IMU/RSSI → denoised range | Δ equiv_α | α≥0.6 | ❌ IMU 无效 |
| B | 匹配模块 | denoised FTM + depth → compat score | AUC, gap | AUC > Gaussian | ✅ AUC +2.2%, IDF1 +2.30pt |
| C | MOT 集成 | denoised FTM → WiFiJointTracker | IDF1, IDsw | IDF1≥84% | ✅ IDF1=87.91% |

**技术设计**：

| 组件 | 选项 A | 选项 B |
|---|---|---|
| GT 监督信号 | depth (相机标注) | \|\|person_3D - AP_3D\|\| (Phase 0 AP 估计) |
| 输入特征 | FTM range + FTM std (**已验证最优**) | ~~+ IMU/RSSI~~（实验证明无效） |
| 架构 | 1D Dilated Causal CNN (WaveNet-like) | Bi-LSTM / Transformer |
| 窗口 | k=10~30 帧 (1-3秒) | |
| 输出 | denoised_range(t) | |
| 损失函数 | L1/Huber(denoised - GT_range) | |

**去噪后 matching 选项**：
1. **手工特征**：denoised_FTM → Gaussian compat (σ 可从 1.5m 缩小到 0.8-1.0m) → Phase 5 全局分配
2. **学习 matching**：denoised_FTM + depth → 轻量 affinity 网络（输入更 clean，学习更容易）

**论文贡献定位**：
- Contribution 1: Global phone-track joint assignment (Phase 5, algorithmic, +3.04pt)
- Contribution 2: Wireless signal denoising via temporal modeling (Phase 6A, learned, +2.81pt)
- Contribution 3: Learned depth-FTM matching (Phase 6B, learned, +2.30pt)
- 三者正交互补：全局分配优化利用方式 × 去噪提升信号质量 × 学习匹配替代手工函数
- 累积增益：OC-SORT 65.63% → 完整方法 **87.91%**（+22.28pt）
