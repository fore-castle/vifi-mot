# Phase 3: Reliability-Gated WiFi Matching

> 时间：2026-06-26
> 目的：提高无线-视觉匹配率，减少误匹配，增加可靠性判断机制
> 完整报告：`results/phase3_summary.md`

## 动机

Method A 已知的三个问题：

1. **IDsw +106%** (1269→2613)：bind/unbind 抖动导致短期 ID 切换增多
2. **Phone 换手** (scene4)：单 phone 索引覆盖 31 个不同 GT，"same phone = same person" 假设崩塌
3. **无可靠性判断**：已绑定的 track 始终使用 WiFi cost 修正关联，无论绑定是否可信

Phase 3 尝试通过 **多维可靠性评分** 和 **FTM 跳变检测** 解决这些问题。

## 方法

### 多维可靠性评分 R

对每个 (track, phone) 对，计算：

$$R = w_c \cdot \text{consistency} + w_s \cdot \text{stability} + w_x \cdot \text{exclusivity} + \text{maturity bonus}$$

| 分量 | 定义 | 物理意义 |
|------|------|----------|
| consistency | compat 滑动窗口均值 | 绑定平均质量 |
| stability | 1 − std(compat)/0.5 | FTM 是否稳定 |
| exclusivity | (best − 2nd) / best | 是否只有一个 phone 匹配 |
| maturity | +0.05 (≥5 帧) | 数据量足够 |

默认权重：w_c=0.40, w_s=0.30, w_x=0.30；compat_history_size=10。

### FTM 跳变检测

Per-phone 维护 FTM 滑动窗口（8 帧）。若当前 FTM 与窗口均值差异 > 阈值 → 判定为 phone 换手 → 强制 unbind 所有绑定该 phone 的 track。

### Reliability-Gated WiFi Cost

关联代价矩阵中：
- **二元 gate**：R < gate → 完全跳过 WiFi cost
- **R-weighted**：cost += wifi_weight × R × (1 − compat)，低可靠性 → 弱 WiFi 影响

### Bind/Unbind 决策

实验发现 R 用于 bind/unbind 会引入错误（R 与 EMA compat 高度相关且噪声更大），最终设计：**bind/unbind 仍用 EMA compat（同 Method A）**，R 仅用于 WiFi cost 加权。

## 实验配置

| 参数 | 值 |
|------|-----|
| wifi_weight | 0.10 |
| depth_sigma | 1.5m |
| ema_alpha | 0.85 |
| bind_threshold | 0.60 (EMA compat) |
| unbind_threshold | 0.15 (EMA compat) |
| bind_init_threshold | 0.70 |
| reliability_gate | 0.0 (disabled, R-weighted only) |
| ftm_jump_threshold | 999.0 (disabled) |

## 4-fold LOSO 主结果

| 场景 | Method A IDF1 | Phase 3 IDF1 | Δ | Method A IDsw | Phase 3 IDsw | Δ |
|------|--------------|-------------|---|--------------|-------------|---|
| scene1 | 81.58% | 79.21% | −2.37 | 477 | 491 | +14 |
| scene2 | 81.48% | 82.00% | **+0.52** | 688 | 657 | −31 |
| scene3 | 72.92% | 70.39% | −2.53 | 1073 | 1035 | −38 |
| scene4 | 78.68% | 78.52% | −0.16 | 827 | 796 | −31 |
| **avg** | **78.67%** | **77.53%** | **−1.14** | **3065** | **2979** | **−86** |

## 关键发现

### 1. FTM 跳变检测有害

FTM 自然波动达 2-5m（Phase 0 已证实），2m 阈值导致频繁 false unbind：

| ftm_jump_threshold | scene4 IDF1 | scene4 IDsw |
|--------------------|------------|-------------|
| 999 (disabled) | 77.7% | 815 |
| 8.0m | 77.5% | 835 |
| 5.0m | 75.9% | 980 |
| 4.0m | 72.6% | 1185 |
| 3.0m | 69.6% | 1529 |
| 2.0m | 66.0% | 2135 |

**结论**：单 AP 单帧 FTM 信号无法区分"正常波动"和"phone 换手"。FTM 跳变检测不适合作为 unbind 信号。

### 2. R-weighted WiFi cost 小幅减少 IDsw

IDsw 从 3065 降到 2979（−86，−2.8%），但 IDF1 回退 1.14pt。R 加权让弱绑定的 WiFi 影响更小，减少了错误关联引起的 IDsw，但同时也减弱了正确绑定的 WiFi 修正能力。

### 3. scene2 是唯一改善的场景

scene2 IDF1 +0.52pt，IDsw −31。scene2 的 FTM 噪声最轻（Phase 0: 中位误差最小），R 信号质量最高，因此 R-weighting 最有帮助。

### 4. scene3 严重回退

scene3 IDF1 −2.53pt。Phase 0 已发现 scene3 有重尾 FTM 噪声（中位 0.81m，RMS 2.39m）。R 中的 stability 分量被 FTM 噪声压低 → WiFi cost 被削弱 → 正确绑定得不到 WiFi 修正 → 关联变差。

### 5. 可靠性评分与 EMA compat 高度相关

R 的三个分量（consistency, stability, exclusivity）全部基于 depth-FTM compat 值，而 Method A 的 EMA compat 已经对 compat 做了时间平滑。**R 没有提供独立于 EMA compat 的信息**。在 FTM 噪声大的场景，R 引入的是噪声而不是信号。

## Ablation 矩阵

| 配置 | avg IDF1 | 说明 |
|------|----------|------|
| Method A (wifi_weight=0.10) | 78.67% | 基线 |
| Phase 3 R-weighted (gate=0.0) | 77.53% | 主配置，−1.14pt |
| Phase 3 binary gate=0.40 | 77.54% | 几乎相同 |
| Phase 3 binary gate=0.70 | 77.56% | 更高 gate 无帮助 |
| Phase 3 + FTM jump 2.0m | 66.0% | 灾难性回退 |
| Phase 3 + FTM jump 5.0m | 75.9% | 仍然有害 |

## 新增实现

- `trackers/reliability_gate.py`：`ReliabilityGateTracker`，多维 R 评分 + FTM 跳变检测 + R-weighted WiFi cost
- `scripts/run_baseline.py`：注册 `wifi_reliability` tracker，Phase 3 CLI 参数

## 新增数据

- `exps/wifi_reliability_phase3_final.json`：4-fold 主结果
- `exps/wifi_reliability_all_constant.json`：R-weighted 配置

## 路线决策

**Phase 3 不作为论文主创新点。** R-weighting 的改进不足以弥补 IDF1 回退。

**根因分析**：FTM 信号质量是硬约束。Phase 0 发现 FTM std 与实际误差相关性极弱 (r=0.11)；Phase 3 进一步证实 FTM 自然波动 2-5m，无法区分噪声与信号。在当前单 AP、depth-only 的设定下，**EMA compat 已经是近乎最优的 WiFi 匹配信号**。

**有潜力的下一步**：

1. **引入 IMU heading**：提供 FTM 之外的独立方向信号，解锁 Phase 1 的 3D Mahalanobis cost
2. **学习 compat 函数**：用数据驱动方法替代固定高斯（方案 D 的严格版），需要解决跨场景泛化问题
3. **真实检测器**：Phase 1/2 的 EKF/bridge 在 GT detection 下无效，真实检测器可能让无线修正发挥作用
4. **FTM 变化率 + IMU 联合**：用 IMU 加速度方向与 FTM 变化率的一致性作为 binding 质量信号（需要 IMU heading 校准）
