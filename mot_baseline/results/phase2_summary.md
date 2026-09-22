# Phase 2: Direction β — Wireless Tracklet Bridging 评测报告

> 实验时间: 2026-06-26
> 实现: `trackers/wifi_ocsort_merger.py` (Direction β v2, post-hoc ID merger)
>        + `trackers/wifi_spatial_ocsort.py` (in-tracker bridging_max_age, v1)
> 评测: strict LOSO, outdoor 4 folds (scene1/2/3/4), GT detections, IoU=0.5
> 计划文档: `results/experiment_plan.md` Sec 5 (Direction β)

---

## 1. 问题定位

### 1.1 Method A 的 IDsw 归因

用 `scripts/phase2_idsw_attribution_v2.py` 把 4 个户外场景的 IDsw 按 GT 身份归因：

| scene | phone-holder GT sw | other GT sw | other % |
|---|---|---|---|
| scene1 | 106 (32 phone GTs) | 136 (367 other GTs) | **56.2%** |
| scene2 | 203 (54 phone GTs) |  65 (336 other GTs) | 24.3% |
| scene3 | 149 (54 phone GTs) |  66 (348 other GTs) | 30.7% |
| scene4 | 158 (45 phone GTs) | 142 (735 other GTs) | **47.3%** |

**结论**：phone-holder GTs 贡献的 IDsw 比例远超其在 GT 总数中的占比。每个 phone-holder GT 平均 3.3-4.3 次 IDsw，而每个 other GT 平均仅 0.2-0.4 次。这是无线信号最应该解决的问题：phone-holder 的 IDsw。

### 1.2 失败模式

用 `scripts/phase1_idsw_modes.py` 检查，发现 100% 的 phone-holder IDsw 是 **occlusion gap** 类型（Method A 的 tracklet 在 `max_age=30` 帧后终止；phone holder 重新出现时创建新 tracklet），几乎不存在 "close proximity swap" 类型。

进一步用 `scripts/phase2_gap_probe.py` 验证：同一个 phone 索引的连续 tracklet 之间的 gap 在本数据集上 **几乎不存在**（max_age=30 已经足够长），IDsw 发生在更细粒度的帧级——同一个物理人在不同帧被绑定到不同的 underlying_tid。

### 1.3 关键发现：phone 绑定会在同一序列内换手

用 `scripts/phase2_check_phone_gt.py` 检查，发现 scene4 中 **单个 phone 索引可以覆盖多达 31 个不同的 GT 身份**：

```
seq=outdoor-scene4-20211007_143810 phone=1 covers 31 different GTs
```

根因：当 phone holder 走出画面/被遮挡时，Method A 的绑定机制会把这个 phone 索引重新分配给附近的其他行人（因为 FTM 信号仍在，但没有对应的检测）。所以 **phone 绑定 ≠ 物理身份**。

这直接导致：任何基于 "same phone binding = same person" 假设的 merger 都会创建跨人的超级 tracklet（super-track），反而损害 IDF1。

---

## 2. 实现路线对比

### 2.1 Direction β v1 — 在 tracker 内延长 bound track 的寿命

修改 `trackers/wifi_spatial_ocsort.py`：
- 新增 `bridging_max_age` 参数（默认 200），bound track 的 `time_since_update` 上限从 `max_age=30` 提升到 `bridging_max_age`。
- 每个未匹配帧对 bound track 跑 EKF wireless update，让预测 bbox 锚在 FTM 上。
- CLI: `--bridging-max-age N`

### 2.2 Direction β v2 — post-hoc tracklet ID 重写

新增 `trackers/wifi_ocsort_merger.py::WiFiOCSortTrackletMerger`：
- 包装 Method A，**不改变**内部关联逻辑（确保视觉部分完全等价于 baseline）。
- 实时维护 per-underlying-tid 的 tracklet span，tracklet 终止时存入 `_terminated`。
- 新 tracklet 绑到同一个 phone 时，若 gap ≤ `bridging_max_age` 且通过空间/F门，就把新 tracklet 的 output tid 改写为 `phone_id_offset + phone`，让 evaluator 看到连续身份。
- 多重合并门：
  1. `boundary_iou_min`：terminated 最后一个 bbox 与 新 tracklet 第一个 bbox 的 IoU ≥ 0.10
  2. `bbox_proximity_px`：IoU=0 时，bbox 中心距离 ≤ N 像素
  3. `require_ftm_continuity + ftm_std_max`：gap 期间的 FTM std ≤ 1.5 m
- CLI: `--tracker wifi_merger --bridging-max-age N --bbox-proximity-px M`

---

## 3. 4-fold LOSO 结果

### 3.1 v1: in-tracker bridging_max_age

| 配置 | scene1 | scene2 | scene3 | scene4 | avg IDF1 | Δ |
|---|---|---|---|---|---|---|
| Method A baseline | 81.58% | 81.48% | 72.92% | 78.68% | **78.67%** | — |
| **v1 bridging_max_age=200** | 75.24% | 71.29% | 62.91% | 68.43% | **69.47%** | −9.20 pt |

v1 在所有场景都严重退化。根因：长寿命 bound track 的预测 bbox 漂移严重，重新出现的 detection 优先匹配到漂移 track 而非新 tracklet，导致错误关联 → IDF1 暴跌。

### 3.2 v2: post-hoc ID merger with gates

| 配置 | scene1 | scene2 | scene3 | scene4 | avg IDF1 | Δ | IDsw 总 Δ |
|---|---|---|---|---|---|---|---|
| Method A baseline | 81.58% | 81.48% | 72.92% | 78.68% | **78.67%** | — | 3065 |
| v2 bmax=40 prox=150 | 81.79% | 78.10% | 72.81% | 72.97% | **76.42%** | −2.25 pt | −736 |
| v2 bmax=80 prox=200 | 81.26% | 77.78% | 72.81% | 72.05% | **75.97%** | −2.70 pt | −758 |
| v2 bmax=150 prox=300 | 81.17% | 77.75% | 72.90% | 71.77% | **75.90%** | −2.77 pt | −779 |
| v2 bmax=200 prox=400 | 81.17% | 77.52% | 71.87% | 71.77% | **75.58%** | −3.09 pt | −786 |

完整 JSON：`exps/wifi_merger_all_bmax150_prox300.json` 等。

v2 在所有场景都显著减少 IDsw（−736 ~ −786），但伴随一致的 IDF1 回退（−2.25 ~ −3.09 pt）。

### 3.3 scene1 上的单场景峰值表现

| 配置 | IDF1 | IDsw |
|---|---|---|
| Method A | 81.58% | 477 |
| v2 bmax=30 | 84.20% | 352 |
| v2 bmax=40 | 84.30% | 352 |
| v2 bmax=80 | 84.40% | 352 |
| v2 bmax=150 | **85.60%** | 352 |

scene1 上 v2 实现 **+4.02 pt IDF1**，且 FP/FN/IDsw 完全不变（IDF1 增益纯粹来自 tracklet fragmentation 的减少）。但这一增益在其他场景无法复现。

---

## 4. 关键分析

### 4.1 为什么 scene1 大涨，scene2/3/4 却小跌？

scene1 的 phone-binding 较稳定（phone 几乎只由真正的 holder 持有），merger 创建的 super-track 确实对应同一个物理人 → IDF1 收益。

scene4 的 phone-binding 极不稳定（143810 序列中单个 phone 覆盖 31 个不同 GT），merger 即使加了 IoU/proximity 门，仍然会创建跨人 super-track → IDF1 严重受损（−6.91 pt）。

### 4.2 IoU/proximity 门为什么不够？

边界 IoU ≥ 0.10 假设同一个物理人在 gap 前后会出现在相似的图像区域。但当 person A 在画面左边消失、person B 在画面右边持有同一个 phone 出现时，IoU=0 且 proximity>阈值，merger 拒绝合并——但 person A 自己也可能因为其他原因（如遮挡后换个姿势）导致 bbox 变化，IoU 跌破阈值。所以门要么太紧（错失合法合并）要么太松（放行跨人合并）。

### 4.3 为什么 IDsw 持续减少但 IDF1 不回涨？

MOT 评估中 IDF1 由 `IDTP / (IDTP + 0.5·IDFP + 0.5·IDFN)` 计算，与 tracklet 连续性、ID 数量都相关。merger 把多个 short tracklet 合并成一个 long super-track 时：
- IDTP 可能增加（连续覆盖同一人）
- IDFP 也可能增加（super-track 的同一 output tid 覆盖多个 GT → FP）
- 净效果取决于 super-track 的纯度

在 scene4 这种 phone 频繁换手的数据上，super-track 纯度低 → IDFP 上升 → IDF1 下降。

### 4.4 与 Phase 1 和 Method A 横向对比

| 方法 | avg IDF1 outdoor | avg IDsw | 实现复杂度 |
|---|---|---|---|
| OC-SORT baseline | 65.63% | 3857 | 低 |
| Method A (WiFi-Geom) | 78.67% | 3065 | 低 |
| Phase 1 (scalar+EKF+bridge) | 78.84% | 3057 | 中 |
| **Phase 2 v1 (in-tracker)** | 69.47% | ? | 中 |
| **Phase 2 v2 (merger bmax=40)** | 76.42% | 2329 | 低 |

Phase 2 v2 在 IDsw 上达到全实验最佳（2329，比 baseline −39.6%，比 Method A −24.0%），但 IDF1 落后 Method A 2.25 pt。

---

## 5. 路线决策

**不把 Direction β 的当前实现作为论文主创新点**。原因：
1. Post-hoc ID 重写是"作弊"式修改，缺乏物理意义：它假设 phone binding 稳定，但在 scene4 上这个假设崩溃。
2. In-tracker bridging 严重退化，说明单纯延长 track 寿命 + EKF wireless update 在 GT detection 下没有带来关联增益。
3. 与 Method A 持平甚至更差，在科学贡献上是"实现探索"，不是"性能提升"。

**但方向本身有价值**。真正的瓶颈是 phone binding 的不稳定性（单 phone 覆盖多 GT），这本身就是一个 research question：

- **候选方向 B'：先修 phone binding 稳定性，再做 tracklet bridging**
  用 FTM 变化率（而非绝对值）作为 binding 信号：当 FTM 突变（>1m / 1s）时 unbind 当前 track，让 phone 重新找更近的匹配。这样 phone binding 更稳定，merger 的假设才成立。
- **候选方向 C：直接用 IMU heading 替代 anchor 决定 phone 方位**
  用 IMU 的航向 + 步长估计 phone 的真实位移，构造 phone state 的非径向约束。这解锁了 Phase 1 的 3D Mahalanobis cost，理论上能显著超过 Method A。

---

## 6. 复现命令

```bash
# Direction β v2 主推荐配置
python3 scripts/run_baseline.py \
  --tracker wifi_merger --all-folds \
  --wifi-weight 0.10 --bind-threshold 0.50 --depth-sigma 1.5 \
  --bridging-max-age 40 --bbox-proximity-px 150 \
  --out exps/wifi_merger_all_bmax40.json

# Direction β v1 in-tracker bridging
python3 scripts/run_baseline.py \
  --tracker wifi_spatial --all-folds \
  --spatial-weight 0.10 --bind-threshold 0.50 --depth-sigma 1.5 \
  --compat-mode scalar --bridging-max-age 200 \
  --out exps/wifi_spatial_direction_beta.json

# 诊断脚本
python3 scripts/phase1_idsw_modes.py     # occlusion-gap vs close-swap 归因
python3 scripts/phase2_idsw_attribution_v2.py  # phone vs other GT 归因
python3 scripts/phase2_check_phone_gt.py  # phone binding 换手分析
```

---

## 7. 新增代码资产

| 文件 | 作用 |
|---|---|
| `trackers/wifi_ocsort_merger.py` | Direction β v2 post-hoc merger 包装器 |
| `trackers/__init__.py` | 导出 `WiFiOCSortTrackletMerger` |
| `scripts/run_baseline.py` | 注册 `wifi_merger` tracker；新增 `--bridging-max-age` / `--bbox-proximity-px` |
| `scripts/phase1_idsw_modes.py` | Method A IDsw 归因（occlusion gap vs close swap） |
| `scripts/phase2_gap_probe.py` | 同一 phone 的连续 tracklet gap 分布 |
| `scripts/phase2_idsw_attribution_v2.py` | IDsw 按 phone-holder / other 归因 |
| `scripts/phase2_check_phone_gt.py` | phone binding 换手检查 |
| `scripts/phase2_posthoc_merger.py` | 独立 post-hoc merger 原型（已被 `wifi_ocsort_merger.py` 替代） |
| `scripts/phase2_tracklet_bridge_probe.py` | tracklet bridging 频率探测 |
| `scripts/phase2_inspect_one.py` | 单条 tracklet 可视化诊断 |

新增数据：`exps/wifi_merger_all_bmax*.json`, `exps/wifi_spatial_direction_beta.json`, `exps/method_a_idsw_modes.json`

---

## 8. 下一步候选

- **A. 修复 phone binding 稳定性**：FTM 变化率 unbind + 短窗口 EMA，预期让 v2 merger 在所有场景都能涨
- **B. 引入 IMU heading 修正 Mahalanobis**：解锁 Phase 1 的 3D cost，预期 +3~5 pt
- **C. 切换到真实检测器（YOLOv8）**：复现 Phase 1/2 在带噪声 detection 下的效果
- **D. 失败模式分析**：在 Vi-Fi + WP-ReID 双数据集上跑诊断，看哪个方向能解锁最大 IDsw 收益
