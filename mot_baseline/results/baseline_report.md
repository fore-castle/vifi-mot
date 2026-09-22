# ViFi MOT Baseline 报告

实验日期：2026-06-22
数据集：RAN4model_dfv4p4（同步预处理版 ViFi）
代码：`/Users/zstar/test/vifi/mot_baseline/`

## 1. 任务定义

把 ViFi 数据集重新解释为多目标跟踪（MOT）任务：

- **输入**：每帧的 BBX5 检测框（合法用户 + 旁观者），无 RGB 外观、无真实检测置信度
- **输出**：跨帧一致的 track ID 序列
- **GT**：BBX5 中的 `subj_idx`（合法用户记为 `L{idx}`）和 `Others_id_ls` 的旁观者 ID（记为 `O{idx}`）
- **划分**：Leave-one-scene-out，scene1~4 各做一次测试集（共 67 序列，约 11 万帧）

## 2. Baseline 设计

三个 tracker 都基于 SORT 系列（共享一个 7 状态恒速 Kalman + 匈牙利 IoU 匹配）：

| Tracker | 关联策略 | 关键参数 |
|---|---|---|
| SORT | 单轮 IoU + Hungarian | iou=0.3, max_age=30, min_hits=3 |
| OC-SORT | + Observation-Centric Re-Update / Momentum / Recovery | inertia=0.2, delta_t=3 |
| ByteTrack | 高/低分两阶段（需置信度） | track_thresh=0.5, new=0.6 |

ByteTrack 因为 ViFi 没有真实检测置信度，分别试过：
- `score=1.0`：全部进入高分桶，第二阶段失效，等价于退化但 min_hits 设置让短暂目标全部丢失
- `score=clip(1-depth/20, 0.05, 1)`：远距离目标 score 太低无法建立轨迹，FN 灾难性

## 3. 主实验结果（4-fold leave-one-scene-out）

### 3.1 SORT

| 测试场景 | MOTA | IDF1 | IDsw | FP | FN |
|---|---|---|---|---|---|
| scene1 | 97.73% | 61.32% | 427 | 257 | 1433 |
| scene2 | 97.75% | 53.49% | 435 | 453 | 1582 |
| scene3 | 98.16% | 56.37% | 425 | 171 | 1534 |
| scene4 | 97.01% | 53.95% | 611 | 389 | 2372 |
| **avg** | **97.66%** | **56.28%** | — | — | — |

### 3.2 OC-SORT

| 测试场景 | MOTA | IDF1 | IDsw | FP | FN |
|---|---|---|---|---|---|
| scene1 | 97.65% | 71.39% | 317 | 527 | 1345 |
| scene2 | 97.90% | 61.19% | 308 | 615 | 1386 |
| scene3 | 98.25% | 67.26% | 256 | 412 | 1366 |
| scene4 | 96.94% | 62.66% | 388 | 841 | 2219 |
| **avg** | **97.69%** | **65.63%** | — | — | — |

### 3.3 ByteTrack（参考）

构造伪置信度后明显劣化，**不建议作为对比基线**。Constant=1.0 时 MOTA=46.73%, IDF1=14.43%；depth-derived 时 MOTA=45.27%, IDF1=9.43%。
ByteTrack 的双阶段关联依赖真实检测分布，在 ViFi 这种 GT-as-detection 设置下不成立。

## 4. 关键观察

1. **MOTA 已接近天花板（~97%）**：因为 GT 框直接当检测，FP/FN 仅来自跟踪逻辑（建轨/丢轨延迟），改进空间有限。
2. **IDF1 是主战场**：跨场景 56%-65%，意味着 **每个 GT 目标在 tracker 输出中平均会被切成 1.7~2.5 段不同 ID**。
3. **OC-SORT 比 SORT 在 IDF1 上提升约 9.3 个点**，IDsw 减少约 33%。说明 motion-only 的二次关联（Observation-Centric Recovery）已经能 captured 一部分长时遮挡。
4. **FM（fragmentation）数量与 IDsw 接近**：大多数 ID switch 来自"目标短暂消失→重新出现时被新 ID 接管"。
5. **scene4 IDF1 最低 (62.7%)**：旁观者最密集（平均~19个），且户外有大量出入视野行为。

## 5. Wireless 信号介入点分析

把 BBX5/Others 视作"检测器"已经几乎完美，wireless 信号唯一有用的方向是 **降低合法用户的 ID switch / fragmentation**。下面是几个具体可落地的方向，难度递增：

### A. WiFi-as-ReID（最低成本，强可解释性）
把每个合法用户的 `FTM_li + IMU19` 经过一个轻量 Bi-LSTM（直接复用你 my_vifi 的相机/手机 encoder）映射为 32 维 embedding。每个 track 维护一个 EMA embedding；关联时把 `1 - cos(emb_track, emb_phone)` 作为附加 cost。

- 优点：纯插件式改造，OC-SORT 框架一行代码加一个 cost
- 难点：旁观者没有 wireless → 只有合法用户有这层增强；要设计 fallback（fall back 到纯 IoU）
- 论文卖点：第一个 wireless ReID for MOT，且只在 *partial subjects* 有信号的真实部署条件下工作

### B. FTM-as-Depth-Prior（修 Kalman 协方差）
合法用户多了一个 ~1m 精度的距离测量 (FTM range)。把它转成"图像平面深度噪声"通过 ZED 内参反投影，融进 Kalman 的观测协方差 R。等价于：合法用户的轨迹在被 depth 测量约束的方向上更稳定。

- 优点：理论清晰，可以做 ablation 表
- 难点：要拿到 ZED 内参；FTM 在多路径下噪声非高斯

### C. IMU-as-Motion-Prediction-Correction（修 F 矩阵）
SORT 的恒速假设在行人转向/启停时崩。用 IMU 的瞬时角速度修正 F 的方向项，把 `du/dt, dv/dt` 用 IMU 投影修正。

- 优点：纯 motion 层修改，对 IDsw 直接有用
- 难点：需要标定相机-手机相对姿态；只对合法用户有用

### D. Wireless-Vision Joint Affinity（你已有的亲和矩阵复用）
直接把你 my_vifi 的预训练亲和模型搬进 tracker：每次需要解决新-旧关联时，把"现有 tracks 的 BBX 历史 + 候选 detection 的 BBX 历史"过亲和网络得到一个 W 矩阵，与 IoU 矩阵以可学习的 alpha 加权。
- 优点：复用现有 91% 的亲和模型权重，工程量小
- 难点：亲和模型期望窗口=10 帧；新轨迹只有 1 帧时不能直接用

### 推荐路线

第一步先做 **A (WiFi-as-ReID)** 来快速验证 wireless 增强带来 IDsw 下降；第二步上 **D**（复用你已有亲和矩阵）做更强消融；A/D 都能给出 *partial coverage*（旁观者无信号）下的真实增益曲线，这是 ViFi 数据天然的故事。

## 6. 复现命令

```bash
cd /Users/zstar/test/vifi/mot_baseline

# 4-fold leave-one-scene-out
python3 scripts/run_baseline.py --tracker sort     --all-folds
python3 scripts/run_baseline.py --tracker ocsort   --all-folds
python3 scripts/run_baseline.py --tracker bytetrack --all-folds --score-mode constant

# 单场景调试
python3 scripts/run_baseline.py --tracker ocsort --test-scene scene4
```

结果 JSON 保存在 `exps/`，每个文件是一个数组，每个元素一个测试场景。

## 7. 待办

- [ ] 把 RGB 时间戳读进来，给 motchallenge 评测加一个真实 fps
- [ ] 加 HOTA 评测（需要 TrackEval 包）
- [ ] 实现方案 A：wireless ReID embedding + EMA + cost fusion
- [ ] 实现方案 D：复用 my_vifi 亲和矩阵作为关联打分器
- [ ] indoor scene0 的 GT 也评测一次（5 用户、无旁观者，简单环境的天花板对照）
