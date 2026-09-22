# 方案 D — 学习版亲和矩阵作 wireless cost

> **TL;DR**：把 ViFi 复现的亲和矩阵学习模型接入 MOT tracker 作 wireless ReID，
> 在单 fold（leaky）评测下 outdoor avg IDF1 达 90.91%，但经严格 leave-one-scene-out
> 4-fold 重训后回落到 **75.62%**，仍低于几何版方案 A 的 78.51%。
> 域内很强、跨域退化，建议下一步做 A+D ensemble。
> 完整研究总结见 [wifi_reid_mot_study_report.md](./wifi_reid_mot_study_report.md)。

实验日期：2026-06-22
代码：`trackers/wifi_affinity.py` + `trackers/affinity_helper.py`
模型：`my_vifi/checkpoints_strict/full/fold1_best_epoch29_acc0.8552.pth`

## 1. 思路

把 my_vifi（用户已复现的 ViFi 论文亲和矩阵学习）的预训练 Bi-LSTM + 1×1 Conv
模型直接作为 wireless 信号的"软 ReID encoder"，**替换方案 A 的几何高斯
compatibility 函数**。OC-SORT 的关联代价、Phone Monopoly、bind-on-creation 等
机制完全保留。

模型架构（来自 my_vifi/model.py）：

- 相机分支：Bi-LSTM(2 层, hidden=32) on (cx, cy, depth) × k=10 帧
- 手机分支：Bi-LSTM(2 层, hidden=32) on (FTM_range, FTM_std, IMUagm9 9d) × k=10 帧
- 1×1 Conv 4 层（128→64→32→16→1）压成 Affinity tensor (Np+1, Nc+1)
- Row-softmax × Col-softmax 平均得到对称化软概率

参数量：90,817（轻量）。

## 2. 实现要点

`AffinityHelper` 维护两个滚动 buffer：

```
track_buf:  track_id -> deque[(cx, cy, depth)]  (max k=10)
phone_buf:  phone_idx -> deque[(ftm_r, ftm_std, imu9...)]  (max k=10)
```

每帧步骤：

1. OC-SORT IoU + OCM 关联，匹配 → push (cx, cy, depth) 进 track_buf
2. 所有 phone 推送 (FTM, IMU9) 进 phone_buf
3. Forward MultimodalNetwork → affinity (Np, N_active_track)
4. EMA 更新每个 track 的 phone_scores（用 affinity 替换几何 compat）
5. greedy 绑定（max score 优先 + Phone Monopoly）

输出 ID 与方案 A 完全一致：未绑定用 short ID，绑定后用 `phone_id_offset + phone_idx`。

## 3. 关键工程细节

- **Z-score 归一化**：`scripts/precompute_norm_stats.py` 调用 my_vifi 的
  `compute_norm_stats`，在 fold1 训练序列上抽样 ~2000 个 window，把 cam_mean/std
  和 ph_mean/std 落盘为 `data/affinity_norm_stats.npz`。tracker 启动时一次性加载。
- **历史不足时跳过**：track 或 phone 历史 < `min_history=3` 帧 → affinity 列/行 mask 为 0
  （即不参与该帧的绑定决策）。这避免新生 track 在 frame 1 就用 garbage history 推理。
- **track 上限**：模型固定 Nc=15 槽位，超出时只取前 Nc 个 active tracks 进入推理；
  其余仍走纯 OC-SORT。outdoor scene 单帧很少 >15 active tracks。
- **设备**：CPU 推理，全部 4-fold 仅 ~35s（vs OC-SORT ~9s），4× 慢但仍 real-time-ish。
- **数据加载扩展**：`data/vifi_mot.py` 新增 `frame.imu_agm9` 字段以支持
  9 维 IMUagm9（与 my_vifi 训练一致）；旧的 `frame.imu19` 仍保留兼容方案 A。

## 4. 实验结果

最佳超参（沿用方案 A 的 EMA + bind 逻辑参数）：
`wifi_weight=0.10  ema_alpha=0.85  bind_threshold=0.50
bind_init_threshold=0.60  unbind_threshold=0.10`

### 4.1 IDF1 主表（关键指标）

| Scene | OC-SORT | WiFi-Geom (A) | **WiFi-Affinity (D)** | Δ vs OC-SORT | Δ vs A |
|---|---|---|---|---|---|
| scene1 | 71.39 | 81.76 | **91.21** | +19.82 | +9.45 |
| scene2 | 61.19 | 77.26 | **91.70** | +30.51 | +14.44 |
| scene3 | 67.26 | 74.99 | **92.60** | +25.34 | +17.61 |
| scene4 | 62.66 | 80.02 | **88.11** | +25.45 | +8.09 |
| **avg outdoor** | **65.63** | **78.51** | **90.91** | **+25.28** | **+12.40** |
| scene0 (indoor strict) | 19.74 | 39.27 | 37.81 | +18.07 | -1.46 |

### 4.2 完整指标（4-fold avg outdoor）

| Tracker | MOTA | IDF1 | IDsw | FP | FN | 推理速度 |
|---|---|---|---|---|---|---|
| OC-SORT | 97.69 | 65.63 | 1269 | 2395 | 6316 | ~9s |
| WiFi-Geom (方案 A) | 97.38 | 78.51 | 2613 | 2499 | 6210 | ~10s |
| **WiFi-Affinity (方案 D)** | **97.45** | **90.91** | 2306 | 2395 | 6316 | ~35s |

注意：WiFi-Affinity 在 IDF1 + 25.3 pt 的同时，MOTA 仅降 0.24 pt，
**FP/FN 与 OC-SORT 完全一致**——这是因为 affinity 模型只影响 phone 软绑定（即输出 ID），
**完全不修改 detection-track 关联结果**，所以不会引入额外 FP/FN。

### 4.3 IDsw 对比

| Tracker | scene1 | scene2 | scene3 | scene4 | total |
|---|---|---|---|---|---|
| OC-SORT | 317 | 308 | 256 | 388 | 1269 |
| WiFi-Geom (A) | 448 | 647 | 838 | 680 | 2613 |
| WiFi-Affinity (D) | 472 | 641 | 516 | 677 | 2306 |

学习版 IDsw 比几何版减少 ~12%（scene3 减少 38%）——更准确的绑定决策减少了
unbind→rebind 切换。

## 5. 严格 Leave-One-Scene-Out 评测

§4 那张表是**乐观估计、有 train-test leakage**：所有数字都来自 my_vifi 提供的
fold1 checkpoint，而 fold1 的训练集包含 scene1–4 中 ~70/75 个 outdoor 序列。
当用同一个 fold1 模型对 4 个 outdoor scene 做 inference 时，绝大多数测试序列
模型都已经"见过"。要拿到真正的跨场景泛化数字，必须按 scene 切分重新训练。

### 5.1 协议

**Leave-One-Scene-Out (LOSO) 4-fold**：

- fold k：测试集 = scene k 的全部序列，训练集 = scene0 (indoor) ∪ scene{1..4}\{k}
- 每个 fold 独立计算 norm_stats、独立训 30 epoch（CPU，bs=64，SGD+MultiStep）
- 选 best val checkpoint（按 sequence-level val_ratio=0.1, val_seed=2026）
- 在测试 scene 上跑 wifi_affinity tracker（沿用 §4 同一组超参）

代码：`scripts/loso_train.py` 和 `scripts/loso_eval.py`，
ckpt 落盘到 `checkpoints_loso/full/fold{1..4}_best_*.pth`。

每 fold val accuracy（仅作训练健康度参考，不是测试指标）：
fold1=85.25%, fold2=87.11%, fold3=86.32%, fold4=85.36%。

### 5.2 严格 4-fold 结果

| Fold | Test scene | MOTA | IDF1 | IDsw | FP | FN |
|---|---|---|---|---|---|---|
| 1 | scene1 | 97.38% | **87.18%** | 568 | 527 | 1345 |
| 2 | scene2 | 97.18% | **80.81%** | 1101 | 615 | 1386 |
| 3 | scene3 | 97.72% | **69.68%** | 864 | 412 | 1366 |
| 4 | scene4 | 96.09% | **64.81%** | 1349 | 841 | 2219 |
| **avg outdoor** | — | **97.09%** | **75.62%** | — | — | — |

### 5.3 与方案 A / leaky D 对比（outdoor 4-fold avg）

| Tracker | MOTA | IDF1 | Δ vs A |
|---|---|---|---|
| OC-SORT | 97.69 | 65.63 | — |
| WiFi-Geom (A) | 97.38 | 78.51 | 0.00 |
| **WiFi-Affinity (D, leaky)** | 97.45 | 90.91 | +12.40 |
| **WiFi-Affinity (D, strict LOSO)** | 97.09 | **75.62** | **−2.89** |

### 5.4 关键观察

1. **leakage 解释了大部分增益**：去掉训练泄漏后 IDF1 从 90.91% 落到 75.62%（−15.3 pt），
   实际跨场景泛化能力远低于"看起来很强"的乐观数字。
2. **严格 LOSO 仍未超过几何版**：strict D = 75.62 < A = 78.51，说明当前小模型
   （~90k 参数）在新场景上的几何线索利用还不如人工设计的 EMA + bind 启发式。
3. **fold-by-fold 异质性极大**：scene1 strict IDF1 = 87.18%（已超几何版 81.76%），
   而 scene4 = 64.81%（远低于几何版 80.02%）。**不是模型容量问题**——实测全数据集
   每序列 phones ≤ 3、同帧总人数（含 passersby）≤ 12，`Nm_phone=5` 和
   `Nm_camera=15` 从未饱和。`20211007_143810` 报出的 "187 GT IDs" 是 motmetrics
   累计计数：路人不断进出场，每次入画都拿一个新 GT ID，与同帧并发量无关。
   scene4 的失败模式需要 per-sequence 误差分析（IMU 失效、scene 视角差异等）。
4. **MOTA 仍稳**：所有 strict fold MOTA 都 ≥ 96%，说明 affinity 错误绑定不会回灌
   到 detection-track 关联，FP/FN 仍由 OC-SORT 主导，这是软绑定架构的稳定性优势。
5. **scene0 indoor 的 OOD 数字现在更有意义**：原 §4 的 fold1 在 indoor 上 IDF1=37.81%，
   是"训练含 outdoor、测试 indoor"的跨域泛化数字；它和 strict scene4 的退化模式
   一致——训练分布外的场景，学习版会显著退化。

## 6. 优势与局限

**优势**

- 利用了 wireless 信号的时间结构（IMU 时序、FTM 时序）：单帧几何只能用 depth 一个标量
- 学习到的非高斯噪声模型：FTM 多路径偏差不再是简单高斯
- IDF1 在域内逼近 90%，远超几何版上限（78.5%）
- 完全不修改 OC-SORT 的关联代价矩阵 → 不引入额外 FP/FN，MOTA 几乎不动

**局限**

- 需要预训练 checkpoint + norm_stats（部署成本变高）
- CPU 推理 4× 慢于纯 OC-SORT（每帧 forward 一次 Bi-LSTM）
- 跨 domain 退化：fold1 模型在 indoor 上不如几何版稳健
- 依赖 IMUagm9 信号；缺失或噪声大时回退弱于几何版的 depth-only 启发式

## 7. 复现命令

```bash
cd /Users/zstar/test/vifi/mot_baseline

# 1) 一次性预计算 norm_stats（产出 data/affinity_norm_stats.npz）
python3 scripts/precompute_norm_stats.py

# 2) 单场景对比
python3 scripts/run_baseline.py --tracker wifi_affinity --test-scene scene4 \
    --wifi-weight 0.10 --bind-threshold 0.50 \
    --bind-init-threshold 0.60 --unbind-threshold 0.10

# 3) 4-fold（注意：用同一个 fold1 checkpoint，存在 leakage）
python3 scripts/run_baseline.py --tracker wifi_affinity --all-folds \
    --wifi-weight 0.10 --bind-threshold 0.50 \
    --bind-init-threshold 0.60 --unbind-threshold 0.10

# 4) Indoor 严格未见
python3 scripts/run_baseline.py --tracker wifi_affinity --test-scene scene0 \
    --wifi-weight 0.10 --bind-threshold 0.50 \
    --bind-init-threshold 0.60 --unbind-threshold 0.10

# 5) 严格 LOSO：按 scene 切，重新训 4 个 fold
python3 scripts/loso_train.py --folds 1,2,3,4 --epochs 30 --batch-size 64

# 6) 严格 LOSO 评测：每个 fold 用对应 ckpt 在 held-out scene 上测
python3 scripts/loso_eval.py --folds 1,2,3,4
```

## 8. 后续

- 严格 LOSO 已经完成（§5），下一步重点是缩小 strict D (75.62) 与 A (78.51) 的差距：
  - **A+D ensemble**（首选）：affinity 高置信时用 D，低置信回退 A；
    比如对每个 phone 取 softmax 最大概率，> threshold 用学习版绑定，否则回退几何版
  - **训练分布扩展**：indoor + outdoor 混合训练，提升跨场景视角鲁棒性
  - 更长训练 / 更大模型；但要警惕 overfit 到训练 scene
- 拿 strict LOSO ckpt 在 indoor scene0 上评测，得到真正的 OOD 数字
  （当前 37.81% 仍是 leaky fold1 ckpt 上的）
- 量化推理速度：导出 ONNX / TorchScript，看是否能压到 OC-SORT 同档
- **scene4 误差分析**：scene4 strict IDF1 比 A 低 15 pt，但**不是模型容量问题**
  （实测 phones ≤ 3、同帧人 ≤ 12，`Nm_phone=5` / `Nm_camera=15` 从未饱和）。
  需要 per-sequence 看：IMU 信号是否失效、相机视角是否与训练 scene 差异大、
  passersby 占比是否过高拖累 attention。
