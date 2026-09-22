# WiFi-as-ReID MOT 实验总结

> **作者**：zstar（整理：QoderWork）  
> **实验时间**：2026-06-22  
> **数据集**：RAN4model_dfv4p4（ViFi 论文同步预处理版）  
> **代码目录**：`/Users/zstar/test/vifi/mot_baseline/`  
> **相关详细报告**：`baseline_report.md`（§2）, `wifi_ocsort_report.md`（§3）,
> `wifi_affinity_report.md`（§4–§7）

---

## TL;DR

把 ViFi 论文复现的"WiFi + 视觉亲和矩阵学习模型"接入 MOT tracker，作为 wireless ReID 信号，
**看起来在 4-fold 平均 IDF1 上比几何版 +12.4 pt（78.51% → 90.91%），
但经严格 leave-one-scene-out 重训后落到 75.62%，仍比几何版低 2.89 pt**。

| 版本 | outdoor 4-fold avg IDF1 | Δ vs 几何版 A |
|---|---:|---:|
| OC-SORT 裸跑 | 65.63% | −12.88 |
| WiFi-Geom (方案 A，纯几何，无需训练) | **78.51%** | — |
| WiFi-Affinity (方案 D, leaky 4-fold) | 90.91% | +12.40 |
| **WiFi-Affinity (方案 D, strict LOSO)** | **75.62%** | **−2.89** |

**结论**：学习版模型在跨场景泛化上**尚未超过人工几何启发式**，
但其域内表现极强（scene1 strict IDF1 = 87.18%），
**A+D ensemble 是下一步最有价值的方向**。

---

## 1. 任务与数据集

**任务定义**  
把 ViFi 的多人 WiFi 定位数据集重新解释为 MOT：

- **输入**：每帧 RGB-D 相机 BBX5 检测框（合法用户 + 旁观者 passersby），
  无 RGB 外观、无检测置信度；外加每个手机的 FTM（距 AP 距离，米）、FTM 标准差、
  9 维 IMU（agm9，加速度+陀螺+磁力计）
- **输出**：跨帧一致的 track ID 序列
- **GT**：BBX5 的 `subj_idx`（合法用户 `L{i}`）和 `Others_id_ls`（旁观者 `O{j}`）

**数据集结构**（RAN4model_dfv4p4）

| Scene | 类别 | 序列数 | 帧数 | 每序列 phones | 同帧最多人数 | 备注 |
|---:|---:|---:|---:|---:|---:|---|
| scene0 | indoor | 15 | ~26k | 5 | 7 | OOD 测试集 |
| scene1 | outdoor | 16 | ~29k | 2 | 12 | LOSO fold 1 |
| scene2 | outdoor | 18 | ~32k | 3 | 11 | LOSO fold 2 |
| scene3 | outdoor | 18 | ~32k | 3 | 11 | LOSO fold 3 |
| scene4 | outdoor | 15 | ~27k | 3 | 11 | LOSO fold 4 |
| **合计** | | **82** | **~146k** | | | |

评测采用 **leave-one-scene-out 4-fold**（outdoor），
每次 4 个 outdoor scene 的 1 个做 test、其余 3 个 + indoor 做训练。

---

## 2. 方法总览

### 2.1 Tracker 基线

所有 tracker 共享同一份 7 状态恒速 Kalman + Hungarian IoU 匹配，
主要差别在关联策略和 ID 维护：

| Tracker | 关键差异 | 关键参数 |
|---|---|---|
| SORT | 单轮 IoU 匹配 | iou=0.3, max_age=30, min_hits=3 |
| OC-SORT | + 观测中心更新、动量、轨迹恢复 | inertia=0.2, delta_t=3 |
| ByteTrack | 高/低分两阶段（需置信度） | track_thresh=0.5, new=0.6 |

### 2.2 WiFi-Geom (方案 A，纯几何，无需训练)

把每个手机作为无外观 ReID 信号，**不需要训练**：

- **Phone 软绑定**：每个 track 维护 `phone_scores` 字典，用 EMA 累积与每个 phone 的兼容度
- **Compatibility 函数**：  
  `compat(depth, FTM) = exp(-(depth − FTM)² / (2σ²))`
  （depth = RGB-D 深度，FTM = 手机到 AP 距离，两者理论一致）
- **Phone Monopoly**：同一时刻每个 phone 最多绑一个 track
- **Bind-on-Creation**：新 track 创建立刻尝试绑定，减少后期 ID 切换
- **旁观者不受影响**：无 wireless 信号，走原 OC-SORT 路径

### 2.3 WiFi-Affinity (方案 D，学习版)

把 my_vifi 已复现的 ViFi Bi-LSTM 亲和矩阵模型作为"软 ReID encoder"，
**替换方案 A 的高斯 compatibility**，其他机制（软绑定、Phone Monopoly、bind-on-creation）保留。

**模型架构**（约 90k 参数）：

- 相机分支：Bi-LSTM(2 层, hidden=32) on (cx, cy, depth) × k=10 帧
- 手机分支：Bi-LSTM(2 层, hidden=32) on (FTM, FTM_std, IMU9) × k=10 帧
- 1×1 Conv × 4 层（128→64→32→16→1）压成 Affinity 张量 (Np+1, Nc+1)
- Row-softmax + Col-softmax 平均得到对称化软概率

推理成本：CPU 4× OC-SORT（每帧 forward 一次 Bi-LSTM）。

---

## 3. 主实验结果

### 3.1 Tracker 基线对比（4-fold avg outdoor）

| Tracker | MOTA | IDF1 | IDsw | FP | FN | 推理 |
|---|---:|---:|---:|---:|---:|---:|
| SORT | 97.66% | 56.28% | 1898 | 1270 | 7321 | ~5s |
| OC-SORT | 97.69% | **65.63%** | 1269 | 2395 | 6316 | ~9s |
| ByteTrack | 97.45% | 60.13% | 1751 | 2919 | 6353 | ~10s |

→ **OC-SORT 作为 baseline**（IDF1 最高，IDsw 最低）。

### 3.2 WiFi-Geom 方案 A（vs OC-SORT baseline）

| Scene | OC-SORT IDF1 | WiFi-Geom IDF1 | Δ |
|---:|---:|---:|---:|
| scene1 | 71.39 | 81.76 | +10.37 |
| scene2 | 61.19 | 77.26 | +16.07 |
| scene3 | 67.26 | 74.99 | +7.73 |
| scene4 | 62.66 | 80.02 | +17.36 |
| **avg outdoor** | **65.63** | **78.51** | **+12.88** |
| scene0 (indoor) | 19.74 | 39.27 | +19.53 |

**优势**：纯几何、无需训练、跨域稳健、对 IMU 失效有回退路径。  
**代价**：IDsw 增加 106%（2613 vs 1269），EMA 累积带来的 bind/unbind 抖动。

### 3.3 WiFi-Affinity 方案 D（leaky 4-fold，乐观估计）

用 my_vifi 作者提供的单一 fold1 ckpt（训练集包含 scene1-4 中 ~70/75 序列）
在 4 个 outdoor scene 上推理：

| Scene | OC-SORT | WiFi-Geom (A) | WiFi-Affinity (D, leaky) | Δ vs OC-SORT | Δ vs A |
|---:|---:|---:|---:|---:|---:|
| scene1 | 71.39 | 81.76 | **91.21** | +19.82 | +9.45 |
| scene2 | 61.19 | 77.26 | **91.70** | +30.51 | +14.44 |
| scene3 | 67.26 | 74.99 | **92.60** | +25.34 | +17.61 |
| scene4 | 62.66 | 80.02 | **88.11** | +25.45 | +8.09 |
| **avg outdoor** | **65.63** | **78.51** | **90.91** | **+25.28** | **+12.40** |
| scene0 (indoor) | 19.74 | 39.27 | 37.81 | +18.07 | −1.46 |

注意：IDF1 +25 pt 的同时 **MOTA 仅降 0.24 pt，FP/FN 与 OC-SORT 完全一致**——
affinity 模型只影响 phone 软绑定，**完全不修改 detection-track 关联**。

**但这张表有严重 train-test leakage**：
同一个 fold1 ckpt 的训练集包含了 4 个 outdoor scene 中绝大多数序列，
推理时 ~70/75 个测试序列模型都见过。91% 的数字只是乐观上界。

### 3.4 WiFi-Affinity 方案 D（strict LOSO，本次核心贡献）

为得到真实跨场景泛化能力，按 scene 切分重训 4 个独立 fold：

**协议**：

- fold k：测试集 = scene k 全部序列；训练集 = scene0 + scene{1..4}\{k}
- 每 fold 独立计算 norm_stats、独立训 30 epoch（CPU，bs=64，SGD+MultiStep）
- 选 best val checkpoint（val_ratio=0.1, val_seed=2026）
- 在测试 scene 上跑 wifi_affinity tracker（同一组超参）

**训练健康度**（val_acc，仅作训练参考）：  
fold1=85.25%, fold2=87.11%, fold3=86.32%, fold4=85.36%。

**严格 4-fold 结果**：

| Fold | Test scene | MOTA | IDF1 | IDsw | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | scene1 | 97.38% | **87.18%** | 568 | 527 | 1345 |
| 2 | scene2 | 97.18% | **80.81%** | 1101 | 615 | 1386 |
| 3 | scene3 | 97.72% | **69.68%** | 864 | 412 | 1366 |
| 4 | scene4 | 96.09% | **64.81%** | 1349 | 841 | 2219 |
| **avg outdoor** | — | **97.09%** | **75.62%** | 3882 | 2395 | 6316 |

### 3.5 综合对比（outdoor 4-fold avg）

| 版本 | MOTA | IDF1 | Δ vs A |
|---|---:|---:|---:|
| OC-SORT | 97.69 | 65.63 | −12.88 |
| WiFi-Geom (方案 A) | 97.38 | 78.51 | 0.00 |
| WiFi-Affinity (方案 D, leaky) | 97.45 | 90.91 | +12.40 |
| **WiFi-Affinity (方案 D, strict LOSO)** | **97.09** | **75.62** | **−2.89** |

---

## 4. 核心发现

1. **学习版域内潜力巨大**：scene1 strict IDF1 = 87.18%（**真的超过了几何版 81.76%**），
   证明 Bi-LSTM + IMUagm9 时序建模在分布匹配时能学到比单帧 depth-FTM 高斯更强的表征。

2. **Leakage 解释了 +15 pt 的虚增**：leaky 90.91% → strict 75.62%，
   **实际跨场景泛化能力远低于"看起来很强"的乐观数字**。

3. **严格 LOSO 仍未超过几何版**：strict D = 75.62 < A = 78.51，
   当前 ~90k 参数小模型在新场景上的几何线索利用还不如人工设计的 EMA + bind 启发式。

4. **Fold 间方差极大**：scene1 strict = 87.18% 而 scene4 = 64.81%（几何版在 scene4 是 80.02%）。
   实测**不是模型容量问题**——全数据集 phones ≤ 3、同帧人 ≤ 12，
   `Nm_phone=5` / `Nm_camera=15` 从未饱和。scene4 的失败模式需要 per-sequence 误差分析。

5. **MOTA 稳健、FP/FN 不引入**：所有 strict fold MOTA ≥ 96%，FP/FN 与 OC-SORT 完全一致。
   软绑定架构保证了 affinity 模型的错误不会回灌到 detection-track 关联。

6. **跨域退化明显**：scene0 indoor 用 leaky fold1 ckpt 测得 IDF1 = 37.81%，
   低于几何版 39.27%。训练分布外的场景，学习版会显著退化，与 strict scene4 模式一致。

---

## 5. 优势与局限

**优势**

- 利用 wireless 信号的时间结构（IMU 时序、FTM 时序）：单帧几何只能用 depth 一个标量
- 学习到的非高斯噪声模型：FTM 多路径偏差不再是简单高斯
- 域内逼近 90% IDF1，远超几何版上限 78.5%
- 完全不修改 OC-SORT 的关联代价矩阵 → 不引入额外 FP/FN，MOTA 几乎不动

**局限**

- 需要预训练 checkpoint + norm_stats（部署成本变高）
- CPU 推理 4× 慢于纯 OC-SORT（每帧 forward 一次 Bi-LSTM）
- 跨场景泛化不足（strict LOSO 低于几何版 2.89 pt），跨域（outdoor → indoor）退化明显
- 依赖 IMUagm9 信号；缺失或噪声大时回退弱于几何版的 depth-only 启发式

---

## 6. 复现命令

```bash
cd /Users/zstar/test/vifi/mot_baseline

# 1) Tracker 基线（OC-SORT）4-fold 评测
python3 scripts/run_baseline.py --tracker ocsort --all-folds

# 2) 方案 A（WiFi-Geom）单场景 / 全 fold
python3 scripts/run_baseline.py --tracker wifi_ocsort --test-scene scene4 \
    --wifi-weight 0.10 --bind-threshold 0.50 \
    --bind-init-threshold 0.60 --unbind-threshold 0.10

# 3) 方案 D（WiFi-Affinity）leaky 4-fold（注意：同一 fold1 ckpt，存在 leakage）
python3 scripts/precompute_norm_stats.py
python3 scripts/run_baseline.py --tracker wifi_affinity --all-folds

# 4) 方案 D strict LOSO：按 scene 切，重训 4 fold（CPU ~33min/fold，总 ~2.2h）
python3 scripts/loso_train.py --folds 1,2,3,4 --epochs 30 --batch-size 64
python3 scripts/loso_eval.py --folds 1,2,3,4

# 5) Indoor OOD（leaky ckpt）
python3 scripts/run_baseline.py --tracker wifi_affinity --test-scene scene0
```

---

## 7. 产物清单

| 类型 | 路径 |
|---|---|
| LOSO 训练脚本 | `scripts/loso_train.py` |
| LOSO 评测脚本 | `scripts/loso_eval.py` |
| LOSO ckpt | `checkpoints_loso/full/fold{1..4}_best_epoch{N}_acc{X}.pth` |
| LOSO norm_stats | `data/affinity_norm_stats_loso_fold{1..4}.npz` |
| LOSO 原始结果 | `exps/wifi_affinity_loso_fold{1..4}_scene{N}.json` |
| LOSO 汇总 | `exps/wifi_affinity_loso_summary.json` |
| 训练日志 | `exps/loso_train.log` |

---

## 8. 后续工作（按优先级）

1. **A+D ensemble（首选）**：取 affinity softmax 最大概率 > τ 时用 D 绑定，否则回退 A。
   预期能同时保留 A 的跨域鲁棒性和 D 的域内高精度，缩小甚至抹平 Δ vs A。
2. **训练分布扩展**：indoor + outdoor 混合训练，缓解跨域退化。
3. **Scene4 误差分析**：per-sequence 追踪绑定轨迹，定位 IMU 失效 / 视角差异 / passersby 干扰。
4. **Indoor OOD 重测**：用 strict LOSO ckpt 重测 scene0，得到真 OOD 数字
   （当前 37.81% 仍是 leaky fold1 ckpt 上的）。
5. **推理加速**：导出 ONNX / TorchScript，看是否能压到 OC-SORT 同档。
6. **模型扩展**（谨慎）：更长训练 / 更大容量，但要警惕 overfit 到训练 scene。

---

## 附录 A：场景-序列-phone 数量实测

为避免后续误判"模型容量饱和"，专门做了数据集 sanity check：

| Scene | 序列数 | 每序列 phone 数 | 同帧 BBX 峰值 | 同帧路人峰值 | 同帧总人峰值 |
|---:|---:|---:|---:|---:|---:|
| indoor/scene0 | 15 | 5 | 5 | 3 | 7 |
| outdoor/scene1 | 16 | 2 | 2 | 10 | 12 |
| outdoor/scene2 | 18 | 3 | 3 | 6 | 9 |
| outdoor/scene3 | 18 | 3 | 3 | 6 | 9 |
| outdoor/scene4 | 15 | 3 | 3 | 8 | 11 |

- `Nm_phone=5` / `Nm_camera=15` **全场景不饱和**
- 不要把 motmetrics 的 "GT IDs" 累计计数误解为"同帧并发人数"——
  `20211007_143810` 的 187 个 GT IDs 只是路人进出刷新 ID 的累计，与同帧人数无关

---

## 附录 B：关键超参

所有 WiFi 实验统一使用：

```
wifi_weight = 0.10
ema_alpha = 0.85
bind_threshold = 0.50
bind_init_threshold = 0.60
unbind_threshold = 0.10
```

训练超参（LOSO）：

```
epochs = 30
batch_size = 64
optimizer = SGD, lr=1e-3, momentum=0.9
scheduler = MultiStepLR, milestones=[15], gamma=0.1
val_ratio = 0.1, val_seed = 2026
norm_stats = per-fold (从训练序列抽样 ~2000 windows 估计)
```

---

**TL;DR 总结**：WiFi 信号作为无外观 ReID 对 MOT 有显著提升作用，
但学习版模型在跨场景泛化上**尚未超过人工几何启发式**。
下一步应做 A+D ensemble：让学习版在分布匹配的场景主导绑定，
几何版在分布外场景兜底。
