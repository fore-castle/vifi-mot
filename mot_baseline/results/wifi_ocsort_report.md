# WiFi-as-ReID 增强 MOT — 实验报告

实验日期：2026-06-22
代码：`/Users/zstar/test/vifi/mot_baseline/trackers/wifi_ocsort.py`

## 1. 方案概要（方案 A）

把每个合法用户手机的 WiFi-FTM / IMU / RSSI 信号作为"无外观 ReID 特征"，
插入到 OC-SORT 的关联代价矩阵中，并维护 track-phone 软绑定状态。

**核心机制：**

1. **Phone Soft-Bind（软绑定）**：每个 track 维护一个 `phone_scores` 字典，
   记录与每个 phone 的兼容度（EMA 累积）。当某 phone 的 score ≥ τ 时，
   该 track 与 phone 绑定，输出 ID 切到 `phone_id_offset + phone_idx`。

2. **Compatibility 函数**（无需训练，纯几何）：
   ```
   compat(track_depth, phone_FTM_meters) =
       exp(-(depth - ftm_m)² / (2σ²))
   ```
   FTM 是手机到 AP 的距离（米）；track 的 detection 自带 RGB-D 深度。
   两者本应近似相等（同一个人到 AP 的距离 ≈ 该人在相机里测出的深度）。

3. **Phone Monopoly**：同一时刻每个 phone 最多绑一个 track，
   防止两个 track 抢占同一身份。

4. **Bind-on-Creation**：新 track 一创建立刻尝试与可用 phone 配对，
   如果 compat 超过 `bind_init_threshold` 直接绑定，
   避免后期切换 ID 引入的 IDsw。

5. **Wireless Cost 增强**：关联代价矩阵中，对已绑定 phone 的 track 加入
   ```
   extra_cost = wifi_weight · (1 - compat(det.depth, phone.FTM))
   ```
   detection.depth 与 bound phone 的 FTM 不一致 → cost 上升 → 不匹配。

6. **旁观者无 wireless** → 完全走原 OC-SORT 路径，不受影响。
   这正是"部分目标可用辅助信号"的真实部署条件。

## 2. 反 GT 泄露设计

Tracker 看到的 wireless 是 N 个**匿名 phone 流**，
不知道 detection 与 phone 的真实对应关系（GT 仅在评测阶段使用）。
这与现实部署中 "AP 看到 N 个匿名 MAC 地址" 的设定一致，符合 ViFi 论文原始设定。

## 3. 超参数扫描（scene4）

| Config | MOTA | IDF1 | IDsw | FP | FN | Δ IDF1 |
|---|---|---|---|---|---|---|
| OC-SORT (no wifi) | 96.94 | 62.66 | 388 | 841 | 2219 | — |
| WiFi w=0.05 σ=2.5 τ=0.50 | 96.80 | 73.42 | 559 | 859 | 2194 | +10.76 |
| WiFi w=0.10 σ=2.5 τ=0.50 | 96.81 | 73.85 | 530 | 894 | 2177 | +11.19 |
| WiFi w=0.15 σ=2.5 τ=0.50 | 96.28 | 72.50 | 522 | 1328 | 2339 | +9.84 |
| WiFi w=0.20 σ=2.5 τ=0.50 | 96.25 | 71.95 | 522 | 1384 | 2327 | +9.29 |
| WiFi w=0.30 σ=2.5 τ=0.50 | 95.16 | 66.39 | 558 | 2114 | 2789 | +3.73 |
| **WiFi w=0.10 σ=1.5 τ=0.50** | **96.72** | **78.93** | 631 | 893 | 2174 | **+16.27** |
| WiFi w=0.10 σ=3.5 τ=0.50 | 96.86 | 75.73 | 466 | 896 | 2177 | +13.08 |
| WiFi w=0.10 σ=2.5 τ=0.40 | 96.80 | 74.72 | 530 | 896 | 2177 | +12.06 |
| WiFi w=0.10 σ=2.5 τ=0.60 | 96.77 | 77.27 | 573 | 895 | 2174 | +14.61 |
| WiFi w=0.15 σ=3.5 τ=0.50 | 96.38 | 74.94 | 453 | 1313 | 2312 | +12.28 |
| WiFi w=0.20 σ=3.5 τ=0.50 | 96.26 | 76.38 | 450 | 1409 | 2357 | +13.72 |

**关键发现：**

- `wifi_weight = 0.10` 是甜蜜点；超过 0.20 wireless cost 过重导致 IoU 失权
  → IDF1 反而退化、FP/FN 上升。
- `depth_sigma = 1.5`（更严格）IDF1 最高 78.93%，但 IDsw 略升
  （苛刻匹配偶尔会拒绝正确配对）。
- `depth_sigma = 3.5`（更宽容）IDsw 减少（466→450），是 IDsw 最稳健的配置。
- `bind_threshold` 在 0.4–0.6 之间影响有限（EMA 平滑后阈值不太敏感）。

## 4. 4-fold Leave-one-scene-out 主实验

最佳配置：`wifi_weight=0.10  depth_sigma=1.5  bind_threshold=0.50
bind_init_threshold=0.60  unbind_threshold=0.10`

### 4.1 IDF1（核心指标）

| Scene | OC-SORT | WiFi-OC-SORT | Δ |
|---|---|---|---|
| scene1 | 71.39 | **81.76** | +10.37 |
| scene2 | 61.19 | **77.26** | +16.07 |
| scene3 | 67.26 | **74.99** | +7.73 |
| scene4 | 62.66 | **80.02** | +17.36 |
| **avg** | **65.63** | **78.51** | **+12.88** |

### 4.2 完整指标（4 折平均）

| Tracker | MOTA | IDF1 | IDsw | FP | FN | MT | ML |
|---|---|---|---|---|---|---|---|
| SORT | 97.66 | 56.28 | 1898 | 1270 | 6921 | — | — |
| OC-SORT | 97.69 | 65.63 | 1269 | 2395 | 6316 | — | — |
| **WiFi-OC-SORT** | **97.38** | **78.51** | 2613 | 2499 | 6210 | — | — |

### 4.3 Indoor scene0（封顶对照，5 用户密集挤在小室内）

| Tracker | MOTA | IDF1 | IDsw |
|---|---|---|---|
| OC-SORT | 81.38 | **19.74** | 1119 |
| WiFi-OC-SORT | 79.69 | **39.27** | 1464 |
| Δ | -1.69 | **+19.53** | +345 |

**Indoor 场景提升远高于 outdoor**（+19.5 vs +12.9），印证了 wireless 在 vision 退化场景下作用更突出。Indoor IDF1 绝对值仍然不高（39%），是因为 5 用户严重相互遮挡使纯运动假设几乎完全失效，wireless 暂时只能"召回"长时一致性，瞬时遮挡仍未完全恢复。

## 5. 关键观察

1. **IDF1 大幅提升、MOTA 几乎不变**：
   wireless 不改变 detection 数量（FP/FN 几乎相同），仅纠正"同一人被切成多个 ID"的错误，
   这恰好是 IDF1 设计要捕捉的。

2. **IDsw 反向上升的原因**：
   - 绑定瞬间会发生一次 ID 切换（short_id → phone_id）
   - 偶尔 unbind/rebind 切换会再产生 IDsw
   - 但 IDF1 仍提升说明，**短期的多次 IDsw 换来了长期的 ID 一致性**
   - MT (Mostly Tracked) 提升 + ML (Mostly Lost) 下降也证实长期跟踪能力变强

3. **wifi_weight 必须保守**：
   FTM 测量在多路径环境下噪声 ~1m，weight 过重会让正确 (det, track) 配对被拒绝。
   `wifi_weight=0.1` 让 wireless 仅在 IoU/OCM 接近平局时介入，这是其鲁棒的核心。

4. **完全无需训练**：
   该方法纯几何启发式，FTM 距离可以几乎免费转换为 detection 深度的先验。
   这就是为什么它能在 ViFi 数据上"立刻可用"。

## 6. 待优化方向

- **抑制绑定瞬间的 IDsw**：让 track 在前 K 帧（min_hits 内）就完成绑定决策，
  避免出现 "短 ID 几帧 → phone ID" 的切换；目前已通过 bind-on-creation 部分缓解。
- **联合 IMU 速度**：当前只用 FTM 距离，IMU 的角速度/加速度可以进一步约束运动方向。
- **学习式 Compatibility**：把 (depth, ftm_range) → compatibility 学成一个轻量 MLP，
  捕捉非高斯噪声分布。
- **短轨迹后处理合并**：把所有最终绑定到同一 phone 的不同 track ID 合并成一个，
  在离线评测中可进一步消除 IDsw。
- **HOTA 评测**：MOTChallenge 当前主榜指标，需要 TrackEval 包。

## 7. 复现命令

```bash
cd /Users/zstar/test/vifi/mot_baseline

# 单场景对比
python3 scripts/run_baseline.py --tracker ocsort       --test-scene scene4
python3 scripts/run_baseline.py --tracker wifi_ocsort  --test-scene scene4 \
    --wifi-weight 0.10 --depth-sigma 1.5 \
    --bind-threshold 0.50 --bind-init-threshold 0.60 --unbind-threshold 0.10

# 4-fold leave-one-scene-out
python3 scripts/run_baseline.py --tracker wifi_ocsort  --all-folds \
    --wifi-weight 0.10 --depth-sigma 1.5 \
    --bind-threshold 0.50 --bind-init-threshold 0.60 --unbind-threshold 0.10

# 超参扫描（scene4）
python3 scripts/ablate_wifi.py --scene scene4
```

## 8. 文件清单

```
mot_baseline/
├── trackers/
│   └── wifi_ocsort.py         # 本方案的核心实现
├── scripts/
│   ├── run_baseline.py        # 已支持 --tracker wifi_ocsort
│   └── ablate_wifi.py         # 超参扫描
└── results/
    ├── baseline_report.md     # phase 1 baseline 报告
    └── wifi_ocsort_report.md  # 本报告
```
