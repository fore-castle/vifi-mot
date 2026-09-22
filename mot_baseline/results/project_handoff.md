## Vi-Fi MOT 项目交接文档

> 生成日期：2026-07-07
> 项目路径：`/Users/zstar/test/vifi/mot_baseline`
> 数据集路径：`/Users/zstar/test/vifi/RAN4model_dfv4p4`
> 目标会议：ACM MM 2027

---

### 一、项目概述

本项目研究 WiFi FTM 信号辅助的多目标跟踪（Visual MOT + WiFi Fusion）。核心思路是用 WiFi 测距（FTM）与单目深度估计（depth）的匹配关系，将视觉跟踪 tracklet 绑定到手机持有者，从而实现 phone-stable 的身份输出。

**当前最佳方法**（Phase 5 + 6A + 6B 联合）：

| 指标 | OC-SORT (bare) | 当前最佳 | 提升 |
|---|---|---|---|
| IDF1 | 65.63% | **87.91%** | +22.28pt |
| IDsw | 1,269 | 556 | −56% |
| MOTA | 97.69% | 97.47% | ≈ 不变 |

**评测协议**：Leave-One-Scene-Out (LOSO) 4-fold 交叉验证，scene1-4 轮流作测试集。

---

### 二、累积增益路径

```
OC-SORT (65.63%)                                    ← 纯运动 baseline
  ↓ +14.73pt
Phase 5: Global Phone-Track Assignment (80.36%)      ← Hungarian 联合分配
  ↓ +2.81pt
Phase 6A: FTM Denoising + σ Tuning (83.17%)          ← CausalDilatedCNN 去噪
  ↓ +4.74pt
Phase 6B: Learned Matching MLP (87.91%)              ← 257 params MLP 替代 Gaussian
```

每个阶段的增益互相正交：全局分配优化"如何利用信号"，去噪优化"信号质量本身"，学习匹配优化"信号的匹配函数"。

---

### 三、目录结构总览

```
mot_baseline/
├── trackers/          # 跟踪器实现（12 个 .py，3840 行）
│   ├── wifi_joint.py  # ★ 当前主力 tracker（Phase 5+6B）649行
│   ├── ocsort.py      # OC-SORT baseline
│   ├── sort.py        # SORT baseline
│   ├── bytetrack.py   # ByteTrack（不适用，已废弃）
│   ├── wifi_ocsort.py # Method A: WiFi 几何绑定
│   ├── wifi_affinity.py # Method D: BiLSTM affinity（过拟合）
│   ├── reliability_gate.py # Phase 3: 可靠性门控
│   ├── spatial_projector.py # Phase 1: 3D 空间投影
│   ├── wifi_spatial_ocsort.py # Phase 1+2: 空间+EKF+桥接
│   ├── wifi_ocsort_merger.py # Phase 2: post-hoc merger
│   └── utils.py       # KalmanBox, IoU, Hungarian
├── denoising/         # FTM 去噪模块（4 个 .py，1154 行）
│   ├── models.py      # 3 种架构定义
│   ├── dataset.py     # 滑窗数据集构造
│   ├── train.py       # 训练脚本
│   └── inference.py   # FTMDenoiser（在线）+ FTMDenoiserBatch（离线）
├── data/
│   ├── vifi_mot.py    # 数据加载（SequenceMeta, Detection, Frame, list_sequences, load_sequence）
│   └── *.npz          # 归一化统计（LOSO 4-fold）
├── eval/
│   └── mot_eval.py    # Evaluator: 封装 py-motmetrics（MOTA/IDF1/IDsw）
├── scripts/           # 实验脚本（22 个 .py，5462 行）
│   ├── run_baseline.py          # 主入口：跑 baseline + noise injection
│   ├── phase6b_integrated_mot.py # ★ 当前最佳：去噪+学习匹配+MOT
│   ├── phase6b_learned_matching.py # Phase 6B 匹配模型训练
│   ├── phase6_denoise_mot.py    # Phase 6A: 去噪+MOT 集成
│   ├── phase6_oracle_mot.py     # Oracle 实验
│   ├── phase6_noise_analysis.py # FTM 噪声分析
│   ├── loso_train.py / loso_eval.py # Method D 的 LOSO 训练/评测
│   └── ...（Phase 0-2 分析脚本）
├── checkpoints_denoising/  # 12 个去噪 checkpoint（3 model × 4 fold）
├── checkpoints_matching/   # 4 个匹配 checkpoint（1 model × 4 fold）
├── checkpoints_loso/full/  # 4 个 Method D checkpoint（已废弃）
├── exps/                   # 62 个 JSON 实验结果
└── results/                # 14 个 Markdown 分析报告
    └── experiment_summary.md # ★ 总汇总（45K，843行）
```

---

### 四、核心代码 API 速查

#### 4.1 数据加载

```python
from data.vifi_mot import list_sequences, load_sequence, Frame

# 列出所有 outdoor 序列
seqs = list_sequences(scenes=["scene1"])  # 可选过滤
seqs = [s for s in seqs if s.kind == "outdoor"]

# 加载一序列
frames = load_sequence(meta)  # -> List[Frame]
# Frame.frame_id, Frame.detections (List[Detection])
# Frame.ftm (N,2), Frame.detections[i].depth, .gt_track_id, .bbox
```

数据集根路径硬编码在 `data/vifi_mot.py:44`：
```python
DATASET_ROOT = "/Users/zstar/test/vifi/RAN4model_dfv4p4"
```

#### 4.2 WiFi Joint Tracker（主力）

```python
from trackers.wifi_joint import WiFiJointTracker

tracker = WiFiJointTracker(
    max_age=30, min_hits=3, iou_threshold=0.3,
    delta_t=3, inertia=0.2,
    wifi_weight=0.10,           # WiFi 在 cost matrix 中的权重
    depth_sigma=1.3,            # ★ 去噪后用 1.3（raw 用 1.5）
    ema_alpha=0.85,
    bind_init_threshold=0.7,
    unbind_threshold=0.15,
    assign_threshold=0.40,
    no_ghost_pool=True,         # ghost pool 当前禁用
    matching_model=None,        # ★ Phase 6B: 传入 MatchingMLP 实例
)

# 每帧更新
outputs = tracker.update(
    dets_xyxy,     # (D,4) 检测框
    det_depths,    # (D,) 深度值（米）
    scores,        # (D,) 检测分数
    ftm_m,         # (P,) FTM 测距（米）——去噪后的
    ftm_valid,     # (P,) bool mask
    ftm_std_m,     # (P,) FTM std（米）——Phase 6B 需要
)
# -> List[(track_id, bbox_xyxy)]
```

#### 4.3 FTM 去噪（在线 + 离线）

```python
from denoising.inference import FTMDenoiser, FTMDenoiserBatch

# 在线：frame-by-frame（MOT 集成用）
denoiser = FTMDenoiser("causal_cnn", test_scene="scene1", window_size=15)
denoiser.reset()  # 新序列开始时
denoised = denoiser.denoise(ftm_m, ftm_valid, ftm_std_m)

# 离线：整序列去噪（分析/训练用）
batch = FTMDenoiserBatch("causal_cnn", test_scene="scene1", window_size=15)
denoised = batch.denoise_sequence(ftm_series, std_series, valid_series)
```

Checkpoint 路径：`checkpoints_denoising/{model}_{scene}_best.pth`
命名约定：`causal_cnn` / `causal_lstm` / `bilstm` × `scene1-4`

#### 4.4 学习匹配 MLP（Phase 6B）

```python
from scripts.phase6b_learned_matching import MatchingMLP
import torch

model = MatchingMLP(in_features=6, hidden=16)  # 257 params
ckpt = torch.load("checkpoints_matching/matching_scene1_best.pth",
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 输入: [Δ, |Δ|, depth, denoised_ftm, ftm_std, Δ²/(2σ²)]
# 输出: [0,1] 匹配概率
```

#### 4.5 MOT 评测

```python
from eval.mot_eval import Evaluator

evaluator = Evaluator()
evaluator.add_sequence(seq_name, gt_seq, hyp_seq)
# gt_seq/hyp_seq: List[(frame_id, List[(track_id, (x,y,w,h))])]
summary = evaluator.summarize()  # -> DataFrame with OVERALL row
idf1 = summary.loc["OVERALL", "idf1"] * 100
```

---

### 五、实验阶段总表

| 阶段 | 时间 | 核心方法 | avg IDF1 | Δ vs 上一步 | 状态 |
|---|---|---|---|---|---|
| OC-SORT | 06-19 | 纯运动 baseline | 65.63% | — | ✅ |
| Method A | 06-20 | WiFi EMA 几何绑定 | 78.51% | +12.88pt | ✅ 但 IDsw 恶化 |
| Method D (strict) | 06-22 | BiLSTM 90k params | 75.62% | −2.89pt | ❌ 过拟合 |
| Phase 0 | 06-26 | 数据可行性验证 | — | — | ✅ 全部 PASS |
| Phase 1 | 06-26 | 3D Spatial Radio Prior + EKF | 78.84% | +0.33 vs A | ⚠️ 增益微小 |
| Phase 2 | 06-26 | Wireless Tracklet Bridging | 76.42% | −2.25 vs A | ⚠️ phone 换手 |
| Phase 3 | 06-26 | Reliability-Gated WiFi | 77.53% | −1.14 vs A | ⚠️ FTM 噪声是硬约束 |
| Phase 4 | 06-26 | Detection Noise Robustness | — | WiFi 噪声下更有价值 | ✅ 补充实验 |
| **Phase 5** | 06-27 | **Global Phone-Track Assignment** | **80.36%** | **+3.04 vs A** | ✅ **首个突破** |
| **Phase 6A** | 06-30 | **CausalDilatedCNN 去噪** | **83.17%** | **+2.81** | ✅ |
| **Phase 6B** | 07-01 | **Learned Matching MLP** | **87.91%** | **+4.74 vs P5** | ✅ **当前最佳** |

---

### 六、关键文件速查表

#### 6.1 跑实验的命令

```bash
cd /Users/zstar/test/vifi/mot_baseline

# 当前最佳配置（Phase 5 + 6A + 6B）
python3 -u scripts/phase6b_integrated_mot.py

# Phase 6A only（去噪 + Gaussian matching）
python3 -u scripts/phase6_denoise_mot.py --model causal_cnn --wifi-weight 0.10

# 训练匹配模型
python3 -u scripts/phase6b_learned_matching.py

# Raw baseline
python3 -u scripts/run_baseline.py --tracker ocsort
```

#### 6.2 依赖说明

| 包 | 用途 | 安装方式 |
|---|---|---|
| torch | 去噪/匹配模型 | `pip3 install torch` |
| numpy | 数值计算 | 系统自带 |
| scipy | Hungarian 算法 | `pip3 install scipy` |
| motmetrics | MOT 评测 | `pip3 install motmetrics` |
| pandas | 数据处理 | `pip3 install pandas` |

**NumPy 2.0 兼容 shim**（必须在 `import motmetrics` 之前）：
```python
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
```

#### 6.3 Checkpoint 清单

| 目录 | 文件命名 | 数量 | 大小 |
|---|---|---|---|
| `checkpoints_denoising/` | `{model}_{scene}_best.pth` | 12 | ~1.1 MB |
| `checkpoints_matching/` | `matching_{scene}_best.pth` | 4 | ~17 KB |
| `checkpoints_loso/full/` | `fold{N}_best_epoch{E}_acc{A}.pth` | 4 | ~1.5 MB（已废弃） |

#### 6.4 关键 JSON 结果文件

| 文件 | 内容 |
|---|---|
| `exps/phase6b_integrated_mot.json` | ★ 当前最佳：Gaussian vs Learned MOT 对比 |
| `exps/phase6b_matching.json` | 匹配质量评估（gap, AUC） |
| `exps/phase6_denoise_mot.json` | 去噪 MOT sigma sweep |
| `exps/phase6_denoising_results.json` | 去噪模型训练结果 |
| `exps/phase6_oracle_mot.json` | Oracle 实验（理论上限） |
| `exps/phase6_feature_extension.json` | IMU 特征扩展（无效） |
| `exps/wifi_joint_scene*_constant.json` | Phase 5 逐场景结果 |

---

### 七、已知问题与注意事项

1. **DATASET_ROOT 硬编码**：`data/vifi_mot.py:44` 和各 `phase6*.py` 脚本中都有硬编码路径，迁移机器需修改。

2. **Ghost pool 已禁用**：`no_ghost_pool=True` 是当前最佳配置。Ghost pool 重连与全局分配冲突，IDF1 退化约 2pt。

3. **Method D (BiLSTM affinity) 已废弃**：90k 参数过拟合，LOSO 泛化差（75.62% < Method A 78.51%），22pt fold 方差。

4. **IMU/RSSI 特征无效**：Phase 6A-ext 实验证明 IMU 加速度与 FTM 噪声相关性 ≈ 0（r < 0.03），加入后去噪效果反而下降。

5. **scene4 是最难场景**：大量路人（Others），FTM 噪声大。但当前方法在 scene4 仍达 85.42%（vs OC-SORT 62.66%）。

6. **GT detection 不现实**：detection 来自 ZED2 SDK 的 GT 标注（score=1.0），MOTA 无区分力。Phase 4 噪声注入实验部分缓解，但真实检测器评测待做。

7. **`weights_only=False`**：所有 `torch.load()` 调用需要此参数，因为 checkpoint 中包含 numpy 标量对象。

---

### 八、下一步候选方向

1. **更大去噪模型**：当前 CausalDilatedCNN（13.8k params, equiv_α=0.51）可能未充分拟合。尝试 deeper/wider 架构或 attention 机制。
2. **真实检测器评测**：用 YOLOv8 替换 GT detection，验证方法在真实条件下的增益（Phase 4 暗示噪声下 WiFi 更有价值）。
3. **σ 联合优化**：当前 σ=1.3 是手动选的。可以做 per-phone adaptive σ（基于 FTM std 或 denoiser confidence）。
4. **Oracle gap 分析**：Oracle α=1.0 可达 92.17%，当前 87.91% 仍有 ~4pt gap，分析 gap 来源。
5. **论文写作**：当前实验充分，可以开始组织 Introduction / Method / Experiment 章节。

---

### 九、快速验证命令

新开对话后，可用以下命令快速验证环境：

```bash
cd /Users/zstar/test/vifi/mot_baseline

# 1. 验证数据加载
python3 -c "from data.vifi_mot import list_sequences; print(len(list_sequences()), 'sequences')"

# 2. 验证去噪模型
python3 -c "from denoising.inference import FTMDenoiser; d = FTMDenoiser('causal_cnn','scene1'); print('OK')"

# 3. 验证 tracker
python3 -c "from trackers.wifi_joint import WiFiJointTracker; t = WiFiJointTracker(); print('OK')"

# 4. 跑一个 fold 的完整 pipeline（约 1 分钟）
python3 -u scripts/phase6b_integrated_mot.py --test-scene scene1
```

---

### 十、核心发现摘要（论文可用）

1. **FTM 噪声具有强时间相关性**（ACF lag-1 = 0.932, 半衰期 10 帧），使得因果去噪模型能有效消除 ~50% 噪声。

2. **去噪 + tighter σ 协同效应**：raw FTM 下 σ=1.0 导致 tracker 崩溃（IDF1 63.78%），去噪后 σ=1.0 仍维持 82.16%。去噪的核心价值是让 tracker 能安全使用更紧的匹配门限。

3. **257 params 胜过 90k params**：Phase 6B 的极小 MLP 在去噪信号上学习匹配，LOSO 4-fold 全部正向（+0.9 ~ +5.1pt），而 Method D 的 90k 参数 BiLSTM 泛化失败。模块化设计（去噪 → 匹配 → 分配解耦）是关键。

4. **WiFi 辅助 MOT 在检测噪声下价值更大**：bbox jitter 使 OC-SORT IDF1 下降 2.85pt，但 WiFi 方法仅下降 0.42pt。Phone-stable ID 天然抗检测抖动。

5. **全局联合分配（Hungarian）比贪心绑定优 +3.04pt**：消除了 track 处理顺序的排序偏差，在多人近距离交叉时优势显著。
