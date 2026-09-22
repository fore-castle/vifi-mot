# ViFi MOT Baseline

把 ViFi 数据集（RAN4model_dfv4p4）重新定义为 MOT 任务的最小基线代码。

```
mot_baseline/
├── data/
│   └── vifi_mot.py        # BBX5 + Others -> 每帧 detections + GT track ID
├── trackers/
│   ├── utils.py            # IoU / 匈牙利 / Kalman (7状态恒速)
│   ├── sort.py             # SORT
│   ├── ocsort.py           # OC-SORT (ORU + OCM + OCR)
│   ├── bytetrack.py        # ByteTrack (双阶段，需要置信度)
│   └── wifi_ocsort.py      # WiFi-as-ReID OC-SORT（方案 A）
├── eval/
│   └── mot_eval.py         # motmetrics 包装（MOTA / IDF1 / IDsw / MOTP / MT / ML）
├── scripts/
│   ├── run_baseline.py     # 主入口
│   └── ablate_wifi.py      # 超参扫描
├── exps/                   # JSON 实验结果
└── results/
    ├── baseline_report.md   # phase 1 baseline 实验报告
    └── wifi_ocsort_report.md  # phase 2 WiFi 增强报告
```

## 快速开始

```bash
# 单场景
python3 scripts/run_baseline.py --tracker ocsort --test-scene scene4

# 4-fold leave-one-scene-out
python3 scripts/run_baseline.py --tracker sort   --all-folds
python3 scripts/run_baseline.py --tracker ocsort --all-folds
```

## 主要结果（详见 results/）

### Phase 1: 纯运动 Baseline (`baseline_report.md`)

| Tracker | MOTA(avg) | IDF1(avg) | IDsw(total) |
|---|---|---|---|
| SORT | 97.66% | 56.28% | 1898 |
| OC-SORT | 97.69% | 65.63% | 1269 |
| ByteTrack | 不适用（无真实置信度） | | |

### Phase 2: WiFi-as-ReID 增强 (`wifi_ocsort_report.md`)

| Tracker | MOTA(avg) | **IDF1(avg)** | IDsw(total) |
|---|---|---|---|
| OC-SORT | 97.69% | 65.63% | 1269 |
| **WiFi-OC-SORT** | 97.38% | **78.51% (+12.88pt)** | 2613 |

### Phase 3: 学习版 Affinity 增强 (`wifi_affinity_report.md`)

复用 my_vifi 已训练的 Bi-LSTM + 1×1 Conv 亲和矩阵模型替换方案 A 的几何 compat：

| Tracker | MOTA(avg) | **IDF1(avg outdoor)** | indoor scene0 IDF1 |
|---|---|---|---|
| OC-SORT | 97.69% | 65.63% | 19.74% |
| WiFi-Geom (方案 A) | 97.38% | 78.51% | 39.27% |
| **WiFi-Affinity (方案 D)** | **97.45%** | **90.91% (+25.28pt)** | 37.81% |

注意：fold1 模型在 outdoor 4-fold 上有 train leakage（仅训练了 fold1 checkpoint），
indoor scene0 是严格未见。学习版在域内显著超越几何版（+12.4pt），跨域略弱（-1.5pt）。
两者互补。

## 快速开始（推荐配置）

```bash
# 4-fold WiFi-OC-SORT
python3 scripts/run_baseline.py --tracker wifi_ocsort --all-folds \
    --wifi-weight 0.10 --depth-sigma 1.5 \
    --bind-threshold 0.50 --bind-init-threshold 0.60 --unbind-threshold 0.10

# 超参扫描
python3 scripts/ablate_wifi.py --scene scene4
```
