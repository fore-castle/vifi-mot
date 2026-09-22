# S11: 量化基线中的 LOSO 数据泄漏

> 日期：2026-07-31 | 脚本：`s11_clean_matching.py`（不修改原始基线脚本）

## 1. 发现的两处泄漏（均已读码验证）

### L1 — 匹配 MLP 训练集未排除测试场景 🔴

`mot_baseline/scripts/phase6b_learned_matching.py:59`
```python
def build_matching_dataset(test_scene: str, model_name="causal_cnn"):
    denoiser = FTMDenoiserBatch(model_name, test_scene, window_size=15)
    #                                       ↑ test_scene 仅用于选去噪器 ckpt
    for scene in SCENES:      # ← 遍历全部 4 场景，缺少 `if scene == test_scene: continue`
```
- `:196-201` 的 80/20 "验证集"从同一含测试场景的池中随机切分 → **选模型也被污染**
- `:340-342` 注释写 `# Evaluate on held-out test set`，实际是在同一全场景数据集（含训练样本）上评估 → `exps/phase6b_matching.json` 的 AUC/gap 是训练集指标
- **同一缺陷被复制到本轮新增脚本**：`t9_hetero_matching.py:83`、`t12_hmu_matching.py:29`、`t15_dual_matching.py:45`
- 影响面：`checkpoints_matching/` 全部 16 个 ckpt → `exps/v2_experiments.json` 所有 `*_learned` 配置 → 包括被称为"当前最佳"的 **87.91%**

### L2 — 去噪 checkpoint 按测试折 MAE 选出 🔴（本次未修，见 §4）

`mot_baseline/denoising/train.py:220, 269-295`
```python
test_ds = FTMDenoiseDataset(test_series, ...)   # held-out scene 构成
...
if test_mae < best_test_mae:
    best_test_mae = test_mae
    torch.save({...})            # ← 按测试集 MAE 保存 best ckpt，全程无 val split
```
12 个去噪 ckpt（`causal_cnn/causal_lstm/bilstm × scene1-4`）全部是测试集选出的。

**对照**：`denoising/hetero.py:74-79,105,112-121`（本轮新写）协议正确——10% val split、按 val NLL 选模、温度在 val 上标定。可作为修复 `train.py` 的模板。

## 2. 修复与量化（只修 L1，隔离其单独影响）

`s11_clean_matching.py` 与原脚本唯一差别是加了那一行 `if scene == test_scene: continue`；特征、架构、epoch、seed、类平衡、去噪器 ckpt 全部保持一致。

| 配置 | scene1 | scene2 | scene3 | scene4 | **avg IDF1** |
|---|---|---|---|---|---|
| `base_learned`（泄漏，原"最佳"） | 91.20 | 86.61 | 88.41 | 85.42 | **87.91** |
| `base_clean`（L1 已修） | 91.19 | 86.47 | 85.86 | 85.87 | **87.35** |
| **逐折差值** | −0.01 | −0.14 | **−2.55** | **+0.45** | **−0.56** |

**结论**：
- L1 泄漏使基线被高估 **+0.56pt**，几乎全部来自 **scene3（−2.55）**；scene4 反而略降（+0.45，说明泄漏并非在每折都有利）
- 泄漏的影响量级（0.56pt）**与本轮追逐的"改进"（+0.31pt）同量级** → 进一步印证之前所有 sub-1pt 结论都在伪影/噪声范围内
- 干净 MLP 的 val AUC：0.849/0.865/0.876/0.847（泄漏版为 ~0.86 但那是污染的 val，不可比）

### T3b 在干净基线上的表现

| 配置 | avg IDF1 | Δ vs 对应基线 |
|---|---|---|
| `base_learned` → `t3b_learned` | 87.91 → 88.06 | +0.15（泄漏基线上） |
| `base_clean` → `clean_t3b` | 87.35 → 87.36 | **+0.01（干净基线上）** |

T3b 的微小增益在干净基线上基本消失 → 与 S10 的显著性结论一致（该项从未通过检验）。

## 3. 修正后的诚实数字

| 阶段 | 原报告 | 修正后 | 备注 |
|---|---|---|---|
| OC-SORT（纯视觉） | 65.63 | 65.62 | ✅ JSON 可查，无训练，可信 |
| Phase 5（全局分配，无训练） | 80.36 | 待核（同配置存在 3 套互斥数字） | 无训练参数，不涉泄漏，但文档需统一 |
| Phase 6A（去噪 + 高斯匹配） | 83.17 | 85.61（`base_gaussian`，同协议） | 83.17 的产出脚本不可复现；L2 泄漏仍在 |
| Phase 6B（+ 学习匹配） | **87.91** | **87.35** | L1 已修；L2 仍在 |
| 本轮三思路组合 | 88.22 | 需在干净基线上重跑 | 且其自身 MLP 也有 L1 泄漏 |

## 4. 剩余待修

1. **L2**：给 `denoising/train.py` 加训练池内部 val split（照抄 `hetero.py:74-79`），按 val MAE 选模，重训 12 个去噪 ckpt。预计基线还会有变动
2. **T15/T9/T12 的 MLP** 需用干净协议重训后重新评估三思路组合
3. **超参重选**：`wifi_weight=0.10`、`depth_sigma=1.3/1.5` 当前是在测试折（scene4/全 4 折平均）上选的，应改为在训练场景上选，或每折做内层 leave-one-more-scene-out
4. **去噪器 buffer 跨序列泄漏**（S10 §2 发现）：`FTMDenoiser` 每折只建一次、从不 reset，需统一为每序列 reset

## 5. 方法学教训

这次泄漏之所以长期未被发现，是因为：
- 函数签名有 `test_scene` 参数，看起来像做了 LOSO（实际只用于选去噪器 ckpt）
- 注释明确写 `# Evaluate on held-out test set`，与实际行为相反
- 4 折结果看起来"合理"（87.91 相对 85.61 的 +2.30 并不夸张），没有触发怀疑

**建议**：为每个训练脚本加一个断言，例如
```python
assert test_scene not in {s for s in scenes_used_for_training}, "LOSO violation"
```
并在每个 ckpt 里存 `protocol` 字段（`s11_clean_matching.py` 已这样做：`"protocol": "clean_loso"`）。
