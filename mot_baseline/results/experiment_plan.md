## 后续实验计划（2026-06-26 确认版）

> 决策记录：  
> 1. 先做纯物理 baseline，验证有效后再加深度学习  
> 2. 深度学习阶段两个架构都试（Cross-Attention A + Spatial GNN B），取好的  
> 3. 暂不做 WP-ReID 迁移，集中精力在 Vi-Fi  

---

### 总体时间线

```
Week 1      Week 2      Week 3      Week 4      Week 5-6     Week 7-8     Week 9-10
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Phase 0   | Phase 1a  | Phase 1b  | Phase 1c  | Phase 2a  | Phase 2b  | Phase 3  |
| 数据验证  | 3D投影    | EKF融合   | 遮挡bridg | Cross-Attn| GNN       | 消融+补充|
```

总计 ~10 周实验 + 3-4 周写作 = ~14 周（~3.5 个月）

---

### Phase 0：数据可行性验证（Week 1，2-3 天）

| 实验 | 内容 | 产出 | 判定标准 |
|---|---|---|---|
| 0.1 | depth vs FTM 误差分布 | mean/std/per-scene 统计 | \|mean\|<1.5m, std<2.5m |
| 0.2 | AP 位置最小二乘推断 | AP 3D 坐标 + 拟合残差 | 残差 < 2m |
| 0.3 | IDsw 分布分析 | per-sequence IDsw 排名 + 遮挡事件统计 | — |

**决策门 0**：0.1+0.2 通过 → Phase 1；不通过 → 调整策略（可能需要换用 FTM 差分而非绝对距离）

---

### Phase 1：纯物理 Spatial Radio Prior（Week 2-4，3 周）

#### Week 2 (1a)：3D Mahalanobis 空间兼容性

| 实验 | 内容 | 对比 |
|---|---|---|
| 1a.1 | 实现 2D→3D 投影模块 | 可视化 3D 投影结果 |
| 1a.2 | 实现 phone 3D 位置估计（FTM 球面 + 地面 + IMU heading） | 与 det_3d 对比 |
| 1a.3 | Mahalanobis 兼容性替换 cost matrix | vs 方案 A（标量高斯）|
| 1a.4 | Strict LOSO 评测 | IDF1 对比表 |

**新代码**：`trackers/spatial_projector.py`, 修改 `wifi_ocsort.py` → `wifi_spatial_ocsort.py`

#### Week 3 (1b)：Kalman EKF 无线观测融合

| 实验 | 内容 | 对比 |
|---|---|---|
| 1b.1 | KalmanBox 增加 wireless_update() | — |
| 1b.2 | 实现 EKF Jacobian + adaptive R_wire | vs 无 wireless update |
| 1b.3 | Strict LOSO 评测 | 1a.4 + 1b.2 |

**修改**：`trackers/utils.py` (KalmanBox), `trackers/wifi_spatial_ocsort.py`

#### Week 4 (1c)：遮挡 bridging + 人工遮挡实验

| 实验 | 内容 | 对比 |
|---|---|---|
| 1c.1 | 遮挡期间 wireless-only Kalman update | vs 无 bridging |
| 1c.2 | 人工遮挡实验（mask 20%/40% detection） | IDsw 退化曲线 |
| 1c.3 | Strict LOSO 完整评测 | Phase 1 最终数字 |

**决策门 1**：Phase 1 综合 IDF1 > 78.51%（方案 A）→ 物理模型独立有价值；< 78.51% → 物理模型作为特征输入 Phase 2

---

### Phase 2：Physics-Guided 深度学习（Week 5-8，4 周）

#### Week 5-6 (2a)：Cross-Attention + Physics Bias

| 实验 | 内容 | 对比 |
|---|---|---|
| 2a.1 | Physics feature 提取（6-8 维/对） | — |
| 2a.2 | 双分支 TCN 编码器实现 | — |
| 2a.3 | Cross-attention + MLP_bias(物理特征) | — |
| 2a.4 | LOSO 训练 + 评测 | vs Phase 1 |
| 2a.5 | Soft measurement update 集成 | vs hard bind |

**新代码**：`trackers/physics_cross_attn.py`, `models/physics_matcher.py`

#### Week 7-8 (2b)：Spatial GNN

| 实验 | 内容 | 对比 |
|---|---|---|
| 2b.1 | 图构造（track 节点 + phone 节点 + 物理边） | — |
| 2b.2 | 2 层 message passing + 边分类 | — |
| 2b.3 | LOSO 训练 + 评测 | vs 2a (Cross-Attn) |
| 2b.4 | Soft measurement update 集成 | vs 2a.5 |

**新代码**：`trackers/spatial_gnn.py`, `models/spatial_gnn_model.py`

**决策门 2**：取 2a 和 2b 中 IDF1 更高的作为最终方法

---

### Phase 3：消融 + 补充实验（Week 9-10，2 周）

| 实验 | 内容 |
|---|---|
| 3.1 | 完整消融表（A0-A7，8 行）|
| 3.2 | 人工遮挡退化曲线（20%/40%/60% mask）|
| 3.3 | scene0 (indoor) strict LOSO 跨域评测 |
| 3.4 | Per-sequence error analysis（最差/最好序列 case study）|
| 3.5 | Attention 可视化 / GNN 边权重可视化 |

---

### 关键里程碑

| 时间点 | 里程碑 | 成功标准 |
|---|---|---|
| Week 1 末 | Phase 0 通过 | depth-FTM 误差可控，AP 位置可推断 |
| Week 4 末 | Phase 1 完成 | 纯物理 IDF1 > 78.51%（超过方案 A）|
| Week 8 末 | Phase 2 完成 | 混合模型 IDF1 > Phase 1 结果 |
| Week 10 末 | Phase 3 完成 | 完整消融 + case study |
| Week 14 末 | 论文完成 | 投稿 ACM MM 2027 |

---

### 风险与 Mitigation

| 风险 | 概率 | 影响 | Mitigation |
|---|---|---|---|
| depth 质量不够 | 中 | Phase 0 阻塞 | 用 FTM 差分（ΔFTM/Δt）替代绝对距离 |
| AP 位置推断不准 | 中 | Phase 1 精度受限 | 把 AP 位置 uncertainty 放大到 R_wire 中 |
| 纯物理 IDF1 < 方案 A | 中 | Phase 1 价值降低 | 直接进入 Phase 2，物理特征作为学习模型输入 |
| Cross-Attention overfit | 高 | Phase 2a 失败 | Data augmentation + dropout + early stopping |
| GNN 实现太复杂 | 中 | Phase 2b 延期 | 如果 2a 效果好，2b 可以简化或跳过 |
| scene4 仍然拉胯 | 高 | 整体 IDF1 受限 | Per-scene analysis，scene4 作为 hard case 单独讨论 |
