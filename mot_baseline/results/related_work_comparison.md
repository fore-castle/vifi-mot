# Related Work Comparison: Vision + Wireless MOT

> 整理日期：2026-06-24  
> 阅读范围：5 篇核心工作（2020-2023）  
> 用途：作为新 idea 的差异化定位参考

---

## Page 1 — 方法维度对照表

### 1.1 基础信息

| 论文 | 年份 / 会议 | 作者 / 单位 | 核心信号 | 问题类型 |
|---|---|---|---|---|
| **Vi-Fi** | IPSN 2022 | Abrar Alali, Hansi Liu, Marco Gruteser, Shubham Jain 等 (Rutgers WINLAB + 多机构) | WiFi FTM + 9 轴 IMU + RGB-D depth | 视觉-无线 **身份关联**（单相机 + 多手机） |
| **ViTag** | IEEE SECON 2022 | Bryan Bo Cao 等 (Rutgers WINLAB，Vi-Fi 同团队) | WiFi FTM + 19 维 IMU + BBX5 | 视觉-无线 **身份关联**（online，跨模态翻译） |
| **VMWP (RCPM)** | ACM MM 2020 | Yiheng Liu, Wengang Zhou 等 (USTC) | **GPS** (~10 m 误差，模拟 WiFi/5G) + 视觉轨迹 | 跨相机 **ReID** + video-to-signal matching |
| **MCPF** | IEEE TMM 2022 | 同上（VMWP 的 journal 扩展版） | 同上 | 同上 + **UMM-ReID 无监督跨域** |
| **UMTF** | IEEE TPAMI 2023 | Yiheng Liu, Wengang Zhou, Qiaokang Xie, Houqiang Li (USTC) | 手机定位轨迹（cellular/WiFi 均可） | **弱标注下的无监督 ReID**（仅需相机位置，不需场景 GPS 标注） |

### 1.2 信号与特征对照

| 论文 | 无线信号细节 | 视觉特征 | 时间窗口 |
|---|---|---|---|
| Vi-Fi | FTM @ 3 Hz（range + std，2D）；9 轴 IMU @ 50 Hz（resample 到 FTM 时间戳） | BBX (x, y, depth) 3D，**无 RGB 外观** | k=10 帧 (~3s) |
| ViTag | FTM @ 3-5 Hz（range + std）；19 维 IMU（含四元数 + 重力 + 线性加速度）@ 100 Hz | BBX5 (x, y, depth, w, h)，**无 RGB 外观** | 10 samples (~1-3s) |
| VMWP / MCPF | GPS (lat, lon) @ 1 Hz，~10 m 误差；**无 IMU** | ResNet-50 / I3D / M3D / STMP 2048-D 外观特征 | 序列级（整条 video） |
| UMTF | 同上（cellular/WiFi 定位，弱监督） | ResNet-50 2048-D 外观特征，2 帧平均 | 序列级 |

### 1.3 融合方法

| 论文 | 融合架构 | 训练方式 | 是否需要身份标注 |
|---|---|---|---|
| **Vi-Fi** | 双分支 Bi-LSTM (hidden=32) + 1×1 Conv 压缩成 (Nm+1, Nc+1) affinity matrix；baseline 为 DTW + Hungarian | **有监督**（cross-entropy on affinity） | **需要**（per-frame ID） |
| **ViTag** | X-Translator：跨模态 seq2seq 翻译（vision → reconstructed phone），6 项加权损失；用 Euclidean/Bhattacharyya 距离做 Hungarian 匹配 | **有监督**（paired vision-phone） | 需要（训练时） |
| **VMWP / RCPM** | RCPM：交替更新 visual affinity S 和 trajectory distance D，K=4 轮迭代；**test-time only**，**无训练** | **无训练**（post-hoc 图传播） | **不需要**（但需场景 GPS 标注） |
| **MCPF** | RCPM + UMM-ReID（伪标签 + triplet loss 训练视觉分类器） | **无监督跨域** | 不需要（伪标签自举） |
| **UMTF** | MMDA（时序-空间聚类 → 路径一致性 score）+ MMGN（图神经网络传播 multimodal feature） | **无监督弱标注**（仅相机位置） | **不需要** |

### 1.4 身份关联形式化

| 论文 | 形式化 | 关联算法 | 在线/离线 |
|---|---|---|---|
| Vi-Fi | 软亲和矩阵（row-softmax + col-softmax） | Hungarian (offline) / majority voting (online) | 两者 |
| ViTag | 跨模态重构后 Euclidean/BD 距离 | Hungarian | Online (90% overlap sliding) |
| VMWP / MCPF | ReID ranking（CMC / mAP） | 相似度排序 | Offline |
| UMTF | 多模态特征 ranking | 相似度排序 | Offline |

### 1.5 数据集

| 论文 | 数据集 | 规模 | 场景 | 公开 |
|---|---|---|---|---|
| **Vi-Fi** | Vi-Fi Dataset（自采） | 90 序列（15 indoor + 75 outdoor）；5 subjects indoor + 2-3 outdoor；最多 12 同帧行人 | Indoor + outdoor | **是**（含代码） |
| **ViTag** | 自采（IRB） | 15 indoor (5 subjects) + 15 outdoor (2 subjects) + 1 crowded (11 同帧) | Indoor + outdoor | **是** |
| **VMWP / MCPF** | WP-ReID（自采） | 79 IDs，868 video 序列，106,578 帧；6 相机，**仅 29 有无线轨迹** | 户外操场 | **是**（仅数据） |
| **UMTF** | WP-ReID + Campus4K（合成无线） | WP-ReID 同上 + Campus4K (3849 seqs, 1567 IDs, 6 cams, 模拟无线) | Outdoor | 数据 + 代码 |

### 1.6 关键指标

| 论文 | 主指标 | 最好结果 | 备注 |
|---|---|---|---|
| **Vi-Fi** | IDP (Identification Precision) | Outdoor online 81.1% / offline 90.2%；Indoor ~96% | 仅 2-3 手机同时存在 |
| **ViTag** | IDP | **平均 88.39%**（Indoor 90.21% / Outdoor 87.85% / Crowded 87.11%） | FTM 贡献 +12.56% |
| **VMWP** | ReID mAP / R1 + Signal Matching R1 | MMT+RCPM: ReID mAP=61.6%, R1=77.1%; **SM R1=89.6%** (+35.9 vs baseline) | RCPM 贡献巨大 |
| **MCPF** | 同上 | 与 VMWP 相同（journal 扩展） | 加 UMM-ReID 无监督 |
| **UMTF** | mAP / R1 | WP-ReID: **mAP=66.3%, R1=91.0%**（vs vision-only 54.7 / 85.6）；Campus4K: mAP=86.6% | 弱标注下超越 RCPM |

### 1.7 跨场景 / 泛化能力（你最该关注的维度）

| 论文 | 跨场景评测 | 跨场景表现 | 泛化性评估 |
|---|---|---|---|
| Vi-Fi | 5-fold outdoor CV；无 cross-scene test | 未测 | **未验证**（同一场景切分训练/测试） |
| ViTag | LOOCV within dataset | 未测 | **未验证** |
| VMWP / MCPF | MARS → WP-ReID / Duke → WP-ReID 跨域 | UMM-ReID 跨域 +16~27 pt | **部分验证**（跨数据集，但 WP-ReID 本身无训练集） |
| UMTF | Campus4K (模拟) → WP-ReID | 跨域 mAP 提升 +5~12 pt | **部分验证**（但 Campus4K 无线是合成的） |

**关键观察**：**所有 5 篇都未做真正的 leave-one-scene-out 泛化评测**。Vi-Fi/ViTag 用 K-fold 在同一场景内切分；VMWP/UMTF 用跨数据集做"跨域"，但目标数据集 WP-ReID 没有训练集、不能严格比较。你已经在 ViFi 数据上跑出 strict LOSO 数字（leaky 90.91% → strict 75.62%），**这是该领域第一次公开的 strict cross-scene 结果**。

---

## Page 2 — 差异化分析 & 你的研究机会

### 2.1 已有工作的局限（按论文）

| 论文 | 作者自承的局限 | 你补充的观察 |
|---|---|---|
| **Vi-Fi** | (1) affinity matrix 维度固定 (Nm=5, Nc=15)，不能扩展；(2) 隐私问题未解决 | **最大盲点**：仅测 IDP，不报 IDF1，不能与 MOT 社区比较；仅 2-3 手机同时存在，没压力测试；tracktor++ 检测错误未分析；需手动时钟同步 |
| **ViTag** | (1) 数据规模小（5 indoor / 2 outdoor subjects）；(2) 单相机 + 单 AP；(3) FTM 受环境多径影响；(4) 泛化到新视角列为 future work | X-Translator 的 6 项损失超参敏感；ED vs BD 选择依场景而定无自适应 |
| **VMWP** | (1) WP-ReID 仅 79 IDs / 29 无线，scale 太小；(2) GPS 代替 WiFi，未真实评估；(3) 需 per-camera 地理标定；(4) 超参 (K, σ, iterations) 敏感 | 无端到端训练，性能受上游 ReID 限制；仅户外 |
| **MCPF** | 同上 + UMM-ReID 伪标签噪声 | RCPM 是 post-hoc，与上游 ReID 解耦；无线覆盖率稀疏（29/79） |
| **UMTF** | (1) Campus4K 无线是**合成的**；(2) 需行人带手机且开启定位；(3) 感知半径超参（默认 50m）需调；(4) WP-ReID 无 train/test split | 合成无线不能反映真实噪声；聚类中心数启发式估计 |

### 2.2 现有工作未覆盖的 5 个关键维度

1. **真正的跨场景泛化（strict LOSO）**：所有 5 篇都没测过"训在 scene A、测在 scene B"，而这正是部署的核心诉求。你的实验（leaky 90.91 → strict 75.62）**首次暴露了这个问题**。
2. **外观一致人群（uniform appearance）**：没有任何一篇论文构造"穿制服/校服/工服"的测试子集。这是 visual ReID 必然失效、wireless 必然有效的 corner case，是天然的 motivation 场景。
3. **在线 MOT 而非离线 ReID**：Vi-Fi/ViTag 做 association 但非标准 MOT；VMWP/UMTF 做离线 ReID。**没有一篇用标准 MOT 评测（MOTA/IDF1/IDsw）做 online tracking**。你已经用 OC-SORT 跑通了 online MOT 框架。
4. **跨场景泛化的方法论**：现有工作都是"在某场景训练、同场景（或换数据集）测试"，没人讨论 meta-learning / domain adaptation / ensemble with geometric prior。
5. **隐私保护合规**：Vi-Fi 在 discussion 提了一句"privacy concerns"，但没有把"无外观 ReID"作为方法论贡献来论证。GDPR / 中国《个人信息保护法》场景下这是真痛点。

### 2.3 你的 idea 定位矩阵

把你的想法放在"现有工作 × 你的差异化"的二维空间里：

| 差异化角度 | 已有工作覆盖度 | 你的已有进展 | 论文可行性 |
|---|---|---|---|
| **A. 跨场景泛化 (strict LOSO)** | **空白**（0/5 篇测过） | 已跑出 leaky vs strict 数字，证明问题存在 | ★★★★★ 强烈推荐 |
| **B. 外观一致人群 ID 恢复** | **空白**（0/5 篇构造过） | 需自采 1 个子集（~2-4 个 uniform 序列） | ★★★★ 推荐 |
| **C. Online MOT 评测（MOTA/IDF1）** | **部分**（Vi-Fi/ViTag 是 association，不是标准 MOT） | 已实现 OC-SORT + 方案 A/D，指标齐全 | ★★★★ 推荐（作为副贡献） |
| **D. 隐私保护范式论证** | **弱**（Vi-Fi 仅 discussion） | 需补法律 + 技术对照 | ★★★ 作为 motivation 章节 |
| **E. A+D ensemble / meta-learning** | **空白** | 方案 A、D 已独立实现 | ★★★★ 作为方法贡献 |

### 2.4 推荐的研究定位

**主标题候选**：  
"**Cross-Scene Multi-Object Tracking with Wireless-Assisted Identity Recovery: 
A Strict Leave-One-Scene-Out Benchmark and Geometric-Learned Ensemble**"

**One-sentence story**：  
"我们首次暴露了视觉+无线 MOT 在跨场景上的显著退化（strict LOSO IDF1 从 90.91% 跌至 75.62%），
提出 (1) 一个 strict cross-scene benchmark、(2) 一个几何-学习 ensemble 方法把 strict IDF1 推回 80%+，
(3) 并构造了一个外观一致人群的 corner-case 评测展示 wireless 的不可替代性。"

**三个并列贡献**：

1. **Benchmark 贡献**：第一个 strict LOSO cross-scene 评测协议（在 ViFi 数据集上），
   附带一个外观一致人群的 corner-case 子集（自采 ~4-6 序列）
2. **方法贡献**：A+D ensemble（geometric prior + learned affinity，two-stage gating），
   解决"学习版跨场景退化"这个你自己首次暴露的问题
3. **洞察贡献**：外观一致人群场景下 wireless 的不可替代性实证（motivation figure）

### 2.5 差异化防御（应对 reviewer 拒稿意见）

**可能的拒稿意见**：

- "Vi-Fi 2022 已经做过 vision + wireless association 了，新颖性不足"  
  → **防御**：我们首次暴露并量化了 strict LOSO 跨场景退化（−15.3 pt），
  并提出 ensemble 解法；Vi-Fi 仅测了同场景 K-fold。
- "WP-ReID 系列已经做了 wireless-assisted ReID"  
  → **防御**：他们做的是**离线、跨相机 ReID**（用 GPS 做 graph propagation）；
  我们做的是**在线、单相机 MOT**（用 FTM/IMU 做 phone-track 绑定）。
  问题形式、信号类型、评测协议都完全不同。
- "你的 ViFi 数据集规模太小（82 序列）"  
  → **防御**：承认，但指出 Vi-Fi / ViTag 数据集规模相当；
  重点是**评测协议的严格性**（strict LOSO）和**corner-case 子集**（外观一致）的洞察价值。
  作为 future work 提到扩展到更大规模数据集。

### 2.6 Timeline 建议

| 阶段 | 工作 | 时长 |
|---|---|---|
| **Phase 1** | Related work 精读 + 现有方案 A/D 复现已跑通 | **已完成** |
| **Phase 2** | A+D ensemble 实现 + strict LOSO 评测 | 4-6 周 |
| **Phase 3** | 自采外观一致人群子集（4-6 序列，校服/制服场景） | 2-4 周 |
| **Phase 4** | 实验消融 + 写论文 | 6-8 周 |
| **Phase 5** | 投稿前打磨 | 2 周 |
| **总计** | | **~4-5 个月** |

**目标会议**（按时间窗口）：

- ICCV 2027 (submission ~Mar 2027) — 充足时间做完整实验
- CVPR 2027 (submission ~Nov 2026) — 紧凑但可行
- AAAI / IJCAI 2027 — 备选
- 如仅做 Phase 2（仅 ensemble）可投 ECCV / BMVC workshop

---

## 附录：5 篇论文 quick-reference

| # | Short name | Key takeaway |
|---|---|---|
| 1 | **Vi-Fi (IPSN 2022)** | First vision + WiFi (FTM+IMU) association via learned affinity; no appearance; small scale |
| 2 | **ViTag (SECON 2022)** | Cross-modal seq2seq translation vision→phone; online; same team as Vi-Fi |
| 3 | **VMWP (ACM MM 2020)** | First vision + wireless (GPS proxy) ReID via RCPM graph propagation; test-time only |
| 4 | **MCPF (TMM 2022)** | Journal version of VMWP + UMM-ReID unsupervised domain adaptation |
| 5 | **UMTF (TPAMI 2023)** | Weak-label unsupervised ReID (only camera locations needed); surpasses supervised RCPM |

---

**Bottom line for your idea**:

你的方向（vision + wireless MOT）**成立且有价值**，但必须在以下至少一点上与这 5 篇做出明确差异：

1. **strict LOSO cross-scene 评测**（强烈推荐，已有数据支持）
2. **外观一致人群 corner-case**（推荐，需自采）
3. **geometric-learned ensemble 方法**（推荐，已有 A/D 雏形）

三选二即可成文。最稳的组合是 **1 + 3**（已有数据 + 已有代码），可在 4-5 个月内投稿。
