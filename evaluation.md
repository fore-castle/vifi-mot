# 论文质量评估记录（第 2 轮）

> 评估日期：2026-07-29 | 数据源：Semantic Scholar / OpenAlex / GitHub API / IEEE·ACM 官网核实
> 标准：合格 = CCF A/B 正式发表（或 ICLR 等公认顶会）；边缘 = CCF C 或非 CCF 但高引高相关；不合格 = workshop 低引 / 纯预印本 / 零引用

## 合格（12 篇，计入 20 篇目标）

| 编号 | 论文 | Venue | CCF | 引用 | 认可度 |
|---|---|---|---|---|---|
| A3 | Mission (mmWave+RGB ReID) | SenSys 2024 | B | 13-15 | 被后续 mmWave ReID 引用 |
| A5 | OOSTraj | CVPR 2024 | A | 17 | 代码 16★，被引作 baseline |
| A6 | LTrajDiff | ACM MM 2023 | A | 4 | 弱（无代码低引），但正式 A 会 |
| A7 | Out-of-Sight Embodied Agents | TPAMI 2026（DOI 10.1109/TPAMI.2026.3676710，已核实） | A | 0(新) | OOSTraj 期刊扩展 |
| B2 | ImmTrack | IPSN 2023 | B | 8-10 | 被 survey 引用 |
| B4 | X-Fi | ICLR 2025 | 非CCF目录/公认顶会 | 20 | 代码 21★ |
| B5 | PmTrack | IMWUT 2023 | A | 8-17 | IMWUT 正刊 |
| C1 | NLOS Survey | ACM CSUR 2024 | A | 30 | 权威综述 |
| D2 | SUSHI | CVPR 2023 | A | 85 | 代码 141★，常用 baseline |
| D3 | Hybrid-SORT | AAAI 2024 | A | 140 | 代码 266★，高采用 |
| D4 | UCMCTrack | AAAI 2024 | A | 95-138 | 代码 372★（本批最高） |
| D5 | SparseTrack | TCSVT 2025（DOI 10.1109/TCSVT.2024.3524670，已核实） | B | 57-148 | 代码 166★ |

## 边缘（5 篇，不计入 20 篇，作参考保留）

| 编号 | 论文 | Venue | CCF | 引用 | 保留理由 |
|---|---|---|---|---|---|
| A2 | ViFi-Loc | ICMI 2023 | C | 6 | Vi-Fi 一作后续，related work 可引 |
| B1 | Opt-in Camera | IROS 2025 | C | 2-3 | 与用户方法同构性最高的 UWB 版本 |
| C2 | WhereArtThou | IEEE Access 2024 | 无 | 33 | FTM+人体遮挡实证，场景同构，引用好 |
| C3 | UWB CIR 误差缓解 | Sensors 2024 | 无 | 15 | 误差分型校正思路可借鉴 |
| C5 | UWB BERT NLOS | IPIN 2024 | 无 | 5 | 领域专业会，ablation 对照可用 |

## 不合格（5 篇，剔除）

| 编号 | 论文 | 剔除理由 |
|---|---|---|
| A1 | ViFiT | MobiCom workshop（非主会），仅 2 引，代码 1★ |
| A4 | ViFiCon | CVPR workshop（arXiv 挂 4 年未进主会），2-3 引，无代码。注意：仍是相关性极高的对标方法，写论文时可引用但不算高质量证据 |
| B3 | ViFi-ReID | 纯 arXiv 预印本，0 引用，无同行评审 |
| C4 | UWB Bayesian | Scientific Reports 非 CCF，0 引用 |
| D1 | CAMELTrack | 仍为预印本（GitHub 无录用标记），但代码 129★ 有潜力，**建议持续追踪**，若录用可升级 |

## 第 3 轮新增（13 篇，均为 CCF A/B 或领域公认顶刊，已核实 venue+引用）

| 编号 | 论文 | Venue | CCF | 引用 | PDF |
|---|---|---|---|---|---|
| A8 | RoVaR (视觉+UWB 双层融合跟踪) | IMWUT 2023 | A | 3 | ✅ |
| A9 | Invisibility Cloak (手表IMU-视频绑定+隐私) | UIST 2025 | A | 0(新) | ✅ |
| A10 | STNet (音视频说话人跟踪) | TMM 2024 | B | 11 | ✅ |
| A11 | MM-Fi (五模态人体感知数据集) | NeurIPS 2023 D&B | A | 170 | ✅ |
| A12 | Babel (可扩展多模态感知预训练) | SenSys 2025 | B | 3(新) | ✅ |
| C6 | RLoc (WiFi 定位不确定性量化) | IMWUT 2023 | A | 31 | ✅ |
| C7 | Bayesian KalmanNet (学习KF不确定性) | IEEE TSP 2025 | 非CCF目录/领域顶刊 | 15 | ✅ |
| C8 | Self-Training NLOS Mitigation | TMC 2023 | A | 46 | ✅ |
| C9 | RefLoc (锚链路 NLOS 判别) | ACM TOSN 2024 | B | 18 | ❌付费墙 DOI 10.1145/3657639 |
| D6 | MotionTrack (Refind 长时找回) | CVPR 2023 | A | 159 | ✅ |
| D7 | DiffMOT (扩散运动预测) | CVPR 2024 | A | 96 | ✅ |
| D8 | SMILEtrack (相似度学习匹配) | AAAI 2024 | A | 106 | ✅ |
| D9 | U2MOT (不确定性感知关联) | ICCV 2023 | A | 24 | ✅ |

## 最终状态（第 3 轮结束）

- **合格论文：12 + 13 = 25 篇 ≥ 20，目标达成** ✅
- 边缘参考：5 篇（A2, B1, C2, C3, C5）
- 剔除：5 篇（A1, A4, B3, C4, D1；PDF 保留在 papers/ 供参考）
- 分方向合格数：A 无线+视觉融合 9 篇 | B 人-设备绑定 4 篇 | C 测距去噪/不确定性 5 篇 | D 视觉 MOT 7 篇
- 未获取 PDF 的合格论文：B5 PmTrack (IMWUT)、C1 NLOS Survey (CSUR)、C9 RefLoc (TOSN)，均为 ACM 付费墙，DOI 已记录

## 第 4 轮：E 批次（2026-07-30，2025-2026 CVPR/ICCV/NeurIPS 主会专项）

评估口径调整：因发表过新，不设引用量门槛，只验证主会收录（openaccess.thecvf.com / papers.nips.cc / 官方接收列表核实，排除 workshop）。全部 CCF A。

| 编号 | 论文 | Venue | 主题 |
|---|---|---|---|
| E1 | OA-SORT | CVPR 2026 | 遮挡感知代价修正（免训练） |
| E2 | FDTA | CVPR 2026 | 关联嵌入学习（含深度线索） |
| E3 | HyperSSM | CVPR 2026 | 超图+SSM 遮挡运动推理 |
| E4 | GRAE-3DMOT | CVPR 2025 | 几何距离学习进关联代价 |
| E5 | ADMCMT | CVPR 2025 | RGBT 多相机全天候跟踪 |
| E6 | MVTrajecter | ICCV 2025 | 多时间戳运动+外观代价 |
| E7 | RAPTR | NeurIPS 2025 | 弱监督雷达 3D 姿态 |
| E8 | MVDoppler-Pose | CVPR 2025 | mmWave vs 相机遮挡鲁棒性 |
| E9 | Single-Chip Radar FM | ICCV 2025 | 雷达基础模型 |
| E10 | SATM Adapter | ICCV 2025 | 视频-雷达空间对齐+时序匹配 |
| E11 | RCTDistill | ICCV 2025 | 各向异性误差建模跨模态蒸馏 |
| E12 | MotionBind | NeurIPS 2025 | 运动模态统一嵌入对齐 |

## 最终状态（第 4 轮结束）

- **合格论文：25 + 12 = 37 篇**（其中 2025-2026 年 18 篇，占 49%）
- 边缘参考 5 篇，剔除 5 篇（PDF 保留）
- PDF 共 43 个，付费墙未获取 3 篇（DOI 已记录）

## 标注更正记录（2026-07-31 复核）

### 已更正
- **C7 Bayesian KalmanNet（IEEE TSP 2025）：原标 `CCF A` → 更正为 `非CCF目录/领域顶刊`**
  - 核实依据：CCF 推荐目录（2022/2026 版）**不覆盖信号处理领域**，IEEE Transactions on Signal Processing 未出现在任何领域分类中（已逐一检索"人机交互与普适计算"等相关领域列表确认）
  - 该刊仍为 IEEE 信号处理学会旗舰期刊、JCR Q1 / 中科院一区，学术声誉不受影响，故仍计入合格（与 B4 X-Fi/ICLR 相同处理口径）
  - 合格总数不变（37 篇），但引用它时不应声称 CCF A

### 已复核无误
- IMWUT/UbiComp = **CCF A**（"人机交互与普适计算"A 类，目录条目名 `UbiComp/IMWUT`）→ B5 PmTrack、A8 RoVaR、C6 RLoc 标注正确
- UIST = CCF A（A9 正确）；ICMI = CCF C（A2 正确）
- 其余条目（CVPR/ICCV/AAAI/NeurIPS/ACM MM = A；TPAMI/TMC/CSUR = A；SenSys/IPSN/TCSVT/TMM/TOSN = B；IEEE Access/Sensors/Scientific Reports/IPIN = 非 CCF）均与 CCF 目录一致

**核实来源**：[CCF 人机交互与普适计算目录](https://www.ccf.org.cn/Academic_Evaluation/HCIAndPC/)、[CCF 目录在线检索版](https://ccf.atom.im/)
