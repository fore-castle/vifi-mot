# Vi-Fi MOT 相关论文调研报告（2023-2026）

> 生成日期：2026-07-29
> 研究背景：WiFi FTM 测距 + 单目深度匹配，将视觉 MOT tracklet 绑定到手机持有者，实现 phone-stable ID（OC-SORT + 全局 Hungarian 分配 + CausalCNN 去噪 + 极小 MLP 学习匹配，IDF1 87.91%），目标 ACM MM 2027。
> 共收录 **22 篇**论文（20 篇 PDF 已下载至 `papers/`，2 篇因付费墙仅存元数据）。
> 相关性评级：★★★ 必引/直接对标 | ★★ 强相关/方法借鉴 | ★ 背景支撑

---

## 方向 A：RF/无线+视觉融合跟踪与定位（7 篇）

### A1. ViFiT: Reconstructing Vision Trajectories from IMU and Wi-Fi Fine Time Measurements ★★★
- **出处**：ACM MobiCom 2023 Workshop (ISACom) | Vi-Fi 原班团队（Rutgers/Stony Brook）
- **PDF**：`papers/A_1_ViFiT.pdf` | [arXiv](https://arxiv.org/abs/2310.03140)
- **摘要**：基于 Transformer，直接从手机端数据（IMU + WiFi FTM）重建行人在视频中的 bbox 轨迹，在摄像头帧丢失或不传输时维持视觉跟踪。在 Vi-Fi 数据集 5 个真实场景上评估，重建质量优于 LSTM 基线（ViTag），并提出 MRFR 等新指标。动机是降低视频回传延迟与带宽开销，用手机模态"补帧"。
- **相关性**：同源同数据集，是 phone→vision 的逆问题（你做 vision→phone 绑定）。其 Transformer 时序建模可与 CausalCNN 去噪对比；ACM MM 投稿必引的 Vi-Fi 后续工作，可作 tracklet 缺失段补全的 baseline。

### A2. ViFi-Loc: Multi-modal Pedestrian Localization using GAN with Camera-Phone Correspondences ★★★
- **出处**：ACM ICMI 2023 | Hansi Liu（Vi-Fi 一作）后续工作
- **PDF**：`papers/A_2_ViFi-Loc.pdf`
- **摘要**：用 GAN 学习行人 camera-phone 对应关系，推理时仅凭手机数据（GPS+IMU+FTM）生成精化 3D 位置（1-2m 误差）。生成坐标可反向与 bbox 关联，实现自学习式数据扩充。
- **相关性**：证明 camera-phone 对应可被生成模型内化；"生成坐标再关联 bbox"的自监督闭环对你的 MLP 匹配器伪标签自举有直接启发；也是行人出视野时维持 phone-stable ID 的互补方案。

### A3. Mission: mmWave Radar Person Identification with RGB Cameras ★★
- **出处**：ACM SenSys 2024
- **PDF**：`papers/A_3_Mission_SenSys24.pdf`
- **摘要**：首个 mmWave 雷达↔RGB 相机跨模态行人 ReID：雷达检测到的目标可在其他相机区域图像库中检索。跨模态相似度估计挖掘 2D 图像与 3D 雷达点云的协同，免预注册，58 人实验 85% top-1。
- **相关性**：代表 RF-视觉身份关联从"轨迹几何匹配"到"跨模态特征嵌入检索"的路线，与你的测距+深度几何匹配形成方法论对照；open-set 免注册设定与度量学习损失可为 MLP 匹配头升级为对比嵌入提供参考。

### A4. ViFiCon: Vision and Wireless Association Via Self-Supervised Contrastive Learning ★★★
- **出处**：CVPR 2026 Workshop MULA | Vi-Fi 团队
- **PDF**：`papers/A_4_ViFiCon_CVPRW26.pdf` | [arXiv](https://arxiv.org/abs/2210.05513)
- **摘要**：自监督对比学习框架，学习 RGB-D 深度序列与 WiFi FTM 序列的跨模态关联，无需人工标注。多人深度序列堆叠为图像表示，"场景级时间同步"pretext 任务预训练。仅用 depth+FTM 在 2.5s 滑窗内达 92.63% vision-to-wireless 关联准确率。
- **相关性**：**输入模态与你几乎一致（depth+FTM）——最直接的对标方法**。论文中应正面对比关联精度/IDF1；其自监督路线反衬你监督方案的轻量高效；时间同步 pretext 任务可借鉴以缓解标注稀缺。

### A5. OOSTraj: Out-of-Sight Trajectory Prediction With Vision-Positioning Denoising ★★★
- **出处**：CVPR 2024 | Northeastern/Toyota
- **PDF**：`papers/A_5_OOSTraj_CVPR24.pdf` | [arXiv](https://arxiv.org/abs/2404.02227)
- **摘要**：定义"视野外轨迹预测"新任务：行人被遮挡时仅凭含噪无线定位序列预测无噪视觉轨迹。无监督视觉-定位去噪模块 + 传感器轨迹到像素坐标映射。在 Vi-Fi 和 JRDB 上双 SOTA。
- **相关性**：同在 Vi-Fi 数据集上处理 FTM 噪声，其无监督去噪与你的 CausalCNN 监督去噪构成直接对比，审稿人很可能要求引用；相机标定投影做法可增强测距-深度匹配的几何一致性。

### A6. LTrajDiff: Layout Sequence Prediction From Noisy Mobile Modality ★★★
- **出处**：**ACM MM 2023**（目标会议！）
- **PDF**：`papers/A_6_LTrajDiff_ACMMM23.pdf` | [arXiv](https://arxiv.org/abs/2310.06138)
- **摘要**：用含噪手机传感器通过去噪扩散模型预测被遮挡行人的完整 bbox 布局序列（位置+尺寸）。粗到细扩散去噪 + RMS 模块 + Siamese Masked Encoder，随机遮挡与超短输入设定下 SOTA。
- **相关性**：**发表于目标会议 ACM MM，证明"手机模态辅助视觉跟踪"在 MM 社区的接受度，写作框架可直接参考**；扩散式 bbox 生成可作长遮挡段插补备选，与你的轻量判别路线形成"重生成 vs 轻判别"对比。

### A7. Out-of-Sight Embodied Agents: Multimodal Tracking, Sensor Fusion, and Trajectory Forecasting ★★
- **出处**：IEEE TPAMI 2025/2026（OOSTraj 扩展版）
- **PDF**：`papers/A_7_OutOfSight_TPAMI25.pdf` | [arXiv](https://arxiv.org/abs/2509.15219)
- **摘要**：扩展到行人+车辆，改进 Vision-Positioning 去噪，系统对比 Kalman 等经典去噪方法，在 Vi-Fi 与 JRDB 上建立更强 benchmark。
- **相关性**：提供 Vi-Fi 数据集上 FTM 去噪的最完整 benchmark（含 Kalman 基线），你的去噪模块可直接借用其实验协议与对比表格设计；related work"无线辅助视觉跟踪"小节核心引文。

---

## 方向 B：跨模态身份匹配 / 人-设备绑定（5 篇）

### B1. Opt-in Camera: Person Identification in Video via UWB Localization ★★★
- **出处**：IROS 2025 | CyberAgent AI Lab
- **PDF**：`papers/B_01_OptInCamera_UWB.pdf` | [arXiv](https://arxiv.org/abs/2409.19891)
- **摘要**：隐私保护相机：仅记录佩戴 UWB 标签且同意入镜的人。UWB 轨迹与视觉 tracklet 时空对齐，Kalman 平滑 + 带约束线性分配求解 tag-track 全局匹配，8-23 人环境 10fps 实时。
- **相关性**：**与你的工作同构性极高**——同为"无线测距轨迹↔视觉 tracklet"全局约束分配，其线性分配与你的全局 Hungarian 可直接对比；UWB vs FTM 测距精度差异是好的讨论点；opt-in 隐私应用支撑 phone-stable ID 的应用价值叙事。

### B2. ImmTrack: Interpersonal Distance Tracking with mmWave Radar and IMUs ★★
- **出处**：ACM/IEEE IPSN 2023
- **PDF**：`papers/B_02_ImmTrack_mmWave_IMU.pdf`
- **摘要**：首个 mmWave 雷达 + 随身 IMU 的"跟踪+重识别"系统：匹配雷达轨迹与 IMU 惯性轨迹，把设备伪身份迁移到雷达轨迹。最多 27 人实验，分米-秒级精度。
- **相关性**："设备轨迹→匿名感知轨迹身份迁移"与你的 phone→tracklet 绑定是同一问题族（模态换成雷达）；其轨迹相似度匹配与多人歧义消解可作 baseline 思想对比；支撑"随身设备作为身份锚点"立论。

### B3. ViFi-ReID: A Two-Stream Vision-WiFi Multimodal Approach for Person Re-identification ★★
- **出处**：arXiv 2024 预印本（2410.09875，注意引用时标注）
- **PDF**：`papers/B_03_ViFiReID.pdf`
- **摘要**：视觉 + WiFi CSI 双流 ReID：商用路由器 CSI 捕捉步态特征，与视频流特征经跨模态对比学习融合，缓解遮挡/换装下纯视觉 ReID 失效。
- **相关性**：RF-vision ReID 直接代表；CSI 步态与 FTM 测距是互补的 WiFi 信息维度，可讨论"信道级 vs 测距级"两条 WiFi-视觉绑定路线的对比。

### B4. X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing ★
- **出处**：ICLR 2025 | NTU MARS Lab
- **PDF**：`papers/B_04_XFi.pdf` | [arXiv](https://arxiv.org/abs/2410.10167)
- **摘要**：模态不变人体感知基础模型，支持 RGB/深度/LiDAR/mmWave/WiFi-CSI 任意组合即插即用，transformer + X-fusion 跨模态融合，姿态估计与动作识别 SOTA。
- **相关性**：为引言提供"跨模态人体感知"大背景；模态缺失鲁棒融合对 FTM 缺失/NLOS 时的匹配退化处理有借鉴；与你的极小 MLP 形成"foundation model vs 轻量匹配头"定位区分。

### B5. PmTrack: Enabling Personalized mmWave-based Human Tracking ★★★
- **出处**：IMWUT/UbiComp 2023 (Vol.7 No.4) | 天津大学
- **PDF**：**未获取**（ACM 付费墙）| [ACM DL](https://dl.acm.org/doi/10.1145/3631433)
- **摘要**：用手机/手表 IMU 作身份指示器实现个性化 mmWave 多人跟踪。创新点：用"朝向"作雷达-IMU 匹配特征，规避模态异构与累积误差；配套检测增强、干扰抑制、轨迹连续性方案。5 人场景身份识别 98%/95%。
- **相关性**：与你是"同一问题、不同锚点特征"的镜像（朝向 vs 距离）；其匹配特征选择分析可直接启发你的消融实验设计；IMWUT 正刊，人-设备绑定小节必引。

---

## 方向 C：无线测距去噪与不确定性建模（5 篇）

### C1. NLOS Identification and Mitigation for Time-based Indoor Localization: Survey ★★
- **出处**：ACM Computing Surveys 2024
- **PDF**：**未获取**（可在线读：https://dl.acm.org/doi/full/10.1145/3663473）
- **摘要**：时间测距（ToF/TDoA/RTT）NLOS 识别与缓解系统综述，按机器学习/统计检验/单一与混合测量分类评述，聚焦 WiFi 与 UWB。
- **相关性**：为 CausalCNN 去噪模块提供最权威的相关工作分类框架；其"识别→缓解"两阶段范式可用于论证你端到端去噪+σ门限一体化设计的差异性。

### C2. WhereArtThou: A WiFi-RTT-Based Indoor Positioning System ★★★
- **出处**：IEEE Access 2024 | Samsung Research
- **PDF**：`papers/C_2_WhereArtThou_WiFiRTT.pdf`
- **摘要**：商用级 WiFi RTT（FTM）定位系统，EKF + 随机游走模型，可融合 IMU。核心贡献：**距离依赖的测量噪声模型**，并实证**人体持机遮挡对 RTT 测距误差的显著影响**。18 小时数据，90 分位误差 1.45-1.65m。
- **相关性**：**场景与你几乎完全同构**——手机持有者 FTM 测距 + 人体遮挡噪声。距离依赖噪声模型是 σ 自适应门限最直接的实证依据；人体遮挡 case study 可直接引用论证 FTM 噪声非平稳性，支撑可学习去噪的必要性。

### C3. UWB Ranging Error Mitigation with Novel CIR Features and Two-Step NLOS Identification ★★
- **出处**：Sensors 2024
- **PDF**：`papers/C_3_UWB_CIR_ErrorMitigation_Sensors.pdf`
- **摘要**：基于 CIR 特征的 UWB 误差缓解：决策树+FNN 两步 NLOS 识别，NLOS 误差细分三类分别校正。NLOS 识别 95.05%，LOS 误差降 50.4%，定位精度提升 54.46%。人体/墙体/玻璃三种遮挡实验。
- **相关性**："误差分类→分类型校正"可启发 CausalCNN 增加 LOS/NLOS 辅助分类头（multi-task）；人体遮挡是其显式实验变量，与持机遮挡场景直接对应。

### C4. Deep Bayesian Neural Networks for UWB Phase Error Correction ★★
- **出处**：Scientific Reports (Nature) 2025
- **PDF**：`papers/C_4_UWB_Bayesian_SciRep.pdf`
- **摘要**：双层贝叶斯 NN 融合框架校正 UWB 相位系统误差，物理约束 + 不确定性感知建模，**输出校准的置信区间**。角度误差降 94.7%。
- **相关性**：测距不确定性估计+置信度学习最新代表：贝叶斯 NN 输出校准置信区间可类比到你的 σ 估计——去噪网络同时回归校正值和逐测量方差，供自适应门限使用。

### C5. UWB NLOS Identification and Mitigation based on BERT ★★
- **出处**：IPIN 2024
- **PDF**：`papers/C_5_UWB_BERT_NLOS.pdf`
- **摘要**：BERT 自注意力做信道识别与误差校正：NLOS 识别 96.65%，比 CNN/LSTM 提升 11.86%/10.80%，NLOS 测距误差降 41.97%。
- **相关性**：自注意力序列模型对测距去噪的强 baseline，正好衬托 CausalDilatedCNN 在**在线因果、低延迟、小参数量**上的优势，是 ablation/讨论的理想对照。

---

## 方向 D：视觉 MOT 前沿方法（5 篇）

### D1. CAMELTrack: Context-Aware Multi-cue ExpLoitation for Online MOT ★★★
- **出处**：arXiv 2025 (2505.01257) | EPFL/UCLouvain
- **PDF**：`papers/D_01_CAMELTrack.pdf` | [GitHub](https://github.com/TrackingLaboratory/CAMELTrack)
- **摘要**：完全从数据学习关联策略的在线关联模块：注意力模块编码轨迹历史上下文 + 多线索（外观/运动/位姿）融合，取代手工代价矩阵。保持模块化 TbD，DanceTrack/SportsMOT/MOT17 SOTA。
- **相关性**：**与你的"学习匹配 MLP"最直接对应**——证明学习式关联函数优于手工加权。可借鉴"每线索独立编码+上下文融合打分"，把 FTM 距离作为额外 cue token；引用其作为"learned association 优于手工代价"的关键论据。

### D2. SUSHI: Unifying Short and Long-Term Tracking with Graph Hierarchies ★★
- **出处**：CVPR 2023
- **PDF**：`papers/D_02_SUSHI_GraphHierarchies.pdf`
- **摘要**：层次化 GNN 统一短时/长时关联：底层帧间关联成短 tracklet，高层复用同一 GNN 在 tracklet 级做长时距全局关联，端到端训练。长遮挡恢复能力突出。
- **相关性**：与"全局 Hungarian phone-track 分配 + 长时遮挡恢复"高度契合。可将 WiFi-track 全局分配视为其最高层的"外部身份锚点"，引用论证"纯视觉长时关联仍会失败、需额外模态锚定身份"。

### D3. Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking ★★★
- **出处**：AAAI 2024
- **PDF**：`papers/D_03_HybridSORT.pdf`
- **摘要**：在 OC-SORT 框架引入弱线索代价项（检测置信度状态、高度状态、速度方向），与 IoU/ReID 线性加权进 Hungarian 代价矩阵。几乎零开销，DanceTrack/MOT17/20 显著提升。
- **相关性**：**"在 OC-SORT 代价矩阵加新代价项"的最标准范式，与你的做法结构完全同构**——FTM 距离本质是更强的"物理弱线索"。其逐线索消融与权重敏感性实验可直接作你实验章节模板；必引对比基线。

### D4. UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation ★★
- **出处**：AAAI 2024
- **PDF**：`papers/D_04_UCMCTrack.pdf` | [GitHub](https://github.com/corfyi/UCMCTrack)
- **摘要**：放弃图像平面 IoU，在地平面（世界坐标）关联：检测框脚点投影到地面，Kalman 建模 + 归一化 Mahalanobis 距离作代价。纯运动线索，MOT17/20/DanceTrack/KITTI SOTA。
- **相关性**：证明"把关联搬到几何/世界坐标"能提升身份保持，为 FTM（米制几何量）与视觉轨迹在统一几何空间计算代价提供方法论支撑；提示 WiFi 代价项应按 FTM 方差做不确定性归一化（Mahalanobis 形式），而非固定权重。

### D5. SparseTrack: MOT by Scene Decomposition based on Pseudo-Depth ★★★
- **出处**：IEEE TCSVT 2025（arXiv 2306.05238）
- **PDF**：`papers/D_05_SparseTrack.pdf` | [GitHub](https://github.com/hustvl/SparseTrack)
- **摘要**：用单目"伪深度"（检测框底边 y 坐标）把稠密场景分解为稀疏深度层，由近到远级联匹配（DCM），每层内仅用 IoU 即可鲁棒关联。无需 ReID，遮挡场景 IDsw 明显降低。
- **相关性**：**"深度线索辅助关联"的代表作，直接支撑你的核心动机**——粗糙伪深度都能显著改善遮挡关联，FTM 真实米制测距应带来更强增益；"按深度分层级联匹配"可与 WiFi 距离分桶匹配结合。

---

## 总体分析与写作建议

### 领域格局（2023-2026）
1. **Vi-Fi 生态两条主线**：(a) Rutgers/Stony Brook 团队持续推进 phone↔vision 关联（ViFiT、ViFi-Loc、ViFiCon）；(b) Northeastern/Toyota 团队用 Vi-Fi 数据做遮挡轨迹预测，冲进 CVPR/ACM MM/TPAMI。**你的 phone-stable ID + 在线 MOT 层面的全局分配恰好填补两条线之间的空白**。
2. **人-设备绑定已成独立问题族**：UWB tag（B1）、IMU-雷达（B2/B5）、CSI 步态（B3）——你的 FTM-深度匹配是该族中"最低硬件门槛"（仅需手机+单目相机）的成员，这是重要卖点。
3. **测距去噪的学界共识**：噪声非平稳（人体遮挡主导，C2 实证）、需要不确定性输出（C4）、序列模型有效（C5）。你的 CausalCNN + 学习 σ 路线站得住，且在线因果性是差异化优势。
4. **视觉 MOT 的两个趋势与你同频**：learned association（D1）取代手工代价、几何/深度空间关联（D4/D5）取代纯 2D IoU。

### 论文定位建议
- 叙事主线：**"手机作为身份锚点的在线 MOT"**——对比 ViFiCon（离线关联，92.63% 窗口级精度）你做的是在线逐帧 MOT 集成与 IDF1 端到端提升。
- 必做实验对比：ViFiCon（A4）、Hybrid-SORT（D3）加 WiFi 线索版本、Opt-in Camera（B1）式约束分配。
- Related work 四小节直接对应 A/B/C/D 四方向。

### 下载统计
| 方向 | 找到 | PDF 成功 | 未获取 |
|---|---|---|---|
| A 无线+视觉融合 | 7 | 7 | 0 |
| B 跨模态身份绑定 | 5 | 4 | PmTrack (B5) |
| C 测距去噪 | 5 | 4 | CSUR Survey (C1) |
| D 视觉 MOT | 5 | 5 | 0 |
| **合计** | **22** | **20** | 2 |

---

# 第 2-3 轮：质量评估与补充调研（详见 evaluation.md）

第 2 轮对 22 篇逐篇核实 venue/CCF/引用/认可度后，剔除 5 篇（ViFiT、ViFiCon 为 workshop；ViFi-ReID、CAMELTrack 为预印本；C4 零引用），5 篇降为边缘参考。第 3 轮补充 13 篇 CCF A/B 论文如下。

## 第 3 轮新增论文（13 篇）

### A8. RoVaR: Robust Multi-agent Tracking through Dual-layer Diversity in Visual and RF Sensing ★★★
- **出处**：IMWUT 2023 (CCF A) | CMU/NEC/Georgia Tech/Stony Brook | 引用 3
- **PDF**：`papers/A_8_RoVaR_IMWUT23.pdf`
- **摘要**：视觉（VIO）与 RF（UWB 测距）双层多样性融合的多智能体跟踪：算法层用贝叶斯+学习混合方法特征级融合两模态的互补误差特性，系统层利用多智能体协同测距。NLOS/光照差/视觉特征稀疏等单模态失效场景下保持分米级精度。
- **相关性**：与你的 FTM+单目深度融合思路同构（RF 补视觉深度歧义、视觉补 RF 噪声）；其不确定性感知融合权重设计可直接借鉴到绑定代价函数。

### A9. Invisibility Cloak: Personalized Smartwatch-Guided Camera Obfuscation ★★★
- **出处**：ACM UIST 2025 (CCF A) | UCLA | 新作
- **PDF**：`papers/A_9_InvisibilityCloak_UIST25.pdf`
- **摘要**：智能手表 IMU 流与视频中各人腕部运动跨模态匹配，找出佩戴者对应 tracklet，再按隐私偏好选择性模糊画面。真实多人场景验证关联精度与实时性。
- **相关性**：核心问题与你完全一致（person-device binding），模态用 IMU 而非 FTM；其关联算法、多人歧义消解与隐私应用叙事可作直接对比/引用。

### A10. STNet: Deep Audio-Visual Fusion Network for Robust Speaker Tracking ★★
- **出处**：IEEE TMM 2024 (CCF B) | 北大 | 引用 11
- **PDF**：`papers/A_10_STNet_TMM24.pdf`
- **摘要**：统一坐标空间下音频-视觉融合说话人跟踪：声源似然图与视觉特征映射到同一空间，跨模态注意力对齐，质量感知模块在遮挡/噪声下自适应加权。AV16.3 超 SOTA。
- **相关性**：audio-visual 是"非视觉信号+视觉跨模态关联"成熟范式；统一坐标空间对齐与模态质量加权可类比迁移到 FTM 与深度序列的匹配。

### A11. MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset ★★
- **出处**：NeurIPS 2023 Datasets & Benchmarks (CCF A) | NTU | 引用 170
- **PDF**：`papers/A_11_MMFi_NeurIPS23.pdf`
- **摘要**：首个 WiFi CSI/mmWave/LiDAR/RGB-D/红外五模态同步标注 4D 人体感知数据集（40 人、27 万+帧），已成无线-视觉跨模态标准基准。
- **相关性**：related work"跨模态人体感知基准"必引；可用于跨模态匹配模型预训练或消融。

### A12. Babel: Scalable Pre-trained Model for Multi-Modal Sensing via Expandable Modality Alignment ★★
- **出处**：ACM SenSys 2025 (CCF B) | UW-Madison/MSR | 新作
- **PDF**：`papers/A_12_Babel_SenSys25.pdf`
- **摘要**：把六模态（WiFi/mmWave/IMU/LiDAR/视频/骨架）对齐分解为两两渐进对齐，解决多模态配对数据稀缺；多任务提升最高 12%。
- **相关性**："部分配对数据下的渐进式模态对齐"可用于 FTM-视觉配对样本有限时训练跨模态匹配网络；与 X-Fi 互补。

### C6. RLoc: Towards Robust Indoor Localization by Quantifying Uncertainty ★★★
- **出处**：IMWUT 2023 (CCF A) | 中科大 | 引用 31
- **PDF**：`papers/C6_RLoc_IMWUT23.pdf`
- **摘要**：对 WiFi AoA 估计不确定性量化（KL 散度损失+模型集成），用量化的不确定性指导多 AP 融合与时序跟踪，显著优于角度法与指纹法。
- **相关性**：与你的 σ 自适应匹配高度同源——给每次无线观测输出可信度再加权融合；同为 WiFi、同为学习型不确定性，是方法学参照与引用锚点。

### C7. Bayesian KalmanNet: Quantifying Uncertainty in Deep Learning Augmented Kalman Filter ★★
- **出处**：IEEE TSP 2025（非 CCF 目录 / 信号处理领域旗舰期刊，JCR Q1）| 引用 15
- **PDF**：`papers/C7_BayesianKalmanNet_TSP25.pdf`
- **摘要**：将贝叶斯深度学习（MC dropout/集成）引入学习型 Kalman 滤波，输出经校准的可信协方差，精度与不确定性校准均优于 KalmanNet 与 EKF。
- **相关性**："深度网络+滤波器+校准不确定性"的正统范式，支撑你 σ 自适应门控的理论合理性论证。

### C8. Indoor Localization System with NLOS Mitigation Based on Self-Training ★★★
- **出处**：IEEE TMC 2023 (CCF A) | 清华+BCAM | 引用 46
- **PDF**：`papers/C8_SelfTrainingNLOS_TMC23.pdf`
- **摘要**：自训练免标注 NLOS 缓解：地图+惯性+UWB 多源融合自动产生并迭代精化测距弱标签，NLOS 测距误差降 80%，90 分位定位误差 0.5m。
- **相关性**："多传感融合自动生成弱标签"可借鉴到 FTM 去噪——用视觉轨迹为去噪网络提供伪监督；高引 TMC 正刊对标。

### C9. RefLoc: Exploiting Anchor Links for NLOS Combating in UWB Localization ★★
- **出处**：ACM TOSN 2024 (CCF B) | 清华 | 引用 18
- **PDF**：未获取（付费墙）| DOI 10.1145/3657639
- **摘要**：利用锚点间已知位置的"锚链路"作参考在线判别 LOS/NLOS：NLOS 识别 96%（超 SOTA 10%），定位误差降 80%，开销小。
- **相关性**：免标注、按环境自适应判别测距质量的结构化先验路线，可作 σ 自适应之外的对比讨论。

### D6. MotionTrack: Learning Robust Short-Term and Long-Term Motions ★★★
- **出处**：CVPR 2023 (CCF A) | 西安交大 | 引用 159
- **PDF**：`papers/D_06_MotionTrack_CVPR23.pdf`
- **摘要**：短时用图卷积交互模块防密集场景漂移，长时用 Refind 模块基于轨迹历史特征×候选检测相关性矩阵找回长时丢失目标。MOT17/20 当时 SOTA。
- **相关性**：Refind 是"长时遮挡后身份恢复"代表方案，与你用 FTM 在视觉丢失期间维持身份的动机直接对标，是纯视觉基线与写作引用。

### D7. DiffMOT: Real-time Diffusion-based Tracker with Non-linear Prediction ★★
- **出处**：CVPR 2024 (CCF A) | 引用 96
- **PDF**：`papers/D_07_DiffMOT_CVPR24.pdf`
- **摘要**：扩散概率模型作运动预测器（D²MP）替代 Kalman，一步解码预测，实时 22.7 FPS，DanceTrack/SportsMOT 显著超 KF 类。
- **相关性**："学习式运动先验替代 KF"路线的高引代表，related work 运动模型改进方向的对照。

### D8. SMILEtrack: SiMIlarity LEarning for Occlusion-Aware MOT ★★★
- **出处**：AAAI 2024 (CCF A) | 引用 106
- **PDF**：`papers/D_08_SMILEtrack_AAAI24.pdf`
- **摘要**：类 Siamese 相似度学习模块（含 Patch Self-Attention）学习遮挡感知外观相似度 + 相似度匹配级联（SMC）。MOT17/20 超 ByteTrack。
- **相关性**：**与你的匹配 MLP 最直接对应的高引工作**——同样学习成对相似度打分器喂给匈牙利分配；其结构与消融设计可直接参照。

### D9. U2MOT: Uncertainty-aware Unsupervised Multi-Object Tracking ★★
- **出处**：ICCV 2023 (CCF A) | 阿里/浙大 | 引用 24
- **PDF**：`papers/D_09_U2MOT_ICCV23.pdf`
- **摘要**：显式度量关联不确定性：不确定性分层关联验证伪轨迹一致性 + 不确定性引导难样本挖掘。无监督达到部分有监督水平。
- **相关性**："关联不确定性→分层匹配决策"与你按 FTM 置信度自适应加权 WiFi 代价项同构，支撑不确定性建模动机。

---

## 最终清单统计（第 3 轮结束）

| 类别 | 数量 | 说明 |
|---|---|---|
| **合格（CCF A/B 或公认顶会）** | **25** | A方向 9 + B方向 4 + C方向 5 + D方向 7 |
| 边缘参考 | 5 | A2, B1, C2, C3, C5（CCF C 或非 CCF 但高相关） |
| 已剔除 | 5 | A1, A4, B3, C4, D1（workshop/预印本/零引用，PDF 保留） |
| PDF 已下载 | 31 | 含剔除与边缘论文 |
| 付费墙未获取 | 3 | PmTrack (B5)、NLOS Survey (C1)、RefLoc (C9)，DOI 已记录 |

逐篇质量核实数据（CCF 等级、引用数、代码 star、venue DOI 验证）见 `evaluation.md`。

---

# 第 4 轮：2025-2026 CVPR/ICCV/NeurIPS 主会补充（E 批次，12 篇）

> 调研日期：2026-07-30。全部核实为主会收录（openaccess.thecvf.com / papers.nips.cc），CCF A。因发表过新，不设引用量门槛，以 venue 硬指标为准。

## E 批次 I：MOT / 跨模态跟踪（E1-E6）

### E1. Occlusion-Aware SORT (OA-SORT) — CVPR 2026 ★★★
- **PDF**：`papers/E1_OA-SORT.pdf`
- **摘要**：即插即用免训练框架：遮挡感知模块（高斯图抑制背景）+ 遮挡感知偏移 + 偏差感知动量，修正部分遮挡下的关联代价。DanceTrack 63.1 HOTA；接入 4 个跟踪器平均 +2.08 HOTA / +3.05 IDF1。
- **相关性**：与 OC-SORT 同血统、直击遮挡代价失真、以 IDF1 为主指标。"视觉代价失效时引入外部证据"与你的 WiFi 代价项互补，强对比基线。

### E2. FDTA: Learning Discriminative Object Embeddings for MOT — CVPR 2026 ★★
- **PDF**：`papers/E2_FDTA.pdf`
- **摘要**：发现端到端 MOT 共享 DETR 嵌入类间相似度过高、缺实例级区分度。三路适配器（空间/深度感知、时间、身份质量感知对比学习）精炼关联嵌入。
- **相关性**："深度线索注入关联嵌入"与"FTM 作额外几何代价"动机同构，支撑学习匹配 MLP 的设计合理性。

### E3. HyperSSM: Hypergraph-State Collaborative Reasoning for MOT — CVPR 2026 ★★
- **PDF**：`papers/E3_HyperSSM.pdf`
- **摘要**：让运动状态相似的目标相互约束修正，遮挡时推断合理运动延续：超图模块捕捉空间运动相关性 + SSM 保证时间平滑。
- **相关性**：直击长时遮挡轨迹断裂/身份恢复；"预测噪声导致关联不稳"的设定与 FTM 去噪+不确定性代价动机呼应。

### E4. GRAE-3DMOT: Geometry Relation-Aware Encoder for Online 3D MOT — CVPR 2025 ★★★
- **PDF**：`papers/E4_GRAE-3DMOT.pdf`
- **摘要**：用几何关系感知编码器替代硬距离阈值门控：空间关系编码 + 时空关系编码 + 距离感知特征融合层，将几何距离显式编码进关联特征。
- **相关性**："把几何距离学习进关联代价而非硬阈值"与你的可学习 WiFi 代价项设计哲学高度一致，可借鉴其距离感知融合层。

### E5. ADMCMT: All-Day Multi-Camera Multi-Target Tracking — CVPR 2025 ★★
- **PDF**：`papers/E5_ADMCMT.pdf`
- **摘要**：低光下引入红外模态做多相机跟踪，首个 RGBT MCMT 数据集 M3Track + All-Day Mamba 融合模块（光照引导自适应加权）。
- **相关性**："视觉退化时引入辅助模态维持身份"与你的叙事完全平行；自适应模态加权可对照 WiFi 权重学习。

### E6. MVTrajecter: Multi-View Pedestrian Tracking with Trajectory Motion/Appearance Cost — ICCV 2025 ★★★
- **PDF**：`papers/E6_MVTrajecter.pdf`
- **摘要**：引入轨迹运动代价+轨迹外观代价，跨多个历史时间戳计算同一性代价并聚合，单时刻误关联可被其他时刻纠正。
- **相关性**：与"运动+外观+WiFi 三路代价→全局 Hungarian"结构直接可比；多时间戳代价聚合对跨遮挡窗口的身份恢复有直接借鉴价值。

## E 批次 II：RF/多模态感知与跨模态对齐（E7-E12）

### E7. RAPTR: Radar-based 3D Pose Estimation using Transformer — NeurIPS 2025 ★★
- **PDF**：`papers/E7_RAPTR_NeurIPS25.pdf`
- **摘要**：弱监督雷达 3D 姿态估计（仅 3D BBox+2D 关键点标签），伪 3D 可变形注意力+3D 模板/重力损失缓解深度歧义。HIBER/MMVR 误差降 34.3%/76.9%，开源。
- **相关性**："弱监督+几何先验损失缓解 RF 深度歧义"可借鉴到 FTM 不确定性建模；MERL 团队是 RF 感知必引。

### E8. MVDoppler-Pose: Multi-View mmWave for Self-Occluded Walking Pose — CVPR 2025 ★★★
- **PDF**：`papers/E8_MVDopplerPose_CVPR25.pdf`
- **摘要**：首个系统对比 mmWave 与相机做行走姿态估计，论证 RF 对距离和遮挡的天然鲁棒性；多视角 RF 融合解决雷达方向性问题，远距/自遮挡超相机方案。
- **相关性**：直接支撑"视觉遮挡/远距失效、RF 补盲"的动机叙事；多视角 RF 融合与 FTM 多 AP 去噪同构，motivation 有力引文。

### E9. Towards Foundational Models for Single-Chip Radar — ICCV 2025 ★
- **PDF**：`papers/E9_SingleChipRadarFM_ICCV25.pdf`
- **摘要**：100 万样本原始 mmWave 数据集 + GRT 4D 基础模型，3D 占据预测/语义分割媲美高端传感器；原始数据优于压缩表示（等效 10 倍数据）。
- **相关性**：为"廉价 RF 硬件+学习方法获得高层语义"提供背书；scaling 分析可支撑数据规模论证。

### E10. SATM Adapter: Spatial Alignment and Temporal Matching for Video-Radar — ICCV 2025 ★★★
- **PDF**：`papers/E10_SATM_VideoRadar_ICCV25.pdf`
- **摘要**：视频-雷达跨模态生理测量：空间对齐模块对齐两模态特征分布 + 时序匹配模块消除波形差异，适配器式微调复用单模态预训练。
- **相关性**：视觉-RF 跨模态对齐最新范式；"空间对齐+时序匹配"双分支与"深度轨迹-FTM 距离序列时空匹配"高度同构。

### E11. RCTDistill: Cross-Modal KD for Radar-Camera 3D Detection — ICCV 2025 ★★
- **PDF**：`papers/E11_RCTDistill_ICCV25.pdf`
- **摘要**：三模块跨模态蒸馏：RAKD 显式建模雷达距离-方位方向各向异性误差，TKD 对齐时序 BEV 特征，RDKD 蒸馏关系知识。nuScenes 雷达-相机 SOTA。
- **相关性**："按 RF 误差方向各向异性建模不确定性再跨模态对齐"正是 FTM（距离向噪声大）与视觉深度（深度向噪声大）联合建模可借鉴的方法论。

### E12. MotionBind: Multi-Modal Human Motion Alignment — NeurIPS 2025 ★★
- **PDF**：`papers/E12_MotionBind_NeurIPS25.pdf`
- **摘要**：人体运动序列纳入语言绑定的统一嵌入空间，运动-文本/视频/音频对比对齐 + 多条件运动生成，检索/识别/生成 SOTA。
- **相关性**：可类比地把 FTM 距离序列视为"运动模态"，与视觉 tracklet 嵌入共享空间做检索式绑定；对比对齐训练策略可参考。

---

## 最终清单统计（第 4 轮结束，2026-07-30）

| 类别 | 数量 | 说明 |
|---|---|---|
| **合格论文** | **37** | 第 2-3 轮 25 篇（CCF A/B）+ E 批次 12 篇（2025-2026 三大会主会） |
| 其中 2025-2026 年 | 18 | E 批次 12 + X-Fi、SparseTrack、Invisibility Cloak、Babel、Bayesian KalmanNet、TPAMI OOS |
| 边缘参考 | 5 | A2, B1, C2, C3, C5 |
| 已剔除 | 5 | A1, A4, B3, C4, D1（PDF 保留） |
| PDF 已下载 | 43 | papers/ 目录 |
| 付费墙未获取 | 3 | PmTrack、CSUR Survey、RefLoc（DOI 已记录） |
