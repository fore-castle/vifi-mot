# 补充实验方案（投稿 ACM MM 前必做/可选清单）

> 生成日期：2026-07-31 | 基于两轮代码库调研（GitHub API 核实仓库状态与源码结构）
> 当前主结果：三思路组合 88.22% IDF1（+0.31 vs Phase 6B 87.91），完整消融矩阵见 ../findings.md
> 本文件夹用途：存放全部补充实验的代码、结果与记录

---

## 实验总览（按优先级）

| # | 实验 | 类型 | 工作量 | 状态 |
|---|---|---|---|---|
| S1 | OC-SORT 官方版对齐 | 堵审稿质疑 | 小 (<1天) | 待做 |
| S2 | Hybrid-SORT w/o ReID 基线 | 更强视觉基线 | 小 (<1天) | 待做 |
| S3 | Vi-Fi (IPSN'22) 关联逻辑重实现 | 同数据集方法对比 | 小-中 (1-2天) | 待做 |
| S4 | ViTag 官方复现 + 协议对齐 | 同数据集方法对比 | 小 (跑通) + 中 (对齐) | 待做 |
| S5 | YOLOv8 真实检测器重跑全套 | 真实性/增益放大 | 中 (1-2天) | 待做 |
| S6 | UCMCTrack 基线 | 更强视觉基线 | 中 (1-2天) | 可选 |
| S7 | SparseTrack DCM 抽取（伪深度 vs 真深度消融） | 强化卖点 | 中 (1-2天) | 可选 |
| S8 | σ̂ 时序可视化 + Oracle gap 分析 | 论文图表 | 小 | 待做 |
| S9 | ViFiCon / ViFiT 数字引用 | 论文表格 | 零（引用原文） | 写作时做 |

---

## S1. OC-SORT 官方版对齐（最高优先级）

**动机**：全部结果建立在自实现 OC-SORT 上，审稿人可质疑"基线偏弱抬高增益"。

**调研结论**（已核实 noahcao/OC_SORT 源码）：官方版与自实现有 5 处实质差异，最关键的是——
1. **输出框来源**：官方匹配帧输出 `last_observation`（=原始检测框，GT 场景下与 GT IoU=1）；自实现输出 KF 平滑框 → GT 检测评测下官方天然 MOTA/IDF1 更高
2. min_hits 用 `hit_streak`（连续命中）而非累计 hits
3. 第一轮关联：arccos 角度差 + 分数加权 + 匹配后 IoU 剔除
4. ORU 在 KF 内 freeze/unfreeze（含协方差恢复）
5. 有 giou/ciou/diou 选项

**做法**：拷贝官方 `trackers/ocsort_tracker/{ocsort,association,kalmanfilter}.py`（纯 numpy/filterpy/lap，无检测器耦合），薄封装 `update(np.hstack([dets, scores[:,None]]), img_size, img_size)`（scale=1）。跑 4 折 LOSO 纯视觉，然后把 WiFi 模块移植到官方底座上重跑最佳组合。

**预期结论**：若官方底座上 WiFi 增益依旧（65→88 量级不变），结果免疫"底座偏弱"质疑；若官方纯视觉更高（可能，因输出框差异），如实更新所有表格。

## S2. Hybrid-SORT w/o ReID（更强视觉基线）

**调研结论**（ymzis69/HybridSORT）：`hybrid_sort.py` 纯 numpy 无 torch，`update()` 接口与 OC-SORT 同构，直接吃 `[x1,y1,x2,y2,score]`。需从 `tools/run_hybrid_sort_dance.py` 抄 args 默认值。

**⚠️ 公平性声明（必须写进论文）**：GT 检测分数恒 1.0 → 其核心弱线索 TCM（置信度状态建模）恒为 0、BYTE 低分段恒空。实际生效的只有"四角点速度 OCM + 高度调制 IoU"。同理 ByteTrack 完全退化为 SORT（一句话说明即可，不接入）。

**做法**：接入后跑 4 折纯视觉；可选再做"Hybrid-SORT + 我们的 WiFi 绑定模块"验证方法可移植性。

## S3. Vi-Fi (IPSN 2022) 关联逻辑重实现（最重要的方法对比）

**调研结论**：官方仓库（vifi2021/Vi-Fi，2022-08 后无维护）是 MATLAB 管线 + 原始数据格式，直接跑不可行。但其核心只有：三个 affinity（FTM-深度差、IMU 航位推算轨迹 vs bbox 轨迹相关、步态事件）+ munkres 匈牙利，**约几百行 Python 可在 dfv4p4 上重实现**。

**做法**：
1. 重实现三 affinity + 窗口级（3s）bipartite 匹配，作为 phone-binding 模块插入我们的 OC-SORT 底座
2. 报两个口径：(a) 它原生的窗口级关联 accuracy（与原文 ~81-83% 对照 sanity check）；(b) 插入 tracker 后的逐帧 IDF1（与我们的 88.22 直接对比）
3. 诚实标注"重实现（reimplemented）"

**这是论文对比表的核心行**：同数据集、同任务血统的最直接前作。

## S4. ViTag（SECON 2022）官方复现

**调研结论**（bryanbocao/vitag，2024-03 仍在维护）：**唯一开箱即用 dfv4p4 的开源方法**——其 DATA.md 定义的正是我们用的目录结构（sync_ts16_dfv4p4/BBX5/FTM_li pkl）。TF 2.3/Keras 老环境，建议用其 conda spec 或 Docker。原生指标：1-3s 滑窗 IDP（论文报 88.39%）。

**做法**：
1. 按官方 `train.py -tsid_idx N` 跑其 leave-one-sequence-out，先复现原文 IDP
2. 协议对齐（二选一或都做）：(a) 把我们的绑定输出也算成窗口级 IDP，进它的协议；(b) 把它的 X-Translator 窗口匹配投影到逐帧 track ID，进我们的 IDF1 协议
3. 注意其划分是 leave-one-**sequence**-out 而我们是 leave-one-**scene**-out——需统一到 LOSO scene（对它更难，需说明）

## S5. YOLOv8 真实检测器重跑（真实性实验）

**动机**：GT 检测（score=1.0）不真实且使 MOTA 无区分力；Phase 4 噪声实验暗示检测退化时 WiFi 价值放大（bbox jitter 下 OC-SORT 掉 2.85pt、WiFi 法仅掉 0.42pt）。预期真实检测下我们的相对增益扩大。

**做法**：
1. **已核实：本地 dfv4p4 无原始图像帧**（`RGB_ts16_dfv4p4_ls.json` 仅是 2143 个时间戳的列表）。两条路径：
   - 路径 A：从 Vi-Fi 官网（sites.google.com/winlab.rutgers.edu/vi-fidataset）申请/下载原始 RGB 序列（需评估体积与可得性），YOLOv8x 推理生成检测
   - 路径 B（退化方案）：在 GT 框上注入系统性检测噪声模型（漏检率 p_miss(遮挡相关)、误检 FP/帧、bbox jitter σ 分级），比 Phase 4 更完整，可控且可复现
2. 检测框与 GT 用 IoU≥0.5 匹配获得深度（或 ZED 深度图采样，若可得）
3. 重跑：OC-SORT / 官方 OC-SORT / Hybrid-SORT / Phase6B / 三思路组合，报完整表
4. 附带收益：检测分数不再恒 1.0，S2 的 TCM、BYTE 二阶段恢复生效，基线对比更公平

**风险**：若本地无原始图像帧且官网数据太大/不可得，退化方案是在 GT 框上注入更真实的噪声模型（漏检+误检+jitter，比 Phase 4 更系统）。

## S6. UCMCTrack（可选）

纯运动 tracker，与"把关联搬到几何空间"的叙事契合。主要成本：cam_para 需交互式 GUI 手标（每场景 10-20 分钟，固定相机可行）；**更优路径**：用 ZED 真深度直接反投地面坐标替代其单目投影，顺便构成"单目估计 vs RGB-D"对照。

## S7. SparseTrack DCM 抽取（可选，卖点消融）

DCM（深度级联匹配）纯 numpy 自包含，把其伪深度（2000−bbox底边y）换成 ZED 真深度只需一行。构成三行消融：无深度 / 伪深度 (SparseTrack) / 真深度 / 真深度+FTM（我们）——直接支撑"真实测距 > 图像启发式深度"的核心论点。整仓不接（detectron2+pbcvt 编译负担）。

## S8. σ̂ 可视化与 Oracle gap 分析（论文图）

1. 选 2-3 段含遮挡事件的序列，画时序图：raw FTM / hetero μ / σ̂ 带 / GT 深度 / 遮挡事件标记——展示"σ̂ 在遮挡时自动膨胀"（网络知道自己不准）
2. Oracle gap 分解：当前 88.22 vs Oracle 92.17，逐序列对比找出剩余错误类型分布（绑定歧义 vs 轨迹断裂 vs 深度缺失），指导下一步

## S9. 引用数字（写作时）

ViFiCon（无代码，92.63% 窗口 IDP）、ViFiT（任务不同）、OOSTraj（任务不同）在对比表中引用原文数字并标注协议差异。**重要叙事点**：所有前作只报窗口级关联指标，无一报逐帧 MOT IDF1——逐帧在线 MOT 协议本身是我们的差异化贡献。

---

## 协议对齐总原则（写论文时的公平性声明）

1. GT 检测下分数相关机制（BYTE/TCM/fuse_score）失效——逐 tracker 说明实际生效模块
2. 窗口级 IDP 与逐帧 IDF1 不可直接比——双口径报告或明确标注
3. 重实现方法标注 "reimplemented"，官方代码标注 commit hash
4. 所有新基线同一 LOSO 4-fold scene 划分、同一 py-motmetrics 版本

## 目录规划

```
supplementary_experiments/
├── PLAN.md                  ← 本文件
├── s1_ocsort_official/      ← 官方 OC-SORT 接入
├── s2_hybridsort/
├── s3_vifi_reimpl/          ← Vi-Fi affinity 重实现
├── s4_vitag/
├── s5_yolo_detector/
├── s6_ucmctrack/            （可选）
├── s7_dcm_ablation/         （可选）
├── s8_visualization/
└── results/                 ← 统一 JSON 结果与汇总表
```

## 参考仓库（已核实）

- https://github.com/noahcao/OC_SORT
- https://github.com/ymzis69/HybridSORT
- https://github.com/vifi2021/Vi-Fi （MATLAB，仅参考逻辑）
- https://github.com/bryanbocao/vitag （dfv4p4 兼容）
- https://github.com/bryanbocao/vifit
- https://github.com/corfyi/UCMCTrack
- https://github.com/hustvl/SparseTrack
- https://github.com/Hai-chao-Zhang/OOSTraj （任务不同，仅引用）
