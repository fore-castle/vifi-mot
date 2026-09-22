## Action 1 详细实现计划：3D Spatial Radio Prior + Kalman Filter 深度融合

> 日期：2026-06-25  
> 目标：用物理空间约束替代标量 depth-FTM 兼容性，并将无线信号融入 Kalman filter 的状态估计  

---

### 0. 现状诊断：当前方法为什么不够深

当前 Kalman filter 状态向量 `x = [u, v, s, r, du, dv, ds]`（像素空间），WiFi 信号完全不参与 filter 的 predict/update 循环。方案 A 的 `exp(-(depth-FTM)²/2σ²)` 只在 association cost matrix 中作为一个额外 penalty，且仅对已绑定 track 生效。

**三个根本问题：**

1. **标量比较丢失方向信息**：depth=5m 和 FTM=5m → 兼容性=1.0。但如果人在相机正前方 5m（camera frame Z=5m），而 AP 在人背后（AP→person 方向不同于 camera→person 方向），标量兼容性无法发现这种空间不一致。

2. **不修改状态估计**：即使 WiFi 告诉 tracker "这个人应该在 FTM=5m 处"，Kalman filter 的状态和协方差不受影响。遮挡期间 WiFi 信息被完全浪费。

3. **无法处理不确定性传播**：FTM std 大（多径严重）时应该增大 uncertainty 而非简单降低权重，但当前方法只用固定 σ=1.5m。

---

### Phase 0：深度数据可用性验证（2-3 天）

**目标**：确认 BBX5 中的 depth 值是否足够准确，能与 FTM 做有意义的空间比较。

**步骤 0.1**：检查原始 depth map 是否可访问

```python
# 检查 RAN 目录下是否有 Depth 数据
import os, pickle, numpy as np
raw_dir = "/Users/zstar/test/vifi/RAN"
for scene in os.listdir(os.path.join(raw_dir, "seqs", "outdoor")):
    depth_dir = os.path.join(raw_dir, "seqs", "outdoor", scene, "<seq_id>", "Depth")
    if os.path.exists(depth_dir):
        print(f"Found depth maps in {scene}")
```

如果 raw depth maps 不可用（IRB 限制），则只用 BBX5 中的 pre-extracted depth。

**步骤 0.2**：统计 depth vs FTM 的误差分布

对每个序列，逐帧比较：
- `det_depth` = BBX5 中 detection bbox 中心处的 depth（米，ZED SDK 输出）
- `ftm_range` = FTM_li 中的 range_mm / 1000（米，WiFi FTM 测距）

```python
# 对每个合法用户（有 phone 的 subject）
errors = det_depth - ftm_range  # 正值 = depth > FTM
```

**需要统计的量**：
- 误差分布的 mean（系统偏差，可能来自时钟不同步或 AP 位置偏移）
- 误差分布的 std（随机噪声，多径 + depth 估计误差）
- FTM std 与实际误差的相关性（FTM std 是否是好的不确定性指标？）
- 按 scene 分组统计（scene1-4 是否有不同的噪声特征？）

**判定标准**：
- |mean| < 1.5m 且 std < 2.5m → 直接可行
- |mean| > 1.5m → 需要先做 per-sequence bias 校准
- std > 4m → 3D 空间约束的区分力太弱，需要调整策略（比如加大 IMU 权重）

**步骤 0.3**：AP 位置推断

Vi-Fi 数据集中 AP 位置没有直接给出。但可以通过以下方式推断：

**方法 A**：利用 FTM 数据反推。如果某序列中只有 1 个 phone，且 phone 的运动轨迹可以从 BBX5 depth + 相机投影推断出来，那么 AP 位置 = 最小化 Σ|dist(track_3d, AP) - FTM|² 的点。

**方法 B**：查看 Vi-Fi 论文或 supplementary material 中的实验设置图。

**方法 C**（fallback）：不显式建模 AP 位置，而是把 "AP→person 方向" 用 FTM 差分（ΔFTM/Δt）+ IMU 航向联合估计。这样不需要 AP 绝对坐标。

---

### Phase 1：3D Spatial Radio Prior（核心创新 A）

#### 1.1 坐标系统定义

**Camera frame**（ZED2 右手坐标系，Y-UP）：
- Z_cam = depth（垂直于像平面的距离，米）
- X_cam = (col - cx) / fx × depth（水平方向，米）
- Y_cam = -(row - cy) / fy × depth（竖直方向，米，取负因为图像 y 轴向下）

**相机内参**（HD720，已知）：
- fx = 528.365, fy = 527.925
- cx = 638.925, cy = 359.2805

**2D detection → 3D camera frame**：

给定 detection 的 (cx_det, cy_det, depth_det)，3D 位置为：

```
Z_det = depth_det
X_det = (cx_det - cx) / fx × depth_det
Y_det = -(cy_det - cy) / fy × depth_det
```

**Camera frame → World frame**（简化，假设相机水平放置、高度 h_cam）：
- X_world = X_det
- Y_world = h_cam + Y_det（地面 Y_world ≈ 0）
- Z_world = Z_det

**注意**：ZED2 实际有轻微的俯仰角（tilt），但 Vi-Fi 的 tracking_parameters.set_as_static = True 表示使用静态场景模型，且相机高度 ~2.5m，tilt 角很小（< 10°）。对于距离精度 ~1m 的需求，忽略 tilt 引入的误差 < 0.5m，可以接受。

#### 1.2 Phone 3D 位置估计

**FTM 给出的信息**：phone 到 AP 的距离 r_ftm（米），标准差 σ_ftm（米）。

**单 AP 不能三角定位**，但我们可以联合多个约束：

**约束 1：FTM 距离球面**
```
||p_phone - p_AP||² = r_ftm²
```
这定义了一个以 AP 为圆心、r_ftm 为半径的球面。

**约束 2：地面平面**（phone 在人行高度 ~1.0m）
```
Y_phone = h_phone ≈ 1.0 m
```
球面与平面相交为一个圆。

**约束 3：IMU 航向**（9 轴 IMU 中的 magnetometer + gyroscope）
```
heading = atan2(mag_y, mag_x)  # 或者从 quaternion 提取 yaw
```
这给出了 phone 的运动方向。结合前一个时刻的 phone 位置估计，可以约束当前位置在 heading 方向上。

**综合估计**（per-frame，per-phone）：

```python
# 已知：AP 位置 p_AP（3D），FTM range r_ftm，IMU heading θ
# 假设：phone 高度 h_phone = 1.0m

# 1. 在 Y = h_phone 平面上，FTM 约束变成 2D 圆：
#    (X - X_AP)² + (Z - Z_AP)² = r_ftm² - (h_phone - Y_AP)²
r_horizontal = sqrt(max(r_ftm² - (h_phone - Y_AP)², 0))

# 2. IMU heading 给出圆上的方向约束：
#    如果前一帧 phone 位置为 p_prev，则当前位置应在 p_prev + heading 方向上
#    与圆的交点即为 phone 估计位置

# 3. 如果没有前一帧估计（首帧），取圆上离相机最近的点
#    （因为 camera 视角有限，远处的人不太可能被检测到）
```

**输出**：`p_phone_est = (X, Y, Z)` 3D 估计位置 + `Σ_phone` 3×3 协方差矩阵。

**协方差建模**：
```
Σ_phone = R_heading × diag(σ_along², σ_height², σ_cross²) × R_heading^T

其中：
- σ_along = σ_ftm（FTM 距离方向的 uncertainty，由 FTM std 给出）
- σ_height = 0.3m（人行高度的先验 uncertainty）
- σ_cross = r_ftm × sin(σ_heading)（垂直于 heading 方向的 uncertainty）
- σ_heading ≈ 15° = 0.26 rad（magnetometer 航向精度的典型值）
- R_heading = rotation matrix from heading angle
```

这个协方差矩阵的关键性质：沿 heading 方向 uncertainty 小（FTM 精度），垂直方向 uncertainty 大（heading 精度限制）。这比当前的标量 σ=1.5m 高斯更有物理意义。

#### 1.3 3D Spatial Compatibility（替代当前标量兼容性）

**当前方法**：
```
compat_scalar = exp(-(depth - r_ftm)² / (2 × 1.5²))
```
比较两个标量距离，固定 σ=1.5m。

**新方法**：
```
# track 的 3D 位置（从 detection depth 投影）
p_track = (X_det, Y_det, Z_det)  # camera frame → world frame

# phone 的 3D 位置估计 + 协方差
p_phone = (X_phone, Y_phone, Z_phone)
Σ_phone = 3×3 covariance matrix

# Mahalanobis 距离（比 Euclidean 更合理，因为它考虑了方向性 uncertainty）
Δp = p_track - p_phone
d_maha = sqrt(Δp^T × Σ_phone⁻¹ × Δp)

# 空间兼容性（3D 高斯）
compat_3d = exp(-0.5 × d_maha²)
```

**为什么 Mahalanobis > Euclidean > 标量深度差**：

| 方法 | 信息利用 | 问题 |
|---|---|---|
| 标量 depth-FTM | 只用 1D 距离 | 丢失方向，固定 σ |
| Euclidean 3D | 用 3D 位置差 | 各方向等权，不区分 FTM 精度和 heading 精度 |
| **Mahalanobis 3D** | 用 3D 位置差 + 方向性 uncertainty | **沿 heading 方向严格，垂直方向宽松**——符合物理 |

**具体例子**：假设 phone 在 AP 正前方 5m，heading 朝北。FTM 精度 1m，heading 精度 15°。
- Track A 在 phone 正前方 1m（沿 heading）：d_maha ≈ 1/σ_ftm = 1.0 → compat ≈ 0.61
- Track B 在 phone 侧面 1m（垂直 heading）：d_maha ≈ 1/σ_cross = 1/(5×sin15°) = 1/1.29 = 0.78 → compat ≈ 0.74
- 两个 track 到 phone 的 Euclidean 距离都是 1m，但 Mahalanobis 区分了方向

#### 1.4 Cost Matrix 集成

**修改 OC-SORT 的 `_associate` 方法**：

当前 cost matrix（方案 A，仅对已绑定 track）：
```
cost[i,j] = (1 - IoU) + inertia × angular_cost + wifi_weight × (1 - compat_scalar)
```

新的 cost matrix（对所有 track，不需要先绑定）：
```
cost[i,j] = (1 - IoU) + inertia × angular_cost + λ_spatial × (1 - compat_3d[i,j])
```

**关键改进**：
- `compat_3d[i,j]` 对所有 (detection_i, track_j) 对计算，不限于已绑定 track
- 这意味着 WiFi 信号从第一帧就参与 association，不需要先 "bind" 再 "use"
- `λ_spatial` 替代 `wifi_weight`，初始值通过 Phase 0 的误差统计确定

**新增：AP 位置推断模块**（如果 AP 位置未知）：

前 10 帧的 warmup 期间，收集所有 (detection, FTM) 对，用最小二乘法估计 AP 位置：
```
p_AP = argmin_{p} Σ_t Σ_subject (||p_det(t, subj) - p|| - r_ftm(t, subj))²
```
其中 `p_det(t, subj)` 是 t 时刻 subject 的 3D 位置（从 depth 投影），`r_ftm(t, subj)` 是对应的 FTM 测距。

---

### Phase 2：Kalman Filter 无线观测融合（核心创新 B）

#### 2.1 为什么要在 Kalman filter 中融合无线信号

**当前 Kalman filter 的信息流**：
```
Predict:  x_{k|k-1} = F × x_{k-1}    (纯视觉运动模型)
Update:   x_k = x_{k|k-1} + K × (z_vis - H × x_{k|k-1})   (纯视觉观测)
```

WiFi 信号完全不参与。这意味着：
- 遮挡期间（无 z_vis）：track 只能靠 predict 漂移，没有 WiFi 修正
- 近距离行人：视觉 association 容易混淆，WiFi 约束被浪费
- Kalman 协方差 P 不反映 WiFi 提供的额外信息

**融合后的信息流**：
```
Predict:       x_{k|k-1} = F × x_{k-1}                    (运动模型)
Visual Update: x_k* = x_{k|k-1} + K_vis × (z_vis - H × x_{k|k-1})  (视觉观测)
Wireless Update: x_k = x_k* + K_wire × (z_wire - h(x_k*))           (无线观测)
```

这是一个 **sequential update**（也叫 cascaded Kalman update）：先用视觉更新，再用无线更新。在卡尔曼滤波理论中，当两个观测独立时，顺序更新等价于联合更新（见 Bar-Shalom, "Estimation with Applications to Tracking and Navigation", Ch.9）。

#### 2.2 状态向量选择

**方案选择：保持 2D 像素空间状态 + 3D 观测模型**

不改 Kalman 状态向量 `x = [u, v, s, r, du, dv, ds]`，但把无线观测建模为状态的非线性函数。

**理由**：
- 最小改动量——现有的 IoU matching、OCM、ORU 全部不用改
- 2D→3D 投影是确定性的（depth 已知），不需要在状态中维护 3D 位置
- 无线观测的非线性通过 EKF（Extended Kalman Filter）处理

#### 2.3 无线观测模型（EKF 推导）

**无线观测量**：`z_wire = r_ftm`（FTM 测距，米）

**观测函数 h(x)**：从 Kalman 状态 x 预测 FTM 测距

```
# 从状态提取 2D 位置 + depth
u = x[0]  # bbox 中心 x (pixels)
v = x[1]  # bbox 中心 y (pixels)
depth = last_depth  # 最近一次观测的 depth（米）

# 2D→3D 投影
X = (u - cx) / fx × depth
Y = -(v - cy) / fy × depth
Z = depth

# 预测的 FTM 距离（从 AP 到 track 的 3D 距离）
h(x) = ||(X, Y, Z) - p_AP|| = sqrt((X-X_AP)² + (Y-Y_AP)² + (Z-Z_AP)²)
```

**Jacobian H_wire = ∂h/∂x**（7×1 向量）：

设 `d = h(x) = ||p_track - p_AP||`，则：

```
∂h/∂u = [(u - cx) / fx × depth - X_AP] / d × depth / fx
       = (X - X_AP) / d × depth / fx

∂h/∂v = [-(v - cy) / fy × depth - Y_AP] / d × (-depth / fy)
       = -(Y - Y_AP) / d × depth / fy

∂h/∂s = 0  (area 不直接影响 3D 位置)

∂h/∂r = 0  (aspect ratio 不直接影响 3D 位置)

∂h/∂du = 0  (速度不直接影响位置)
∂h/∂dv = 0
∂h/∂ds = 0
```

**简写**：
```
H_wire = [∂h/∂u, ∂h/∂v, 0, 0, 0, 0, 0]^T   (1×7 行向量)
```

其中：
```
∂h/∂u = (X - X_AP) × depth / (d × fx)
∂h/∂v = -(Y - Y_AP) × depth / (d × fy)
```

**直觉**：Jacobian 告诉我们 "track 在图像上移动 1 像素，FTM 距离会变化多少"。这取决于 track 相对于 AP 的方向——如果 track 沿着 AP 方向移动，FTM 变化大；如果垂直于 AP 方向移动，FTM 变化小。

#### 2.4 观测噪声 R_wire（从 FTM 不确定性推导）

**标准 Kalman update 中**：`R` 是观测噪声的方差（对于标量观测，R 是一个标量）。

**FTM 测距噪声的物理来源**（参考 WiFi FTM 文献）：
1. **硬件偏差**（manufacturer-dependent）：~0.5m 系统偏差
2. **多径效应**（blocker-dependent）：室内可达 2-3m
3. **时钟漂移**：Vi-Fi 已做时钟同步修正
4. **NLOS（非视距）**：遮挡导致信号绕路，距离偏大

**R_wire 建模**：
```
R_wire = σ_ftm² + σ_depth² + σ_AP²

其中：
- σ_ftm = FTM 自带的 std（每帧给出，通常 0.5-2m）
- σ_depth = depth 估计误差（ZED2 典型值 ~0.1m for <5m, ~0.3m for 5-10m）
- σ_AP = AP 位置 uncertainty（Phase 0 推断精度，估计 ~0.5m）
```

**实际计算**：
```python
sigma_ftm = ftm_std / 1000.0  # mm → m, per-frame value
sigma_depth = 0.1 + 0.02 * depth  # 经验模型：距离越远误差越大
sigma_ap = 0.5  # AP 位置 uncertainty (fixed after Phase 0 calibration)

R_wire = sigma_ftm**2 + sigma_depth**2 + sigma_ap**2
```

**关键性质**：R_wire 是 per-frame、per-phone 自适应的。FTM std 大（多径严重）→ R_wire 大 → Kalman gain K_wire 小 → 无线观测对状态修正小。这自动实现了"信号差时降权"的效果，不需要人工设定权重。

#### 2.5 完整 Wireless Update 步骤

在每个帧的视觉 update 之后（如果有可用的 FTM 观测）：

```python
def wireless_update(track, phone_idx, ftm_range_m, ftm_std_m, p_ap, depth):
    """
    对一个已关联 phone 的 track，执行无线观测 update。
    
    参数：
        track: _OCTrack 对象（含 KalmanBox kf）
        phone_idx: 关联的 phone 索引
        ftm_range_m: FTM 测距（米）
        ftm_std_m: FTM 标准差（米）
        p_ap: AP 3D 位置（world frame，米）
        depth: track 最近一次的 depth 值（米）
    """
    kf = track.kf
    x = kf.x  # [u, v, s, r, du, dv, ds]
    P = kf.P  # 7×7 协方差
    
    # 1. 从状态提取 2D 位置，投影到 3D
    u, v = x[0], x[1]
    X = (u - CX) / FX * depth
    Y = -(v - CY) / FY * depth
    Z = depth
    
    # 2. 计算预测 FTM 距离
    dx = X - p_ap[0]
    dy = Y - p_ap[1]
    dz = Z - p_ap[2]
    d_pred = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # 3. Innovation（观测 - 预测）
    innovation = ftm_range_m - d_pred
    
    # 4. Jacobian H_wire (1×7)
    H_wire = np.zeros((1, 7))
    H_wire[0, 0] = dx * depth / (d_pred * FX)   # ∂h/∂u
    H_wire[0, 1] = -dy * depth / (d_pred * FY)   # ∂h/∂v
    # 其余为 0
    
    # 5. 观测噪声 R_wire
    sigma_depth = 0.1 + 0.02 * depth
    R_wire = np.array([[ftm_std_m**2 + sigma_depth**2 + SIGMA_AP**2]])
    
    # 6. Kalman gain
    S = H_wire @ P @ H_wire.T + R_wire     # 1×1
    K = P @ H_wire.T @ np.linalg.inv(S)     # 7×1
    
    # 7. State update
    kf.x = x + (K * innovation).flatten()
    
    # 8. Covariance update (Joseph form for numerical stability)
    I_KH = np.eye(7) - K @ H_wire
    kf.P = I_KH @ P @ I_KH.T + K @ R_wire @ K.T
```

**关键设计决策**：

- **用 Joseph form 而非简化 form**：因为 wireless update 的 K 可能较大（FTM 精度高时），简化 form `(I-KH)P` 可能导致 P 不正定。
- **depth 的处理**：depth 不在 Kalman 状态中（状态只有 u,v,s,r + 速度）。每次 wireless update 使用最近一次观测的 depth 值。如果 track 已丢失（遮挡中），使用最后一次有效 depth。
- **只对已关联 phone 的 track 执行**：初始关联仍靠 IoU + OCM + 3D spatial compatibility（Phase 1），wireless update 在 track-phone 绑定后才激活。

#### 2.6 遮挡期间的 Wireless-Only Update（Tracklet Bridging 入口）

当 track 连续 n 帧未匹配到 detection（`time_since_update = n > 0`）时：

```python
# 正常帧：predict + visual_update + wireless_update
# 遮挡帧：predict + wireless_update（跳过 visual_update）

if track.time_since_update > 0 and track.bound_phone is not None:
    # 只有 predict 已经执行（在 _associate 之前）
    # 直接用 FTM 做 wireless update
    wireless_update(track, ...)
    # 这会修正 track 的位置估计，减少遮挡漂移
```

**为什么这有效**：
- Kalman predict 用常速度模型外推位置，遮挡越久漂移越大
- Wireless update 用 FTM 测距把 track "拉回" 正确的距离
- 虽然 FTM 精度只有 ~1m，但比纯靠 predict 漂移 10 帧（可能漂移 3-5m）好得多
- 这直接减少了 track 重新出现时的 ID switch 概率

#### 2.7 IMU 增强的运动模型（可选扩展）

当前运动模型是 constant velocity（F 矩阵中 du, dv 不变）。可以用 IMU 数据提供更准确的 velocity prior：

```python
# 从 IMU 估计 phone 的 2D 图像速度
# IMU 给出加速度（m/s²），积分得到速度
# 但 IMU 是 phone 端的，需要投影到图像平面

# 简化版本：用 IMU 的步态检测（step detector）和航向来修正速度
if imu_step_detected:
    # 人在走路 → 速度约 1.2-1.5 m/s
    speed_prior = 1.3  # m/s
    # 投影到图像平面：需要 depth 和相机参数
    du_image = speed_prior * cos(heading) * fx / depth  # pixels/frame
    dv_image = speed_prior * sin(heading) * fy / depth  # pixels/frame
else:
    # 人站着不动
    speed_prior = 0.0
    du_image = 0
    dv_image = 0

# 用 IMU 速度作为 process noise 的先验
# 修改 Q 矩阵中的速度项：
Q[4, 4] = max(du_image**2 * 0.1, 0.01)  # du 的 process noise
Q[5, 5] = max(dv_image**2 * 0.1, 0.01)  # dv 的 process noise
```

**这个扩展的价值**：
- 当 track 在图像中静止但 IMU 检测到步行时，Q 增大 → Kalman filter 对速度变化更敏感
- 当 track 在移动但 IMU 检测到静止时，Q 减小 → Kalman filter 更信任当前速度
- 这解决了"视觉和无线信号对人运动状态判断不一致"的问题

**判定**：先不实现（Phase 2 的可选扩展），优先保证 Phase 1 + Phase 2 基础版跑通。

---

### Phase 3：训练与评测协议

#### 3.1 不需要额外训练

Phase 1 和 Phase 2 的核心模块都是**无需训练的**：
- 3D Spatial Radio Prior：基于物理投影 + Mahalanobis 距离，无可学习参数
- Kalman EKF update：基于推导出的 Jacobian 和噪声模型，无可学习参数
- AP 位置推断：最小二乘法，不需要标注数据

**超参数**（需要调优，但不需要训练）：
- `λ_spatial`：3D spatial compatibility 在 cost matrix 中的权重（grid search on [0.1, 0.2, 0.4, 0.6]）
- `h_phone`：手机高度先验（默认 1.0m，可调 [0.8, 1.0, 1.2]）
- `σ_heading`：IMU 航向精度（默认 15°，可按 IMU 校准调整）

#### 3.2 评测协议：严格 Leave-One-Scene-Out 4-fold

沿用之前跑通的 LOSO 协议：
- 4 fold：scene1/scene2/scene3/scene4 轮流做 test
- 评测指标：MOTA / IDF1 / IDsw / FP / FN
- Baselines：
  1. OC-SORT（纯视觉）
  2. WiFi OC-SORT 方案 A（标量 depth-FTM 兼容性）
  3. WiFi OC-SORT 方案 D（Bi-LSTM affinity, strict LOSO）

#### 3.3 Ablation Study 设计

| 实验 | 内容 | 预期 |
|---|---|---|
| A1: 3D spatial only | Phase 1 only（替换标量兼容为 3D 兼容），不做 Kalman update | 3D 投影 + Mahalanobis 比标量深度差好多少？ |
| A2: Kalman EKF only | Phase 2 only（无线 update），用标量兼容做 association | Wireless Kalman update 对遮挡恢复的贡献 |
| A3: Full (A1 + A2) | Phase 1 + Phase 2 | 两者是否互补？ |
| A4: + Artificial occlusion | 在 A3 基础上，随机 mask 30% 的 detection，测 IDsw | 遮挡场景下的 bridging 效果 |
| A5: Fixed σ vs Adaptive R_wire | 对比固定 σ=1.5m 和自适应 R_wire | 自适应 uncertainty 的价值 |

---

### Phase 4：实现路线图

| 周 | 任务 | 产出 |
|---|---|---|
| Week 1 | Phase 0: depth 质量验证 + AP 位置推断 | 误差分布报告，AP 坐标 |
| Week 2 | Phase 1: 3D 投影模块 + Mahalanobis compatibility | `spatial_radio_prior.py` |
| Week 3 | Phase 2: Kalman EKF wireless update | 修改 `utils.py` KalmanBox + 新建 `wifi_spatial_ocsort.py` |
| Week 4 | Phase 3: LOSO 评测 + ablation | 结果表格 + 分析报告 |
| Week 5 | 调优 + bug fix + artificial occlusion 实验 | 最终数字 |

---

### 涉及文件变更清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `mot_baseline/trackers/utils.py` | 修改 | KalmanBox 增加 `last_depth` 和 `wireless_update()` 方法 |
| `mot_baseline/trackers/wifi_spatial_ocsort.py` | 新建 | 主 tracker：OC-SORT + 3D Spatial Radio Prior + EKF wireless update |
| `mot_baseline/trackers/spatial_projector.py` | 新建 | 2D→3D 投影 + AP 位置推断 + phone 位置估计 |
| `mot_baseline/data/vifi_mot.py` | 修改 | Frame 数据增加 IMU heading、FTM std 字段 |
| `mot_baseline/scripts/run_spatial.py` | 新建 | 运行 + 评测脚本 |
| `mot_baseline/results/spatial_radio_report.md` | 新建 | 实验报告 |

---

### 公式汇总（方便论文引用）

**2D→3D 投影**：
```
X = (u - cx)/fx × d,    Y = -(v - cy)/fy × d,    Z = d
```

**Phone 协方差**：
```
Σ_phone = R(θ) × diag(σ_ftm², σ_h², (r·sin σ_θ)²) × R(θ)^T
```

**Mahalanobis 空间兼容性**：
```
compat(p_track, p_phone) = exp(-0.5 × (p_track - p_phone)^T Σ_phone⁻¹ (p_track - p_phone))
```

**EKF 无线观测 Jacobian**：
```
∂h/∂u = (X - X_AP) × d / (||p - p_AP|| × fx)
∂h/∂v = -(Y - Y_AP) × d / (||p - p_AP|| × fy)
H_wire = [∂h/∂u, ∂h/∂v, 0, 0, 0, 0, 0]
```

**自适应观测噪声**：
```
R_wire = σ_ftm² + σ_depth² + σ_AP²
```

**Sequential Kalman Update**：
```
x* = x + K_vis(z_vis - Hx)                    # visual update
x** = x* + K_wire(z_ftm - h(x*))              # wireless update
```
