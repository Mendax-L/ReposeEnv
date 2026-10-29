# ReposeEnv

[Mendax-L/ReposeEnv](https://github.com/Mendax-L/ReposeEnv) 提供 PushT 和双臂 AlignBox3D 仿真、示范采集与策略执行；模型训练由相邻的 [ActionExpert](../ActionExpert/README.md) 负责。

## 安装与结构

```powershell
Set-Location Z:\ReposeEnv
conda env create -f environment.yml
conda activate ReposeEnv
```

已有环境直接激活即可。本机解释器为 `E:\anaconda3\envs\ReposeEnv\python.exe`。

```text
ReposeEnv/
├── README.md                 # 安装、采集、动作与数据说明
├── environment.yml           # 仿真环境依赖
├── environment.py            # 统一环境工厂
├── policy_interface.py       # 策略接口、契约导出与 TCP 客户端
├── common/                   # 共享环境、遥操作、录制和 LeRobot 读写
│   └── recording.py          # 人工采集、动作回放、数据校验入口
├── alignbox3d/               # 双臂箱体对齐任务
│   ├── collect.py            # 自动成功示范采集入口
│   ├── grasp_collection.py   # 随机抓点与接触抓取专家
│   ├── parallel_collection.py # 独立进程采集与合并
│   ├── diversity.py          # 外观随机化与评估保留组合
│   ├── env.py                # 公共环境接口
│   └── native/               # 物理仿真、运动学和机器人资源
├── pushT/                    # PushT 环境与人工控制
└── training_data/            # 本地训练数据，不进入 Git
```

功能代码按任务归属放置，参数在代码中给出默认值并支持命令行覆盖。测试、静态预览工具及其生成资料由开发时在仓库外维护。机器人 XML、OBJ 网格、来源和许可证位于 `alignbox3d/native/assets/arx_l5/`，运行时必须保留；训练数据及运行输出加入 `.gitignore`。

## 自动采集

```powershell
python -m alignbox3d.collect --episodes 100 --max-attempts 500 --workers 4
```

流程为：接近 → 渐进加紧 → 抬起 → 搬运对齐 → 放下 → 渐进松开并撤离 → 稳定验收。每步控制两个末端和夹爪，箱体通过真实碰撞接触运动。只保存成功示范，默认输出到 `training_data/lerobot/alignbox3d/grasp-carry-日期时间/`；`--output` 可以指定一个尚不存在的目录。

`--workers` 默认 1，设为 4 即启动四个独立仿真进程。主进程按 seed 顺序合并成功轨迹，重新生成全局编号和统计，视频直接复制。`--episodes` 指成功条数；失败只记录到 `meta/collection.json`。达到 `--max-attempts` 仍不足目标时保留已成功的数据并报错；默认最多尝试成功目标条数的五倍。

抓点在上箱局部坐标系中采样：各自前后偏移 ±10 mm、高度偏移 ±5 mm，两点前后差 ≤2 mm、高度差 ≤5 mm。夹持朝向共享扰动各轴 ±1.5°，每臂再加各轴 ±0.25°；初始箱体 roll/pitch 各 ±2°，允许通过 `--initial-tilt-deg` 调到最大 5°。相同 seed 复现初态与抓点；重力稳定后按实测箱体姿态规划。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--seed` | 500000 | 首次尝试的随机种子，后续依次递增 |
| `--render-size` | 256 | 两相机录制宽度 |
| `--max-steps` | 6000 | 单次尝试的动作步数上限 |
| `--approach-speed-mm-s` / `--withdraw-speed-mm-s` | 50 | 空载接近、撤离速度上限 |
| `--contact-speed-mm-s` | 15 | 箱沿附近接近、退出速度上限 |
| `--lift-speed-mm-s` / `--carry-speed-mm-s` / `--lower-speed-mm-s` | 20 | 抬起、搬运、放下速度上限 |
| `--free-angular-speed-deg-s` / `--angular-speed-deg-s` | 24 / 12 | 空载、接触或持箱时旋转速度上限 |
| `--gripper-speed-mm-s` | 4 | 驱动手指目标行程的变化速度上限 |
| `--lift-mm` | 20 | 箱体几何中心的规划抬升量 |

夹爪用五次平滑曲线渐进开合，默认 12 mm 行程变化约需 5.6 s，再停稳至少 0.4 s。闭合时箱体异常升起超过 1.5 mm 或峰值速度超过 25 mm/s，该尝试会被拒绝。运动默认不启用额外加速度限制；需要时用 `--acceleration-mm-s2` 指定。其余随机范围和选项见 `python -m alignbox3d.collect --help`。

### 视觉随机化

默认 `--visual-mode all`：工业柔和配色、三档箱体结构、物料、相机和光照随机化。三档结构为 `ribbed_level=0` 无筋、`1` 边缘加强筋、`2` 边缘与面板加强筋；箱体尺寸 240×300×120 mm。加强筋和物料只影响外观，保持箱体质量、惯量和碰撞结构一致。

相机位置各轴 ±10 mm、旋转各轴 ±1°，灯光强度系数 0.8–1.2。可选模式为 `baseline/color/ribbing/contents/camera/lighting/all`。六个评估专用组合集中定义在 `alignbox3d/diversity.py` 的 `HELDOUT_VISUAL_CONFIGS` 中，采集器拒绝将这些组合写入训练集。

## 人工采集、回放与校验

```powershell
python -m common.recording --env pusht
python -m common.recording --env alignbox3d
python -m common.recording --validate training_data/lerobot/pusht
python -m common.recording --env pusht --actions training_data/pusht_commands.npy --output training_data/lerobot/pusht/replay-001
```

人工窗口使用原生宽度 512 的预览；三维同时显示总览与近景。空格开始/暂停、回车保存、R 丢弃、Esc 保存并退出。PushT 使用鼠标或 WASD；AlignBox3D 用 `1/2/3` 选择 inner/outer/双臂，方向键控制 XY，PageUp/PageDown 控制 Z，Q/E、I/K、J/L 控制旋转，O/P 调整夹爪开度。也可使用 `python -m alignbox3d.teleop` 或 `python -m pushT.teleop`。

动作文件为 `.npy` 的 `[帧数, 动作维度]` 数组，或 `.npz` 中的 `action` 数组；只接受真实控制器动作。人工录制和外部动作回放保存实际成功标志；自动采集入口额外执行“仅保存成功”的筛选。

## 环境与动作

契约 = 环境与模型共同遵守的字段、单位、动作顺序和采样频率。

| 环境 | 状态维数 | 动作 | 频率 |
| --- | ---: | --- | ---: |
| `pusht` | 20 | 2 维 agent 绝对目标 XY，范围 `[0,512]` | 10 Hz |
| `alignbox3d` | 66 | 每臂 XYZ/旋转向量增量＋夹爪绝对开度，共 14 维 | 50 Hz |

AlignBox3D 动作顺序为 `[inner.dx,dy,dz,rx,ry,rz,grip, outer.dx,dy,dz,rx,ry,rz,grip]`。位姿分量范围 `[-1,1]`，每步平移分量缩放 1 mm、旋转向量分量缩放 0.5°；旋转向量 = 方向表示旋转轴、长度表示旋转角的三维向量，使用共享世界坐标系右手方向。增量叠加到上一目标位姿。夹爪是绝对开度，`0` 全闭、`1` 全开，对应驱动手指行程 0–44 mm；`env.hold_action()` 保持当前目标位姿和夹爪开度。

重置后夹爪张开并与箱体分离，最低指尖至少高于旋转后箱体最高点 20 mm；平放时工具参考点 Z=0.362 m，最大初始倾斜时约 0.408 m。环境由动作直接控制末端，不执行隐式抓取或升降。

```python
from environment import make_environment

env = make_environment("alignbox3d", render_size=256)
try:
    state = env.reset(seed=0)
    observation = env.observe()
    poses = env.get_end_effector_poses()  # (2,7)，两末端的绝对 XYZ/WXYZ
    result = env.step(env.hold_action())
finally:
    env.close()
```

两个末端共用 MuJoCo 世界坐标系，XY 原点为下箱中心，桌面 Z=0.080 m；四元数按 WXYZ 排列，表示工具到世界的朝向。66 维状态的零起始切片如下：

| 切片 | 内容 |
| --- | --- |
| `[0:7]` / `[7:13]` | 上箱底面中心 XYZ/WXYZ、线速度与角速度 |
| `[13:25]` / `[25:37]` | 双臂关节位置、速度 |
| `[37:40]` / `[40:43]` | inner、outer 实际绝对 XYZ |
| `[43:49]` / `[49:52]` / `[52:54]` | 双末端目标 XYZ、箱体 XY/yaw 误差、双臂接触力 |
| `[54:56]` / `[56:58]` | 两夹爪实际、目标驱动手指行程，单位 m |
| `[58:62]` / `[62:66]` | inner、outer 实际绝对 WXYZ |

箱体 yaw 为顺时针正，与原生右手四元数解出的 Z 角符号相反。`step()` 返回 `StepResult`，包含 `obs/reward/done/info/terminated/truncated`；其他公共方法包括 `reset/observe/render/get_keypoints/get_agent_positions/teleop_agent/contract/close`。

三维生命周期为 control/complete/failed。提前结束用 `env.request_finish()`，再执行一次 `env.step(env.hold_action())` 取得结束结果，该步不推进仿真。成功需要 XY 误差 ≤5 mm、yaw 误差 ≤3°、高度误差 <0.2 mm、倾斜 <0.02 rad、自由关节速度范数 <0.015，连续满足 0.5 s；自动专家在松爪撤离后启用验收。

## 数据与语言标签

LeRobot v3.0 = Parquet 数值数据、同步 MP4 视频和字段/轨迹元数据组成的数据格式。每帧保存动作前观测和动作，`next.*` 保存动作后结果；两个相机同时采集，视频采用 H.264，不含界面或动作标记。

每批目录包含 `data/`、`videos/`、`meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`meta/episodes/`、`meta/pushenv_schema.json` 与 `meta/provenance.json`。`pushenv_schema.json` 保留固定格式名以兼容已有数据；`observation.end_effector_poses` 保存两末端 XYZ/WXYZ，展开为 14 维。自动示范额外保存采集报告、源码快照、视觉字典及语言字典。

逐帧语言标签固定为：**接近 → 加紧 → 抬起 → 搬运对齐 → 放下 → 松开撤离**。`language.stage_id` 为 0–5，`language.stage` 为对应的 6 维独热向量 = 该阶段位置为 1、其余为 0；每帧 `task_index` 指向 `meta/tasks.parquet` 中相应中文标签。未标注时 ID 为 -1、向量全零。这些字段独立于 66 维物理状态与 14 维动作；15 个内部控制子阶段由 `alignbox3d/stage_labels.py` 统一映射为六个语言标签。

`collection.*` 保存抓点、扰动、内部阶段和视觉参数，作为采集元数据。数据校验检查阶段 ID、独热向量和任务文本的一致性。`meta/collection.json` 记录所有尝试与拒绝原因；`meta/provenance.json` 记录参数、运行版本和源码指纹。

## 策略接口

`PolicyInterface` 根据模型选择 state/keypoint/image/all 观测，录制仍保留所有原生观测和双相机。导出契约不需要模型服务：

```powershell
python policy_interface.py --env alignbox3d --obs image --export-contract training_data/alignbox3d_contract.json
python policy_interface.py --env alignbox3d --host 127.0.0.1 --port 8765 --episodes 10
```

模型侧在 ActionExpert 环境运行；仿真侧在 ReposeEnv 环境运行。TCP = 本机或网络进程之间的通信连接，策略启动时核对维数、字段顺序、单位和动作语义。
