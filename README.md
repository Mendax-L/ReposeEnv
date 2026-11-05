# ReposeEnv

[Mendax-L/ReposeEnv](https://github.com/Mendax-L/ReposeEnv) 提供 PushT 和双臂 AlignBox3D 仿真、示范采集与策略执行；本项目的 π₀.₅ 数据导出、训练及验证由相邻的 [Pi05AlignBox3D](../Pi05AlignBox3D/README.md) 负责。

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
├── config/                   # 接口、频率与共用的训练/验证场景 YAML
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

功能代码按任务归属放置，接口、频率和训练/验证场景在 `config/` 中声明，其余采集参数支持命令行覆盖。测试、静态预览工具及其生成资料由开发时在仓库外维护。机器人 XML、OBJ 网格、来源和许可证位于 `alignbox3d/native/assets/arx_l5/`，运行时必须保留；训练数据及运行输出加入 `.gitignore`。

## 自动采集

流程为：接近 → 渐进加紧 → 抬起 → 搬运对齐 → 放下 → 渐进松开并撤离 → 稳定验收。每步控制两个末端和夹爪，箱体通过真实碰撞接触运动。只保存成功示范；指定 `--output` 时目录必须尚不存在；未指定时使用带时间戳的默认目录。默认对每只夹爪的停放位姿独立施加各轴 ±8 mm 扰动。`--episode-mode mixed` 按 `--recovery-fraction`、`--contact-error-fraction` 和 `--grasp-ready-fraction` 概率采样类型。下方生产配方使用 20% 错误抵箱和 20% 抓取就绪的尝试比例。

训练集和验证集共用 `config/alignbox3d_data.yml`。当前 π₀.₅ 生产配方在各 split 内混合完整正常、错误抵箱恢复和抓取就绪轨迹，训练共 200 条、验证共 24 条：

```powershell
python -m alignbox3d.collect --split train --episodes 200 --workers 4 --seed 500000 --recovery-fraction 0 --contact-error-fraction 0.20 --grasp-ready-fraction 0.20 --output training_data/lerobot/alignbox3d/pi05_train
python -m alignbox3d.collect --split validation --episodes 24 --workers 4 --seed 900000 --recovery-fraction 0 --contact-error-fraction 0.20 --grasp-ready-fraction 0.20 --output training_data/lerobot/alignbox3d/pi05_validation
```

采集完成后，使用 [Pi05AlignBox3D 的完整 recipe](../Pi05AlignBox3D/README.md) 导出固定高层任务数据，并从 LeRobot 的 `lerobot/pi05_base` 微调 π₀.₅。正式 recipe 使用 20000 步并每 5000 步保存 checkpoint；原来的 3000 步仅适合作为流程预检。原始轨迹的阶段句子仅供导出时校验，不直接作为模型训练指令。

`--data-config` 可指定另一份同结构配置。当前 `.yml` 文件内容必须使用 **JSON 语法**，由 Python 标准库 `json` 解析；后缀是历史命名，不能直接写成普通 YAML 缩进语法。采集前可运行 `python -m json.tool config/alignbox3d_data.yml > /dev/null` 检查语法。文件声明相机/光照随机范围、两侧的 `[start, stop)` seed 范围以及验证专用搭配规则。默认训练范围 `[500000,900000)`、验证范围 `[900000,1000000)`；采集器拒绝越界种子，工作进程使用同一配置。训练侧排除所有匹配 `holdout_rules` 的搭配，验证侧按 `visuals` 样例顺序轮换。配置和实际 split 保存到 `meta/data_split.json`、schema 和采集报告中。训练和验证使用不重叠 seed 段，避免重复物理场景。

`--workers` 默认 1，设为 4 即启动四个独立仿真进程。主进程按 seed 顺序合并成功轨迹，重新生成全局编号和统计，视频直接复制。`--episodes` 指成功条数；失败只记录到 `meta/collection.json`。达到 `--max-attempts` 仍不足目标时保留已成功的数据并报错；默认最多尝试成功目标条数的五倍。上述配方对尝试按 60% 完整正常、20% 错误抵箱重抓、20% 抓取就绪采样；连续 20 个 seed 分层轮换，使四种横向错误组合均能被抽到。成功筛选会改变落盘比例；检查 `meta/collection.json` 的 `saved_by_recovery_stage` 和 `saved_by_contact_pattern`。逐帧 `collection.recovery_mode_id` 为 0 正常、1 接近偏移、2 持箱偏移、3 错误抵箱重抓、4 抓取就绪，属于采集元数据，不进入模型输入。

抓点在上箱局部坐标系中采样：各自前后偏移 ±10 mm、高度偏移 ±5 mm，两点前后差 ≤2 mm、高度差 ≤5 mm。完整正常和错误恢复轨迹的夹持倾角每条采样 20–25°，共享扰动各轴 ±1.5°，每臂再加各轴 ±0.25°；从箱沿上方到闭合保持同一夹持方向。抓取就绪轨迹从箱沿已处于两指之间、下一步直接闭合的状态录制，半数为近垂直的 10°，半数为 20–25°；就绪初态之前的预定位动作不进入训练。初始箱体 roll/pitch 各 ±2°，允许通过 `--initial-tilt-deg` 调到最大 5°。夹爪起点扰动可用 `--initial-tool-jitter-mm` 调整到 0–15 mm。三个 fraction 分别控制偏移恢复、错误抵箱和抓取就绪的尝试比例，三者之和不得超过 1。错误抵箱时两只夹爪分别沿各自箱壁局部横向偏移 19 mm，按 seed 轮换外/外、内/内、外/内、内/外四种组合；必须测得至少一侧接触力 ≥1 N，并校验两只夹爪的指垫中心均处于对应箱壁的同一侧，避免把已经夹住箱壁的状态误标为失败。此后录制退出、重新抓取和最终对齐；制造错误接触的动作不进入训练。相同 seed 复现初态、抓点与恢复场景；重力稳定后按实测箱体姿态规划。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--seed` | 当前 split 的起点 | 首次尝试的随机种子，后续依次递增 |
| `--render-size` | 256 | 两相机录制宽度 |
| `--max-steps` | 3000 | 单次尝试的动作步数上限，默认按 YAML 的采样频率保留 100 s |
| `--approach-speed-mm-s` / `--withdraw-speed-mm-s` | 50 | 空载接近、撤离速度上限 |
| `--contact-speed-mm-s` | 15 | 箱沿附近接近、退出速度上限 |
| `--lift-speed-mm-s` / `--carry-speed-mm-s` / `--lower-speed-mm-s` | 20 | 抬起、搬运、放下速度上限 |
| `--free-angular-speed-deg-s` / `--angular-speed-deg-s` | 24 / 12 | 空载、接触或持箱时旋转速度上限 |
| `--gripper-speed-mm-s` | 4 | 驱动手指目标行程的变化速度上限 |
| `--lift-mm` | 20 | 箱体几何中心的规划抬升量 |
| `--initial-tool-jitter-mm` | 8 | 双夹爪停放位置各 XYZ 轴独立扰动范围，单位 mm |
| `--recovery-fraction` / `--recovery-offset-mm` | 0.25 / 10 | 尝试恢复采样的概率与偏离幅度上限 |
| `--contact-error-fraction` / `--contact-error-lateral-mm` | 0 / 19 | 错误抵箱重抓的尝试概率与两臂横向偏移；范围 18–20 mm |
| `--grasp-ready-fraction` | 0 | 从箱沿已在两指之间的就绪状态开始录制的尝试概率 |

夹爪用五次平滑曲线渐进开合，默认 12 mm 行程变化约需 5.6 s，再停稳至少 0.4 s。闭合时箱体异常升起超过 1.5 mm 或峰值速度超过 25 mm/s，该尝试会被拒绝。运动默认不启用额外加速度限制；需要时用 `--acceleration-mm-s2` 指定。其余随机范围和选项见 `python -m alignbox3d.collect --help`。

末端跟踪的停稳判据使用实测末端的帧间位姿变化：连续 10 帧满足实测平移速度 <1.5 mm/s、角速度 <0.5°/s，并且位置误差 <1 mm、朝向误差 <0.5°。默认 30 Hz 下这对应约 0.333 s，每帧位移 <0.05 mm、角变化 <0.0167°；等待上限 2.5 s。常量集中在 `alignbox3d/grasp_collection.py` 的 `TOOL_STILL_*` / `TOOL_SETTLE_TIMEOUT_S`。固定位置偏差不会产生帧间位移，因此不会再将 P 控制器用于补偿静载的指令速度当成实际运动。

### 视觉随机化

默认 `--visual-mode all`：工业柔和配色、三档箱体结构、物料、相机和光照随机化。三档结构为 `ribbed_level=0` 无筋、`1` 边缘加强筋、`2` 边缘与面板加强筋；箱体尺寸 240×300×120 mm。加强筋和物料只影响外观，保持箱体质量、惯量和碰撞结构一致。夹爪保留原始 ARX L5 可视网格；手指内部指垫碰撞盒增大，侧壁抓点上移 5 mm，以减少网格进入箱沿。环境记录夹爪接触的最大物理穿透深度；箱沿穿透超过 1.5 mm 的尝试判为失败。

相机位置各轴默认 ±10 mm、旋转各轴 ±1°，灯光强度系数 0.8–1.2，均由共用数据配置设置。训练可选模式为 `baseline/color/ribbing/contents/camera/lighting/all`；验证固定使用 `all`。搭配划分集中定义在 `config/alignbox3d_data.yml` 的 `splits.validation.holdout_rules` 中，例如以下 JSON 片段：

```json
{"holdout_rules": [{"name": "edge_dusty_blue", "ribbed_level": 1, "upper_color_id": 6}]}
```

这条规则保留“边缘加强筋＋灰蓝色上箱”，不论下箱颜色、桌面、物料类型、数量或配色如何变化，训练都排除这个搭配。单条规则内的字段同时匹配，多条规则任一匹配即保留；未列出的字段不参与判断。规则支持 `ribbed_level`、`upper_color_id`、`lower_color_id`、`table_color_id`、`content_type_id`、`content_amount_id`、`content_palette_id` 的任意非空子集。`visuals` 是具体验证/演示样例，必须匹配至少一条规则。

公共 `make_environment("alignbox3d")` 默认使用验证场景，可传入 `split="train"` 或 `data_config=...`。小 seed（如 0）会映射到所选范围内；直接传入范围内 seed 则保持原值。单独的颜色或箱体结构仍可在训练中出现，指定的属性搭配仅用于验证。旧数据不会自动过滤，规则变更后应重新生成训练/验证集。

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

| 环境 | 状态维数 | 动作 | 策略/采样频率 |
| --- | ---: | --- | ---: |
| `pusht` | 2 | 2 维 agent 绝对目标 XY，范围 `[0,512]` | 10 Hz |
| `alignbox3d` | 20 | 左右末端 XYZ、局部 roll/pitch/yaw 增量及夹爪绝对开度，共 14 维 | 30 Hz |

三个频率均写在环境 YAML 中：`fps` 是策略动作与数据采样频率，`control_hz` 是控制器目标更新频率，`physics_hz` 是物理积分频率。AlignBox3D 默认为 30/60/600 Hz；每个策略步用两次控制更新插值位置、朝向和夹爪目标，每次更新执行 10 个物理子步，只保存一帧观测和动作。PushT 保持 10/10/100 Hz。频率必须为正整数，后两项分别是前一项的整数倍；PushT 当前要求 `control_hz == fps`。

AlignBox3D 动作顺序由 `config/alignbox3d.yml` 定义：左末端 XYZ、左末端 RPY、左夹爪开度、右末端 XYZ、右末端 RPY、右夹爪开度。XYZ 使用世界坐标，RPY 使用末端局部坐标，按通用 roll、pitch、yaw 顺序排列；metadata 显式保存该顺序及组合公式 `R_next = R_previous @ Rz(yaw) @ Ry(pitch) @ Rx(roll)`。XYZ/RPY 分量范围为 `[-1,1]`，步长由 YAML 的 `translation_speed_m_s: 0.06`、`rotation_speed_deg_s: 30.0` 除以 `fps` 得到：30 Hz 下每个分量每策略步分别缩放 2 mm、1°，经 60 Hz 插值后每次目标更新为 1 mm、0.5°。这些限制按分量计算，单轴目标速度上限为 60 mm/s，三轴满幅约为 104 mm/s；实测末端运动仍取决于 IK 和伺服跟踪。`action.left_gripper`、`action.right_gripper` 是 `[0,1]` 的绝对开度比例，0 全闭、1 全开，映射驱动手指行程 0–44 mm。全零动作会闭合夹爪；保持当前目标位姿与开度请使用 `env.hold_action()`。`robot.*_gripper` 观测仍是实测手指行程，单位 m。

频率和步长语义随数据集、接口契约和 checkpoint 保存，其中 `timing` 记录 policy/recording/control/physics 四项频率。AlignBox3D 当前动作语义版本为 `dual_ee_delta_xyz_local_roll_pitch_yaw_gripper_abs_0_1/v4`；旧数据与 checkpoint 应匹配原环境，新配置需重新采集和训练。策略频率表示动作执行的仿真时间间隔；一次预测可返回多个动作，模型请求频率由执行的动作数决定，也不保证仿真实时运行。

重置后夹爪张开并与箱体分离，最低指尖至少高于旋转后箱体最高点 20 mm；未指定停放偏移且箱体平放时工具参考点 Z=0.362 m。`reset(options={"initial_tool_offsets_m": [[dx,dy,dz],[dx,dy,dz]]})` 可以分别设置两臂起点偏移，各分量限于 ±15 mm，并仍执行安全高度与 IK 检查。环境由动作直接控制末端，不执行隐式抓取或升降。

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

两个末端共用 MuJoCo 世界坐标系，XY 原点为下箱中心，桌面 Z=0.080 m。公共 robot state 只包含左右末端 XYZ、Rot6D 和夹爪位置，共 20 维；公共字段使用 `robot.left_end_effector_pos` 等名称，相机使用 `images.camera2`。旋转字段后缀约定为 `rot6d`、`wxyz` 或 `rotm`，分别表示 Rot6D、WXYZ 四元数和行优先 3×3 旋转矩阵。

箱体 yaw 为顺时针正，与原生右手四元数解出的 Z 角符号相反。`step()` 返回 `StepResult`，包含 `obs/reward/done/info/terminated/truncated`；其他公共方法包括 `reset/observe/render/get_keypoints/get_agent_positions/teleop_agent/contract/close`。

三维生命周期为 control/complete/failed。提前结束用 `env.request_finish()`，再执行一次 `env.step(env.hold_action())` 取得结束结果，该步不推进仿真。成功需要 XY 误差 ≤5 mm、yaw 误差 ≤3°、高度误差 <0.2 mm、倾斜 <0.02 rad、自由关节速度范数 <0.015，连续满足 0.5 s；自动专家在松爪撤离后启用验收。

## 数据与语言标签

LeRobot v3.0 = Parquet 数值数据、同步 MP4 视频和字段/轨迹元数据组成的数据格式。每帧保存动作前观测和动作，`next.*` 保存动作后结果；两个相机同时采集，视频采用 H.264，不含界面或动作标记。

每批目录包含 `data/`、`videos/`、`meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`meta/episodes/`、`meta/pushenv_schema.json` 与 `meta/provenance.json`。`pushenv_schema.json` 保留固定格式名以兼容已有数据；公共观测字段由环境 YAML 决定。自动示范额外保存采集报告、源码快照、视觉字典及语言字典。

逐帧语言阶段为：**接近 → 加紧 → 抬起 → 搬运对齐 → 放下 → 松开撤离**；错误抵箱轨迹先出现第七类 **重新抓取**，然后重新进入接近阶段。每类提供 4 种完整中文指令；每条轨迹按 seed 为各阶段选择一句，同一阶段保持不变。`language.stage_id` 为 0–6，`language.stage` 为 7 维独热向量；每帧 `task_index` 指向 `meta/tasks.parquet` 中实际使用的完整句子，不能当作阶段 ID。全部候选句与阶段关系保存在 `meta/language.json` 和 schema 中。未标注时 ID 为 -1、向量全零。这些字段独立于 20 维 robot state 与 14 维动作；16 个内部控制子阶段由 `alignbox3d/stage_labels.py` 映射为七类语言阶段。

`collection.*` 保存抓点、扰动、内部阶段和视觉参数，作为采集元数据。数据校验检查阶段 ID、独热向量和任务文本的一致性。`meta/collection.json` 记录所有尝试与拒绝原因；`meta/provenance.json` 记录参数、运行版本和源码指纹。

## 策略接口

`PolicyInterface` 对两个环境统一返回 YAML 选定的字段：`robot_state` 中是 `robot.*`，`env` 中是 `images.*` 与 `instruction`。接口字段的维度、单位、RPY 顺序和坐标系由原生 metadata 定义，YAML 只列出最终所需字段。导出契约不需要模型服务：

```powershell
python policy_interface.py --env alignbox3d --obs image --export-contract training_data/alignbox3d_contract.json
python policy_interface.py --env alignbox3d --host 127.0.0.1 --port 8765 --episodes 10
```

π₀.₅ 模型侧在 `lerobot-pi05` 环境运行；仿真侧在 `ReposeEnv` 环境运行。TCP = 本机或网络进程之间的通信连接，策略启动时核对维数、字段顺序、单位和动作语义。

排查在线推理时，可给 `policy_interface.py` 增加 `--trace 路径.jsonl`。它逐行刷新 `episode_start`、`prediction_request`、`prediction`、`action`、`episode_end` 事件：请求行含动作前 20 维状态和指令，预测行含 `[n,14]` 动作块，动作行含实际执行的 14 维动作、下一状态、奖励、成功标志及阶段。模型报错时，最后一条 `prediction_request` 也会保留。

在线 `predict.obs` 保留 `robot_state` 和 `env` 的命名字段；图像用 RGB uint8 HWC 的 shape/base64 编码，指令使用自然语言字符串（历史窗口使用字符串列表）。Pi05AlignBox3D 依据导出数据保存的接口契约拼接 `robot_state`，与训练的 `observation.state` 保持同一字段顺序。数据落盘时相机使用 LeRobot 的 `observation.images.camera2` / `observation.images.front`，语言通过逐帧 `task_index` 关联 `meta/tasks.parquet`。观测、图像和语言都是动作前同一时刻的数据。字段顺序、旋转语义或 FPS 改变后需重新采集和训练，旧 checkpoint 不会自动转换。

训练模型的闭环验证入口位于相邻 [Pi05AlignBox3D](../Pi05AlignBox3D/README.md) 的 `validate.py`。它启动 π₀.₅ 模型服务，再通过 `conda run -n ReposeEnv python` 执行本仓库的 `policy_interface.py`，输出成功率、逐步轨迹和 camera2 录像。运行命令及路径设置见该项目 README 第 6 步。
