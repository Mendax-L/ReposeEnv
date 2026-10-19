# pushenv

三个环境使用同一条链路：**环境 → 人工采集 / 策略动作 → 全部观测 → LeRobot v3.0 数据**。数采始终记录关键点、RGB 和原生状态；算法在读取时选择自己的输入。

## 目录与入口

```text
pushenv/
├── environment.py          # 创建环境，统一 reset / step / observe
├── record.py               # 人工采集、直线路径生成、动作回放与数据校验
├── policy_interface.py     # 算法观测选择、动作执行、TCP 策略连接
├── pushT/                  # env.py、teleop.py、native/
├── pushBox2D/              # env.py、teleop.py、native/
├── pushBox3D/              # env.py、teleop.py、native/，含 ARX L5 模型资源
├── common/                 # 必需的共享环境、遥操作、录制与 LeRobot 实现
├── training_data/
│   ├── lerobot/
│   │   ├── pusht/          # 已有 PushT 数据，也可作为新批次的上级目录
│   │   ├── pushbox2d/      # PushBox2D 人工采集批次
│   │   └── pushbox3d/      # PushBox3D 人工采集批次
│   ├── llerobot/           # 程序生成的数据，按本项目指定路径单独存放
│   │   └── pushbox3d/      # scripted-straight-*，与 human-* 分开
│   ├── pusht/              # 保留的原始 Zarr 数据
│   └── pusht.zip           # 保留的原始归档
├── environment.yml         # 统一依赖
└── README.md
```

三个环境分别导出 `PushTEnv`、`PushBox2DEnv`、`PushBox3DEnv`；各自的 `env.py` 提供公共接口，`teleop.py` 生成真实人工指令，`native/` 保存仿真和渲染实现。`common/` 是三个入口实际依赖的运行代码。

## 安装与运行

在 `Z:\pushenv` 打开终端。首次创建环境，之后只需激活：

```powershell
conda env create -f environment.yml
conda activate pushenv
```

本机解释器为 `E:\anaconda3\envs\pushenv\python.exe`。仿真与数采运行在 `pushenv`，模型训练和推理使用模型自己的环境。

## 当前三个任务

| 环境名 | 当前控制方式 | state 维度 | action 维度 | 采样频率 |
| --- | --- | ---: | ---: | ---: |
| `pusht` | 单个圆形 agent 推动 T 块；绝对目标 XY | 20 | 2 | 10 Hz |
| `pushbox2d` | 左右两个抓点固定在盒边，控制两者中点的绝对目标 XY | 12 | 2 | 10 Hz |
| `pushbox3d` | ARX L5 双臂固定抓住上盒，同步执行世界坐标 XY 增量；升降自动完成 | 54 | 2 | 50 Hz |

PushT 和 PushBox2D 的动作范围是 `[0, 512]`，单位为原生 512×512 工作空间像素。PushBox2D 的两抓点同时平移，初始角度为 0；它们不再接受独立的四维动作。

PushBox3D 的动作是 `[dx, dy]`，每项限制为 `[-1, 1]`，每步乘以 `0.001 m`，同时应用到两只抓具。环境先将上盒从起始面自动抬升 **20 mm**，开放 XY 控制，对齐后自动降回起始面并验证落稳。升降由原生环境执行，人工和策略使用相同流程；共同工作空间和逆运动学可达性限制仍有效，双抓点间距不变。

三维盒体外尺寸为 `0.24×0.30×0.12 m`，壁厚 `0.008 m`。默认每次重置在 `X∈[-25,25] mm、Y∈[-20,20] mm` 内均匀采样上盒初始偏移，剔除距目标不足 10 mm 的点，偏航保持 0，初始 Z 为 0.200 m。同一个 `reset(seed)` 复现同一初态，换 seed 得到不同扰动；人工保存下一条时 seed 加 1，丢弃重试保持同 seed。需要固定测试位置时可用 `env.reset(0, options={"offset": [0.035, -0.020]})`，或设置 `options={"randomize": False}`。PushT、PushBox2D 沿用各自已有的 seed 随机初态。

## 人工采集

从 `Z:\pushenv` 运行所需的一条命令：

```powershell
python record.py --env pusht
python record.py --env pushbox2d
python record.py --env pushbox3d
```

窗口初始暂停。二维按 **空格开始 → 人工操作 → 回车保存**；三维按 **空格开始 → 自动抬升 20 mm → 方向键调整 XY → 对齐后自动降落、落稳并保存**。每条结束后暂停，空格开始下一条。

三维窗口底部增加中文操作状态：抬升期间显示“抬升未完成，暂不能操作 XY”，升起并稳定后显示“抬升已完成，可以开始 XY 操作”；暂停、落放和保存时会显示对应的锁定原因。`Actual lift` 是实际抬高量，目标为 20 mm。

| 操作 | PushT | PushBox2D | PushBox3D |
| --- | --- | --- | --- |
| 移动 | 按住鼠标左键移动，或 WASD | 方向键控制共同抓点中心 XY | 方向键控制世界坐标 XY；上/下为 +Y/−Y |
| 升降 | — | — | 自动抬升和降落，无 Z 手控 |
| 空格 | 开始 / 暂停 | 相同 | 开始 / 暂停；收尾降落期间继续执行 |
| 回车 | 保存当前非空片段并重置 | 相同 | 先降落、落稳，再保存并重置 |
| R | 丢弃当前未保存片段并重来 | 相同 | 先降落，再丢弃并重置 |
| Esc / 关闭窗口 | 保存当前非空片段并退出 | 相同 | 先降落、落稳，再保存并退出 |

PushBox2D 和 PushBox3D 只有共同控制，不使用 `1/2` 切换。二维键盘目标每步偏移 6 像素；三维每步产生最多 1 mm 的 XY 目标增量，松开按键发送零增量。主动暂停时不推进仿真；失去焦点会暂停人工控制，三维自动升降和收尾仍继续并完整记录。

三维成功要求 **XY 中心距离不超过 5 mm**，即 `sqrt(dx² + dy²) <= 0.005 m`，包含 5 mm 边界；回到起始面后的高度误差小于 0.2 mm、偏航误差小于 0.01 rad、倾斜小于 0.02 rad，并低速落稳 0.5 秒。回车或 Esc 提前结束时，也会完成降落；未达到对齐要求的轨迹标记为截断，不标成功。

抬高量 `lifted_height = 当前 Z - 原始落位 Z`，并非误差；抬升误差 `lift_error = lifted_height - 0.020 m`。原始落位 Z 为 0.200 m，升起目标为 0.220 m。`height_error` 相对当前阶段的目标高度：抬升/XY 阶段比较 0.220 m，落放/落稳阶段比较 0.200 m。高度检查用于确认自动升降到位，XY 距离用于衡量平面对齐。

默认输出到 `Z:\pushenv\training_data\lerobot\<环境名>\human-日期时间\`。指定批次目录、条数或图像分辨率：

```powershell
python record.py --env pushbox2d --output training_data/lerobot/pushbox2d/human-001 --episodes 20 --render-size 256
```

每保存一条就写入视频、Parquet 和元数据，非空输出目录会拒绝覆盖。二维默认上限为 3000 步，即 300 秒；三维为 6000 步，即 120 秒，到达上限后继续记录完成降落所需的步骤。`--max-steps` 修改上限，`--episodes 0` 表示持续采集，条数按实际保存数量计算。

GUI 在落放后使用单独的后台线程计算统计量、编码视频并写入数据，界面持续显示 `SAVING` 和耗时。写入完成后才增加保存条数、重置并开放下一条采集；保存期间不能重置或重复保存，关闭窗口也会等待当前写入完成。高清分辨率和视频质量保持不变。

成功、失败和环境时间限制会自动收尾并保存该条。二维人工提前保存或退出标记 `next.truncated=true`；三维在落稳后按实际结果标记成功或截断。丢弃重来使用相同 seed，保存下一条后 seed 加 1。

## 直线路径生成数据

同一个 Python 入口可生成 PushBox3D 的程序轨迹：**随机初态 → 自动抬升 20 mm → 沿固定直线移动到目标 → 自动落放、落稳并判断成功**。

```powershell
python record.py --env pushbox3d --straight-line --episodes 100 --seed 0
```

默认保存到 `Z:\pushenv\training_data\llerobot\pushbox3d\scripted-straight-日期时间\`；这里按指定使用 `llerobot`，与人工数据的 `training_data/lerobot/pushbox3d/human-*` 分开。`--output` 可指定新的空目录，`--speed-mm-s` 设置平面速度，默认 20 mm/s。此模式仅用于 PushBox3D，默认 100 条，seed 依次递增。

直线路径在 XY 控制开始时固定方向和终点，每步共同移动两只抓具，不写物体位置、不改变物理状态。采集内容仍包含完整原生状态、关键点、双相机 RGB、阶段及动作；升降阶段动作是 `[0, 0]`。程序轨迹在 `meta/provenance.json` 标记 `kind="scripted_straight_line"`、`human_demonstration=false`，不会标为人工示范。`meta/collection.json` 逐条保存 seed、初始偏移、实际成功标记、最终误差和直线横向偏差；失败轨迹如实保留，不通过改标签或换 seed 隐藏失败。

## 保存的完整数据

每一行按 `observation_t → action_t → next.*` 对齐：观测在动作执行前读取，`next.*` 描述动作执行后的结果。`action` 是送入环境的指令，原生控制器仍会施加工作空间限制。

| 字段 | PushT | PushBox2D | PushBox3D |
| --- | ---: | ---: | ---: |
| `observation.state` | 20：9 个关键点 XY + agent XY | 12：4 个角点 XY + 左右抓点 XY | 54：物体、关节、末端、目标和接触状态 |
| `observation.keypoints` | 18 | 8 | 24：8 个世界坐标 XYZ 角点 |
| `observation.agent_pos` | 2 | 4 | 6：双末端世界坐标 XYZ |
| `observation.images.front` | 默认 96×96 RGB | 默认 96×96 RGB | 默认 512×512 RGB，总览相机 |
| `observation.images.camera2` | — | — | 默认 308×512 RGB，胸前斜俯近景相机，可见箱体侧壁与双抓具 |
| `observation.phase` | — | — | 阶段编号：0 抬升、1 XY 控制、2 降落、3 落稳、4 完成、5 失败 |
| 其他观测 | 原生位姿、速度、目标、接触计数 | 相同 | 原生状态、盒体速度、目标位姿、关节位置/速度、末端目标、接触力与接触计数 |
| `next.*` | 奖励、done、terminated、truncated、success、success_known | 相同 | 相同 |

三维两个相机始终一起采集。`--render-size` 修改采集宽度，总览为正方形，近景高度为 `max(2, 2×round(宽度×0.3))`；例如宽度 256 时近景为 154×256。窗口始终使用原生宽度 512 的双相机预览，保持近景比例。RGB 均为 uint8，视频不含界面、鼠标光标或动作标记，采用 H.264 有损编码；数值状态和动作存入 Parquet。

抬升、降落和落稳阶段也逐帧记录，`action` 为二维零向量，`observation.phase` 描述动作前阶段。高清 RGB 使用采集目录内的临时磁盘缓冲，每条保存或丢弃后清理，避免整条双相机图像堆积在内存。

三维 54 维状态顺序是：盒体位置/四元数 7、盒体线/角速度 6、双臂关节位置 12、双臂关节速度 12、双末端位置 6、双末端目标 6、对齐误差 3、接触力 2。`inner/outer` 和二维 `external_agent/internal_agent` 是保留的字段名，当前分别指两臂、左右抓点；完整字段名称和单位保存在每批数据的元数据中。

LeRobot v3.0 = 用 Parquet 保存数值、MP4 保存同步 RGB、元数据描述字段和轨迹边界的数据格式。每批目录包含：

```text
human-日期时间/
├── data/chunk-000/file-000.parquet
├── videos/observation.images.front/chunk-000/file-000.mp4
├── videos/observation.images.camera2/chunk-000/file-000.mp4  # 仅三维
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    ├── episodes/chunk-000/file-000.parquet
    ├── pushenv_schema.json
    └── provenance.json
```

校验已有数据，或从真实控制器动作数组回放采集：

```powershell
python record.py --validate training_data/lerobot/pusht
python record.py --env pushbox2d --actions training_data/pushbox2d_commands.npy --output training_data/lerobot/pushbox2d/replay-001
```

动作文件由控制器提供：`.npy` 为 `[帧数, 动作维度]` 数组，`.npz` 使用 `action` 键。三维仅在 XY 控制阶段消耗动作行，自动阶段补零并完整记录；动作耗尽后先降落再结束。现有 PushT 原始 Zarr、压缩包及已转换的 206 条轨迹、25,650 帧全部保留，新数据默认写入独立批次。

## 环境与策略接口

三个环境共同支持 `reset(seed)`、`step(action)`、`observe(include_images=True)`、`get_keypoints()`、`get_agent_positions()`、`teleop_agent()`、`render()`、`close()`、`contract()`。`step()` 返回 `StepResult`，包含 `obs`、`reward`、`done`、`info`、`terminated`、`truncated`。

三维 `reset()` 从抬升阶段开始，持续调用 `step()` 推进整个周期；`result.info["phase"]` 为 `control` 时才执行 XY 指令。提前结束可调用 `env.request_finish()`，随后继续发送 `[0, 0]` 直到 `done`，让环境完成降落。

```python
from environment import make_environment

env = make_environment("pushbox2d", render_size=96)
try:
    state = env.reset(seed=0)
    observation = env.observe()
    action = env.teleop_agent().act(direction=(1, 0))
    result = env.step(action)
finally:
    env.close()
```

算法用 `PolicyInterface` 选择观测，不影响人工数据的记录内容：

```python
from policy_interface import PolicyInterface

with PolicyInterface("pushbox3d", obs="image") as interface:
    observation = interface.reset(seed=0)
    action = interface.env.teleop_agent().act(direction=(1, 0))
    result = interface.step(action)
    next_observation = result.obs
```

| `obs` | 返回值 |
| --- | --- |
| `state` | 统一状态向量，三个环境依次为 20 / 12 / 54 维 |
| `keypoint` | 关键点与 agent/末端位置拼接，依次为 20 / 12 / 30 维 |
| `image` | 相机图像与 `observation.agent_pos` 字典；三维包含两个相机 |
| `all` | 全部观测字典 |

`interface.contract()["input_features"]` 给出所选输入字段；向量输入还提供 `input_dim`。原生动作范围和含义由 `action_dim`、`action_low`、`action_high`、`action_semantics` 描述。

模型服务运行在独立环境后，环境侧可连接默认 `127.0.0.1:8765`：

```powershell
python policy_interface.py --env pusht --episodes 5 --max-steps 300
python policy_interface.py --env pushbox3d --obs image --export-contract training_data/pushbox3d_contract.json
```

TCP 协议版本为 1：4 字节网络字节序长度 + UTF-8 JSON，单条消息最多 1 MiB。交互顺序为 `health → reset → predict → step → predict → close`；客户端按模型声明的状态或视觉输入建立历史，视觉历史使用 NHWC uint8 + Base64 无损传输。当前 TCP 视觉协议每个模型声明一个 `image_key`；三维采集仍保存两个相机，本地 Python `obs="image"` 也同时提供两个相机。模型是否使用一台或两台相机，由算法决定。

旧 PushBox2D 四维动作、旧 PushBox3D 三维 XYZ 或六维独立双臂动作模型与新环境不兼容，客户端会在执行动作前拒绝不匹配的接口约定。三维 `state` 保持 54 维，当前模型需使用新的二维动作约定。`Z:\ActionExpert` 的调用继续通过上述三个根入口连接。

## 模型资源

三维运行所需的 ARX L5 XML、网格、`LICENSE` 和来源记录保存在 `pushBox3D/native/assets/arx_l5/`，运行不需要下载。模型来源是 Google DeepMind MuJoCo Menagerie 的 `arx_l5`，来源提交为 `71f066ad0be9cd271f7ed58c030243ef157af9f4`，许可证为 BSD-3-Clause；更详细的原始来源与文件校验值见同目录 `SOURCE.json`。旧 UR5e 资源不再属于当前环境。
"# pushenv" 
