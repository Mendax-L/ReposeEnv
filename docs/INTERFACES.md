# 环境、数据与接口参考

这是字段、坐标和生命周期参考。以下命令均在激活 pushenv 后、从 `Z:\pushenv` 执行；回放示例中的动作数组需由控制器预先提供。日常安装、采集与模型连接见 [README](../README.md)，全部参数见 [CLI](CLI.md)。

## 当前三个任务

| 环境名 | 当前控制方式 | state 维度 | action 维度 | 采样频率 |
| --- | --- | ---: | ---: | ---: |
| `pusht` | 单个圆形 agent 推动 T 块；绝对目标 XY | 20 | 2 | 10 Hz |
| `pushbox2d` | 左右两个抓点固定在盒边，控制两者中点的绝对目标 XY | 12 | 2 | 10 Hz |
| `pushbox3d` | ARX L5 双臂固定抓住上盒，执行箱体中心 XY 平移与 yaw 旋转；升降自动完成 | 54 | 3 | 50 Hz |

PushT 和 PushBox2D 的动作范围是 `[0, 512]`，单位为原生 512×512 工作空间像素。PushBox2D 的两抓点同时平移，初始角度为 0；它们不再接受独立的四维动作。

PushBox3D 的动作是 `[dx, dy, dyaw]`，每项限制为 `[-1, 1]`。XY 每步最大移动 `0.001 m`，yaw 每步最大旋转 `0.5°`（`0.008726646 rad`）。**yaw = 绕箱体中心竖直 Z 轴的平面转向，从上方看顺时针为正、逆时针为负**；动作、角度观测、目标角度和界面都使用此约定。只发送 yaw 时中心目标 XY 不动。两只抓具的位置和末端朝向一起绕箱体中心转动，抓点间距保持不变，工作空间和双臂逆运动学可达性限制仍有效。环境先自动抬升 **20 mm**，开放 XY/yaw 控制，对齐后自动降回起始面并验证落稳；Z 不属于手动动作。

三维盒体外尺寸为 `0.24×0.30×0.12 m`，壁厚 `0.008 m`。默认以 `X∈[-40,40] mm、Y∈[-30,30] mm` 为均匀采样范围，剔除距目标不足 20 mm 的点；初始 yaw 的绝对值为 **3°–8°**，正负均可，避免生成几乎看不出旋转的初态。位置和角度组合必须同时满足两臂原有工作空间，不可达组合会重新采样，不会直接裁剪或扩大机器人运动范围。初始 Z 为 0.200 m，控制目标 yaw 限于 ±12°。同一个 `reset(seed)` 复现相同位置和角度；保存下一条时 seed 加 1，丢弃重试保持同 seed。固定位置和角度可用 `env.reset(0, options={"offset": [0.035, -0.020], "yaw": 0.0})`，其中 yaw 单位是弧度；`options={"randomize": False}` 使用原固定位置和零角度。PushT、PushBox2D 沿用各自已有的 seed 随机初态。

两只夹爪的夹持面中心固定在上箱左右侧壁的腰部中点，箱体局部坐标为 `(-0.116, 0, 0.060) m` 和 `(0.116, 0, 0.060) m`，即距箱底 6 cm、前后居中；原始指垫、碰撞和机械臂网格均保留。camera2 位于 `(0, -1.5, 0.6719) m`，向下看约 16.8°，优先露出上下箱体的前侧壁、相邻边缘及位置和 yaw 偏差；手腕位于画面上方，夹持处可能有局部遮挡。总览相机保持原视角。新采集数据的元数据记录实际夹持位置。

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
| `observation.box_target_pose` | — | — | 4：箱体中心目标 X/Y/Z（m）及 yaw（rad） |
| `observation.yaw` | — | — | 1：当前箱体 yaw（rad），顺时针正、逆时针负 |
| 其他观测 | 原生位姿、速度、目标、接触计数 | 相同 | 原生状态、盒体速度、目标位姿、关节位置/速度、末端目标、接触力与接触计数 |
| `next.*` | 奖励、done、terminated、truncated、success、success_known | 相同 | 相同 |

三维两个相机始终一起采集。`--render-size` 修改采集宽度，总览为正方形，近景高度为 `max(2, 2×round(宽度×0.3))`；例如宽度 256 时近景为 154×256。窗口始终使用原生宽度 512 的双相机预览，保持近景比例。RGB 均为 uint8，视频不含界面、鼠标光标或动作标记，采用 H.264 有损编码；数值状态和动作存入 Parquet。

抬升、降落和落稳阶段也逐帧记录，`action` 为三维零向量，`observation.phase` 描述动作前阶段。三维动作字段按 `box.delta.x`、`box.delta.y`、`box.delta.yaw` 排列，缩放与旋转中心写在元数据中。高清 RGB 使用临时磁盘缓冲，每条保存或丢弃后清理，避免整条双相机图像堆积在内存。

三维 54 维状态顺序是：盒体位置/四元数 7、盒体线/角速度 6、双臂关节位置 12、双臂关节速度 12、双末端位置 6、双末端目标 6、对齐误差 3、接触力 2。`observation.state[51]` 与单独的 `observation.yaw` 都是顺时针为正的角度；MuJoCo 原生四元数和角速度保留标准右手坐标，不能直接将四元数解出的 Z 角当作本项目 yaw，其符号相反。`inner/outer` 和二维 `external_agent/internal_agent` 是保留的字段名，完整字段名称、单位和 yaw 约定保存在每批元数据中。

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

动作文件由控制器提供：`.npy` 为 `[帧数, 动作维度]` 数组，`.npz` 使用 `action` 键。三维仅在 XY 控制阶段消耗动作行，自动阶段补零并完整记录；动作耗尽后先降落再结束。现有 PushT 压缩包与已转换的 206 条轨迹、25,650 帧保留；当前没有单独解压的 training_data/pusht 目录，新数据默认写入独立批次。

## 环境与策略接口

三个环境共同支持 `reset(seed)`、`step(action)`、`observe(include_images=True)`、`get_keypoints()`、`get_agent_positions()`、`teleop_agent()`、`render()`、`close()`、`contract()`。`step()` 返回 `StepResult`，包含 `obs`、`reward`、`done`、`info`、`terminated`、`truncated`。

三维 `reset()` 从抬升阶段开始，持续调用 `step()` 推进整个周期；`result.info["phase"]` 为 `control` 时才执行 XY/yaw 指令。提前结束可调用 `env.request_finish()`，随后继续发送 `[0, 0, 0]` 直到 `done`，让环境完成降落。

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
    action = interface.env.teleop_agent().act(direction=(1, 0), yaw=0.5)
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

旧 PushBox2D 四维动作、旧 PushBox3D 二维 XY、三维 XYZ、逆时针为正的 XY/yaw 或六维独立双臂动作模型与新环境不兼容。当前三维动作是顺时针为正的 XY/yaw；TCP 会核对更新后的 `action_semantics` 和 `observation_layout`，防止同为三维但方向约定不同的模型接入。三维 `state` 保持 54 维；已有数据以各批 schema 为准。`Z:\ActionExpert` 的调用继续通过上述三个根入口连接。

## 模型资源

三维运行所需的 ARX L5 XML、网格、`LICENSE` 和来源记录保存在 `pushBox3D/native/assets/arx_l5/`，运行不需要下载。模型来源是 Google DeepMind MuJoCo Menagerie 的 `arx_l5`，来源提交为 `71f066ad0be9cd271f7ed58c030243ef157af9f4`，许可证为 BSD-3-Clause；更详细的原始来源与文件校验值见同目录 `SOURCE.json`。旧 UR5e 资源不再属于当前环境。
