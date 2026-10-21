# pushenv

pushenv 负责仿真、采集和策略执行，ActionExpert 负责模型。主链路：**环境 → 完整观测和动作 → LeRobot 数据 → 模型训练 → 环境评估**。采集保留全部可用观测，算法在训练时选择输入。

## 项目结构

```text
Z:\pushenv\
├── environment.py          # make_environment，统一环境入口
├── record.py               # 人工、程序、回放采集和校验
├── policy_interface.py     # Python 策略接口、TCP 客户端、契约导出
├── pushT/                  # env.py 公共接口、teleop.py、native/ 仿真
├── pushBox2D/              # 相同分层
├── pushBox3D/              # 相同分层；native/assets/arx_l5/ 模型资源
├── common/
│   ├── environment.py      # 环境包装与 StepResult
│   ├── schema.py           # 字段与单位
│   ├── controllers.py      # 控制器
│   ├── recording.py        # 回放和程序采集
│   ├── teleoperation.py    # 人工窗口和保存生命周期
│   └── lerobot.py          # 数据写入和校验
├── training_data/
│   ├── lerobot/            # pusht、pushbox2d、pushbox3d 批次
│   └── pusht.zip           # 原始数据归档
├── docs/                   # 完整 CLI、接口、数据清单、整理记录
├── environment.yml
├── .gitignore
└── README.md
```

当前没有 training_data/pusht 解压目录。源码和 ARX L5 资源保持当前路径，避免破坏导入及 XML 相对引用。

## 安装与入口

以下命令使用 Windows PowerShell；环境已存在时跳过创建。

```powershell
Set-Location Z:\pushenv
conda env create -f Z:\pushenv\environment.yml
conda activate pushenv
python record.py --help
python policy_interface.py --help
```

本机解释器为 E:\anaconda3\envs\pushenv\python.exe，依赖以 environment.yml 为准。environment.py 提供 Python API，没有独立命令行。模型安装见 [ActionExpert](../ActionExpert/README.md)。

## 当前契约

契约 = 状态字段、动作维数、单位和方向约定；LeRobot v3.0 = Parquet 数值、同步 MP4 和轨迹元数据。

| 环境 | state 维数 | 动作 | 频率 |
| --- | ---: | --- | ---: |
| pusht | 20 | 2 维绝对 XY，[0,512] 像素 | 10 Hz |
| pushbox2d | 12 | 2 维共同抓点中心绝对 XY，[0,512] 像素 | 10 Hz |
| pushbox3d | 54 | 3 维 dx/dy/dyaw，各项 [-1,1] | 50 Hz |

yaw = 从上方看绕箱体中心 Z 轴的转向，顺时针为正。三维每步最大 XY 1 mm、yaw 0.5°，升降自动完成。当前版本是 stacked_box_state_yaw_cw/v2 和 box_center_xy_yaw_cw_delta_1mm_0.5deg_auto_lift_20mm/v2。旧四维 PushBox2D、旧二维 XY PushBox3D 和不同 yaw 方向数据不能直接用于当前环境。

当前三维程序批次为 `training_data/lerobot/pushbox3d/scripted-straight-yaw-cw-waist-camera2-20261020-144758`，100 条、34,966 帧、200 个双相机视频，100 条均成功对齐并落放。该批使用侧壁腰部夹持和抬高后的 camera2（Z = 0.6719 m），保留全部状态、关键点及双相机；`meta/provenance.json` 记录相机配置、源码指纹与校验结果。其他批次的兼容性见 [数据清单](docs/DATASETS.md)，完整字段、成功标准和 Python API 见 [接口参考](docs/INTERFACES.md)。


## 人工采集

从 `Z:\pushenv` 运行所需的一条命令：

```powershell
python record.py --env pusht
python record.py --env pushbox2d
python record.py --env pushbox3d
```

窗口初始暂停。二维按 **空格开始 → 人工操作 → 回车保存**；三维按 **空格开始 → 自动抬升 20 mm → 方向键调整 XY、Q/E 调整 yaw → 对齐后自动降落、落稳并保存**。每条结束后暂停，空格开始下一条。

三维窗口底部显示抬升是否完成、能否操作 XY/yaw；暂停、落放和保存时显示对应的锁定原因。`Lift` 是实际抬高量，目标为 20 mm；`yaw` 显示当前箱体角度，单位为度。

| 操作 | PushT | PushBox2D | PushBox3D |
| --- | --- | --- | --- |
| 移动 | 按住鼠标左键移动，或 WASD | 方向键控制共同抓点中心 XY | 方向键控制世界坐标 XY；上/下为 +Y/−Y |
| yaw 旋转 | — | — | Q：逆时针（负）；E：顺时针（正），从上方看 |
| 升降 | — | — | 自动抬升和降落，无 Z 手控 |
| 空格 | 开始 / 暂停 | 相同 | 开始 / 暂停；收尾降落期间继续执行 |
| 回车 | 保存当前非空片段并重置 | 相同 | 先降落、落稳，再保存并重置 |
| R | 丢弃当前未保存片段并重来 | 相同 | 先降落，再丢弃并重置 |
| Esc / 关闭窗口 | 保存当前非空片段并退出 | 相同 | 先降落、落稳，再保存并退出 |

PushBox2D 和 PushBox3D 只有共同控制，不使用 `1/2` 切换。二维键盘目标每步偏移 6 像素；三维 XY 每步最大 1 mm，yaw 每步最大 0.5°，可以同时平移和旋转，松开按键发送零增量。主动暂停时不推进仿真；失去焦点会暂停人工控制，三维自动升降和收尾仍继续并完整记录。

三维成功要求 **XY 中心距离不超过 5 mm**，即 `sqrt(dx² + dy²) <= 0.005 m`，包含 5 mm 边界；回到起始面后的高度误差小于 0.2 mm、偏航误差小于 0.01 rad、倾斜小于 0.02 rad，并低速落稳 0.5 秒。回车或 Esc 提前结束时，也会完成降落；未达到对齐要求的轨迹标记为截断，不标成功。

抬高量 `lifted_height = 当前 Z - 原始落位 Z`，并非误差；抬升误差 `lift_error = lifted_height - 0.020 m`。原始落位 Z 为 0.200 m，升起目标为 0.220 m。`height_error` 相对当前阶段的目标高度：抬升/XY/yaw 阶段比较 0.220 m，落放/落稳阶段比较 0.200 m。高度检查用于确认自动升降到位，XY 距离与 yaw 角误差共同衡量平面对齐；yaw 成功阈值 `0.01 rad` 约为 `0.573°`。

默认输出到 `Z:\pushenv\training_data\lerobot\<环境名>\human-日期时间\`。指定批次目录、条数或图像分辨率：

```powershell
python record.py --env pushbox2d --output training_data/lerobot/pushbox2d/human-001 --episodes 20 --render-size 256
```

每保存一条就写入视频、Parquet 和元数据，非空输出目录会拒绝覆盖。二维默认上限为 3000 步，即 300 秒；三维为 6000 步，即 120 秒，到达上限后继续记录完成降落所需的步骤。`--max-steps` 修改上限，`--episodes 0` 表示持续采集，条数按实际保存数量计算。

GUI 在落放后使用单独的后台线程计算统计量、编码视频并写入数据，界面持续显示 `SAVING` 和耗时。写入完成后才增加保存条数、重置并开放下一条采集；保存期间不能重置或重复保存，关闭窗口也会等待当前写入完成。高清分辨率和视频质量保持不变。

成功、失败和环境时间限制会自动收尾并保存该条。二维人工提前保存或退出标记 `next.truncated=true`；三维在落稳后按实际结果标记成功或截断。丢弃重来使用相同 seed，保存下一条后 seed 加 1。

## 直线路径生成数据

同一个 Python 入口可生成 PushBox3D 的程序轨迹：**随机位置与 yaw → 自动抬升 20 mm → 中心沿固定 XY 直线移动，同时将 yaw 转回 0 → 自动落放、落稳并判断成功**。

```powershell
python record.py --env pushbox3d --straight-line --episodes 100 --seed 0
```

默认保存到 `Z:\pushenv\training_data\lerobot\pushbox3d\scripted-straight-yaw-cw-日期时间\`；`yaw-cw` 表示此批包含旋转扰动，并使用顺时针为正的 yaw，与 `human-*` 人工示范和旧 `scripted-straight-*` 批次区分。`--output` 可指定新的空目录，`--speed-mm-s` 设置平面速度，默认 20 mm/s；`--yaw-speed-deg-s` 设置旋转速度，默认 10°/s。此模式仅用于 PushBox3D，默认 100 条，seed 依次递增。

直线路径在控制开始时固定箱体中心的 XY 方向和终点，同时以限速 yaw 指令校正角度，不直接写物体位置或姿态。采集内容包含完整原生状态、关键点、双相机 RGB、阶段、箱体目标位姿及三维动作；升降阶段动作是 `[0, 0, 0]`。程序轨迹在 `meta/provenance.json` 标记 `kind="scripted_straight_line"`、`human_demonstration=false`，不会标为人工示范。`meta/collection.json` 逐条保存 seed、初始位置、`initial_yaw_deg`、实际成功标记、最终误差和直线横向偏差；失败轨迹如实保留。原来的 100 条二维动作数据初始 yaw 全为 0，保持原格式作为旧批次，新 yaw 数据单独生成。


## 数据校验与动作回放

```powershell
python record.py --validate training_data/lerobot/pusht
python record.py --validate training_data/lerobot/pushbox3d/scripted-straight-yaw-cw-waist-camera2-20261020-144758
```

回放需要先准备真实控制器动作文件 training_data/pushbox3d_commands.npy，此文件当前未提供。npy 形状为 [帧数,3]；npz 使用 action 键。准备后执行：

```powershell
python record.py --env pushbox3d --actions training_data/pushbox3d_commands.npy --output training_data/lerobot/pushbox3d/replay-run01
```

三维只在控制阶段消耗动作行，自动阶段补零，动作耗尽后先降落。actions、validate、straight-line 三个模式互斥。全部参数、默认值和限制见 [完整命令参考](docs/CLI.md)。

## 模型连接与契约导出

导出当前图像输入契约，不需要模型服务：

```powershell
python policy_interface.py --env pushbox3d --obs image --export-contract outputs/contracts/pushbox3d_image.json
```

TCP = 环境与模型进程之间的网络连接。先在另一终端启动已有 PushT 模型服务：

```powershell
conda run --no-capture-output -n actionexpert python Z:\ActionExpert\inference.py Z:\ActionExpert\outputs\pusht\keypoint_100k\policy.pt --mode serve --env-name pusht --device cuda --host 127.0.0.1 --port 8765
```

出现 MODEL_READY 后，在 pushenv 终端运行：

```powershell
python policy_interface.py --env pusht --host 127.0.0.1 --port 8765 --episodes 5 --seed 100000 --max-steps 300 --output outputs/policy/pusht-run01/metrics.json
```

TCP 输入由模型协商，支持 state/image；Python 接口额外支持 keypoint/all。预览、录像与训练命令见 [ActionExpert README](../ActionExpert/README.md)。旧 camera2_resnet_fm/best.pt 为二维动作模型，不能接入当前三维环境。资源来源与许可证见 [接口参考](docs/INTERFACES.md)，清理范围见 [整理记录](docs/CLEANUP.md)。
