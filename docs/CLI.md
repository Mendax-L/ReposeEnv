# 完整命令参数

2026-10-20 从当前入口实际执行 --help，并提取源码参数声明。README 提供完整流程。

## record.py

工作目录：Z:\pushenv；环境：pushenv。

```powershell
python record.py --help
```

```text
usage: record.py [-h] [--env {pusht,pushbox2d,pushbox3d}] [--output OUTPUT]
                 [--task TASK] [--seed SEED] [--max-steps MAX_STEPS]
                 [--episodes EPISODES] [--render-size RENDER_SIZE]
                 [--speed-mm-s SPEED_MM_S] [--yaw-speed-deg-s YAW_SPEED_DEG_S]
                 [--actions ACTIONS | --validate DATASET | --straight-line]

Record human, supplied or scripted straight-line actions as LeRobot v3.0.

options:
  -h, --help            show this help message and exit
  --env {pusht,pushbox2d,pushbox3d}
  --output OUTPUT
  --task TASK
  --seed SEED
  --max-steps MAX_STEPS
  --episodes EPISODES   Human: 0/unset means unlimited. Straight-line: default
                        100.
  --render-size RENDER_SIZE
                        Recorded camera width; default 512 for PushBox3D, 96
                        for 2-D.
  --speed-mm-s SPEED_MM_S
                        Straight-line shared XY speed in mm/s; default 20.
  --yaw-speed-deg-s YAW_SPEED_DEG_S
                        Straight-line centre yaw speed in degrees/s; default
                        10.
  --actions ACTIONS     Replay one episode of controller .npy/.npz actions
                        instead of human control.
  --validate DATASET    Check an existing LeRobot dataset without starting a
                        simulator.
  --straight-line       Generate scripted PushBox3D straight trajectories;
                        these are not human demonstrations.
```

参数声明（包含 help 未展示的默认值；未声明 default 的普通参数默认 None，store_true 默认 False）：

```python
parser.add_argument("--env", choices=SUPPORTED_ENVIRONMENTS, default=default_env)
parser.add_argument("--output", type=Path)
parser.add_argument("--task")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-steps", type=int)
parser.add_argument("--episodes", type=int, help="Human: 0/unset means unlimited. Straight-line: default 100.")
parser.add_argument("--render-size", type=int, help="Recorded camera width; default 512 for PushBox3D, 96 for 2-D.")
parser.add_argument("--speed-mm-s", type=float, default=20.0, help="Straight-line shared XY speed in mm/s; default 20.")
parser.add_argument("--yaw-speed-deg-s", type=float, default=10.0, help="Straight-line centre yaw speed in degrees/s; default 10.")
source.add_argument("--actions", type=Path, help="Replay one episode of controller .npy/.npz actions instead of human control.")
source.add_argument("--validate", type=Path, metavar="DATASET", help="Check an existing LeRobot dataset without starting a simulator.")
source.add_argument("--straight-line", action="store_true", help="Generate scripted PushBox3D straight trajectories; these are not human demonstrations.")
```

## policy_interface.py

工作目录：Z:\pushenv；环境：pushenv。

```powershell
python policy_interface.py --help
```

```text
usage: policy_interface.py [-h] [--env {pusht,pushbox2d,pushbox3d}]
                           [--host HOST] [--port PORT] [--episodes EPISODES]
                           [--seed SEED] [--max-steps MAX_STEPS]
                           [--connect-timeout CONNECT_TIMEOUT]
                           [--socket-timeout SOCKET_TIMEOUT] [--output OUTPUT]
                           [--export-contract EXPORT_CONTRACT]
                           [--obs {state,keypoint,image,all}]

Policy interaction: local observations/actions and the TCP rollout client.

options:
  -h, --help            show this help message and exit
  --env {pusht,pushbox2d,pushbox3d}
  --host HOST
  --port PORT
  --episodes EPISODES
  --seed SEED
  --max-steps MAX_STEPS
  --connect-timeout CONNECT_TIMEOUT
  --socket-timeout SOCKET_TIMEOUT
  --output OUTPUT
  --export-contract EXPORT_CONTRACT
                        Export feature and action contracts without a policy
                        connection.
  --obs {state,keypoint,image,all}
                        Contract input mode, or an expected TCP input mode
                        (otherwise negotiated with the model).
```

参数声明（包含 help 未展示的默认值；未声明 default 的普通参数默认 None，store_true 默认 False）：

```python
parser.add_argument("--env", choices=SUPPORTED_ENVIRONMENTS, default="pusht")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=100000)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--connect-timeout", type=float, default=30.0)
parser.add_argument("--socket-timeout", type=float, default=120.0)
parser.add_argument("--output", type=Path, default=Path("outputs/policy/metrics.json"))
parser.add_argument("--export-contract", type=Path, help="Export feature and action contracts without a policy connection.")
parser.add_argument("--obs", choices=("state", "keypoint", "image", "all"),
                        help="Contract input mode, or an expected TCP input mode (otherwise negotiated with the model).")
```
