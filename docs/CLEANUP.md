# 整理记录（2026-10-20）

保留现有源码分层，重写 README；详细参数放 docs/CLI.md，领域说明与历史实验分别单独归档。
删除范围仅为 __pycache__ 下 Python 字节码缓存，并忽略后续生成的缓存。
pushenv 共清理 4 个缓存目录；沿用已有 .gitignore，不删除任何数据批次。
数据集、原始 pusht.zip、模型权重、日志、录像、评估、ARX L5 资源及许可证全部保留。
未改动历史 JSON 内路径与结果，未重新训练，未更改仿真或算法行为。
当前三维动作由旧二维 XY 升级为三维 XY/yaw，README 已移除直接使用旧三维任务模型的操作路径。

验证：record.py 与 policy_interface.py 的 --help 成功；README 与 CLI 中 PowerShell 块语法通过；文档链接存在。record.py --validate 校验通过：PushT 为 206 条、25,650 帧、3 个视频；当前三维 yaw-cw 批次为 100 条、35,847 帧、200 个视频。校验包括字段、轨迹边界、视频帧数、尺寸与 FPS；本次未重新采集数据。
