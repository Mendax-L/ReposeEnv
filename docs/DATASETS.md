# 数据清单

当前默认批次为 2026-10-20 重新生成的 `scripted-straight-yaw-cw-waist-camera2-20261020-144758`：腰部夹持，camera2 高度 0.6719 m，100 条全部成功，完整数值字段与 200 个视频的帧数、尺寸、频率均已校验。初始 yaw 顺时针为正，正向 49 条、负向 51 条；最大最终 XY 误差 0.887 mm 以内。

下表同时保留此前批次的元数据记录，历史记录不代表对应目录当前仍在磁盘。旧批次用于追溯，不能混同当前夹持方式、相机配置或动作契约。

| 批次（相对 training_data/lerobot） | 条数 | 帧数 | 动作形状 | 动作语义 |
| --- | ---: | ---: | --- | --- |
| pushbox3d/scripted-straight-yaw-cw-waist-camera2-20261020-144758 | 100 | 34966 | [3] | box_center_xy_yaw_cw_delta_1mm_0.5deg_auto_lift_20mm/v2；腰部夹持、抬高后 camera2 |
| pushbox2d/human-20261015-164705-581008 | 14 | 147 | [2] | attached_pair_absolute_xy |
| pushbox3d/human-20261015-164756-419966 | 2 | 9998 | [3] | attached_pair_xyz_delta_1mm |
| pushbox3d/human-20261019-094803-899869 | 5 | 3507 | [2] | attached_pair_xy_delta_1mm_auto_lift_20mm |
| pushbox3d/scripted-straight-20261019-104056-258472 | 100 | 32787 | [2] | attached_pair_xy_delta_1mm_auto_lift_20mm |
| pushbox3d/scripted-straight-yaw-cw-20261019-161948-257259 | 100 | 35847 | [3] | box_center_xy_yaw_cw_delta_1mm_0.5deg_auto_lift_20mm/v2 |
| pusht | 206 | 25650 | [2] | absolute_agent_xy |
