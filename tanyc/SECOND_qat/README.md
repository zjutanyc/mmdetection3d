# SECOND Quantization-Aware Training (QAT)

This directory mirrors the structure of official `projects` entries so that
custom configs, modules, and scripts stay organized for experimentation with
SECOND model quantization.

Contents (trimmed for KITTI single-class Car detection):
- `configs/`: QAT configs (defaults to `second_hv_secfpn_8xb6-80e_kitti-3d-car.py`).
- `_base_/`: minimal base files only for SECOND on KITTI (model/dataset/schedule/runtime).
- `pretrained/`: contains `hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth`.
- `second_qat/`: python package for custom modules/hooks.
- `test.sh`: distributed eval launcher (defaults to GPUs 4,5,6,7).
- `prepare_kitti.sh`: helper to verify KITTI raw data placement and build info files.

Dataset prep (manual download required):
1) Download KITTI object detection data from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
   and place under `data/kitti` as:
   - `training/{velodyne,image_2,label_2,calib}`
   - `testing/{velodyne,image_2,calib}`
2) Run `bash tanyc/SECOND_qat/prepare_kitti.sh` to generate `kitti_infos_*.pkl` and reduced point clouds.

Evaluation:
- Default: `bash tanyc/SECOND_qat/test.sh`
- Override: `bash tanyc/SECOND_qat/test.sh <CONFIG> <CHECKPOINT> [PY_ARGS]`


---


```
# 浮点数结果
----------- AP11 Results ------------

Car AP11@0.70, 0.70, 0.70:
bbox AP11:95.1688, 89.6431, 87.9990
bev  AP11:89.9659, 87.2761, 84.3051
3d   AP11:88.3587, 78.2174, 76.0258
aos  AP11:94.91, 89.06, 87.21
Car AP11@0.70, 0.50, 0.50:
bbox AP11:95.1688, 89.6431, 87.9990
bev  AP11:95.2968, 89.9348, 88.7258
3d   AP11:95.2452, 89.8569, 88.5259
aos  AP11:94.91, 89.06, 87.21

----------- AP40 Results ------------

Car AP40@0.70, 0.70, 0.70:
bbox AP40:97.4337, 92.4421, 89.2350
bev  AP40:92.6351, 88.4149, 85.2521
3d   AP40:90.4473, 81.3402, 76.1788
aos  AP40:97.16, 91.79, 88.40
Car AP40@0.70, 0.50, 0.50:
bbox AP40:97.4337, 92.4421, 89.2350
bev  AP40:97.5416, 94.7324, 91.7199
3d   AP40:97.4362, 94.5609, 91.4855
aos  AP40:97.16, 91.79, 88.40
```