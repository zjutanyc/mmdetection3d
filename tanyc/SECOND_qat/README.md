# SECOND Quantization (SparseEncoder only)

用于 SECOND 稀疏编码器（traveller59/spconv）量化的最小化工具集：校准生成 PTQ 权重、评测 FP/PTQ。

## 核心文件与目录
```
tanyc/SECOND_qat/
├─ configs/                         # 默认配置（KITTI 单类）
├─ CUDA-SECOND/
│  ├─ sparseencoder_quantization.py # 量化工具，仅 SparseEncoder
│  ├─ second_sparse_encoder_qat.py  # 校准/生成 PTQ 权重
│  └─ second_ptq_eval.py            # 使用 PTQ 权重评测
├─ fp_eval.sh                       # 单卡浮点评测
├─ ptq_eval.sh                      # 单卡 PTQ 评测
└─ README.md
```

预训练权重下载地址：https://github.com/open-mmlab/mmdetection3d/tree/main/configs/second

## 典型流程
1) 校准（PTQ 起点）：
   ```bash
   python tanyc/SECOND_qat/CUDA-SECOND/second_sparse_encoder_qat.py \
     --config tanyc/SECOND_qat/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
     --checkpoint work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth \
     --work-dir work_dirs/SECOND_qat \
     --calib-batches 600
   ```
   生成 `work_dirs/SECOND_qat/sparse_encoder_ptq_calib600.pth`。
2) 评测 PTQ：
   ```bash
   bash tanyc/SECOND_qat/ptq_eval.sh \
     tanyc/SECOND_qat/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
     work_dirs/SECOND_qat/sparse_encoder_ptq_calib600.pth
   ```
3) 浮点评测：
   ```bash
   bash tanyc/SECOND_qat/fp_eval.sh \
     tanyc/SECOND_qat/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
     work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth
   ```


## 参考结果
```bash
# 全浮点
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

```bash
# 稀疏卷积量化
----------- AP11 Results ------------

Car AP11@0.70, 0.70, 0.70:
bbox AP11:93.4153, 88.4264, 85.3411
bev  AP11:89.5396, 85.5942, 81.7580
3d   AP11:87.2914, 77.1879, 73.7662
aos  AP11:93.24, 87.79, 84.51
Car AP11@0.70, 0.50, 0.50:
bbox AP11:93.4153, 88.4264, 85.3411
bev  AP11:93.5213, 88.9610, 86.6461
3d   AP11:93.4859, 88.8427, 86.2926
aos  AP11:93.24, 87.79, 84.51

----------- AP40 Results ------------

Car AP40@0.70, 0.70, 0.70:
bbox AP40:95.8594, 90.7850, 86.7078
bev  AP40:91.8231, 86.8896, 83.1111
3d   AP40:89.0894, 79.4300, 74.0808
aos  AP40:95.68, 90.07, 85.81
Car AP40@0.70, 0.50, 0.50:
bbox AP40:95.8594, 90.7850, 86.7078
bev  AP40:95.9794, 92.7698, 89.0764
3d   AP40:95.8585, 92.5440, 88.7045
aos  AP40:95.68, 90.07, 85.81
```