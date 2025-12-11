# TensorRT INT8 deploy config for SECOND (KITTI Car only)
# Note: mmdeploy pip 包未自带 mmdet3d 模板，此文件提供最小配置，
#       需要搭配 mmdeploy 的 deploy 工具/脚本使用。

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        max_workspace_size=1 << 30,
        fp16_mode=False,
        int8_mode=True,
        int8_param=dict(algorithm='entropy')  # 可改为 'minmax'/'percentile'
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(
                    # actual input shape is [num_voxels, points_per_voxel, 4]
                    min_shape=[1, 5, 4],
                    opt_shape=[16000, 5, 4],
                    max_shape=[40000, 5, 4]),
                num_points=dict(
                    # shape [num_voxels]
                    min_shape=[1],
                    opt_shape=[16000],
                    max_shape=[40000]),
                coors=dict(
                    # shape [num_voxels, 4]
                    min_shape=[1, 4],
                    opt_shape=[16000, 4],
                    max_shape=[40000, 4])
            ))
    ])

codebase_config = dict(type='mmdet3d', task='VoxelDetection')

# ONNX/IR export settings
ir_config = dict(
    backend='tensorrt',
    opset_version=11,
    input_names=['voxels', 'num_points', 'coors'],
    # Flattened outputs per level to match ONNX export wrapper.
    output_names=[
        'cls_score0', 'cls_score1',
        'bbox_pred0', 'bbox_pred1',
        'dir_cls_pred0', 'dir_cls_pred1'
    ],
    dynamic_axes={
        'voxels': {0: 'voxels', 1: 'points_per_voxel'},
        'num_points': {0: 'voxels'},
        'coors': {0: 'voxels'},
        # outputs keep default static axes by backend
    })

onnx_config = dict(
    input_names=ir_config['input_names'],
    output_names=ir_config['output_names'],
    dynamic_axes=ir_config['dynamic_axes'],
    opset_version=ir_config['opset_version'],
    keep_initializers_as_inputs=True,
    optimize=False)

# 校准数据配置（需要先生成 calib_file，供 TensorRT 构建 INT8 引擎使用）
calib_config = dict(
    create_calib=True,
    calib_file='calib_kitti_car.h5',
    algo_type='entropy',
    seed=0,
    subset=500,  # 校准样本数量，可按需调整
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne_reduced'),
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        metainfo=dict(classes=['Car']),
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
                 transforms=[
                     dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1., 1.], translation_std=[0, 0, 0]),
                     dict(type='RandomFlip3D'),
                     dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1])]),
            dict(type='Pack3DDetInputs', keys=['points'])
        ])
)
