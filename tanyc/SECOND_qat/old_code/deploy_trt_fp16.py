# TensorRT FP16 deploy config for SECOND (KITTI Car only)

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        max_workspace_size=1 << 30,
        fp16_mode=True,
        int8_mode=False,
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(
                    min_shape=[1, 5, 4],
                    opt_shape=[16000, 5, 4],
                    max_shape=[40000, 5, 4]),
                num_points=dict(
                    min_shape=[1],
                    opt_shape=[16000],
                    max_shape=[40000]),
                coors=dict(
                    min_shape=[1, 4],
                    opt_shape=[16000, 4],
                    max_shape=[40000, 4])
            ))
    ])

codebase_config = dict(type='mmdet3d', task='VoxelDetection')

ir_config = dict(
    backend='tensorrt',
    opset_version=11,
    input_names=['voxels', 'num_points', 'coors'],
    output_names=[
        'cls_score0', 'cls_score1',
        'bbox_pred0', 'bbox_pred1',
        'dir_cls_pred0', 'dir_cls_pred1'
    ],
    dynamic_axes={
        'voxels': {0: 'voxels', 1: 'points_per_voxel'},
        'num_points': {0: 'voxels'},
        'coors': {0: 'voxels'},
    })

onnx_config = dict(
    input_names=ir_config['input_names'],
    output_names=ir_config['output_names'],
    dynamic_axes=ir_config['dynamic_axes'],
    opset_version=ir_config['opset_version'],
    keep_initializers_as_inputs=True,
    optimize=False)
