_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]


# Dataset settings
dataset_type = 'WaymoDataset'
data_root = '/home/012392471@SJSUAD/master_project/mmdetection3d/data/waymo/kitti_format'
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
voxel_size = [0.32, 0.32, 6]

model = dict(
    type='MVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(32000, 32000))),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[point_cloud_range[0], point_cloud_range[1], -0.0345,
                    point_cloud_range[3], point_cloud_range[4], -0.0345],
                   [point_cloud_range[0], point_cloud_range[1], 0,
                    point_cloud_range[3], point_cloud_range[4], 0],
                   [point_cloud_range[0], point_cloud_range[1], -0.1188,
                    point_cloud_range[3], point_cloud_range[4], -0.1188]],
            sizes=[[4.73, 2.08, 1.77], [0.91, 0.84, 1.74], [1.81, 0.84, 1.77]],
            rotations=[0, 1.57],
            reshape_out=False)))

# Data pipeline
train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=6,
         use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample',
         db_sampler=dict(
             data_root=data_root,
             info_path=data_root + 'waymo_dbinfos_train.pkl',
             rate=1.0,
             prepare=dict(
                 filter_by_difficulty=[-1],
                 filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
             classes=['Car', 'Pedestrian', 'Cyclist'],
             sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10))),
    dict(type='ObjectNoise',
         num_try=100,
         translation_std=[0.25, 0.25, 0.25],
         global_rot_range=[0.0, 0.0],
         rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.78539816, 0.78539816],
         scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=6,
         use_dim=5),
    dict(type='Pack3DDetInputs',
         keys=['points'],
         meta_keys=['sample_idx', 'context_name', 'timestamp'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        data_prefix=dict(
            pts='training/velodyne',
            sweeps='training/velodyne'),
        pipeline=train_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=False))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_val.pkl',
        data_prefix=dict(
            pts='training/velodyne',
            sweeps='training/velodyne'),
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_test.pkl',
        data_prefix=dict(
            pts='testing/velodyne',
            sweeps='testing/velodyne'),
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True))

val_evaluator = dict(
    type='WaymoMetric',
    ann_file=data_root + 'waymo_infos_val.pkl',
    waymo_bin_file='waymo/waymo_format/gt.bin',
    data_root=data_root,
    convert_kitti_format=False)

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')