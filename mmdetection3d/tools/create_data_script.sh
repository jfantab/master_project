python tools/create_data.py waymo \
    --root-path ./data/waymo \
    --version v1.4 \
    --out-dir ./data/waymo \
    --workers 64 \
    --extra-tag waymo \
    --skip-cam_instances-infos


: '
Saving validation dataset infos into ./data/waymo/kitti_format/waymo_infos_val.pkl
Traceback (most recent call last):
  File "/home/012392471@SJSUAD/master_project/mmdetection3d/tools/create_data.py", line 376, in <module>
    waymo_data_prep(
  File "/home/012392471@SJSUAD/master_project/mmdetection3d/tools/create_data.py", line 244, in waymo_data_prep
    create_ImageSets_img_ids(out_dir, splits)
  File "/home/012392471@SJSUAD/master_project/mmdetection3d/tools/dataset_converters/waymo_converter.py", line 717, in create_ImageSets_img_ids
    open(save_dir + 'val.txt', 'w').writelines(idx_all[1])
IndexError: list index out of range
'