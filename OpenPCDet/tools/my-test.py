import os
import datetime
from pathlib import Path
import argparse

import numpy as np

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils 

from eval_utils import eval_utils

cfg_file = '/home/012392471@SJSUAD/master_project/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml'

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_file, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


args, cfg = parse_config()

if args.infer_time:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if args.launcher == 'none':
    dist_test = False
    total_gpus = 1
else:
    if args.local_rank is None:
        args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
    args.tcp_port, args.local_rank, backend='nccl'
    )
    dist_test = True

if args.batch_size is None:
    args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
else:
    assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
    args.batch_size = args.batch_size // total_gpus

# Load configuration
# cfg_file = '/home/012392471@SJSUAD/master_project/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml'
cfg_from_yaml_file(cfg_file, cfg)

# Build model
# os.mkdir("log_outputs")
log_file = os.path.join("log_outputs", ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=4,
    dist=False, 
    # workers=args.workers, 
    logger=logger,
    training=False
)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

# Load pre-trained model
# ckpt_path = 'pp_multihead_nds5823_updated.pth'
ckpt_path = 'cbgs_pp_centerpoint_nds6070.pth'
model.load_params_from_file(filename=ckpt_path, logger=logger)
# model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')))
model.cuda()
model.eval()

args = None
epoch_id = 0
eval_utils.eval_one_epoch(
    cfg, args, model, test_loader, epoch_id, logger, dist_test=False,
    result_dir=Path("eval_output_results")
)