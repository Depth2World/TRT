"""Training arguments. Legacy spellings are retained as aliases."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Train TRT-NLOS with paired transient supervision.')
    parser.add_argument('--data-dir', '--data_dir', dest='data_dir', required=True)
    parser.add_argument('--model-dir', '--model_dir', dest='model_dir', default='outputs/trt-nlos')
    parser.add_argument('--clean-measurement-pattern', required=True,
                        help='Glob for the clean MP4 in each scene/view directory; must match exactly one file.')
    parser.add_argument('--batch-size', '--bacth_size', dest='bacth_size', type=int, default=1)
    parser.add_argument('--target-size', '--target_size', dest='target_size', type=int, choices=[128, 256], default=256)
    parser.add_argument('--num-epoch', '--num_epoch', dest='num_epoch', type=int, default=50)
    parser.add_argument('--num-save', '--num_save', dest='num_save', type=int, default=600)
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int, default=8)
    parser.add_argument('--num-coders', '--num_coders', dest='num_coders', type=int, default=1)
    parser.add_argument('--lr-rate', '--lr_rate', dest='lr_rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', '--weit_decay', dest='weit_decay', type=float, default=1e-4)
    parser.add_argument('--opter', choices=['adamw', 'adam'], default='adamw')
    parser.add_argument('--resume', dest='resmod_dir', default='')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dist-url', dest='dist_url', default='env://')
    args = parser.parse_args()
    if min(args.bacth_size, args.num_epoch, args.num_save, args.num_coders) < 1:
        parser.error('batch size, epochs, save interval and coder count must be positive')
    args.model_name = 'trt-nlos'
    args.resmue = bool(args.resmod_dir)
    args.loc_rank = 0
    args.world_size = 1
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    return args
