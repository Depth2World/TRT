"""Reconstruct aligned FK or NLOST-system MAT measurements."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['fk', 'ours'], required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, default=Path('outputs/real'))
    parser.add_argument('--target-size', type=int, choices=[128, 256], default=256)
    parser.add_argument('--num-coders', type=int, default=1)
    parser.add_argument('--bin-len', type=float, default=0.0096,
                        help='Light travel distance per input time bin, in meters.')
    return parser.parse_args()


def load_measurement(path, dataset, target_size):
    import numpy as np
    import scipy.io as sio
    import torch
    import torch.nn.functional as F
    data = sio.loadmat(path)
    if dataset == 'fk':
        x = np.asarray(data['final_meas'], dtype=np.float32)
        if x.shape != (256, 256, 512):
            raise ValueError(f'{path}: expected final_meas [256,256,512], got {x.shape}')
        if target_size == 128:
            x = x[::2] + x[1::2]
            x = x[:, ::2] + x[:, 1::2]
    else:
        x = np.asarray(data['data'], dtype=np.float32)
        if x.shape != (128, 128, 512):
            raise ValueError(f'{path}: expected data [128,128,512], got {x.shape}')
        x = x.transpose(1, 0, 2)
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
    if dataset == 'ours' and target_size == 256:
        x = F.interpolate(x, size=(256, 256), mode='nearest')
    return x[None]


def main(args):
    import cv2
    import numpy as np
    import scipy.io as sio
    import torch
    from models.nlost import NLOST
    from utils.checkpoint import load_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError('Full-resolution TRT-NLOS inference requires a CUDA GPU.')
    files = sorted(args.data_path.glob('*.mat'))
    if not files:
        raise FileNotFoundError(f'No MAT measurements in {args.data_path}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model = NLOST(num_coders=args.num_coders, spatial=128, tlen=256,
                  bin_len=args.bin_len, target_size=args.target_size).cuda().eval()
    load_checkpoint(model, args.checkpoint)
    args.output_path.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for path in files:
            measurement = load_measurement(path, args.dataset, args.target_size).cuda()
            _, intensity, depth, _ = model(measurement)
            intensity = ((intensity + 1) / 2)[0, 0].cpu().numpy()
            depth = ((depth + 1) / 2)[0, 0].cpu().numpy()
            sio.savemat(args.output_path / f'{path.stem}.mat',
                        {'intensity': intensity, 'depth': depth})
            preview = intensity / max(float(intensity.max()), 1e-8)
            for suffix, value in [('int', preview), ('dep', depth)]:
                output = args.output_path / f'{path.stem}_{suffix}.png'
                if not cv2.imwrite(str(output), (np.clip(value, 0, 1) * 255).astype(np.uint8)):
                    raise OSError(f'Could not write {output}')
            print(f'Saved {path.stem}')


if __name__ == '__main__':
    main(parse_args())
