# TRT-NLOS

TRT-NLOS extends [NLOST (CVPR 2023)](https://github.com/Depth2World/NLOST)
with a light denoising network before transient normalization and reconstruction.

## Architecture

![TRT-NLOS pipeline](asset/trtnlospipeline.png)

```text
Transient [B, 1, 512, H, W]
  -> LightDenoisingNet
  -> min-max normalization
  -> transient feature expansion / downsampling
  -> differentiable FK migration
  -> local / global spatio-temporal encoders
  -> cross-attention integration
  -> volume reconstruction
  -> intensity and depth rendering
```

The released configuration uses H=W=128 or 256, an internal spatial size of
128, and one encoder/decoder block. `models/nlost.py` retains the class name
`NLOST` and all parameter names for compatibility with TRT-NLOS checkpoints.
It returns `(volume, intensity, depth, denoised_measurement)` in both modes.
Convert intensity and depth with `(prediction + 1) / 2`, as in the original
training and evaluation scripts. Depth is the dataset's normalized target;
the exported prediction is not automatically converted to physical meters.

## Environment

Use Python 3.9+ and install a matching PyTorch / torchvision build for your
CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Full-resolution training and inference require an NVIDIA CUDA GPU. Exact CUDA
versions and peak memory must be verified on the training server; no GPU memory
claim is made for the added denoiser. Datasets and weights are distributed separately. Download `trt_nlos.pth` from
[the shared model folder](https://pan.quark.cn/s/23463e95c2b3). Original NLOST weights do not contain the denoiser
and are not interchangeable with TRT-NLOS weights.

## Real-data inference

Run commands from this directory. Supply aligned, temporally binned MAT files;
this entry point does not preprocess raw acquisition files.

```bash
python infer.py --dataset fk \
  --data-path /path/to/aligned_fk \
  --checkpoint /path/to/trt_nlos.pth \
  --target-size 256 --output-path outputs/fk

python infer.py --dataset ours \
  --data-path /path/to/aligned_ours \
  --checkpoint /path/to/trt_nlos.pth \
  --target-size 256 --output-path outputs/ours
```

`bash test.sh` accepts the same arguments. Use the target size and coder count
that match the checkpoint. Checkpoints are loaded strictly; both plain and
`module.`-prefixed state dictionaries are accepted.

| Dataset | MAT key and shape | Preserved preprocessing |
| --- | --- | --- |
| FK | `final_meas`, `[256,256,512]` (H,W,T) | 2×2 spatial summation for target size 128 |
| NLOST system (`ours`) | `data`, `[128,128,512]` (H,W,T) | Swap H/W; nearest-neighbor upsampling for target size 256 |

The default `--bin-len 0.0096` is light travel distance per input bin in meters
(corresponding to 32 ps with c=3e8 m/s), not a time in seconds. It is doubled
inside the model after temporal downsampling. The default wall size is 2 m.
Outputs include intensity/depth MAT arrays and clipped PNG previews; intensity
previews are additionally normalized by their maximum. MAT arrays preserve the
unclipped predictions.

## Synthetic evaluation

```bash
python validate_syn.py \
  --syn_data_path /path/to/bike --target_size 256 \
  --pretrained_model /path/to/trt_nlos.pth --output_path outputs/bike

python validate_syn.py --unseen \
  --syn_data_path /path/to/unseen --target_size 256 \
  --pretrained_model /path/to/trt_nlos.pth --output_path outputs/unseen
```

Synthetic evaluation uses `bin_len=0.01`. It retains the original foreground
bounding-box crop with 40-pixel padding and reports intensity PSNR/SSIM and
depth MAD/RMSE. Intensity and depth crops are calculated from their respective
targets; these are not full-frame scores. The synthetic scripts use one coder.

The bike loader expects:

```text
bike/0/<object>/shin*/
  video-confocalspad*.mp4
  confocal-0-*.hdr
  depth-0-*.hdr
```

Object directories are reverse-lexicographically sorted: the first 250 are
training, and the remainder validation (the original split convention).
The unseen loader reads all objects with `video-confocal-gray-full*.mp4`,
`confocal-0*.png` and `depth-0*.png`. The original PNG depth clipping to [0,1]
is retained; confirm the actual depth encoding before interpreting these scores.
No extra synthetic noise is introduced by these loaders.

## Training and denoiser supervision

**The downloaded loader did not provide `raw_meas`, although the training loop
requires it. The original clean-target source and noise-generation procedure
still need confirmation. Training reproduction is therefore not yet verified.**

A paired-file interface is provided: `--clean-measurement-pattern` must identify
one separate clean MP4 in the same view directory as each noisy MP4. Clean and
noisy measurements undergo the same preprocessing and must have matching shapes.
The loader rejects a missing/ambiguous match or the noisy file itself. This is
an explicit interface for paired data, not a claim about the original dataset's
clean filenames. If the original setup generates noise online, restore that
specific procedure instead of using an arbitrary noise model.

After confirming the paired data layout:

```bash
bash train.sh --data-dir /path/to/bike \
  --clean-measurement-pattern 'ACTUAL_CLEAN_FILENAME.mp4' \
  --model-dir outputs/trt-nlos --target-size 256

NPROC_PER_NODE=2 bash train.sh --data-dir /path/to/bike \
  --clean-measurement-pattern 'ACTUAL_CLEAN_FILENAME.mp4' \
  --model-dir outputs/trt-nlos --target-size 256 --batch-size 1
```

`--batch-size` is per GPU. The preserved objective is
`10 * RMSE(denoised, clean) + RMSE(intensity, target) + RMSE(depth, target)`.
Despite the legacy function name `criterion_L2`, each term is root MSE.
Validation uses the same scaling and weights. `--resume /path/to/checkpoint.pth`
requires an original training checkpoint with optimizer state and restores model/optimizer state and continues at the next epoch; it does not
restore an exact mid-epoch sampler position. TensorBoard logs, config and
checkpoints are written to `--model-dir`.

## Source layout

- `models/`: denoiser, active NLOST blocks, rendering and FK migration.
- `utils/`: datasets, checkpoint loading/saving, arguments and distributed setup.
- `pro/`: training, validation and the reconstruction loss.
- `train.py`, `infer.py`, `validate_syn.py`: the three entry points.
- `metric.py`: original evaluation metrics and crop convention.

Unused ablation architectures, alternate physics operators, traditional-baseline
scripts, repeated evaluation scripts, profiling and temporary files are excluded.
See `CLEANUP_NOTES.md` for compatibility and verification notes.

## Acknowledgements

This code builds on [NLOST](https://github.com/Depth2World/NLOST), including its
LFE/PTR-derived components. Dataset links and the prior-work citation are available
in that repository. Please cite the TRT paper as well when its final citation is
added to the root README.
