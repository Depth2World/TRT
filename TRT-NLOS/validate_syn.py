import os
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from utils.LFEDataset import LFEDataset, LFEDataset_unseen
from metric import SSIM, MAD, crop_to_cal, cal_psnr

cudnn.benchmark = True
from models import nlost
import logging

def main(args):

    model = nlost.NLOST(ch_in=1, num_coders=1,spatial=128,tlen=256,bin_len=0.01,target_size=args.target_size)
    model.cuda()
    from utils.checkpoint import load_checkpoint
    load_checkpoint(model, args.pretrained_model)

    print("Start eval...")
    folder_path = [args.syn_data_path]
    shineness = [0]
    dataset_cls = LFEDataset_unseen if args.unseen else LFEDataset
    val_data = dataset_cls(root=folder_path,  # dataset root directory
                          shineness=shineness,
                          for_train=False,
                          ds=1,  # temporal down-sampling factor
                          clip=512,  # time range of histograms
                          size=256,  # measurement size (unit: px)
                          scale=1,  # scaling factor (float or float tuple)
                          background=[0.05, 2],  # background noise rate (float or float tuple)
                          target_size=args.target_size,  # target image size (unit: px)
                          target_noise=0.01,  # standard deviation of target image noise
                          color='gray')  # color channel(s) of target image

    if not len(val_data):
        raise ValueError('No evaluation samples found; check dataset layout and split.')
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    out_path = args.output_path + '/syn/'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler(args.output_path + '/crop_syn_evluation.log' ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )

    niter = 1
    l_psnr = []
    l_ssim = []
    l_mad = []
    l_rmse = []

    cal_ssim = SSIM().cuda()
    cal_mad = MAD().cuda()
    mse = nn.MSELoss()
    with torch.no_grad():
        model.eval()
        for sample in tqdm(val_loader):
            M_wnoise = sample["ds_meas"].cuda()
            dep_gt = sample["dep_gt"].cuda()
            img_gt = sample["img_gt"].cuda()
            mask = dep_gt > 0
            mask = mask.float()
            _,pred,depth_pred,_ = model(M_wnoise)

            pred = (pred + 1) / 2
            depth_pred = (depth_pred + 1) / 2


            box_gt, box_pred = crop_to_cal(img_gt,pred)
            box_dep_gt, box_dep_pred = crop_to_cal(dep_gt,depth_pred)

            box_gt, box_pred = box_gt.cuda(), box_pred.cuda()
            box_dep_gt, box_dep_pred = box_dep_gt.cuda(), box_dep_pred.cuda()

            im_psnr = cal_psnr(box_pred, box_gt)
            l_psnr.append(im_psnr.item())
            im_ssim = cal_ssim(box_pred, box_gt)
            l_ssim.append(im_ssim.item())
            dep_mad = cal_mad(box_dep_pred, box_dep_gt)
            l_mad.append(dep_mad.item())
            dep_rmse = torch.sqrt(mse(box_dep_pred, box_dep_gt))
            l_rmse.append(dep_rmse.item())


            front_view = pred[0].cpu().numpy().transpose(1,2,0)
            depth_view = depth_pred[0].cpu().numpy().transpose(1,2,0)
            view_gt = img_gt[0].cpu().numpy().transpose(1,2,0)
            cv2.imwrite(out_path + f'{niter}_int.png', (np.clip(front_view/max(float(np.max(front_view)), 1e-8), 0, 1)*255).astype(np.uint8))
            cv2.imwrite(out_path + f'{niter}_gt.png', (np.clip(view_gt/max(float(np.max(view_gt)), 1e-8), 0, 1)*255).astype(np.uint8))
            cv2.imwrite(out_path + f'{niter}_dep.png', (np.clip(depth_view, 0, 1)*255).astype(np.uint8))
            niter += 1

        logging.info("img_ssim: %f" % (float(sum(l_ssim))/float(len(l_ssim))))
        logging.info("dep_mad: %f" % (float(sum(l_mad))/float(len(l_mad))))
        logging.info("img_psnr: %f" % (float(sum(l_psnr))/float(len(l_psnr))))
        logging.info("dep_rmse: %f" % (float(sum(l_rmse))/float(len(l_rmse))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_data_path", type=str, required=True,help="Path to the syn dataset.")
    parser.add_argument("--target_size", type=int, choices=[128, 256], default=256,help="The spatial resolution of the input transient, i.e., 256 or 128")
    parser.add_argument("--output_path", type=str, required=True,help="Path to output.")
    parser.add_argument("--pretrained_model", type=str, required=True,help="Prtrained Model Path.")
    parser.add_argument('--unseen', action='store_true', help='Evaluate the unseen-object dataset.')
    args = parser.parse_args()

    return args

def test():
    args = parse_args()
    main(args)

if __name__=="__main__":
    test()

