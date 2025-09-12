import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import scipy.io as scio
import scipy
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score as ACC
from pathlib import Path
 
from metric import RMSE, PSNR, SSIM ,AverageMeter, crop_to_cal,MAD,cal_psnr
# from tools import*
import torch.nn.functional as F
from models.network import dy_nlos
from utils import dynamic_dataset
from pro.Loss import criterion_KL, criterion_L2
from matplotlib import pyplot as plt
import cv2

def main():
    
    model = dy_nlos(in_ch=1,out_ch=1,spatial=256,tlen=1024,frames=3,local_rank=-1)
    model_root = 'xxxx'
    out_path = model_root + '/middle2014/'
    data_path = '/data1/yueli/dataset/sp_syn/Middlebury2014_256/'
    sbrs = os.listdir(data_path)
    pathsbr = [Path(data_path) / x for x in sbrs if os.path.isdir(Path(data_path) / x)]

    all_model_file = []
    model_files = os.listdir(model_root)
    for fi in model_files:
        if ('pth' in fi) and ('END' not in fi):
                fi_d = os.path.join(model_root, fi)
                all_model_file.append(fi_d)
    
    all_model_file.sort() 
    model.cuda()
    model = torch.nn.DataParallel(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Numbers of parameters are: {}".format(num_params))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(os.path.join(out_path,'dep'), exist_ok=True)
    
    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler(out_path + '/middlesyn_all_evluation.log' ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )
    
    cal_ssim = SSIM().cuda()
    cal_rmse = RMSE().cuda()
    total_time = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        model.eval()
        for i in range(len(all_model_file)):
            logging.info(all_model_file[i])
            checkpoint = torch.load(all_model_file[i], map_location="cpu")
            # checkpoint = torch.load(all_model_file[i])
            model_dict = model.state_dict()
            ckpt_dict = checkpoint['state_dict']
            model_dict.update(ckpt_dict)
            model.load_state_dict(model_dict)
            
            ppp2 = []
            ppp10 = []
            ppp50 = []
            ppp100 = []     
            for j in range(len(pathsbr)):
                os.makedirs(os.path.join(out_path+'/dep/',sbrs[j]), exist_ok=True)
                seqs = os.listdir(pathsbr[j])
                seqss = [Path(pathsbr[j]) / x for x in seqs]
                l_ssim = []
                l_rmse = []
                dep_mat = np.zeros([256,256,2,len(seqss)])
                for k in range(len(seqss)):
                    all_data = scio.loadmat(seqss[k])
                    deps = np.asarray(all_data['depth']).astype(np.float32)    # 72 88  
                    h,w = deps.shape
                    M_mea = scipy.sparse.csc_matrix.todense(all_data['spad'])
                    M_mea = np.ascontiguousarray(M_mea).astype(np.float32).reshape([1, w, h, 1024])
                    M_mea = torch.from_numpy(np.transpose(M_mea,[0,3,2,1])[None]).cuda()

                    if 'Right' in str(seqss[k]):
                        M_mea = M_mea[:,:,:,:,w-256:]
                        deps = deps[:,w-256:]
                    elif 'Left' in str(seqss[k]):
                        M_mea = M_mea[:,:,:,:,:256]
                        deps = deps[:,:256]
                    else:
                        M_mea = M_mea[:,:,:,:,75:75+256]
                        deps = deps[:,75:75+256]

                    model.module.evluation_rw(M_mea)
                    M_mea_re, dep_re = model.module.vlo, model.module.rendered_depth

                    C = 3e8
                    Tp = 100e-9 / 1024  ## simulation setting  nyu is just 80ps
                    dep_re = dep_re * 1024 * Tp * C / 2   

                    l_rmse.append(cal_rmse(dep_re,torch.from_numpy(deps[None][None]).cuda()))
                    l_ssim.append(cal_ssim(dep_re,torch.from_numpy(deps[None][None]).cuda()))
                    # print(l_ssim[-1])
                    dep_re = dep_re.detach().cpu().numpy()[0, 0]

                    dep_mat[:,:,0,k] = dep_re
                    dep_mat[:,:,1,k] = deps
                    
                    fig = plt.figure(figsize=(18, 12))
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(dep_re, vmin=1.4, vmax=5.2)
                    ax2 = fig.add_subplot(1, 2, 2)      
                    ax2.imshow(deps, vmin=1.4, vmax=5.2)
                    save_path = f'{out_path}/dep/{sbrs[j]}/{k}.png'
                    plt.savefig(save_path) 
                    plt.close()
                    scio.savemat(f'{out_path}/dep/{sbrs[j]}/all.mat',{'pre_gt':dep_mat})

                infor = str(pathsbr[j]) +'    img_ssim:' + str(float(sum(l_ssim))/float(len(l_ssim)))+'     dep_rmse:' +str(float(sum(l_rmse))/float(len(l_rmse)))
                logging.info(infor)
                # print(infor)
                if int(sbrs[j].split('_')[-1])==2:
                    ppp2.append(float(sum(l_rmse))/float(len(l_rmse)))
                elif int(sbrs[j].split('_')[-1])==10:
                    ppp10.append(float(sum(l_rmse))/float(len(l_rmse)))
                elif int(sbrs[j].split('_')[-1])==50:
                    ppp50.append(float(sum(l_rmse))/float(len(l_rmse)))
                else:
                    ppp100.append(float(sum(l_rmse))/float(len(l_rmse)))
            
            infor_ave = 'x_2 average:' + str(float(sum(ppp2))/float(len(ppp2))) +'x_10 average:' + str(float(sum(ppp10))/float(len(ppp10))) +'x_50 average:' + str(float(sum(ppp50))/float(len(ppp50))) +'x_100 average:' + str(float(sum(ppp100))/float(len(ppp100)))
            logging.info(infor_ave)


if __name__=="__main__":
    main()




