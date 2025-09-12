import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import scipy.io as scio
import imageio
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import scipy

cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2
from skimage.metrics import structural_similarity as ssim
from models.network import dy_nlos
from matplotlib import pyplot as plt


from matplotlib.colors import ListedColormap


colormapc = np.loadtxt('colors.csv', delimiter=',')
cmap = ListedColormap(colormapc, name='parula')

def get_mpl_colormap(cmap_name):
    if cmap_name == 'parula':
        # cmap = loadmat('parula.mat')['parula']
        # cmap = cmocean.cm.parula
        cmap = ListedColormap(colormapc, name='parula')
    else:
        cmap = plt.get_cmap(cmap_name)
        
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    # return cmap
    return color_range.reshape(256, 1, 3)

def color_code(im, start, stop, name='parula'): #parula #viridis
    # msk = np.zeros_like(im)
    # msk[np.logical_and(start < im, im < stop)] = 1.0
    im = np.clip(im, start, stop)
    im = (im-start) / float(stop-start)
    im = im * 255.0
    im = im.astype(np.uint8)
    im = cv2.applyColorMap(im, get_mpl_colormap(name))
    # im[msk != 1.0] = 0
    return im


start_all = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,  1,    0.5] 
stop_all  = [2.5, 1.5, 5  , 5  , 2.5, 2.5, 2.5, 2.5, 2  ]
name = ['checkerboard','elephant','hallway','kitchen',\
    'lamp','roll','stairs_ball','stairs_walking',\
        'stuff']


def main():
    
    # baseline   
    model = dy_nlos(in_ch=1,out_ch=1,spatial=256,tlen=1024,frames=3,local_rank=-1)
    
    model_root = 'xxx'
    
    out_root = model_root + '/rw_lindell/'
    all_model_file = []
    model_files = os.listdir(model_root)
    for fi in model_files:
        if ('pth' in fi) and ('END' not in fi):
                fi_d = os.path.join(model_root, fi)
                all_model_file.append(fi_d)

    model.cuda()
    model = torch.nn.DataParallel(model)
    print("Start eval...")
    rw_path  = '/data1/yueli/dataset/single_photon_rw/lindell/'
    
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
        
    for k in range(len(all_model_file)):
        model_path = all_model_file[k]

        if not os.path.exists(out_root):
            os.makedirs(out_root, exist_ok=True)

  
        checkpoint = torch.load(model_path, map_location="cpu")
        ckpt_dict = checkpoint['state_dict']
        model.load_state_dict(ckpt_dict)
        C = 3e8
        Tp = 26e-12

        for i in range(len(all_file)):
            
            out_path = out_root + '/n_iter' + model_path.split('_')[-1][:-4] 
            print(all_file[i])
            M_mea = np.asarray(scio.loadmat(rw_path + name[i] + '.mat')["spad_processed_data"])[0][0]
            M_mea = scipy.sparse.csc_matrix.todense(M_mea)
            M_mea = np.ascontiguousarray(M_mea).astype(np.float32).reshape([1, 1, 1536, 256, 256])
            M_mea = M_mea.transpose((0,1,2,4,3))
            M_mea = torch.from_numpy(M_mea).cuda()
            M_mea = F.interpolate(M_mea,[1024,256,256])
            print(M_mea.shape)
            with torch.no_grad():
                model.module.evluation_rw(M_mea)
                M_mea_re, dep_re = model.module.vlo, model.module.rendered_depth
                M_mea_re = M_mea_re.data.cpu().numpy().squeeze()
                dep_re = dep_re.detach().cpu().numpy()[0, 0]
                tile_out = np.argmax(M_mea_re, axis=0)

                # dist1 = (tile_out/1024) * 1536 * Tp * C /2
                dist2 = dep_re  * 1536 * Tp * C /2
                
                start = start_all[i] #gt.min() -0.5
                stop = stop_all[i] #gt.max() + 0.5
                # pre_c = color_code(dist1,start,stop)        
                pre_l = color_code(dist2,start,stop)        
                
                # print(out_path + name[i]+'_prec.png')
                # cv2.imwrite(os.path.join(out_path+ name[i]+'_prec.png'),pre_c)
                cv2.imwrite(os.path.join(out_path+ name[i]+'_prel.png'),pre_l)
                
                
                # fig = plt.figure(figsize=(18, 12))
                # ax1 = fig.add_subplot(1, 2, 1)
                # im1 = ax1.imshow(dist1, vmin=0.5, vmax=2.5)
                # fig.colorbar(im1,ax=ax1)
                # ax2 = fig.add_subplot(1, 2, 2)      
                # im2 = ax2.imshow(dist2, vmin=0.5, vmax=2.5)
                # fig.colorbar(im2,ax=ax2)
                # plt.savefig(out_path + f'_color.png')  
                # plt.close()
                # scio.savemat(out_path + '.mat', {"vlo_pre":dist1,"dep_pre":dist2})

if __name__=="__main__":
    main()




