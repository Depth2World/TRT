from enum import EnumMeta
import os 
import glob
from pickle import TRUE
import numpy as np
import cv2
from torch.utils.data import Dataset,DataLoader
import scipy.io as sio
import random
import torch
import torch.nn.functional as F
from math import log2
import sys
import scipy
import time
import h5py
import scipy.sparse as sp

def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)


def read_dataset(source):
    if not os.path.exists(source):
        print("Setting file %s for dataset doesn't exist." % (source))
        sys.exit()
    else:
        with open(source) as split_f:
            path_samples = split_f.readlines()
        print('{} sequences have been loaded'.format(len(path_samples)))
    return path_samples



def load_sparse_hdf5_group(group, name):
    """
    从 HDF5 文件中的一个 group 读取稀疏矩阵。
    """
    data = group[name + '/data'][:]
    indices = group[name + '/indices'][:]
    indptr = group[name + '/indptr'][:]
    shape = tuple(group[name + '/shape'][:])

    return sp.csc_matrix((data, indices, indptr), shape=shape)

class DynamicNLOSDataseth5(Dataset):
    def __init__(
        self, 
        root_path,
        filter_path,               # dataset root directory
        target_noise=0,     # standard deviation of target image noise
        frame_num = 3,
        spatial_size = 64,
        training = True
    ):
        super(DynamicNLOSDataseth5, self).__init__()
        self.root_path = root_path
        self.filter_path = read_dataset(filter_path)
        self.target_noise = target_noise
        self.frame_num = frame_num
        self.sps = spatial_size
        self.training = training
        self.output_size = 64
        if not self.training:
            self.filter_path = self.filter_path[:len(self.filter_path)//10]
        print('len',len(self.filter_path))

    def load_mea_int_dep(self, seq_path):
        """
        从 HDF5 文件读取稀疏矩阵。
        """
        with h5py.File(seq_path[:-4]+'.h5', 'r') as h5f:
            spad = load_sparse_hdf5_group(h5f, 'spad')
            rates = load_sparse_hdf5_group(h5f, 'rates')
            bins = h5f['bins']
            intensity = h5f['intensity']

            spad = np.asarray(scipy.sparse.csc_matrix.todense(spad)).reshape([1, self.sps, self.sps, -1]) # 1,self.sps,self.sps,1024
            spad = np.transpose(spad, (0, 3, 2, 1))

            # normalized pulse as GT histogram
            rates = np.asarray(scipy.sparse.csc_matrix.todense(rates)).reshape([1, self.sps, self.sps, -1])
            rates = np.transpose(rates, (0, 3, 1, 2))
            rates = rates / np.sum(rates, axis=1)[None, :, :, :] 

            bins = (np.asarray(bins).astype(np.float32).reshape([self.sps, self.sps])-1)[None, :, :] / 1023
            intensity = (np.asarray(intensity).astype(np.float32).reshape([self.sps, self.sps]))[None, :, :]
        
            return spad, rates, bins, intensity

    def tryitem(self,idx):
        path = self.root_path + self.filter_path[idx][:-1]
        # print(path)
        spad, rates, bins, intensity = self.load_mea_int_dep(path)
        if True:
            h, w = spad.shape[2:]
            new_h = self.output_size
            new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            rates = rates[:, :, top: top + new_h,
                        left: left + new_w]
            spad = spad[:, :, top: top + new_h,
                        left: left + new_w]
            bins = bins[:, top: top + new_h,
                        left: left + new_w]
            intensity = intensity[:, top: top + new_h,
                        left: left + new_w]


        rates = torch.from_numpy(rates).float()
        spad = torch.from_numpy(spad).float()
        bins = torch.from_numpy(bins).float()
        intensity = torch.from_numpy(intensity).float()
        # print(rates.shape, spad.shape, bins.shape, intensity.shape)
        sample = {'rates': rates, 'spad': spad, 'bins': bins, 'intensity': intensity}
        return sample
    
    def __len__(self):
        return len(self.filter_path)

    def __getitem__(self, idx):
        try:
            sample = self.tryitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self.tryitem(idx)
        return sample



class DynamicNLOSDataset(Dataset):
    def __init__(
        self, 
        root_path,
        filter_path,               # dataset root directory
        target_noise=0,     # standard deviation of target image noise
        frame_num = 3,
        spatial_size = 64,
        training = True
    ):
        super(DynamicNLOSDataset, self).__init__()
        self.root_path = root_path
        self.filter_path = read_dataset(filter_path)
        self.target_noise = target_noise
        self.frame_num = frame_num
        self.sps = spatial_size
        self.training = training
        self.output_size = 64
        if not self.training:
            self.filter_path = self.filter_path[:len(self.filter_path)//10]
        print('len',len(self.filter_path))

    def load_mea_int_dep(self, seq_path):
        all_data = sio.loadmat(seq_path)
        spad = np.asarray(scipy.sparse.csc_matrix.todense(all_data['spad'])).reshape([1, self.sps, self.sps, -1]) # 1,self.sps,self.sps,1024
        spad = np.transpose(spad, (0, 3, 2, 1))

        # normalized pulse as GT histogram
        rates = np.asarray(all_data['rates']).reshape([1, self.sps, self.sps, -1])
        rates = np.transpose(rates, (0, 3, 1, 2))
        rates = rates / np.sum(rates, axis=1)[None, :, :, :] 

        bins = (np.asarray(all_data['bin']).astype(np.float32).reshape([self.sps, self.sps])-1)[None, :, :] / 1023
        intensity = (np.asarray(all_data['intensity']).astype(np.float32).reshape([self.sps, self.sps]))[None, :, :]
        return spad, rates, bins, intensity

    def tryitem(self,idx):
        path = self.root_path + self.filter_path[idx][:-1]
        # print(path)
        spad, rates, bins, intensity = self.load_mea_int_dep(path)
        if True:
            h, w = spad.shape[2:]
            new_h = self.output_size
            new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            rates = rates[:, :, top: top + new_h,
                        left: left + new_w]
            spad = spad[:, :, top: top + new_h,
                        left: left + new_w]
            bins = bins[:, top: top + new_h,
                        left: left + new_w]
            intensity = intensity[:, top: top + new_h,
                        left: left + new_w]


        rates = torch.from_numpy(rates).float()
        spad = torch.from_numpy(spad).float()
        bins = torch.from_numpy(bins).float()
        intensity = torch.from_numpy(intensity).float()
        # print(rates.shape, spad.shape, bins.shape, intensity.shape)
        sample = {'rates': rates, 'spad': spad, 'bins': bins, 'intensity': intensity}
        return sample
    
    def __len__(self):
        return len(self.filter_path)

    def __getitem__(self, idx):
        try:
            sample = self.tryitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self.tryitem(idx)
        return sample
    

if __name__ == '__main__': 
    root_path = '/data/yueli/dataset/sp_syn/nyu_syn_sp_256_h5/'
    total_path = '/data/yueli/code/tapami_single_photon_v2/utils/256fortraining.txt'
    train_data = DynamicNLOSDataseth5(root_path, total_path,0,3,256,True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True) # drop_last would also influence the performance
    print(len(train_data))
    now = time.time()
    for index,data in enumerate(train_loader):
        print(time.time() - now)
        print(data['rates'].shape,data['spad'].shape,data['bins'].shape,data['intensity'].shape)   
        now = time.time()
        # print(torch.max(data['meas']),torch.max(data['meas_gt']))