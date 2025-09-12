import torch
import logging
from .worker import worker
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler

import imageio
from models.network import trt_sp
from pro.Loss import criterion_KL, criterion_L2
from utils import dynamic_dataset

from metric import RMSE, PSNR, SSIM, AverageMeter
from sklearn.metrics import accuracy_score as ACC
import os 
import scipy.io as scio
import numpy as np
import torch.nn.functional as F
import cv2
###DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
metric_list = ['rmse', 'psnr', 'ssim','acc']
val_metrics = {k: AverageMeter() for k in metric_list}


class dy_model(worker):
    def __init__(self,args):
        super().__init__(args=args)
       
        
    def load_resume(self):
        
        if self.args.resume:
            if os.path.exists(self.args.resmod_dir):
                logging.info("Loading checkpoint from {}".format(self.args.resmod_dir))
                checkpoint = torch.load(self.args.resmod_dir, map_location="cpu")
                # load start epoch
                try:
                    self.start_epoch = checkpoint['epoch']
                    logging.info("Loaded and update start epoch: {}".format(self.start_epoch))
                except KeyError as ke:
                    self.start_epoch = 0
                    logging.info("No epcoh info found in the checkpoint, start epoch from 1")
                # load iter number
                try:
                    self.n_iter = checkpoint["n_iter"]
                    logging.info("Loaded and update start iter: {}".format(self.n_iter))
                except KeyError as ke:
                    self.n_iter = 0
                    logging.info("No iter number found in the checkpoint, start iter from 0")

                # load learning rate
                try:
                    self.args.lr_rate = checkpoint["lr"]
                except KeyError as ke:
                    logging.info("No learning rate info found in the checkpoint, use initial learning rate:")
                # load model params
                # model_dict = model.state_dict()
                try:
                    ckpt_dict = checkpoint['state_dict']
                    self.model.load_state_dict(ckpt_dict)
                    logging.info("Loaded and update model states!")
                except KeyError as ke:
                    logging.info("No model states found!")
                    sys.exit("NO MODEL STATES")

                logging.info("Checkpoint load complete!!!")

            else:
                logging.info("No checkPoint found at {}!!!".format(self.args.resmod_dir))
                sys.exit("NO FOUND CHECKPOINT ERROR!")
        else:
            logging.info("Do not resume! Use initial params and train from scratch.")

    def init_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.opter == 'adamw':
            self.optimizer = torch.optim.AdamW(params, lr=self.args.lr_rate, weight_decay=self.args.weit_decay)
        else:
            self.optimizer = torch.optim.Adam(params, lr=self.args.lr_rate)

    def train_model(self): 
        train_data = dynamic_dataset.DynamicNLOSDataset(self.args.root_path, self.args.train_total_path,0,3,self.args.data_size,True)

        train_sampler = DistributedSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler,batch_size=self.args.train_bacth_size, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)
        
        test_data = dynamic_dataset.DynamicNLOSDataset(self.args.root_path, self.args.test_total_path,0,3,self.args.data_size,False)
        val_loader = DataLoader(test_data,batch_size=self.args.test_bacth_size,shuffle=True, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)
                
        self.model = trt_sp(in_ch=1,out_ch=1,spatial=64,tlen=1024,frames=3,local_rank=self.local_rank)
        ### load resume
        self.model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank],find_unused_parameters=True)
        self.load_resume()
        
        logging.info("Models constructed complete! Paralleled on {} GPUs".format(torch.cuda.device_count()))
        
        # optimizer 
        self.init_optimizer()

        for epoch in range(self.start_epoch,self.args.num_epoch):
            self.current_epoch = epoch
            
            if True:
                self.model.train()
                train_loader.sampler.set_epoch(epoch)
                logging.info('{} epcoh start training!'.format(epoch))
                # self.train_new_epoch()
                loss_kl,loss_tv = 0, 0
                ### training
                for iteration, sequence in enumerate(train_loader):
                    # print('training',self.n_iter)
                    self.n_iter += 1
                    data = self.prepare_data(sequence)
                    self.model.module.set_input(data)
                    
                    self.model.module.train_one_iteration(self.optimizer)
                    # self.loss = self.model
                    errs = self.model.module.get_loss()
                    
                    loss_kl += errs[0].item()
                    loss_tv += errs[1].item()
                
                if dist.get_rank() == 0:
                    loss_kl /= (iteration+1)
                    loss_tv /= (iteration+1)
                    self.writer.add_scalar('Training/Loss_kl', loss_kl, epoch)
                    self.writer.add_scalar('Training/Loss_tv', loss_tv, epoch)

            if True:    
                logging.info('{} epcoh start validation!'.format(epoch))
                ## validation
                loss_kl_test, loss_tv_test = 0, 0
                rmse = RMSE().cuda()
                # psnr = PSNR().cuda()
                ssim = SSIM().cuda()
                l_rmse = []
                # l_psnr = []
                l_ssim = []
                self.model.eval()
                with torch.no_grad():
                    for iters, seq in enumerate(val_loader):
                        data = self.prepare_data(seq)
                        self.model.module.set_input(data)
                        self.model.module.test_one_iteration()
                        errs = self.model.module.get_loss()
                        
                        loss_kl_test += errs[0].item()
                        loss_tv_test += errs[1].item()
                              
                        l_rmse.append(rmse(self.model.module.rendered_depth * 1024 *3e8 * 80e-12 / 2, self.model.module.dep_gt * 1024 *3e8 * 80e-12 / 2))
                        # l_psnr.append(psnr(self.model.module.rendered_img, self.model.module.int_gt))
                        l_ssim.append(ssim(self.model.module.rendered_depth * 1024 *3e8 * 80e-12 / 2, self.model.module.dep_gt * 1024 *3e8 * 80e-12 / 2))

                if dist.get_rank() == 0:
                    loss_kl_test /= (iters+1)
                    loss_tv_test /= (iters+1)

                    self.writer.add_scalar('Testing/Loss_kl', loss_kl_test, epoch)
                    self.writer.add_scalar('Testing/Loss_tv', loss_tv_test, epoch)

                    self.writer.add_scalar('Evluation/rmse', sum(l_rmse)/len(l_rmse),epoch)
                    # self.writer.add_scalar('Evluation/psnr', sum(l_psnr)/len(l_psnr),epoch)
                    self.writer.add_scalar('Evluation/ssim', sum(l_ssim)/len(l_ssim),epoch)

                    self.writer.add_images("Testing/dep", self.model.module.rendered_depth, epoch,dataformats="NCHW")
                    self.writer.add_images("Testing/gt", self.model.module.dep_gt, epoch,dataformats="NCHW")
            
             ### save checkpoint
            file_path = self.args.model_dir+"/epoch_{}_iter_{}.pth".format(epoch, self.n_iter)
            self.save_checkpoint(self.n_iter, epoch, self.model, self.optimizer, file_path) 
            # for g in self.optimizer.param_groups:
                # g['lr'] *= 0.9

    def prepare_data(self,sequence):
        spad = sequence['spad'].to(self.local_rank)
        rates = sequence['rates'].to(self.local_rank)
        bins = sequence['bins'].to(self.local_rank)
        intensity = sequence['intensity'].to(self.local_rank)
        data = {'rates': rates, 'spad': spad, 'bins': bins, 'intensity': intensity}
        return data


