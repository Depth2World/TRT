import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from pro.Loss import criterion_KL, criterion_L2, criterion_TV
import torch.nn.init as init
from math import sqrt, pow, log2
from .trtsp_modules import *



class trt_sp(nn.Module):
    def __init__(self,in_ch=1,out_ch=1, spatial=256,tlen=1024,frames=3, coders = 12, local_rank=-1):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.spatial = spatial
        self.tlen = tlen
        self.frames = frames
        self.coders = coders
        self.local_rank = local_rank
        channels_m = 64
        self.feature_extraction = MsFeat_3(self.in_channels, channels_m)
        self.tds0 = nn.Sequential(nn.Conv3d(channels_m * 4,channels_m * 4,kernel_size=3,stride=(2,1,1),padding=(1,1,1),bias=True),nn.ReLU(inplace=True))
        self.tds1 = nn.Sequential(nn.Conv3d(channels_m * 4, channels_m * 4, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),nn.ReLU(inplace=True))

        self.ds0 = SPDS(channels_m * 4, 2) 
        self.posenc_l = PosiEncCNN(channels_m * 4)
        self.posenc_g = PosiEncCNN(channels_m * 4)
         # local encoders
        self.loc_encds = modelClone(WindowEncoderSep(dim=channels_m * 4,input_resolution=[spatial//4,spatial//4,tlen//16],num_heads=8,window_size=8), self.coders)
        # global encoders
        self.glb_encds = modelClone(GlobalEncoderSep(dim=channels_m * 4,input_resolution=[spatial//8,spatial//8,tlen//16],num_heads=8), self.coders)
        # local-global integration
        self.locglb_inte = LocGlbInteNBlks_LCGC_l1d2(channels_m * 4, [spatial//4,spatial//4,tlen//4],8, self.coders)
        self.inte_rec = NLOSInteRec_4(self.in_channels,channels_m)       

    
    def forward(self, meas):
        ms_fea = self.feature_extraction(meas) # b c t/4 h/4 w/4
        tds_fea = self.tds0(ms_fea)  # b c t/8 h/4 w/4 
        tds_fea = self.tds1(tds_fea)  # b c t/16 h/4 w/4

        ds_fea = self.ds0(tds_fea) # b c t/16 h/8 w/8 
        
        posenc_l = self.posenc_l(tds_fea)    
        posenc_g = self.posenc_g(ds_fea)    
        
        # local information extraction 
        for loc_encd in self.loc_encds: # b c t/16 h w 
            posenc_l = loc_encd(posenc_l)      
        # global information extraction
        for glb_encd in self.glb_encds:  #  b c t/16 h/4 w/4 
            posenc_g = glb_encd(posenc_g)   

        inte_loc, inte_glb = self.locglb_inte(posenc_l, posenc_g)    # b c t/16 h/4 w/4 

        out_vlo = self.inte_rec(inte_loc, inte_glb, ms_fea)  # b c t h w   
        denoise_out = torch.squeeze(out_vlo,dim=1)
        weights = Variable(torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * torch.nn.Softmax2d()(denoise_out)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)
        return denoise_out, soft_argmax # torch.Size([1, 1024, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
            

    def train_one_iteration(self,optimizer):
        pred_vlo, rendered_depth = self.forward(self.meas)
        errs = self.compute_train_loss(pred_vlo, self.meas_gt, rendered_depth, self.dep_gt)
        if not isinstance(errs, list) and not isinstance(errs, tuple):
            errs = [errs]
        loss = sum(errs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def test_one_iteration(self):
        pred_vlo, rendered_depth = self.forward(self.meas)
        errs = self.compute_train_loss(pred_vlo, self.meas_gt, rendered_depth, self.dep_gt)
        self.rendered_depth = rendered_depth
        # if not isinstance(errs, list) and not isinstance(errs, tuple):
        #     errs = [errs]
        # loss = sum(errs)
        
    def set_input(self,sequence):
        self.meas_gt = sequence['rates']
        self.meas = sequence['spad']
        self.int_gt = sequence['intensity']
        self.dep_gt = sequence['bins']
    
  
    def evluation_rw(self,meas):
        self.vlo, self.rendered_depth = self.forward(meas)

    def compute_train_loss(self,input0,gt0,input1,gt1):
        self.vals = []
        lsmx = torch.nn.LogSoftmax(dim=1)
        vlo_lsmx = lsmx(input0).unsqueeze(1)
        kl_loss = criterion_KL(vlo_lsmx,gt0)
        tv_loss = criterion_TV(input1)

        self.vals.append(1*kl_loss)
        self.vals.append(1e-5*tv_loss)
        return self.vals
    
    
    def get_loss(self):
        return self.vals
    
