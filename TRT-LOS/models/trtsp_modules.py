
from math import sqrt, pow, log2
from operator import pos
import numpy as np
from numpy.core.shape_base import block
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.modules.activation import Hardswish
from torch.nn.modules.container import T
from torch.nn.modules.normalization import CrossMapLRN2d


def window_partition_swin(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_partition(inputs, window_size):
    r"""The window_partition function
    Put the input images into non-overlapped windows patches
    Input: (B, C, D, H, W); window_size [int]
    Output: (num_windows*B, C, D, window_size, window_size)
    """

    B, C, D, H, W = inputs.shape
    patches = inputs.view(B, C, D, H // window_size, window_size, W // window_size, window_size)
    windows = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous().view(-1, C, D, window_size, window_size)
    return windows


def window_reverse(inputs, H, W):
    """The window_reverse function
    Put the patch windows into one image
    Input: (num_windows*B, C, D, window_size, window_size); window_size, H, W
    Output: (B, C, D, H, W)
    """

    NB, C, D, window_size, _ = inputs.shape
    B = int(NB / (H * W / window_size / window_size))
    patches = inputs.view(B, H // window_size, W // window_size, C, D, window_size, window_size)
    out = patches.permute(0, 3, 4, 1, 5, 2, 6).contiguous().view(B, C, D, H, W)
    return out



def softArgMax(inputs):
    r"""The softArgMax function
    It computes the softargmax along a certain dimension of the input tensor
    Input: (B,1024, H, W)
    Output: (B, 1, H, W)
    """

    smax = nn.Softmax2d()
    weights = torch.linspace(0,1,steps=inputs.shape[1]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor)
    weighted_smax = weights * smax(inputs)
    out = weighted_smax.sum(1).unsqueeze(1)

    return out


def modelClone(module, N):
    r"""The modelClone function
    It clones the given model with N copies.
    """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PosiEncCNN(nn.Module):
    r"""The PosiEncCNN block
    It generates the positional encodings for the input tensor, which is a simple CNN on the input.
    Input:
        in_chans (int): channel size of input tokens
        out_chans (int): channel size of output tokens
    Output: out= inputs+ position
    """

    def __init__(self, in_chans):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = in_chans
        self.proj = nn.Conv3d(self.in_chans,self.out_chans,3,1,1,bias=True)
        fan_out = self.proj.kernel_size[0]* self.proj.kernel_size[1]* self.proj.kernel_size[2]* self.out_chans
        init.normal_(self.proj.weight, 0, sqrt(2.0 / fan_out)); init.constant_(self.proj.bias, 0.0)
        # init.kaiming_normal_(self.proj.weight, 0, 'fan_in', 'relu'); init.constant_(self.proj.bias, 0.0)
        
        print("\033[0;33;40m=> WARN: PosiEncCNN block in/out channel sizes {} / {} \033[0m".format(self.in_chans, self.out_chans))

    def forward(self, inputs):
        r"""
        Input: inputs has size of (B, Cin, D, H, W)
        Output: encoded output has size of (B, Cout, D, H, W)
        """

        pos_code = self.proj(inputs)
        out = pos_code + inputs

        return out

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

    def flops(self, N):
        flops = 0
        # self.proj(inputs)
        flops += self.proj.kernel_size[0]* self.proj.kernel_size[1]* self.proj.kernel_size[2]* self.out_chans* self.in_chans* N

        return flops



class MsFeat_3(nn.Module):
    """The feature extraction module 
    Extract features with downsampled temporal dim
    Input: B, C, D, H, W
    Output: B, 32, D/4, H, W
    """

    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.ds = nn.Sequential(nn.Conv3d(self.in_chans, self.in_chans, 7, stride=(2,2,2), padding=3, dilation=1, groups=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.ds[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds[0].bias, 0.0)

        self.conv1 = nn.Sequential(nn.Conv3d(self.in_chans, self.out_chans, 3, stride=(2,2,2), padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(nn.Conv3d(self.in_chans, self.out_chans, 3, stride=(2,2,2), padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)

        self.conv3 = nn.Sequential(nn.Conv3d(self.out_chans, self.out_chans, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)

        self.conv4 = nn.Sequential(nn.Conv3d(self.out_chans, self.out_chans, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)

        print("\033[0;33;40m=> WARN: MsFeat block in/out channel sizes {} / {}\033[0m".format(self.in_chans, 4* self.out_chans))

    def forward(self, inputs):
        ds = self.ds(inputs)
        conv1 = self.conv1(ds)
        conv2 = self.conv2(ds)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1)
        return torch.cat((conv1, conv2, conv3, conv4), 1)
        # return torch.cat((conv1,conv4), 1) + torch.cat((conv2, conv3), 1)
        

    def flops(self, N):
        flops = 0
        # self.ds(inputs)
        flops += self.in_chans* self.in_chans* N* 7* 7* 7//2
        # self.conv1(ds), self.conv2(ds)
        flops += 2* self.in_chans* self.out_chans* N//4* 9
        # self.conv3(conv2), self.conv4(conv2)
        flops += 2* self.out_chans* self.out_chans* N//4* 9

        return flops


class SPDS(nn.Module):
    """The SPDS block
    Downsample spatial dim and keep the temporal dimension
    Input: B, C, D, H, W
    Output: B, C, D, H/ds, W/ds (default)
    """

    def __init__(self, ch_in, ds=2):
        super().__init__()

        self.in_chans = ch_in
        self.ds = ds

        self.dsop = nn.Sequential(
            nn.Conv3d(self.in_chans, self.in_chans, 2*self.ds+ 1, (1,self.ds,self.ds), self.ds, bias=True),
            nn.ReLU(inplace=True)
        )
        init.kaiming_uniform_(self.dsop[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.dsop[0].bias, 0.)

    def forward(self, inputs):

        out = self.dsop(inputs)

        return out

    def flops(self, N):
        flops = 0
        # self.dsop(inputs)
        flops += self.in_chans* self.in_chans* pow((2*self.ds+ 1),3)* N

        return flops


class DsFusion(nn.Module):
    """The DsFusion block
    Downsample temporal dim and fuse the them along channel
    Input: B, Cin, D, H, W
    Output: B, Cout, D/4, H, W
    """

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.in_chans = ch_in
        self.out_chans = ch_out
        self.ds = nn.Sequential(
            nn.Conv3d(self.in_chans, self.in_chans, kernel_size=5,stride=(2,1,1),padding=(2,2,2),bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.in_chans, self.in_chans, kernel_size=5,stride=(2,1,1),padding=(2,2,2),bias=True),
            nn.ReLU(inplace=True)#,
            #nn.Conv3d(self.in_chans, self.in_chans, kernel_size=5,stride=(2,1,1),padding=(2,2,2),bias=True),   #  nyu2
            #nn.ReLU(inplace=True)                                                                               #  nyu2
        )
        init.kaiming_normal_(self.ds[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds[0].bias, 0.0)
        # init.kaiming_normal_(self.ds[2].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds[2].bias, 0.0)
        #init.kaiming_normal_(self.ds[4].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds[4].bias, 0.0)   #  nyu2

        self.fusion = nn.Sequential(
            nn.Conv3d(self.in_chans, self.out_chans, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.ReLU(inplace=True)
        )
        init.kaiming_normal_(self.fusion[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.fusion[0].bias, 0.0)

        print("\033[0;33;40m=> WARN: DsFusion block in/out channel sizes and DS scale {} / {} / {}\033[0m".format(self.in_chans, self.out_chans, 4))
    
    def forward(self, inputs):
        ds = self.ds(inputs)
        out = self.fusion(ds)

        return out

    def flops(self, N):
        flops = 0
        # self.ds(inputs)
        flops += 2* self.in_chans* self.in_chans* self.ds[0].kernel_size[0]* self.ds[0].kernel_size[1]* self.ds[0].kernel_size[2]* N
        # self.fusion(ds)
        flops += self.in_chans* self.out_chans * N

        return flops


        
class Transient_AllD2(nn.Module):
    def __init__(self, in_channels,out_channels, norm=nn.InstanceNorm3d):
        super(Transient_AllD2, self).__init__()
        ###############################################
        # assert in_channels == 1
        weights = np.zeros((in_channels, in_channels, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        # tfweights = torch.from_numpy(weights)

        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=(2,2,2),
                      bias=True),
            ResConv3D(out_channels, inplace=False),
            ResConv3D(out_channels, inplace=False)
        )
    def forward(self, x0):
        # x0 is from 0 to 1
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=(2,2,2), padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)
        # print(x1.shape)
        # re = torch.cat([x0_conv, x1], dim=1)
        re = x0_conv + x1

        return re
        # return x0_conv   # wo fea

   
class Transient_TDown_3(nn.Module):
    def __init__(self, in_channels,out_channels, norm=nn.InstanceNorm3d):
        super(Transient_TDown_3, self).__init__()
        ###############################################
        # assert in_channels == 1
        # weights = np.zeros((1, in_channels, 3, 3, 3), dtype=np.float32) #  inistial 14 
        weights = np.zeros((in_channels, in_channels, 3, 3, 3), dtype=np.float32) #  up 14 

        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        # tfweights = torch.from_numpy(weights)

        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=(2,1,1),
                      bias=True),
            ResConv3D(out_channels, inplace=False),
            ResConv3D(out_channels, inplace=False)
        )
    def forward(self, x0):
        # x0 is from 0 to 1
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=(2,1,1), padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)
        # print(x1.shape)
        re = x0_conv + x1          #  this use the broadcast  is that suitable?
        return re
       
class LocGlbInteNBlks_LCGC_l1d2(nn.Module):
    r"""The LocGlbInteNBlks block
    It integrates the information from local patches and global images with several numbers (can assign) of transformer decoders.
    *It use our CGNLMHAttention instead of built-in MHA in pytorch.*
    Note that these decoders do NOT use the positional encodings. And the number of decoders can be assigned.
    For local pathes, its resolution is: B, C, D, H, W.
    For global images, its resolution is: B, C, D, H/2, W/2.
    Workflow: local patches-> as Q to refine the global images; global images-> as Q to refine the local pathes.
    Input:
        ch_in (int): Number of input channels.
        input_resolution (tuple[int, int, int]): Input resulotion of H, W, and D.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.1
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    Output: Integrated output, two tensor with resolution of (B, C, D, H, W).
    """

    def __init__(self, ch_in=64, input_resolution=[32,32,64], num_heads=8, num_ders = 3, res_con = 0,
                mlp_ratio=4, mlp_drop=0., attn_drop=0.1, 
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_chans = ch_in
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.num_ders = num_ders
        self.mlp_ratio = mlp_ratio
        self.drop = mlp_drop
        self.resid = res_con
        # self.upop_L = SPUP(self.in_chans,up=2)
        self.upop_G = SPUP(self.in_chans,up=2)

        # local and global pathes decoders
        # self.loc_decs = modelClone(CrossDecoder_spatial(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
        self.loc_decs = modelClone(CrossDecoder(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
        # self.loc_decs = modelClone(CrossDecoderVIT(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
        
        # self.glb_decs = modelClone(CrossDecoder_spatial(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
        self.glb_decs = modelClone(CrossDecoder(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
        # self.glb_decs = modelClone(CrossDecoderVIT(self.in_chans, self.input_resolution, self.num_heads, self.resid), self.num_ders)
            
        attn_mask_loc = None
        attn_mask_glb = None

        self.register_buffer("attn_mask_loc", attn_mask_loc)
        self.register_buffer("attn_mask_glb", attn_mask_glb)

        print("\033[0;33;40m=> WARN: LocGlbInteNBlks block in/out channel sizes, num_head, num_decoders, and resolution {} / {} / {} / {} / {},{},{} \033[0m".format(self.in_chans, self.in_chans, self.num_heads, self.num_ders, self.input_resolution[0],self.input_resolution[1],self.input_resolution[2]))

    def forward(self, inpt_loc, inpt_glb):
        r"""
        Input:
            inputs: input local and global features with shape of (B, C, D, H, W). Use the local patches as Q to process the global images; use the global images as Q to process the local patches.
            workflow: local patches: Q-> global images;
            global images: Q-> local patches
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None, defult: None.
        Output: out: two tensor with resolution of (B, C, D, H, W).
        MIND THE SHAPE CHANGES!!!
        """
        # up_loc = self.upop_L(inpt_loc)    # B,C,D,H,W
        up_glb = self.upop_G(inpt_glb)    # B,C,D,H,W
        up_loc = inpt_loc

        raw_loc = up_loc
        raw_glb = up_glb
        # cross correlations between decoders
        for loc_dec, glb_dec in zip(self.loc_decs, self.glb_decs):
            # print(up_loc.shape,up_glb.shape)
            #version 3 by yl  self and cross attention lcgs
            out_loc = loc_dec(up_loc, up_glb)
            out_glb = glb_dec(up_glb, up_loc)
            up_loc = out_loc
            up_glb = out_glb
        return raw_loc + up_loc, raw_glb + up_glb
        
    def flops(self):
        flops = 0
        H, W, D = self.input_resolution
        # self.upop(inpt_glb)
        flops += self.upop.flops(H* W* D)
        # local and global attension
        flops += self.num_ders* 2* self.in_chans* H* W* D
        flops += self.num_ders* 2* (4* H*W* pow(self.in_chans,2)+ pow(H*W,2)*self.in_chans) + (4* D* pow(self.in_chans,2)+ pow(D,2)*self.in_chans)
        flops += self.num_ders* 2* 2* H* W* D* self.in_chans* self.in_chans* self.mlp_ratio
        flops += self.num_ders* 2* self.in_chans* H* W* D

        return flops
   
        
class ResConv3D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv3D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
        )
        self.inplace = inplace
    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re


class binsregression(nn.Module):
    def __init__(self, maxbins):
        super(binsregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxbins))/(maxbins-1),[1, maxbins,1,1])).cuda()

    def forward(self, x):
        pred_vlo = x*self.disp.data
        bins = torch.sum(pred_vlo,1, keepdim=True)
        return pred_vlo,bins
    

class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        upscale_factor = self.upscale_factor
        
        # Ensure the number of input channels is divisible by upscale_factor^3
        assert channels % (upscale_factor ** 3) == 0, \
            f"Number of input channels ({channels}) must be divisible by (upscale_factor ** 3) ({upscale_factor ** 3})"
        
        new_channels = channels // (upscale_factor ** 3)
        
        # Reshape input tensor from (B, C, D, H, W) to (B, C/(r^3), D*r, H*r, W*r)
        x = x.view(batch_size, new_channels, upscale_factor, upscale_factor, upscale_factor, depth, height, width)
        
        # Permute dimensions to (B, new_C, D*r, H*r, W*r)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        
        # Merge dimensions to get the final shape
        x = x.view(batch_size, new_channels, depth * upscale_factor, height * upscale_factor, width * upscale_factor)
        
        return x

class PixelShuffle3DD(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3DD, self).__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        upscale_factor = self.upscale_factor
        
        # Ensure the number of input channels is divisible by upscale_factor
        assert channels % upscale_factor == 0, \
            f"Number of input channels ({channels}) must be divisible by upscale_factor ({upscale_factor})"
        
        new_channels = channels // upscale_factor
        
        # Reshape input tensor from (B, C, D, H, W) to (B, C/r, r, D, H, W)
        x = x.view(batch_size, new_channels, upscale_factor, depth, height, width)
        
        # Permute dimensions to (B, C/r, D*r, H, W)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # Merge dimensions to get the final shape
        x = x.view(batch_size, new_channels, depth * upscale_factor, height, width)
        
        return x


class NLOSInteRec_4(nn.Module):
    r"""The InteRec Block
    It generates the reconstructions by upsampling the temporal dimension of input features (local, global, and shallow ones). And then integrate them with 1*1*1 conv to get a 3D reconstructed volume.
    The reconstruction is based on a softArgMax. Previous experiments have validated that KL loss is good for training.
    Input:
        ch_in (int): Number of input channels, default: 32.
        input_resolution (tuple[int, int, int]): Input resulotion of H, W, and D.
    Output: 3D volume and 2D depth map with resolution of ((B, 1024)H, W).
    """

    def __init__(self, ch_in=1,channels_m=8, input_resolution=[32,32,64]):
        super().__init__()
        self.in_chans = ch_in
        self.input_resolution = input_resolution

        # upsample the temporal resolution
        self.tmup0 = PixelShuffle3DD(4)   #channels_m * 4 / 4  
        self.tmup1 = PixelShuffle3DD(4)   #channels_m * 4 / 4  
        # adjust 3D volume features
        self.ad1 = nn.Sequential(
                                    nn.Conv3d(channels_m*2,channels_m*2,3,1,1),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(channels_m * 2,channels_m * 4,3,1,1),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(channels_m*4,4**3,3,1,1),)
        self.ad2 = nn.Sequential(
                                    nn.Conv3d(channels_m*4,channels_m*4,3,1,1),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv3d(channels_m * 4,4**3,3,1,1))

        self.up = PixelShuffle3D(4)     #64 ->1 

        print("\033[0;33;40m=> WARN: InteRec block in/out channel sizes {} / {} \033[0m".format(self.in_chans, 1))

    def forward(self, inp_loc_feat, inp_glb_feat, inp_sha_feat):
        r"""
        Compute the 3D volume and 2D depth maps with three input tensors.
        Input: 
            deep local and global feature maps (B,C,D,H,W);
            shallow feature maps (B,C,D,H,W). They should have same dimension space.
        Output: 3D volume and 2D depth map with resolution of ((B, 1024)H, W).
        """
        # temporal upsample
        up_loc_feat = self.tmup0(inp_loc_feat) 
        up_glb_feat = self.tmup1(inp_glb_feat) 
        # print(up_loc_feat.shape,up_glb_feat.shape,inp_sha_feat.shape)
        # adjust1 
        ad_loc_glb = self.ad1(torch.cat([up_loc_feat,up_glb_feat],dim=1))
        # ad_loc_glb = self.ad1(up_loc_feat+up_glb_feat)
        # ad_loc_glb = self.ad1(up_loc_feat)
        # ad_inp = self.ad2(inp_sha_feat)
        # out = self.up(ad_loc_glb+ad_inp)
        out = self.up(ad_loc_glb)
        return out

    def flops(self, N):
        flops = 0
        # temporal upsampling
        flops += 3* self.tmup1.flops(N)
        flops += 3* 1* N 
        # softArgMax
        flops += N / self.input_resolution[2]* 1024

        return flops


        





class CGNL1DATTN(nn.Module):
    r"""The CGNL1DATTN block
    It computes the non-local attentions along 1D dimensions with dot production kernel.
    When groups=1, the correlations will be computed among channels.
    Input: (B, C, D)
    Output: (B, C, D)
    """
    def __init__(self, inplanes, ch_ds=False, attn_scale=1, groups=1, resid = 0):
        self.attn_scale = attn_scale
        self.groups = groups
        self.residual = resid

        super().__init__()
        self.in_chans = inplanes
        if ch_ds:
            ch_tmp = self.in_chans //4
        else:
            ch_tmp = self.in_chans
        # conv theta
        self.t = nn.Conv1d(self.in_chans, ch_tmp, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.t.weight, 0, 'fan_in', 'relu')
        # conv phi
        self.p = nn.Conv1d(self.in_chans, ch_tmp, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.p.weight, 0, 'fan_in', 'relu')
        # conv g
        self.g = nn.Conv1d(self.in_chans, ch_tmp, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.g.weight, 0, 'fan_in', 'relu')
        # conv z
        self.z = nn.Conv1d(ch_tmp, self.in_chans, kernel_size=1, stride=1, groups=self.groups, bias=False)
        init.kaiming_normal_(self.z.weight, 0, 'fan_in', 'relu')

        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=self.in_chans)
        init.constant_(self.gn.weight, 0); nn.init.constant_(self.gn.bias, 0)

        print("\033[0;33;40m=> WARN: 1D CGNL Attention block uses SCALE= {}\033[0m".format(self.attn_scale))
        if self.groups > 1:
            print("\033[0;33;40m=> WARN: Temporal CGNL block uses '{}' groups\033[0m".format(self.groups))

    def kernel(self, t, p, g, b, c, d):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            d: depth channels
        """
        t = t.view(b, 1, c * d)
        p = p.view(b, 1, c * d)
        g = g.view(b, c * d, 1)

        att = torch.bmm(p, g)* self.attn_scale

        x = torch.bmm(att, t)
        x = x.view(b, c, d)

        return x

    def forward(self, inp_q, inp_k, inp_v):
        residual = inp_v* self.residual

        t = self.t(inp_v)
        p = self.p(inp_q)
        g = self.g(inp_k)

        b, c, d = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, d)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, d)

        x = self.z(x)
        x = self.gn(x) + residual

        return x

    def flops(self, N):
        # N: number of depth dimension
        flops = 0
        # self.t(x), self.p(x), and self.g(x)
        flops += 3* self.in_chans* self.in_chans//4 * N
        # self.kernel(t, p, g, b, c, d)
        flops += self.in_chans//4 * N* 2

        return flops


class TEMUP(nn.Module):
    r"""The TEMUP block
    It consists of several deconvolution block to upsample temporal dimension of input features while keep the spatial dimension.
    Input: ch_in, Ns (scales to be upsampled), (B, C, D, H, W)
    Output: (B, 1024, H, W)
    """
    def __init__(self, ch_in, Ns=16,ch_out=1):
        super().__init__()

        self.in_chans = ch_in
        self.Ns = int(log2(Ns))

        upblk = []
        for n in range(self.Ns - 1):
            in_chans_cur = self.in_chans // (pow(2,n))
            block_ft = nn.Sequential(
                nn.ConvTranspose3d(int(in_chans_cur), int(in_chans_cur//2), kernel_size=(6,3,3), stride=(2,1,1), padding=(2,1,1),bias=False),
                nn.ReLU(inplace=True)
            )
            init.kaiming_normal_(block_ft[0].weight, 0, 'fan_in', 'relu')
            upblk.append(block_ft)

        if self.in_chans // (pow(2,self.Ns-1)) <= 1:
            chans_last = 1
        else:
            chans_last = int(self.in_chans // (pow(2,self.Ns-1)))
        block_re = nn.Sequential(
            nn.ConvTranspose3d(chans_last, ch_out, kernel_size=(6,3,3), stride=(2,1,1), padding=(2,1,1),bias=False)
        )
        init.normal_(block_re[0].weight, mean=0, std=0.001)
        upblk.append(block_re)
        self.temup = nn.Sequential(*upblk)

    def forward(self, inputs):
        r"""
        Upsample the temporal dimension of the input features.
        Input: (B, C, D, H, W)
        Output: (B, 1024, H, W)
        """

        out = self.temup(inputs)    # B,C,D,H,W
        # out = torch.squeeze(out, 1) # B,1024,H,W

        return out

    def flops(self, N):
        # N is the size of the tensor (HW or HWD)
        flops = 0
        for n in range(self.Ns - 1):
            chan_cur = self.in_chans // (pow(2,n))
            flops += chan_cur* chan_cur// 2* 54* N* pow(2,n)// 2

        flops += self.in_chans // (pow(2,self.Ns))* 54* N* 8// 2

        return flops


class TEMSPUP(nn.Module):
    r"""The TEMSPUP block
    It consists of several deconvolution block to upsample temporal and spatial dimension of input features.
    Input: ch_in, Ns (scales to be upsampled), (B, C, D, H, W)
    Output: (B, 1024, H, W)
    """

    def __init__(self, ch_in, Ns=16,ch_out=1):
        super().__init__()

        self.in_chans = ch_in
        self.Ns = int(log2(Ns))

        upblk = []
        for n in range(self.Ns - 1):
            in_chans_cur = self.in_chans // (pow(2,n))
            block_ft = nn.Sequential(
                nn.ConvTranspose3d(int(in_chans_cur), int(in_chans_cur//2), kernel_size=(6,6,6), stride=(2,2,2), padding=(2,2,2),bias=False),
                nn.ReLU(inplace=True)
            )
            init.kaiming_normal_(block_ft[0].weight, 0, 'fan_in', 'relu')
            upblk.append(block_ft)

        if self.in_chans // (pow(2,self.Ns-1)) <= 1:
            chans_last = 1
        else:
            chans_last = int(self.in_chans // (pow(2,self.Ns-1)))
        block_re = nn.Sequential(
            nn.ConvTranspose3d(chans_last, ch_out, kernel_size=(6,6,6), stride=(2,2,2), padding=(2,2,2),bias=False)
        )
        init.normal_(block_re[0].weight, mean=0, std=0.001)
        upblk.append(block_re)
        self.temup = nn.Sequential(*upblk)

    def forward(self, inputs):
        r"""
        Upsample the temporal dimension of the input features.
        Input: (B, C, D, H, W)
        Output: (B, 1024, H, W)
        """
        out = self.temup(inputs)    # B,C,D,H,W
        out = torch.squeeze(out, 1) # B,1024,H,W
        return out

    def flops(self, N):
        # N is the size of the tensor (HW or HWD)
        flops = 0
        for n in range(self.Ns - 1):
            chan_cur = self.in_chans // (pow(2,n))
            flops += chan_cur* chan_cur// 2* 54* N* pow(2,n)// 2

        flops += self.in_chans // (pow(2,self.Ns))* 54* N* 8// 2

        return flops


class MLP(nn.Module):
    """The MLP block
    The feed forward layer of the network
    Input: B, L, C
    Output: B, L, C
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, inputs):
        out_fc1 = self.fc1(inputs)
        out_act = self.act(out_fc1)
        out_drop1 = self.drop(out_act)
        out_fc2 = self.fc2(out_drop1)
        out = self.drop(out_fc2)
        return out


class WindowAttention(nn.Module):
    r"""The Window based multi-head self attention (W-MSA) module with relative position bias (better for training)
    It computes the multi-head self attention on the window sized image patches. First code the windows with relative position bias. It supports both of shifted (use the exstra mask) and non-shifted window.
    Input:
        dim (int): Number of input channels.
        dim_t (int): Number of temporal dimensions.
        window_size (tuple[int, int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Output: attention output
    """

    def __init__(self, dim, dim_t, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert window_size > 0, f"window size {window_size} should >0."
        self.dim = dim
        self.dim_t = int(np.sqrt(dim_t))
        self.window_size = to_2tuple(window_size * int(np.sqrt(dim_t)))  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros([int((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)), num_heads]))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask=None):
        """
        Input:
            inputs: input features with shape of (num_windows*B, D*window_size*window_size, C). Before using this attention block, the whole image should be put into window patches using func: window_partition. The dependency is computed independently among <num_windows*B> but within part of channel <C>. Note that we should reshape the input tensor. Note that for multi-head attention, C will be divided into several groups and correlations among C will be broken.
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        Output: out: inner dependency
        """
        B_, N, C = inputs.shape
        qkv = self.qkv(inputs).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) B_,heads,N,C//heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    #B_,heads,N,N
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1).type(torch.long)].view(
            int(self.window_size[0] * self.window_size[1]), int(self.window_size[0] * self.window_size[1]), -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(attn.shape,relative_position_bias.unsqueeze(0).shape)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, dim_t={self.dim_t}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        """The function for calculating flops for one window attention
        Input: N: the length of the token, that is: B, <N>, C
        output: calculating flops
        """
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class WindowAttentionTem(nn.Module):
    r"""The Window based multi-head self attention (W-MSA) module with relative position bias (better for training)
    It computes the multi-head self attention on the window sized image patches. First code the windows with relative position bias. It supports both of shifted (use the exstra mask) and non-shifted window.
    Input:
        dim (int): Number of input channels.
        dim_t (int): Number of temporal dimensions.
        window_size (tuple[int, int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Output: attention output
    """

    def __init__(self, dim, dim_t, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert window_size > 0, f"window size {window_size} should >0."
        self.dim = dim
        self.dim_t = int(np.sqrt(dim_t))
        self.window_size = window_size #to_2tuple(window_size * int(np.sqrt(dim_t)))  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2*self.window_size - 1, num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        relative_coords = torch.arange(window_size)[:, None] - torch.arange(window_size)[None, :]
        relative_coords += window_size - 1
        self.register_buffer("relative_position_index", relative_coords)

        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask=None):
        """
        Input:
            inputs: input features with shape of (num_windows*B, D*window_size*window_size, C). Before using this attention block, the whole image should be put into window patches using func: window_partition. The dependency is computed independently among <num_windows*B> but within part of channel <C>. Note that we should reshape the input tensor. Note that for multi-head attention, C will be divided into several groups and correlations among C will be broken.
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        Output: out: inner dependency
        """
        B_, N, C = inputs.shape
        qkv = self.qkv(inputs).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) B_,heads,N,C//heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    #B_,heads,N,N
        
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1).type(torch.long)].view(
            # int(self.window_size[0] * self.window_size[1]), int(self.window_size[0] * self.window_size[1]), -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # print(attn.shape,relative_position_bias.unsqueeze(0).shape)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class GlobalAttention(nn.Module):
    r"""The GroupAttention block => can be replaced by the nn.MultiHeadAttention()
    It is just a multi-head attention block computed without relative position bias.
    Input:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Output: attention output
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inputs):
        """
        Input:
            inputs: input features with shape of (B, H*W, C). The dependency is computed independently among <B> but within channel <C>. Note that we should reshape the input tensor.
            H, W (int): height and width of the input tensor.
        Output: out
        """
        B, N, C = inputs.shape
        qkv = self.qkv(inputs).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, n_head, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, n_head, N, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, n_head, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)   # B, N, C
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, num_head={self.num_heads}"

    def flops(self, N):
        """The function for calculating flops for attention
        Input: N: the length of the token, that is: B, <N>, C
        output: calculating flops
        """
        flops = 0
        # qkv = self.qkv(inputs)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.dim * N * N
        #  x = (attn @ v)
        flops += self.dim * N * N
        # x = self.proj(x)
        flops += N * self.dim * self.dim

        return flops




class CrossGlobalAttention(nn.Module):
    r"""The GroupAttention block => can be replaced by the nn.MultiHeadAttention()
    It is just a multi-head attention block computed without relative position bias.
    Input:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Output: attention output
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inputq, input_kv):
        # print(inputq.shape,input_kv.shape)
        """
        Input:
            inputs: input features with shape of (B, H*W, C). The dependency is computed independently among <B> but within channel <C>. Note that we should reshape the input tensor.
            H, W (int): height and width of the input tensor.
        Output: out
        """
        B, N, C = inputq.shape
        # qkv = self.qkv(inputs).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, n_head, N, head_dim
        # q, k, v = qkv[0], qkv[1], qkv[2]  # B, n_head, N, head_dim
        
        # q, k ,v = inputq, input_kv, input_kv
        
        q = self.to_q(inputq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.to_kv(input_kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, n_head, N, head_dim
        k, v = kv[0], kv[1]
        # print(q.shape,k.shape,v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, n_head, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)   # B, N, C
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, num_head={self.num_heads}"

    def flops(self, N):
        """The function for calculating flops for attention
        Input: N: the length of the token, that is: B, <N>, C
        output: calculating flops
        """
        flops = 0
        # qkv = self.qkv(inputs)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.dim * N * N
        #  x = (attn @ v)
        flops += self.dim * N * N
        # x = self.proj(x)
        flops += N * self.dim * self.dim

        return flops


class CGNL2DMHAttention(nn.Module):
    r"""The CGNL2DMHAttention block => a kind of replacement to nn.MultiHeadAttention(), which requires less GPU memory
    It is just a multi-head attention block with CGNLATTN to commpute attention.
    *Since the CGNLATTN has the codings in channel, please mind use it or not.*
    Input:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Output: attention output
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, proj_drop=0., h=16, w=16, resi=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.h = h
        self.w = w
        self.resi = resi

        # self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn = CGNL2DATTN(self.dim, attn_scale=self.scale, groups=self.num_heads, resid=self.resi)
        # self.proj = nn.Linear(dim, dim)   # can add or not
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inp_q, inp_k, inp_v):
        r"""
        Input:
            inputs: the three input features with shape of (B, H*W, C) as Q, K, and V. Note that we should reshape the input tensor.
            H, W (int): height and width of the input tensor.
        Output: out
        """
        HW, BD, C = inp_q.shape
        inter_q = inp_q.permute(1,2,0).reshape(BD, C, self.h, self.w) #BD, C, H, W
        inter_k = inp_k.permute(1,2,0).reshape(BD, C, self.h, self.w) #BD, C, H, W
        inter_v = inp_v.permute(1,2,0).reshape(BD, C, self.h, self.w) #BD, C, H, W
        out_att = self.attn(inter_q, inter_k, inter_v) # HW, BD, C
        # out = self.proj(out)
        out = self.proj_drop(out_att)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, num_head={self.num_heads}"

    def flops(self, N):
        """The function for calculating flops for attention
        Input: N: the length of the token, that is: B, <N>, C
        output: calculating flops
        """
        flops = 0
        # qkv = self.qkv(inputs)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.attn.flops(N* self.dim)

        return flops


class CGNL1DMHAttention(nn.Module):
    r"""The CGNL1DMHAttention block => a kind of replacement to nn.MultiHeadAttention(), which requires less GPU memory
    It is just a multi-head attention block with CGNLATTN to commpute attention.
    *Since the CGNLATTN has the codings in channel, please mind use it or not.*
    Input:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Output: attention output
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, proj_drop=0., resi=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.resi = resi

        self.attn = CGNL1DATTN(self.dim, attn_scale=self.scale, groups=self.num_heads, resid=self.resi)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inp_q, inp_k, inp_v):
        r"""
        Input:
            inputs: the three input features with shape of (B, H*W, C) as Q, K, and V. Note that we should reshape the input tensor.
            H, W (int): height and width of the input tensor.
        Output: out
        """

        inter_q = inp_q.permute(1,2,0) #Dkp, Datt, C
        inter_k = inp_k.permute(1,2,0) #Dkp, Datt, C
        inter_v = inp_v.permute(1,2,0) #Dkp, Datt, C
        out_att = self.attn(inter_q, inter_k, inter_v) # Datt, Dkp, C
        # out = self.proj(out)
        out = self.proj_drop(out_att)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, num_head={self.num_heads}"

    def flops(self, N):
        """The function for calculating flops for attention
        Input: N: the length of the token, that is: B, <N>, C
        output: calculating flops
        """
        flops = 0
        # qkv = self.qkv(inputs)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.attn.flops(N* self.dim)

        return flops



class WindowEncoderSep(nn.Module):
    r"""The WindowEncoder Block
    It is the transformer encoder with WindowAttention, which doesn't contain the patch embedding and positional encoding.
    The dependency is computed among H and W first and D finally.
    Formation: Norm-> Partion-> W-MSA_HW-> W-MSA_T-> Merge-> ADD-> Norm-> MLP-> ADD.
    Input:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int, int]): Input resulotion of H, W, and D.
        num_heads (int): Number of attention heads.
        window_size (int): Window images size. If >=input_resolution, dependency will be computed on whole size of measurements.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value, better for training. Default: True.
        qk_scale (float | None, optional): If set, override default qk scale as head_dim ** -0.5.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    Output: Encoded output with internal dependency, (B, C, D, H, W).
    """
    def __init__(self, dim, input_resolution, num_heads=8, window_size=8, mlp_ratio=4., 
                qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        # if the window size is larger than input resolution, don't partition the windows
        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(self.dim)
        self.attn_hw = WindowAttention(self.dim, dim_t=1, window_size=self.window_size, num_heads=self.num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=self.drop)
        # self.attn_t = WindowAttention(self.dim, dim_t=1, window_size=np.sqrt(self.input_resolution[2]),num_heads=self.num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=self.drop)
        # self.attn_t = WindowAttention(self.dim, dim_t=1, window_size=self.window_size,num_heads=self.num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=self.drop)
        # self.attn_t = WindowAttentionTem(self.dim, dim_t=1, window_size=self.window_size,num_heads=self.num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=self.drop)
        self.attn_t = WindowAttentionTem(self.dim, dim_t=1, window_size=self.input_resolution[2],num_heads=self.num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=self.drop)

        self.drop_path1 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(in_features=self.dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=self.drop)
        self.drop_path2 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        print("\033[0;33;40m=> WARN: WindowEncoder block in/out channel sizes, num_head, resolution, and window_size {} / {} / {} / {},{},{} / {} \033[0m".format(self.dim, self.dim, self.num_heads, self.input_resolution[0],self.input_resolution[1],self.input_resolution[2], self.window_size))

    def forward(self, inputs):
        r"""
        Input:
            inputs: input features with shape of (B, C, D, H, W). The dependency is computed independently among <B> and window patches but partly within channel <C>. In this seperate version, we first compute dependency among H and W and then the D.
            workflow: B,C,DHW-> nB,C,D,h,w-> nBD,hw,C-> nBhw,D,C-> B,H,W,D,C-> B,C,D,H,W
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None, defult: None.
        Output: out: encoded output with full internal dependency, (B, C, D, H, W)
        """

        H, W, D = self.input_resolution
        B, C, _, _, _ = inputs.shape
        inputs = inputs.contiguous().view(B, C, -1)
        _, _, L = inputs.shape
        assert L == H* W* D, f"{L} should be equal to {H}*{W}*{D}."

        shortcut = inputs.permute(0, 2, 1)  # B, L, C
        out_norm1 = self.norm1(inputs.permute(0, 2, 1)) # B, L, C
        out_norm1 = out_norm1.permute(0, 2, 1).reshape(B, C, D, H, W) # B, C, D, H, W

        # partition windows
        x_windows = window_partition(out_norm1, self.window_size)  # nW*B, C, D, window_size, window_size
        # W-MSA on spatial dimension
        x_windows_hw = x_windows.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.window_size * self.window_size, C)   # nW*B*D, window_size*window_size, C
        # print(x_windows_hw.shape)
        attn_windows_hw = self.attn_hw(x_windows_hw, mask=self.attn_mask)  # nW*B*D, window_size*window_size, C
        
        
        ## reshape and repartition for temporal attention
        # x_windows_t = out_norm1.contiguous().view(B,C,D//self.window_size,self.window_size,H,W)
        x_windows_t = out_norm1.contiguous().view(B,C,D,H,W)
        x_windows_t = x_windows_t.permute(0,3,4,2,1).contiguous().view(-1,D,C) #HW*nD, windowsize, C

        # W-MSA on temporal dimension
        attn_windows_t = self.attn_t(x_windows_t, mask=self.attn_mask)    # nW*B*window_size*window_size, D, C

        # print(attn_windows_t.shape)
        # merge windows, (num_windows*B, C, D, window_size, window_size)
        # attn_windows = attn_windows_t.contiguous().view(-1, self.window_size* self.window_size, D, C).permute(0,3,2,1).contiguous().view(-1,C,D,self.window_size,self.window_size)    # (nW*B, C, D, window_size, window_size)
        # out_wmsa = window_reverse(attn_windows, H, W).permute(0,2,3,4,1)  # B, D, H, W, C
        
        out_wmsa_hw = attn_windows_hw.contiguous().view(B,H//self.window_size,W//self.window_size,D, self.window_size, self.window_size, C)
        out_wmsa_hw = out_wmsa_hw.permute(0,3,1,4,2,5,6).contiguous().view(B,D,H,W,C).contiguous().view(B, D*H*W, C)


        # out_wmsa_t = attn_windows_t.contiguous().view(B,H,W,D//self.window_size,self.window_size,C).contiguous().view(B,H,W,D,C).permute(0,3,1,2,4)
        out_wmsa_t = attn_windows_t.contiguous().view(B,H,W,D,C).permute(0,3,1,2,4)
        out_wmsa_t = out_wmsa_t.contiguous().view(B, D*H*W, C)  # B, L, C
        # print(out_wmsa_hw.shape,out_wmsa_t.shape)
        out_wmsa = out_wmsa_hw + out_wmsa_t
        # FFN
        out_res = shortcut + self.drop_path1(out_wmsa)
        out_ffn = out_res + self.drop_path2(self.mlp(self.norm2(out_res)))
        out = out_ffn.permute(0, 2, 1).reshape(B, C, D, H, W)

        return out
        
    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, N):
        flops = 0
        # norm1
        flops += self.dim * N
        # W-MSA_HW and W-MSA_T
        nW = N / self.window_size / self.window_size / self.input_resolution[2]
        flops += nW * (self.attn_hw.flops(self.window_size * self.window_size) + self.attn_t.flops(D))
        # mlp (it has two linear layers)
        flops += 2 * N * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * N

        return flops



class GlobalEncoderSep(nn.Module):
    r"""The GlobalEncoderSep Block
    It is the transformer encoder with GlobalAttention, which doesn't contain the patch embedding and positional encoding.
    The dependency is computed among H and W first and D finally.
    Formation: Norm-> W-MSA_HW-> W-MSA_T-> ADD-> Norm-> MLP-> ADD.
    Input:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int, int]): Input resulotion of H, W, and D.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value, better for training. Default: False.
        qk_scale (float | None, optional): If set, override default qk scale as head_dim ** -0.5.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic drop rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    Output: Encoded output with internal dependency, (B, C, D, H, W).
    """

    def __init__(self, dim, input_resolution, num_heads=8, mlp_ratio=4, 
                qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop = drop

        self.norm1 = norm_layer(self.dim)
        self.attn_hw = GlobalAttention(self.dim, self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=self.drop)
        self.attn_t = GlobalAttention(self.dim, self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=self.drop)
        
        self.drop_path1 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(self.dim, mlp_hidden_dim, act_layer=act_layer, drop=self.drop)
        self.drop_path2 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        print("\033[0;33;40m=> WARN: GlobalEncoderSep block in/out channel sizes, num_head and resolution {} / {} / {} / {},{},{}, \033[0m".format(self.dim, self.dim, self.num_heads, self.input_resolution[0],self.input_resolution[1],self.input_resolution[2]))

    def forward(self, inputs):
        r"""
        Input:
            inputs: input features with shape of (B, C, D, H, W). The dependency is computed independently among <B> and window patches but partly within channel <C>. In this seperate version, we first compute dependency among H and W and then the D.
            workflow: B,C,DHW-> nB,C,D,h,w-> nBD,hw,C-> nBhw,D,C-> B,H,W,D,C-> B,C,D,H,W
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None, defult: None.
        Output: out: encoded output with full internal dependency, (B, C, D, H, W)
        MIND THE SHAPE CHANGES!!!
        """

        H, W, D = self.input_resolution
        B, C, _, _, _ = inputs.shape
        inputs = inputs.contiguous().view(B, C, -1)
        _, _, L = inputs.shape
        assert L == H* W* D, f"{L} should be equal to {H}*{W}*{D}."

        shortcut = inputs.permute(0, 2, 1)  # B, L, C
        out_norm1 = self.norm1(inputs.permute(0,2,1)).reshape(B, D, H* W, C)   # B, D, HW, C

        # MSA on spatial dimension
        out_normhw = out_norm1.contiguous().view(-1, H* W, C)
        out_attn_hw = self.attn_hw(out_normhw)   # BD, HW, C
        out_attn_hw = out_attn_hw.contiguous().view(B,-1, C)

        # MSA on temporal dimension

        out_normt = out_norm1.permute(0,2,1,3).contiguous().view(-1, D, C)
        out_attn_t = self.attn_t(out_normt)   # BHW, D, C
        out_attn_t = out_attn_t.reshape(B, H* W, D, C).contiguous().view(B, -1, C)  # B, L, C

        out_attn = out_attn_hw + out_attn_t
        # FFN
        out_res = shortcut + self.drop_path1(out_attn)
        out_ffn = out_res + self.drop_path2(self.mlp(self.norm2(out_res)))   # B, L, C
        out = out_ffn.reshape(B, H, W, D, C).permute(0,4,3,1,2) # B,C, D, H, W

        return out

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_head={self.num_heads}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W, D = self.input_resolution
        # norm1
        flops += self.dim * H* W* D
        # MSA_HW and MSA_T
        flops += self.attn_hw.flops(H* W) + self.attn_t.flops(D)
        # mlp
        flops += 2* H* W* D* self.dim* self.dim* self.mlp_ratio
        # norm2
        flops += self.dim* H* W* D
        
        return flops


class CrossDecoder(nn.Module):
    r"""The CrossDecoder block
    It consists of several CGNL1DMHAttention to generate high integrated features.
    It is the one block and will integrate information with others
    Input:
        ch_in (int): Number of input channels.
        input_resolution (tuple[int, int, int]): Input resulotion of H, W, and D.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.1
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    Output: Integrated output with resolution of (B, C, D, H, W).
    """

    def __init__(self, ch_in=64, input_resolution=[32,32,64], num_heads=8, res_con = 0,
                mlp_ratio=4, mlp_drop=0., attn_drop=0.1, 
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_chans = ch_in
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropnum = mlp_drop
        self.resid = res_con

        # decoder block (we can use built-in MHSA to replace it)
        self.norm1 = norm_layer(self.in_chans)
        self.norm12 = norm_layer(self.in_chans)
        self.attn_hw = CGNL1DMHAttention(self.in_chans, self.num_heads, resi=self.resid)
        self.attn_t = CGNL1DMHAttention(self.in_chans, self.num_heads, resi=self.resid)
        self.drop1 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(self.in_chans)
        mlp_hidden_dim_loc = int(self.in_chans* self.mlp_ratio)
        self.mlp = MLP(self.in_chans, mlp_hidden_dim_loc,act_layer=act_layer, drop=self.dropnum)
        self.drop2 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        print("\033[0;33;40m=> WARN: CrossDecoder block in/out channel sizes, num_head and resolution {} / {} / {} / {},{},{} \033[0m".format(self.in_chans, self.in_chans, self.num_heads, self.input_resolution[0],self.input_resolution[1],self.input_resolution[2]))

    def forward(self, inpt_kv, inpt_q):
        r"""
        Input: two input features with shape of (B, C, D, H, W).
        Output: integrated features with shape of (B, C, D, H, W).
        """

        B, C, D, H, W = inpt_kv.shape
        shortcut = inpt_kv.reshape(B, C, D*H*W).permute(0, 2, 1) # B, L, C

        # decoder for spatial and temporal independently
        k = v = self.norm1(inpt_kv.permute(0,2,3,4,1)).reshape(B, D, H* W, C)  # B, D, HW, C
        q = self.norm12(inpt_q.permute(0,2,3,4,1)).reshape(B, D, H* W, C)  # B, D, HW, C
        q_in_hw = q.contiguous().view(-1, H* W, C).permute(1,0,2) # HW, BD, C
        q_in_t = q.permute(1,0,2,3).contiguous().view(D, -1, C) # D,BHW,C
        k_in = k.contiguous().view(-1, H* W, C).permute(1,0,2) # HW, BD, C
        v_in = v.contiguous().view(-1, H* W, C).permute(1,0,2) # HW, BD, C
        attn_hw = self.attn_hw(q_in_hw, k_in, v_in) # HW, BD, C
        attn_hw = attn_hw.reshape(H* W, B, D, C).contiguous().view(-1, D, C).permute(1, 0, 2) # D, BHW, C
        attn_t = self.attn_t(q_in_t, attn_hw, attn_hw) # D, BHW, C
        attn = attn_t.reshape(D, B, H, W, C).permute(1,0,2,3,4).contiguous().view(B, -1, C) # B, L, C

        # FFN
        out_drop = shortcut + self.drop1(attn)
        out_ffn = out_drop + self.drop2(self.mlp(self.norm2(out_drop)))
        out = out_ffn.reshape(B, D, H, W, C).permute(0,4,1,2,3) # B, C, D, H, W

        return out

    def flops(self, N):
        flops = 0
        H, W, D = self.input_resolution
        # local and global attension
        flops += 2* self.in_chans* H* W* D
        flops += 2* (4* H*W* pow(self.in_chans,2)+ pow(H*W,2)*self.in_chans) + (4* D* pow(self.in_chans,2)+ pow(D,2)*self.in_chans)
        flops += 2* 2* H* W* D* self.in_chans* self.in_chans* self.mlp_ratio
        flops += 2* self.in_chans* H* W* D

        return flops




class SPUP(nn.Module):
    """The SPUP block
    Upsample spatial dim and keep the temporal dimension
    Input: B, C, D, H, W
    Output: B, C, D, H*up, W*up (default)
    """

    def __init__(self, ch_in, up=2):
        super().__init__()

        self.in_chans = ch_in
        self.up = int(log2(up))

        upblk = []
        for n in range(self.up):
            sp_upblk = nn.Sequential(
            nn.ConvTranspose3d(self.in_chans, self.in_chans, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.ReLU(inplace=True)
            )
            init.kaiming_normal_(sp_upblk[0].weight, 0, 'fan_in', 'relu')
            upblk.append(sp_upblk)
        
        self.upop = nn.Sequential(*upblk)

    def forward(self, inputs):

        out = self.upop(inputs)

        return out

    def flops(self, N):
        flops = 0
        # self.upop(inputs)
        flops += self.in_chans* self.in_chans* 108* N

        return flops
        