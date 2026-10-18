"""TRT-NLOS: light denoiser followed by the NLOST reconstruction backbone.

The legacy NLOST class and submodule names are retained for checkpoint compatibility.
"""
from torch import nn
from .nlost_modules import (
    LightDenoisingNet, Transient_TDown_2, MsFeat_2, Transient_TDown_3,
    DsFusion, SPDS, PosiEncCNN, modelClone, WindowEncoderSep,
    GlobalEncoderSep, LocGlbInteNBlks_LCGC_l1d2, NLOSInteRec_2, VisbleNet,
)
from .modules import Rendering
from .utils_pytorch import fk_1_10


class NLOST(nn.Module):
    """TRT-NLOS model returning volume, intensity, depth and denoised transient."""
    def __init__(self, ch_in=1, num_coders=1,spatial=128,tlen=256,bin_len=0.01,target_size=256):
        super().__init__()
        if num_coders < 1:
            raise ValueError('num_coders must be positive')
        if target_size not in (spatial, 2 * spatial):
            raise ValueError('target_size must equal spatial or twice spatial')
        self.sa = target_size // spatial
        assert self.sa in [1,2]
        self.in_chans = ch_in
        self.coders = num_coders
        self.spatial = spatial
        self.tlen = tlen
        self.bin_len = bin_len
        if self.sa==2:
            ts_stride = [2,2,2]    ## convlutional stride across the temporal and spatial dimensions
        else:
            ts_stride = [2,1,1]
        self.light_denoise = LightDenoisingNet(ch_in,ch_in)
        self.sig_expand = Transient_TDown_2(ch_in,2,ts_stride)
        self.tra2vlo = fk_1_10.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len*2,dnum=self.in_chans*4)

        channels_m = 8
        self.msfeat = MsFeat_2(self.in_chans*4, channels_m) #B,32,128,H,W
        self.sig_feat = Transient_TDown_3(channels_m * 4,channels_m * 4)

        self.dsfusion = DsFusion(channels_m * 4, channels_m * 4)    #B,16,64,H,W

        self.spds = SPDS(channels_m * 4,ds=2)
        self.posenc_l = PosiEncCNN(channels_m * 4)
        self.posenc_g = PosiEncCNN(channels_m * 4)

        self.loc_encds = modelClone(WindowEncoderSep(dim=channels_m * 4,input_resolution=[self.spatial,self.spatial,16],num_heads=4,window_size=self.spatial//2), self.coders)
        self.glb_encds = modelClone(GlobalEncoderSep(dim=channels_m * 4,input_resolution=[self.spatial//2,self.spatial//2,16],num_heads=4), self.coders)
        self.locglb_inte = LocGlbInteNBlks_LCGC_l1d2(channels_m * 4, [self.spatial,self.spatial,16],8, self.coders)
        self.inte_rec = NLOSInteRec_2(self.in_chans * 4,channels_m)        #  64
        self.project = VisbleNet()
        self.inten_refine = Rendering(nf0=self.in_chans * 8, out_channels=1,factor=self.sa)
        self.dep_refine = Rendering(nf0=self.in_chans * 8, out_channels=1,factor=self.sa,isdep=True)


    def normalize(self, data_bxcxdxhxw):   #  min max scaling
        b, c, d, h, w = data_bxcxdxhxw.shape
        data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
        data_min = data_bxcxk.min(2, keepdim=True)[0]
        data_zmean = data_bxcxk - data_min
        data_max = data_zmean.max(2, keepdim=True)[0]
        data_norm = data_zmean / (data_max + 1e-15)
        return data_norm.view(b, c, d, h, w)


    def forward(self, inputs):
        expected = (self.in_chans, 2 * self.tlen, self.spatial * self.sa, self.spatial * self.sa)
        if inputs.ndim != 5 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(f'Expected [B, {expected}], got {tuple(inputs.shape)}')
        inputs_denoise = self.light_denoise(inputs)
        inputs_norm = self.normalize(inputs_denoise)             # b 1 512 h w
        sigal_expand = self.sig_expand(inputs_norm)  # b 4 256 128 128
        vlo = self.tra2vlo(sigal_expand, [0] * inputs.shape[0], [self.tlen] * inputs.shape[0])  # b 4 256 128 128
        vlo = nn.ReLU()(vlo)
        vlo = self.normalize(vlo)

        msfeat = self.msfeat(vlo)
        sig_feat = self.sig_feat(msfeat)
        patem_l = self.dsfusion(sig_feat)
        patem_s = self.spds(patem_l)
        posenc_l = self.posenc_l(patem_l)
        for loc_encd in self.loc_encds:
            winenc = loc_encd(posenc_l)
            posenc_l = winenc
        posenc_g = self.posenc_g(patem_s)
        for glb_encd in self.glb_encds:
            glbenc = glb_encd(posenc_g)
            posenc_g = glbenc

        inte_loc, inte_glb = self.locglb_inte(winenc, glbenc)
        out_vlo = self.inte_rec(inte_loc, inte_glb, msfeat)

        raw = self.project(out_vlo)
        refine_intensity = self.inten_refine(raw)
        refine_dep = self.dep_refine(raw)

        refine_intensity = refine_intensity * 2 - 1
        refine_dep = refine_dep * 2 - 1
        return out_vlo, refine_intensity, refine_dep, inputs_denoise
