

import torch
import torch.nn as nn
import torch.nn.functional as F
import  numpy as np


class lct_fk(nn.Module):

    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2, dnum = 2,align_corners=False):
        super(lct_fk, self).__init__()

        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop

        self.bin_len = bin_len
        self.wall_size = wall_size

        self.align_corners = align_corners

        self.parpareparam()

        self.register_buffer("gridz_1xMx1x1_todev", self.gridz_1xMx1x1, persistent=False)
        gridz_square_1xMx1x1 = self.gridz_1xMx1x1_todev ** 2
        self.register_buffer("gridz_square_1xMx1x1", gridz_square_1xMx1x1, persistent=False)
        datapad_Dx2Tx2Hx2W = torch.zeros((dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32)
        self.register_buffer("datapad_Dx2Tx2Hx2W", datapad_Dx2Tx2Hx2W, persistent=False)
        self.register_buffer("newcoord_dx2Mx2Nx2Nx3_todev", self.newcoord_1x2Mx2Nx2Nx3.repeat(dnum, 1, 1, 1, 1), persistent=False)
        self.register_buffer("gridz_2Mx2Nx2N_todev", self.gridz_2Mx2Nx2N, persistent=False)
        self.register_buffer("gridznew_todev", self.gridznew, persistent=False)


    def change_bin_len(self, bin_len):
        print('change bin_len from %f to %f' % (self.bin_len, bin_len))

        self.bin_len = bin_len
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution

        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        M = temprol_grid
        N = sptial_grid

        fkrange = ((N * self.trange) / (M * self.width * 4)) ** 2
        gridznew = fkrange * self.gridxy_change + self.gridz_change
        gridznew = np.sqrt(gridznew)
        self.gridznew = torch.from_numpy(gridznew)

        newsame_1x2Mx2Nx2Nx1 = self.gridznew.unsqueeze(0).unsqueeze(4)
        newx = self.gridx_2Mx2Nx2N.unsqueeze(0).unsqueeze(4)
        newy = self.gridy_2Mx2Nx2N.unsqueeze(0).unsqueeze(4)
        self.newcoord_1x2Mx2Nx2Nx3 = torch.cat([newx, newy, newsame_1x2Mx2Nx2Nx1], dim=4)

        dnum = self.newcoord_dx2Mx2Nx2Nx3_todev.shape[0]
        dev = self.newcoord_dx2Mx2Nx2Nx3_todev.device
        self.newcoord_dx2Mx2Nx2Nx3_todev = self.newcoord_1x2Mx2Nx2Nx3.to(dev).repeat(dnum, 1, 1, 1, 1)
        self.gridznew_todev = self.gridznew.to(dev)

    def parpareparam(self,):

        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution

        temprol_grid = self.crop
        sptial_grid = self.spatial_grid

        gridz_M = np.arange(temprol_grid, dtype=np.float32)
        gridz_M = gridz_M / (temprol_grid - 1)
        gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))

        zdim = np.arange(2 * temprol_grid, dtype=np.float32)
        xdim = np.arange(2 * sptial_grid, dtype=np.float32)

        zdim = (zdim - temprol_grid) / temprol_grid
        xdim = (xdim - sptial_grid) / sptial_grid
        ydim = xdim

        [gridy_2Nx2Nx2M, gridx_2Nx2Nx2M, gridz_2Nx2Nx2M] = np.meshgrid(xdim, ydim, zdim)
        gridz_2Mx2Nx2N = np.transpose(gridz_2Nx2Nx2M, [2, 1, 0])
        gridy_2Mx2Nx2N = np.transpose(gridy_2Nx2Nx2M, [2, 1, 0])
        gridx_2Mx2Nx2N = np.transpose(gridx_2Nx2Nx2M, [2, 1, 0])

        '''
        print(gridz_2Mx2Nx2N[:, 0, 0])
        print(gridy_2Mx2Nx2N[0, :, 0])
        print(gridx_2Mx2Nx2N[0, 0, :])
        '''

        self.gridz_2Mx2Nx2N = torch.from_numpy(gridz_2Mx2Nx2N)
        self.gridy_2Mx2Nx2N = torch.from_numpy(gridy_2Mx2Nx2N)
        self.gridx_2Mx2Nx2N = torch.from_numpy(gridx_2Mx2Nx2N)

        self.gridxy_change = gridx_2Mx2Nx2N ** 2 + gridy_2Mx2Nx2N ** 2
        self.gridz_change = gridz_2Mx2Nx2N ** 2

        M = temprol_grid
        N = sptial_grid

        fkrange = ((N * self.trange) / (M * self.width * 4)) ** 2
        gridznew = fkrange * self.gridxy_change + self.gridz_change
        gridznew = np.sqrt(gridznew)
        self.gridznew = torch.from_numpy(gridznew)

        newsame_1x2Mx2Nx2Nx1 = self.gridznew.unsqueeze(0).unsqueeze(4)
        newx = self.gridx_2Mx2Nx2N.unsqueeze(0).unsqueeze(4)
        newy = self.gridy_2Mx2Nx2N.unsqueeze(0).unsqueeze(4)
        self.newcoord_1x2Mx2Nx2Nx3 = torch.cat([newx, newy, newsame_1x2Mx2Nx2Nx1], dim=4)

        self.xdim = xdim
        self.zdim = zdim
        self.z0pos = np.where(zdim > 0)[0][0]
        print('zzeropos %d' % self.z0pos)


    def roll_1(self, x_bxtxhxwx2, dim, n):
        if dim == 1:
            a = torch.cat((x_bxtxhxwx2[:, -n:], x_bxtxhxwx2[:, :-n]), dim=dim)
        if dim == 2:
            a = torch.cat((x_bxtxhxwx2[:, :, -n:], x_bxtxhxwx2[:, :, :-n]), dim=dim)
        if dim == 3:
            a = torch.cat((x_bxtxhxwx2[:, :, :, -n:], x_bxtxhxwx2[:, :, :, :-n]), dim=dim)
        return a

    def forward(self, feture_bxdxtxhxw, tbes=[0,0,0], tens=[256,256,256]):

        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        for tbe, ten in zip(tbes, tens):
            assert tbe >= 0
            assert ten <= self.crop
        dev = feture_bxdxtxhxw.device

        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.crop - tens[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)

        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop

        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, self.crop, hnum, wnum)

        gridz_square_1xMx1x1 = self.gridz_square_1xMx1x1
        data_BDxTxHxW = data_BDxTxHxW * gridz_square_1xMx1x1

        data_BDxTxHxW = F.relu(data_BDxTxHxW, inplace=False)
        data_BDxTxHxW = torch.sqrt(data_BDxTxHxW)

        datapad_Dx2Tx2Hx2W = self.datapad_Dx2Tx2Hx2W
        datapad_BDx2Tx2Hx2W = datapad_Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)

        datapad_BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = data_BDxTxHxW

        datafre_temp = torch.fft.fftn(datapad_BDx2Tx2Hx2W, dim = (-3,-2,-1))
        datafre_BDX2Tx2Hx2Wx2 = torch.stack((datafre_temp.real, datafre_temp.imag), -1)

        datafre_BDX2Tx2Hx2Wx2 = self.roll_1(datafre_BDX2Tx2Hx2Wx2, dim=1, n=temprol_grid)
        datafre_BDX2Tx2Hx2Wx2 = self.roll_1(datafre_BDX2Tx2Hx2Wx2, dim=2, n=sptial_grid)
        datafre_BDX2Tx2Hx2Wx2 = self.roll_1(datafre_BDX2Tx2Hx2Wx2, dim=3, n=sptial_grid)


        if True:
            datafre_BDx2x2Hx2Wx2T = datafre_BDX2Tx2Hx2Wx2.permute(0, 4, 1, 2, 3)
            '''
            size = datafre_BDx2x2Hx2Wx2T.shape
            theta = torch.from_numpy(np.eye(3, 4, dtype=np.float32)).unsqueeze(0)
            gridstmp = F.affine_grid(theta, size, align_corners=self.align_corners)
            x = gridstmp[:, :, :, :, 0:1]
            y = gridstmp[:, :, :, :, 1:2]
            z = gridstmp[:, :, :, :, 2:3]
            '''
            newcoord_BDx2Mx2Nx2Nx3 = self.newcoord_dx2Mx2Nx2Nx3_todev.repeat(bnum, 1, 1, 1, 1)
            if True:
                datafrenew = F.grid_sample(datafre_BDx2x2Hx2Wx2T, newcoord_BDx2Mx2Nx2Nx3, \
                                           mode='bilinear', padding_mode='zeros', \
                                           align_corners=self.align_corners)

            else:
                datafrenew = F.grid_sample(datafre_BDx2x2Hx2Wx2T, newcoord_BDx2Mx2Nx2Nx3, \
                                       mode='bilinear', padding_mode='zeros')
            tdata_BDx2Tx2Hx2Wx2 = datafrenew.permute(0, 2, 3, 4, 1)
            tdata_BDx2Tx2Hx2Wx2 = tdata_BDx2Tx2Hx2Wx2.contiguous()
        else:
            import scipy.interpolate as si
            zdim = self.zdim
            xdim = self.xdim
            ydim = xdim

            gridznew = self.gridznew.numpy()
            gridy_2Mx2Nx2N = self.gridy_2Mx2Nx2N.numpy()
            gridx_2Mx2Nx2N = self.gridx_2Mx2Nx2N.numpy()

            datafre_bdxtxhxwx2 = datafre_BDX2Tx2Hx2Wx2.detach().cpu().numpy()
            datafre_bdxtxhxw = datafre_bdxtxhxwx2[:, :, :, :, 0] + 1j * datafre_bdxtxhxwx2[:, :, :, :, 1]

            re = []
            for datafre in datafre_bdxtxhxw:
                tvol = si.interpn(points=(zdim, ydim, xdim), values=datafre, \
                          xi=np.stack([gridznew, gridy_2Mx2Nx2N, gridx_2Mx2Nx2N], axis=3), \
                          method='linear', bounds_error=False, fill_value=0)
                re.append(tvol)

            re_bdxtxhxw = np.stack(re)
            re_real_bdxtxhxw = np.real(re_bdxtxhxw)
            re_imag_bdxtxhxw = np.imag(re_bdxtxhxw)

            re_real_bdxtxhxw = torch.from_numpy(re_real_bdxtxhxw).to(dev)
            re_imag_bdxtxhxw = torch.from_numpy(re_imag_bdxtxhxw).to(dev)
            tdata_BDx2Tx2Hx2Wx2 = torch.stack([re_real_bdxtxhxw, re_imag_bdxtxhxw], dim=4)

        samplez_1xMxNxNx1 = self.gridz_2Mx2Nx2N_todev.unsqueeze(0).unsqueeze(4)
        sampleznew = self.gridznew_todev.unsqueeze(0).unsqueeze(4)

        tdata_BDx2Tx2Hx2Wx2[:, :self.z0pos, :, :, :] = 0
        tdata_BDx2Tx2Hx2Wx2 = tdata_BDx2Tx2Hx2Wx2 * samplez_1xMxNxNx1.abs()

        tdata_BDx2Tx2Hx2Wx2 = tdata_BDx2Tx2Hx2Wx2 / (sampleznew + 1e-8)

        tdata_BDx2Tx2Hx2Wx2 = self.roll_1(tdata_BDx2Tx2Hx2Wx2, dim=1, n=temprol_grid)
        tdata_BDx2Tx2Hx2Wx2 = self.roll_1(tdata_BDx2Tx2Hx2Wx2, dim=2, n=sptial_grid)
        tdata_BDx2Tx2Hx2Wx2 = self.roll_1(tdata_BDx2Tx2Hx2Wx2, dim=3, n=sptial_grid)

        data_pre = torch.complex(tdata_BDx2Tx2Hx2Wx2[...,0],tdata_BDx2Tx2Hx2Wx2[...,1])
        data_temp = torch.fft.ifftn(data_pre,dim=(-3,-2,-1))
        data = torch.stack((data_temp.real, data_temp.imag), -1)

        data = data[:, :temprol_grid, :sptial_grid, :sptial_grid]
        data = data[:, :, :, :, 0] ** 2 + data[:, :, :, :, 1] ** 2;

        volumn_BDxTxHxW = data.view(bnum * dnum, self.crop, hnum, wnum)

        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)

        return volumn_BxDxTxHxW

