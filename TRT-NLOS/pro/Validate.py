import numpy as np
import torch
from tqdm import tqdm
import utils.SetDistTrain as utils
from pro.Loss import criterion_L2

from metric import RMSE, PSNR, SSIM
items = ["ALL", "int", "dep"]
metric_list = ['rmse', 'psnr', 'ssim']

def validate(model, val_loader, n_iter, val_loss, params, logWriter,val_metrics):

    device = torch.device(params.device)
    rmse = RMSE().to(device)
    psnr = PSNR().to(device)
    ssim = SSIM().to(device)
    model.eval()
    for meter in val_metrics.values():
        meter.reset()
    l_all = []
    l_l2 = []
    l_l2_dep = []
    l_kl = []

    for sample in tqdm(val_loader):
        raw_meas = sample["raw_meas"].to(device)
        M_mea = sample["ds_meas"].to(device)
        dep_gt = sample["dep_gt"].to(device)
        img_gt = sample["img_gt"].to(device)
        _, int_re,dep_re,meas_denoise = model(M_mea)
        inten_re = (int_re + 1) / 2
        dep_re = (dep_re + 1) / 2

        loss_l2 = criterion_L2(inten_re, img_gt).data.cpu().numpy()
        loss_l2_dep = criterion_L2(dep_re, dep_gt).data.cpu().numpy()
        measurement_loss = criterion_L2(meas_denoise, raw_meas).data.cpu().numpy()

        loss = loss_l2 + loss_l2_dep + 10 * measurement_loss
        l_all.append(loss)
        l_l2.append(loss_l2)
        l_l2_dep.append(loss_l2_dep)
        l_kl.append(measurement_loss)

        pred = torch.clamp(inten_re.detach(), 0, 1)
        target = torch.clamp(img_gt, 0, 1)
        metric_dict = {
            'rmse': rmse(pred, target).cpu(),
            'psnr': psnr(pred, target).cpu(),
            'ssim': ssim(pred, target).cpu(),
        }
        for k in metric_list:
            val_metrics[k].update(metric_dict[k].item(), M_mea.shape[0])
    if utils.is_main_process():
        logWriter.add_scalar("loss_val/all", np.mean(l_all), n_iter)
        logWriter.add_scalar("loss_val/l2", np.mean(l_l2), n_iter)
        logWriter.add_scalar("loss_val/l2_dep", np.mean(l_l2_dep), n_iter)
        logWriter.add_scalar("loss_val/l_kl", np.mean(l_kl), n_iter)

        logWriter.add_images("inten_rec", inten_re.clamp(0,1), n_iter,dataformats="NCHW")
        logWriter.add_images("inten_gt", img_gt, n_iter,dataformats="NCHW")
        logWriter.add_images("dep_rec", dep_re.clamp(0,1), n_iter,dataformats="NCHW")
        logWriter.add_images("dep_gt", dep_gt, n_iter,dataformats="NCHW")

        for k in metric_list:
            logWriter.add_scalars(k, {'test': val_metrics[k].item()}, n_iter)
        val_loss[items[0]].append(np.mean(l_all))
        val_loss[items[1]].append(np.mean(l_l2))
        val_loss[items[2]].append(np.mean(l_l2_dep))
    return val_loss, logWriter, val_metrics


