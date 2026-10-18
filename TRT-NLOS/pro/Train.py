import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import scipy.io as scio
from pro.Validate import validate
from utils.SaveChkp import save_checkpoint
import utils.SetDistTrain as utils
from pro.Loss import criterion_L2
import logging

from metric import RMSE, PSNR, SSIM, AverageMeter
cudnn.benchmark = True
items = ["ALL", "int", "dep"]

metric_list = ['rmse', 'psnr', 'ssim']
train_metrics = {k: AverageMeter() for k in metric_list}
val_metrics = {k: AverageMeter() for k in metric_list}


def train(model, train_loader, val_loader, optimer, epoch, n_iter,
            train_loss, val_loss, params, logWriter):
    device = torch.device(params.device)
    rmse = RMSE().to(device)
    psnr = PSNR().to(device)
    ssim = SSIM().to(device)
    for meter in train_metrics.values():
        meter.reset()
    for sample in tqdm(train_loader):
        model.train()
        raw_meas = sample["raw_meas"].to(device)
        M_mea = sample["ds_meas"].to(device)
        dep_gt = sample["dep_gt"].to(device)
        img_gt = sample["img_gt"].to(device)

        _, inten_re, dep_re, meas_denoise = model(M_mea)
        inten_re = (inten_re + 1) / 2
        dep_re = (dep_re + 1) / 2

        measurement_loss = criterion_L2(meas_denoise, raw_meas)
        loss_l2 = criterion_L2(inten_re, img_gt)
        loss_l2_dep = criterion_L2(dep_re, dep_gt)
        loss =  measurement_loss + loss_l2 + loss_l2_dep
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        n_iter += 1

        metric_dict = {
            'rmse': rmse(inten_re.detach(), img_gt).item(),
            'psnr': psnr(inten_re.detach(), img_gt).item(),
            'ssim': ssim(inten_re.detach(), img_gt).item(),
        }
        for k in metric_list:
            train_metrics[k].update(metric_dict[k], M_mea.shape[0])

        if utils.is_main_process():
            logWriter.add_scalar("loss_train/all", loss, n_iter)
            logWriter.add_scalar("loss_train/measurement_rmse", measurement_loss, n_iter)
            logWriter.add_scalar("loss_train/l2", loss_l2, n_iter)
            logWriter.add_scalar("loss_train/l2_dep", loss_l2_dep, n_iter)

            for k in metric_list:
                logWriter.add_scalars(k, {'train': train_metrics[k].item()}, n_iter)
            train_loss[items[0]].append(loss.data.cpu().numpy())
            train_loss[items[1]].append(loss_l2.data.cpu().numpy())
            train_loss[items[2]].append(loss_l2_dep.data.cpu().numpy())

        if n_iter % params.num_save == 0:
            logging.info("Sart validation...")
            with torch.no_grad():
                val_loss, logWriter, val_metric = validate(model, val_loader, n_iter, val_loss, params, logWriter,val_metrics)
                log_str = 'test {} | '.format(n_iter)
                for k in metric_list:
                    log_str += '{:s} {:.5f} | '.format(k, val_metric[k].item())
                logging.info(log_str)

            if utils.is_main_process():
                scio.savemat(file_name=params.model_dir+"/train_loss.mat", mdict=train_loss)
                scio.savemat(file_name=params.model_dir+"/val_loss.mat", mdict=val_loss)
                logging.info("Validation complete!")
                save_checkpoint(n_iter, epoch, model, optimer,
                file_path=params.model_dir+"/epoch_{}_{}.pth".format(epoch, n_iter))
                logging.info("Checkpoint saved!")

    if n_iter % params.num_save != 0:
        with torch.no_grad():
            val_loss, logWriter, _ = validate(model, val_loader, n_iter, val_loss, params, logWriter, val_metrics)
    return model, optimer, n_iter, train_loss, val_loss, logWriter,train_metrics,val_metrics

