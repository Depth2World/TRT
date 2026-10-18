import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils.LFEDataset import LFEDataset
from utils.ParseArgs import parse_args
from utils.SaveChkp import save_checkpoint
import utils.SetDistTrain as utils
from pro.Train import train
import logging
from models import nlost


cudnn.benchmark = True

def main():

    opt = parse_args()
    utils.init_distributed_mode(opt)
    if not torch.cuda.is_available():
        raise RuntimeError('TRT-NLOS training requires CUDA.')
    device = torch.device('cuda', opt.loc_rank)
    torch.cuda.set_device(device)
    opt.device = str(device)

    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler(opt.model_dir + '/train.log' ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )
    logging.info('='*80)
    logging.info(f'Start of experiment: {opt.model_name}')
    logging.info('='*80)

    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    logging.info(opt)
    logging.info("Current main process GPUs: {}".format((opt.loc_rank)))
    logging.info("Number of available GPUs: {} {}".format(torch.cuda.device_count(), \
        torch.cuda.get_device_name(torch.cuda.current_device())))
    logging.info("Number of Encoder-Decoders: {}".format(opt.num_coders))
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")

    logging.info("Loading training and validation data...")
    seed = opt.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    folder_path = [opt.data_dir]
    shineness = [0]
    logging.info(folder_path[0])
    train_data = LFEDataset(root=folder_path,  # dataset root directory
                            shineness=shineness,
                            for_train=True,
                            ds=1,  # temporal down-sampling factor
                            clip=512,  # time range of histograms
                            size=256,  # measurement size (unit: px)
                            scale=1,  # scaling factor (float or float tuple)
                            background=[0.05, 2],  # background noise rate (float or float tuple)
                            target_size=opt.target_size,  # target image size (unit: px)
                            target_noise=0.01,  # standard deviation of target image noise
                            color='gray', clean_measurement_pattern=opt.clean_measurement_pattern)  # color channel(s) of target image

    val_data = LFEDataset(root=folder_path,  # dataset root directory
                          shineness=shineness,
                          for_train=False,
                          ds=1,  # temporal down-sampling factor
                          clip=512,  # time range of histograms
                          size=256,  # measurement size (unit: px)
                          scale=1,  # scaling factor (float or float tuple)
                          background=[0.05, 2],  # background noise rate (float or float tuple)
                          target_size=opt.target_size,  # target image size (unit: px)
                          target_noise=0.01,  # standard deviation of target image noise
                          color='gray', clean_measurement_pattern=opt.clean_measurement_pattern)  # color channel(s) of target image

    if not len(train_data) or not len(val_data):
        raise ValueError('Empty train/validation split; check the LFE dataset layout.')
    train_data[0]
    val_data[0]

    if opt.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = DistributedSampler(train_data,num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler = SequentialSampler(val_data)
    else:
        train_sampler = RandomSampler(train_data)
        val_sampler = SequentialSampler(val_data)

    train_loader = DataLoader(train_data, sampler=train_sampler,batch_size=opt.bacth_size, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, sampler=val_sampler,batch_size=opt.bacth_size, num_workers=opt.num_workers, pin_memory=True)
    if not len(train_loader):
        raise ValueError('Batch size exceeds the number of training samples.')
    logging.info("Load training and validation data complete!")
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")

    logging.info("Constructing Models...")
    model = nlost.NLOST(ch_in=1, num_coders=opt.num_coders,spatial=128,tlen=256,bin_len=0.01,target_size=opt.target_size)
    model.to(device)
    logging.info(model)
    if opt.distributed:
        model = DDP(model, device_ids=[opt.loc_rank],find_unused_parameters=True)
        logging.info("Models constructed complete! Paralleled on {} GPUs".format(torch.cuda.device_count()))
    else:
        logging.info("Models constructed complete on SINGLE GPU!")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Numbers of parameters are: {}".format(num_params/1e6))
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")

    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt.opter == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=opt.lr_rate, weight_decay=opt.weit_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=opt.lr_rate)

    n_iter = 0
    start_epoch = 1
    items = ["ALL", "int", "dep"]
    train_loss = {items[0]: [], items[1]: [], items[2]: []}
    val_loss = {items[0]: [], items[1]: [], items[2]: []}
    logWriter = SummaryWriter(opt.model_dir) if utils.is_main_process() else None
    logging.info("Parameters initialized")
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")

    if opt.resmue:
        from utils.checkpoint import load_checkpoint
        checkpoint = load_checkpoint(model, opt.resmod_dir)
        start_epoch = checkpoint.get('epoch', 0) + 1
        n_iter = checkpoint.get('n_iter', 0)
        if 'optimizer' not in checkpoint:
            raise ValueError('Resume requires the original checkpoint with optimizer state.')
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info('Resumed from %s', opt.resmod_dir)

    if utils.is_main_process():
        import json
        from pathlib import Path
        (Path(opt.model_dir) / 'config.json').write_text(json.dumps(vars(opt), indent=2))

    logging.info("Start training...")
    for epoch in range(start_epoch, opt.num_epoch + 1):
        logging.info("Epoch: {}, LR: {}".format(epoch, optimizer.param_groups[0]["lr"]))

        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)

        model, optimizer, n_iter, train_loss, val_loss, logWriter,train_metrics,val_metrics = \
            train(model, train_loader, val_loader, optimizer, \
                epoch, n_iter, train_loss, val_loss, opt, logWriter)

        log_str = 'Epoch_Train_{} | '.format(epoch)
        for k in train_metrics:
            log_str += '{:s} {:.5f} | '.format(k, train_metrics[k].item())
        if utils.is_main_process():
            logging.info(log_str)

        log_str = 'Epoch_Test_{} | '.format(epoch)
        for k in val_metrics:
            log_str += '{:s} {:.5f} | '.format(k, val_metrics[k].item())
        if utils.is_main_process():
            logging.info(log_str)


        if utils.is_main_process():
            save_checkpoint(n_iter, epoch, model, optimizer,\
                file_path=opt.model_dir+"/epoch_{}_{}_END.pth".format(epoch, n_iter))

        logging.info("End of epoch: {}. Checkpoint saved!".format(epoch))

    if logWriter is not None:
        logWriter.close()
    if opt.distributed:
        torch.distributed.destroy_process_group()


if __name__=="__main__":
    main()


