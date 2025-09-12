import torch
import torch.nn as nn
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter
import os 
import torch.distributed as dist




class worker(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.start_epoch = 0
        self.n_iter = 0
        
    def update_parse_args(self):
        today = datetime.today()
        self.args.model_dir += self.args.model_name +"_"+ str(today.year)+"_"+str(today.month)+str(today.day)
        # mkdirs if necessary
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir, exist_ok=True)
        # save args to files
        if self.args.param_store:
            args_dict = self.args.__dict__
            config_bk_pth = self.args.model_dir + "/config_bk.txt"
            with open(config_bk_pth, "w") as cbk_pth:
                cbk_pth.writelines("------------------Start------------------"+ "\n")
                for key, valus in args_dict.items():
                    cbk_pth.writelines(key + ": " + str(valus) + "\n")
                cbk_pth.writelines("------------------End------------------"+ "\n")
            print("Config file load complete! \nNew file saved to {}".format(config_bk_pth))
        return self.args
    
    def init_log(self):
        if logging.root: del logging.root.handlers[:]
        logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(self.args.model_dir + '/train.log' ),
            logging.StreamHandler()
        ],
        format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
        )
    
    def init_tensorboard(self):
        self.writer = SummaryWriter(self.args.model_dir + "/")
    
    def setup_experiments(self):
        if dist.get_rank() == 0:
            self.update_parse_args()
            self.init_log()
            self.init_tensorboard()
            logging.info("Number of available GPUs: {} {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device())))
            command = f'cp -r ./ {self.args.model_dir}/code/'
            os.system(command)
            logging.info(f"Copy Code to: {self.args.model_dir}/code")

    def save_checkpoint(self, n_iter, epoch, model, optimer, file_path):
        """
        params:
        epcoh:the current epoch
        n_iter: the current iter
        model: the model dict
        optimer: the optimizer dict
        """
        state = {}
        state["n_iter"] = n_iter
        state["epoch"] = epoch
        state["lr"] = optimer.param_groups[0]["lr"]
        state["state_dict"] = model.state_dict()
        state["optimizer"] = optimer.state_dict()
        if dist.get_rank() == 0:
            torch.save(state, file_path)


    def train_model(self):
        pass
            
    def do(self,local_rank,nprocs):
        self.local_rank = local_rank
        self.nprocs = nprocs
        # torch.cuda.set_device(self.local_rank)
        # dist.init_process_group(backend="nccl")

        self.setup_experiments()
        self.train_model()
        
        
    
    def test_model(self):
        pass
    
    def evluation_real_data(self):
        pass
        
    def forward(self,input):
        pass