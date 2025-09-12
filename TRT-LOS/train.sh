pip install imageio[ffmpeg] pip install imageio[pyav] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ipdb scikit-learn scikit-image tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA_VISIBLE_DEVICES=0

#  CUDA_VISIBLE_DEVICES=4,5,6,7 
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 29538 train.py \
--model_name 'TRT_SP' \
--train_bacth_size 1 \
--test_bacth_size 4 \
--lr_rate 1e-4 \
--num_epoch 50 \
--num_workers 8 \
--data_size 256 \
--model_dir '/data1/yueli/TRT_SP/' \
--root_path '/data1/yueli/dataset/sp_syn/nyu_syn_sp_256/' \
--train_total_path 'utils/256fortraining.txt' \
--test_total_path 'utils/256fortesting.txt' \
--num_save 300 \
# --resume True \
# --resmod_dir xxx.pth


