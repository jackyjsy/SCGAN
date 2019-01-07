CUDA_VISIBLE_DEVICES=1 python main.py \
--mode='train' --dataset='CelebA' \
--experiment_path='./experiment_celeba_mod_final' \
--port=8097 --web_dir='experiment_celeba_mod_final' \
--batch_size=16 #--pretrained_model='20_12000'
