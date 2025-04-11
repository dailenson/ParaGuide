CUDA_VISIBLE_DEVICES=6 python main.py --base_dir "/home/SSD/qiangya/vae3-0/IAM_nobg" --dataset "IAM64_real" "VATr_recreat" \
    --round "1" --batchsize 64 --exp 'proto_mix' --size 512 512 \
    --learning_rate 0.005