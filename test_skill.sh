epoch=(5 30 60 90 100 110 120 150 180 200)
model_dir="auxiliary-20241027_131705"
gpu_id=7

for e in "${epoch[@]}"
do
    CUDA_VISIBLE_DEVICES=$gpu_id python eva_skill.py --base_dir "/home/SSD/qiangya/vae3-0/IAM_nobg" --dataset "IAM64_real" "VATr_recreat" \
        --batchsize 128 --proxy_mode real_fake --model_path "saved/$model_dir/models/DTL_epoch=$e.pt" \
        --stepsize 5e-4 --size 256 256
done