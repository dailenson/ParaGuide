epoch=(30)
model_dir="debug-20250420_134049"
gpu_id=5

for e in "${epoch[@]}"
do
    CUDA_VISIBLE_DEVICES=$gpu_id python test_skill.py --base_dir "/home/SSD/qiangya/ForensicsIAM" --dataset "IAM" "VATr" \
        --model_path "saved/$model_dir/models/epoch=$e.pt" 
done