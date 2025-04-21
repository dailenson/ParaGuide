epoch=(30)
model_dir="debug-20250420_134049"
device_id=5

for e in "${epoch[@]}"
do
    CUDA_VISIBLE_DEVICES=$device_id python test_random.py --base_dir "/home/SSD/qiangya/ForensicsIAM" \
        --model_path "saved/$model_dir/models/epoch=$e.pt"
done