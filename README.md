# 🍔 ForensicsIAM Benchmark

  We provide the ForensicsIAM dataset in [Google Drive](https://drive.google.com/drive/folders/1vivXSekLLr06UA7pBdZFLzqOnUiSwhSV?usp=sharing). Please download the dataset, uzip it and move the extracted files to /ForensicsIAM.

## 📂 Dataset Folder Structure
  ```
  ForensicsIAM/
  │
  ├── Genuine/ - one genuine subset  
  │   └── IAM/
  │       ├── train/
  │       ├── test/
  │       ├── train.txt
  │       └── test.txt
  │
  └── Forged/ - three forged subsets
      ├── One-DM/
      │   ├── train/
      │   ├── test/
      │   ├── train.txt
      │   └── test.txt
      ├── VATr/
      │   ├── train/
      │   ├── test/
      │   ├── train.txt
      │   └── test.txt
      └── HWT/
          ├── train/
          ├── test/
          ├── train.txt
          └── test.txt               
  ```


# 🤖 ParaGuide Algorithm

## 🚀 Training & Test
**Training**
- To train the ParaGuide, run this command:
```
python main.py --base_dir "/home/data/ForensicsIAM" --dataset "IAM" "Forged subset" \
--batchsize 64 --exp 'saved_path' --size 512 512 --learning_rate 0.005
```
**Note**:
Please modify ``base_dir`` and ``exp`` according to your own path. Please modify ``Forged subset`` to the subset of data you want to use,``One-DM``,``VATr``or``HWT``.

**Test**
- To test the ParaGuide on the skilled forgery scenario, run this command:
```
python eva_skill.py --base_dir "/home/data/ForensicsIAM" --dataset "IAM64" "Forged subset" \
        --batchsize 128 --model_path "saved_path/DTL_epoch=200.pt" --size 512 512
```
**Note**:
Please modify ``base_dir`` and ``exp`` according to your own path. Please modify ``Forged subset`` to the subset of data you want to use,``One-DM``,``VATr``or``HWT``.
- To test the ParaGuide on the random forgery scenario, run this command:
```
python eval_random.py --base_dir "/home/data/ForensicsIAM" --dataset "IAM" \
        --batchsize 128 --model_path "saved_path/DTL_epoch=200.pt" --size 512 512
```
**Note**:
Please modify ``base_dir`` and ``model_path`` according to your own path. 