import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pad
import math

random.seed(1)
np.random.seed(1)

def debug_image_v1(img, name):
    new_image = (img + 1)/2
    topil = transforms.ToPILImage()
    pil_img = topil(new_image)
    pil_img.save(name)



class Writer_Dataset(Dataset):
    def __init__(self, args, mode):

        self.args = args
        self.mode = mode
        self.writer_set = set()
        self.basic_transforms = transforms.Compose([transforms.Resize([int(args.size[0]), int(args.size[1])]),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                         std=[0.5, 0.5, 0.5])
                                                    ])
        self.prototype_transforms = transforms.Compose([transforms.Resize([int(args.size[0]), int(args.size[1])]),
                                                        transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                         std=[0.5, 0.5, 0.5])
                                                    ])
        #data_root = os.path.join(self.args.base_dir, self.args.dataset)  # ./../BHSig260/Bengali
        if self.mode == 'Train':
            data_dict, self.writer_dict = self.load_data(os.path.join(self.args.base_dir, 'IAM64_train.txt'))
        elif self.mode == 'Test':
            data_dict, self.writer_dict = self.load_data(os.path.join(self.args.base_dir, 'IAM64_test.txt'))
        
        data_list = []
        for dataset in self.args.dataset:
            for idx in data_dict:
                if self.mode == 'Train':
                    img_path = os.path.join(self.args.base_dir, dataset, 'train', 
                                            data_dict[idx]['s_id'], data_dict[idx]['image'])
                elif self.mode == 'Test':
                    img_path = os.path.join(self.args.base_dir, dataset, 'test', 
                                            data_dict[idx]['s_id'], data_dict[idx]['image'])
                label = 1 if 'IAM64_real' in dataset else 0
                data_list.append({'img_path': img_path, 'writer_id':data_dict[idx]['s_id'], 'label': label})
                #self.data_df = self.data_df._append({'img_path': img_path, 's_id':data_dict[idx]['s_id'], 'label': label}, ignore_index=True)
        self.data_df = pd.DataFrame(data_list)
        print(f'all {self.mode} datasets comprises total {len(self.data_df)} images !!')

    def __len__(self):
        return len(self.data_df)
        

    def load_data(self, data_path):
        
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split(' ') for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                s_id = i[0].split(',')[0]
                image = i[0].split(',')[1] + '.png'
                transcription = i[1]
                if self.mode == 'Train':
                    full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
                elif self.mode == 'Test':
                    if len(transcription) > 1:
                        full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
                    else:
                        continue
                self.writer_set.add(s_id)
                idx += 1
        
        sorted_writer_list = sorted(self.writer_set)
        print("total writers", len(sorted_writer_list))
        writer_dict = {}
        index = 0
        for i in sorted_writer_list:
            writer_dict[i] = index
            index += 1
        return full_dict, writer_dict

    def get_prototype(self, img_path, num=16):
        writer_dirs = os.path.dirname(img_path)
        writer_imgs = os.listdir(writer_dirs)
        img_paths = [os.path.join(writer_dirs, i) for i in writer_imgs]
        if len(writer_imgs) > num:
            writer_imgs = random.sample(img_paths, num)
        else:
            writer_imgs = random.choices(img_paths, k=num)
        writer_imgs = [os.path.join(writer_dirs, i) for i in writer_imgs]
        writer_imgs = [Image.open(i).convert("RGB") for i in writer_imgs]
        rows = [writer_imgs[i:i+4] for i in range(0, num, 4)]

        # 先将每组4张图片拼接成一行
        rows = [np.hstack([np.array(img) for img in row]) for row in rows]

        # 确定所有行的最小宽度
        min_width = min(row.shape[1] for row in rows)

        # 根据最小宽度裁剪每行
        rows = [row[:, :min_width] for row in rows]

        # 将所有裁剪后的行垂直拼接成最终图片
        img = np.vstack(rows)
        return Image.fromarray(img)


        

    def __getitem__(self, index):
        sample = {}
        img_path = self.data_df.iloc[index]['img_path']
        sig_image = Image.open(img_path).convert("RGB")
        prototype_img = self.get_prototype(img_path)
        contrast_img = self.get_prototype(img_path)
        writer_id = self.data_df.iloc[index]['writer_id']
        label = self.data_df.iloc[index]['label']
        if self.mode == 'Train':
            if label == 0:
                writer_id = len(self.writer_dict) +  self.writer_dict[writer_id]
            else:
                writer_id = self.writer_dict[writer_id]
        elif self.mode == 'Test':
            writer_id = self.writer_dict[writer_id]
        else:
            raise ValueError('Invalid mode')
        #cropped_sig = self.__get_com_cropped__(sig_image)
        sig_image = self.basic_transforms(sig_image)
        prototype_img = self.prototype_transforms(prototype_img)
        contrast_img = self.prototype_transforms(contrast_img)
        sample = {'image' : sig_image, 'prototype_img' : prototype_img, 'contrast_img' : contrast_img, 'label' : label,
                    'writer_id' : int(writer_id), 'img_name' : os.path.basename(img_path)}
        return sample
        

def get_dataloader(args):
    train_dset = Writer_Dataset(args, mode='Train')
    train_loader = DataLoader(train_dset, batch_size=args.batchsize, shuffle=True, num_workers=8)
    # print('==> Train data loaded')
    
    test_dset = Writer_Dataset(args, mode='Test')
    test_loader = DataLoader(test_dset, batch_size=args.batchsize, shuffle=False, num_workers=8)
    # print('==> Test data loaded')

    return train_loader, test_loader, test_dset, train_dset