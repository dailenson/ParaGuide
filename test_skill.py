import numpy as np
import pandas as pd
import torch
from data_loader.loader import *
from models.model import *
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import argparse
from utils.metrics import compute_accuracy_roc
from utils.util import fix_seed

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser('ParaGuide -- Evaluation')
    parser.add_argument('--base_dir', type=str, default='./data/ForensicsIAM')
    parser.add_argument('--dataset', type=str, nargs=2, default="IAM" "VATr")
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--size', type=int, nargs=2, default=[512, 512])
    args = parser.parse_args()

    print('\n'+'*'*100)


    # fix the random seed for reproducibility
    random_seed = 1
    fix_seed(random_seed)

    # setup test dataloader
    _, test_loader, test_set, _ = get_dataloader(args)
    
    # build model architecture
    model = ParaGuide(args)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    print(f"Loading model from: {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # feature extraction from test dataset
    features, labels, writer_ids, img_names = np.zeros((len(test_set), 256), dtype=np.float32), [], [], []
    count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            feat = model.pro_fea(model.encoder(batch['image'].to(device), pool=True))
            features[count:(count+len(feat)), :] = feat.cpu().numpy()
            labels.append(batch['label'].cpu().numpy().flatten())    
            writer_ids.append(batch['writer_id'].cpu().numpy().flatten())
            img_names.extend(batch['img_name'])
            count += len(feat)
    labels = np.concatenate(labels)
    writer_ids = np.concatenate(writer_ids)
 

    ### randomly choose the reference set
    df = pd.DataFrame(features)
    df['label'] = labels.copy()
    df['writer_id'] = writer_ids.copy()
    df['img_name'] = img_names.copy()
    df_ref_writer_list = []
    for writer in test_set.writer_dict.values():
        df_ref = df[(df['writer_id']==writer) & (df['label']==1)]
        df_ref = df_ref.sample(8, random_state=0, replace=False)  
        assert (len(df_ref) == 8)
        df_ref_writer_list.append(df_ref)
    df_ref_writer = pd.concat(df_ref_writer_list)
    print(f"Each reference set has 8 samples,  {len(test_set.writer_dict)} sets contain {len(df_ref_writer)} samples")

    dist, y_true, writer_id = [], [], []

    # construct the test pairs between reference sets and test samples
    preds = pd.DataFrame(columns=['img_name', 'writer_id', 'y_true', 'y_pred'])
    for i in tqdm(range(len(df)), leave=False):
        feature = np.array(df.iloc[i][0:256]).flatten()
        label = df.iloc[i]['label']
        writer = df.iloc[i]['writer_id']
        img_name = df.iloc[i]['img_name']

        ''' calculate the distance between the reference sets and the test samples, 
            including genuine samples and skilled forgeries.'''
        if img_name not in set(df_ref_writer['img_name']):
            df_ref = df_ref_writer[(df_ref_writer['writer_id']==writer)]
            assert (len(df_ref) == 8)
            df_ref = df_ref.drop(['label', 'writer_id', 'img_name'], axis=1)
            mean_ref = np.mean(np.array(df_ref, dtype=np.float32), axis=0)
            mse_diff = np.dot(feature, mean_ref) / (np.linalg.norm(feature) * np.linalg.norm(mean_ref))
            dist.append(mse_diff)
            y_true.append(label)
            writer_id.append(writer)

    print(f">> Total numbers of tested samples: {len(dist)}")
    metrics, thresh_optimal = compute_accuracy_roc(np.array(dist), np.array(y_true))
    print("Metrics obtained: \n" + '-'*50)
    epoch = args.model_path.split('.')[-2][-5:]
    print(f"current epoch is {epoch}")
    print(f"Acc: {metrics['best_acc'] * 100 :.3f} %")
    print(f"FAR: {metrics['best_far'] * 100 :.3f} %")
    print(f"FRR: {metrics['best_frr'] * 100 :.3f} %")