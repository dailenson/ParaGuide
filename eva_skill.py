import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import torch
from torch.nn.functional import mse_loss
from torchvision.utils import save_image
import random
import json
# from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
from iam_dataset import *
from model import *
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

    
### Taken from SigNet paper
def compute_accuracy_roc(predictions, labels, step=None):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    print(f"Max: {dmax}, Min: {dmin}")
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
    if step is None:
        step = 0.00005

    max_acc, min_frr, min_far = 0.0, 1.0, 1.0
    d_optimal = 0.0
    tpr_arr, far_arr = [], []
    for d in tqdm(predictions):
    #for d in tqdm(np.arange(dmin, dmax + step, step)):
        idx1 = predictions.ravel() >= d     # pred = 1
        idx2 = predictions.ravel() < d      # pred = 0
        #idx1 = predictions.ravel() <= d     # pred = 1
        #idx2 = predictions.ravel() > d      # pred = 0

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        #tnr = float(np.sum(labels[idx2] == 0)) / ndiff

        #frr = float(np.sum(labels[idx2] == 1)) / nsame
        far = float(np.sum(labels[idx1] == 0)) / ndiff

        tpr_arr.append(tpr)
        far_arr.append(far)

        #acc = 0.5 * (tpr + tnr)
        acc = (float(np.sum(labels[idx1] == 1)) + float(np.sum(labels[idx2] == 0))) / len(labels)

        # print(f"Threshold = {d} | Accuracy = {acc:.4f}")

        if acc > max_acc:
            max_acc = acc
            d_optimal = d
            
            # FRR, FAR metrics
            min_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_far = float(np.sum(labels[idx1] == 0)) / ndiff
            
    
    metrics = {"best_acc" : max_acc, "best_frr" : min_frr, "best_far" : min_far, "tpr_arr" : tpr_arr, "far_arr" : far_arr}
    return metrics, d_optimal

#### compute accuracy per writer
def compute_accuracy_per_writer(predictions, labels, writers, step=None):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    mean_val = np.mean(predictions)
    std_val = np.std(predictions)
    predictions = np.array([(x - mean_val) / std_val for x in predictions])
    print(f"Max: {dmax}, Min: {dmin}")
    if step is None:
        step = 0.00005

    max_acc = 0.0
    optimal_frr, optimal_far = 1.0, 1.0
    d_optimal = 0.0
    unique_writers = np.unique(writers)

    for d in tqdm(np.arange(dmin, dmax + step, step)):
    #for d in tqdm(np.arange(dmin, 5, step)):
    #for d in tqdm(predictions):
        writer_accs, writer_frrs, writer_fars = [], [], []
        for writer in unique_writers:
            writer_idx = writers == writer
            writer_preds = predictions[writer_idx].ravel()
            writer_labels = labels[writer_idx]

            idx1 = writer_preds <= d  # pred = 1
            idx2 = writer_preds > d   # pred = 0

            writer_acc = (np.sum(writer_labels[idx1] == 1) + np.sum(writer_labels[idx2] == 0)) / len(writer_labels)
            writer_frr = np.sum(writer_labels[idx2] == 1) / np.sum(writer_labels == 1)  # False Rejection Rate
            writer_far = np.sum(writer_labels[idx1] == 0) / np.sum(writer_labels == 0)  # False Acceptance Rate

            writer_accs.append(writer_acc)
            writer_frrs.append(writer_frr)
            writer_fars.append(writer_far)

        avg_acc = np.mean(writer_accs)
        avg_frr = np.mean(writer_frrs)
        avg_far = np.mean(writer_fars)

        if avg_acc > max_acc:
            max_acc = avg_acc
            d_optimal = d
            optimal_frr = avg_frr
            optimal_far = avg_far

    metrics = {"best_acc": max_acc, "best_frr": optimal_frr, "best_far": optimal_far}
    return metrics, d_optimal



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Dual Triplet -- Evaluation | SSL for Writer Identification')
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--dataset', type=str, nargs=2, default='./../BHSig260/Bengali')
    parser.add_argument('--saved_models', type=str, default='./saved_models')
    parser.add_argument('--load_model', type=str, default='./../Autoencoder/saved_models/BHSig260_Bengali_SSL_Encoder_RN18_AE.pth')
    parser.add_argument('--batchsize', type=int, default=128)  # change for Testing; default=16
    parser.add_argument('--print_freq', type=int,default=10)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--learning_rate_AE', type=float, default=0.005)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--lambd', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='./saved_models/BHSig260_Bengali_200epochs_LR=0.1_LR-AE=0.0001_LBD=1.0_backbone=AE.pt')
    parser.add_argument('--stepsize', type=float, default=5e-5)
    parser.add_argument('--eval_type', type=str, default='self', choices=['self','cross'])
    parser.add_argument('--proxy_mode', type=str, default='only_real', choices=['only_real', 'real_fake'])
    parser.add_argument('--roc', type=bool, default=False)
    parser.add_argument('--roc_name', type=str, default=None)
    parser.add_argument('--size', type=int, nargs=2, required=True, help='(h , w) of the input image')
    args = parser.parse_args()

    print('\n'+'*'*100)

    # 1. get data
    _, test_loader, test_set, _ = get_dataloader(args)
    
    # 2. load model
    MODEL_PATH = args.model_path
    # THRESHOLD = 0.001934

    checkpoint = torch.load(MODEL_PATH)
    model = Triplet_Model(args)
    model.load_state_dict(checkpoint)
    print(f"Loading model from: {MODEL_PATH}")
    model.to(device)
    model.eval()

    # 3. feature extraction from train and test
    features, labels, writer_ids, img_names = np.zeros((len(test_set), 256), dtype=np.float32), [], [], []

    count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            #feat = model.projector(model.encoder(batch['image'].to(device), pool=True))
            feat = model.pro_fea(model.encoder(batch['image'].to(device), pool=True))
            #feat = model.encoder(batch['image'].to(device), pool=True)
            features[count:(count+len(feat)), :] = feat.cpu().numpy()
            labels.append(batch['label'].cpu().numpy().flatten())    
            writer_ids.append(batch['writer_id'].cpu().numpy().flatten())
            img_names.extend(batch['img_name'])
            count += len(feat)
            ### debug 
            # if count > 8000:
            #     features = features[:count]
            #     break
    labels = np.concatenate(labels)
    writer_ids = np.concatenate(writer_ids)
 

    ### choose the reference set
    df = pd.DataFrame(features)
    df['label'] = labels.copy()
    df['writer_id'] = writer_ids.copy()
    df['img_name'] = img_names.copy()

    #writer_set = {int(writer) for writer in test_set.writer_set}

    df_ref_writer_list = []

    for writer in test_set.writer_dict.values():
        df_ref = df[(df['writer_id']==writer) & (df['label']==1)]
        df_ref = df_ref.sample(8, random_state=0, replace=False)  
        assert (len(df_ref) == 8)
        df_ref_writer_list.append(df_ref)

    df_ref_writer = pd.concat(df_ref_writer_list)
    print(f"Length of reference set: {len(df_ref_writer)}")

    dist, y_true, writer_id = [], [], []

    preds = pd.DataFrame(columns=['img_name', 'writer_id', 'y_true', 'y_pred'])
    for i in tqdm(range(len(df)), leave=False):
        feature = np.array(df.iloc[i][0:256]).flatten() # D = 512 or 128 -- change accordingly
        label = df.iloc[i]['label']
        writer = df.iloc[i]['writer_id']
        img_name = df.iloc[i]['img_name']

        if img_name not in set(list(df_ref_writer['img_name'])):
            ## img is not a part of reference set
            df_ref = df_ref_writer[(df_ref_writer['writer_id']==writer)]
            assert (len(df_ref) == 8)
            df_ref = df_ref.drop(['label', 'writer_id', 'img_name'], axis=1)
            mean_ref = np.mean(np.array(df_ref, dtype=np.float32), axis=0)
            mse_diff = np.dot(feature, mean_ref) / (np.linalg.norm(feature) * np.linalg.norm(mean_ref))
            #mse_diff = np.abs(np.mean(np.subtract(feature, mean_ref)))
            # y_pred = 1 if mse_diff <= THRESHOLD else 0
            # preds = preds.append({'img_name' : img_name, 'writer_id' : writer, 'y_true' : label, 'y_pred' : y_pred}, ignore_index=True)
            dist.append(mse_diff)
            y_true.append(label)
            writer_id.append(writer)

    print(f">> Total numbers of tested samples: {len(dist)}")
    metrics, thresh_optimal = compute_accuracy_roc(np.array(dist), np.array(y_true), step=args.stepsize)
    #metrics, thresh_optimal = compute_accuracy_per_writer(np.array(dist), np.array(y_true), np.array(writer_id), step=args.stepsize)
    print("Metrics obtained: \n" + '-'*50)
    epoch = args.model_path.split('.')[-2][-5:]
    print(f"current epoch is {epoch}")
    print(f"Acc: {metrics['best_acc'] * 100 :.3f} %")
    print(f"FAR: {metrics['best_far'] * 100 :.3f} %")
    print(f"FRR: {metrics['best_frr'] * 100 :.3f} %")