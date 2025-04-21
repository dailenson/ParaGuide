''' Adapted from DetailSemNet [ECCV 2024] and SURDS [ICPR 2022].
    Repositories:
    - https://github.com/nycu-acm/DetailSemNet_OSV
    - https://github.com/soumitri2001/SURDS-SSL-OSV
'''

import numpy as np
from tqdm import tqdm

def compute_accuracy_roc(predictions, labels, step=5e-4):
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
        idx1 = predictions.ravel() >= d     # genuine samples, if the distance is greater than the threshold
        idx2 = predictions.ravel() < d      # forged samples, if the distance is less than the threshold

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        far = float(np.sum(labels[idx1] == 0)) / ndiff
        tpr_arr.append(tpr)
        far_arr.append(far)

        acc = (float(np.sum(labels[idx1] == 1)) + float(np.sum(labels[idx2] == 0))) / len(labels)

        if acc > max_acc:
            max_acc = acc
            d_optimal = d
            
            # FRR, FAR metrics
            min_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_far = float(np.sum(labels[idx1] == 0)) / ndiff   
    
    metrics = {"best_acc" : max_acc, "best_frr" : min_frr, "best_far" : min_far, "tpr_arr" : tpr_arr, "far_arr" : far_arr}
    return metrics, d_optimal