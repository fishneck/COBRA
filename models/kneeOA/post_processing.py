import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.metrics import confusion_matrix
    
def get_acc_precision(file):
    # file: str, path to mpunt output prediction
    ds = '/'.join(file.split('/')[:-1])+'/'
    f = file.split('/')[-1]
    sub = f.split('_')[0]
    group = ds.split('/')[-3].split('_pred')[0]
    test_pred = nib.load(ds + f).get_fdata().argmax(axis=3)
    label_path = f'./MultiPlanarUNet/data/OAI_ZIB/{group}/labels/{sub}.nii.gz'
    test_label = nib.load(label_path).get_fdata()
    
    cm = confusion_matrix(test_label.flatten(),test_pred.flatten())
    accuracy = np.diag(cm).sum()/cm.sum()
    precision = (np.diag(cm[1:,1:])/cm[1:,:].sum(axis=1)).mean()
    
    return sub, group, accuracy, precision

def get_ci_percentile(x,confidence=0.95):
    # x: list or array
    data = np.random.choice(x, 5000, replace = True)
    u_percentile = confidence + (1-confidence)/2
    l_percentile = (1-confidence)/2
    u_ci = np.percentile(data,u_percentile*100)
    l_ci = np.percentile(data,l_percentile*100)
    return l_ci, u_ci


def get_COBRA_from_seg_confidence(dirs):
    # dirs, list of str, paths to mpunt output prediction
    COBRA = {}
    COBRA['sub'] = []
    for i in range(5):
        COBRA[f'confidence_{i}_mean'] = []

    COBRA['seq_prob_2'] = [] # femur cartilage
    COBRA['seq_prob_4'] = [] # tibia cartilage
    COBRA['split'] = []
    for d in dirs:
        for f in os.listdir(d):
            print(f)
            test_pred_impaired_image_load = nib.load(d + f).get_fdata()
            sub = f.split('_')[0]
            COBRA['sub'].append(sub)
            print(d.split('/')[-3].split('_pred')[0])
            COBRA['split'].append(d.split('/')[-3].split('_pred')[0])
            
            for k in range(5):
                v = test_pred_impaired_image_load.max(axis=3)[test_pred_impaired_image_load.argmax(axis=3)==k]
                COBRA[f'confidence_{k}_mean'].append(v.mean()/6) # divided by 6 planes for multi-planr sampling
                if k==2 or k==4:
                    prob_sampled = np.random.choice(v,size=1000,replace=True) # collect samples for distribution plot
                    COBRA[f'seq_prob_{k}'].append(prob_sampled)
    return pd.DataFrame(COBRA)
  