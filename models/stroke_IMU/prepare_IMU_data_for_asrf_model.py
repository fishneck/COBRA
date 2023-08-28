import os
import pandas as pd
import numpy as np
import shutil
import random


HC_data_dir = # Your data directory to healthy subjects' data
#'./kinematic_data/HC-complete-data-processed/'

HAR_data_dir = # Your data directory to impaired subjects' data
#'./kinematic_data/HAR-data-processed-2/'

os.makedirs('./dataset/sensors_all') # healthy subjects
os.makedirs('./dataset/sensors_all/groundTruth')
os.makedirs('./dataset/sensors_all/features')

os.makedirs('./dataset/sensors') # impaired subjects
os.makedirs('./dataset/sensors/groundTruth')
os.makedirs('./dataset/sensors/features')


mapping = np.array([['0', 'idle'],
                       ['1', 'reach'],
                       ['2', 'reposition'],
                       ['3', 'stabilize'],
                       ['4', 'transport']])
np.savetxt('./dataset/sensors/mapping.txt', mapping, delimiter=" ", fmt="%s") 
np.savetxt('./dataset/sensors_all/mapping.txt', mapping, delimiter=" ", fmt="%s") 


def feature_normalize(dataset, mu = None, sigma = None):
    if mu is None:
        mu = np.mean(dataset,axis = 0)
    if sigma is None:
        sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

def make_features_groundTruth(root_path,f,save_path):
    sample_data = pd.read_csv(root_path+f)
    sample_data['motion_apd'] = sample_data.iloc[:,-6:-1].idxmax(1)
    sample_data.drop(labels = ['markers','markernames'],axis = 1, inplace = True)
    sample_data.drop(labels = ['upperspinecoursedeg','upperspinepitchdeg','upperspinerolldeg',\
                                'upperarmcourseltdeg','upperarmpitchltdeg','upperarmrollltdeg',\
                                'forearmcourseltdeg','forearmpitchltdeg','forearmrollltdeg',\
                                'handcourseltdeg','handpitchltdeg','handrollltdeg','upperarmcoursertdeg',\
                                'upperarmpitchrtdeg','upperarmrollrtdeg','forearmcoursertdeg',\
                                'forearmpitchrtdeg','forearmrollrtdeg','handcoursertdeg','handpitchrtdeg',\
                                'handrollrtdeg','lowerspinecoursedeg','lowerspinepitchdeg','lowerspinerolldeg',\
                                'pelviscoursedeg','pelvispitchdeg','pelvisrolldeg'], axis = 1, inplace = True)
    sample_data.iloc[:,0:-17] = feature_normalize(sample_data.iloc[:,0:-17])
    readings = np.array(sample_data.iloc[:,1:-16])
    readings = np.transpose(readings)
    gt = np.array(sample_data['class'])
    gt[gt == 'retract'] = 'reposition'
    gt[gt=='rest'] = 'idle'
    np.save(save_path+'features/'+f[:-4],readings)
    np.savetxt(save_path+'groundTruth/'+f[:-4]+'.txt',gt,fmt='%s')
    
for f in os.listdir(HC_data_dir):
    make_features_groundTruth(HC_data_dir,f,f'./dataset/sensors_all/')
for f in os.listdir(HAR_data_dir):
    make_features_groundTruth(HAR_data_dir,f,'./dataset/sensors/')

# make gr_array and boundary using python scripts
cmd = 'python utils/generate_gt_array.py --dataset_dir dataset'
os.system(cmd)

cmd = 'python utils/generate_boundary_array.py --dataset_dir dataset'
os.system(cmd)


activity_c_id = {}

d = 'sensors_all'
activity_c_id[d]=[]
for f in os.listdir(f'./dataset/{d}/groundTruth/'):
    c = f.split('_')[0]
    if len(activity_c_id)>0 and c not in activity_c_id[d]:
        activity_c_id[d].append(c)
            
print('num of healthy subjects:', len(activity_c_id['sensors_all']))


def make_csv(d,fold=5):
    all_subjects_ori = activity_c_id[d]
    all_subjects_ori.sort(key = lambda x: int(x[1:]))
    print(all_subjects_ori)
    #test_subjects = np.random.choice(all_subjects_ori,size=4,replace=False)
    test_subjects =  ['C15','C23','C4','C30']
    print('test_subject: ', test_subjects)

    train_val_subjects = [i for i in all_subjects_ori if i not in test_subjects]
    all_features = os.listdir(f'./dataset/{d}/features/')
    root_path_feat = f'./dataset/{d}/features/' 
    root_path_gt = f'./dataset/{d}/gt_arr/'
    root_path_gtb = f'./dataset/{d}/gt_boundary_arr/'

    s_all_features = os.listdir('./dataset/sensors/features/')
    s_root_path_feat = './dataset/sensors/features/'
    s_root_path_gt = './dataset/sensors/gt_arr/'
    s_root_path_gtb = './dataset/sensors/gt_boundary_arr/'
    
    all_subjects = train_val_subjects
    random.seed(0)
    random.shuffle(all_subjects)
    n_val = len(all_subjects)//fold
    for i in range(fold):#for i,s in enumerate(all_subjects_ori):
        val_subjects = all_subjects[i*n_val:min((i+1)*n_val,len(all_subjects))]
        train_subjects = [i for i in all_subjects if i not in val_subjects]
        
        print(train_subjects,val_subjects,test_subjects)
        af_feats = []
        af_gt = []
        af_gtb = []
        for f in all_features:
            for s in train_subjects:
                if s+'_' in f:
                    af_feats.append(root_path_feat+f)
                    af_gt.append(root_path_gt+f)
                    af_gtb.append(root_path_gtb+f)
        df = pd.DataFrame({'feature':af_feats,'label':af_gt,'boundary':af_gtb})
        df.to_csv(f'./csv/{d}/train'+str(int(i+1))+'.csv',index=False)

        af_feats = []
        af_gt = []
        af_gtb = []
        for f in all_features:
            for s in val_subjects:
                if s+'_' in f:
                    af_feats.append(root_path_feat+f)
                    af_gt.append(root_path_gt+f)
                    af_gtb.append(root_path_gtb+f)
        df = pd.DataFrame({'feature':af_feats,'label':af_gt,'boundary':af_gtb})
        df.to_csv(f'./csv/{d}/val'+str(int(i+1))+'.csv',index=False)

        af_feats = []
        af_gt = []
        af_gtb = []
        for f in all_features:
            for s in test_subjects:
                if s+'_' in f:
                    af_feats.append(root_path_feat+f)
                    af_gt.append(root_path_gt+f)
                    af_gtb.append(root_path_gtb+f)
        df = pd.DataFrame({'feature':af_feats,'label':af_gt,'boundary':af_gtb})
        df.to_csv(f'./csv/{d}/test'+str(int(i+1))+'.csv',index=False)

        af_feats = []
        af_gt = []
        af_gtb = []
        for f in s_all_features:
            if '_'+d.split('_')[1] in f:
                #print('case5')
                af_feats.append(s_root_path_feat+f)
                af_gt.append(s_root_path_gt+f)
                af_gtb.append(s_root_path_gtb+f)

        df = pd.DataFrame({'feature':af_feats,'label':af_gt,'boundary':af_gtb})
        df.to_csv(f'./csv/{d}/s_test'+str(int(i+1))+'.csv',index=False)
    


for k,v in activity_c_id.items():
    print(k,len(v))
    if '_' in k:
        print('generating csv files for ',k)
        make_csv(k)
        
for k,v in activity_c_id.items():
    print(k,len(v))
    for i in range(1, 6):# 5-fold cross validation
        cmd = f'python utils/make_configs.py --root_dir ./result/{k} --dataset {k} --in_channel 77 --param_search True --split {i}'
        os.system(cmd)