import json
import numpy as np
import os
import shutil
import random
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--HC_dir', type=str, required=True) #healthy
parser.add_argument('--HAR_dir', type=str, required=True) #impaired
parser.add_argument('--fold', type=str, required=True) #split index = 1,2,3,4
args = parser.parse_args()

videodir = args.HC_dir

os.makedirs(f'splits/split{fold}/test', exist_ok=True)
os.makedirs(f'splits/split{fold}/test_stroke', exist_ok=True)

outlist_train = f'splits/split{fold}/train.csv'
outlist_val = f'splits/split{fold}/val.csv'
outlist_test = f'splits/split{fold}/test/test.csv'

train_val_dict = {'fold_1':['C00011','C00022','C00031','C00028'],
                  'fold_2':['C00027','C00020','C00032','C0007'],
                  'fold_3':['C00019','C00024','C00025','C0009'],
                  'fold_4':['C00026','C00012','C0005','C00029']}

test_subjects =  ['C00015','C00023','C0004','C00030']


f = open(outlist_train, 'w')

val_person_ID = train_val_dict[f'fold_{args.fold}']
test_person_ID = test_subjects
class_count_dict = {'0':0, '1':0, '2':0, '3':0, '4':0}

person_list = os.listdir(videodir)


for person_ID in person_list:
    if not person_ID in val_person_ID and  not person_ID in test_person_ID:

        videodir_person_ID = os.path.join(videodir,person_ID)
        activity_list = os.listdir(videodir_person_ID)

        for activity_ID in activity_list:

            activity_elements = activity_ID.split(' ')
            newfolder_name = ''
            for j in range(len(activity_elements)):
                newfolder_name = newfolder_name + activity_elements[j]
                if j < len(activity_elements) - 1:
                    newfolder_name = newfolder_name + '_'

            # print(newfolder_name)

            folder_name_v1 = os.path.join(videodir_person_ID, activity_ID)
            folder_name_cmb = os.path.join(videodir_person_ID, newfolder_name)

            shutil.move(folder_name_v1, folder_name_cmb)

            activity_ID = newfolder_name

            videodir_person_ID_activity_ID = os.path.join(videodir_person_ID, activity_ID)
            clip_list = [i for i in os.listdir(videodir_person_ID_activity_ID) if '.mp4' in i]
            

            print(videodir_person_ID_activity_ID)
            print(clip_list)
            
            def takeStart(elem):
                time_begin = elem.split('e')[0]
                return int(time_begin[1:])
            clip_list_filtered=filter(lambda x: int(x.split('e')[0][1:]) % 2 == 0, clip_list)
            
            clip_list = sorted(clip_list_filtered, key=takeStart)

            if (len(clip_list)==0):
                print(videodir_person_ID_activity_ID)
            for clip_ID in clip_list:
                class_ID_tmpt = clip_ID.split('.')[0]
                class_ID = class_ID_tmpt.split('_')[-1]
                class_count_dict[class_ID] +=1

                file_name = os.path.join(videodir_person_ID_activity_ID,clip_ID)
                f.write(file_name + ' ' + class_ID + '\n')
f.close()


class_count_dict = {'0':0, '1':0, '2':0, '3':0, '4':0}
f = open(outlist_val, 'w')
for person_ID in person_list:
    if person_ID in val_person_ID:
        videodir_person_ID = os.path.join(videodir, person_ID)
        activity_list = os.listdir(videodir_person_ID)

        for activity_ID in activity_list:
            activity_elements = activity_ID.split(' ')
            newfolder_name = ''
            for j in range(len(activity_elements)):
                newfolder_name = newfolder_name + activity_elements[j]
                if j < len(activity_elements) - 1:
                    newfolder_name = newfolder_name + '_'


            folder_name_v1 = os.path.join(videodir_person_ID, activity_ID)
            folder_name_cmb = os.path.join(videodir_person_ID, newfolder_name)

            shutil.move(folder_name_v1, folder_name_cmb)

            activity_ID = newfolder_name
            videodir_person_ID_activity_ID = os.path.join(videodir_person_ID, activity_ID)
            clip_list = [i for i in os.listdir(videodir_person_ID_activity_ID) if '.mp4' in i]
            
            def takeStart(elem):
                time_begin = elem.split('e')[0]
                return int(time_begin[1:])


            clip_list_filtered=filter(lambda x: int(x.split('e')[0][1:]) % 2 == 0, clip_list)

            clip_list = sorted(clip_list_filtered, key=takeStart)

            for clip_ID in clip_list:
                class_ID_tmpt = clip_ID.split('.')[0]
                class_ID = class_ID_tmpt.split('_')[-1]

                class_count_dict[class_ID] += 1

                file_name = os.path.join(videodir_person_ID_activity_ID, clip_ID)
                f.write(file_name + ' ' + class_ID + '\n')
f.close()

print('val_class_distri')
print(class_count_dict)




class_count_dict = {'0':0, '1':0, '2':0, '3':0, '4':0}
f = open(outlist_test, 'w')
for person_ID in person_list:
    if person_ID in test_person_ID:
        videodir_person_ID = os.path.join(videodir, person_ID)
        activity_list = os.listdir(videodir_person_ID)

        for activity_ID in activity_list:
            activity_elements = activity_ID.split(' ')
            newfolder_name = ''
            for j in range(len(activity_elements)):
                newfolder_name = newfolder_name + activity_elements[j]
                if j < len(activity_elements) - 1:
                    newfolder_name = newfolder_name + '_'

            folder_name_v1 = os.path.join(videodir_person_ID, activity_ID)
            folder_name_cmb = os.path.join(videodir_person_ID, newfolder_name)

            shutil.move(folder_name_v1, folder_name_cmb)

            activity_ID = newfolder_name

            videodir_person_ID_activity_ID = os.path.join(videodir_person_ID, activity_ID)
            clip_list = [i for i in os.listdir(videodir_person_ID_activity_ID) if '.mp4' in i]
            
            def takeStart(elem):
                time_begin = elem.split('e')[0]
                return int(time_begin[1:])

            clip_list_filtered=filter(lambda x: int(x.split('e')[0][1:]) % 2 == 0, clip_list)

            clip_list = sorted(clip_list_filtered, key=takeStart)

            for clip_ID in clip_list:
                class_ID_tmpt = clip_ID.split('.')[0]
                class_ID = class_ID_tmpt.split('_')[-1]

                class_count_dict[class_ID] += 1

                file_name = os.path.join(videodir_person_ID_activity_ID, clip_ID)
                f.write(file_name + ' ' + class_ID + '\n')
f.close()

print('test_class_distri')
print(class_count_dict)


videodir = args.HAR_dir

outlist_test = f'splits/split{fold}/test_stroke/test.csv'


person_list = os.listdir(videodir)
if 'test.csv' in person_list:
    person_list.remove('test.csv')
test_person_ID = person_list

class_count_dict = {'0':0, '1':0, '2':0, '3':0, '4':0}
f = open(outlist_test, 'w')
for person_ID in person_list:
    if person_ID in test_person_ID:
        videodir_person_ID = os.path.join(videodir, person_ID)
        activity_list = os.listdir(videodir_person_ID)

        for activity_ID in activity_list:
            activity_elements = activity_ID.split(' ')
            newfolder_name = ''
            for j in range(len(activity_elements)):
                newfolder_name = newfolder_name + activity_elements[j]
                if j < len(activity_elements) - 1:
                    newfolder_name = newfolder_name + '_'

            folder_name_v1 = os.path.join(videodir_person_ID, activity_ID)
            folder_name_cmb = os.path.join(videodir_person_ID, newfolder_name)

            shutil.move(folder_name_v1, folder_name_cmb)

            activity_ID = newfolder_name

            videodir_person_ID_activity_ID = os.path.join(videodir_person_ID, activity_ID)
            clip_list = [i for i in os.listdir(videodir_person_ID_activity_ID) if '.mp4' in i]

            def takeStart(elem):
                time_begin = elem.split('e')[0]
                return int(time_begin[1:])

            clip_list_filtered=filter(lambda x: int(x.split('e')[0][1:]) % 2 == 0, clip_list)

            clip_list = sorted(clip_list_filtered, key=takeStart)

            for clip_ID in clip_list:
                class_ID_tmpt = clip_ID.split('.')[0]
                class_ID = class_ID_tmpt.split('_')[-1]

                class_count_dict[class_ID] += 1

                file_name = os.path.join(videodir_person_ID_activity_ID, clip_ID)
                f.write(file_name + ' ' + class_ID + '\n')
f.close()

print('test_class_distri')
print(class_count_dict)

