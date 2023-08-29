import subprocess
import glob
import os
import pandas as pd
import numpy as np
import argparse
from joblib import delayed
from joblib import Parallel
import json
import cv2



name_to_ID = {'reach':0, 'transport':1, 'reposition':2, 'stabilize':3, 'rest':4, 'retract':2, 'idle':4 ,0 : 999} 
#0 : 999, placeholder for ignorable labels

camera_pos = pd.read_csv(args.HC_video_position)

def count_frames(path, override=False):
    # grab a pointer to the video file and initialize the total
    # number of frames read
    video = cv2.VideoCapture(path)
    total = 0
    # if the override flag is passed in, revert to the manual
    # method of counting frames
    if override:
        total = count_frames_manual(video)
    # otherwise, let's try the fast way first
    else:
        # lets try to determine the number of frames in a video
        # via video properties; this method can be very buggy
        # and might throw an error based on your OpenCV version
        # or may fail entirely based on your which video codecs
        # you have installed
        try:
            total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total = count_frames_manual(video)
    # release the video file pointer
    fps = video.get(cv2.CAP_PROP_FPS)

    video.release()
    # return the total number of frames in the video
    print(f'total={total}, fps={fps}')
    return total,fps

def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break
        # increment the total number of frames read
        total += 1
    # return the total number of frames in the video file
    return total



def load_file(filename):
    f = lambda x: int(name_to_ID[x])
    df = pd.read_csv(filename, usecols= ['class'], converters={'class':f})


    # how many seconds does it have
    length = int(len(df.index)/100)

    label = np.asarray(df)

    return label, length


def load_file_from_raw(file_path):
#   load labels from raw data
    
    col_order = ['times',
         'lumbarflexiondeg',
         'lumbarlateralrtdeg',
         'lumbaraxialrtdeg',
         'thoracicflexiondeg',
         'thoraciclateralrtdeg',
         'thoracicaxialrtdeg',
         'elbowflexionltdeg',
         'elbowflexionrtdeg',
         'shouldertotalflexionltdeg',
         'shouldertotalflexionrtdeg',
         'shoulderflexionltdeg',
         'shoulderflexionrtdeg',
         'shoulderabductionltdeg',
         'shoulderabductionrtdeg',
         'shoulderrotationoutltdeg',
         'shoulderrotationoutrtdeg',
         'wristextensionltdeg',
         'wristextensionrtdeg',
         'wristradialltdeg',
         'wristradialrtdeg',
         'wristsupinationltdeg',
         'wristsupinationrtdeg',
         'upperspinecoursedeg',
         'upperspinepitchdeg',
         'upperspinerolldeg',
         'upperarmcourseltdeg',
         'upperarmpitchltdeg',
         'upperarmrollltdeg',
         'forearmcourseltdeg',
         'forearmpitchltdeg',
         'forearmrollltdeg',
         'handcourseltdeg',
         'handpitchltdeg',
         'handrollltdeg',
         'upperarmcoursertdeg',
         'upperarmpitchrtdeg',
         'upperarmrollrtdeg',
         'forearmcoursertdeg',
         'forearmpitchrtdeg',
         'forearmrollrtdeg',
         'handcoursertdeg',
         'handpitchrtdeg',
         'handrollrtdeg',
         'lowerspinecoursedeg',
         'lowerspinepitchdeg',
         'lowerspinerolldeg',
         'pelviscoursedeg',
         'pelvispitchdeg',
         'pelvisrolldeg',
         'upperspineaccelsensorxmg',
         'upperspineaccelsensorymg',
         'upperspineaccelsensorzmg',
         'upperarmaccelsensorxltmg',
         'upperarmaccelsensoryltmg',
         'upperarmaccelsensorzltmg',
         'forearmaccelsensorxltmg',
         'forearmaccelsensoryltmg',
         'forearmaccelsensorzltmg',
         'handaccelsensorxltmg',
         'handaccelsensoryltmg',
         'handaccelsensorzltmg',
         'upperarmaccelsensorxrtmg',
         'upperarmaccelsensoryrtmg',
         'upperarmaccelsensorzrtmg',
         'forearmaccelsensorxrtmg',
         'forearmaccelsensoryrtmg',
         'forearmaccelsensorzrtmg',
         'handaccelsensorxrtmg',
         'handaccelsensoryrtmg',
         'handaccelsensorzrtmg',
         'lowerspineaccelsensorxmg',
         'lowerspineaccelsensorymg',
         'lowerspineaccelsensorzmg',
         'pelvisaccelsensorxmg',
         'pelvisaccelsensorymg',
         'pelvisaccelsensorzmg',
         'upperspinerotx',
         'upperspineroty',
         'upperspinerotz',
         'ltupperarmrotx',
         'ltupperarmroty',
         'ltupperarmrotz',
         'ltforearmrotx',
         'ltforearmroty',
         'ltforearmrotz',
         'lthandrotx',
         'lthandroty',
         'lthandrotz',
         'rtupperarmrotx',
         'rtupperarmroty',
         'rtupperarmrotz',
         'rtforearmrotx',
         'rtforearmroty',
         'rtforearmrotz',
         'rthandrotx',
         'rthandroty',
         'rthandrotz',
         'lowerspinerotx',
         'lowerspineroty',
         'lowerspinerotz',
         'pelvisrotx',
         'pelvisroty',
         'pelvisrotz',
         'markers',
         'markernames']
    classes = ['rest','reach','retract','reposition','stabilize','transport','idle']
    minimal_motion = ['idle','rest','stabilize']
    
    def check_and_replace_col_names(data):
        cols = list(data.columns)
        if '_' in cols[0]:
            new_names = []
            for i in cols:
                temp = i.split('_')
                new_col_name = ''.join(temp)
                new_names.append(new_col_name.lower())
            data.columns = new_names
        else:
            data.columns = [i.lower() for i in cols]
        return data

    try:
        temp_data = pd.read_csv(file_path)
        print(temp_data.shape)
        if temp_data.shape[1] != 106:
            print('incorrect number of columns')
            return [], 0
    except:
        try:
            temp_data = pd.read_csv(file_path.split('.')[0]+'_4.23.csv')
        except:
            try:
                temp_data = pd.read_csv(file_path.split('.')[0]+'_labelled.csv')
            except:
                print('file not loaded')
                return [], 0
    temp_data = check_and_replace_col_names(temp_data)
    print('start time: {:3f}, end time: {:3f}'.format(temp_data.times.values[0],temp_data.times.values[-1]))
    try:
        temp_data = temp_data[col_order]
    except:
        print('incorrect cols names')
        return [], 0
    start_time = temp_data.iloc[0,0]
    if start_time != 0:
        temp_data.iloc[:,0] -= start_time
    temp_data['markernames'].fillna(method = 'ffill', inplace=True)
    if np.sum(np.sum(temp_data.isna()))>0:
        print('NaNs detected')
        return [], 0

    if '_4.23' in file_path:
        file_path = file_path[:-9]+file_path[-4:] 
    if '_labelled' in file_path:
        file_path = file_path[:-13] + file_path[-4:]

    temp_data['class'] = 0
    for j in classes:
        temp_data.loc[temp_data['markernames'].str.match('.*'+j+'.*'),'class'] = j
    
    #temp_data = temp_data[temp_data['class'] != 0]
    f = lambda x: int(name_to_ID[x])
    label = temp_data['class'].map(f)
    label = np.asarray(label).reshape(-1,1)
    length = length = int(temp_data.shape[0] / 100) 

    return label, length

def divide_clip(source_filename, output_dir, label, length, flip):
    """
    Construct command to trim the videos (ffmpeg required)
    arguments:
    ---------
    source_filename: str
        the source video file name
    output_dir: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    for start_time in range(length-2):
        end_time = start_time + 2
        output_filename = os.path.join(output_dir,'s'+str(start_time)+'e'+str(end_time)+'_'+str(label[start_time*100+100,0])+'.mp4')

        if not flip:
            command = ['ffmpeg',
                       '-i', '"%s"' % source_filename, '-vf', "'scale=512:512'", '-colorspace','5', '-color_primaries', '2', '-color_trc', '2',
                       '-ss', str(start_time),
                       '-t', str(end_time - start_time),
                       '-c:v', 'libx264', '-c:a', 'copy',
                       '-threads', '1',
                       '-loglevel', 'panic',
                       '"%s"' % output_filename]
        else:
            command = ['ffmpeg',
                       '-i', '"%s"' % source_filename, '-vf', "'hflip,scale=512:512'", '-colorspace','5', '-color_primaries', '2', '-color_trc', '2',
                       '-ss', str(start_time),
                       '-t', str(end_time - start_time),
                       '-c:v', 'libx264', '-c:a', 'copy',
                       '-threads', '1',
                       '-loglevel', 'panic',
                       '"%s"' % output_filename]

        command = ' '.join(command)
        print(command)
        # '-vf', "scale=trunc(oh*a/2)*2:480",
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            status ='failed'
            print('divide_clip---- subprocess.CalledProcessError \t',err.output)
            return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    #delete "ignore label" clips
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if '999' in f:
                os.remove(os.path.join(output_dir,f))
    return status, 'Downloaded'

def get_camera_position(video):
    video = camera_pos[camera_pos['filepath']==video].camera_pos.values[0]
    #print(video)
    return video

def get_video_orientation(leftorright,pos):
    flip=False
    keep=False
    if pos=='Left' and leftorright=='Right':
        flip=False
        keep=False
    if pos=='Right' and leftorright=='Right':
        flip=False
        keep=True
    if pos=='Left' and leftorright=='Left':
        flip=False
        keep=True
    if pos=='Right' and leftorright=='Left':
        flip=False
        keep=False
    if pos=='Top-up' and leftorright=='Right':
        flip=True
        keep=True
    if pos=='Top-up' and leftorright=='Left':
        flip=False
        keep=True
    if pos=='Top-b' and leftorright=='Right':
        flip=False
        keep=True
    if pos=='Top-b' and leftorright=='Left':
        flip=True
        keep=True        
    print('pos=',pos, 'leftorright=',leftorright)
    return keep,flip



def process_one_csv(video_src_root,csv_root_dir, csv_file, save_video_root):
    person_ID = csv_file.split('_')[0] # c2
    person_ID = 'C000'+ person_ID[1:] # C0002

    leftorright = csv_file.split('_')[1]
    if leftorright=='RT':
        flip = False
    else:
        flip = True

    video_person_root = os.path.join(save_video_root, person_ID)
    if not os.path.exists(video_person_root):
        os.mkdir(video_person_root)

    activity_ID = csv_file.split('_')[2] # glasses
    if 'RTT' in activity_ID:
        if 'left' in csv_file:
            activity_ID = 'RTT left side'
        else:
            activity_ID = 'RTT right side'
    if 'FM' in activity_ID:
        if 'left' in csv_file:
            activity_ID = 'FM left side'
        else:
            activity_ID = 'FM right side'
    repeatID = csv_file.split(' ')[-1][0] # 5

    if repeatID in ['1', '2', '3', '4', '5', '6']:#C00022_glasses6_1
        video_name_base = person_ID + '_' + activity_ID + repeatID
    elif "C00026" in person_ID and (activity_ID == 'combing'  or activity_ID == 'deodrant' or activity_ID == 'deodorant' or activity_ID == 'glasses' or activity_ID == 'drinking'):
        video_name_base = person_ID + '_' + activity_ID + '1'
    elif "C00027" in person_ID and (activity_ID == 'deodorant' or activity_ID == 'deodrant' or activity_ID == 'drinking' or activity_ID == 'combing' or activity_ID == 'FM left side'):
        video_name_base = person_ID + '_' + activity_ID + '1'
    elif "C00028" in person_ID and (activity_ID == 'glasses' or activity_ID == 'deodorant' or activity_ID == 'deodrant' or activity_ID == 'combing' or activity_ID == 'drinking' or activity_ID == 'FM left side' or activity_ID == 'FM right side'):
        video_name_base = person_ID + '_' + activity_ID + '1'
    elif "C00029" in person_ID and (activity_ID == 'deodorant' or activity_ID == 'deodrant' or activity_ID == 'drinking' or activity_ID == 'combing' or activity_ID == 'face wash' or activity_ID == 'glasses' or activity_ID == 'FM right side' or activity_ID == 'FM left side'):
        video_name_base = person_ID + '_' + activity_ID + '1'
    elif "C00030" in person_ID and (activity_ID == 'combing' or activity_ID == 'drinking' or activity_ID == 'brushing' or activity_ID == 'face wash' or activity_ID == 'glasses' or activity_ID == 'deodorant' or activity_ID == 'deodrant' or activity_ID == 'FM left side'  or activity_ID == 'FM right side'):
        video_name_base = person_ID + '_' + activity_ID + '1'
    elif "C00031" in person_ID and (activity_ID == 'glasses' or activity_ID == 'drinking' or activity_ID == 'face wash' or activity_ID == 'FM left side' or activity_ID == 'FM right side'  or activity_ID == 'deodorant' or activity_ID == 'deodrant'  or activity_ID == 'combing' or activity_ID == 'brushing'):
        video_name_base = person_ID + '_' + activity_ID + '1'
    elif repeatID=='.':#C25_RT_RTT right side_SB_mod_RTT right side 3 .csv
        video_name_base = person_ID + '_' + activity_ID + csv_file.split(' ')[-2][0]
    else:
        video_name_base = person_ID + '_' + activity_ID + 's'
    print(f'person_ID:{person_ID},activity_ID:{activity_ID},repeatID:{repeatID}')
    print(f'video_name_base:{video_name_base}')
    video_names = []
    if person_ID=='C00032':
        return tuple([video_name_base,'no video raw data',video_name_base,'no video raw data'])
    for v in os.listdir(os.path.join(video_src_root, person_ID)):
        if video_name_base in v:
            video_names.append(v)
        elif activity_ID == 'deodorant':
        # deodorant may be incorrectly named
            video_name_base = video_name_base.replace('deodorant','deodrant')
            if video_name_base in v:
                video_names.append(v)
        
    video_name_1 = video_names[0]
    try:
        video_name_2 = video_names[1] #some videos don't have 'xxxx_2'
    except:
        print('only one camera view is recorded')
        
    if video_name_1 not in os.listdir(os.path.join(video_src_root, person_ID)):
        print(f'video_name_1 not in ...  ==== video_name_1:{video_name_1}')

    label, length = load_file(os.path.join(csv_root_dir,csv_file))

    video_source_path = os.path.join(video_src_root, person_ID, video_name_1)
    video_num_frames,FPS = count_frames(video_source_path)

    # if the time
    if abs(int(video_num_frames/FPS) - length) >= 2:
        status1=False
        status2=False
        return tuple([video_name_1,status1,video_name_2,status2])


    video_person_activity_1 = os.path.join(video_person_root,leftorright+video_name_base+'_1')
    if not os.path.exists(video_person_activity_1):
        os.mkdir(video_person_activity_1)    
    pos = get_camera_position(os.path.join(video_src_root,person_ID,video_name_1))
    keep, flip = get_video_orientation(leftorright,pos)
    if keep:
        status1, _ = divide_clip(os.path.join(video_src_root,person_ID,video_name_1), video_person_activity_1, label, length, flip)
    else:
        status1=False
    if len(video_names)>1:
        video_person_activity_2 = os.path.join(video_person_root,leftorright+video_name_base+'_2')
        if not os.path.exists(video_person_activity_2):
            os.mkdir(video_person_activity_2)
        pos = get_camera_position(os.path.join(video_src_root,person_ID,video_name_2))
        keep, flip = get_video_orientation(leftorright,pos)
        if keep:
            status2, _ = divide_clip(os.path.join(video_src_root,person_ID,video_name_2), video_person_activity_2, label, length, flip)
        else:
            status2=False
    else:
        video_name_2=''
        video_person_activity_2=''
    #print(f'video_person_activity_1:{video_person_activity_1},video_person_activity_2:{video_person_activity_2}')

    return tuple([video_name_1,status1,video_name_2,status2])



def main(csvidx,video_src_root,csv_root_dir, save_video_root, activity, num_jobs=24):
    # Download all clips.
    status_lst = []
    csv_to_be_process = []
    
    for f in os.listdir(csv_root_dir):
        csv_to_be_process.append(f)
    for csv_file in [csv_to_be_process[csvidx]]:
        status_lst.append(process_one_csv(video_src_root,csv_root_dir, csv_file, save_video_root))


    with open('download_report_HC.json', 'a') as fobj:
        fobj.write(json.dumps(status_lst))
        fobj.write('\n')




if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--csv_root_dir', type=str, 
                   help=('CSV file containing the following format: '
                         'Identifier,Start time,End time,Class label'))
    p.add_argument('--save_video_root', type=str, 
                   help='Output directory where videos will be saved.')
    p.add_argument('--csvidx', type=int, default=0)
    p.add_argument('--video_src_root', type=str, 
                   help='Raw video directory')
    p.add_argument('--HC_video_position', type=str, 
                   help='csv file for camera position')
    
    main(**vars(p.parse_args()))


