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

camera_pos = pd.read_csv(args.HAR_video_position_path)

#which hand is impaired
subject_whichhand = {'s1': 'Right',
 's2': 'Left',
 's3': 'Left',
 's4': 'Left',
 's5': 'Left',
 's6': 'Right',
 's7': 'Right',
 's8': 'Left',
 's9': 'Left',
 's10': 'Left',
 's11': 'Right',
 's12': 'Right',
 's13': 'Right',
 's16': 'Right',
 's17': 'Right',
 's18': 'Left',
 's19': 'Left',
 's20': 'Left',
 's21': 'Left',
 's22': 'Left',
 's23': 'Left',
 's24': 'Left',
 's25': 'Right',
 's26': 'Left',
 's27': 'Left',
 's28': 'Left',
 's29': 'Right',
 's30': 'Left',
 's31': 'Right',
 's32': 'Right',
 's33': 'Left',
 's34': 'Right',
 's35': 'Right',
 's36': 'Right',
 's37': 'Right',
 's39': 'Right',
 's40': 'Left',
 's41': 'Right',
 's42': 'Left',
 's43': 'Left',
 's44': 'Left',
 's45': 'Right',
 's46': 'Right',
 's47': 'Right',
 's48': 'Right',
 's49': 'Right',
 's50': 'Left',
 's51': 'Left',
 's53': 'Left',
 's54': 'Left',
 's55': 'Left'}

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

        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            status ='failed,'
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
    """
    Args:
        csv_root_dir:
        csv_file:

    Returns:
    s2_glasses_RT_SB_mod_glasses 5.csv
    """
    person_ID = csv_file.split('_')[0] # s2
    person_ID = 'S000'+ person_ID[1:] # S0002


    video_person_root = os.path.join(save_video_root, person_ID)
    if not os.path.exists(video_person_root):
        os.mkdir(video_person_root)


    activity_ID = csv_file.split('_')[1] # glasses
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

    if repeatID in ['1', '2', '3', '4', '5']:
        video_name_base = person_ID + '_' + activity_ID + repeatID
    else:
        video_name_base = person_ID + '_' + activity_ID + 's'

    video_names = []
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


    # if the time doesn't match
    if abs(int(video_num_frames/FPS) - length) >= 2:
        return tuple([video_name_1,"frame doesn't match",video_name_2,"frame doesn't match"])
    
    # if recorded motion does not belong to the impaired hand
    paretic_hand = subject_whichhand[csv_file.split('_')[0]]#which hand is impaired
    df = pd.read_csv(os.path.join(csv_root_dir,csv_file), usecols= ['hand_left'])
    motion_hand = 'Left' if pd.read_csv(os.path.join(csv_root_dir,csv_file), usecols= ['hand_left'])['hand_left'][0]>0 else 'Right'
    if motion_hand!=paretic_hand:
        return tuple([video_name_1,"motion/paretic hand doesn't match",video_name_2,"motion/paretic hand doesn't match"])
    
     
    video_person_activity_1 = os.path.join(video_person_root,video_name_1.split('.')[0])
    if not os.path.exists(video_person_activity_1):
        os.mkdir(video_person_activity_1)
    pos = get_camera_position(os.path.join(video_src_root,person_ID,video_name_1))
    keep, flip = get_video_orientation(motion_hand,pos)
    print(video_src_root,person_ID,video_name_1,video_person_activity_1)
    if keep:
        status1, _ = divide_clip(os.path.join(video_src_root,person_ID,video_name_1), video_person_activity_1, label, length, flip)
    else:
        status1=False
    
    if len(video_names)>1:
        video_person_activity_2 = os.path.join(video_person_root,video_name_2.split('.')[0])
        if not os.path.exists(video_person_activity_2):
            os.mkdir(video_person_activity_2)
        pos = get_camera_position(os.path.join(video_src_root,person_ID,video_name_2))
        keep, flip = get_video_orientation(motion_hand,pos)
        if keep:
            status2, _ = divide_clip(os.path.join(video_src_root,person_ID,video_name_2), video_person_activity_2, label, length, flip)
        else:
            status2=False
    else:
        video_name_2=''
        video_person_activity_2=''
    print(f'video_person_activity_1:{video_person_activity_1},video_person_activity_2:{video_person_activity_2}')

    return tuple([video_name_1,status1,video_name_2,status2])



def main(csvidx,video_src_root,csv_root_dir, save_video_root, activity, num_jobs=24):
    status_lst = []
    csv_to_be_process = []
    for f in os.listdir(csv_root_dir):
        csv_to_be_process.append(f)
    for csv_file in [csv_to_be_process[csvidx]]:
        status_lst.append(process_one_csv(video_src_root,csv_root_dir, csv_file, save_video_root))

    with open('download_report.json', 'a') as fobj:
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
    p.add_argument('--HAR_video_position', type=str, 
                   help='csv file for camera position')
    
    main(**vars(p.parse_args()))
