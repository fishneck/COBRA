import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import cv2
import argparse

def save_blurry_video(input_video_path, resize = 128):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_clip_path = input_video_path.split('/')[0]
    outcrop = cv2.VideoWriter(input_video_path.replace(video_clip_path,video_clip_path+'_blurry'), fourcc, 60.0, (512, 512))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            #print(frame.shape, ret)
            center = cv2.resize(frame, (resize, resize))
            center = cv2.resize(center, (512, 512))
            outcrop.write(center)   
            #cv2.imshow("frame", frame)
            cv2.waitKey(1)
        else:
            break

    cap.release()
    outcrop.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set Transformer ++ with positional embedding analysis for hematoma expansion prediction")
    parser.add_argument("--video_clip_path", type=str,
                        help="directory to original video clips to be blurred")
    parser.add_argument("--blurry_corrupted_file", type=str,
                        help="file to save blurred video clips paths")
    args = parser.parse_args()
    
    activity = args.activity

    #blur all subject
    sub = os.listdir(args.video_clip_path)
    for s in sub:
        print(s)
        for v in os.listdir(args.video_clip_path+'/' +s):
            print(v)
            original_dir = args.video_clip_path + '/' + s + '/' + v
            blurred_dir = args.video_clip_path + '_blurry/' + s + '/' + v
            os.makedirs(blurred_dir,exist_ok=True)
            if len(os.listdir(blurred_dir))<len(original_dir):            
                for clip in os.listdir(original_dir):
                    save_blurry_video(original_dir +'/'+clip)


    blurry_corrupted_file = args.blurry_corrupted_file 
    lines = []
    os.makedirs('/'.join(blurry_corrupted_file.split('/')[:-1]),exist_ok=True)

    sub = os.listdir(args.video_clip_path+'_blurry') 
    for s in sub:
        print(s)
        for v in os.listdir(args.video_clip_path+'_blurry/'+s):
            for clip in os.listdir(args.video_clip_path+'_blurry/'+s+'/'+v):
                if int(clip.split('e')[0][1:])%2==0:
                    lines.append(args.video_clip_path+'_blurry/'+s+'/'+v+'/'+clip + ' ' + clip.split('_')[1].split('.')[0]+'\n')


    with open(blurry_corrupted_file,'w') as f:
        f.writelines(lines)

