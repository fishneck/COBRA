# Data for COBRA

This direcotory contains data preparation instructions from Quantifying Impairment and Disease Severity Using AI Models Trained on Healthy Subjects by Boyang Yu, Aakash Kaku, Kangning Liu, Avinash Parnandi, Emily Fokas, Anita Venkatesan, Natasha Pandit, Rajesh Ranganath, Heidi Schambra and Carlos Fernandez-Granda. 

For more information please visit our website https://fishneck.github.io/COBRA/



# Clinical Applications

We provide healthy trained model outputs and model weights. Raw data and model training instructions are also be included. 

## Stroke Impairment - Wearable sensors 

Model outputs for wearable sensor are shared via Google Drive([link](https://drive.google.com/drive/folders/1YBgIZJhYRgd7IiChn7yWOsT6HCIKYPhl?usp=drive_link)). 

Stroke patients' clinical assessment scores is provided in this [repo](https://drive.google.com/drive/folders/1tbpq0z6C5aGIdJRrIuF_jAAoN8SWc3KZ?usp=drive_link).

After downloading files, there should be a directory structure like this

```tree
root ── Stroke_IMU/
      ├─ fold1/...
      ├─ fold2/...
      ├─ fold3/...
      ├─ fold4/...
      ├─ fold5/ ───  predictions_loss_model.p
      │           ├─ predictions_loss_model.stroke.p
      │           └─ best_loss.prm
      ├ stroke_raw.pkl
      └ healthy_raw.pkl
```

Each `fold*/` contains prediction for 1) held-out healthy subjects 2) stroke subjects and 3) model weights for that fold. Two `.pkl` files contains processed output for calculating COBRA.

Please use codes in this [repo](https://github.com/fishneck/COBRA/tree/main/models/stroke_IMU) for generating COBRA score.

Raw wearble sensor data and patient meta data can also be accessed from [StrokeRehab](https://simtk.org/projects/primseq) public directory.

To train a model from scratch on healthy individuals' inertial measurement units (IMUs) data, please follow instructions in [ASRF](https://github.com/yiskw713/asrf). 


## Stroke Patients - Video

Model outputs for videos are shared via Google Drive([link](https://drive.google.com/drive/folders/1tbpq0z6C5aGIdJRrIuF_jAAoN8SWc3KZ?usp=share_link). 

Stroke patients' clinical assessment scores is provided in this [repo](https://drive.google.com/drive/folders/1tbpq0z6C5aGIdJRrIuF_jAAoN8SWc3KZ?usp=drive_link).

After downloading files, there should be a directory structure like this

```tree
root ── Stroke_video/
      ├─ fold1/...
      ├─ fold2/...
      ├─ fold3/...
      └─ fold4/ ──── test.tar.gz
                  ├─ test_stroke.tar.gz
                  └─ test_blurred/ ──── test_blurred_healthy.tar.gz
                                     └─ test_blurred_stroke.tar.gz
      
```

Each `fold*/` contains predictions for 1) held-out healthy subjects 2) stroke subjects 3) held-out healthy subjects' blurred clips and 4) stroke subjects' blurred clips for that fold. Feel free to use `tar -xzvf file.tar.gz` to extract model outputs.

Please use codes in this [repo](https://github.com/fishneck/COBRA/tree/main/models/stroke_video) for generating COBRA score.

To train a model from scratch on healthy individuals' video data, please follow instructions in [SlowFast](https://github.com/facebookresearch/SlowFast). 


## Knee Osteoarthritis - MRI

We provide 5 sample model outputs via Google Drive [link](https://drive.google.com/drive/folders/1KK473GI1OF2U44euHYA9fVIxsYKoTZsW?usp=drive_link). You can use segmentation model weights in [here](https://drive.google.com/drive/folders/1cBWEblKSqg1uN88ZRWC7ikKmOLTYa-HC?usp=drive_link) to generate all predictions after downloading raw imaging data from [OAI-ZIB](https://pubdata.zib.de/).

For the ease of implementation, please organize the files like this

```tree
root ── OAI_ZIB/
      ├─ train/...
      ├─ val/...
      └─ test/ ──── images/ ──── xxxxxx.nii
                  │           ├─ xxxxxx.nii
                  │           ├─ xxxxxx.nii
                  │           └─ ...
                  └─ labels/ ──── xxxxxx.nii
                              ├─ xxxxxx.nii
                              ├─ xxxxxx.nii
                              └─ ...
```

Knee Osteoarthritis patients' clinical assessment scores is provided in this [repo](https://github.com/fishneck/COBRA/tree/main/data/kneeOA). We only include 479 scans with valid clinical assessment scores from the entire 507 scans in OAI-ZIB.





To train a model from scratch on healthy individuals' video data, please follow instructions in [Multi-Planar-Unet](https://github.com/perslev/MultiPlanarUNet).


