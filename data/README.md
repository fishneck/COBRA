# Data for COBRA

This direcotory contains source data from Quantifying Impairment and Disease Severity Using AI Models Trained on Healthy Subjects by Boyang Yu*, Aakash Kaku*, Kangning Liu*, Avinash Parnandi, Emily Fokas, Anita Venkatesan, Natasha Pandit, Rajesh Ranganath, Heidi Schambra and Carlos Fernandez-Granda [* - Equal Contribution].

We provide model output for each clinical application and postprocessing pipelines for COBRA scores and analysis. For more information please visit our website https://fishneck.github.io/COBRA/

## Application

### Quantification of Impairment in Stroke Patients

The application of the COBRA score to the impairment quantification in stroke patients was carried out using the publicly available StrokeRehab dataset. StrokeRehab contains wearable-sensor data and video from a cohort of 29 healthy individuals and 51 stroke patients performing multiple trials of 9 rehabilitation activities.

The impairment level of each patient was quantified via the Fugl-Meyer assessment (FMA). The FMA score is a number between 0 (maximum impairment) and 66 (healthy) equal to the sum of itemized scores (each from 0 to 2) for 33 upper body mobility assessments carried out in-clinic by a trained expert.


#### Stroke Patients - Wearable sensors

We trained a model to identify functional primitives from healthy individuals' inertial measurement units (IMUs) data. We utilized a Multi-Stage Temporal Convolutional Network (MS-TCN). We used the model confidence for motion related primitives (transport, reposition, reach) to calculate COBRA score. The model is trained using 5-fold cross validation on healthy subjects. Held-out helathy subjects(id=\[C0004, C0015, C0023, C0030\]) and all stroke subjects are saved for evaluating COBRA score.

To access model output, download wearable sensors model output from [here](https://drive.google.com/drive/folders/1YBgIZJhYRgd7IiChn7yWOsT6HCIKYPhl?usp=drive_link). Patient's FMA score is stored in [here](https://github.com/fishneck/COBRA/tree/main/data/Stroke). After extracting files, using `tar -xzvf file.tar.gz`, use codes in this [repo](https://github.com/fishneck/COBRA/tree/main/models/stroke_IMU) for generating COBRA score.


#### Stroke Patients - Video

We performed functional primitive identification from healthy individuals' video data. We utilized the X3D model, a 3D convolutional neural network designed for primitive classification from video data. We used the model confidence for motion related primitives (transport, reposition, reach) to calculate COBRA score. The model is trained using 4-fold cross validation on healthy subjects. Held-out helathy subjects(id=\[C0004, C0015, C0023, C0030\]) and all stroke subjects are saved for evaluating COBRA score.

To access model output, download video model output from [here](https://drive.google.com/drive/folders/1tbpq0z6C5aGIdJRrIuF_jAAoN8SWc3KZ?usp=drive_link). Patient's FMA score is stored in [here](https://github.com/fishneck/COBRA/tree/main/data/Stroke). For visual confounding factors, please use model output under `test_blur/` directory. After extracting files, using `tar -xzvf file.tar.gz`, use codes in this [repo](https://github.com/fishneck/COBRA/tree/main/models/stroke_IMU) for generating COBRA score. 


### Quantification of Severity of Knee Osteoarthritis

The application of the COBRA score to the quantification of knee osteoarthritis (OA) severity was carried out using the publicly available OAI-ZIB dataset. This dataset provides 3D MRI scans of 101 healthy right knees and 378 right knees affected by knee osteoarthritis (OA), a long-term degenerative joint condition.

![image](https://github.com/fishneck/COBRA/blob/main/Data-KneeOA.png)


Each knee is labeled with the corresponding Kellgren-Lawrence (KL) grades, retrieved from the NIH Osteoarthritis Initiative collection. The KL grade quantifies OA severity on a scale from 0 (healthy) to 4 (severe).

We developed a medical segmentation model to predict pixel-wise tissue type on healthy knees. We adopted a Multi-Planar U-Net architecture. We used the model confidence for cartilage tissues (femur cartilage, tibia cartilage) to calculate COBRA score. 


We provide sample segmentation outputs in [here](https://drive.google.com/drive/folders/1KK473GI1OF2U44euHYA9fVIxsYKoTZsW?usp=drive_link) and segmentation model weights in [here](https://drive.google.com/drive/folders/1cBWEblKSqg1uN88ZRWC7ikKmOLTYa-HC?usp=drive_link). Feel free to follow [Multi-Planar-Unet](https://github.com/perslev/MultiPlanarUNet) instructions and use model weights to generate all outputs. Patient's KL grade and train-validation-test split is stored in [here](https://github.com/fishneck/COBRA/tree/main/data/kneeOA).

Codes for generating COBRA score is in [here](https://github.com/fishneck/COBRA/tree/main/models/kneeOA).


