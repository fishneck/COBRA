
This github repo contains instructions, codes, and results from Quantifying Impairment and Disease Severity Using AI Models Trained on Healthy Subjects by Boyang Yu*, Aakash Kaku*, Kangning Liu*, Avinash Parnandi, Emily Fokas, Anita Venkatesan, Natasha Pandit, Rajesh Ranganath, Heidi Schambra and Carlos Fernandez-Granda [* - Equal Contribution].

For more information please visit our website https://fishneck.github.io/COBRA/

# Overview

## Prepare data

**Stroke Impairment** - For COBRA score calculation, we utilized healthy-trained action segmentation model outputs on held-out test set. Model outputs for wearable sensor data can be downladed from [here](https://drive.google.com/drive/folders/1YBgIZJhYRgd7IiChn7yWOsT6HCIKYPhl?usp=drive_link). Model outputs for video data can be downladed from [here](https://drive.google.com/drive/folders/1tbpq0z6C5aGIdJRrIuF_jAAoN8SWc3KZ?usp=drive_link). Stroke patients' clinical assessment scores is provided in this [repo](https://github.com/fishneck/COBRA/tree/main/data/Stroke). Raw wearble sensor data and patient meta data can also be downloaded from [StrokeRehab](https://simtk.org/projects/primseq) public directory. 

**Knee OA Severity** - For COBRA score calculation, we utilized healthy-trained medical imaging segmentation model outputs on held-out test set. Segmentation model weights can be downladed from [here](https://drive.google.com/file/d/1KIppYLu1i3HN_d985rB7H8CugHy26K_o/view?usp=drive_link). Due to the large volume of data, we provide some [sample model output](https://drive.google.com/drive/folders/1KK473GI1OF2U44euHYA9fVIxsYKoTZsW?usp=drive_link) files for illustration. Raw MRI data is publicly available at OAI-ZIB [website](https://pubdata.zib.de/). Raw meta data is publicly available at NIH-OAI [database]([https://pubdata.zib.de/](https://nda.nih.gov/oai/). We provide a pre-processed version of patient meta data in this [repo](https://github.com/fishneck/COBRA/tree/main/data/kneeOA).

To fully replicate the COBRA score in Knee OA replication, please follow [Multi-Planar UNet official repo](https://github.com/perslev/MultiPlanarUNet) to generate test prediction using provided model weights and model configuration file.


## Calculate COBRA score



### Generate plots



# Quantifying Impairment and Disease Severity Using AI Models Trained on Healthy Subjects

Automatic assessment of impairment and disease severity is a key challenge in data-driven medicine. We propose a novel framework to address this challenge, which leverages AI models trained exclusively on healthy subjects. The models are designed to predict a clinically-meaningful attribute of the healthy patients. When presented with data where the attribute is affected by the medical condition of interest, the models experience a decrease in confidence that can be used to quantify deviation from the healthy population. The resulting **COnfidence-Based chaRacterization of Anomalies (COBRA)** score was applied to quantification of upper-body motion impairment in stroke patients, and severity of knee osteoarthritis from magneticresonance imaging scans.


## Application

### Quantification of Impairment in Stroke Patients

The application of the COBRA score to the impairment quantification in stroke patients was carried out using the publicly available StrokeRehab dataset. StrokeRehab contains wearable-sensor data and video from a cohort of 29 healthy individuals and 51 stroke patients performing multiple trials of 9 rehabilitation activities.

The impairment level of each patient was quantified via the Fugl-Meyer assessment (FMA). The FMA score is a number between 0 (maximum impairment) and 66 (healthy) equal to the sum of itemized scores (each from 0 to 2) for 33 upper body mobility assessments carried out in-clinic by a trained expert.

#### Stroke Patients - Wearable sensors


We trained a model to identify functional primitives from healthy individuals' inertial measurement units (IMUs) data. We utilized a Multi-Stage Temporal Convolutional Network (MS-TCN). We used the model confidence for motion related primitives (transport, reposition, reach) to calculate COBRA score. Source code for model training and calculating COBRA score is in [here](https://github.com/fishneck/COBRA/tree/main/models/stroke_IMU).


#### Stroke Patients - Video


We performed functional primitive identification from healthy individuals' video data. We utilized the X3D model, a 3D convolutional neural network designed for primitive classification from video data. We used the model confidence for motion related primitives (transport, reposition, reach) to calculate COBRA score. Source code for model training and calculating COBRA score is in [here](https://github.com/fishneck/COBRA/tree/main/models/stroke_video).


### Quantification of Severity of Knee Osteoarthritis

The application of the COBRA score to the quantification of knee osteoarthritis (OA) severity was carried out using the publicly available OAI-ZIB dataset. This dataset provides 3D MRI scans of 101 healthy right knees and 378 right knees affected by knee osteoarthritis (OA), a long-term degenerative joint condition.

![image](https://github.com/fishneck/COBRA/blob/main/Data-KneeOA.png)


Each knee is labeled with the corresponding Kellgren-Lawrence (KL) grades, retrieved from the NIH Osteoarthritis Initiative collection. The KL grade quantifies OA severity on a scale from 0 (healthy) to 4 (severe).

We developed a medical segmentation model to predict pixel-wise tissue type on healthy knees. We adopted a Multi-Planar U-Net architecture. We used the model confidence for cartilage tissues (femur cartilage, tibia cartilage) to calculate COBRA score. Source code for model training and calculating COBRA score is in [here](https://github.com/fishneck/COBRA/tree/main/examples/kneeOA).

