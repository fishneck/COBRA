
# COBRA: Confidence-based Anomaly Detection for Medicine


This website provides materials relevant to the paper:

Yu, B., Kaku, A., Liu, K., Parnandi, A., Fokas, E., Venkatesan, A., Pandit, N., Ranganath, R., Schambra, H. and Fernandez-Granda, C., Quantifying impairment and disease severity using AI models trained on healthy subjects. npj Digit. Med. 7, 180 (2024). 

DOI: https://doi.org/10.1038/s41746-024-01173-x

For more information please visit our website https://fishneck.github.io/COBRA/

# Overview


The paper presents a method to perform automatic assessment of impairment and disease severity using AI models trained only on healthy individuals. The COnfidence-Based chaRacterization of Anomalies (COBRA) score exploits the decrease in confidence of these models when processing data from impaired or diseased patients to quantify their deviation from the healthy population. This diagram explains how the method quantifies impairment in stroke patients (top) and disease severity of knee osteoarthritis:

![plot](https://github.com/fishneck/COBRA/blob/main/COBRA-Overview-Stroke.png)

![plot](https://github.com/fishneck/COBRA/blob/main/COBRA-Overview-KneeOA.png)

The plots below compare the COBRA score against expert-based metrics: (1) Fugl-Meyer assessment (FMA) score for impairment quantification in stroke patients, (2) Kellgren-Lawrence (KL) grade for knee impairment. The COBRA score computed automatically in under a minute is strongly correlated with the expert-based metrics.

![plot](https://github.com/fishneck/COBRA/blob/main/Results_stroke_kneeOA.png)


# Implementation


## Prepare data 

We provide healthy trained model weights and fine-grained model outputs used for calculating and evaluating COBRA on all clinical appliations. For full instruction regarding COBRA data, please visit our [data page](https://github.com/fishneck/COBRA/tree/main/data).




## Calculate COBRA score

After getting fine-grained model outputs, please follow code snippets in `models/*/1 - Calculate_COBRA.ipynb` to calculate COBRA score. 

Note: Replace `*` with `stroke_IMU` or `stroke_video` or `kneeOA` 

Stroke impairment using [wearable sensors](https://github.com/fishneck/COBRA/tree/main/models/stroke_IMU/).

Stroke impairment using [video](https://github.com/fishneck/COBRA/tree/main/models/stroke_video/).

Knee OA Severity using [MRI scans](https://github.com/fishneck/COBRA/tree/main/models/kneeOA/).



## Generate plots

After getting fine-grained model outputs, please follow code snippets in `models/*/2 - Generate_plots.ipynb` to calculate COBRA score. 

Note: Replace `*` with `stroke_IMU` or `stroke_video` or `kneeOA` 

[Stroke impairment using wearable sensors](https://github.com/fishneck/COBRA/tree/main/models/stroke_IMU/).

[Stroke impairment using video](https://github.com/fishneck/COBRA/tree/main/models/stroke_video/).

[Knee OA Severity using MRI scans](https://github.com/fishneck/COBRA/tree/main/models/kneeOA/).




# Clinical Application

### Quantification of Impairment in Stroke Patients

The application of the COBRA score to the impairment quantification in stroke patients was carried out using the publicly available StrokeRehab dataset. StrokeRehab contains wearable-sensor data and video from a cohort of 29 healthy individuals and 51 stroke patients performing multiple trials of 9 rehabilitation activities.

The impairment level of each patient was quantified via the Fugl-Meyer assessment (FMA). The FMA score is a number between 0 (maximum impairment) and 66 (healthy) equal to the sum of itemized scores (each from 0 to 2) for 33 upper body mobility assessments carried out in-clinic by a trained expert.

We provide a full implementation of using COBRA to quantify impairment in stroke patients using video data in `example.ipynb`. This [example](https://github.com/fishneck/COBRA/blob/main/example.ipynb) covers loading healthy model outputs, calculating COBRA scores and generating the plots in the paper.


### Quantification of Severity of Knee Osteoarthritis

The application of the COBRA score to the quantification of knee osteoarthritis (OA) severity was carried out using the publicly available OAI-ZIB dataset. This dataset provides 3D MRI scans of 101 healthy right knees and 378 right knees affected by knee osteoarthritis (OA), a long-term degenerative joint condition.


Each knee is labeled with the corresponding Kellgren-Lawrence (KL) grades, retrieved from the NIH Osteoarthritis Initiative collection. The KL grade quantifies OA severity on a scale from 0 (healthy) to 4 (severe).

We developed a medical segmentation model to predict pixel-wise tissue type on healthy knees. We adopted a Multi-Planar U-Net architecture. We used the model confidence for cartilage tissues (femur cartilage, tibia cartilage) to calculate COBRA score. 
