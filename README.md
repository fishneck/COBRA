
This github repo contains instructions, codes, and results from Quantifying Impairment and Disease Severity Using AI Models Trained on Healthy Subjects by Boyang Yu, Aakash Kaku, Kangning Liu, Avinash Parnandi, Emily Fokas, Anita Venkatesan, Natasha Pandit, Rajesh Ranganath, Heidi Schambra and Carlos Fernandez-Granda.

For more information please visit our website https://fishneck.github.io/COBRA/

# Overview

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
