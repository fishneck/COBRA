
This github repo contains instructions, codes, and results from Quantifying Impairment and Disease Severity Using AI Models Trained on Healthy Subjects by Boyang Yu*, Aakash Kaku*, Kangning Liu*, Avinash Parnandi, Emily Fokas, Anita Venkatesan, Natasha Pandit, Rajesh Ranganath, Heidi Schambra and Carlos Fernandez-Granda [* - Equal Contribution].

For more information please visit our website https://fishneck.github.io/COBRA/

# Overview

## Prepare data 

For full information regarding COBRA data, please visit our [data page](https://github.com/fishneck/COBRA/tree/main/data).

**Stroke Impairment** 

For COBRA score calculation, we utilized healthy-trained action segmentation model outputs on held-out test set. 

Model outputs for [wearable sensor](https://drive.google.com/drive/folders/1YBgIZJhYRgd7IiChn7yWOsT6HCIKYPhl?usp=drive_link) and [video](https://drive.google.com/drive/folders/1tbpq0z6C5aGIdJRrIuF_jAAoN8SWc3KZ?usp=drive_link) are shared. Stroke patients' clinical assessment scores is provided in this [repo](https://github.com/fishneck/COBRA/tree/main/data/Stroke).

Raw wearble sensor data and patient meta data can also be downloaded from [StrokeRehab](https://simtk.org/projects/primseq) public directory. 


**Knee OA Severity** 

For COBRA score calculation, we utilized healthy-trained medical imaging segmentation model outputs on held-out test set. 

Segmentation [model weights](https://drive.google.com/file/d/1KIppYLu1i3HN_d985rB7H8CugHy26K_o/view?usp=drive_link) and [sample model output](https://drive.google.com/drive/folders/1KK473GI1OF2U44euHYA9fVIxsYKoTZsW?usp=drive_link) are provided. Raw MRI data and raw patient meta data are publicly available at OAI-ZIB [website](https://pubdata.zib.de/) and NIH-OAI [database](https://nda.nih.gov/oai/). We provide a pre-processed version of patient meta data in this [repo](https://github.com/fishneck/COBRA/tree/main/data/kneeOA).

To fully replicate the COBRA score in Knee OA replication, please follow [Multi-Planar UNet](https://github.com/perslev/MultiPlanarUNet) instructions to generate test prediction using provided model weights and model configuration file.


## Calculate COBRA score

After getting fine-grained model outputs, please follow code snippets in `models/*/1 - Calculate_COBRA.ipynb` to calculate COBRA score. 

Note: Replace `*` with `stroke_IMU` or `stroke_video` or `kneeOA` for 3 applications


## Generate plots

After getting fine-grained model outputs, please follow code snippets in `models/*/2 - Generate_plots.ipynb` to calculate COBRA score. 

Note: Replace `*` with `stroke_IMU` or `stroke_video` or `kneeOA` for 3 applications



# Clinical Application

### Quantification of Impairment in Stroke Patients

The application of the COBRA score to the impairment quantification in stroke patients was carried out using the publicly available StrokeRehab dataset. StrokeRehab contains wearable-sensor data and video from a cohort of 29 healthy individuals and 51 stroke patients performing multiple trials of 9 rehabilitation activities.

The impairment level of each patient was quantified via the Fugl-Meyer assessment (FMA). The FMA score is a number between 0 (maximum impairment) and 66 (healthy) equal to the sum of itemized scores (each from 0 to 2) for 33 upper body mobility assessments carried out in-clinic by a trained expert.

We provide a full implementation of using COBRA to quantify impairment in stroke patients using video data in `example.ipynb`


### Quantification of Severity of Knee Osteoarthritis

The application of the COBRA score to the quantification of knee osteoarthritis (OA) severity was carried out using the publicly available OAI-ZIB dataset. This dataset provides 3D MRI scans of 101 healthy right knees and 378 right knees affected by knee osteoarthritis (OA), a long-term degenerative joint condition.


Each knee is labeled with the corresponding Kellgren-Lawrence (KL) grades, retrieved from the NIH Osteoarthritis Initiative collection. The KL grade quantifies OA severity on a scale from 0 (healthy) to 4 (severe).

We developed a medical segmentation model to predict pixel-wise tissue type on healthy knees. We adopted a Multi-Planar U-Net architecture. We used the model confidence for cartilage tissues (femur cartilage, tibia cartilage) to calculate COBRA score. 
