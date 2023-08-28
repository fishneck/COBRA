#!/bin/bash

#SBATCH --job-name=train_asrf
#SBATCH --nodes=1
#SBATCH --array=0-4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition gpu4_medium
#SBATCH --time=35:59:59         
#SBATCH --output=slurm/%x-%A-%a-%j.out
#SBATCH --error=slurm/%x-%A-%a-%j.err

module purge
module load python/gpu/3.7.6

cd /gpfs/scratch/by2026/asrf_hc

i=0;

for s in "1" "2" "3" "4" "5";
    do
        for a in "all" 
            do
                sub[$i]=$s;
                act[$i]=$a;
                i=$(($i+1));
            done
    done

f=${sub[$SLURM_ARRAY_TASK_ID]}
activity=${act[$SLURM_ARRAY_TASK_ID]}


echo ${f}
echo ${activity}

python train.py ./result/sensors_${activity}/in_channel-77_param_search-True_dataset-sensors_${activity}_split-${f}/config.yaml 
