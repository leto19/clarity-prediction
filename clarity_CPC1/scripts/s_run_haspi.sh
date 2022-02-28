#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH  --account=dcs-res
#SBATCH --partition=dcs-gpu
#SBATCH --nodes=1
#SBATCH --time=80:00:00
#SBATCH --gpus-per-node=1

#load the modules
module load Anaconda3/5.3.0
module load fosscuda/2019b  # includes GCC 8.3
module load imkl/2019.5.281-iimpi-2019b
module load CMake/3.15.3-GCCcore-8.3.0
module load MATLAB/2021a/binary



# python env



source activate speechbrain
srun --export=ALL python3  calculate_SI_haspi.py metadata/CPC1.train.json ../../../clarity_CPC1_data/clarity_data/scenes/ ../../../clarity_CPC1_data/clarity_data/HA_outputs/train/ metadata/listeners.CPC1_train.json haspi.train.csvls

