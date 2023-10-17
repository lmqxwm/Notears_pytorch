#!/bin/sh

#SBATCH --job-name=exp2
#SBATCH --account=pi-naragam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# TO USE V100 specify --constraint=v100
# TO USE RTX600 specify --constraint=rtx6000
#SBATCH --constraint=v100   # constraint job runs on V100 GPU use
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu
#SBATCH --cpus-per-task=1   # set this to the desired number of threads

# LOAD MODULES
module load tensorflow
module load cuda/11.7

# DO COMPUTE WORK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python training.py