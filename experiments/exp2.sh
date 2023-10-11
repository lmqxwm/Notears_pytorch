#!/bin/bash
 
#---------------------------------------------------------------------------------
# Account information
 
#SBATCH --account=pi-naragam              # basic (default), phd, faculty, pi-<account>
 
#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=8G           # requested memory
#SBATCH --time=2-00:00:00          # wall clock limit (d-hh:mm:ss)


#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=exp2    # user-defined job name

#---------------------------------------------------------------------------------
# Print some useful variables

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python/booth/3.10
module load cuda/11.4


#---------------------------------------------------------------------------------
# Commands to execute below...

python3 -u exp2.py