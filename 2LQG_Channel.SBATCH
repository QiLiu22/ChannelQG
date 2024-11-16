#!/bin/bash

#SBATCH --nodes=2
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=0-06:00:00
#SBATCH --job-name=2LayQG
#SBATCH --mail-type=END
#SBATCH --mail-user=ql2221@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --array=2

module purge

cd /scratch/ql2221/dedalus_sing/Channel
srun /scratch/work/sd3201/dedalus3/run-dedalus-ryanver.bash python 2LQG_channel.py $SLURM_ARRAY_TASK_ID
