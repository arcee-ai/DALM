#!/usr/bin/bash

#SBATCH --job-name=falcon
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=Partition
#SBATCH --cpus-per-task=8
#SBATCH -n 1
#SBATCH -N 1


source ~/anaconda3/bin/activate torch

python data/unlabelled/falcon_refinedweb.py
