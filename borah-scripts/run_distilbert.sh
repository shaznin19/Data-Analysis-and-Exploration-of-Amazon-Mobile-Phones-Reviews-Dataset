#!/bin/bash
#SBATCH -J distilbert_Results      # job name
#SBATCH -o distilbert_1.o%j      # output and error file name (%j expands to jobID)
#SBATCH -n 48                   # total number of tasks requested
#SBATCH -N 1                    # number of nodes you want to run on
#SBATCH -p gpu                  # queue (partition) -- defq, eduq, gpuq, shortq
#SBATCH --gres=gpu:1            # request one gpu
#SBATCH -t 120:00:00             # run time (hh:mm:ss)

module load slurm
module load cudnn8.0-cuda11.0/8.0.5.39

# Activate the conda environment
. ~/.bashrc
conda activate llm_env
pip3 install scikit-learn matplotlib seaborn

# Your code goes here
python distilbert_amazon.py
