#!/bin/bash
#SBATCH --time=6:15:00
# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=22G
#SBATCH -J ar_binclass
# Specify an output file
#SBATCH -o arbin1.out
#SBATCH -e arbin1.err
# Set up the environment by loading modules
# Run a script
source ../ecg_env/bin/activate
python train_ar.py --model_path="arbin1" --epochs=1
