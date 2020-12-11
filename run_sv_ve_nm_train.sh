#!/bin/bash
#SBATCH --time=6:15:00
# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=22G
#SBATCH -J svvenm1
# Specify an output file
#SBATCH -o svvenm1.out
#SBATCH -e svvenm1.err
# Set up the environment by loading modules
# Run a script
source ../ecg_env/bin/activate
python train.py --model_path="sv_ve_nm.pth.tar" --epochs=8 --lr=.0001 --train_path="/gpfs/data/ceickhof/ecg_data/data/sv_ve_nm_train.txt" --val_path="/gpfs/data/ceickhof/ecg_data/data/sv_ve_nm_test.txt" --outsz=3
