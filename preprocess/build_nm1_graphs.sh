#!/bin/bash 
#SBATCH -n 8 
#SBATCH --mem=50G
#SBATCH -t 30:00:00

python3 preprocess_nm1.py
