#!/bin/bash 
#SBATCH -n 8 
#SBATCH --mem=40G
#SBATCH -t 30:00:00

python3 preprocess_nm2.py
