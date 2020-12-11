#!/bin/bash
#SBATCH -n 10
#SBATCH --mem=50G
#SBATCH -t 24:00:00

python3 preprocess_ar.py
