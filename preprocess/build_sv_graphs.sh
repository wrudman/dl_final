#!/bin/bash
#SBATCH -n 20
#SBATCH --mem=50G
#SBATCH -t 24:00:00

python3 preprocess_sv.py
