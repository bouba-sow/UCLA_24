#!/bin/bash
#SBATCH --job-name=zhang2023_full
#SBATCH --output=zhang2023_full_%j.out
#SBATCH --error=zhang2023_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
cd /store/scratch/elefranc/Documents/UCLA_24

source ~/.bashrc 2>/dev/null || true
conda activate torch 2>/dev/null || true

# Vision is slow on full video — use reference bootstrap for stage 3, or pre-built labels.
python code3_Zhang_2023/scripts/run_full_pipeline.py \
    --video data/40m_act_24_S06E01_30fps.m4v \
    --reference-csv data/40m_act_24_S06E01_30fps_characters.csv \
    --bids-dir data/bids \
    --sub 572 --ses 01 \
    --device cuda \
    --epochs 100 \
    --skip-vision \
    --labels-csv results/code3_Zhang_2023/vision/characters_30fps.csv
