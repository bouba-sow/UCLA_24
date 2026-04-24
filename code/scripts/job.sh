#!/bin/bash
#SBATCH --job-name=dinov3_s06e01
#SBATCH --partition=gpu-p2        # Changed to CPU partition
#SBATCH --nodelist=puck7          # Target puck1 as requested
#SBATCH --cpus-per-task=4        # Increased to 22 processors
#SBATCH --mem=32G                 # Increased memory slightly for 22 workers
#SBATCH --output=/store/scratch/bsow/Documents/UCLA_24/code/scripts/dinov3_%j.out
#SBATCH --export=ALL
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --time=10:00:00            # Set a reasonable time limit for the job

echo "Job started at $(date)!"

source .venv/bin/activate

python3 /store/scratch/bsow/Documents/UCLA_24/code/scripts/extract_dinov3_frame_features.py \
  --video-path /store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps.m4v \
  --batch-size 32 \
  --image-size 518 \
  --device cuda

echo "Job completed at $(date)!"