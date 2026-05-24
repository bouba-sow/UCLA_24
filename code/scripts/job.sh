#!/bin/bash
#SBATCH --job-name=dinov3_s06e01
#SBATCH --partition=gpu-p1        # Changed to CPU partition
#SBATCH --nodelist=puck6          # Target puck1 as requested
#SBATCH --cpus-per-task=4        # Increased to 22 processors
#SBATCH --mem=45G                 # Increased memory slightly for 22 workers
#SBATCH --output=/store/scratch/bsow/Documents/UCLA_24/code/scripts/dinov3_%j.out
#SBATCH --export=ALL
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --time=10:00:00            # Set a reasonable time limit for the job

echo "Job started at $(date)!"

source .venv/bin/activate

bash /store/scratch/bsow/Documents/UCLA_24/code/scripts/run_dinov3_multi_layers.sh

echo "Job completed at $(date)!"