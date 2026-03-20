#!/bin/bash
#SBATCH --job-name=stanza_24
#SBATCH --partition=gpu-p1        # Changed to CPU partition
#SBATCH --nodelist=puck6          # Target puck1 as requested
#SBATCH --cpus-per-task=4        # Increased to 22 processors
#SBATCH --mem=8G                 # Increased memory slightly for 22 workers
#SBATCH --output=stanza_%j.out
#SBATCH --export=ALL
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --time=1:00:00            # Set a reasonable time limit for the job

echo "Job started at $(date)!"

source .venv/bin/activate
python run_nlp.py

echo "Job completed at $(date)!"