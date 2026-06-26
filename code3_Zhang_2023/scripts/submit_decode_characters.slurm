#!/bin/bash
#SBATCH --job-name=zhang2023_decode
#SBATCH --output=/store/scratch/elefranc/Documents/UCLA_24/zhang2023_decode_%j.out
#SBATCH --error=/store/scratch/elefranc/Documents/UCLA_24/zhang2023_decode_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-p1,gpu-p2
#SBATCH --gres=gpu:1

# Zhang 2023 character decoding (code3) — Oberon GPU (same layout as code2).
# Usage:
#   sbatch code3_Zhang_2023/scripts/submit_decode_characters.slurm
#
# Overrides, e.g.:
#   EPOCHS=3 N_FOLDS=2 sbatch code3_Zhang_2023/scripts/submit_decode_characters.slurm

set -euo pipefail

cd /store/scratch/elefranc/Documents/UCLA_24

PY=${PY:-/store/scratch/elefranc/Documents/UCLA_24/.venv/bin/python}
LABELS_CSV=${LABELS_CSV:-data/40m_act_24_S06E01_30fps_characters.csv}
SUB=${SUB:-572}
SES=${SES:-01}
EPOCHS=${EPOCHS:-100}
N_FOLDS=${N_FOLDS:-5}
BATCH_SIZE=${BATCH_SIZE:-256}
OUTPUT_DIR=${OUTPUT_DIR:-results/code3_Zhang_2023}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Job started at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "PY=$PY"
echo "labels=$LABELS_CSV  sub=$SUB ses=$SES  epochs=$EPOCHS folds=$N_FOLDS"

mkdir -p "$OUTPUT_DIR"

"$PY" code3_Zhang_2023/scripts/decode_characters.py \
    --bids-dir data/bids \
    --sub "$SUB" \
    --ses "$SES" \
    --labels-csv "$LABELS_CSV" \
    --epochs "$EPOCHS" \
    --n-folds "$N_FOLDS" \
    --batch-size "$BATCH_SIZE" \
    --device cuda \
    --output-dir "$OUTPUT_DIR"

echo "Job completed at $(date)"
