#!/bin/bash
#SBATCH --job-name=zhang2023_bis
#SBATCH --output=/store/scratch/elefranc/Documents/UCLA_24/zhang2023_bis_%j.out
#SBATCH --error=/store/scratch/elefranc/Documents/UCLA_24/zhang2023_bis_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-p1,gpu-p2
#SBATCH --gres=gpu:1

set -euo pipefail
cd /store/scratch/elefranc/Documents/UCLA_24

PY=${PY:-/store/scratch/elefranc/Documents/UCLA_24/.venv/bin/python}
LABELS_CSV=${LABELS_CSV:-data/40m_act_24_S06E01_30fps_characters.csv}
OUTPUT_DIR=${OUTPUT_DIR:-results/code4_Zhang_2023_bis}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

"$PY" code4_Zhang_2023_bis/scripts/decode_characters.py \
    --labels-csv "$LABELS_CSV" \
    --epochs "${EPOCHS:-100}" \
    --n-folds "${N_FOLDS:-5}" \
    --device cuda \
    --output-dir "$OUTPUT_DIR"
