#!/bin/bash
#SBATCH --job-name=decode_concepts
#SBATCH --output=/store/scratch/elefranc/Documents/UCLA_24/decode_concepts_%j.out
#SBATCH --error=/store/scratch/elefranc/Documents/UCLA_24/decode_concepts_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu-p1,gpu-p2
#SBATCH --gres=gpu:1

# Run from the project root (absolute path required — Slurm copies the script)
cd /store/scratch/elefranc/Documents/UCLA_24

# Activate environment if needed
# source .venv/bin/activate

PY=${PY:-/store/scratch/elefranc/Documents/UCLA_24/.venv/bin/python}
CLUSTERLESS=${CLUSTERLESS:-data/clusterless/sub-572_ses-01_clusterless.npz}
CSV=${CSV:-data/24_S06E01_8concepts_merged.csv}

# Build the clusterless cache once if missing (CPU-bound, heavy I/O).
if [ ! -f "$CLUSTERLESS" ]; then
    echo "Clusterless cache not found; building it..."
    $PY code2/src/decoding/preprocessing.py --n-jobs "${SLURM_CPUS_PER_TASK:-8}" --out "$CLUSTERLESS"
fi

$PY code2/scripts/decode_concepts.py \
    --clusterless_npz "$CLUSTERLESS" \
    --csv_path "$CSV" \
    --epochs 49 --batch_size 256 --num_workers "${SLURM_CPUS_PER_TASK:-8}"
