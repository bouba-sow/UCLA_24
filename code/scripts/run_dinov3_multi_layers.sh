#!/bin/bash
set -euo pipefail

ROOT="/store/scratch/bsow/Documents/UCLA_24"
SCRIPT="${ROOT}/code/scripts/extract_dinov3_frame_features.py"
VIDEO="${ROOT}/data/40m_act_24_S06E01_30fps.m4v"
BASE_OUT_DIR="${ROOT}/data/features/dinov3_vithplus_distilled/s06e01"

# 1-based layer numbers requested by user.
LAYERS=(1 8 16 24 32)
TOTAL_LAYERS=32

for LAYER_NUM in "${LAYERS[@]}"; do
  if (( LAYER_NUM < 1 || LAYER_NUM > TOTAL_LAYERS )); then
    echo "Invalid 1-based layer number ${LAYER_NUM}; expected 1..${TOTAL_LAYERS}" >&2
    exit 1
  fi

  # Python script expects 0-based layer index.
  LAYER_INDEX=$((LAYER_NUM - 1))
  LAYER_TAG="layer_${LAYER_NUM}"

  OUT_DIR="${BASE_OUT_DIR}/${LAYER_TAG}/frame_npy"
  MANIFEST_CSV="${BASE_OUT_DIR}/${LAYER_TAG}/frame_npy_paths.csv"
  MANIFEST_TXT="${BASE_OUT_DIR}/${LAYER_TAG}/frame_npy_paths.txt"
  METADATA_JSON="${BASE_OUT_DIR}/${LAYER_TAG}/extraction_metadata.json"

  echo "============================================================"
  echo "Running DINOv3 extraction for layer ${LAYER_NUM} (index ${LAYER_INDEX})"
  echo "Output directory: ${OUT_DIR}"
  echo "============================================================"

  python3 "${SCRIPT}" \
    --video-path "${VIDEO}" \
    --batch-size 300 \
    --image-size 518 \
    --device cuda \
    --layer-index "${LAYER_INDEX}" \
    --output-dir "${OUT_DIR}" \
    --manifest-csv "${MANIFEST_CSV}" \
    --manifest-txt "${MANIFEST_TXT}" \
    --metadata-json "${METADATA_JSON}"
done

echo "All requested layers completed: ${LAYERS[*]}"
