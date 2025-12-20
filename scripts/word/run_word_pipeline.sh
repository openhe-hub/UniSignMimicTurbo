#!/usr/bin/env bash
# Pipeline to run word-level filtering and boundary extraction for a single folder.
# Defaults target: AslToHiya-01 under output/word_level/word_videos.

set -euo pipefail

has_jpg() {
  local dir=$1
  [[ -d "${dir}" ]] && compgen -G "${dir}"/*.jpg > /dev/null
}

# Configurable parameters (override via env vars if needed)
WORD_ID=${WORD_ID:-AslToHiya-01}
MP4_ROOT=${MP4_ROOT:-output/word_level/mp4_asltohiya}
FRAMES_ROOT=${FRAMES_ROOT:-output/word_level/frames_asltohiya}
FILTER1_ROOT=${FILTER1_ROOT:-output/word_level/frames_asltohiya_filtered1}
FILTER2_ROOT=${FILTER2_ROOT:-output/word_level/frames_asltohiya_filtered2}
BOUNDARY_ROOT=${BOUNDARY_ROOT:-output/word_level/word_boundary_frames}
REF_IMAGE=${REF_IMAGE:-assets/example_data/images/test5_576x576.jpg}

DUP_THRESHOLD=${DUP_THRESHOLD:-3.0}
MIN_DUP_LEN=${MIN_DUP_LEN:-2}
HAND_THRESHOLD=${HAND_THRESHOLD:-0.8}
HEAD_THRESHOLD=${HEAD_THRESHOLD:-0.9}
HAND_HEIGHT_THRESHOLD=${HAND_HEIGHT_THRESHOLD:-0.1}
DEVICE=${DEVICE:-cuda}

echo "[INFO] WORD_ID=${WORD_ID}"

MP4_SRC="output/word_level/word_videos/${WORD_ID}"
FRAMES_DIR="${FRAMES_ROOT}/${WORD_ID}"
FILTER1_DIR="${FILTER1_ROOT}/${WORD_ID}"
FILTER2_DIR="${FILTER2_ROOT}/${WORD_ID}"
BOUNDARY_DIR="${BOUNDARY_ROOT}/${WORD_ID}"

if [[ ! -d "${MP4_SRC}" ]]; then
  echo "[ERROR] Source MP4 folder not found: ${MP4_SRC}"
  exit 1
fi

if [[ ! -f "${REF_IMAGE}" ]]; then
  echo "[ERROR] Reference image not found: ${REF_IMAGE}"
  exit 1
fi

echo "[INFO] Staging MP4 root: ${MP4_ROOT}"
mkdir -p "${MP4_ROOT}"
if [[ ! -e "${MP4_ROOT}/${WORD_ID}" ]]; then
  ln -s ../word_videos/"${WORD_ID}" "${MP4_ROOT}/${WORD_ID}"
fi

if has_jpg "${FRAMES_DIR}"; then
  echo "[SKIP] Extract frames (existing JPGs in ${FRAMES_DIR})"
else
  echo "[STEP] Extract frames"
  python scripts/sentence/extract_all_frames_seq.py \
    --mp4-root "${MP4_ROOT}" \
    --out-root "${FRAMES_ROOT}"
fi

if has_jpg "${FILTER1_DIR}"; then
  echo "[SKIP] Filter duplicates (existing JPGs in ${FILTER1_DIR})"
else
  if ! has_jpg "${FRAMES_DIR}"; then
    echo "[ERROR] No input frames found for duplicate filter in ${FRAMES_DIR}"
    exit 1
  fi
  echo "[STEP] Filter duplicates"
  python scripts/sentence/filter_duplicate_frames.py \
    --frames-dir "${FRAMES_ROOT}" \
    --subfolder "${WORD_ID}" \
    --save-cleaned-frames \
    --output-dir "${FILTER1_ROOT}" \
    --duplicate-threshold "${DUP_THRESHOLD}" \
    --min-duplicate-length "${MIN_DUP_LEN}"
fi

if has_jpg "${FILTER2_DIR}"; then
  echo "[SKIP] Filter by pose (existing JPGs in ${FILTER2_DIR})"
else
  if ! has_jpg "${FILTER1_DIR}"; then
    echo "[ERROR] No input frames found for pose filter in ${FILTER1_DIR}"
    exit 1
  fi
  echo "[STEP] Filter by pose"
  python scripts/sentence/filter_frames_by_pose.py \
    --frames-dir "${FILTER1_ROOT}" \
    --subfolder "${WORD_ID}" \
    --save-filtered \
    --output-dir "${FILTER2_ROOT}" \
    --hand-threshold "${HAND_THRESHOLD}" \
    --head-threshold "${HEAD_THRESHOLD}" \
    --hand-height-threshold "${HAND_HEIGHT_THRESHOLD}" \
    --device "${DEVICE}"
fi

if has_jpg "${BOUNDARY_DIR}"; then
  echo "[SKIP] Extract boundary frames (existing JPGs in ${BOUNDARY_DIR})"
else
  if ! has_jpg "${FILTER2_DIR}"; then
    echo "[ERROR] No input frames found for boundary extraction in ${FILTER2_DIR}"
    exit 1
  fi
  echo "[STEP] Extract boundary frames (word-level)"
  python scripts/word/extract_boundary_frames.py \
    --frames-root "${FILTER2_ROOT}" \
    --ref-image "${REF_IMAGE}" \
    --out-root "${BOUNDARY_ROOT}" \
    --subfolder "${WORD_ID}"
fi

echo "[DONE] Pipeline finished for ${WORD_ID}"
