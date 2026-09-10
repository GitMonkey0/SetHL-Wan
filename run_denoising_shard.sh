#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 REPRESENTATION TRAINING_SEED NPU_INDEX CHECKPOINT_DIR" >&2
  exit 2
fi

representation=$1
training_seed=$2
npu_index=$3
checkpoint_dir=$4
root=$(cd "$(dirname "$0")" && pwd)
latents="${SETHL_LATENTS:-$root/data_sources/wlasl/latents_hand256}"
denoising_root="${SETHL_DENOISING_ROOT:-$root/results/denoising}"
checkpoint="$checkpoint_dir/checkpoint-3000.pt"
[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
extra=()
if [[ "$representation" == "vq" ]]; then
  extra=(--codebook "${SETHL_CODEBOOK:-$root/data_sources/wlasl/vq26_codebook.pt}")
fi

for policy in interval full; do
  output="$denoising_root/${policy}/${representation}_seed${training_seed}.json"
  if [[ -f "$output" ]]; then
    echo "already complete: $output"
    continue
  fi
  python "$root/evaluate_denoising.py" \
    --latents "$latents" \
    --checkpoint "$checkpoint" --representation "$representation" \
    --mask-policy "$policy" --compare-zero-control --device "npu:${npu_index}" \
    --output "$output" "${extra[@]}"
done

echo "completed denoising ${representation} seed ${training_seed}"
