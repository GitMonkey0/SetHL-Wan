#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "usage: $0 REPRESENTATION TRAINING_SEED NPU_INDEX CHECKPOINT_DIR [MASK_POLICY] [POSTERIOR_TEMPERATURE] [OUTPUT_LABEL]" >&2
  exit 2
fi

representation=$1
training_seed=$2
npu_index=$3
checkpoint_dir=$4
mask_policy=${5:-interval}
posterior_temperature=${6:-1}
output_label=${7:-$representation}
root=$(cd "$(dirname "$0")" && pwd)
latents="$root/data_sources/wlasl/latents_hand256"
checkpoint="$checkpoint_dir/checkpoint-3000.pt"
subset="$root/results/generation_subset.json"
base="$root/results/generated/${mask_policy}/${output_label}_seed${training_seed}"

[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
mapfile -t indices < <(jq -r '.selected[].sample_index' "$subset")
extra=()
if [[ "$representation" == "vq" ]]; then
  extra=(--codebook "$root/data_sources/wlasl/vq26_codebook.pt")
fi
scales=(0 0.5 1 1.5)
if [[ "$mask_policy" == "full" ]]; then
  scales=(1)
fi

for sample_index in "${indices[@]}"; do
  generation="$base/sample_${sample_index}"
  report="$base/sample_${sample_index}.json"
  if [[ ! -f "$generation/generation.json" ]]; then
    python "$root/generate_sethl_wan.py" \
      --latents "$latents" --checkpoint "$checkpoint" \
      --representation "$representation" --sample-index "$sample_index" \
      --mask-policy "$mask_policy" --inference-steps 30 \
      --seeds 0 1 2 3 --control-scales "${scales[@]}" \
      --posterior-temperature "$posterior_temperature" \
      --device "npu:${npu_index}" --output "$generation" "${extra[@]}"
  fi
  python "$root/evaluate_generated_videos.py" \
    --generation "$generation" --latents "$latents" --output "$report"
done

echo "completed ${representation} seed ${training_seed} ${mask_policy} on ${#indices[@]} sources"
