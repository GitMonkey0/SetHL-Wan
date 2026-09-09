#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 NPU_INDEX CHECKPOINT_DIR" >&2
  exit 2
fi
npu_index=$1
checkpoint_dir=$2
root=$(cd "$(dirname "$0")" && pwd)
subset="$root/results/temperature_validation_subset.json"
latents="$root/data_sources/wlasl/latents_hand256"
checkpoint="$checkpoint_dir/checkpoint-3000.pt"

for temperature in 1 1.5 2; do
  tag=${temperature/./p}
  while read -r sample_index; do
    generation="$root/results/temperature_validation/temp${tag}/sample_${sample_index}"
    report="$root/results/temperature_validation/temp${tag}/sample_${sample_index}.json"
    if [[ ! -f "$generation/generation.json" ]]; then
      python "$root/generate_sethl_wan.py" \
        --latents "$latents" --checkpoint "$checkpoint" --representation sethl \
        --split val --sample-index "$sample_index" --mask-policy interval \
        --inference-steps 30 --seeds 0 1 2 3 --control-scales 0 .5 1 1.5 \
        --posterior-temperature "$temperature" --device "npu:${npu_index}" \
        --output "$generation"
    fi
    python "$root/evaluate_generated_videos.py" \
      --generation "$generation" --latents "$latents" --output "$report"
  done < <(jq -r '.selected[].sample_index' "$subset")
done

python - "$root/results/temperature_validation" <<'PY'
import glob,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); result={}
for directory in sorted(root.glob('temp*')):
    rows=[json.load(open(p)) for p in directory.glob('sample_*.json')]
    result[directory.name]={'sources':len(rows),
        'mean_frontier_hypervolume':float(np.mean([r['frontier_hypervolume'] for r in rows]))}
best=max(result, key=lambda x: result[x]['mean_frontier_hypervolume'])
output={'selection_split':'val','selection_metric':'mean_frontier_hypervolume',
        'candidates':result,'selected':best}
(root/'selection.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
PY
