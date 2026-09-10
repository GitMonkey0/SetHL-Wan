#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 {sethl|sethl_nohier|hardhl|vq|continuous|raster} SEED NPU_INDEX" >&2
  exit 2
fi
method=$1
seed=$2
device="npu:$3"
root=$(cd "$(dirname "$0")" && pwd)
latents="$root/data_sources/wlasl/latents_hand256_content_disjoint"
codebook="$root/data_sources/wlasl/vq26_codebook_content_disjoint.pt"
run_root="$root/runs_content_disjoint"
mkdir -p "$run_root"
common=(--latents "$latents" --steps 3000 --seed "$seed" --device "$device")
train_common=(--latents "$latents" --steps 3000 --rank 16 --seed "$seed"
              --lr 1e-4 --wan-lr 1e-5 --mask-policy mixed --device "$device")

case "$method" in
  sethl)
    init="$run_root/pretrain_sethl_aw1_seed${seed}.pt"
    [[ -f "$init" ]] || python "$root/pretrain_completion.py" "${common[@]}" \
      --representation sethl --angular-weight 1 --output "$init"
    python "$root/train_sethl_wan.py" "${train_common[@]}" \
      --representation sethl --angular-weight 1 --completion-init "$init" \
      --output "$run_root/wlasl_hand_cell_sethl_seed${seed}"
    ;;
  sethl_nohier)
    init="$run_root/pretrain_sethl_aw0_seed${seed}.pt"
    [[ -f "$init" ]] || python "$root/pretrain_completion.py" "${common[@]}" \
      --representation sethl --angular-weight 0 --output "$init"
    python "$root/train_sethl_wan.py" "${train_common[@]}" \
      --representation sethl_nohier --angular-weight 0 --completion-init "$init" \
      --output "$run_root/wlasl_hand_cell_no_spherical_seed${seed}"
    ;;
  hardhl)
    init="$run_root/pretrain_sethl_aw1_seed${seed}.pt"
    [[ -f "$init" ]] || python "$root/pretrain_completion.py" "${common[@]}" \
      --representation sethl --angular-weight 1 --output "$init"
    python "$root/train_sethl_wan.py" "${train_common[@]}" \
      --representation hardhl --angular-weight 1 --completion-init "$init" \
      --output "$run_root/wlasl_hand_pre_hardhl_seed${seed}"
    ;;
  vq)
    init="$run_root/pretrain_vq_aw1_seed${seed}.pt"
    [[ -f "$init" ]] || python "$root/pretrain_completion.py" "${common[@]}" \
      --representation vq --angular-weight 1 --codebook "$codebook" --output "$init"
    python "$root/train_sethl_wan.py" "${train_common[@]}" \
      --representation vq --angular-weight 1 --completion-lr 1e-5 \
      --completion-init "$init" --codebook "$codebook" \
      --output "$run_root/wlasl_hand_stable_vq_seed${seed}"
    ;;
  continuous)
    init="$run_root/pretrain_continuous_seed${seed}.pt"
    [[ -f "$init" ]] || python "$root/pretrain_completion.py" "${common[@]}" \
      --representation continuous --output "$init"
    python "$root/train_sethl_wan.py" "${train_common[@]}" \
      --representation continuous --completion-init "$init" \
      --output "$run_root/wlasl_hand_pre_continuous_seed${seed}"
    ;;
  raster)
    python "$root/train_sethl_wan.py" "${train_common[@]}" \
      --representation raster --output "$run_root/wlasl_hand_raster_seed${seed}"
    ;;
  *) echo "unknown method: $method" >&2; exit 2 ;;
esac
