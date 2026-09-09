#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
target=${1:-"$root/vendor/VideoX-Fun"}
if [[ -e "$target" ]]; then
  echo "target already exists: $target" >&2
  exit 1
fi
mkdir -p "$(dirname "$target")"
git clone https://github.com/aigc-apps/VideoX-Fun.git "$target"
git -C "$target" checkout 968f0e2192ba4c7a12868bf36d73260d135424ca
git -C "$target" apply "$root/patches/videox_fun_ascend.patch"
echo "VideoX-Fun prepared at $target"
echo "export VIDEOX_FUN_ROOT=$target"
