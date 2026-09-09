#!/usr/bin/env python3
"""Read-only architecture and LoRA-budget audit for the local Wan checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open


LORA_SUFFIXES = (".q.weight", ".k.weight", ".v.weight",
                 ".ffn.0.weight", ".ffn.2.weight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    checkpoint = args.model_dir / "diffusion_pytorch_model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as file:
        shapes = {key: file.get_slice(key).get_shape() for key in file.keys()}
    total = sum(math.prod(shape) for shape in shapes.values())
    targets = {key: shape for key, shape in shapes.items()
               if key.startswith("blocks.") and key.endswith(LORA_SUFFIXES)}
    result = {
        "model_dir": str(args.model_dir),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "tensor_count": len(shapes),
        "parameter_count": total,
        "config": {key: config[key] for key in (
            "num_layers", "dim", "ffn_dim", "num_heads", "in_dim", "out_dim",
            "patch_size", "text_dim", "text_len", "add_ref_conv")},
        "lora_target_suffixes": list(LORA_SUFFIXES),
        "lora_target_matrix_count": len(targets),
        "lora_parameter_count": {
            str(rank): sum((shape[0] + shape[1]) * rank for shape in targets.values())
            for rank in (8, 16, 32, 64)
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
