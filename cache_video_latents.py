#!/usr/bin/env python3
"""Cache frozen Wan VAE latents for prepared RGB windows on Ascend NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
VIDEOX_FUN_ROOT = Path(os.environ.get("VIDEOX_FUN_ROOT", ROOT / "vendor" / "VideoX-Fun"))
for path in (ROOT / "vendor" / "python", VIDEOX_FUN_ROOT, ROOT):
    sys.path.insert(0, str(path))

from videox_fun.models import AutoencoderKLWan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get(
        "WAN_MODEL_PATH", ROOT / "models" / "Wan2.1-Fun-V1.1-1.3B-Control")))
    parser.add_argument("--config", type=Path,
                        default=VIDEOX_FUN_ROOT / "config" / "wan2.1" / "wan_civitai.yaml")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("npu"):
        torch.npu.set_device(args.device)
    device = torch.device(args.device)

    cfg = OmegaConf.load(args.config)
    vae = AutoencoderKLWan.from_pretrained(
        str(args.model / cfg.vae_kwargs.vae_subpath),
        additional_kwargs=OmegaConf.to_container(cfg.vae_kwargs),
    ).to(device).eval()
    records = [json.loads(x) for x in (args.windows / "manifest.jsonl").read_text().splitlines()]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    records = records[args.shard_index::args.num_shards]
    if args.limit:
        records = records[:args.limit]
    output_rows = []
    started = time.time()
    with torch.no_grad():
        for index, row in enumerate(records):
            sample = torch.load(args.windows / row["path"], weights_only=True)
            rgb = sample["rgb"]
            if rgb.dtype == torch.uint8:
                rgb = rgb.float().div(127.5).sub(1.0)
            rgb = rgb.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            clean = vae.encode(rgb)[0].mode()
            first = vae.encode(rgb[:, :, :1])[0].mode()
            first_grid = torch.zeros_like(clean)
            first_grid[:, :, :1] = first
            out_path = args.output / row["path"]
            torch.save({
                "clean_latent": clean.cpu().to(torch.bfloat16),
                "first_frame_latent": first_grid.cpu().to(torch.bfloat16),
                "joints": sample["joints"],
                "visible": sample["visible"],
                "source_clip": sample["source_clip"],
                "frame_indices": sample["frame_indices"],
            }, out_path)
            output_rows.append({**row, "path": out_path.name})
            print(f"[{index + 1}/{len(records)}] {row['id']}", flush=True)
    manifest_name = ("manifest.jsonl" if args.num_shards == 1 else
                     f"manifest.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl")
    manifest = args.output / manifest_name
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows))
    summary = {
        "windows": len(output_rows),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "seconds": time.time() - started,
        "device": args.device,
        "vae_checkpoint_sha256": hashlib.sha256(
            (args.model / cfg.vae_kwargs.vae_subpath).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    summary_name = ("summary.json" if args.num_shards == 1 else
                    f"summary.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json")
    (args.output / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
