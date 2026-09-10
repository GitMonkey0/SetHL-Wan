# SetHL-Wan

Research code for **SetHL: Partially Specified Hand Articulation Control for
Video Diffusion**. SetHL turns Hand Labanotation (HL) into an editable control
interface for Wan2.1-Fun-Control: a user fixes selected bone/time symbols while
the model completes and varies the remaining articulation.

The key operation is not ordinary quantization. A hard straight-through sample
selects one of the fixed 26 HL directions, after which SetHL samples a
continuous direction strictly inside that symbol's spherical Voronoi cell.
Thus geometry can vary without changing the requested symbol. Wrist paths,
palm frames, and reference bone lengths are always supplied; full joint
coordinates never bypass the symbolic stream.

This directory is separate from the frozen first-paper submission and does not
modify it.

## Layout

- `DESIGN_LOCK.md`: research question, falsification rule, and evaluation lock.
- `prepare_wlasl.py`: deterministic WLASL clip/landmark preprocessing.
- `cache_video_latents.py`: Wan-VAE latent cache construction.
- `pretrain_completion.py`: masked symbolic or continuous completion.
- `train_sethl_wan.py`: joint bridge and rank-16 Wan LoRA training.
- `run_training_seed.sh`: exact per-method training recipe used in the paper.
- `kinematic_spatializer.py`: certified within-HL-cell sampling and kinematics.
- `generate_sethl_wan.py`: reproducible ancestral video generation.
- `evaluate_completion.py`, `evaluate_denoising.py`,
  `evaluate_generated_videos.py`, and `evaluate_video_features.py`: held-out
  metrics and the explicitly post-hoc video-feature diagnostic.
- `aggregate_generation_results.py`: paired source-level bootstrap.
- `build_public_release.py`: whitelist-only, data-free repository export.
- `paper/`: ICASSP manuscript and figures.

## Data and model

The current protocol uses one 17-frame, 256-by-256 hand-focused clip per WLASL
source video. Exact decoded-RGB deduplication removes 193 records and prevents
content overlap across the resulting 1,402/392/265 train/validation/test split.
WLASL RGB files are governed by their original license and are not
redistributed. The backbone is
`Wan2.1-Fun-V1.1-1.3B-Control`; its VAE and 1.3B backbone are frozen, while
rank-16 LoRA, the completion network, and the compact control bridge train.
All compared methods use zero text and CLIP embeddings, isolating first-frame
and motion-control conditioning.

## Reproduction

Commands below assume execution from this repository's root and an Ascend PyTorch
environment with `torch_npu`, the Wan model, and WLASL sources available at the
paths passed on the command line. The shell wrappers additionally require
`git` and `jq`; paper packaging uses `tectonic` and `zip`.

```bash
python -m pip install -r requirements.txt
```

Install the CANN-matched `torch` and `torch-npu` wheels separately as described
in `ENVIRONMENT.md`.

Prepare the exact VideoX-Fun revision and its two-line Ascend complex-dtype
patch, then point the scripts to the upstream code and downloaded model:

```bash
./setup_videox_fun.sh vendor/VideoX-Fun
export VIDEOX_FUN_ROOT=$PWD/vendor/VideoX-Fun
export WAN_MODEL_PATH=/path/to/Wan2.1-Fun-V1.1-1.3B-Control
```

The model weights are available as
`alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control` on Hugging Face or ModelScope and
are not redistributed here. WLASL videos likewise remain subject to the
dataset's original terms.

Generated-video evaluation additionally uses the versions pinned in
`requirements-eval.txt`; in particular MediaPipe 0.10.21 requires a compatible
protobuf runtime.

Install `requirements-eval.txt` in a separate evaluation environment before
running the generated-video evaluator. The core unit tests deliberately keep
MediaPipe and VideoX-Fun imports lazy, so they can run before those optional
components are installed.

```bash
# Validate all manifests and implementation invariants.
PYTHONPATH=. pytest -q

# Remove exact decoded-RGB duplicates before fitting or training anything.
python deduplicate_wlasl_rgb.py \
  --windows data_sources/wlasl/windows_hand256 \
  --latents data_sources/wlasl/latents_hand256 \
  --source-manifest data_sources/wlasl/windows_hand256/manifest.jsonl \
  --output data_sources/wlasl/latents_hand256_content_disjoint

# Pretrain the masked SetHL completion model.
python pretrain_completion.py \
  --latents data_sources/wlasl/latents_hand256_content_disjoint \
  --representation sethl --angular-weight 1 --seed 2027 \
  --output runs_content_disjoint/pretrain_sethl_aw1_seed2027.pt

# Jointly train SetHL and Wan LoRA on one NPU.
python train_sethl_wan.py \
  --latents data_sources/wlasl/latents_hand256_content_disjoint \
  --representation sethl --steps 3000 --rank 16 --seed 2027 \
  --completion-init runs_content_disjoint/pretrain_sethl_aw1_seed2027.pt \
  --angular-weight 1 --device npu:0 \
  --output runs_content_disjoint/wlasl_hand_cell_sethl_seed2027

# Equivalent locked wrapper (repeat for seeds 2027, 2028, and 2029).
bash run_training_content_disjoint.sh sethl 2027 0
```

Run matched commands for `continuous`, `vq`, `hardhl`, `sethl_nohier` (the
no-spherical-loss ablation), and `raster`; VQ additionally requires
`--codebook`. Exact masks and random
seeds are serialized with every generated sample. Statistical aggregation
resamples source videos rather than treating diffusion samples as independent.

The reported generation grid uses three training seeds, four paired diffusion
seeds, four control scales, and all 25 test videos satisfying the locked
two-hand visibility criterion. Posterior temperature is selected only on the
eight-source validation manifest in
`results/temperature_validation_subset_content_disjoint.json`.

## Reproduce the paper tables

Run posterior calibration once on validation data, then generate every locked
test condition. The wrappers resume completed samples rather than overwriting
them.

```bash
export SETHL_LATENTS=$PWD/data_sources/wlasl/latents_hand256_content_disjoint
export SETHL_SUBSET=$PWD/results/generation_subset_content_disjoint.json
export SETHL_GENERATION_ROOT=$PWD/results/generated_content_disjoint
export SETHL_CODEBOOK=$PWD/data_sources/wlasl/vq26_codebook_content_disjoint.pt
export SETHL_TEMPERATURE_SUBSET=$PWD/results/temperature_validation_subset_content_disjoint.json
export SETHL_TEMPERATURE_ROOT=$PWD/results/temperature_validation_content_disjoint
export SETHL_DENOISING_ROOT=$PWD/results/denoising_content_disjoint

bash run_temperature_validation.sh 0 runs_content_disjoint/wlasl_hand_cell_sethl_seed2027

# Repeat each method for seeds 2027, 2028, and 2029 on available NPUs.
bash run_generation_shard.sh sethl 2027 0 runs_content_disjoint/wlasl_hand_cell_sethl_seed2027
bash run_generation_shard.sh continuous 2027 0 runs_content_disjoint/wlasl_hand_pre_continuous_seed2027

# Full-control comparison.
bash run_generation_shard.sh sethl 2027 0 runs_content_disjoint/wlasl_hand_cell_sethl_seed2027 full

# Held-out denoising evaluation (both interval and full policies).
bash run_denoising_shard.sh sethl 2027 0 runs_content_disjoint/wlasl_hand_cell_sethl_seed2027
```

After all six methods and three training seeds finish, aggregate at the source
video level and generate the LaTeX macros used verbatim by the paper:

```bash
python aggregate_denoising_results.py \
  --root results/denoising_content_disjoint --policy interval \
  --method sethl --baseline continuous \
  --output results/final_denoising_sethl_vs_continuous_interval_content_disjoint.json

# Post-hoc source-consistency diagnostic at control scale 1.0. The official
# torchvision R3D-18 Kinetics-400 V1 weights have SHA-256
# b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d.
python evaluate_video_features.py \
  --generation-root results/generated_content_disjoint/interval \
  --windows data_sources/wlasl/windows_hand256 \
  --weights model_cache/hub/checkpoints/r3d_18-b3b3357e.pth \
  --device npu:0 --output results/final_video_feature_consistency_content_disjoint.json

python summarize_content_disjoint.py \
  --root results/generated_content_disjoint/interval \
  --full-root results/generated_content_disjoint/full \
  --subset results/generation_subset_content_disjoint.json \
  --codebook-results results/codebook_test_content_disjoint.json \
  --temperature-selection results/temperature_validation_content_disjoint/selection.json \
  --denoising-results results/final_denoising_sethl_vs_continuous_interval_content_disjoint.json \
  --video-feature-results results/final_video_feature_consistency_content_disjoint.json \
  --output-json results/final_generation_summary_content_disjoint.json \
  --output-tex paper/results_content_disjoint.tex

cd paper && make
cd .. && python release_audit_content_disjoint.py --evidence-only
```

The distributed repository omits large checkpoints and RGB videos, so the
command above audits all released evidence. After reproducing the checkpoints
locally, omit `--evidence-only` to additionally verify every final checkpoint.

MediaPipe evaluation is best installed in a separate environment using
`requirements-eval.txt`, because its protobuf constraint may conflict with
unrelated packages in a general-purpose NPU image.

## Current evidence policy

The locked evaluation did **not** pass the preregistered joint
frontier-superiority gate, and the manuscript makes no such claim. The supported
finding is more diagnostic: hard HL significantly improves compliance over
continuous control but sacrifices localized diversity; probabilistic SetHL
significantly recovers that diversity while retaining accuracy, leakage, and
detection gains over continuous control. SetHL also reduces conditional
denoising MSE on all 265 content-disjoint test clips. R3D-18 similarity is
reported only as a post-hoc diagnostic. `paper/results_content_disjoint.tex` is
generated directly from archived reports; no result is inferred from training
loss.
