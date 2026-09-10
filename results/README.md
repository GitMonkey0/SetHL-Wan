# Result artifacts

The public release contains metrics and experiment manifests, but not WLASL
frames, cached Wan latents, model checkpoints, or generated MP4 files.

- `experiment_lock.json` at the repository root fixes the dataset hashes,
  methods, seeds, masks, inference budget, and primary endpoint.
- `generation_subset_content_disjoint.json` and
  `temperature_validation_subset_content_disjoint.json` are the
  source-video selections made without inspecting generated results.
- `generated_content_disjoint/{interval,full}/*/sample_*.json` are per-source reports. Training
  seeds are averaged within a source before any source-level bootstrap.
- `denoising_content_disjoint/{interval,full}/*.json` cover all 265 test sources
  for SetHL and continuous control.
- `final_video_feature_consistency_content_disjoint.json` records the explicitly post-hoc,
  source-conditioned R3D-18 diagnostic at control scale 1.0.
- `final_generation_summary_content_disjoint.json` is the machine-readable source for
  `paper/results_content_disjoint.tex`; the latter is never edited by hand.
- `temperature_validation_content_disjoint/selection.json` records validation-only posterior
  temperature selection.
Run `python release_audit_content_disjoint.py --evidence-only` on the data-free
public release, or omit `--evidence-only` after full reproduction with checkpoints, to
verify the locked hashes, complete grids, paper macros, page geometry, and PDF
fonts.
