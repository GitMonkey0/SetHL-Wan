# Result artifacts

The public release contains metrics and experiment manifests, but not WLASL
frames, cached Wan latents, model checkpoints, or generated MP4 files.

- `experiment_lock.json` at the repository root fixes the dataset hashes,
  methods, seeds, masks, inference budget, and primary endpoint.
- `generation_subset.json` and `temperature_validation_subset.json` are the
  source-video selections made without inspecting generated results.
- `generated/{interval,full,finger}/*/sample_*.json` are per-source reports. Training
  seeds are averaged within a source before any source-level bootstrap.
- `denoising/{interval,full}/*.json` cover all 269 test sources.
- `final_video_feature_consistency.json` records the explicitly post-hoc,
  source-conditioned R3D-18 diagnostic at control scale 1.0.
- `final_generation_summary.json` is the machine-readable source for
  `paper/results.tex`; the latter is never edited by hand.
- `temperature_validation/selection.json` records validation-only posterior
  temperature selection.
- `policy_validation_selection.json` records validation-only selection of the
  whole-finger robustness condition;
  `final_generation_sethl_vs_continuous_finger.json` records its one-shot
  three-seed test comparison.

Run `python release_audit.py --evidence-only` on the data-free public release,
or `python release_audit.py` after full reproduction with checkpoints, to
verify the locked hashes, complete grids, paper macros, page geometry, and PDF
fonts.
