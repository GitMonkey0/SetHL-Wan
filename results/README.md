# Result artifacts

The public release contains metrics and experiment manifests, but not WLASL
frames, cached Wan latents, model checkpoints, or generated MP4 files.

- `experiment_lock.json` at the repository root fixes the dataset hashes,
  methods, seeds, masks, inference budget, and primary endpoint.
- `generation_subset.json` and `temperature_validation_subset.json` are the
  source-video selections made without inspecting generated results.
- `generated/{interval,full}/*/sample_*.json` are per-source reports. Training
  seeds are averaged within a source before any source-level bootstrap.
- `denoising/{interval,full}/*.json` cover all 269 test sources.
- `final_generation_summary.json` is the machine-readable source for
  `paper/results.tex`; the latter is never edited by hand.
- `temperature_validation/selection.json` records validation-only posterior
  temperature selection.

Run `python release_audit.py --evidence-only` on the data-free public release,
or `python release_audit.py` after full reproduction with checkpoints, to
verify the locked hashes, complete grids, paper macros, page geometry, and PDF
fonts.
