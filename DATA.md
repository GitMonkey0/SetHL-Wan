# Data provenance

The experiments use the public WLASL metadata and RGB videos together with
MediaPipe hand landmarks. RGB data are not redistributed by this repository.

- WLASL processed videos and metadata: Kaggle dataset
  `risangbaskoro/wlasl-processed`.
- Landmark archive: Kaggle dataset `abd0kamel/mutemotion-output`, version 12,
  file `landmarks_V3.npz`.
- The archive orders landmarks as right hand (21), left hand (21), pose (33),
  and face (478). `prepare_wlasl.py` converts this to the repository's
  left/right convention.

The preparation command is:

```bash
python prepare_wlasl.py \
  --videos /path/to/wlasl-processed/videos \
  --metadata /path/to/WLASL_v0.3.json \
  --parsed /path/to/WLASL_parsed_data.json \
  --landmarks /path/to/landmarks_V3.npz \
  --output data_sources/wlasl/windows_hand256 \
  --frames 17 --size 256 --padding 0.12 --crop-mode hands --min-visible 0.70
```

Each source video contributes at most one uniformly sampled window. We hash the
decoded 17-frame RGB tensor and retain one record per hash, with the fixed
priority test, validation, train and an ID tie-break. This removes 193 exact
duplicates, including every cross-split duplicate, and gives 2,059 clips in a
1,402/392/265 train/validation/test split. The authoritative content-disjoint
latent manifest has SHA-256
`a2f304db7d7e5a60ec7556c206f2f75f44ba175041d5430c1c82683c5a33b15d`.
The 26-entry VQ baseline codebook is fitted on training directions only; the
released `vq26_codebook_content_disjoint.pt` has SHA-256
`3bfbe6436169876e0ec3b97723ccfaeeda80909b98f7db97245c99c65f86fea5`.

Generated-video evaluation uses every test example for which both source hands
are visible in at least 90% of frames: 25 sources listed in the checked-in
`results/generation_subset_content_disjoint.json` (SHA-256
`b4afea51b7886e9d239b0c10ed50f8d9768d5fcf27e60682f5435b031e284104`).
No source was selected by its generated result. Posterior calibration uses a
separate fixed eight-source validation manifest.
