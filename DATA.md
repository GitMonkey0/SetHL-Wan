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

Each source video contributes at most one uniformly sampled window and keeps
its official train/validation/test label. The resulting 2,252-example split is
1,579/404/269. After Wan-VAE caching, the authoritative latent manifest has
SHA-256
`5693326da395d89911e91194a49a36e44342f43d4bc91762fcd282edd99f3dd7`.
The 26-entry VQ baseline codebook is fitted on training directions only; the
released `vq26_codebook.pt` has SHA-256
`862c60ff9f2b8071e364727933d8ac1e178e2c4cd21f542f5e74fe514dce2e1c`.

Generated-video evaluation uses every test example for which both source hands
are visible in at least 90% of frames: 27 sources listed in the checked-in
`results/generation_subset.json` (SHA-256
`adbd86ff74b7485f3e8dec3f43dde5fecd9e1d106e665c9f201f87d5483a8cf8`).
No source was selected by its generated result. Posterior calibration uses a
separate fixed eight-source validation manifest.
