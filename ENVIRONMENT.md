# Reproduction environment

The reported runs used one Ascend 910B2C NPU per training or evaluation
process under CANN 8.5.1, Python 3.11, PyTorch 2.9.0, and
torch-npu 2.9.0.post1. Mixed precision was bfloat16. Each training seed used
one device; no result relies on multi-device numerical reduction.

The Wan implementation is VideoX-Fun commit
`968f0e2192ba4c7a12868bf36d73260d135424ca`. Run
`setup_videox_fun.sh` to fetch that revision and apply the included Ascend
complex-dtype compatibility patch. Set `VIDEOX_FUN_ROOT` if it is not located
at the default sibling workspace path, and set `WAN_MODEL_PATH` to the
downloaded `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control` directory.

Python package versions used for the final run are listed in
`requirements.txt` and `requirements-eval.txt`. The locally installed
Diffusers build reported version `0.41.0.dev0`; the public requirement admits
the compatible 0.35--0.41 API range because that development wheel has no
stable package artifact.

The RGB source data and Wan weights are not redistributed. Every reported
checkpoint records the SHA-256 of the latent manifest, and the release audit
checks that hash together with the fixed test and validation subset manifests.
