# Design lock: HL-Wan for underspecified hand articulation control

## Research question

Can a video diffusion model obey a *partial, camera-invariant hand-motion
specification* without freezing all unspecified finger motion?  Existing hand
control interfaces usually provide a complete 2-D/3-D trajectory.  That is a
poor interface for animation and interaction: an author may know that the
index finger must point inward over an interval, while leaving its exact angle,
the other fingers, and the transition timing free.

## Relation to the advisor's prior work

Li et al., *Translating Motion to Notation* (ACM MM 2024), introduced Hand
Labanotation (HL): twenty hand-local regional vectors, each quantized into one
of 26 directional symbols.  Its Sec. 5.3 reconstructs a single approximate
skeleton by replacing every symbol with the center direction of its cell.  The
new work starts exactly where that paper stops.  It treats an HL cell as a set
of admissible directions, preserves a posterior over that set, and trains Wan
to model the corresponding set of videos.  It therefore tests the earlier
paper's proposed downstream role rather than repeating motion-to-notation
recognition.

## Method under test

**SetHL-Wan** uses Wan2.1-Fun-Control 1.3B with trainable rank-16 LoRA and a
small kinematic control encoder.  A control program contains an always-given
wrist trajectory and palm frame, reference bone lengths, per-bone HL-cell
distributions, and an explicit specified/unspecified *articulation* mask.  The
differentiable spatializer first draws a hard straight-through symbol, then
samples a continuous direction strictly inside that symbol's spherical
Voronoi cell, and forward-kinematically reconstructs a control map.  The
within-cell radius is below half the nearest-code separation, so sampling
cannot change the selected HL symbol.  Hard-HL and learned-VQ controls remain
point-center baselines. Training uses
standard Wan flow matching plus masked categorical cross entropy and an
expected angular cost on valid, hidden cells.  At inference, the completion
posterior and diffusion process are sampled once per output.  No claim is made
for losses that are not present in the released training code.

The method must not transmit full joint coordinates in parallel.  The earlier
all-joint-XY bridge is a rejected design because its coordinate bypass made HL
semantics nearly irrelevant.

## Primary benchmark protocol

WLASL supplies licensed real RGB motion clips and official train/validation/test
metadata.  Hand landmarks are pseudo-labelled consistently and converted to HL
in a hand-local coordinate system.  Splits remain at original video identity;
overlapping windows from one video may never cross a split.  A public ten-clip
EgoGrasp sample is used only as a cross-domain diagnostic, not as training-scale
evidence.  STB remains an optional controlled-motion benchmark if its author
archive becomes reachable.

For each held-out trajectory, create predeclared partial controls by masking
distal bones, contiguous time spans, and entire non-dominant fingers.  Generate
multiple videos per first frame and control.  Report:

- specified-cell HL accuracy and angular error;
- hidden-cell conditional angular diversity;
- specified-cell symbol disagreement as semantic leakage, with legal
  within-symbol angular variation reported separately;
- temporal acceleration error and hand-detection rate;
- sign-recognition consistency, identity similarity, and FVD/FID-VID;
- paired bootstrap confidence intervals over source videos and three seeds.

The primary endpoint is the Pareto hypervolume dominated by the
compliance--hidden-diversity operating points relative to the fixed origin
$(0,0)$. The diversity coordinate is hidden-cell angular diversity multiplied
by one minus specified-cell symbol disagreement and by hand-detection rate.
Thus legal within-symbol variation is retained, while global semantic drift
and failed detections cannot win by merely inflating motion.

## Required matched baselines

All representation baselines use the same Wan checkpoint, rank-16 LoRA budget,
updates, data, seeds, and kinematic encoder capacity:

- a matched-capacity partial raster-skeleton bridge;
- exact continuous hand-local directions plus the same mask;
- continuous Gaussian directions with learned variance;
- learned VQ motion tokens with the same vocabulary size;
- hard center-of-cell HL;
- SetHL without the spherical loss;
- full SetHL-Wan.

## Falsification gate

The paper proceeds under this title only if SetHL-Wan beats continuous Gaussian
and learned VQ controls on the preregistered frontier with a paired 95% interval
excluding zero, while its fully specified control accuracy is non-inferior
(margin: 1 percentage point HL accuracy).  Otherwise the result does not show
that HL is needed and the topic must be changed rather than cosmetically
reframed.

## Evidence status (final locked evaluation, 2026-09-10)

- Actual Wan2.1-Fun-Control 1.3B NPU forward/backward: **passed**.
- LoRA gradients: **480/480 finite** in the systems smoke test.
- WLASL RGB/landmark preparation: **2,252 clips complete** with exact manifest
  coverage and official 1,579/404/269 train/validation/test labels.
- Hand-focused Wan-VAE latent cache: **2,252/2,252 complete** across eight
  independently verified shards.
- End-to-end Wan training: **18/18 primary runs complete** (six methods,
  three seeds, 3,000 updates each).
- Actual Wan ancestral generation: **complete** for all locked interval,
  full-control, and center-only conditions.
- Denoising evaluation: **36/36 reports complete**, each covering all 269 test
  sources.
- Falsification outcome: **the joint gate failed** because the interval-control
  hypervolume differences against continuous and VQ-26 include zero. Full-control
  non-inferiority passed. The paper consequently does not claim frontier
  superiority; it reports the supported semantic-compliance, leakage, and
  denoising results, including confidence intervals.
- R3D-18 source-conditioned feature similarity was added after the locked
  analysis as a **post-hoc diagnostic**. It was not an endpoint, gate, stopping
  rule, or model-selection signal and cannot rescue the failed primary gate.
