# Figure: RAR Architecture Overview

**Label:** `fig:rar-architecture`
**Used in:** §3 Method (after the section opener, before §3.1)
**Archetype:** architecture_overview
**Backend:** TikZ
**Standalone layout:** full-width single column (`\linewidth`)

## Intent

Show the end-to-end RAR pipeline at the architecture level: a single
horizontal flow from raw input to calibrated classification logits, with
all five components visible. Reader should be able to point at any
component in the figure and locate the subsection of §3 that defines it.

## Components (left to right)

1. **Raw input** $x$ — packet sequence / session. (ptgrey fill.)
2. **Frozen encoder** $f$ — ET-BERT-style feature extractor. (ptblue fill.)
   Subtitle: "frozen, shared across heads".
3. **Feature** $f(x)$ — D-dim vector fed forward to all three branches.
4. **Reconstruction autoencoder** $g_\phi$ — trained on replayed old
   samples. Produces reconstruction error $e(x)$. (ptcyan fill.)
   Branch goes downward, then up to the sigmoid block.
5. **Sigmoid + temperature** — $\sigma(-\alpha \cdot e(x))$, then
   temperature $\tau$ scaling produces routing weight $w_{\mathrm{old}}(x)$.
6. **Old-knowledge head** $h_{\mathrm{old}}$ — trained at base step,
   frozen at incremental step. (ptgreen fill.)
7. **New-knowledge head** $h_{\mathrm{new}}$ — initialized at incremental
   step. (ptred fill.)
8. **Logit calibration + combine** — per-head $T_h, b_h$ rescale +
   weighted combination. Outputs logits over all classes. (ptblue fill.)

## Visual cues

- Branch downward from feature to autoencoder (no routing role).
- Branch to the routing weight above the heads.
- Both heads weighted into the calibration block (the only place where
  routing weight is consumed).
- Soft routing emphasis: arrow widths into calibration reflect
  $w_{\mathrm{old}}$ vs $w_{\mathrm{new}}$ but with NO claim that one
  head dominates (avoid the "hard assignment" framing).

## Caption contract

The caption must state that $p_{\mathrm{old}}(x)$ is the per-sample
old/new probability, that soft routing keeps both heads active, and that
calibration unifies logits across disjoint class subsets.

## Color usage

- ptblue for "shared / frozen" elements (encoder, calibration).
- ptcyan for the reconstruction branch (the new mechanism).
- ptgreen for "old" semantics (head + replay anchor).
- ptred for "new" semantics.
- ptgrey for raw data / input.