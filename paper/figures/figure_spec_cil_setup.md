# Figure: Class-Incremental Learning Setup in Encrypted Traffic

**Label:** `fig:cil-setup`
**Used in:** §2 Background, §2.1 The encrypted-traffic CIL setup
**Archetype:** concept_illustration (time-flow)
**Backend:** TikZ
**Standalone layout:** full-width single column (`\linewidth`)

## Intent

Show the temporal structure of the CIL setup in the encrypted-traffic
domain: a base step followed by $T$ incremental steps, with the
classification head expanding each time. The reader should see all three
structural gaps (shared classifier, uniform distillation, limited
replay) as labeled failure modes in the figure, not as separate diagrams.

## Time axis (left to right)

- **Base step ($t_0$):** encoder $f$ trained + base head $h_0$ over
  $K_0$ classes. Label: "$K_0$ classes, full data".
- **Incremental step $t_1$:** new classes $K_1$ added; head expands to
  $h_1$ (now covers $K_0 + K_1$); replay buffer of old samples present.
- **Incremental step $t_2$:** same operation with $K_2$ new classes,
  head $h_2$ covers $K_0 + K_1 + K_2$.
- (Dotted vertical continuation to $t_T$.)

## Three structural gaps as callouts

Three labeled arrows point at the failure modes inside each incremental
step's training loop:

1. **Shared classifier** — arrow into the expanded head.
2. **Uniform distillation** — arrow into the cross-entropy block.
3. **Limited replay buffer** — arrow into the small buffer icon.

## Visual cues

- Time arrow runs left-to-right under the timeline.
- Each step shows: encoder (frozen after base), head block, replay
  buffer (small stack), cross-entropy loss node.
- The head block visibly grows wider at each incremental step to
  dramatize the "head expansion" failure mode.
- A small "Old samples" icon (ptgreen) and "New samples" icon (ptred)
  flow into the training step.
- The three gap callouts use ptblue, ptcyan, ptgreen respectively to
  distinguish them.

## Caption contract

The caption must state that (a) time flows left to right, (b) the head
expands at each step and the replay buffer is the only bridge to the
past, and (c) the three labeled gaps are the structural limitations
prior CIL methods leave open on encrypted traffic.

## Color usage

- ptblue for the encoder (frozen, shared across time).
- ptgreen for "old" components (replay buffer, old samples).
- ptred for "new" components (new samples, expanded head).
- ptcyan for distillation / loss arrow.
- ptgrey for time axis and box outlines.