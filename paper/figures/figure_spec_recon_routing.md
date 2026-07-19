# Figure: Reconstruction Routing Mechanism (detail)

**Label:** `fig:recon-routing`
**Used in:** Per-Flow Reconstruction Compatibility
**Archetype:** component_detail
**Backend:** TikZ
**Standalone layout:** full-width single column (`\linewidth`)

## Intent

Zoom into the reconstruction branch to show the exact transformation
chain from frozen-encoder features to the routing weight, including the
intermediate error magnitude and the temperature scaling. The figure
should make the role of $\tau$ visible: it sits between the probability
and the routing weight, not between the error and the probability.

## Components

Inside a single dashed bounding box labeled "Reconstruction routing":

1. **Feature input** $f(x)$ — entry point on the left.
2. **Autoencoder** $g_\phi$ — produces reconstruction $\hat{f}(x)$.
3. **Error computation** — $\| \hat{f}(x) - f(x) \|_2^2$, drawn as a
   circle node.
4. **Sigmoid** $\sigma(-\alpha \cdot e(x))$ — converts error into
   probability $p_{\mathrm{old}}(x)$.
5. **Temperature scaling** $\mathrm{softmax}_\tau$ — explicitly labeled
   $\tau$ knob; output is $w_{\mathrm{old}}(x)$.
6. **Routing weight output** $w_{\mathrm{old}}(x) \in [0, 1]$ on the right.

## External connection

The figure shows that $f(x)$ also feeds (via a thin arrow) to both
classification heads — but the heads themselves are NOT inside the
bounding box. This is the key separation: the routing weight is consumed
by the calibration step in the architecture figure, not by the heads.

## Visual cues

- The reconstruction branch runs horizontally; vertical labels annotate
  the transformations.
- The $\tau$ symbol is rendered large and labeled "temperature knob" to
  emphasize that it is the only inference-time routing parameter (§3.4).
- A small inset equation under the sigmoid: $p_{\mathrm{old}} =
  \sigma(-\alpha \cdot e(x))$.
- Under the temperature node: $w_{\mathrm{old}} = \mathrm{softmax}_\tau
  (p_{\mathrm{old}})$.

## Caption contract

The caption must clarify (a) the autoencoder is trained only on
replayed old samples, so reconstruction error correlates with how
"old" $x$ is, and (b) $\tau$ controls routing softness, not the
probability itself.

## Color usage

- Entire bounding box border: ptcyan (matches architecture figure).
- Autoencoder block: ptcyan fill.
- Sigmoid + temperature nodes: ptblue fill (they are the
  probability/weight interface).
- Error computation: ptgrey fill (geometric, no learning).
- External thin arrow to heads: ptgrey.
