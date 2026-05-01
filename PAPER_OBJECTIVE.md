# Paper Objective: Parameterization-Dependent Memorization Dynamics

## Executive summary

The paper should not be framed as "epsilon/epsilon memorizes less." That is too easy to dismiss unless quality is controlled. The stronger and more respectable framing is:

> Target parameterization changes the learning dynamics by which diffusion/flow models move from generation to memorization.

The key sentence to aim for is:

> At matched architecture, dataset size, optimizer, and sample quality, different target parameterizations enter different memorization regimes.

The literature split is clean. Bonnaire et al. already study training dynamics, dataset size, capacity, FID, train/test loss, and memorization time. Li and He already study prediction/loss parameterization and finite-capacity high-dimensional denoising. George et al. already study precise train/test learning curves for DSM under random features, with axes `(n,d,p,m)`. The contribution should be their intersection: parameterization-dependent memorization learning curves. Bonnaire gives the time/data/capacity frame; Li-He gives the target-geometry frame; George gives the learning-curve frame.

The paper should show that target parameterization shifts:

- the quality-memorization frontier;
- the generation-to-memorization window;
- the capacity/data phase boundary.

The must-do analyses are:

- add FID/KID or another quality metric at every checkpoint;
- add train/test losses;
- compute common-space losses if possible;
- validate projected `F_mem`;
- plot checkpoint trajectories in the `(F_mem, FID)` plane.

The high-value derived quantities are:

- `F_mem@FID_best`;
- `tau_gen`;
- `tau_mem`;
- `tau_mem - tau_gen`.

The best result would be:

> One parameterization gives a longer good-quality-before-memorization window, or parameterizations memorize differently even at matched FID.

The main thing to avoid is accidentally writing "Bonnaire, but with target labels." Dataset-size sweeps and capacity sweeps alone are not enough. FID-memorization Pareto plots alone are also not novel, since prior work already studies quality-memorization tradeoffs. The novelty is that target parameterization is the axis that shifts the curve/frontier/window, not merely another hyperparameter. MemControl, for example, explicitly optimizes a quality-memorization tradeoff during fine-tuning, so the Pareto framing is useful but cannot be the contribution by itself.

## What we want from internal analysis/Codex

The role of internal analysis is coauthor-level idea generation and filtering, not just "more plots." It should:

1. identify the strongest paper claim;
2. sanity-check against nearby literature;
3. separate publishable ideas from obvious ablations;
4. design analyses that reviewers cannot easily dismiss;
5. translate messy W&B observations into paper sections and figures;
6. help implement the right evaluations without exploding scope.

The job is not to maximize experiment count. The job is to make every experiment answer a reviewer objection.

## Actual paper thesis

Strongest thesis:

> Algebraically related target parameterizations are not equivalent learning problems under finite data, finite capacity, and finite optimization. They induce different learning curves, different overfitting dynamics, and different memorization behavior.

Sharper mechanistic version:

> Target parameterization controls how model capacity is allocated between intrinsic data structure, ambient noise directions, and sample-specific identity.

Li-He matters because clean data lies near a low-dimensional manifold, while noise and velocity/noised quantities live in the ambient space. Their paper separates prediction space from loss space, enumerates the `(x, epsilon, v)` combinations, and shows that direct network output and loss-space choices can matter even when quantities are transformable. They also include a toy experiment where low-dimensional data are buried in a higher-dimensional observed space by a fixed random column-orthogonal projection; as observed dimension increases, only `x` prediction remains reasonable in that setup.

The twist here is not merely "x prediction works better." The twist is:

> Which parameterization learns the distribution before it starts copying the dataset?

That is the Bonnaire connection. Bonnaire et al. identify an early generation time `tau_gen` and a later memorization time `tau_mem`, with `tau_mem` increasing linearly with training set size `N` while `tau_gen` remains roughly constant. Their result creates a good-generation-before-memorization window. The natural extension is whether this window is parameterization-dependent.

George et al. make the train/test learning-curve angle more principled. Their random-features DSM analysis studies train and test errors in a high-dimensional limit where `d,n,p -> infinity` with `psi_n=n/d` and `psi_p=p/d` fixed, and they emphasize that the number of noise samples per data sample, `m`, has a nontrivial role in generalization and memorization. This motivates tracking not only `F_mem`, but also train/test gaps, noise-resampling effects, and learning curves as functions of `N`, capacity, and parameterization.

## What is already done, and what is not

Bonnaire already covers the broad claim that diffusion models can generate well before they memorize, and that memorization time scales with dataset size. They also include standard U-Net experiments and a theoretical random-features component. Therefore, the fixed-capacity dataset-size sweep is valuable only as a baseline if used to show parameterization-dependent shifts in the Bonnaire dynamics.

Li and He already cover the claim that `x` prediction can be better under finite capacity in high-dimensional spaces, and they explicitly discuss prediction space versus loss space. Therefore, the `x/x`, `epsilon/epsilon`, `v/v`, and `x/v` grid is valuable only if connected to memorization and generalization dynamics, not merely sample quality.

Gu et al. already argue that the DSM optimum can be training-data-replicating and study memorization across factors such as data distribution, model configuration, and training procedure. So the broad "diffusion models can memorize small datasets" point is not novel. The added value is that target parameterization changes the route into memorization, not that memorization exists.

Gagneux et al. are especially close on the parameterization side. They study flow-matching objectives with loss weighting and output parameterization, including noise-, clean-image-, and velocity-based formulations, and analyze how these choices interact with intrinsic dimensionality, architecture, and dataset size using denoising accuracy and FID. The gap is that their focus is training objective/quality diagnostics, not memorization dynamics.

The quality-memorization tradeoff itself is also not new. MemControl uses memorization and generation quality metrics as rewards in a bi-level PEFT selection framework and explicitly targets better quality-memorization tradeoffs. So a Pareto plot is not enough. What can be new is showing that target parameterization shifts the quality-memorization frontier even without changing data, architecture, or optimizer.

## Current experiments recast as a paper skeleton

There are two main experimental families.

First, a fixed-capacity training-size sweep. The same U-Net is trained on CelebA 32x32 grayscale subsets, roughly `N=256` to `N~16,000`, across multiple seeds and parameterizations such as `x/x`, `epsilon/epsilon`, `v/v`, and `x/v`. Runs train to about `1e6` steps and track a projected proxy for `F_mem`. Current observations: all models reduce training loss, but raw losses are not comparable across target spaces. `x/x` and `v/v` tend to memorize more than `epsilon/epsilon`, and final memorization decreases as `N` increases. The same qualitative trend appears under Adam and SGD.

Second, a capacity sweep at selected dataset sizes, roughly small/intermediate/large `N`, such as `N~200`, `2,000`, and `8,000`. U-Net capacity varies through base channels or block count. The important part is not that larger U-Nets memorize more; it is whether parameterizations occupy different positions on the capacity-memorization curve.

Current weakness: `F_mem` is a projected monitoring proxy and not yet a final defensible metric. It is usable for exploratory plots, but the paper needs either validation against full nearest-neighbor/perceptual memorization on selected runs or a clear statement that projected `F_mem` is a proxy.

## Core analyses to run

### 1. Quality-controlled memorization

Highest priority.

For each checkpoint, compute:

```text
F_mem(tau)
FID(tau)
```

Plot trajectories in:

```text
x = F_mem
y = FID
```

Lower-left is better: low memorization and good sample quality.

Key statistic:

```text
F_mem@FID_best = F_mem(argmin_tau FID(tau))
```

Also compute matched-quality memorization:

```text
F_mem@FID <= q
```

where `q` is either an absolute FID threshold or within 10% of best FID.

This directly answers the reviewer objection:

> Maybe epsilon/epsilon memorizes less only because it generates worse samples.

The desired result is not "epsilon/epsilon has lower final memorization." The desired result is:

> At comparable FID, parameterizations still differ in memorization.

If this fails, that is informative: lower memorization of `epsilon/epsilon` is mostly explained by lower sample quality.

### 2. Parameterization-specific Bonnaire timescales

For each parameterization `y in {x/x, epsilon/epsilon, v/v}`, define:

```text
tau_gen^(y) = min{tau : FID^(y)(tau) <= (1 + delta) FID_best^(y)}
```

and:

```text
tau_mem^(y) = min{tau : F_mem^(y)(tau) > alpha}
```

Then define:

```text
Delta_tau^(y) = tau_mem^(y) - tau_gen^(y)
```

This may be the cleanest main result. Bonnaire says there is a window between generation and memorization. We ask whether this window changes with target parameterization.

Then fit:

```text
tau_mem^(y)(N) ~= a_y N + b_y
```

If slopes `a_y` differ, the strong claim is:

> The linear memorization-time scaling is parameterization-dependent.

### 3. Train/test loss gaps in native and common spaces

Native losses:

```text
Delta L_native^(y)(tau) = L_test,native^(y)(tau) - L_train,native^(y)(tau)
```

This tells whether each model overfits its own objective.

Native losses across `x`, `epsilon`, and `v` are not directly comparable because target scales and time weighting differ. Therefore, common-space losses are needed. Convert each model prediction into a shared target space and evaluate:

```text
Delta L_x^(y)(tau)
Delta L_epsilon^(y)(tau)
Delta L_v^(y)(tau)
```

Clean question:

> Does the train/test gap open earlier for the parameterizations that memorize earlier?

This is the George connection: memorization should appear as a change in learning curves, not only as a nearest-neighbor statistic.

### 4. Noise-time-resolved overfitting

Evaluate losses as a function of noise level:

```text
L_train(t, tau)
L_test(t, tau)
Delta L(t, tau)
```

Memorization should be most visible at low noise, where the model is close to reconstructing clean data. Gagneux et al. also emphasize denoising accuracy across noise levels as a diagnostic for objective choices.

Result to look for:

> x/x and v/v develop low-noise train/test gaps earlier than epsilon/epsilon.

If it goes the other way, the mechanism is not low-noise overfitting.

### 5. Capacity/data phase boundary by parameterization

Capacity sweeps should become a phase-boundary plot, not just "larger U-Nets memorize more."

For each parameterization `y`, define a memorization boundary:

```text
n_y^*(p) = smallest dataset size for which F_mem^(y) < alpha at a fixed stopping rule
```

The stopping rule can be final training step, best-FID checkpoint, or `tau_gen`. Best-FID or matched-FID is strongest because it avoids comparing bad models to good models.

Claim:

> Target parameterization shifts the capacity/data boundary between generalization and memorization.

### 6. George's m-axis: noise resampling as optional mechanism experiment

George et al. emphasize the number of noise samples per data sample, `m`, as a nontrivial DSM generalization/memorization factor. In normal diffusion training, repeated epochs with fresh noise effectively increase the variety of noised versions per image.

Focused experiment:

```text
Fix N, architecture, optimizer, and total steps.
Compare:
1. fixed noise/timestep bank per image,
2. small finite bank per image,
3. fresh noise every iteration.
Run x/x, epsilon/epsilon, v/v.
Measure FID, F_mem, train/test gap.
```

Question:

> Is parameterization-dependent memorization partly mediated by the effective number of noise views per training example?

Not necessary for the first paper result, but potentially strong as a mechanism section if main results are clean.

### 7. Intrinsic/ambient decomposition

Most conceptually ambitious.

Hypothesis:

> Parameterization changes whether the model spends capacity on intrinsic data directions, ambient noise directions, or sample identity.

On synthetic data, test directly with a known subspace or manifold. Decompose generated samples or prediction errors into tangent/intrinsic and normal/ambient components:

```text
||e||^2 = ||P_M e||^2 + ||(I - P_M)e||^2
```

Question:

> Does the parameterization that memorizes more also fit normal/off-manifold components differently?

This connects synthetic experiments to the Li-He manifold argument.

## MNIST ambient-dimension idea

Use it, but do not oversell it.

Resolution sweep:

```text
D = H x W, H,W in {8,16,32,64,128}
```

while the semantic dataset remains MNIST. The intuition is that intrinsic digit structure should not change much across resolutions.

Critique: changing resolution is not a clean ambient-dimension intervention. Upsampling adds deterministic smoothness and redundancy; downsampling removes information; convolutional U-Nets exploit locality differently at different resolutions. This is not equivalent to Li-He's random projection experiment, where low-dimensional data are buried in a high-dimensional observed space by a fixed random column-orthogonal projection unknown to the model.

Defensible version:

```text
Clean control:
Take MNIST PCA or an autoencoder latent u in R^d.
Embed with x = A u, where A in R^{D x d}, A^T A = I.
Sweep D at fixed d.

Image-space proxy:
Resize MNIST to different resolutions.
Interpret this as a resolution-induced ambient expansion stress test, not a true fixed-intrinsic-dimension experiment.
```

Keep this auxiliary unless CelebA results are weak.

## Figure plan

Main figure: four-panel result.

Panel A: `F_mem` vs training step. Split by parameterization at fixed `N`, or one row per `N`. Shows raw memorization dynamics.

Panel B: FID vs training step. Same runs. Shows whether lower memorization corresponds to worse generation.

Panel C: FID vs `F_mem` trajectories. Each checkpoint is a point; connect points over time. Color by parameterization. This is the quality-memorization curve.

Panel D: `F_mem@FID_best` vs `N`. Compact controlled comparison across dataset sizes.

Second figure: timescale result.

```text
tau_gen(N)
tau_mem(N)
tau_mem(N) - tau_gen(N)
```

with separate curves for `x/x`, `epsilon/epsilon`, and `v/v`.

Third figure: capacity phase boundary.

```text
F_mem@FID_best
or
n_y^*(p)
```

as a function of U-Net capacity, split by parameterization.

Appendix:

- optimizer replication;
- projected-vs-full `F_mem` validation;
- native train/test losses;
- common-space losses;
- noise-time-resolved loss heatmaps.

## Codex implementation target

For every run/checkpoint, create one row:

```text
run_id
seed
dataset
N_train
capacity_config
num_params
base_channels
num_blocks
optimizer
parameterization
prediction_space
loss_space
step
epoch_or_seen_samples
train_loss_native
test_loss_native
train_loss_x_eval
test_loss_x_eval
train_loss_eps_eval
test_loss_eps_eval
train_loss_v_eval
test_loss_v_eval
F_mem_projected
F_mem_full_optional
FID
KID_optional
num_generated_samples
num_reference_samples
```

Derived quantities:

```text
FID_best
step_best_FID
F_mem_at_best_FID
tau_gen_delta_5pct
tau_gen_delta_10pct
tau_mem_alpha_001
tau_mem_alpha_005
tau_mem_minus_tau_gen
native_train_test_gap
x_space_train_test_gap
eps_space_train_test_gap
v_space_train_test_gap
pareto_nondominated_indicator
```

Plots:

```text
1. F_mem vs step, grouped by parameterization and N.
2. FID vs step, grouped by parameterization and N.
3. FID vs F_mem trajectories.
4. F_mem_at_best_FID vs N.
5. tau_gen and tau_mem vs N.
6. tau_mem ~= a_y N + b_y fits.
7. capacity vs F_mem_at_best_FID.
8. proxy F_mem vs full F_mem validation scatter.
9. noise-level train/test loss heatmaps.
```

Key implementation requirement: every summary must support grouping by parameterization, dataset size, capacity, optimizer, seed, and checkpoint.

## What not to center

Do not center raw training loss comparisons across parameterizations. They are useful diagnostics, but scales are not comparable without normalization or common-space evaluation.

Do not center final `F_mem` alone. It confounds memorization, sample quality, and training duration.

Do not center the Pareto front as a generic quality-memorization plot. The front matters only if target parameterization changes it.

Do not center MNIST resolution as proof of fixed intrinsic dimension. Use it as a proxy/stress test, and use random projection if a clean Li-He-style ambient-dimension control is needed.

Do not overclaim "epsilon/epsilon generalizes better" unless `epsilon/epsilon` has competitive FID or lower memorization at matched FID.

## Clean paper story

Main claim:

> Target parameterization changes when diffusion models memorize.

Results:

1. At fixed capacity and dataset size, `x/x`, `epsilon/epsilon`, and `v/v` exhibit different memorization trajectories.
2. These differences are not explained solely by optimizer choice, since the qualitative trend persists under Adam and SGD.
3. Lower memorization is meaningful only after controlling for sample quality; therefore compare `F_mem` at best FID and at matched FID.
4. Parameterization changes the Bonnaire window `tau_mem - tau_gen`.
5. Parameterization shifts the capacity/data boundary for memorization.
6. Noise-level-resolved train/test gaps suggest a mechanism: parameterizations differ in how they allocate finite capacity across clean data structure, ambient noise directions, and sample identity.

One-line abstract version:

> Although common diffusion targets are algebraically related, they induce different finite-capacity learning curves: target parameterization shifts the transition from generation to memorization, changes the quality-memorization frontier, and alters the capacity/data phase boundary.

## References to track

- Bonnaire et al., "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training". https://arxiv.org/abs/2505.17638
- MemControl, "Mitigating Memorization in Diffusion Models via Automated Parameter Selection". https://arxiv.org/abs/2405.19458
- Li and He, "Back to Basics: Let Denoising Generative Models Denoise". https://arxiv.org/abs/2511.13720
- George et al., "Denoising Score Matching with Random Features: Insights on Diffusion Models from Precise Learning Curves". https://arxiv.org/abs/2502.00336
- Gu et al., "On Memorization in Diffusion Models". https://arxiv.org/abs/2310.02664
- Gagneux et al., "Training Flow Matching: The Role of Weighting and Parameterization". https://arxiv.org/abs/2603.06454
