# Paper Objective: Target Parameterization and Memorization Dynamics

## Executive summary

The paper should no longer be framed as target parameterization being determined by intrinsic versus ambient dimension. Concurrent work by Qing Jin and Chaoyang Wang, "Revisiting Diffusion Model Predictions Through Dimensionality" (arXiv:2601.21419), is close enough to own that geometry/target-choice story. They introduce the continuous target

```text
y_k = k*x - (1-k)*eps,  k in [0, 1]
```

where `k=0` is noise prediction up to sign, `k=0.5` is velocity prediction up to scale, and `k=1` is clean-data prediction. They derive

```text
k* = D / (D + d)
```

so `k* ~= 0.5` when ambient dimension `D` is close to intrinsic dimension `d`, and `k* -> 1` when `D >> d`. They also propose `k`-Diff, where `k` is learned by backpropagation.

Therefore, our novelty should be:

> Target parameterization changes the memorization dynamics of flow matching.

Sharper objective:

> The target that is optimal for sample quality may not be optimal for memorization. We study how target parameterization shifts the generation-memorization frontier in flow matching.

The central question is:

```text
Does k_FID_star equal k_mem_star?
```

where

```text
k_FID_star = argmin_k FID
k_mem_star = argmin_k F_mem@FID_best
```

The strongest possible result is:

```text
k_FID_star != k_mem_star
```

Paper sentence:

> The prediction target that is optimal for sample quality is not necessarily optimal for memorization.

## Updated internal objective

This project studies how target parameterization affects memorization dynamics in flow matching. We focus on the linear flow path

```text
z_t = t*x + (1-t)*eps
u_fm = x - eps
eps ~ N(0, I)
t in [t_min, t_max]
```

and compare targets along the continuous family

```text
y_k = k*x - (1-k)*eps.
```

Prior work has shown that the optimal target for sample quality depends on data geometry, especially the relation between ambient and intrinsic dimension. We instead ask whether the quality-optimal target is also safe from memorization.

Our central hypothesis is that target parameterization changes not only final sample quality, but the training-time transition from generation to memorization. We therefore evaluate each target using checkpoint-level FID, nearest-neighbor memorization `F_mem`, train/test losses, memorization at best FID, and the timescales `tau_gen` and `tau_mem`. The core result we seek is whether the target minimizing FID differs from the target minimizing memorization at matched quality.

One-liner:

> Target parameterization shifts the generation-memorization frontier in flow matching.

Stronger version if supported:

> Quality-optimal flow-matching targets can be memorization-prone.

## What remains ours after 2601.21419

The 2601.21419 paper does not appear to compute nearest-neighbor memorization, `F_mem`, train/reference memorization ratios, Bonnaire-style memorization time `tau_mem`, or quality-controlled memorization such as `F_mem@FID_best`. Their experiments are centered on FID/IS and ImageNet latent/pixel-space generation quality.

Our contribution should be:

```text
Target parameterization is not only a sample-quality choice.
It also changes when and how flow-matching models memorize.
```

This shifts the framing from quality-optimal target selection to memorization dynamics under controlled quality measurement.

## Claims to test

1. Target parameterization changes memorization dynamics at fixed dataset size, architecture, optimizer, and training budget.
2. Lower memorization is meaningful only after controlling for sample quality.
3. The quality-optimal target and memorization-optimal target need not coincide.
4. Target parameterization changes the Bonnaire-style generation-to-memorization window.

Bonnaire et al. identify an early generation time `tau_gen` and a later memorization time `tau_mem`, with `tau_mem` increasing roughly linearly with dataset size while `tau_gen` remains approximately constant. Our extension indexes those timescales by target parameterization:

```text
tau_gen^(k)
tau_mem^(k)
tau_mem^(k) - tau_gen^(k)
```

## Claims not to make

Do not claim we introduce the continuous `k` target.

Do not claim we are first to connect target parameterization to intrinsic/ambient dimension.

Do not claim `x` prediction being favored when `D >> d` is our explanation.

Do not center MNIST resolution as a clean intrinsic/ambient experiment.

Do not make FID differences across targets the main contribution.

Do not say `epsilon/epsilon` generalizes better unless it has competitive quality or lower memorization at matched FID.

## Experiments to keep

Keep the fixed-capacity CelebA 32x32 grayscale dataset-size sweep:

```text
N in {256, ..., 16000}
```

Keep multiple seeds.

Keep optimizer replication with Adam and SGD.

Keep the capacity sweep at representative dataset sizes.

Keep checkpoint-level `F_mem`, but add FID and train/test losses wherever possible.

## Experiments to add

Add a small fixed-`k` sweep:

```text
k in {0.0, 0.25, 0.5, 0.75, 1.0}
```

At minimum include:

```text
k in {0.0, 0.5, 1.0}
```

These recover noise, velocity, and `x` prediction up to sign/scale.

For each `k`, compute:

```text
FID(tau)
F_mem(tau)
F_mem@FID_best
tau_gen
tau_mem
tau_mem - tau_gen
```

Optional learned-`k` experiment:

```text
k = sigmoid(w_k)
y_k = k*x - (1-k)*eps
u_hat = velocity_from_yk(z_t, y_hat_k, t, k)
```

Initialize `k` at `0.5`, log it at every checkpoint, and save the trajectory. This is optional and should not block the first result.

## Flow-matching target setup

Use flow matching only where possible. Remove diffusion/VP-specific assumptions from new code paths unless needed for backward compatibility.

Core path:

```text
z_t = t*x + (1-t)*eps
u_fm = x - eps
```

Continuous target:

```text
y_k = k*x - (1-k)*eps
```

Velocity induced by a `k`-target prediction:

```text
denom = k*(1-t) + (1-k)*t
u_hat = ((1 - 2*k)*z_t + y_hat_k) / denom
```

For numerical stability:

```text
denom = denom.clamp_min(denom_min)
```

`denom_min` should be configurable. Use `1e-4` if `t_min/t_max` already avoid endpoints, or `0.05` if following the `k`-Diff implementation exactly.

Important sign convention:

```text
k=0 gives y_k = -eps, not +eps.
```

Legacy `eps/eps` predicts `+eps`; it corresponds to the new `k=0` target only after multiplying output/target by `-1`. For new `k`-sweeps, implement `y_k` directly and report results consistently.

## Implementation target for Codex

Add config support:

```text
target_mode: {legacy, fixed_k, learned_k}
k_value: float in [0, 1] for fixed_k
learn_k_init: float, default 0.5
denom_min: float
loss_space: {target, velocity} if feasible
```

Add helpers:

```text
make_fm_batch(x, t, eps):
    z = t*x + (1-t)*eps
    u = x - eps
    return z, u

target_yk(x, eps, k):
    return k*x - (1-k)*eps

velocity_from_yk(z, yk, t, k, denom_min):
    denom = k*(1-t) + (1-k)*t
    denom = denom.clamp_min(denom_min)
    return ((1 - 2*k)*z + yk) / denom
```

Training objective options:

```text
Target-space loss:
    pred_yk = model(z, t)
    loss = mse(pred_yk, yk)

Velocity-space loss:
    pred_yk = model(z, t)
    pred_u = velocity_from_yk(z, pred_yk, t, k, denom_min)
    loss = mse(pred_u, u)
```

Use target-space loss for controlled comparison with existing `x/eps/u` target regressions unless otherwise specified. Add velocity-space loss as optional because `k`-Diff uses it.

Sampling:

```text
pred_yk = model(z, t)
pred_u = velocity_from_yk(z, pred_yk, t, k, denom_min)
z_next = z + dt * pred_u
```

Support Euler and the existing solver if already present.

Backward compatibility mapping:

```text
x/x -> k=1
velocity/u/u or v/v -> k=0.5 up to scale
eps/eps -> k=0 up to sign
```

Do not silently mix legacy `+eps` with new `-eps` in velocity conversion.

## Metrics table target

For every run/checkpoint, save one row with:

```text
run_id
seed
dataset
N_train
capacity_config
num_params
optimizer
target_mode
k_value
learned_k_current
step
train_loss_native
test_loss_native
train_loss_velocity
test_loss_velocity
F_mem_projected
F_mem_full_optional
FID
KID_optional
num_generated_samples
num_reference_samples
```

Derived metrics:

```text
FID_best = min_tau FID(tau)
step_best_FID = argmin_tau FID(tau)
F_mem_at_best_FID = F_mem(step_best_FID)
tau_gen = first step where FID <= (1+delta)*FID_best
tau_mem = first step where F_mem > alpha
gen_mem_window = tau_mem - tau_gen
k_FID_star = k with lowest best FID
k_mem_star = k with lowest F_mem_at_best_FID
pareto_front in (FID, F_mem), lower is better
```

Every summary must support grouping by parameterization, `k`, dataset size, capacity, optimizer, seed, and checkpoint.

## Plot targets

1. `F_mem` vs step, colored by `k`.
2. FID vs step, colored by `k`.
3. FID vs `F_mem` checkpoint trajectories, colored by `k`.
4. `F_mem_at_best_FID` vs `k`.
5. Best FID vs `k`.
6. `tau_gen` and `tau_mem` vs `k`.
7. `tau_mem - tau_gen` vs `k`.
8. For `N` sweep: `F_mem_at_best_FID` vs `N`, split by `k`.
9. For capacity sweep: `F_mem_at_best_FID` vs capacity, split by `k`.

## Revised contribution bullets

1. We study target-parameterization-dependent memorization in flow matching, focusing on the continuous target family `y_k = k*x - (1-k)*eps`.
2. We evaluate memorization jointly with sample quality, using checkpoint-level FID, nearest-neighbor `F_mem`, and `F_mem` at the best-FID checkpoint.
3. We show that target parameterization changes the training-time transition from generation to memorization, quantified by `tau_gen`, `tau_mem`, and `tau_mem - tau_gen`.
4. We test whether the target that minimizes FID is also the target that minimizes memorization at matched quality.
5. We perform dataset-size and capacity sweeps to determine whether target parameterization shifts the effective boundary between generalization and memorization.

If the `k`-sweep works:

```text
We find that the quality-optimal target and memorization-optimal target can differ, revealing a target-dependent quality-memorization tradeoff in flow matching.
```

If it does not:

```text
We find that target parameterization jointly affects quality and memorization, and lower memorization must be interpreted only after controlling for sample quality.
```

## Related/concurrent work: 2601.21419

Jin and Wang, "Revisiting Diffusion Model Predictions Through Dimensionality," studies prediction-target selection through dimensionality. It introduces the target family

```text
y_k = k*x - (1-k)*eps,  k in [0, 1]
```

which recovers noise, velocity, and clean-data prediction as special cases up to sign and scale. Their theory analyzes a linear model trained on data supported in a `d`-dimensional subspace of an ambient `D`-dimensional space. The learning dynamics decompose into components parallel to the data manifold and perpendicular to it. The parallel component is associated with data recovery, while the perpendicular component is associated with noise elimination in the ambient space.

Under simplifying assumptions for the flow-matching path, they derive:

```text
k* = D / (D + d)
```

This predicts velocity-like targets when `D ~= d` and clean-data prediction when `D >> d`. Since intrinsic dimension is hard to estimate in real data, they propose `k`-Diff, which learns scalar `k` by backpropagation. Their ImageNet experiments show `k`-Diff matching or improving fixed-target baselines in latent and pixel space.

For our paper, this is concurrent work on quality-optimal target selection, not memorization work. It motivates the `k`-target family and strengthens the case that target parameterization matters, but it does not answer whether the target that optimizes FID also minimizes memorization. Our work studies that missing axis by measuring `F_mem`, `F_mem@FID_best`, `tau_gen`, and `tau_mem` across target parameterizations.

Paper-ready paragraph:

```latex
Concurrent work by Jin and Wang studies prediction-target selection through the lens of dimensionality. They introduce a continuous target family
\[
    \vy_k = k\vx-(1-k)\veps,
    \qquad k\in[0,1],
\]
which recovers noise, velocity, and clean-data prediction as special cases up to sign and scale. In a linear model with data supported on a $d$-dimensional subspace of an ambient $D$-dimensional space, they show that the learning dynamics decompose into manifold-parallel and ambient-perpendicular components, and derive the optimal target parameter
\[
    k^\star = \frac{D}{D+d}.
\]
This predicts velocity-like targets when $D\approx d$ and clean-data prediction when $D\gg d$. They further propose $k$-Diff, which learns $k$ by backpropagation and improves or matches fixed-target baselines in latent- and pixel-space ImageNet generation. Our work is complementary: rather than optimizing the target for sample quality alone, we ask whether target parameterization also changes the transition from generation to memorization. In particular, we evaluate whether the quality-optimal target is also optimal under nearest-neighbor memorization metrics and quality-controlled memorization summaries such as $F_{\mathrm{mem}}@\mathrm{FID}_{\mathrm{best}}$.
```

Defensive version:

```latex
This concurrent work makes it important to distinguish target-quality optimality from memorization behavior. Its analysis explains why certain targets are preferable for efficient prediction under intrinsic/ambient dimensionality, but it does not study training-set replication or memorization timescales. We therefore use the same target family as a controlled axis for evaluating how target choice affects the quality--memorization frontier.
```

## Minimal LaTeX target setup

```latex
\subsection{Flow matching target parameterization}
\label{sec:fm_target_parameterization}

We consider the linear flow-matching path
\begin{equation}
    \vz_t = t\vx + (1-t)\veps,
    \qquad
    \veps\sim\mathcal{N}(\vzero,\mI),
    \qquad
    t\in[t_{\min},t_{\max}]\subset(0,1).
\end{equation}
The corresponding flow velocity is
\begin{equation}
    \vu = \vx-\veps.
\end{equation}

Following the continuous target parameterization of concurrent work, we write
\begin{equation}
    \vy_k = k\vx-(1-k)\veps,
    \qquad
    k\in[0,1].
    \label{eq:k_target}
\end{equation}
The special cases $k=0$, $k=1/2$, and $k=1$ recover noise, velocity, and clean-data prediction up to sign and scale. For sampling, a prediction $\hat{\vy}_k$ is converted into a velocity estimate by
\begin{equation}
    \hat{\vu}_k(\vz_t,t)
    =
    \frac{(1-2k)\vz_t+\hat{\vy}_k(\vz_t,t)}
    {k(1-t)+(1-k)t}.
    \label{eq:k_to_velocity}
\end{equation}
Thus all targets define a velocity field along the same probability path, but they induce different finite-capacity regression problems.
```

Training objective:

```latex
For fixed $k$, we train a model $f_\theta^k$ using
\begin{equation}
    \mathcal{L}_{\mathrm{target}}^k(\theta)
    =
    \mathbb{E}_{\vx,\veps,t}
    \left[
        \left\|
        f_\theta^k(\vz_t,t)-\vy_k
        \right\|_2^2
    \right].
    \label{eq:k_target_loss}
\end{equation}
Where stated, we also evaluate the induced common velocity loss
\begin{equation}
    \mathcal{L}_{\mathrm{vel}}^k(\theta)
    =
    \mathbb{E}_{\vx,\veps,t}
    \left[
        \left\|
        \hat{\vu}_k(\vz_t,t)-\vu
        \right\|_2^2
    \right].
    \label{eq:k_velocity_loss}
\end{equation}
```

## References to track

- Jin and Wang, "Revisiting Diffusion Model Predictions Through Dimensionality". https://arxiv.org/html/2601.21419v2
- Bonnaire et al., "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training". https://arxiv.org/abs/2505.17638
- MemControl, "Mitigating Memorization in Diffusion Models via Automated Parameter Selection". https://arxiv.org/abs/2405.19458
- Li and He, "Back to Basics: Let Denoising Generative Models Denoise". https://arxiv.org/abs/2511.13720
- George et al., "Denoising Score Matching with Random Features: Insights on Diffusion Models from Precise Learning Curves". https://arxiv.org/abs/2502.00336
- Gu et al., "On Memorization in Diffusion Models". https://arxiv.org/abs/2310.02664
- Gagneux et al., "Training Flow Matching: The Role of Weighting and Parameterization". https://arxiv.org/abs/2603.06454
