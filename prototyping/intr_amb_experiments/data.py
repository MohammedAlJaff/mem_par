"""
data.py
=======

Synthetic dataset with a *known* intrinsic dimension k embedded in ambient
dimension d, for studying how the gap (d - k) shapes memorization dynamics
across x- / eps- / v-prediction in flow matching.

Construction (all affine, so everything stays a GMM and is closed-form)
-----------------------------------------------------------------------
Source GMM in R^d:                x0 ~ sum_m pi_m N(mu_m, mode_sigma^2 I_d)
Bottleneck (nested random P):     A = P^T P,   P in R^{k x d}, P P^T = I_k
                                  => A is the orthogonal projector onto a
                                  random k-dim subspace (A = A^T, A^2 = A)
Thickened reconstruction:         x_tilde = A x0 + off_manifold_sigma (I-A) xi

Dataset GMM (push-forward, exact):
    mu_tilde_m    = A mu_m
    Sigma_tilde_m = mode_sigma^2 A + off_manifold_sigma^2 (I_d - A)

Flow-matching path (STANDARD orientation: t=0 noise, t=1 data):
    alpha_t = t,  x_t = alpha_t x_tilde + (1 - t) eps
    eps ~ N(0, off_manifold_sigma^2 I_d)
    dot_alpha = +1,  dot_sigma = -1

Marginal p_t stays a GMM:
    m_m(t) = alpha_t mu_tilde_m
    S_m(t) = alpha_t^2 Sigma_tilde_m + ((1 - t)^2 off_manifold_sigma^2) I_d

Closed-form score / optimal targets (the ORACLE / Bayes-optimal field):
    score(x,t) = grad_x log p_t(x) = sum_m gamma_m(x,t) [ -S_m(t)^{-1} (x - m_m(t)) ]
    x0_hat  = (x_t + (1 - t)^2 off_manifold_sigma^2 score) / alpha_t
    eps_hat = -(1 - t) * off_manifold_sigma^2 score
    v       = dot_alpha x0_hat + dot_sigma eps_hat = x0_hat - eps_hat

NOTE on two distinct "targets":
  * make_training_target(...)  -> the Monte-Carlo regression target the network
    actually trains on, built from a CLEAN train-set point + sampled (t, eps).
    No score involved:  x->x_tilde, eps->eps, v->x_tilde - eps.
  * true_target(...)           -> the Bayes-optimal field (uses the GMM score).
    This is the ground truth you measure a learned field AGAINST.

Conventions
-----------
  rho : NN-ratio threshold (memorization).      r = d1/d2 in [0,1], small => memorized.
  tau : reserved elsewhere for gradient-step index (NOT used here).
  Vectors/matrices follow paper math symbols; inline comments annotate meaning.
  dtype defaults to float64 for oracle numerical stability (S_m(t) is ill-
  conditioned as t->1 when sigma is small). Use float32 for training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

PARAMS = ("x", "eps", "v")


# ----------------------------------------------------------------------
# manifold Gaussian data model
# ----------------------------------------------------------------------
class ManifoldGMM:
    """Thin dataset/oracle class for the intrinsic-vs-ambient Gaussian mixture experiments."""

    def __init__(
        self,
        mu: Tensor,  # (M, d) source component means
        pi: Tensor,  # (M,) mixture weights (sum to 1)
        k: int,  # intrinsic / bottleneck dimension
        sigma: float | None = None,  # legacy alias: sets both sigmas when split names are omitted
        n_train: int = 2000,  # frozen train-set size
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        t_eps: float = 1e-3,  # clamp t away from {0,1} (x0_hat ~ 1/alpha_t blows up at t=0)
        mode_sigma: float | None = None,  # within-mode std before projection onto the manifold
        off_manifold_sigma: float | None = None,  # perpendicular thickness around the manifold
    ):
        """Build projector, projected/thickened GMM params, and cached training points."""
        self.device = torch.device(device)
        self.dtype = dtype
        self.t_eps = float(t_eps)

        mu = mu.to(self.device, dtype)
        pi = pi.to(self.device, dtype)

        self.M, self.d = mu.shape
        assert pi.shape == (self.M,)
        assert 1 <= k <= self.d, "need 1 <= k <= d"
        self.k = int(k)
        if mode_sigma is None:
            mode_sigma = sigma if sigma is not None else 1e-2
        if off_manifold_sigma is None:
            off_manifold_sigma = sigma if sigma is not None else mode_sigma
        self.mode_sigma = float(mode_sigma)
        self.off_manifold_sigma = float(off_manifold_sigma)
        self.sigma = self.mode_sigma  # legacy attribute for old notebooks/scripts
        assert self.mode_sigma > 0.0, "mode_sigma must be positive"
        assert self.off_manifold_sigma > 0.0, "off_manifold_sigma must be positive"
        self.n_train = int(n_train)
        self.pi = pi / pi.sum()
        # Normalize explicitly here so downstream log(pi) is always well-defined.

        # --- nested random orthonormal projection P (k x d) ----------------
        # One seeded full orthogonal basis; slice first k rows => nested across k.
        g = torch.Generator(device=self.device).manual_seed(seed)
        G = torch.randn(self.d, self.d, generator=g, device=self.device, dtype=dtype)
        Qfull, _ = torch.linalg.qr(G)  # (d, d), columns orthonormal
        P_full = Qfull.T  # rows orthonormal
        self.P = P_full[: self.k].contiguous()  # (k, d), nested slice
        self.A = self.P.T @ self.P  # (d, d) orthogonal projector, A = A^T = A^2

        # --- build-time geometry guard (fail loudly on a projector bug) ----
        eye_k = torch.eye(self.k, device=self.device, dtype=dtype)
        assert torch.allclose(self.P @ self.P.T, eye_k, atol=1e-8), (
            "P rows not orthonormal"
        )
        assert torch.allclose(self.A, self.A.T, atol=1e-8), "A not symmetric"
        assert torch.allclose(self.A @ self.A, self.A, atol=1e-6), "A not idempotent"
        assert abs(self.A.diagonal().sum().item() - self.k) < 1e-4, "rank(A) != k"

        # --- dataset (tilde) GMM, closed form ------------------------------
        I_d = torch.eye(self.d, device=self.device, dtype=dtype)
        Sigma = (self.mode_sigma**2) * I_d.expand(self.M, self.d, self.d).clone()
        self.mu_tilde = mu @ self.A  # (M, d); A symmetric => A mu_m
        self.Sigma_tilde = (
            self.A @ Sigma @ self.A + self.off_manifold_sigma**2 * (I_d - self.A)
        )  # (M, d, d): mode spread along manifold, thickness off manifold
        self.L_tilde = torch.linalg.cholesky(self.Sigma_tilde)  # (M, d, d) for sampling

        # --- frozen train set (sampled ONCE) -------------------------------
        self._gen_train = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.train_set = self._sample(
            self.n_train, self._gen_train
        )  # (n_train, d), frozen

        self._I_d = I_d

    # ----------------------------------------------------------------------
    # constructors / convenience
    # ----------------------------------------------------------------------
    @classmethod
    def random_gmm(
        cls,
        d: int,
        k: int,
        n_modes: int = 3,
        sep: float = 4.0,  # mode-center separation scale
        sigma: float | None = None,
        mode_sigma: float | None = None,
        off_manifold_sigma: float | None = None,
        n_train: int = 2000,
        seed: int = 0,
        **kw,
    ) -> "ManifoldGMM":
        """Build a random isotropic-mode baseline GMM and hand it to the main constructor."""
        if mode_sigma is None:
            mode_sigma = sigma if sigma is not None else 1e-2
        if off_manifold_sigma is None:
            off_manifold_sigma = sigma if sigma is not None else mode_sigma
        g = torch.Generator().manual_seed(seed + 7)
        mu = sep * torch.randn(n_modes, d, generator=g)  # random isotropic means
        pi = torch.full((n_modes,), 1.0 / n_modes)  # equal-mode prior
        return cls(
            mu,
            pi,
            k=k,
            mode_sigma=mode_sigma,
            off_manifold_sigma=off_manifold_sigma,
            n_train=n_train,
            seed=seed,
            **kw,
        )

    # ----------------------------------------------------------------------
    # sampling
    # ----------------------------------------------------------------------
    def _sample(self, n: int, gen: torch.Generator) -> Tensor:
        """Internal helper: sample from the cached dataset distribution using component Cholesky factors."""
        comp = torch.multinomial(self.pi, n, replacement=True, generator=gen)  # (n,)
        z = torch.randn(n, self.d, generator=gen, device=self.device, dtype=self.dtype)
        L = self.L_tilde[comp]  # (n, d, d) component-specific Cholesky factors
        x = self.mu_tilde[comp] + torch.bmm(L, z.unsqueeze(-1)).squeeze(
            -1
        )  # sample via A@x0 + noise
        return x

    def sample(self, n: int, seed: int | None = None) -> Tensor:
        """Public helper for fresh draws from dataset p_tilde, independent of the memoized train_set."""
        gen = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        if gen is None:
            comp = torch.multinomial(self.pi, n, replacement=True)
            z = torch.randn(n, self.d, device=self.device, dtype=self.dtype)
        else:
            comp = torch.multinomial(self.pi, n, replacement=True, generator=gen)
            z = torch.randn(
                n, self.d, generator=gen, device=self.device, dtype=self.dtype
            )
        L = self.L_tilde[comp]
        return self.mu_tilde[comp] + torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)

    # ----------------------------------------------------------------------
    # closed-form ORACLE (Bayes-optimal field)
    # ----------------------------------------------------------------------
    def _gmm_at_t(self, t: Tensor):
        """Compute component marginals m_m(t), S_m(t) for the mixture at each requested time."""
        a = t.view(-1, 1, 1)  # alpha_t
        s = (1.0 - t).view(-1, 1, 1, 1)  # beta_t = 1 - t
        sigma_t = self.off_manifold_sigma * s
        m = a * self.mu_tilde.unsqueeze(0)  # (B,M,d)
        S = (t.view(-1, 1, 1, 1) ** 2) * self.Sigma_tilde.unsqueeze(0) + (
            sigma_t**2
        ) * self._I_d.view(
            1, 1, self.d, self.d
        )  # (B,M,d,d): alpha^2 Sigma_tilde + beta^2 sigma^2 I
        return m, S

    def true_score(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute exact Bayes-oracle score field ∇_x log p_t(x) by mixture responsibilities."""
        x = x.to(self.device, self.dtype)
        t = self._as_t(t, x.shape[0])
        m, S = self._gmm_at_t(t)  # (B,M,d), (B,M,d,d): per-component marginals at t
        diff = x.unsqueeze(1) - m  # (B,M,d)
        L = torch.linalg.cholesky(
            S
        )  # (B,M,d,d): SPD factor for each Gaussian component
        z = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(
            -1
        )  # S^{-1} diff, (B,M,d)
        quad = (diff * z).sum(-1)  # (B,M): Mahalanobis distance term
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (B,M)
        logN = -0.5 * (
            self.d * math.log(2 * math.pi) + logdet + quad
        )  # (B,M): component log-likelihood
        logw = (
            torch.log(self.pi).unsqueeze(0) + logN
        )  # (B,M): prior + likelihood in log-space
        gamma = torch.softmax(
            logw, dim=1
        )  # responsibilities (B,M), sums to one per sample
        comp_score = -z  # (B,M,d) = -S^{-1}(x-m) for each component
        return (gamma.unsqueeze(-1) * comp_score).sum(1)  # (B,d)

    def true_target(self, x: Tensor, t: Tensor, param: str) -> Tensor:
        """Convert oracle score into the requested regression target x_tilde, eps, or v."""
        assert param in PARAMS
        x = x.to(self.device, self.dtype)
        t = self._as_t(t, x.shape[0])
        score = self.true_score(x, t)  # oracle score field at (x,t)
        alpha = t.unsqueeze(-1)  # (B,1)
        beta = (1.0 - t).unsqueeze(
            -1
        )  # (B,1), coefficient in x_t = alpha x_tilde + beta eps
        sigma = self.off_manifold_sigma * beta  # (B,1), effective forward-noise std
        x0_hat = (x + sigma**2 * score) / alpha  # Tweedie: E[x_tilde | x_t]
        eps_hat = (
            -beta * (self.off_manifold_sigma**2) * score
        )  # E[eps | x_t], eps has std off_manifold_sigma (not 1)
        if param == "x":
            return x0_hat
        if param == "eps":
            return eps_hat
        return x0_hat - eps_hat  # v = dot_alpha x0 + dot_sigma eps

    # ----------------------------------------------------------------------
    # training target (MC, from frozen train set)
    # ----------------------------------------------------------------------
    def make_training_target(
        self, param: str, batch_size: int, gen: torch.Generator | None = None
    ):
        """Build one training batch target pair (x_t, t, y) from cached clean train points."""
        assert param in PARAMS
        idx = torch.randint(
            0, self.n_train, (batch_size,), generator=gen, device=self.device
        )
        x_tilde = self.train_set[
            idx
        ]  # (B,d): draw clean latent-clean point x̃ from frozen set
        eps = (
            torch.randn(
                batch_size, self.d, generator=gen, device=self.device, dtype=self.dtype
            )
            * self.off_manifold_sigma
        )  # (B,d) forward-noise std off_manifold_sigma
        t = torch.rand(
            batch_size, generator=gen, device=self.device, dtype=self.dtype
        ).clamp(self.t_eps, 1.0 - self.t_eps)
        alpha = t.unsqueeze(-1)  # alpha_t
        beta = (1.0 - t).unsqueeze(-1)
        x_t = alpha * x_tilde + beta * eps  # (B,d), x_t = alpha x_tilde + (1-t) eps
        target = {"x": x_tilde, "eps": eps, "v": x_tilde - eps}[
            param
        ]  # MC training target for selected parameterization
        return x_t, t, target

    # ----------------------------------------------------------------------
    # geometry + memorization primitives
    # ----------------------------------------------------------------------
    def dist_to_manifold(self, x: Tensor) -> Tensor:
        """Return orthogonal distance of each sample to the k-dimensional manifold subspace."""
        x = x.to(self.device, self.dtype)
        return torch.linalg.norm(x - x @ self.A, dim=-1)  # A symmetric => (A x_b)

    def nn_ratio(self, x_query: Tensor) -> Tensor:
        """Compute nearest-neighbor distance ratio d1/d2 for each query against the frozen set."""
        x_query = x_query.to(self.device, self.dtype)
        D = torch.cdist(x_query, self.train_set)  # (B, n_train)
        d12, _ = torch.topk(D, k=2, largest=False, dim=1)  # nearest two
        return d12[:, 0] / (d12[:, 1] + 1e-12)

    @staticmethod
    def f_mem(r: Tensor, rho: float) -> Tensor:
        """Compute memorized fraction at threshold rho: mean indicator r < rho."""
        return (r < rho).to(r.dtype).mean()  # indicator-average over queries

    # ----------------------------------------------------------------------
    def _as_t(self, t, B: int) -> Tensor:
        # Accept scalar or batched t, then always return shape (B,).
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=self.dtype)
        t = t.to(self.device, self.dtype)
        if t.ndim == 0:
            t = t.expand(B).clone()
        assert t.shape == (B,), f"t must be scalar or ({B},), got {tuple(t.shape)}"
        return t

    def __repr__(self):
        """Human-readable object summary used in print/debug logs."""
        return (
            f"ManifoldGMM(d={self.d}, k={self.k}, M={self.M}, "
            f"mode_sigma={self.mode_sigma:g}, "
            f"off_manifold_sigma={self.off_manifold_sigma:g}, "
            f"n_train={self.n_train}, "
            f"dtype={self.dtype})"
        )


# ==========================================================================
# Notebook-ready usage + sanity checks
# ==========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    d, k, M = 8, 2, 3
    mode_sigma = 1e-2
    off_manifold_sigma = 1e-2

    ds = ManifoldGMM.random_gmm(
        d=d,
        k=k,
        n_modes=M,
        sep=4.0,
        mode_sigma=mode_sigma,
        off_manifold_sigma=off_manifold_sigma,
        n_train=2000,
        seed=0,
    )
    print(ds)

    # -- 1. geometry: A is an orthogonal projector of rank k ----------------
    A = ds.A
    print(
        "\n[1] projector A: symmetric=%s  idempotent=%s  trace=%.4f (==k=%d)"
        % (
            torch.allclose(A, A.T, atol=1e-8),
            torch.allclose(A @ A, A, atol=1e-6),
            A.diagonal().sum().item(),
            k,
        )
    )

    # -- 2. off-manifold thickness matches theory ---------------------------
    #    x = A x0 + off_sigma (I-A) xi  => E[dist^2] = off_sigma^2 (d-k)
    xb = ds.sample(20000, seed=123)
    md2 = (ds.dist_to_manifold(xb) ** 2).mean().item()
    print(
        "[2] mean dist^2 off manifold = %.3e   theory off_sigma^2 (d-k) = %.3e"
        % (md2, off_manifold_sigma**2 * (d - k))
    )

    # -- 3. fresh batch vs frozen train set are DISTINCT objects ------------
    fresh = ds.sample(5, seed=999)
    overlap = torch.cdist(fresh, ds.train_set).min().item()
    print("[3] min dist fresh-sample -> train_set = %.3e (should be > 0)" % overlap)

    # -- 4. ORACLE self-consistency: t*x0_hat + (1-t)*eps_hat == x_t --------
    #    (validates score -> {x0_hat, eps_hat} conversions and the t-orientation)
    B = 64
    x = ds.sample(B, seed=7)
    t = torch.rand(B, dtype=ds.dtype).clamp(ds.t_eps, 1 - ds.t_eps)
    x0_hat = ds.true_target(x, t, "x")
    eps_hat = ds.true_target(x, t, "eps")
    v_hat = ds.true_target(x, t, "v")
    recon = t.unsqueeze(-1) * x0_hat + (1 - t).unsqueeze(-1) * eps_hat
    print(
        "[4] max |t*x0_hat + (1-t)*eps_hat - x| = %.3e (should ~ 0)"
        % ((recon - x).abs().max().item())
    )
    print(
        "    max |v_hat - (x0_hat - eps_hat)|    = %.3e (should ~ 0)"
        % ((v_hat - (x0_hat - eps_hat)).abs().max().item())
    )

    # -- 5. score matches closed form for a single Gaussian (M=1) -----------
    mu1 = torch.zeros(1, d, dtype=torch.float64)
    sigma1 = 0.5
    ds1 = ManifoldGMM(
        mu1,
        torch.ones(1),
        k=d,
        mode_sigma=sigma1,
        off_manifold_sigma=sigma1,
        n_train=10,
        seed=0,
    )  # k=d => A=I, one isotropic Gaussian
    xq = torch.randn(16, d, dtype=torch.float64)
    tq = torch.full((16,), 0.6, dtype=torch.float64)
    # analytic: S = t^2 Sigma_tilde + (sigma(1-t))^2 I ; score = -S^{-1}(x - t*mu)
    S = (0.6**2) * ds1.Sigma_tilde[0] + ((0.4 * sigma1) ** 2) * torch.eye(
        d, dtype=torch.float64
    )
    m = 0.6 * ds1.mu_tilde[0]
    analytic = -(torch.linalg.solve(S, (xq - m).T)).T
    got = ds1.true_score(xq, tq)
    print(
        "[5] single-Gaussian score: max abs err vs analytic = %.3e"
        % ((got - analytic).abs().max().item())
    )

    # -- 6. training target shapes + manual consistency ---------------------
    g = torch.Generator().manual_seed(3)
    x_t, tt, tgt_v = ds.make_training_target("v", 32, gen=g)
    print(
        "[6] training target shapes: x_t=%s t=%s target=%s"
        % (tuple(x_t.shape), tuple(tt.shape), tuple(tgt_v.shape))
    )

    # -- 7. nn_ratio: a train point queried against itself => r ~ 0 ---------
    r_self = ds.nn_ratio(ds.train_set[:200])
    r_fresh = ds.nn_ratio(ds.sample(200, seed=55))
    print(
        "[7] nn_ratio on train points : mean r = %.3e (memorized => ~0)" % r_self.mean()
    )
    print(
        "    nn_ratio on fresh samples: mean r = %.3f   F_mem(rho=0.5)=%.3f"
        % (r_fresh.mean(), ManifoldGMM.f_mem(r_fresh, 0.5))
    )

    print("\nAll sanity checks ran.")
