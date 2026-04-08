from typing import Optional

import torch
from gdfm_components import GDFM, GDFMConfig


def make_discrete_ddpm_schedule(
    T: int,
    beta_min: float,
    beta_max: float,
    device: Optional[torch.device] = None,
):
    """
    Classic discrete DDPM schedule:
        beta_k linearly interpolated over k = 0, ..., T-1
        alpha_k = 1 - beta_k
        alpha_bar_k = prod_{j=0}^k alpha_j

    Returns:
        betas:      [T]
        alphas:     [T]
        alpha_bars: [T]
        signal:     [T] = sqrt(alpha_bar_k)
        noise:      [T] = sqrt(1 - alpha_bar_k)
    """
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")

    betas = torch.linspace(beta_min / T, beta_max / T, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    signal = torch.sqrt(alpha_bars)
    noise = torch.sqrt(torch.clamp(1.0 - alpha_bars, min=0.0))

    return betas, alphas, alpha_bars, signal, noise


def verify_fm_schedule(
    t_min: float = 1e-3,
    t_max: float = 1.0 - 1e-3,
    n_points: int = 9,
    device: Optional[torch.device] = None,
):
    """
    Verify FM:
        a(t) = t
        b(t) = 1 - t

    and print values from t_min to t_max.
    """
    cfg = GDFMConfig(
        family="fm",
        prediction_space="x",
        loss_space="x",
        data_dim=4,
        t_min=t_min,
        t_max=t_max,
    )
    model = GDFM(cfg)

    t = torch.linspace(t_min, t_max, n_points, device=device)
    a = model.a(t)
    b = model.b(t)

    print("=" * 80)
    print("FM schedule check")
    print("=" * 80)
    print(f"{'t':>10} {'a(t)':>14} {'b(t)':>14} {'a+b':>14}")
    for ti, ai, bi in zip(t.tolist(), a.tolist(), b.tolist()):
        print(f"{ti:10.6f} {ai:14.6f} {bi:14.6f} {(ai + bi):14.6f}")

    print()
    print(f"At t≈t_min={t_min}: expect a small, b near 1")
    print(f"At t≈t_max={t_max}: expect a near 1, b small")


def verify_vp_schedule_vs_continuous_and_discrete_ddpm(
    T: int = 1000,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    t_min: float = 1e-3,
    t_max: float = 1.0 - 1e-3,
    n_points: int = 9,
    device: Optional[torch.device] = None,
):
    """
    Compare:
        1. continuous VP schedule in this framework
        2. classic discrete DDPM schedule

    Mapping:
        continuous t in our framework:
            t=0 -> noisy/source end
            t=1 -> data end

        discrete DDPM step k:
            k=0     -> data-like
            k=T-1   -> noisy

        therefore compare at:
            t_k = 1 - k/(T-1)

    Important:
        Our continuous VP uses the analytic linear-beta ODE-style signal coefficient:
            signal_cont(t) = exp(-0.5 * ∫_0^{1-t} beta(u) du)

        The discrete DDPM uses:
            signal_disc[k] = sqrt(prod_j (1-beta_j))

        These should be close, not literally identical, because one is continuous-time
        and one is discrete-time.
    """
    cfg = GDFMConfig(
        family="vp",
        prediction_space="x",
        loss_space="x",
        data_dim=4,
        t_min=t_min,
        t_max=t_max,
        beta_min=beta_min,
        beta_max=beta_max,
    )
    model = GDFM(cfg)

    # --------------------------------------------------------
    # 1. Print continuous VP values over [t_min, t_max]
    # --------------------------------------------------------
    t = torch.linspace(t_min, t_max, n_points, device=device)
    signal_cont = model.a(t)
    noise_cont = model.b(t)

    print("=" * 80)
    print("Continuous VP schedule check")
    print("=" * 80)
    print(f"{'t':>10} {'signal=a(t)':>16} {'noise=b(t)':>16} {'a^2+b^2':>16}")
    for ti, si, ni in zip(t.tolist(), signal_cont.tolist(), noise_cont.tolist()):
        print(f"{ti:10.6f} {si:16.6f} {ni:16.6f} {(si**2 + ni**2):16.6f}")

    print()
    print(f"At t≈t_min={t_min}: expect signal small, noise near 1")
    print(f"At t≈t_max={t_max}: expect signal near 1, noise small")

    # --------------------------------------------------------
    # 2. Discrete DDPM schedule
    # --------------------------------------------------------
    _, _, alpha_bars, signal_disc, noise_disc = make_discrete_ddpm_schedule(
        T=T,
        beta_min=beta_min,
        beta_max=beta_max,
        device=device,
    )

    # --------------------------------------------------------
    # 3. Compare continuous VP against discrete DDPM
    # --------------------------------------------------------
    k = torch.linspace(0, T - 1, n_points, device=device)
    k = torch.round(k).long()

    # Our time orientation is reversed relative to DDPM index k
    t_from_k = 1.0 - k.float() / (T - 1)

    signal_cont_at_k = model.a(t_from_k)
    noise_cont_at_k = model.b(t_from_k)

    signal_disc_at_k = signal_disc[k]
    noise_disc_at_k = noise_disc[k]

    sig_err = (signal_cont_at_k - signal_disc_at_k).abs()
    noise_err = (noise_cont_at_k - noise_disc_at_k).abs()

    print()
    print("=" * 80)
    print("Continuous VP vs classic discrete DDPM")
    print("=" * 80)
    header = (
        f"{'k':>6} {'t=1-k/(T-1)':>16} "
        f"{'sig_cont':>12} {'sig_disc':>12} {'|diff|':>12} "
        f"{'noi_cont':>12} {'noi_disc':>12} {'|diff|':>12}"
    )
    print(header)

    for i in range(len(k)):
        print(
            f"{int(k[i]):6d} {t_from_k[i].item():16.6f} "
            f"{signal_cont_at_k[i].item():12.6f} {signal_disc_at_k[i].item():12.6f} {sig_err[i].item():12.6f} "
            f"{noise_cont_at_k[i].item():12.6f} {noise_disc_at_k[i].item():12.6f} {noise_err[i].item():12.6f}"
        )

    print()
    print(f"max |signal_cont - signal_disc| = {sig_err.max().item():.6e}")
    print(f"max |noise_cont  - noise_disc|  = {noise_err.max().item():.6e}")

    print()
    print("Interpretation:")
    print("1. The time flip is the important part: t = 1 - k/(T-1).")
    print(
        "2. Exact equality is not expected, because one schedule is continuous-time and the other is discrete."
    )
    print(
        "3. Qualitative agreement should be strong: data-like at k=0 <-> t=1, noisy at k=T-1 <-> t=0."
    )


def verify_all_schedules():
    verify_fm_schedule()
    print("\n")
    verify_vp_schedule_vs_continuous_and_discrete_ddpm()
