"""
Lab 09 — External Bonnaire UNet walkthrough.

What this covers:
    - How to import the external Bonnaire UNet and DDPM-style diffusion code
    - The expected interface: model(x_t, t_discrete) -> predicted_noise
    - How forward diffusion constructs x_t from clean data x_0 and noise eps
    - Shape tracing through the encoder, bottleneck, skip connections, and decoder
    - Visual diagnostics: image panels plus scatter plots of predicted-vs-true
      noise norms

Run from project root:
    python reference/walkthroughs/09_external_bonnaire_unet_walkthrough.py
    python reference/walkthroughs/09_external_bonnaire_unet_walkthrough.py --device cuda --image-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).parent.parent
BONNAIRE_UTILS_ROOT = (
    REPO_ROOT / "baselines" / "bonnaire" / "vendor" / "Experiments" / "src" / "Utils"
)


def print_section(title: str) -> None:
    print("\n" + "─" * 72)
    print(title)
    print("─" * 72)


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor_cpu = tensor.detach().float().cpu()
    print(
        f"{name:24s} "
        f"shape={tuple(tensor.shape)!s:18s} "
        f"dtype={str(tensor.dtype):14s} "
        f"device={str(tensor.device):10s} "
        f"min={tensor_cpu.min():8.4f} "
        f"max={tensor_cpu.max():8.4f} "
        f"mean={tensor_cpu.mean():8.4f} "
        f"std={tensor_cpu.std():8.4f}"
    )


def ensure_bonnaire_available() -> None:
    if not BONNAIRE_UTILS_ROOT.exists():
        raise FileNotFoundError(
            f"Expected Bonnaire utils path at {BONNAIRE_UTILS_ROOT}, but it was not found."
        )
    if str(BONNAIRE_UTILS_ROOT) not in sys.path:
        sys.path.insert(0, str(BONNAIRE_UTILS_ROOT))


def build_demo_batch(batch_size: int, image_size: int, device: str) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, image_size, device=device)
    xs = torch.linspace(-1.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    images = []
    for idx in range(batch_size):
        center_x = -0.6 + 1.2 * (idx / max(1, batch_size - 1))
        center_y = 0.6 - 1.2 * (idx / max(1, batch_size - 1))

        ring = torch.exp(-10.0 * ((torch.sqrt(xx**2 + yy**2) - 0.45) ** 2))
        blob = torch.exp(-12.0 * ((xx - center_x) ** 2 + (yy - center_y) ** 2))
        stripes = torch.sin((idx + 1) * np.pi * xx) * torch.cos(np.pi * yy)

        image = 0.55 * ring + 0.35 * blob + 0.25 * stripes
        image = image / image.abs().amax().clamp_min(1e-6)
        images.append(image.unsqueeze(0))

    return torch.stack(images, dim=0)


def to_display_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    if x.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(x.shape)}")

    if x.shape[0] == 1:
        image = x[0]
        image = image - image.min()
        image = image / image.max().clamp_min(1e-6)
        return image.numpy()

    image = (x + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def make_config(device: str, image_size: int, channels: int, timesteps: int) -> SimpleNamespace:
    return SimpleNamespace(
        DEVICE=device,
        IMG_SHAPE=(channels, image_size, image_size),
        TIMESTEPS=timesteps,
        mean=0.0,
        std=1.0,
    )


def trace_unet_forward(unet_module, model, x_t: torch.Tensor, ts: torch.Tensor) -> dict:
    print_section("2. External UNet Trace")
    describe_tensor("noisy input x_t", x_t)
    describe_tensor("discrete times t", ts.float())

    time_emb = model.time_embeddings(ts.flatten().to(torch.long))
    describe_tensor("time embedding", time_emb)

    h = model.init_conv(x_t)
    describe_tensor("after init_conv", h)

    outs = [h]

    for idx, layer in enumerate(model.encoder_blocks):
        layer_name = layer.__class__.__name__
        h = layer(h, time_emb)
        outs.append(h)
        describe_tensor(f"encoder[{idx}] {layer_name}", h)

    for idx, layer in enumerate(model.bottleneck_blocks):
        h = layer(h, time_emb)
        describe_tensor(f"bottleneck[{idx}]", h)

    for idx, layer in enumerate(model.decoder_blocks):
        layer_name = layer.__class__.__name__
        if isinstance(layer, unet_module.ResnetBlock):
            skip = outs.pop()
            describe_tensor(f"skip for decoder[{idx}]", skip)
            h = torch.cat([h, skip], dim=1)
            describe_tensor(f"concat before decoder[{idx}]", h)
        h = layer(h, time_emb)
        describe_tensor(f"decoder[{idx}] {layer_name}", h)

    pred_noise_manual = model.final_block(h)
    pred_noise_direct = model(x_t, ts)

    describe_tensor("manual predicted noise", pred_noise_manual)
    describe_tensor("direct predicted noise", pred_noise_direct)
    diff = (pred_noise_manual - pred_noise_direct).abs().max().item()
    print(f"Max |manual - direct| = {diff:.3e}")

    return {
        "pred_noise_manual": pred_noise_manual,
        "pred_noise_direct": pred_noise_direct,
    }


@torch.no_grad()
def sample_with_ddim(diffusion_module, model, config, df, n_images: int, ddim_steps: int):
    print_section("4. DDIM Sampling Demo")
    generated, init_noise = diffusion_module.sample_diffusion_from_noise_DDIM(
        model,
        n_images=n_images,
        config=config,
        df=df,
        dim=4,
        eta=0.0,
        ddim_steps=ddim_steps,
    )
    describe_tensor("initial noise", init_noise)
    describe_tensor("generated samples", generated)
    return generated, init_noise


def save_image_panel(
    output_path: Path,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    true_noise: torch.Tensor,
    pred_noise: torch.Tensor,
    residual: torch.Tensor,
    generated: torch.Tensor | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()

    panels = [
        ("clean x_0", clean[0]),
        ("noisy x_t", noisy[0]),
        ("true noise eps", true_noise[0]),
        ("predicted noise", pred_noise[0]),
        ("prediction residual", residual[0]),
        ("generated sample" if generated is not None else "clean x_0 (repeat)",
         generated[0] if generated is not None else clean[0]),
    ]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(to_display_image(image), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("External Bonnaire UNet: forward diffusion and denoising")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_norm_scatter(
    output_path: Path,
    ts: torch.Tensor,
    true_noise: torch.Tensor,
    pred_noise: torch.Tensor,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    true_norms = true_noise.flatten(1).norm(dim=1).detach().cpu().numpy()
    pred_norms = pred_noise.flatten(1).norm(dim=1).detach().cpu().numpy()
    ts_np = ts.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(true_norms, pred_norms, s=60, alpha=0.8, c=ts_np, cmap="viridis")
    low = min(true_norms.min(), pred_norms.min())
    high = max(true_norms.max(), pred_norms.max())
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set_xlabel("||true noise||")
    axes[0].set_ylabel("||predicted noise||")
    axes[0].set_title("Per-sample noise norm comparison")

    mse_per_sample = ((true_noise - pred_noise) ** 2).flatten(1).mean(dim=1).detach().cpu().numpy()
    axes[1].scatter(ts_np, mse_per_sample, s=60, alpha=0.8, color="#d95f02")
    axes[1].set_xlabel("diffusion timestep t")
    axes[1].set_ylabel("per-sample MSE")
    axes[1].set_title("Error as a function of timestep")

    fig.suptitle("External Bonnaire UNet: diagnostic scatter plots")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pedagogical walkthrough for the external Bonnaire UNet"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=12)
    parser.add_argument(
        "--save-dir",
        default=str(REPO_ROOT / "output" / "lab-external" / "bonnaire_unet_walkthrough"),
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip the untrained DDIM sampling demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ensure_bonnaire_available()
    import Diffusion
    import Unet

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 72)
    print("Lab 09: External Bonnaire UNet walkthrough")
    print(
        f"device={args.device}  image_size={args.image_size}  "
        f"channels={args.channels}  base_channels={args.base_channels}"
    )
    print("=" * 72)

    clean = build_demo_batch(args.batch_size, args.image_size, args.device)
    if args.channels == 3:
        clean = clean.repeat(1, 3, 1, 1)

    config = make_config(args.device, args.image_size, args.channels, args.timesteps)
    df = Diffusion.DiffusionConfig(
        n_steps=args.timesteps,
        img_shape=config.IMG_SHAPE,
        device=args.device,
    )

    model = Unet.UNet(
        input_channels=args.channels,
        output_channels=args.channels,
        base_channels=args.base_channels,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    ).to(device)
    model.eval()

    print_section("0. Setup")
    describe_tensor("clean batch x_0", clean)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameter count: {n_params:,}")
    print(
        "Time convention: the Bonnaire UNet expects integer diffusion steps "
        f"t in [0, {args.timesteps - 1}] rather than continuous t in [0, 1]."
    )

    ts = torch.linspace(50, args.timesteps - 50, args.batch_size, device=device).round().long()

    print_section("1. Forward Diffusion")
    noisy, true_noise = Diffusion.forward_diffusion(df, clean, ts, config)
    describe_tensor("diffused sample x_t", noisy)
    describe_tensor("ground-truth noise", true_noise)

    trace = trace_unet_forward(Unet, model, noisy, ts)
    pred_noise = trace["pred_noise_direct"]
    residual = pred_noise - true_noise

    print_section("3. Noise Prediction Summary")
    describe_tensor("predicted noise", pred_noise)
    describe_tensor("prediction residual", residual)
    loss = F.mse_loss(pred_noise, true_noise)
    print(f"MSE(pred_noise, true_noise) = {loss.item():.6f}")

    generated = None
    if not args.skip_generate:
        generated, _init_noise = sample_with_ddim(
            Diffusion,
            model,
            config,
            df,
            n_images=min(4, args.batch_size),
            ddim_steps=args.ddim_steps,
        )

    image_panel_path = save_dir / "bonnaire_unet_image_flow.png"
    scatter_path = save_dir / "bonnaire_unet_diagnostics.png"

    save_image_panel(
        image_panel_path,
        clean,
        noisy,
        true_noise,
        pred_noise,
        residual,
        generated,
    )
    save_norm_scatter(scatter_path, ts, true_noise, pred_noise)

    print_section("5. Saved Outputs")
    print(f"Saved image panel: {image_panel_path}")
    print(f"Saved scatter:     {scatter_path}")


if __name__ == "__main__":
    main()
