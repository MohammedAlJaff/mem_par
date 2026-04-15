"""
Lab 08 — External JiT walkthrough.

What this covers:
    - How to import and instantiate the external JiT model stack
    - The raw JiT interface: (x, t, y) -> x_pred
    - The external Denoiser wrapper: how it samples t, builds z_t, and forms
      the velocity-space loss used by Li & He
    - Shape tracing through the patch embedder, transformer blocks, and
      unpatchify step
    - Visual diagnostics: image panels plus a 2D scatter of patch tokens

Run from project root:
    python reference/walkthroughs/08_external_jit_walkthrough.py
    python reference/walkthroughs/08_external_jit_walkthrough.py --device cuda --image-size 64

Notes:
    - This script does NOT modify the external repository on disk.
    - The external JiT code assumes CUDA in a few places. We patch those
      assumptions locally inside this script so the walkthrough can run on
      either CPU or GPU.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).parent.parent
EXTERNAL_JIT_ROOT = REPO_ROOT / "baselines" / "jit" / "vendor"


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


def ensure_external_jit_available() -> None:
    if not EXTERNAL_JIT_ROOT.exists():
        raise FileNotFoundError(
            f"Expected external JiT repository at {EXTERNAL_JIT_ROOT}, but it was not found."
        )
    if str(EXTERNAL_JIT_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_JIT_ROOT))


def install_device_safe_patches() -> tuple[object, object]:
    """
    Patch the external JiT code in-memory so it does not hardcode `.cuda()`.

    We keep the external repo read-only and apply tiny runtime replacements:
      1. A device-safe rotary embedding class
      2. A device-safe attention helper
    """
    ensure_external_jit_available()

    import model_jit
    import util.model_util as model_util

    class DeviceSafeVisionRotaryEmbeddingFast(torch.nn.Module):
        def __init__(
            self,
            dim,
            pt_seq_len=16,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for="lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            num_cls_token=0,
        ):
            super().__init__()

            if custom_freqs is not None:
                freqs = custom_freqs
            elif freqs_for == "lang":
                freqs = 1.0 / (
                    theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
                )
            elif freqs_for == "pixel":
                freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
            elif freqs_for == "constant":
                freqs = torch.ones(num_freqs).float()
            else:
                raise ValueError(f"unknown modality {freqs_for}")

            if ft_seq_len is None:
                ft_seq_len = pt_seq_len

            t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
            freqs = torch.einsum("..., f -> ... f", t, freqs)
            freqs = model_util.repeat(freqs, "... n -> ... (n r)", r=2)
            freqs = model_util.broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

            if num_cls_token > 0:
                freqs_flat = freqs.view(-1, freqs.shape[-1])
                cos_img = freqs_flat.cos()
                sin_img = freqs_flat.sin()

                n_img, d_model = cos_img.shape
                cos_pad = torch.ones(num_cls_token, d_model, dtype=cos_img.dtype)
                sin_pad = torch.zeros(num_cls_token, d_model, dtype=sin_img.dtype)

                freqs_cos = torch.cat([cos_pad, cos_img], dim=0)
                freqs_sin = torch.cat([sin_pad, sin_img], dim=0)
            else:
                freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
                freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        def forward(self, t):
            freqs_cos = self.freqs_cos.to(device=t.device, dtype=t.dtype)
            freqs_sin = self.freqs_sin.to(device=t.device, dtype=t.dtype)
            return t * freqs_cos + model_util.rotate_half(t) * freqs_sin

    def device_safe_scaled_dot_product_attention(query, key, value, dropout_p=0.0):
        l_tokens, s_tokens = query.size(-2), key.size(-2)
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(
            query.size(0),
            1,
            l_tokens,
            s_tokens,
            dtype=query.dtype,
            device=query.device,
        )

        attn_weight = query.float() @ key.float().transpose(-2, -1)
        attn_weight = attn_weight * scale_factor + attn_bias.float()
        attn_weight = torch.softmax(attn_weight, dim=-1).to(query.dtype)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=dropout_p > 0)
        return attn_weight @ value

    model_util.VisionRotaryEmbeddingFast = DeviceSafeVisionRotaryEmbeddingFast
    model_jit.VisionRotaryEmbeddingFast = DeviceSafeVisionRotaryEmbeddingFast
    model_jit.scaled_dot_product_attention = device_safe_scaled_dot_product_attention

    return model_jit, model_util


def build_demo_batch(batch_size: int, image_size: int, device: str) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, image_size, device=device)
    xs = torch.linspace(-1.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    images = []
    for idx in range(batch_size):
        freq = idx + 1
        center_x = -0.5 + (idx / max(1, batch_size - 1))
        center_y = 0.5 - (idx / max(1, batch_size - 1))

        red = torch.sin(freq * math.pi * xx)
        green = torch.cos(freq * math.pi * yy)
        blue = torch.exp(-6.0 * ((xx - center_x) ** 2 + (yy - center_y) ** 2))

        image = torch.stack([red, green, blue], dim=0)
        image = image / image.abs().amax().clamp_min(1e-6)
        images.append(image)

    return torch.stack(images, dim=0)


def to_rgb_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    if x.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(x.shape)}")

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    return x.permute(1, 2, 0).numpy()


def pca_2d(points: torch.Tensor) -> np.ndarray:
    points_np = points.detach().float().cpu().numpy()
    points_np = points_np - points_np.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(points_np, full_matrices=False)
    basis = vt[:2].T
    return points_np @ basis


def make_denoiser_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        img_size=args.image_size,
        class_num=args.class_num,
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        noise_scale=1.0,
        t_eps=5e-2,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.sampling_steps,
        cfg=args.cfg_scale,
        interval_min=0.0,
        interval_max=1.0,
    )


def trace_raw_jit(model, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> dict:
    print_section("1. Raw JiT Trace")
    describe_tensor("input images x", x)
    describe_tensor("continuous time t", t)
    describe_tensor("class labels y", labels.float())

    t_emb = model.t_embedder(t)
    y_emb = model.y_embedder(labels)
    cond = t_emb + y_emb

    describe_tensor("t embedding", t_emb)
    describe_tensor("y embedding", y_emb)
    describe_tensor("conditioning c", cond)

    proj1 = model.x_embedder.proj1(x)
    proj2 = model.x_embedder.proj2(proj1)
    tokens = model.x_embedder(x)
    tokens_pos = tokens + model.pos_embed

    describe_tensor("proj1 patches", proj1)
    describe_tensor("proj2 patches", proj2)
    describe_tensor("patch tokens", tokens)
    describe_tensor("tokens + pos", tokens_pos)

    hidden = tokens_pos
    hidden_after_first = None
    hidden_after_context = None

    for idx, block in enumerate(model.blocks):
        if model.in_context_len > 0 and idx == model.in_context_start:
            in_context_tokens = y_emb.unsqueeze(1).repeat(1, model.in_context_len, 1)
            in_context_tokens = in_context_tokens + model.in_context_posemb
            print(
                f"Inserting {model.in_context_len} in-context tokens at transformer block {idx}."
            )
            describe_tensor("in-context tokens", in_context_tokens)
            hidden = torch.cat([in_context_tokens, hidden], dim=1)
            describe_tensor("tokens after concat", hidden)
            hidden_after_context = hidden

        rope = model.feat_rope if idx < model.in_context_start else model.feat_rope_incontext
        hidden = block(hidden, cond, rope)

        if idx == 0:
            hidden_after_first = hidden
            describe_tensor("after block 0", hidden_after_first)
        if idx == len(model.blocks) - 1:
            describe_tensor("after final block", hidden)

    hidden_no_context = hidden[:, model.in_context_len :]
    patch_predictions = model.final_layer(hidden_no_context, cond)
    x_pred_manual = model.unpatchify(patch_predictions, model.patch_size)
    x_pred_direct = model(x, t, labels)

    describe_tensor("hidden w/o context", hidden_no_context)
    describe_tensor("patch predictions", patch_predictions)
    describe_tensor("manual x_pred", x_pred_manual)
    describe_tensor("direct x_pred", x_pred_direct)

    diff = (x_pred_manual - x_pred_direct).abs().max().item()
    print(f"Max |manual - direct| = {diff:.3e}")
    print(
        f"Patch math: {model.x_embedder.num_patches} patches, "
        f"patch size {model.patch_size}, hidden size {model.hidden_size}"
    )

    return {
        "tokens_pre": tokens_pos[0],
        "tokens_post": hidden_no_context[0],
        "x_pred": x_pred_direct,
        "x_pred_manual": x_pred_manual,
        "hidden_after_first": hidden_after_first,
        "hidden_after_context": hidden_after_context,
    }


def trace_denoiser_objective(denoiser, x: torch.Tensor, labels: torch.Tensor) -> dict:
    print_section("2. Denoiser Objective Trace")

    t = denoiser.sample_t(x.size(0), device=x.device).view(-1, 1, 1, 1)
    e = torch.randn_like(x) * denoiser.noise_scale
    z_t = t * x + (1.0 - t) * e
    v_target = (x - z_t) / (1.0 - t).clamp_min(denoiser.t_eps)

    x_pred = denoiser.net(z_t, t.flatten(), labels)
    v_pred = (x_pred - z_t) / (1.0 - t).clamp_min(denoiser.t_eps)
    loss = ((v_target - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

    describe_tensor("sampled t", t.squeeze(-1).squeeze(-1).squeeze(-1))
    describe_tensor("noise e", e)
    describe_tensor("interpolant z_t", z_t)
    describe_tensor("velocity target", v_target)
    describe_tensor("x_pred from net", x_pred)
    describe_tensor("velocity prediction", v_pred)
    print(f"Velocity-space MSE loss = {loss.item():.6f}")

    return {
        "t": t,
        "noise": e,
        "z_t": z_t,
        "v_target": v_target,
        "x_pred": x_pred,
        "v_pred": v_pred,
        "loss": loss,
    }


@torch.no_grad()
def run_sampling_demo(denoiser, labels: torch.Tensor) -> torch.Tensor:
    print_section("3. JiT Sampling Demo")
    generated = denoiser.generate(labels)
    describe_tensor("generated samples", generated)
    return generated


def save_image_panel(
    output_path: Path,
    x: torch.Tensor,
    objective: dict,
    generated: torch.Tensor | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()

    panels = [
        ("clean x", x[0]),
        ("interpolant z_t", objective["z_t"][0]),
        ("x_pred", objective["x_pred"][0]),
        ("v_target", objective["v_target"][0]),
        ("v_pred", objective["v_pred"][0]),
        ("generated", generated[0] if generated is not None else objective["noise"][0]),
    ]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(to_rgb_image(image))
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("External JiT: image-space checkpoints through the forward path")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_token_scatter(output_path: Path, tokens_pre: torch.Tensor, tokens_post: torch.Tensor) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pre_2d = pca_2d(tokens_pre)
    post_2d = pca_2d(tokens_post)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pre_2d[:, 0], pre_2d[:, 1], s=35, alpha=0.8, c=np.arange(len(pre_2d)))
    axes[0].set_title("Patch tokens before transformer")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(post_2d[:, 0], post_2d[:, 1], s=35, alpha=0.8, c=np.arange(len(post_2d)))
    axes[1].set_title("Patch tokens after transformer")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.suptitle("External JiT: 2D view of one sample's patch tokens")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pedagogical walkthrough for external JiT")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--model", default="JiT-B/16")
    parser.add_argument("--class-num", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--sampling-method", default="heun", choices=["euler", "heun"])
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--save-dir",
        default=str(REPO_ROOT / "output" / "lab-external" / "jit_walkthrough"),
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip the ODE sampling demo and only trace the forward/loss path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Lab 08: External JiT walkthrough")
    print(f"device={args.device}  model={args.model}  image_size={args.image_size}")
    print("=" * 72)

    model_jit, _model_util = install_device_safe_patches()
    import denoiser as jit_denoiser

    device = torch.device(args.device)

    x = build_demo_batch(args.batch_size, args.image_size, args.device)
    labels = torch.arange(args.batch_size, device=device) % args.class_num
    t_demo = torch.linspace(0.1, 0.9, args.batch_size, device=device)

    raw_model = model_jit.JiT_models[args.model](
        input_size=args.image_size,
        in_channels=3,
        num_classes=args.class_num,
        attn_drop=0.0,
        proj_drop=0.0,
    ).to(device)
    raw_model.eval()

    print_section("0. Setup")
    print(
        f"Raw JiT config: patch_size={raw_model.patch_size}, "
        f"num_patches={raw_model.x_embedder.num_patches}, "
        f"hidden_size={raw_model.hidden_size}, "
        f"in_context_len={raw_model.in_context_len}"
    )
    describe_tensor("demo batch", x)

    raw_trace = trace_raw_jit(raw_model, x, t_demo, labels)

    denoiser_args = make_denoiser_args(args)
    model_for_loss = jit_denoiser.Denoiser(denoiser_args).to(device)
    model_for_loss.eval()
    objective = trace_denoiser_objective(model_for_loss, x, labels)

    generated = None
    if not args.skip_generate:
        generated = run_sampling_demo(model_for_loss, labels)

    image_panel_path = save_dir / "jit_image_flow.png"
    token_scatter_path = save_dir / "jit_patch_token_scatter.png"

    save_image_panel(image_panel_path, x, objective, generated)
    save_token_scatter(token_scatter_path, raw_trace["tokens_pre"], raw_trace["tokens_post"])

    print_section("4. Saved Outputs")
    print(f"Saved image panel:  {image_panel_path}")
    print(f"Saved token scatter:{token_scatter_path}")


if __name__ == "__main__":
    main()
