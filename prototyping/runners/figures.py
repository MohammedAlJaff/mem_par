"""Plotting helpers for synthetic prototype runs."""

from pathlib import Path


def make_synthetic_figures(
    *,
    figures_dir,
    train_history,
    metric_records,
    train_tensor,
    reference_tensor,
    final_samples,
    train_latent,
    reference_latent,
    final_samples_latent,
    final_samples_residual_norm,
    baseline_residual_norm,
    latent_means,
    ambient_means,
    context,
):
    figure_paths = {}
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warning_path = Path(figures_dir) / "figure_generation_error.txt"
        warning_path.write_text("{}: {}".format(type(exc).__name__, exc), encoding="utf-8")
        return {"figure_generation_error": str(warning_path)}

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if train_history:
        path = figures_dir / "training_loss_vs_step.png"
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot([row["step"] for row in train_history], [row["loss"] for row in train_history], linewidth=1.8)
        ax.set_title("Training loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths["training_loss_vs_step"] = str(path)

    if metric_records:
        _metric_curves(plt, figures_dir, figure_paths, metric_records, context)

    if train_latent.shape[1] >= 2:
        _latent_scatters(
            plt,
            figures_dir,
            figure_paths,
            train_latent,
            reference_latent,
            final_samples_latent,
            latent_means,
            context,
        )

    if train_tensor.shape[1] == 2:
        _ambient_scatter(
            plt,
            figures_dir,
            figure_paths,
            train_tensor,
            reference_tensor,
            final_samples,
            ambient_means,
            context,
        )

    _residual_histogram(
        plt,
        figures_dir,
        figure_paths,
        baseline_residual_norm,
        final_samples_residual_norm,
        context,
    )

    if metric_records and metric_records[-1].get("component_probs"):
        _component_coverage(plt, figures_dir, figure_paths, metric_records[-1], context)

    return figure_paths


def _metric_curves(plt, figures_dir, figure_paths, metric_records, context):
    steps = [row["step"] for row in metric_records]
    plot_specs = [
        (
            "f_mem_vs_step",
            "Memorization fraction",
            "f_mem",
            [
                ("generated ambient", [row["ambient_f_mem"] for row in metric_records], "-"),
                ("generated latent", [row["latent_f_mem"] for row in metric_records], "-"),
                ("reference ambient", [row["baseline_ambient_f_mem"] for row in metric_records], "--"),
                ("reference latent", [row["baseline_latent_f_mem"] for row in metric_records], "--"),
            ],
        ),
        (
            "oracle_nll_vs_step",
            "Oracle GMM negative log-likelihood",
            "NLL",
            [
                ("generated latent NLL", [row["oracle_latent_nll_mean"] for row in metric_records], "-"),
                ("reference latent NLL", [row["baseline_oracle_latent_nll_mean"] for row in metric_records], "--"),
            ],
        ),
        (
            "latent_distance_vs_step",
            "Latent distribution distances",
            "distance",
            [
                ("latent Frechet", [row["latent_frechet_distance_to_oracle"] for row in metric_records], "-"),
                ("latent sliced-W2", [row["latent_sliced_w2"] for row in metric_records], "-"),
            ],
        ),
        (
            "residual_energy_vs_step",
            "Off-subspace energy",
            "residual / total energy",
            [
                ("generated", [row["off_subspace_energy_fraction"] for row in metric_records], "-"),
                ("reference", [row["baseline_off_subspace_energy_fraction"] for row in metric_records], "--"),
            ],
        ),
    ]
    for name, title, ylabel, series in plot_specs:
        path = figures_dir / "{}.png".format(name)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for label, values, linestyle in series:
            ax.plot(steps, values, marker="o", linestyle=linestyle, label=label)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths[name] = str(path)


def _latent_scatters(plt, figures_dir, figure_paths, train_latent, reference_latent, samples_latent, means, context):
    import torch

    limits = torch.cat([train_latent, reference_latent, samples_latent, means], dim=0)
    x_min = float(limits[:, 0].min().item()) - 0.5
    x_max = float(limits[:, 0].max().item()) + 0.5
    y_min = float(limits[:, 1].min().item()) - 0.5
    y_max = float(limits[:, 1].max().item()) + 0.5

    for name, title, generated in (
        ("latent_data_scatter", "Data in true latent coordinates", False),
        ("latent_generated_scatter", "Generated samples in true latent coordinates", True),
    ):
        path = figures_dir / "{}.png".format(name)
        fig, ax = plt.subplots(figsize=(5.8, 6.1))
        ax.scatter(train_latent[:, 0].numpy(), train_latent[:, 1].numpy(), s=24, alpha=0.35, label="train")
        if generated:
            ax.scatter(samples_latent[:, 0].numpy(), samples_latent[:, 1].numpy(), s=14, alpha=0.75, label="generated")
        else:
            ax.scatter(reference_latent[:, 0].numpy(), reference_latent[:, 1].numpy(), s=8, alpha=0.20, label="reference")
        ax.scatter(means[:, 0].numpy(), means[:, 1].numpy(), s=90, marker="x", linewidths=2.0, label="latent means")
        ax.set_title(title)
        ax.set_xlabel("x @ B [0]")
        ax.set_ylabel("x @ B [1]")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.20)
        ax.legend()
        fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths[name] = str(path)


def _ambient_scatter(plt, figures_dir, figure_paths, train, reference, samples, means, context):
    import torch

    points = torch.cat([train, reference, samples, means], dim=0)
    x_min = float(points[:, 0].min().item()) - 0.5
    x_max = float(points[:, 0].max().item()) + 0.5
    y_min = float(points[:, 1].min().item()) - 0.5
    y_max = float(points[:, 1].max().item()) + 0.5

    path = figures_dir / "ambient_generated_scatter.png"
    fig, ax = plt.subplots(figsize=(5.8, 6.1))
    ax.scatter(train[:, 0].numpy(), train[:, 1].numpy(), s=24, alpha=0.25, label="train")
    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=14, alpha=0.75, label="generated")
    ax.scatter(means[:, 0].numpy(), means[:, 1].numpy(), s=90, marker="x", linewidths=2.0, label="ambient means")
    ax.set_title("Generated samples in ambient coordinates")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths["ambient_generated_scatter"] = str(path)


def _residual_histogram(plt, figures_dir, figure_paths, baseline_residual_norm, samples_residual_norm, context):
    path = figures_dir / "residual_norm_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(baseline_residual_norm.numpy(), bins=32, alpha=0.50, label="reference")
    ax.hist(samples_residual_norm.numpy(), bins=32, alpha=0.50, label="generated")
    ax.set_title("Off-subspace residual norms")
    ax.set_xlabel("||x - (x @ B) B.T||")
    ax.set_ylabel("count")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths["residual_norm_histogram"] = str(path)


def _component_coverage(plt, figures_dir, figure_paths, record, context):
    path = figures_dir / "component_coverage_final.png"
    xs = list(range(len(record["component_probs"])))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar([x - width / 2.0 for x in xs], record["baseline_component_probs"], width=width, label="reference")
    ax.bar([x + width / 2.0 for x in xs], record["component_probs"], width=width, label="generated")
    ax.axhline(1.0 / float(len(xs)), linestyle="--", linewidth=1.2, label="uniform")
    ax.set_title("Final GMM component coverage")
    ax.set_xlabel("component")
    ax.set_ylabel("mean responsibility")
    ax.grid(alpha=0.20, axis="y")
    ax.legend()
    fig.text(0.5, 0.02, context, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths["component_coverage_final"] = str(path)
