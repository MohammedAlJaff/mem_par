"""Aggregate torus results into local CSVs and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate local torus results.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    from experiments.synthetic.high_d.common.runtime import (
        aggregate_over_seeds,
        collect_metrics,
        fit_boundaries_per_condition,
        load_yaml,
    )
    from experiments.synthetic.high_d.common.viz import plot_f_mem_vs_n, plot_n_star_vs_psi

    config = load_yaml(EXPERIMENT_DIR / "config.yaml")
    df = collect_metrics(EXPERIMENT_DIR)
    if df.empty:
        print(f"No metrics found under {EXPERIMENT_DIR / 'runs'}")
        return

    all_results_path = EXPERIMENT_DIR / "runs" / "all_results.csv"
    df.to_csv(all_results_path, index=False)

    agg = aggregate_over_seeds(df)
    agg_path = EXPERIMENT_DIR / "runs" / "aggregated_results.csv"
    agg.to_csv(agg_path, index=False)

    boundaries = fit_boundaries_per_condition(agg)
    boundaries_path = EXPERIMENT_DIR / "runs" / "boundaries.json"
    boundaries_path.write_text(json.dumps(boundaries, indent=2, default=str))

    for d_value in sorted(agg["d"].unique()):
        sub = agg[agg["d"] == d_value]
        psi = int(config["data"]["D"] / d_value)
        plot_f_mem_vs_n(
            sub,
            str(EXPERIMENT_DIR / "figures" / f"f_mem_vs_n_d{d_value}.pdf"),
            title=f"d={d_value}, psi={psi}",
        )

    psi_boundaries = {
        config["data"]["D"] / int(d_value): preds
        for d_value, preds in boundaries.items()
    }
    plot_n_star_vs_psi(
        psi_boundaries,
        str(EXPERIMENT_DIR / "figures" / "n_star_vs_psi.pdf"),
        title="Phase boundary vs. dimension ratio",
    )
    print(f"Saved aggregate outputs to {EXPERIMENT_DIR / 'runs'} and {EXPERIMENT_DIR / 'figures'}")


if __name__ == "__main__":
    main()
