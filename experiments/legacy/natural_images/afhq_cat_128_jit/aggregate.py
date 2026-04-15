"""Aggregate AFHQ cat 128x128 JiT results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate local AFHQ 128 JiT results.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    from experiments.natural_images.common.runtime import (
        aggregate_over_seeds,
        collect_metrics,
        fit_boundaries_per_condition,
    )
    from experiments.natural_images.common.viz import plot_f_mem_vs_n

    df = collect_metrics(EXPERIMENT_DIR)
    if df.empty:
        print(f"No metrics found under {EXPERIMENT_DIR / 'runs'}")
        return

    df.to_csv(EXPERIMENT_DIR / "runs" / "all_results.csv", index=False)
    agg = aggregate_over_seeds(df)
    agg.to_csv(EXPERIMENT_DIR / "runs" / "aggregated_results.csv", index=False)

    boundaries = fit_boundaries_per_condition(agg)
    (EXPERIMENT_DIR / "runs" / "boundaries.json").write_text(
        json.dumps(boundaries, indent=2, default=str)
    )

    plot_f_mem_vs_n(
        agg,
        str(EXPERIMENT_DIR / "figures" / "f_mem_vs_n.pdf"),
        title="AFHQ cat 128 JiT",
    )
    print(f"Saved aggregate outputs to {EXPERIMENT_DIR / 'runs'} and {EXPERIMENT_DIR / 'figures'}")


if __name__ == "__main__":
    main()
