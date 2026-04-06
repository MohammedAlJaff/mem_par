"""Aggregate spiral-demo summaries into one CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate local spiral-demo summaries.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    import pandas as pd

    rows = []
    for summary_path in sorted((EXPERIMENT_DIR / "runs").rglob("summary.json")):
        payload = json.loads(summary_path.read_text())
        payload["run_dir"] = str(summary_path.parent)
        rows.append(payload)

    if not rows:
        print(f"No summary.json files found under {EXPERIMENT_DIR / 'runs'}")
        return

    df = pd.DataFrame(rows)
    output_path = EXPERIMENT_DIR / "runs" / "summary_table.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
