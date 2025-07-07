import csv
import os
from pathlib import Path


def save_metrics_to_csv(
    metrics: dict[str, float],
    csv_path: Path,
    exp_name: str,
    checkpoint: str,
):
    is_empty = not csv_path.exists() or os.path.getsize(csv_path) == 0

    row_data = {"exp": exp_name}
    row_data.update(metrics)
    row_data.update({"checkpoint": checkpoint})

    # Write to CSV file
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["exp"] + list(metrics.keys()) + ["checkpoint"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is empty
        if is_empty:
            writer.writeheader()

        # Write data row
        writer.writerow(row_data)

    print(f"Experiment results saved to: {csv_path}")
