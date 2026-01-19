import argparse
import csv
import os
from glob import glob
from typing import Dict, List, Optional


def is_threshold_metric(name: str) -> bool:
    for thr in ("10", "20", "30"):
        if name.startswith(f"_{thr}") or f"_{thr}_" in name:
            return True
    return False


def read_metrics(path: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return metrics
    header = rows[0]
    mean_idx = header.index("MEAN") if "MEAN" in header else len(header) - 1
    for row in rows[1:]:
        if not row:
            continue
        name = row[0].strip()
        if not name or is_threshold_metric(name):
            continue
        val = row[mean_idx].strip() if len(row) > mean_idx else ""
        metrics[name] = val
    return metrics


def to_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare evaluation CSVs by listing non-threshold metrics per file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory containing evaluation CSV files (default: output).",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="compare.csv",
        help="Optional path to save the combined comparison table as CSV.",
    )
    args = parser.parse_args()

    csv_paths = sorted(glob(os.path.join(args.output_dir, "*.csv")))
    if not csv_paths:
        print(f"No CSV files found under {args.output_dir}")
        return

    metrics_per_file: List[Dict[str, str]] = []
    basenames: List[str] = []
    all_metric_names: set = set()

    for csv_path in csv_paths:
        metrics = read_metrics(csv_path)
        metrics_per_file.append(metrics)
        basenames.append(os.path.basename(csv_path))
        all_metric_names.update(metrics.keys())

    if not all_metric_names:
        print("No metrics found in CSVs.")
        return

    metric_names = sorted(all_metric_names)

    # Build display matrix with bracketed maxima per metric
    display_rows: List[List[str]] = []
    for m in metric_names:
        raw_vals = [metrics.get(m, "") for metrics in metrics_per_file]
        nums = [to_float(v) for v in raw_vals]
        max_val = None
        for n in nums:
            if n is not None:
                max_val = n if max_val is None else max(max_val, n)
        disp_vals: List[str] = []
        for v, n in zip(raw_vals, nums):
            if max_val is not None and n is not None and n == max_val:
                disp_vals.append(f"[{v}]")
            else:
                disp_vals.append(v)
        display_rows.append([m, *disp_vals])

    col_widths = [max(len("METRIC"), max(len(row[0]) for row in display_rows))]
    for idx, name in enumerate(basenames):
        width = max(len(name), max(len(row[idx + 1]) for row in display_rows), 4)
        col_widths.append(width)

    # Header
    header_cells = ["METRIC".ljust(col_widths[0])]
    for name, width in zip(basenames, col_widths[1:]):
        header_cells.append(name.ljust(width))
    print("  ".join(header_cells))
    print("  ".join("-" * w for w in col_widths))

    table_rows: List[List[str]] = []
    for row in display_rows:
        row_cells = [row[0].ljust(col_widths[0])]
        row_csv = [row[0]]
        for cell, width in zip(row[1:], col_widths[1:]):
            row_cells.append(cell.ljust(width))
            row_csv.append(cell)
        print("  ".join(row_cells))
        table_rows.append(row_csv)

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["METRIC", *basenames])
            writer.writerows(table_rows)
        print(f"\nSaved combined table to {args.save_csv}")


if __name__ == "__main__":
    main()
