import argparse
import csv
import os
from glob import glob
from typing import Dict, List, Optional


def is_threshold_metric(name: str) -> bool:
    return name.startswith("_10") or name.startswith("_20") or name.startswith("_30")


def is_useless_metric(name: str) -> bool:
    return name in {
        "_mean_target_depth",
        "_median_target_depth",
        "_mean_prediction_depth",
        "_median_prediction_depth",
    }


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
        if not name or is_threshold_metric(name) or is_useless_metric(name):
            continue
        val = row[mean_idx].strip() if len(row) > mean_idx else ""
        metrics[name] = val
    return metrics


def to_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def prefers_higher(metric_name: str) -> bool:
    return "threshold_delta" in metric_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Markdown comparison table from evaluate_*.csv files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory containing evaluate_*.csv files (default: output).",
    )
    parser.add_argument(
        "--save-md",
        type=str,
        default=os.path.join("output", "output.md"),
        help="Path to save the Markdown table (default: output/output.md).",
    )
    args = parser.parse_args()

    csv_paths = sorted(glob(os.path.join(args.output_dir, "evaluate_*.csv")))
    if not csv_paths:
        print(f"No evaluate_*.csv files found under {args.output_dir}")
        return

    metrics_per_file: List[Dict[str, str]] = []
    basenames: List[str] = []
    all_metric_names: set = set()

    for csv_path in csv_paths:
        metrics = read_metrics(csv_path)
        metrics_per_file.append(metrics)
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        basenames.append(stem.replace("evaluate_", "", 1))
        all_metric_names.update(metrics.keys())

    if not all_metric_names:
        print("No metrics found in CSVs.")
        return

    metric_names = sorted(all_metric_names)

    # Build display matrix with underlined best values per metric
    display_rows: List[List[str]] = []
    for m in metric_names:
        raw_vals = [metrics.get(m, "") for metrics in metrics_per_file]
        nums = [to_float(v) for v in raw_vals]
        best_val = None
        for n in nums:
            if n is None:
                continue
            if best_val is None:
                best_val = n
            else:
                if prefers_higher(m):
                    best_val = max(best_val, n)
                else:
                    best_val = min(best_val, n)
        disp_vals: List[str] = []
        for v, n in zip(raw_vals, nums):
            if best_val is not None and n is not None and n == best_val:
                disp_vals.append(f"<u>{v}</u>")
            else:
                disp_vals.append(v)
        display_rows.append([m, *disp_vals])

    header = ["METRIC", *basenames]
    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in display_rows:
        md_lines.append("| " + " | ".join(row) + " |")

    md_text = "\n".join(md_lines)
    print(md_text)

    if args.save_md:
        os.makedirs(os.path.dirname(args.save_md) or ".", exist_ok=True)
        with open(args.save_md, "w") as f:
            f.write(md_text + "\n")
        print(f"\nSaved Markdown table to {args.save_md}")


if __name__ == "__main__":
    main()
