#!/usr/bin/env python3
"""
Sequential DSEC downloader (train/validation splits)

Per sequence NAME, downloads into:
  NAME/
    disparity_timestamps.txt
    image_timestamps.txt
    NAME_events_left.zip
    NAME_calibration.zip
    NAME_images_rectified_left.zip
    NAME_disparity_event.zip
    NAME_disparity_image.zip

Then extracts each .zip into:
  NAME/NAME_<zip_stem>/

Downloads ONE SEQUENCE AT A TIME.
Strips trailing newline(s) from timestamps files after download.

Usage:
    python download_dsec.py --split all --out datasets/DSEC/data
    python download_dsec.py --split train --out datasets/DSEC/data --seq zurich_city_01_b
    python download_dsec.py --split validation --out datasets/DSEC/data --no-extract
    python download_dsec.py --split train --out datasets/DSEC/data --keep-zips
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


BASE_URL = "https://download.ifi.uzh.ch/rpg/DSEC/"

FILES = [
    "disparity_timestamps.txt",
    "image_timestamps.txt",
    "events_left.zip",
    "calibration.zip",
    "images_rectified_left.zip",
    "disparity_event.zip",
    "disparity_image.zip",
]

DSEC_TRAIN = {
    "interlaken_00_c":269,  "interlaken_00_d":996,   "interlaken_00_e":996,  "zurich_city_00_a":470,
    "zurich_city_00_b":732, "zurich_city_01_a":341,  "zurich_city_01_b":663, "zurich_city_01_c":489,
    "zurich_city_01_d":398, "zurich_city_01_e":996,  "zurich_city_01_f":787, "zurich_city_02_a":118,
    "zurich_city_02_b":613, "zurich_city_02_c":1442, "zurich_city_02_d":922, "zurich_city_02_e":923,
    "zurich_city_03_a":442, "zurich_city_04_a":351,  "zurich_city_04_b":135, "zurich_city_04_c":591,
    "zurich_city_04_d":479, "zurich_city_04_e":135,  "zurich_city_04_f":430, "zurich_city_09_a":907,
    "zurich_city_09_b":184, "zurich_city_09_c":662,  "zurich_city_09_e":409, "zurich_city_10_a":1158,
    "zurich_city_11_a":233, "zurich_city_11_b":967,  "zurich_city_11_c":979,
}

DSEC_VALIDATION = {
    "interlaken_00_f":746,  "interlaken_00_g":668,   "thun_00_a":120,        "zurich_city_05_a":877,
    "zurich_city_05_b":815, "zurich_city_06_a":762,  "zurich_city_07_a":732, "zurich_city_08_a":394,
    "zurich_city_09_d":850, "zurich_city_10_b":1203,
}


SCENES = {
    # "interlaken_00_c": 269,
    # "interlaken_00_d": 996,
    # "interlaken_00_e": 996,
    # "zurich_city_00_a": 470,
    # "zurich_city_00_b": 732,
    # "zurich_city_01_a": 341,
    # "zurich_city_01_b": 663,
    # "zurich_city_01_c": 489,
    # "zurich_city_01_d": 398,
    # "zurich_city_01_e": 996,
    # "zurich_city_01_f": 787,
    # "zurich_city_02_a": 118,
    # "zurich_city_02_b": 613,
    # "zurich_city_02_c": 1442,
    # "zurich_city_02_d": 922,
    # "zurich_city_02_e": 923,
    # "zurich_city_03_a": 442,
    # "zurich_city_04_a": 351,
    # "zurich_city_04_b": 135,
    # "zurich_city_04_c": 591,
    # "zurich_city_04_d": 479,
    # "zurich_city_04_e": 135,
    # "zurich_city_04_f": 430,
    # "zurich_city_09_a": 907,
    # "zurich_city_09_b": 184,
    # "zurich_city_09_c": 662,
    # "zurich_city_09_e": 409,
    # "zurich_city_10_a": 1158,
    # "zurich_city_11_a": 233,
    # "zurich_city_11_b": 967,
    # "zurich_city_11_c": 979,
    # "interlaken_00_f": 746,
    # "interlaken_00_g": 668,
    # "thun_00_a": 120,
    # "zurich_city_05_a": 877,
    # "zurich_city_05_b": 815,
    # "zurich_city_06_a": 762,
    # "zurich_city_07_a": 732,
    # "zurich_city_08_a": 394,
    # "zurich_city_09_d": 850,
    # "zurich_city_10_b": 1203,
}


@dataclass(frozen=True)
class Job:
    split: str
    seq: str
    filename: str

    @property
    def remote_url(self) -> str:
        return f"{BASE_URL}train/{self.seq}/{self.seq}_{self.filename}"

    def local_path(self, out_root: Path) -> Path:
        seq_dir = out_root / self.split / self.seq
        seq_dir.mkdir(parents=True, exist_ok=True)

        # timestamps should NOT have sequence prefix locally
        if self.filename in ("disparity_timestamps.txt", "image_timestamps.txt"):
            return seq_dir / self.filename

        return seq_dir / f"{self.seq}_{self.filename}"


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.2f} {u}" if u != "B" else f"{int(f)} B"
        f /= 1024.0
    return f"{n} B"


def download_with_resume(
    url: str,
    dst: Path,
    *,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
    retries: int = 5,
    backoff: float = 1.5,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")

    existing = part.stat().st_size if part.exists() else 0
    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    attempt = 0
    while True:
        attempt += 1
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(f"404 Not Found: {url}")

                # Server ignored resume
                if existing > 0 and r.status_code == 200:
                    part.unlink(missing_ok=True)
                    existing = 0
                    headers.pop("Range", None)

                r.raise_for_status()

                total = None
                if "Content-Range" in r.headers:
                    try:
                        total = int(r.headers["Content-Range"].split("/")[-1])
                    except Exception:
                        pass
                elif "Content-Length" in r.headers:
                    try:
                        total = existing + int(r.headers["Content-Length"])
                    except Exception:
                        pass

                mode = "ab" if existing > 0 else "wb"
                downloaded = existing
                t0 = time.time()

                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            pct = (downloaded / total) * 100
                            dt = max(time.time() - t0, 1e-6)
                            speed = downloaded / dt
                            sys.stdout.write(
                                f"\r    {dst.name}: {pct:6.2f}% "
                                f"({human_bytes(downloaded)}/{human_bytes(total)}) "
                                f"@ {human_bytes(int(speed))}/s"
                            )
                            sys.stdout.flush()

                if total:
                    sys.stdout.write("\n")

            part.replace(dst)
            return

        except (requests.RequestException, OSError) as e:
            if attempt >= retries:
                raise RuntimeError(f"Failed after {retries} attempts: {url}\nLast error: {e}") from e
            sleep_s = backoff ** (attempt - 1)
            print(f"\n  !! Error: {e} | retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if any(extract_dir.iterdir()):
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def strip_trailing_newlines(path: Path) -> None:
    """
    Remove ONLY trailing newline characters from a file.
    """
    if not path.exists():
        return
    data = path.read_bytes()
    stripped = data.rstrip(b"\n\r")
    if stripped != data:
        path.write_bytes(stripped)


def make_jobs(split: str, seqs: Iterable[str]) -> list[Job]:
    return [Job(split=split, seq=s, filename=f) for s in seqs for f in FILES]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default="datasets/DSEC/data",
        help="Root output directory. Data will be placed in <out>/<split>/<sequence>/.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "all"],
        help="Which split(s) to download.",
    )
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--seq", action="append", default=None)
    args = ap.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": DSEC_TRAIN,
        "validation": DSEC_VALIDATION,
    }
    splits = [args.split] if args.split != "all" else ["train", "validation"]

    print(f"Output root: {out_root}")
    print(f"Splits: {', '.join(splits)}\n")

    for split in splits:
        seqs = list(split_map[split].keys()) if args.seq is None else args.seq
        print(f"Split: {split}")
        print(f"Sequences: {len(seqs)}\n")

        for i, seq in enumerate(seqs, 1):
            print(f"=== [{i}/{len(seqs)}] {split} sequence: {seq} ===")
            jobs = [Job(split=split, seq=seq, filename=f) for f in FILES]

            for job in jobs:
                url = job.remote_url
                dst = job.local_path(out_root)

                if dst.exists() and dst.stat().st_size > 0:
                    print(f"  Skipping {dst.name} (exists)")
                    continue

                print(f"  Downloading {dst.name}")
                try:
                    download_with_resume(url, dst)

                    # Ensure timestamps files have NO trailing newline
                    if dst.name in ("disparity_timestamps.txt", "image_timestamps.txt"):
                        strip_trailing_newlines(dst)

                except Exception as e:
                    print(f"  FAILED {dst.name}: {e}")
                    print("  Continuing to next file...\n")
                    continue

            if not args.no_extract:
                print("  Extracting zips...")
                seq_dir = out_root / split / seq
                for f in FILES:
                    if not f.endswith(".zip"):
                        continue
                    zip_path = seq_dir / f"{seq}_{f}"
                    if not zip_path.exists():
                        continue

                    stem = f[:-4]
                    extract_dir = seq_dir / f"{seq}_{stem}"
                    print(f"    {zip_path.name} -> {extract_dir.name}")
                    extract_zip(zip_path, extract_dir)

                    if not args.keep_zips:
                        zip_path.unlink(missing_ok=True)

            print(f"=== Finished {split} {seq} ===\n")

    print("All sequences done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
