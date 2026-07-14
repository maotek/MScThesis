#!/usr/bin/env python3
"""Download the EventScape Town01-03 training archive with resume support."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests


URL = "http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town01-03_train.zip"
FILENAME = "Town01-03_train.zip"


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def remote_size(url: str, timeout: int) -> int | None:
    """Return the remote file size when the server provides one."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        return int(length) if length is not None else None
    except (requests.RequestException, ValueError):
        return None


def download(
    url: str,
    destination: Path,
    *,
    timeout: int,
    retries: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    total = remote_size(url, timeout)

    if destination.exists():
        if total is None or destination.stat().st_size == total:
            print(f"Already downloaded: {destination}")
            return
        raise RuntimeError(
            f"Existing file has the wrong size: {destination}\n"
            "Move or delete it before retrying."
        )

    existing = partial.stat().st_size if partial.exists() else 0
    if total is not None:
        remaining = max(total - existing, 0)
        free = shutil.disk_usage(destination.parent).free
        if free < remaining:
            raise RuntimeError(
                f"Not enough disk space in {destination.parent}: "
                f"need {human_bytes(remaining)}, have {human_bytes(free)} free."
            )
        print(f"Archive size: {human_bytes(total)}")
    if existing:
        print(f"Resuming from {human_bytes(existing)}")

    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(
                url, headers=headers, stream=True, allow_redirects=True, timeout=timeout
            ) as response:
                if existing and response.status_code == 200:
                    print("Server ignored the resume request; restarting the download.")
                    partial.unlink()
                    existing = 0
                elif existing and response.status_code != 206:
                    response.raise_for_status()
                    raise RuntimeError(
                        f"Server returned HTTP {response.status_code} for a resume request"
                    )
                response.raise_for_status()

                response_total = response.headers.get("Content-Range", "").partition("/")[2]
                if response_total.isdigit():
                    total = int(response_total)
                elif total is None and response.headers.get("Content-Length", "").isdigit():
                    total = existing + int(response.headers["Content-Length"])

                downloaded = existing
                started = time.monotonic()
                with partial.open("ab" if existing else "wb") as output:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded += len(chunk)
                        elapsed = max(time.monotonic() - started, 0.001)
                        speed = (downloaded - existing) / elapsed
                        progress = (
                            f"{downloaded / total:6.2%} " if total else ""
                        )
                        print(
                            f"\r{progress}{human_bytes(downloaded)} "
                            f"at {human_bytes(int(speed))}/s",
                            end="",
                            flush=True,
                        )
                print()

            if total is not None and partial.stat().st_size != total:
                raise RuntimeError(
                    f"Incomplete download: received {human_bytes(partial.stat().st_size)} "
                    f"of {human_bytes(total)}"
                )
            partial.replace(destination)
            print(f"Saved to: {destination}")
            return
        except (requests.RequestException, OSError, RuntimeError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts. Run the script again to "
                    f"resume from {partial}. Last error: {error}"
                ) from error
            delay = min(2 ** (attempt - 1), 30)
            print(f"\nAttempt {attempt} failed: {error}\nRetrying in {delay}s...")
            time.sleep(delay)


def extract(archive: Path, destination: Path) -> None:
    """Extract an archive safely, skipping files that are already complete."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        files = [member for member in members if not member.is_dir()]
        total = sum(member.file_size for member in files)
        completed = 0
        remaining = 0

        for member in members:
            target = destination / member.filename
            target_resolved = target.resolve()
            if not target_resolved.is_relative_to(destination_root):
                raise RuntimeError(f"Unsafe path in ZIP archive: {member.filename}")
            if member.is_dir():
                continue
            if target.is_file() and target.stat().st_size == member.file_size:
                completed += member.file_size
            else:
                remaining += member.file_size

        free = shutil.disk_usage(destination).free
        if free < remaining:
            raise RuntimeError(
                f"Not enough disk space to extract into {destination}: "
                f"need {human_bytes(remaining)}, have {human_bytes(free)} free."
            )

        print(f"Extracting {archive.name} into {destination}")
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == member.file_size:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with source.open(member) as input_file, partial.open("wb") as output_file:
                while chunk := input_file.read(8 * 1024 * 1024):
                    output_file.write(chunk)
                    completed += len(chunk)
                    progress = completed / total if total else 1.0
                    print(
                        f"\r{progress:6.2%} {human_bytes(completed)} / "
                        f"{human_bytes(total)}",
                        end="",
                        flush=True,
                    )
            partial.replace(target)
        print(f"\nExtracted to: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/EventScape"),
        help="output directory (default: datasets/EventScape)",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=10, help="retry attempts")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="download the ZIP without extracting it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.out.expanduser() / FILENAME
    try:
        download(
            URL,
            archive,
            timeout=args.timeout,
            retries=args.retries,
        )
        if not args.no_extract:
            extract(archive, args.out.expanduser())
    except (RuntimeError, requests.RequestException, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
