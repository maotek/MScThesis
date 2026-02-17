#!/usr/bin/env python3
"""
Download and unpack MVSEC ROS bags.

Download layout:
  <out>/<split>/<sequence>/<sequence>_data.bag
  <out>/<split>/<sequence>/<sequence>_gt.bag

Unpack layout (MVSECSequence-compatible, local hdf5 path):
  <out>/<split>/<sequence>/depth/data/depth_XXXXXXXXXX.npy
  <out>/<split>/<sequence>/depth/data/timestamps.txt
  <out>/<split>/<sequence>/rgb/{davis|davis_left_sync}/frame_XXXXXXXXXX.png
  <out>/<split>/<sequence>/rgb/{davis|davis_left_sync}/timestamps.txt
  <out>/<split>/<sequence>/hdf5/data.hdf5        # davis/left/events [x,y,t,p]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import cv2
import h5py
import numpy as np
import requests
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

TRAIN_URLS = [
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day2_data.bag",
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day2_gt.bag",
]

TEST_URLS = [
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night1_data.bag",
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night1_gt.bag",
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night2_data.bag",
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night2_gt.bag",
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night3_data.bag",
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night3_gt.bag",
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day1_data.bag",
    "https://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day1_gt.bag",
]

LEFT_EVENTS_TOPIC = "/davis/left/events"
LEFT_IMAGE_TOPIC = "/davis/left/image_raw"
LEFT_DEPTH_RAW_TOPIC = "/davis/left/depth_image_raw"
LEFT_DEPTH_RECT_TOPIC = "/davis/left/depth_image_rect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and unpack MVSEC ROS bags")
    parser.add_argument(
        "--out",
        type=str,
        default="datasets/MVSEC/data",
        help="Output root",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Which split(s) to process",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["download", "unpack", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Optional sequence filter (can repeat), e.g. outdoor_day2",
    )
    parser.add_argument("--force", action="store_true", help="Redownload existing bag files")
    parser.add_argument("--force-unpack", action="store_true", help="Overwrite unpacked outputs")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    parser.add_argument(
        "--event-chunk-size",
        type=int,
        default=200000,
        help="Buffered events before appending to HDF5",
    )
    return parser.parse_args()


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for unit in units:
        if f < 1024 or unit == units[-1]:
            return f"{f:.2f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024.0
    return f"{n} B"


def get_filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name


def get_sequence_name_from_filename(filename: str) -> str:
    name = filename[:-4] if filename.endswith(".bag") else filename
    if name.endswith("_data"):
        return name[:-5]
    if name.endswith("_gt"):
        return name[:-3]
    return name


def build_download_map() -> Dict[str, List[str]]:
    return {"train": TRAIN_URLS, "test": TEST_URLS}


def maybe_filter_urls(urls: Iterable[str], sequence_filter: List[str] | None) -> List[str]:
    if not sequence_filter:
        return list(urls)
    wanted = set(sequence_filter)
    return [
        url
        for url in urls
        if get_sequence_name_from_filename(get_filename_from_url(url)) in wanted
    ]


def download_with_resume(url: str, dst: Path, timeout: int = 60) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")

    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    with requests.get(url, stream=True, headers=headers, timeout=timeout) as response:
        if response.status_code == 404:
            raise FileNotFoundError(f"404 Not Found: {url}")

        if existing > 0 and response.status_code == 200:
            part.unlink(missing_ok=True)
            existing = 0

        response.raise_for_status()

        total = None
        if "Content-Range" in response.headers:
            try:
                total = int(response.headers["Content-Range"].split("/")[-1])
            except Exception:
                total = None
        elif "Content-Length" in response.headers:
            try:
                total = existing + int(response.headers["Content-Length"])
            except Exception:
                total = None

        mode = "ab" if existing > 0 else "wb"
        downloaded = existing
        t0 = time.time()

        with open(part, mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)

                if total:
                    dt = max(time.time() - t0, 1e-6)
                    speed = downloaded / dt
                    pct = (downloaded / total) * 100
                    sys.stdout.write(
                        f"\r    {dst.name}: {pct:6.2f}% "
                        f"({human_bytes(downloaded)}/{human_bytes(total)}) "
                        f"@ {human_bytes(int(speed))}/s"
                    )
                    sys.stdout.flush()

        if total:
            sys.stdout.write("\n")

    part.replace(dst)


def ros_time_to_sec(stamp) -> float:
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if hasattr(stamp, "sec") and hasattr(stamp, "nsec"):
        return float(stamp.sec) + float(stamp.nsec) * 1e-9
    secs = int(getattr(stamp, "secs", 0))
    nsecs = int(getattr(stamp, "nsecs", 0))
    return float(secs) + float(nsecs) * 1e-9


def write_timestamps(path: Path, timestamps: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for idx, t_sec in enumerate(timestamps):
            handle.write(f"{idx} {t_sec:.9f}\n")


def append_events(h5_dataset, event_rows: Sequence[Tuple[float, float, float, float]]) -> None:
    if not event_rows:
        return
    block = np.asarray(event_rows, dtype=np.float64)
    old = int(h5_dataset.shape[0])
    new = old + int(block.shape[0])
    h5_dataset.resize((new, 4))
    h5_dataset[old:new, :] = block


def decode_image(msg) -> np.ndarray:
    encoding = str(getattr(msg, "encoding", ""))
    height = int(msg.height)
    width = int(msg.width)

    if isinstance(msg.data, (bytes, bytearray, memoryview)):
        raw = bytes(msg.data)
    else:
        raw = np.asarray(msg.data, dtype=np.uint8).tobytes()

    if encoding in {"mono8", "8UC1"}:
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
    if encoding == "bgr8":
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    if encoding == "rgb8":
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    raise ValueError(f"Unsupported RGB image encoding: {encoding}")


def decode_depth(msg) -> np.ndarray:
    encoding = str(getattr(msg, "encoding", ""))
    if encoding != "32FC1":
        raise ValueError(f"Unsupported depth encoding: {encoding} (expected 32FC1)")

    height = int(msg.height)
    width = int(msg.width)
    is_bigendian = int(getattr(msg, "is_bigendian", 0))

    if isinstance(msg.data, (bytes, bytearray, memoryview)):
        raw = bytes(msg.data)
    else:
        raw = np.asarray(msg.data, dtype=np.uint8).tobytes()

    dtype = np.dtype(">f4") if is_bigendian else np.dtype("<f4")
    depth = np.frombuffer(raw, dtype=dtype).reshape(height, width)
    depth = depth.astype(np.float32, copy=False)
    return depth


def build_typestore(reader: Reader):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    custom_types = {}
    for connection in reader.connections:
        msgdef = getattr(connection, "msgdef", None)
        msgtype = getattr(connection, "msgtype", None)
        if msgdef and msgtype:
            custom_types.update(get_types_from_msg(msgdef, msgtype))
    if custom_types:
        typestore.register(custom_types)
    return typestore


def unpack_sequence(sequence_dir: Path, split: str, event_chunk_size: int, force_unpack: bool) -> None:
    data_bag = sequence_dir / f"{sequence_dir.name}_data.bag"
    gt_bag = sequence_dir / f"{sequence_dir.name}_gt.bag"

    if not data_bag.exists():
        print(f"    Missing data bag: {data_bag.name}, skipping")
        return
    if not gt_bag.exists():
        print(f"    Missing gt bag: {gt_bag.name}, skipping")
        return

    rgb_folder = "davis_left_sync" if split == "test" else "davis"
    depth_dir = sequence_dir / "depth" / "data"
    rgb_dir = sequence_dir / "rgb" / rgb_folder
    hdf5_path = sequence_dir / "hdf5" / "data.hdf5"

    if hdf5_path.exists() and depth_dir.exists() and rgb_dir.exists() and not force_unpack:
        print("    Unpacked outputs exist, skipping (use --force-unpack to overwrite)")
        return

    depth_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_timestamps: List[float] = []
    depth_timestamps: List[float] = []
    event_buffer: List[Tuple[float, float, float, float]] = []

    rgb_count = 0
    depth_count = 0
    total_events = 0

    if force_unpack and hdf5_path.exists():
        hdf5_path.unlink()

    with h5py.File(str(hdf5_path), "w") as h5f:
        davis_group = h5f.require_group("davis")
        left_group = davis_group.require_group("left")
        events_ds = left_group.create_dataset(
            "events",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.float64,
            chunks=(max(event_chunk_size, 1), 4),
            compression="gzip",
            compression_opts=4,
        )

        with Reader(str(data_bag)) as reader:
            typestore = build_typestore(reader)
            selected = [
                connection
                for connection in reader.connections
                if connection.topic in {LEFT_EVENTS_TOPIC, LEFT_IMAGE_TOPIC}
            ]

            for connection, ros_t, rawdata in reader.messages(connections=selected):
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

                if connection.topic == LEFT_EVENTS_TOPIC:
                    for ev in getattr(msg, "events", []):
                        event_buffer.append(
                            (
                                float(ev.x),
                                float(ev.y),
                                ros_time_to_sec(ev.ts),
                                1.0 if bool(ev.polarity) else 0.0,
                            )
                        )
                    if len(event_buffer) >= event_chunk_size:
                        append_events(events_ds, event_buffer)
                        total_events += len(event_buffer)
                        event_buffer = []

                elif connection.topic == LEFT_IMAGE_TOPIC:
                    t_sec = ros_time_to_sec(msg.header.stamp) if hasattr(msg, "header") else float(ros_t) * 1e-9
                    image = decode_image(msg)
                    out_png = rgb_dir / f"frame_{rgb_count:010d}.png"
                    if not cv2.imwrite(str(out_png), image):
                        raise RuntimeError(f"Failed to write image: {out_png}")
                    rgb_timestamps.append(t_sec)
                    rgb_count += 1

        if event_buffer:
            append_events(events_ds, event_buffer)
            total_events += len(event_buffer)

    with Reader(str(gt_bag)) as reader:
        typestore = build_typestore(reader)

        depth_connections = [
            connection for connection in reader.connections if connection.topic == LEFT_DEPTH_RAW_TOPIC
        ]
        if not depth_connections:
            depth_connections = [
                connection for connection in reader.connections if connection.topic == LEFT_DEPTH_RECT_TOPIC
            ]

        for connection, ros_t, rawdata in reader.messages(connections=depth_connections):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            t_sec = ros_time_to_sec(msg.header.stamp) if hasattr(msg, "header") else float(ros_t) * 1e-9
            depth = decode_depth(msg)
            out_npy = depth_dir / f"depth_{depth_count:010d}.npy"
            np.save(str(out_npy), depth)
            depth_timestamps.append(t_sec)
            depth_count += 1

    write_timestamps(rgb_dir / "timestamps.txt", rgb_timestamps)
    write_timestamps(depth_dir / "timestamps.txt", depth_timestamps)

    print(
        f"    Unpacked: rgb={rgb_count}, depth={depth_count}, events={total_events}, hdf5={hdf5_path.name}"
    )


def sequence_dirs_for_split(root: Path, split: str, sequence_filter: List[str] | None) -> List[Path]:
    split_dir = root / split
    if not split_dir.exists():
        return []

    dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name)

    if not sequence_filter:
        return dirs

    wanted = set(sequence_filter)
    return [p for p in dirs if p.name in wanted]


def run_download(args: argparse.Namespace, out_root: Path, splits: List[str]) -> None:
    split_to_urls = build_download_map()

    print("Mode: download\n")
    for split in splits:
        urls = maybe_filter_urls(split_to_urls[split], args.sequence)
        print(f"Split: {split} ({len(urls)} files)")

        for idx, url in enumerate(urls, 1):
            filename = get_filename_from_url(url)
            sequence_name = get_sequence_name_from_filename(filename)
            destination = out_root / split / sequence_name / filename

            print(f"  [{idx}/{len(urls)}] {sequence_name}/{filename}")

            if destination.exists() and destination.stat().st_size > 0 and not args.force:
                print("    Skipping (already exists)")
                continue

            try:
                download_with_resume(url, destination, timeout=args.timeout)
            except Exception as exc:
                print(f"    FAILED: {exc}")


def run_unpack(args: argparse.Namespace, out_root: Path, splits: List[str]) -> None:
    print("Mode: unpack existing bags\n")

    for split in splits:
        dirs = sequence_dirs_for_split(out_root, split, args.sequence)
        print(f"Split: {split} ({len(dirs)} sequence folders)")

        for idx, sequence_dir in enumerate(dirs, 1):
            print(f"  [{idx}/{len(dirs)}] {sequence_dir.name}")
            try:
                unpack_sequence(
                    sequence_dir=sequence_dir,
                    split=split,
                    event_chunk_size=args.event_chunk_size,
                    force_unpack=args.force_unpack,
                )
            except Exception as exc:
                print(f"    FAILED unpack: {exc}")


def main() -> int:
    args = parse_args()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [args.split] if args.split != "all" else ["train", "test"]

    print(f"Output root: {out_root}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Stage: {args.stage}\n")

    if args.stage in {"download", "all"}:
        run_download(args, out_root, splits)

    if args.stage in {"unpack", "all"}:
        run_unpack(args, out_root, splits)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
