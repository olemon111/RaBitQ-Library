#!/usr/bin/env python3
"""Merge .fbin meta information and raw data into a single .fbin file.

The script reads a text metadata file (e.g. base.fbin.meta) containing lines like:
    dim: 128
    nvecs: 985462
    uniq: 985462

and a raw binary file containing back-to-back float32 vectors without a header
(e.g. base.fbin.raw). It writes an output .fbin file with a standard header
(two uint32, order: nvecs then dim) followed by the raw data.
"""

import argparse
import os
import struct
from typing import Tuple

CHUNK_SIZE = 8 * 1024 * 1024  # bytes


def parse_meta(path: str) -> Tuple[int, int]:
    """Parse metadata file and return (dim, nvecs)."""
    dim = None
    nvecs = None
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            if key == "dim":
                dim = int(value)
            elif key == "nvecs":
                nvecs = int(value)
    if dim is None or nvecs is None:
        raise ValueError(f"Missing dim or nvecs in {path}")
    return dim, nvecs


def validate_raw_size(raw_path: str, dim: int, nvecs: int) -> None:
    """Ensure raw file size matches expected dimension and vector count."""
    raw_size = os.path.getsize(raw_path)
    if raw_size % 4 != 0:
        raise ValueError(
            f"Raw file size {raw_size} is not divisible by 4 (float32 alignment)"
        )
    expected = dim * 4 * nvecs
    if raw_size != expected:
        raise ValueError(
            f"Raw file size mismatch: expected {expected} bytes for dim={dim}, nvecs={nvecs}, got {raw_size}"
        )


def copy_raw_with_header(meta_path: str, raw_path: str, out_path: str) -> None:
    dim, nvecs = parse_meta(meta_path)
    validate_raw_size(raw_path, dim, nvecs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fout:
        # Write header as expected by common fbin loaders: nvecs first, then dim.
        fout.write(struct.pack("<II", nvecs, dim))
        with open(raw_path, "rb") as fin:
            while True:
                chunk = fin.read(CHUNK_SIZE)
                if not chunk:
                    break
                fout.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build .fbin file from meta and raw data.")
    parser.add_argument("--meta", required=True, help="Path to .fbin.meta file")
    parser.add_argument("--raw", required=True, help="Path to .fbin.raw file")
    parser.add_argument("--out", required=True, help="Path to output .fbin file")
    args = parser.parse_args()

    copy_raw_with_header(args.meta, args.raw, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
