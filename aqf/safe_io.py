"""Writes that never leave a half-written file: write next to the target, then swap it in.

If the process dies or the disk fills mid-write, the previous version stays intact. The temporary
name carries the process id: two runs writing the same file at once (2026-11-12, two catch-up
runs 3 s apart) must not move each other's half-written file away.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def _swap_in(tmp: Path, path: Path, write) -> None:
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # only left over if the write failed


def write_bytes(path: Path, data: bytes) -> None:
    _swap_in(_tmp(path), path, lambda t: t.write_bytes(data))


def write_text(path: Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def to_parquet(df: pd.DataFrame, path: Path) -> None:
    _swap_in(_tmp(path), path, df.to_parquet)
