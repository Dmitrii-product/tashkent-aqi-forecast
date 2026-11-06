"""Writes that never leave a half-written file: write next to the target, then swap it in.

If the process dies or the disk fills mid-write, the previous version stays intact.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def write_bytes(path: Path, data: bytes) -> None:
    tmp = _tmp(path)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def to_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = _tmp(path)
    df.to_parquet(tmp)
    os.replace(tmp, path)
