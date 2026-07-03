"""Minimal CSV + stdout logger (no tensorboard dependency required)."""
from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, Optional


class Logger:
    def __init__(self, log_dir: str, exp_name: str):
        self.dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, "metrics.csv")
        self._fields: list[str] = []
        self._fh = None
        self._writer: Optional[csv.DictWriter] = None
        self._t0 = time.time()

    def log(self, step: int, metrics: Dict[str, Any], prefix: str = "") -> None:
        row = {"step": step, "wall_time": round(time.time() - self._t0, 2)}
        for k, v in metrics.items():
            row[f"{prefix}{k}" if prefix else k] = _to_float(v)

        # (Re)create the writer if new fields appear.
        new_fields = [k for k in row if k not in self._fields]
        if new_fields or self._writer is None:
            self._fields += new_fields
            self._reopen_writer()
        self._writer.writerow(row)  # type: ignore[union-attr]
        self._fh.flush()  # type: ignore[union-attr]

    def _reopen_writer(self) -> None:
        if self._fh is not None:
            self._fh.close()
        # Re-read existing rows so we don't lose history when headers change.
        rows = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Absorb any columns already on disk so rewriting old rows
                # (which may carry fields from a prior run) never fails.
                for k in reader.fieldnames or []:
                    if k not in self._fields:
                        self._fields.append(k)
        self._fh = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
        self._writer.writeheader()
        for r in rows:
            self._writer.writerow(r)

    @staticmethod
    def print(step: int, metrics: Dict[str, Any]) -> None:
        msg = "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
        print(f"[step {step:>8}] {msg}", flush=True)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def _to_float(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)
