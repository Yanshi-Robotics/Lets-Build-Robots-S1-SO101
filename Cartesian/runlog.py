"""One directory per run under `logs/`, so that a failed session can be read back stage by stage.

    logs/2026-09-10_213000_live/
        run.log      human-readable events: arguments, model + calibration used, limits,
                     every target change, every solve, buttons, exceptions
        ticks.jsonl  one JSON object per control tick with the input and output of
                     every stage (see README, "Reading the logs")
    logs/latest -> the newest run

`latest` is a symlink. Windows refuses to create one unless the account is an
administrator or developer mode is on, so there it becomes `latest.txt` holding the
directory name instead. The jq one-liners in the README read through the symlink and
therefore only work on Linux and macOS.

Old runs beyond KEEP_RUNS are deleted at start-up.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

KEEP_RUNS = 30       # enough history to compare a bad session with a good one
ROUND_DIGITS = 4     # 0.1 mm / 0.0001 rad is plenty; keeps ticks.jsonl small


def _round(value):
    if isinstance(value, float):
        return round(value, ROUND_DIGITS)
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return _round(value.tolist())
    return value


class RunLog:
    def __init__(self, root: Path, mode: str):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self._prune(root)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.dir = root / f"{stamp}_{mode}"
        self.dir.mkdir()
        latest, pointer = root / "latest", root / "latest.txt"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        try:
            latest.symlink_to(self.dir.name)
            pointer.unlink(missing_ok=True)   # in case an earlier run fell back to the file
        except OSError:
            # WinError 1314 without admin rights or developer mode. A text pointer keeps the
            # information reachable; `_prune` skips it because it is not a directory.
            pointer.write_text(self.dir.name + "\n", encoding="utf-8")

        self.log = logging.getLogger("cartesian")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        # encoding is explicit because event() uses ensure_ascii=False and the pairing wizard
        # logs emoji: on Windows the locale encoding (cp1252 / cp936) cannot hold them and
        # logging would print a UnicodeEncodeError traceback instead of the line.
        for handler in (logging.FileHandler(self.dir / "run.log", encoding="utf-8"), logging.StreamHandler()):
            handler.setFormatter(fmt)
            self.log.addHandler(handler)

        # newline="" keeps one \n per record on every platform; Windows would write \r\n
        self._ticks = open(self.dir / "ticks.jsonl", "w", buffering=1, encoding="utf-8", newline="")
        self._t0 = time.monotonic()
        self.event("run directory", path=str(self.dir))

    @staticmethod
    def _prune(root: Path) -> None:
        runs = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
        for old in runs[:-KEEP_RUNS] if len(runs) > KEEP_RUNS else []:
            for f in old.iterdir():
                f.unlink()
            old.rmdir()

    def event(self, message: str, **fields) -> None:
        if fields:
            message += " " + json.dumps(_round(fields), ensure_ascii=False)
        self.log.info(message)

    def warning(self, message: str, **fields) -> None:
        if fields:
            message += " " + json.dumps(_round(fields), ensure_ascii=False)
        self.log.warning(message)

    def exception(self, message: str) -> None:
        self.log.exception(message)

    def tick(self, record: dict) -> None:
        record = {"t": round(time.monotonic() - self._t0, 4), **_round(record)}
        self._ticks.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._ticks.close()
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)
