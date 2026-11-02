"""Stage- and item-level checkpointing so a Colab runtime disconnect never loses finished work.

* Stage markers: ``checkpoints/<stage>.json`` holds the input fingerprint + required outputs. A stage is
  skipped on re-run only if the fingerprint matches AND every required output still exists.
* Item cache: ``cached_json`` stores the result of one expensive unit (e.g. one model's metrics) so a
  restart resumes at the model that was running when the runtime died.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from .io_utils import get_logger, read_json, write_json

log = get_logger()


def fingerprint(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class Checkpointer:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _marker(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def status(self) -> dict:
        out = {}
        for p in sorted(self.dir.glob("*.json")):
            data = read_json(p, {}) or {}
            out[p.stem] = {"finished_at": data.get("finished_at"), "seconds": data.get("seconds")}
        return out

    def is_done(self, stage: str, fp: str, outputs: list[Path] | None = None) -> bool:
        marker = read_json(self._marker(stage))
        if not marker or marker.get("fingerprint") != fp:
            return False
        return all(Path(o).exists() for o in (outputs or marker.get("outputs", [])))

    def run(self, stage: str, fn: Callable[[], dict | None], fp: str, outputs: list[Path] | None = None,
            force: bool = False) -> dict:
        outputs = [Path(o) for o in (outputs or [])]
        if not force and self.is_done(stage, fp, outputs):
            log.info("[checkpoint] %-12s already complete - skipping", stage)
            return (read_json(self._marker(stage), {}) or {}).get("summary", {})
        log.info("[stage] %-12s starting", stage)
        start = time.time()
        summary = fn() or {}
        seconds = round(time.time() - start, 1)
        write_json(self._marker(stage), {
            "stage": stage, "fingerprint": fp, "outputs": [str(o) for o in outputs],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": seconds, "summary": summary,
        })
        log.info("[stage] %-12s finished in %.1fs", stage, seconds)
        return summary

    def invalidate(self, stage: str | None = None) -> None:
        targets = [self._marker(stage)] if stage else list(self.dir.glob("*.json"))
        for t in targets:
            t.unlink(missing_ok=True)


def cached_json(path: Path, fn: Callable[[], dict], force: bool = False) -> dict:
    """Run ``fn`` once and persist its JSON-able result; later calls (even after a crash) reuse it."""
    path = Path(path)
    if not force:
        cached = read_json(path)
        if cached is not None:
            return cached
    result = fn()
    write_json(path, result)
    return read_json(path)
