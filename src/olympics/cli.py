"""Command line interface:  python -m olympics.cli <command> [options]"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .pipeline import STAGES, Pipeline


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=None, help="Workspace folder for ALL outputs (e.g. a Google Drive folder). Default: ./workspace")
    p.add_argument("--config", default=None, help="Path to params.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config value, e.g. ml.profile=fast")
    p.add_argument("--demo", action="store_true", help="Use built-in synthetic data (no Kaggle needed)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="olympics", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run the whole pipeline (resumes from checkpoints)")
    _common(r)
    r.add_argument("--stages", nargs="+", choices=STAGES, help=f"Subset of stages: {' '.join(STAGES)}")
    r.add_argument("--force", nargs="*", default=[], help="Stages to recompute, or 'all'")
    r.add_argument("--force-download", action="store_true")

    st = sub.add_parser("status", help="Show finished stages")
    _common(st)

    rt = sub.add_parser("retrain", help="Retrain using human feedback (champion/challenger)")
    _common(rt)
    rt.add_argument("--force", action="store_true", help="Retrain even if there are too few feedback rows")

    ins = sub.add_parser("inspect-source", help="List downloaded files/columns to confirm dataset mapping")
    _common(ins)

    q = sub.add_parser("sql", help="Run a named SQL query")
    _common(q)
    q.add_argument("name")
    q.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")

    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    pipe = Pipeline(cfg, a.root, a.demo)

    if a.cmd == "run":
        out = pipe.run(a.stages, a.force, a.force_download)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "status":
        print(json.dumps(pipe.ck.status(), indent=2))
    elif a.cmd == "retrain":
        from .ml.retrain import retrain_with_feedback
        from .checkpoint import fingerprint
        ml = pipe.ml_frame()
        print(json.dumps(retrain_with_feedback(cfg, pipe.paths, ml, pipe.tracker, fingerprint(len(ml), cfg["data"]["source"]), a.force), indent=2, default=str))
    elif a.cmd == "inspect-source":
        from .data.ingest import download_kaggle, inspect_source
        src = cfg["data"]["source"]
        d = download_kaggle(cfg["data"]["sources"][src]["kaggle_slug"], pipe.paths.raw / src)
        print(json.dumps(inspect_source(d), indent=2))
    elif a.cmd == "sql":
        from .sql.layer import local_copy, run_query
        params = dict(kv.split("=", 1) for kv in a.param)
        params = {k: (None if v.lower() == "none" else int(v) if v.lstrip("-").isdigit() else v) for k, v in params.items()}
        print(run_query(local_copy(pipe.paths.db_file), a.name, params).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
