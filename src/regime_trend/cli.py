"""Command-line entry point.

Examples
--------
    regime-trend run configs/multi_asset.yaml
    regime-trend run configs/*.yaml --refresh-data --workers 8
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

from .config import load_config
from .experiments import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regime-trend", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="run the full research pipeline for one or more configs")
    p.add_argument("configs", nargs="+", help="YAML config path(s); globs allowed")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--refresh-data", action="store_true")
    p.add_argument("--workers", type=int, default=None, help="processes for HMM fitting")
    p.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("yfinance", "matplotlib", "PIL", "peewee", "hmmlearn"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    paths: list[str] = []
    for pattern in args.configs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    for path in paths:
        cfg = load_config(Path(path))
        s = run(cfg, args.data_dir, args.results_dir, args.refresh_data, args.workers)
        print(f"{cfg.name}: {s['evaluation_period'][0]} -> {s['evaluation_period'][1]}, "
              f"{s['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
