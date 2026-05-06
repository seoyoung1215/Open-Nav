#!/usr/bin/env python3
"""Build OpenNav-style GT gzip from ANSR/VLN-CE episode json (reference_path -> locations)."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict


def load_episodes(path: Path) -> Dict[str, Any]:
    """Load ANSR/VLN-CE episodes from ``.json`` or ``.json.gz``."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")
    if path.suffix.lower() == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, encoding="utf-8")
    with opener as f:
        return json.load(f)


def output_path_as_json_gz(raw: str) -> Path:
    """Output is always gzip-compressed JSON; normalize path to ``*.json.gz``."""
    p = Path(raw).expanduser()
    s = str(p).lower()
    if s.endswith(".json.gz"):
        return p
    if p.suffix.lower() == ".json":
        return p.with_suffix(".json.gz")
    # e.g. out.gz → out.json.gz
    if p.suffix.lower() == ".gz":
        return p.parent / f"{p.stem}.json.gz"
    return p.with_suffix(".json.gz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build OpenNav GT map episode_id → {{locations}} from ANSR episodes. "
            "Input may be plain .json or .json.gz; output is always written as .json.gz."
        )
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="VLN-CE/ANSR episode JSON or JSON.gz "
        "(episodes[].reference_path, episodes[].episode_id)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output path (gzip JSON). "
            "If you pass name.json → writes name.json.gz; bare name → name.json.gz"
        ),
    )
    args = parser.parse_args()

    data = load_episodes(args.episodes)
    out_path = output_path_as_json_gz(args.out)

    out: Dict[str, Dict[str, Any]] = {}
    for ep in data["episodes"]:
        eid = str(ep["episode_id"])
        rp = ep.get("reference_path") or []
        out[eid] = {"locations": rp}

    out_path_parent = out_path.parent
    if str(out_path_parent):
        os.makedirs(out_path_parent, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Wrote {len(out)} entries to {out_path}")


if __name__ == "__main__":
    main()
