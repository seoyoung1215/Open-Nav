#!/usr/bin/env python3
"""Build OpenNav-style GT gzip from ANSR/VLN-CE episode json (reference_path -> locations)."""
import argparse
import gzip
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--episodes",
        required=True,
        help="VLN-CE episode .json.gz (with episodes[].reference_path, episodes[].episode_id)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output path, e.g. .../all_gt.json.gz",
    )
    args = p.parse_args()

    with gzip.open(args.episodes, "rt", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for ep in data["episodes"]:
        eid = str(ep["episode_id"])
        rp = ep.get("reference_path") or []
        out[eid] = {"locations": rp}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Wrote {len(out)} entries to {args.out}")


if __name__ == "__main__":
    main()
