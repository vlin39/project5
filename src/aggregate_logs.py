#!/usr/bin/env python3
"""
Aggregate one or more experiment-log directories into a comparison table.

Usage:
  python3 src/aggregate_logs.py [log_dir [log_dir ...]]

Each log file must be JSON-lines, one record per instance, as written by
runAll_sweep.sh. File names should match results_<solver>_<config>_<T>s.log.

Outputs:
  - per-instance best across all (solver, config, time) combinations
  - per-config average gap to per-instance best
  - top configs and worst configs
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Matches the standard sweep filenames like results_savings_balanced_30s.log.
FNAME_RE_FULL = re.compile(r"results_(?P<solver>[a-z_]+)_(?P<config>[a-z0-9_]+?)_(?P<time>\d+)s\.log$")
# Matches the bare per-method filenames like results_pyvrp.log.
FNAME_RE_BARE = re.compile(r"results_(?P<solver>[a-z_]+)\.log$")


def collect(dirs):
    rows = defaultdict(dict)  # rows[instance][(solver,config,time)] = (cost, wall)
    for d in dirs:
        for fp in sorted(Path(d).glob("*.log")):
            m = FNAME_RE_FULL.search(fp.name)
            if m:
                key = (m.group("solver"), m.group("config"), int(m.group("time")))
            else:
                m = FNAME_RE_BARE.search(fp.name)
                if not m:
                    continue
                key = (m.group("solver"), "default", 0)
            for line in fp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inst = rec.get("Instance")
                try:
                    cost = float(rec.get("Result"))
                except (TypeError, ValueError):
                    continue
                try:
                    wall = float(rec.get("Time", 0))
                except ValueError:
                    wall = 0.0
                rows[inst][key] = (cost, wall)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["focused_logs"])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = collect(args.dirs)
    if not rows:
        print("No log records found.")
        return

    # Per-instance best
    best_per_inst = {inst: min(v[0] for v in cfgs.values()) for inst, cfgs in rows.items()}
    print("Per-instance best (across all logged configs):")
    for inst in sorted(rows):
        cfgs = rows[inst]
        best_key = min(cfgs, key=lambda k: cfgs[k][0])
        best_cost = cfgs[best_key][0]
        s, c, t = best_key
        print(f"  {inst:<18}  {best_cost:>10.2f}  via {s}/{c} @ {t}s")

    # Per-config average gap to per-instance best
    gap_by_cfg = defaultdict(list)
    for inst, cfgs in rows.items():
        baseline = best_per_inst[inst]
        for k, (cost, _) in cfgs.items():
            gap_by_cfg[k].append((cost - baseline) / baseline * 100)
    avg_gap = {k: (sum(g) / len(g), len(g)) for k, g in gap_by_cfg.items()}

    print()
    print(f"Top {args.top} configs (lowest avg gap to per-instance best):")
    for (s, c, t), (g, n) in sorted(avg_gap.items(), key=lambda x: x[1][0])[: args.top]:
        print(f"  {s}/{c:<22} @ {t:>3}s  avg_gap={g:>6.2f}%  n={n}")

    print()
    print(f"Worst {args.top} configs:")
    for (s, c, t), (g, n) in sorted(avg_gap.items(), key=lambda x: x[1][0], reverse=True)[: args.top]:
        print(f"  {s}/{c:<22} @ {t:>3}s  avg_gap={g:>6.2f}%  n={n}")


if __name__ == "__main__":
    main()
