#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

EPS = 0.1

"""
Verifies feasibility and objective consistency. It does NOT prove optimality

Compatible usages:
  $ python3 src/verify.py 5_4_10.vrp 5_4_10.vrp.sol
  $ python3 src/verify.py input/386/47_1.vrp results.log
  $ python3 src/verify.py 5_4_10.vrp '{"Instance": "5_4_10.vrp", "Time": 1.23, "Result": 80.6, "Solution": "0 1 2 3 0 0 4 0 0 0 0 0"}'
  $ python3 src/verify.py input/ solutions/
  $ python3 src/verify.py input/ results.log

Checks:
  - correct number of vehicle routes
  - every route starts/ends at depot 0
  - every customer 1..n-1 appears exactly once
  - no invalid node IDs
  - capacity constraints
  - recomputed Euclidean objective matches reported objective within --eps

# good luck to anyone trying to review this
# I made this before going to sleep and now I'm confused, too.
# Hopefully that will change as I go through and add comments.
# 
"""

class VRPData:
    def __init__(self, path):
        self.path = Path(path)
        tokens = self.path.read_text().split()
        it = iter(tokens)

        self.n = int(next(it))
        self.vehicles = int(next(it))
        self.capacity = int(next(it))

        self.demand = []
        self.x = []
        self.y = []

        for _ in range(self.n):
            self.demand.append(int(float(next(it))))
            self.x.append(float(next(it)))
            self.y.append(float(next(it)))


def distance(data, i, j):
    return math.hypot(data.x[i] - data.x[j], data.y[i] - data.y[j])


def parse_flattened_solution(solution, vehicles):
    """
    "0 0 1 2 0 0 3 4 0 0 0 0 0" -> (flag, [[0,1,2,0], [0,3,4,0], [0,0], [0,0]])

    First token is the optimality flag (0 = not proved, 1 = proved); the
    rest is the flattened routes split on consecutive depot-zero pairs.
    """
    if solution is None or solution == "--":
        raise ValueError("No solution string provided")

    tokens = [int(x) for x in str(solution).split()]
    if not tokens:
        raise ValueError("Empty solution string")
    flag = tokens[0]
    tokens = tokens[1:]

    routes = []
    current = []

    for node in tokens:
        current.append(node)

        if node == 0 and len(current) > 1:
            routes.append(current)
            current = []

            if len(routes) == vehicles:
                break

    if len(routes) < vehicles:
        # If the line is short, pad with empty routes rather than crashing.
        # The structural checker will still catch missing customers/capacity issues.
        # ...I think.
        while len(routes) < vehicles:
            routes.append([0, 0])

    return flag, routes[:vehicles]


def parse_sol(text):
    """
    Parse .sol format:
        <objective> <flag>
        <route 0>
        <route 1>
        ...
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty .sol file")

    first = lines[0].split()
    try:
        reported_obj = float(first[0])
    except Exception as exc:
        raise ValueError("First line of .sol must begin with objective value") from exc

    flag = int(first[1]) if len(first) > 1 else 0
    routes = [[int(x) for x in ln.split()] for ln in lines[1:]]
    return routes, reported_obj, flag, None


def parse_json_record(record, vehicles):
    if record.get("Result") in (None, "--") or record.get("Solution") in (None, "--"):
        raise ValueError("JSON record does not contain a usable Result/Solution")

    flag, routes = parse_flattened_solution(record["Solution"], vehicles)
    reported_obj = float(record["Result"])
    return routes, reported_obj, flag, record


def read_json_records_from_file(path):
    records = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no} of {path} is not valid JSON") from exc
    return records


def pick_record_for_instance(records, instance_name):
    matches = [r for r in records if r.get("Instance") == instance_name]
    if not matches:
        available = [r.get("Instance", "?") for r in records[:10]]
        raise ValueError(
            f"No JSON record for instance {instance_name}. "
            f"First available instances: {available}"
        )
    return matches[-1]


def parse_solution_arg(solution_arg, data):
    """
    Accepts:
      - path to .sol file
      - path to JSON log/results file
      - raw JSON string
      - raw .sol text
    """
    arg = str(solution_arg)
    path = Path(arg)

    # check filesystem paths before interpreting the arg as raw text
    if path.exists():
        text = path.read_text()

        # A .sol file usually has suffix .sol, but also detect by first nonempty line.
        if path.suffix == ".sol": return parse_sol(text)

        # otherwise first try results.log
        try:
            records = read_json_records_from_file(path)
            record = pick_record_for_instance(records, data.path.name)
            return parse_json_record(record, data.vehicles)
        except Exception as exc:
            raise ValueError("...check what you're trying to verify?") from exc
    
    # if it looks like a path, check that the folder and file actually exist
    # this is to avoid confusing error messages
    looks_like_path = (
        path.suffix in {".sol", ".log", ".jsonl"}
        or "/" in arg
        or "\\" in arg
    )

    if looks_like_path: parent = path.parent

    # if the folder doesn't exist...
    if parent != Path(".") and not parent.exists():
        raise FileNotFoundError(f"We can't find a folder named {parent}. Try again?")

    # If the folder exists but the file isn't there...
    if parent.exists() and not path.exists():
        raise FileNotFoundError(f"We can't find a file named {path.name} in the folder {parent}. Try again?")

    # raw JSON string
    stripped = arg.strip()
    if stripped.startswith("{"):
        record = json.loads(stripped)
        return parse_json_record(record, data.vehicles)

    # raw text? idk. I probably should find a better fallback. 
    return parse_sol(arg)


def check_solution(data, routes, reported_obj=None, eps=EPS, verbose=False):
    errors = []

    if len(routes) != data.vehicles:
        errors.append(f"Expected {data.vehicles} routes, got {len(routes)}")

    seen = []
    recomputed = 0.0

    for ri, route in enumerate(routes):
        if len(route) < 2:
            errors.append(f"Route {ri} has fewer than two nodes")
            continue

        if route[0] != 0 or route[-1] != 0:
            errors.append(f"Route {ri} does not start/end at depot 0: {route}")

        load = 0
        cost = 0.0

        for node in route:
            if node < 0 or node >= data.n:
                errors.append(f"Route {ri} has invalid node {node}")
            elif node != 0:
                load += data.demand[node]
                seen.append(node)

        for a, b in zip(route, route[1:]):
            if 0 <= a < data.n and 0 <= b < data.n:
                cost += distance(data, a, b)

        if load > data.capacity:
            errors.append(f"Route {ri} exceeds capacity: load={load}, capacity={data.capacity}")

        recomputed += cost

        if verbose:
            print(f"route {ri}: load={load:4d} cost={cost:.4f} {' '.join(map(str, route))}")

    missing = sorted(set(range(1, data.n)) - set(seen))
    duplicated = sorted(c for c in set(seen) if seen.count(c) > 1)

    if missing:
        errors.append(f"Missing customers: {missing[:30]}{'...' if len(missing) > 30 else ''}")
    if duplicated:
        errors.append(f"Duplicated customers: {duplicated[:30]}{'...' if len(duplicated) > 30 else ''}")

    if reported_obj is not None:
        diff = abs(recomputed - float(reported_obj))
        if diff > eps:
            errors.append(f"Reported objective differs from recomputed by {diff:.4f}")

    return not errors, recomputed, errors



def verify_one(instance_path, solution_arg, eps=EPS, verbose=False):
    """When the input wasn't a folder"""
    data = VRPData(instance_path)
    routes, reported_obj, flag, record = parse_solution_arg(solution_arg, data)
    ok, recomputed, errors = check_solution(data, routes, reported_obj, eps, verbose)

    print(f"Routes:               {len(routes)}")
    print(f"Recomputed Objective: {recomputed:.4f}")
    if reported_obj is not None:
        print(f"Reported Objective:   {float(reported_obj):.4f}")
    print(f"Optimality:           {flag} ({'proved optimal' if flag == 1 else 'not proved optimal'})")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print("INVALID")
    else:
        print("VALID")

    return 0 if ok else 1



def verify_folder(input_folder, solutions_arg, eps=EPS, verbose=False):
    """When the input involved a folder"""
    input_folder = Path(input_folder)
    sol_path = Path(solutions_arg)

    total = 0
    valid = 0

    if sol_path.exists() and sol_path.is_file():
        records = read_json_records_from_file(sol_path)
        by_name = {r.get("Instance"): r for r in records}

        for inst in sorted(input_folder.glob("*.vrp")):
            total += 1
            data = VRPData(inst)
            record = by_name.get(inst.name)

            if record is None:
                print(f"{inst.name}: MISSING record")
                continue

            try:
                routes, reported_obj, flag, _ = parse_json_record(record, data.vehicles)
                ok, recomputed, errors = check_solution(data, routes, reported_obj, eps, verbose)
            except Exception as exc:
                ok, recomputed, errors = False, 0.0, [str(exc)]

            if ok:
                valid += 1
                print(f"{inst.name}: VALID objective={recomputed:.2f}")
            else:
                print(f"{inst.name}: INVALID objective={recomputed:.2f}")
                for e in errors:
                    print(f"  - {e}")

    else:
        # Folder of .sol files.
        sol_folder = sol_path
        for inst in sorted(input_folder.glob("*.vrp")):
            total += 1
            data = VRPData(inst)
            sol_file = sol_folder / f"{inst.name}.sol"

            if not sol_file.exists():
                print(f"{inst.name}: MISSING {sol_file}")
                continue

            try:
                routes, reported_obj, flag, _ = parse_solution_arg(str(sol_file), data)
                ok, recomputed, errors = check_solution(data, routes, reported_obj, eps, verbose)
            except Exception as exc:
                ok, recomputed, errors = False, 0.0, [str(exc)]

            if ok:
                valid += 1
                print(f"{inst.name}: VALID objective={recomputed:.2f} flag={flag}")
            else:
                print(f"{inst.name}: INVALID objective={recomputed:.2f}")
                for e in errors:
                    print(f"  - {e}")

    print(f"\nSummary: {valid}/{total} valid")
    return 0 if valid == total else 1

# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", help="Input .vrp instance OR folder of .vrp files")
    ap.add_argument("solution", help=".sol file, JSON line/log file, raw JSON string, or folder of .sol files")
    ap.add_argument("--eps", type=float, default=EPS, help="Tolerance for reported objective comparison")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    instance_path = Path(args.instance)

    if instance_path.exists() and instance_path.is_dir():
        return verify_folder(instance_path, args.solution, args.eps, args.verbose)

    return verify_one(args.instance, args.solution, args.eps, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
