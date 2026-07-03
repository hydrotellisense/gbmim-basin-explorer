"""
diagnose.py
-----------
Checks the geojson/reservoir/ output folder to verify every feature has:
  - timeseries_{gww_id}.json  with non-empty Surface, Sub_Surface, Precip
  - storage_{gww_id}.json     present
  - watershed_{gww_id}.geojson present
  - downstream_{gww_id}.geojson present

All files are keyed by GWW_reservoir_id (leading zeros stripped).
"""

import os
import json
import argparse
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reservoir_dir",  default="geojson/reservoir",
                   help="Directory containing per-feature files")
    p.add_argument("--reservoir_json", default="geojson/reservoir.json",
                   help="Coordinate JSON to map gww_id → gdw_id")
    return p.parse_args()


def scan_gww_ids(res_dir):
    """Return sorted list of GWW ids from timeseries_*.json filenames."""
    ids = []
    for name in os.listdir(res_dir):
        if name.startswith("timeseries_") and name.endswith(".json"):
            ids.append(name[len("timeseries_"):-len(".json")])
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else float('inf'))


def load_gww_to_gdw(json_path):
    """Load reservoir.json and return {gww_id: gdw_id} mapping."""
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path) as f:
            records = json.load(f)
        return {str(r.get("gww_id", "")): str(r.get("id", ""))
                for r in records if r.get("gww_id")}
    except Exception:
        return {}


def check_timeseries(path):
    """Return dict of metric → status string."""
    metrics = ["Surface", "Sub_Surface", "Precip"]
    if not os.path.exists(path):
        return {m: "missing_file" for m in metrics}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {m: "parse_error" for m in metrics}
    result = {}
    for m in metrics:
        vals = data.get(m, [])
        non_null = [v for v in vals if v is not None]
        result[m] = "ok" if non_null else ("empty" if vals else "missing_key")
    return result


def check_file_exists(path):
    return "ok" if os.path.exists(path) else "missing"


def main():
    args = parse_args()
    res_dir = args.reservoir_dir

    if not os.path.isdir(res_dir):
        print(f"[ERROR] Directory not found: {res_dir}")
        return

    gww_ids = scan_gww_ids(res_dir)
    gww_to_gdw = load_gww_to_gdw(args.reservoir_json)

    print(f"Features found (by timeseries files): {len(gww_ids)}")
    if gww_to_gdw:
        print(f"GWW→GDW mapping loaded: {len(gww_to_gdw)} entries")
    print()

    issues = defaultdict(list)
    counters = defaultdict(int)

    for gww in gww_ids:
        # Timeseries metrics
        ts_checks = check_timeseries(os.path.join(res_dir, f"timeseries_{gww}.json"))
        for metric, status in ts_checks.items():
            if status != "ok":
                issues[gww].append(f"runoff/{metric}: {status}")
                counters[f"runoff/{metric}/{status}"] += 1
            else:
                counters[f"runoff/{metric}/ok"] += 1

        # Storage
        st = check_file_exists(os.path.join(res_dir, f"storage_{gww}.json"))
        if st != "ok":
            issues[gww].append(f"storage: {st}")
            counters["storage/missing"] += 1
        else:
            counters["storage/ok"] += 1

        # Watershed
        ws = check_file_exists(os.path.join(res_dir, f"watershed_{gww}.geojson"))
        if ws != "ok":
            issues[gww].append(f"watershed: {ws}")
            counters["watershed/missing"] += 1
        else:
            counters["watershed/ok"] += 1

        # Downstream
        ds = check_file_exists(os.path.join(res_dir, f"downstream_{gww}.geojson"))
        if ds != "ok":
            issues[gww].append(f"downstream: {ds}")
            counters["downstream/missing"] += 1
        else:
            counters["downstream/ok"] += 1

    # ── Summary ──────────────────────────────────────────────────
    print("=== Coverage Summary ===")
    checks = [
        ("runoff/Surface",     "Surface Runoff"),
        ("runoff/Sub_Surface", "Sub-Surface Runoff"),
        ("runoff/Precip",      "Precipitation"),
        ("storage",            "Storage"),
        ("watershed",          "Watershed"),
        ("downstream",         "Downstream"),
    ]
    for key, label in checks:
        ok      = counters.get(f"{key}/ok", 0)
        missing = (counters.get(f"{key}/missing", 0)
                   + counters.get(f"{key}/missing_file", 0)
                   + counters.get(f"{key}/missing_key", 0))
        empty   = counters.get(f"{key}/empty", 0)
        total   = ok + missing + empty
        parts   = [f"{ok}/{total} ok"]
        if missing: parts.append(f"{missing} missing")
        if empty:   parts.append(f"{empty} empty")
        print(f"  {label:<22}: {', '.join(parts)}")

    # ── Per-feature issues ────────────────────────────────────────
    if issues:
        print(f"\n=== Features with Issues ({len(issues)}) ===")
        for gww in sorted(issues.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            gdw = gww_to_gdw.get(gww, "?")
            print(f"  gww={gww:>8}  gdw={gdw:>8} : {'; '.join(issues[gww])}")
    else:
        print("\nAll features complete — no issues found.")


if __name__ == "__main__":
    main()