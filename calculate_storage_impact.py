import os
import json
import argparse
from calendar import monthrange
from collections import defaultdict

SEASONS = {
    "dry":         [1, 2, 3],
    "pre_monsoon": [4, 5, 6],
    "monsoon":     [7, 8, 9],
    "winter":      [10, 11, 12],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corr_dir",       default="geojson/reservoir_corrected")
    p.add_argument("--reservoir_json", default="geojson/reservoir.json")
    p.add_argument("--out_json",       default="geojson/reservoir_corrected/reservoir.json")
    p.add_argument("--out_csv",        default="Dataset/storage_impact_summary.csv")
    return p.parse_args()


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def parse_ymd(date_str):
    parts = str(date_str).split("-")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None, None


def days_in_month(year, month):
    return monthrange(year, month)[1]


def compute_impact(runoff_ts, storage_ts):
    if runoff_ts is None or storage_ts is None:
        return {s: None for s in SEASONS}

    net_area = runoff_ts.get("net_area_km2") or runoff_ts.get("area_km2")
    if not net_area or net_area <= 0:
        return {s: None for s in SEASONS}

    surface = runoff_ts.get("Surface", [])
    subsurf = runoff_ts.get("Sub_Surface", [])
    r_dates = runoff_ts.get("dates", [])

    runoff_vol = {}
    for i, d in enumerate(r_dates):
        y, m = parse_ymd(d)
        if y is None or m is None:
            continue
        sv = surface[i] if i < len(surface) else None
        bv = subsurf[i] if i < len(subsurf) else None
        if sv is None and bv is None:
            continue
        total_rate = (sv or 0.0) + (bv or 0.0)      
        runoff_vol[(y, m)] = total_rate * net_area * days_in_month(y, m) 

    s_vals  = storage_ts.get("Storage", [])
    s_dates = storage_ts.get("dates", [])
    storage = {}  
    for i, d in enumerate(s_dates):
        y, m = parse_ymd(d)
        if y is None or m is None:
            continue
        sv = s_vals[i] if i < len(s_vals) else None
        if sv is not None:
            storage[(y, m)] = sv

    # Per season, per year
    result = {}
    for season, months in SEASONS.items():
        years = set()
        for (y, m) in storage:
            if m in months:
                years.add(y)

        yearly_impacts = []
        for y in sorted(years):
            s_months = [storage[(y, m)] for m in months if (y, m) in storage]
            v_months = [runoff_vol[(y, m)] for m in months if (y, m) in runoff_vol]
            if not s_months or not v_months:
                continue
            S_ys = sum(s_months) / len(s_months)      # mean storage        [MCM]
            V_ys = sum(v_months)                       # total seasonal vol  [MCM]
            if V_ys <= 0:
                continue
            yearly_impacts.append(S_ys / V_ys)         # dimensionless (days)

        result[season] = (round(sum(yearly_impacts) / len(yearly_impacts), 6)
                          if yearly_impacts else None)

    return result


def main():
    args = parse_args()

    reservoirs = load_json(args.reservoir_json)
    print(f"Reservoirs: {len(reservoirs)}")

    impacts_by_gww = {}
    kept_records = []
    computed = skipped = 0

    for rec in reservoirs:
        gww = str(rec.get("gww_id", ""))
        if not gww:
            skipped += 1
            continue
        runoff_ts  = load_json(os.path.join(args.corr_dir, f"timeseries_{gww}.json"))
        storage_ts = load_json(os.path.join(args.corr_dir, f"storage_{gww}.json"))
        impact = compute_impact(runoff_ts, storage_ts)

        has_data = any(v is not None for v in impact.values())
        if not has_data:
            skipped += 1
            continue

        impacts_by_gww[gww] = impact

        for season, val in impact.items():
            rec[season] = val
        kept_records.append(rec)

        with open(os.path.join(args.corr_dir, f"storage_impact_{gww}.json"), "w") as f:
            json.dump(impact, f, separators=(",", ":"))

        computed += 1

    with open(args.out_json, "w") as f:
        json.dump(kept_records, f, separators=(",", ":"))
    print(f"Updated: {args.out_json}  ({len(kept_records)} records)")

    with open(args.out_csv, "w") as f:
        f.write("gww_id,gdw_id,dry,pre_monsoon,monsoon,winter\n")
        for rec in kept_records:
            gww = rec.get("gww_id", "")
            gdw = rec.get("id", "")
            imp = impacts_by_gww.get(str(gww), {})
            def fmt(v): return "" if v is None else f"{v:.6f}"
            f.write(f"{gww},{gdw},{fmt(imp.get('dry'))},{fmt(imp.get('pre_monsoon'))},"
                    f"{fmt(imp.get('monsoon'))},{fmt(imp.get('winter'))}\n")
    print(f"Written: {args.out_csv}")

    print(f"\n  Kept (with data): {computed}")
    print(f"  Dropped (no data): {skipped}")


if __name__ == "__main__":
    main()