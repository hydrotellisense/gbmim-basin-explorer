import os
import json
import argparse

SEASONS = {
    "dry":         [1, 2, 3],
    "pre_monsoon": [4, 5, 6],
    "monsoon":     [7, 8, 9],
    "winter":      [10, 11, 12],
}

WINDOWS = {
    "2001-2005": (2001, 2005),
    "2006-2010": (2006, 2010),
    "2011-2015": (2011, 2015),
    "2016-2021": (2016, 2021),
}

STORAGE_ABS_MAX = 1e6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corr_dir",       default="geojson/reservoir_corrected")
    p.add_argument("--reservoir_json", default="geojson/reservoir.json")
    p.add_argument("--out_json",       default="geojson/reservoir_corrected/reservoir.json")
    p.add_argument("--out_csv",        default="Dataset/storage_impact_summary.csv")
    p.add_argument("--out_annual_csv", default="Dataset/storage_impact_annual.csv")
    p.add_argument("--denom", choices=["total", "mean"], default="total",
                   help="divide by the total (sum) or mean of the window's monthly inflow")
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


def parse_series(runoff_ts, storage_ts):
    if runoff_ts is None or storage_ts is None:
        return None, None
    net_area = runoff_ts.get("net_area_km2") or runoff_ts.get("area_km2")
    if not net_area or net_area <= 0:
        return None, None

    surface = runoff_ts.get("Surface", [])
    subsurf = runoff_ts.get("Sub_Surface", [])
    r_dates = runoff_ts.get("dates", [])

    runoff_vol = {}
    for i, d in enumerate(r_dates):
        y, m = parse_ymd(d)
        if y is None:
            continue
        sv = surface[i] if i < len(surface) else None
        bv = subsurf[i] if i < len(subsurf) else None
        if sv is None and bv is None:
            continue
        rate = (sv or 0.0) + (bv or 0.0)          # m/month
        runoff_vol[(y, m)] = rate * net_area      # MCM

    s_vals  = storage_ts.get("Storage", [])
    s_dates = storage_ts.get("dates", [])
    storage = {}
    for i, d in enumerate(s_dates):
        y, m = parse_ymd(d)
        if y is None:
            continue
        sv = s_vals[i] if i < len(s_vals) else None
        if sv is None:
            continue
        if STORAGE_ABS_MAX is not None and abs(sv) > STORAGE_ABS_MAX:
            continue
        storage[(y, m)] = sv

    return runoff_vol, storage


def compute_annual_seasonal(runoff_vol, storage, denom_mode):
    years = sorted({y for (y, m) in storage})
    out = {}
    for y in years:
        seasons = {}
        for season, months in SEASONS.items():
            iv = _window_impact(storage, runoff_vol, y, months, denom_mode)
            seasons[season] = round(iv, 6) if iv is not None else None
        if any(v is not None for v in seasons.values()):
            out[y] = seasons
    return out


def _window_impact(storage, runoff_vol, y, months, denom_mode):
    present = [(m, storage[(y, m)]) for m in months if (y, m) in storage]
    if len(present) < 2:
        return None
    present.sort()
    vals = [v for _, v in present]

    net_dS = vals[-1] - vals[0]

    volumes = [runoff_vol[(y, m)] for m in months if (y, m) in runoff_vol]
    if not volumes:
        return None
    denom = sum(volumes) if denom_mode == "total" else sum(volumes) / len(volumes)
    if denom == 0:
        return None
    return net_dS / denom


def compute_windowed(runoff_vol, storage, denom_mode):
    result = {}
    for wlabel, (y0, y1) in WINDOWS.items():
        wseasons = {}
        for season, months in SEASONS.items():
            vals = []
            for y in range(y0, y1 + 1):
                iv = _window_impact(storage, runoff_vol, y, months, denom_mode)
                if iv is not None:
                    vals.append(iv)
            wseasons[season] = round(sum(vals) / len(vals), 6) if vals else None
        result[wlabel] = wseasons
    return result


def compute_annual(runoff_vol, storage, denom_mode):
    years = sorted({y for (y, m) in storage})
    out = {}
    for y in years:
        iv = _window_impact(storage, runoff_vol, y, list(range(1, 13)), denom_mode)
        if iv is not None:
            out[y] = round(iv, 6)
    return out


def main():
    args = parse_args()
    reservoirs = load_json(args.reservoir_json)
    print(f"Reservoirs: {len(reservoirs)}  (storage impact = net change; denominator = {args.denom})")
    print(f"Windows: {', '.join(WINDOWS.keys())}")

    windowed_by_gww = {}
    annual_by_gww = {}
    annual_seasonal_by_gww = {}
    kept_records = []
    computed = skipped = 0

    for rec in reservoirs:
        gww = str(rec.get("gww_id", ""))
        if not gww:
            skipped += 1
            continue
        runoff_ts  = load_json(os.path.join(args.corr_dir, f"timeseries_{gww}.json"))
        storage_ts = load_json(os.path.join(args.corr_dir, f"storage_{gww}.json"))
        runoff_vol, storage = parse_series(runoff_ts, storage_ts)
        if runoff_vol is None or not storage:
            skipped += 1
            continue

        windowed = compute_windowed(runoff_vol, storage, args.denom)
        annual   = compute_annual(runoff_vol, storage, args.denom)
        annual_seasonal = compute_annual_seasonal(runoff_vol, storage, args.denom)

        has_data = any(v is not None for w in windowed.values() for v in w.values())
        if not has_data:
            skipped += 1
            continue

        windowed_by_gww[gww] = windowed
        annual_by_gww[gww]   = annual
        annual_seasonal_by_gww[gww] = annual_seasonal

        # Nested impact object on the record: {window: {season: value}}
        rec["impact"] = windowed
        kept_records.append(rec)

        with open(os.path.join(args.corr_dir, f"storage_impact_{gww}.json"), "w") as f:
            json.dump(windowed, f, separators=(",", ":"))
        with open(os.path.join(args.corr_dir, f"storage_impact_annual_{gww}.json"), "w") as f:
            json.dump({str(y): annual[y] for y in annual}, f, separators=(",", ":"))
        with open(os.path.join(args.corr_dir, f"storage_impact_annual_seasonal_{gww}.json"), "w") as f:
            json.dump({str(y): annual_seasonal[y] for y in annual_seasonal}, f, separators=(",", ":"))
        computed += 1

    with open(args.out_json, "w") as f:
        json.dump(kept_records, f, separators=(",", ":"))
    print(f"Updated: {args.out_json}  ({len(kept_records)} records)")

    def fmt(v):
        return "" if v is None else f"{v:.6f}"

    # Wide summary CSV: one row per reservoir, all window×season impact values
    with open(args.out_csv, "w") as f:
        header = ["gww_id", "gdw_id"]
        for w in WINDOWS:
            for s in SEASONS:
                header.append(f"{w}_{s}")
        f.write(",".join(header) + "\n")
        for rec in kept_records:
            gww = str(rec.get("gww_id", ""))
            gdw = rec.get("id", "")
            wd = windowed_by_gww.get(gww, {})
            vals = [fmt(wd.get(w, {}).get(s)) for w in WINDOWS for s in SEASONS]
            f.write(f"{gww},{gdw}," + ",".join(vals) + "\n")
    print(f"Written: {args.out_csv}")

    with open(args.out_annual_csv, "w") as f:
        f.write("gww_id,gdw_id,year,season,impact\n")
        for rec in kept_records:
            gww = str(rec.get("gww_id", ""))
            gdw = rec.get("id", "")
            asd = annual_seasonal_by_gww.get(gww, {})
            for y in sorted(asd):
                for s in SEASONS:
                    v = asd[y].get(s)
                    if v is not None:
                        f.write(f"{gww},{gdw},{y},{s},{v:.6f}\n")

    print(f"\n  Kept (with data): {computed}")
    print(f"  Dropped (no data): {skipped}")


if __name__ == "__main__":
    main()