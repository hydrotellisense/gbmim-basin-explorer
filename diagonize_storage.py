import os
import json
import argparse
from collections import Counter, defaultdict

SEASONS = {
    "dry":         [1, 2, 3],
    "pre_monsoon": [4, 5, 6],
    "monsoon":     [7, 8, 9],
    "winter":      [10, 11, 12],
}
MONTH_SEASON = {m: s for s, months in SEASONS.items() for m in months}

# Common nodata sentinels seen in satellite / GWW-style products
SENTINELS = {-9999.0, -999.0, -99.0, -1.0, -9998.0, -32768.0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corr_dir",       default="geojson/reservoir_corrected")
    p.add_argument("--reservoir_json", default="geojson/reservoir.json")
    p.add_argument("--storage_pattern", default="storage_{gww}.json")
    p.add_argument("--out_csv",        default="Dataset/negative_storage_report.csv")
    p.add_argument("--show", type=int, default=25,
                   help="How many worst reservoirs to print")
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


def main():
    args = parse_args()

    reservoirs = load_json(args.reservoir_json)
    if reservoirs is None:
        print(f"[ERR] could not read {args.reservoir_json}")
        return
    print(f"Scanning storage for {len(reservoirs)} reservoirs in {args.corr_dir}/\n")

    rows = []                       # per-reservoir summary
    total_neg = 0
    total_files = 0
    no_storage = 0
    season_neg = Counter()          # season -> count of negative months (all reservoirs)
    sentinel_hist = Counter()       # exact sentinel value -> count

    for rec in reservoirs:
        gww = str(rec.get("gww_id", ""))
        gdw = rec.get("id", "")
        if not gww:
            continue
        st = load_json(os.path.join(args.corr_dir, args.storage_pattern.format(gww=gww)))
        if st is None:
            no_storage += 1
            continue
        total_files += 1

        vals  = st.get("Storage", [])
        dates = st.get("dates", [])
        n = min(len(vals), len(dates))

        neg_idx = [i for i in range(n)
                   if vals[i] is not None and vals[i] < 0]
        if not neg_idx:
            continue

        neg_vals = [vals[i] for i in neg_idx]
        min_val = min(neg_vals)
        # sentinel detection: a single exact negative value repeated is very
        # likely a nodata code, not a physical undershoot
        vc = Counter(neg_vals)
        top_val, top_count = vc.most_common(1)[0]
        looks_sentinel = (top_val in SENTINELS) or (top_count >= 3 and top_val == min_val)
        if top_val in SENTINELS:
            sentinel_hist[top_val] += top_count

        neg_dates = [dates[i] for i in neg_idx]
        seasons_hit = Counter()
        for d in neg_dates:
            y, m = parse_ymd(d)
            if m is not None:
                seasons_hit[MONTH_SEASON.get(m, "?")] += 1
                season_neg[MONTH_SEASON.get(m, "?")] += 1

        total_neg += len(neg_idx)
        rows.append({
            "gww": gww, "gdw": gdw,
            "n_neg": len(neg_idx), "n_total": n,
            "min_val": min_val,
            "kind": "sentinel" if looks_sentinel else "small-negative",
            "seasons": dict(seasons_hit),
            "first_dates": neg_dates[:5],
        })

    # ---- report ----
    rows.sort(key=lambda r: (r["kind"] != "sentinel", -r["n_neg"], r["min_val"]))

    print("=" * 78)
    print(f"Reservoirs scanned with a storage file : {total_files}")
    print(f"Reservoirs with NO storage file        : {no_storage}")
    print(f"Reservoirs with >=1 negative storage   : {len(rows)}")
    print(f"Total negative storage timesteps       : {total_neg}")
    if sentinel_hist:
        print(f"Sentinel values detected               : "
              + ", ".join(f"{v:g}×{c}" for v, c in sentinel_hist.items()))
    if season_neg:
        print(f"Negative months by season              : "
              + ", ".join(f"{s}={season_neg[s]}" for s in SEASONS if season_neg[s]))
    print("=" * 78)

    if rows:
        print(f"\nWorst {min(args.show, len(rows))} reservoirs "
              f"(sentinels first, then most frequent):\n")
        print(f"{'gww':>10} {'gdw':>8} {'kind':>15} {'#neg':>5} {'#tot':>5} "
              f"{'min_val':>14}  seasons")
        for r in rows[:args.show]:
            seas = ",".join(f"{k}:{v}" for k, v in r["seasons"].items())
            print(f"{r['gww']:>10} {str(r['gdw']):>8} {r['kind']:>15} "
                  f"{r['n_neg']:>5} {r['n_total']:>5} {r['min_val']:>14.3f}  {seas}")

    # ---- CSV ----
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("gww_id,gdw_id,kind,n_negative,n_total,min_value,"
                "dry,pre_monsoon,monsoon,winter,first_negative_dates\n")
        for r in rows:
            s = r["seasons"]
            f.write(f"{r['gww']},{r['gdw']},{r['kind']},{r['n_neg']},{r['n_total']},"
                    f"{r['min_val']:.4f},"
                    f"{s.get('dry',0)},{s.get('pre_monsoon',0)},"
                    f"{s.get('monsoon',0)},{s.get('winter',0)},"
                    f"{'|'.join(r['first_dates'])}\n")
    print(f"\nWritten: {args.out_csv}")

    if rows:
        n_sent = sum(1 for r in rows if r["kind"] == "sentinel")
        print(f"\nInterpretation:")
        print(f"  {n_sent} reservoir(s) look like unmasked NODATA sentinels "
              f"(mask these before computing storage impact).")
        print(f"  {len(rows) - n_sent} have small scattered negatives "
              f"(likely anomaly product or area->volume undershoot near empty).")


if __name__ == "__main__":
    main()