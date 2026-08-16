import os
import csv
import glob
import json
import argparse
from collections import defaultdict, Counter

import numpy as np

try:
    import pymannkendall as mk
except ImportError:
    mk = None

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEASONS = ["dry", "pre_monsoon", "monsoon", "winter"]

MIN_VALID_POINTS = 8


def load_from_dir(d):
    out = {}
    pattern = os.path.join(d, "storage_impact_annual_seasonal_*.json")
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        gww = fname[len("storage_impact_annual_seasonal_"):-len(".json")]
        with open(path) as f:
            data = json.load(f)          # {year: {season: value}}
        per_season = defaultdict(dict)
        for year, seasons in data.items():
            y = int(year)
            for s, v in seasons.items():
                if v is not None:
                    per_season[s][y] = float(v)
        out[gww] = per_season
    return out


def load_from_csv(path):
    out = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("impact", "")
            if v is None or v == "":
                continue
            gww = str(row["gww_id"])
            season = row["season"]
            year = int(row["year"])
            out[gww][season][year] = float(v)
    return {g: dict(s) for g, s in out.items()}


def pettitt_test(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return None, 1.0

    U = np.zeros(n)
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += np.sign(x[k] - x[j])
        U[k] = s
    # cumulative
    Uk = np.cumsum(U)
    K = np.argmax(np.abs(Uk))
    Kstat = np.abs(Uk[K])

    # approximate p-value (Pettitt 1979)
    p = 2.0 * np.exp(-6.0 * Kstat ** 2 / (n ** 3 + n ** 2))
    p = min(1.0, p)
    return int(K), float(p)


def mann_kendall(years, vals):
    y = np.asarray(vals, dtype=float)
    if mk is not None:
        r = mk.original_test(y)
        return {
            "trend": r.trend,
            "p": float(r.p),
            "tau": float(r.Tau),
            "slope": float(r.slope),
        }

    n = len(y)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(y[j] - y[i])
    # variance (no tie correction beyond basic)
    unique, counts = np.unique(y, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s) if var_s > 0 else 0.0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s) if var_s > 0 else 0.0
    else:
        z = 0.0
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1)) if n > 1 else 0.0
    # Theil-Sen slope
    slopes = []
    yr = np.asarray(years, dtype=float)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = yr[j] - yr[i]
            if dt != 0:
                slopes.append((y[j] - y[i]) / dt)
    slope = float(np.median(slopes)) if slopes else 0.0
    if p < 0.05 and s > 0:
        trend = "increasing"
    elif p < 0.05 and s < 0:
        trend = "decreasing"
    else:
        trend = "no trend"
    return {"trend": trend, "p": float(p), "tau": float(tau), "slope": slope}


def analyse(data, min_pts):
    trend_rows = []
    feature_map = defaultdict(dict)
    change_years = defaultdict(list)

    for gww, per_season in data.items():
        for season in SEASONS:
            series = per_season.get(season, {})
            if len(series) < min_pts:
                continue
            years = sorted(series)
            vals = [series[y] for y in years]

            mkres = mann_kendall(years, vals)
            k_idx, p_pettitt = pettitt_test(vals)
            change_year = years[k_idx] if k_idx is not None else None
            change_sig = (p_pettitt is not None and p_pettitt < 0.05)

            trend_rows.append({
                "gww_id": gww,
                "season": season,
                "n": len(years),
                "year_start": years[0],
                "year_end": years[-1],
                "mk_trend": mkres["trend"],
                "mk_p": round(mkres["p"], 5),
                "mk_tau": round(mkres["tau"], 4),
                "sen_slope": round(mkres["slope"], 6),
                "mean_impact": round(float(np.mean(vals)), 6),
                "amplitude": round(float(np.max(vals) - np.min(vals)), 6),
                "pettitt_change_year": change_year,
                "pettitt_p": round(p_pettitt, 5),
                "pettitt_significant": change_sig,
            })

            if change_sig and change_year is not None:
                change_years[season].append(change_year)

            # features for clustering (per season)
            feature_map[gww][f"{season}_slope"] = mkres["slope"]
            feature_map[gww][f"{season}_mean"]  = float(np.mean(vals))
            feature_map[gww][f"{season}_ampl"]  = float(np.max(vals) - np.min(vals))
            feature_map[gww][f"{season}_dir"]   = (
                1.0 if mkres["trend"] == "increasing"
                else -1.0 if mkres["trend"] == "decreasing"
                else 0.0
            )

    return trend_rows, feature_map, change_years


def cluster_reservoirs(feature_map, k_range=(2, 8)):
    feature_names = [f"{s}_{stat}" for s in SEASONS for stat in ("slope", "mean", "ampl", "dir")]

    gwws, matrix = [], []
    for gww, feats in feature_map.items():
        if all(name in feats for name in feature_names):
            gwws.append(gww)
            matrix.append([feats[name] for name in feature_names])

    if len(gwws) < max(k_range[0], 3):
        return {}, feature_names, None, None

    X = StandardScaler().fit_transform(np.asarray(matrix, dtype=float))

    best_k, best_score, best_labels = None, -1.0, None
    kmax = min(k_range[1], len(gwws) - 1)
    for k in range(k_range[0], kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        return {}, feature_names, None, None

    return dict(zip(gwws, best_labels.tolist())), feature_names, best_k, best_score


def write_trend_csv(rows, path):
    if not rows:
        print("No trend rows (all series below MIN_VALID_POINTS).")
        return
    cols = ["gww_id", "season", "n", "year_start", "year_end",
            "mk_trend", "mk_p", "mk_tau", "sen_slope",
            "mean_impact", "amplitude",
            "pettitt_change_year", "pettitt_p", "pettitt_significant"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {path}  ({len(rows)} reservoir-season rows)")


def write_cluster_csv(feature_map, labels, feature_names, path):
    with open(path, "w", newline="") as f:
        cols = ["gww_id", "cluster"] + feature_names
        w = csv.writer(f)
        w.writerow(cols)
        for gww in sorted(labels):
            feats = feature_map[gww]
            row = [gww, labels[gww]] + [round(feats.get(n, float("nan")), 6) for n in feature_names]
            w.writerow(row)
    print(f"Written: {path}  ({len(labels)} clustered reservoirs)")


def write_changeyear_hist(change_years, path):
    all_years = []
    for s in SEASONS:
        all_years.extend(change_years.get(s, []))
    years_sorted = sorted(set(all_years))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year"] + SEASONS + ["overall"])
        per_season_counts = {s: Counter(change_years.get(s, [])) for s in SEASONS}
        overall = Counter(all_years)
        for y in years_sorted:
            w.writerow([y] + [per_season_counts[s].get(y, 0) for s in SEASONS] + [overall.get(y, 0)])
    print(f"Written: {path}")
    if all_years:
        mode_year, mode_count = Counter(all_years).most_common(1)[0]
        print(f"Most common significant change year (all seasons): {mode_year} "
              f"({mode_count} reservoir-seasons)")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    #src = p.add_mutually_exclusive_group(required=True)
    p.add_argument("--annual_seasonal_dir", default="geojson/reservoir_corrected",
                     help="dir with storage_impact_annual_seasonal_{gww}.json files")
    p.add_argument("--annual_seasonal_csv",
                     help="long CSV: gww_id,gdw_id,year,season,impact")
    p.add_argument("--out_dir", default="Dataset/trend_analysis")
    p.add_argument("--min_points", type=int, default=MIN_VALID_POINTS,
                   help="min valid years per (reservoir, season) series")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.annual_seasonal_csv:
        data = load_from_csv(args.annual_seasonal_csv)
    else:
        data = load_from_dir(args.annual_seasonal_dir)

    print(f"Loaded {len(data)} reservoirs")
    if mk is None:
        print("NOTE: pymannkendall not found — using local Mann-Kendall fallback.")

    trend_rows, feature_map, change_years = analyse(data, args.min_points)

    labels, feature_names, k, sil = cluster_reservoirs(feature_map)
    if k is not None:
        print(f"Clustering: k={k}  silhouette={sil:.3f}  "
              f"({len(labels)} reservoirs with complete season coverage)")
    else:
        print("Clustering: skipped (too few reservoirs with complete season coverage)")

    write_trend_csv(trend_rows, os.path.join(args.out_dir, "trend_by_season.csv"))
    if labels:
        write_cluster_csv(feature_map, labels, feature_names,
                          os.path.join(args.out_dir, "reservoir_clusters.csv"))
    write_changeyear_hist(change_years,
                          os.path.join(args.out_dir, "changeyear_histogram.csv"))


if __name__ == "__main__":
    main()