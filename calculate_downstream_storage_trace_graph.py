import os
import json
import math
import csv
import argparse
import calendar
from collections import defaultdict

SEASONS = {
    "dry": [1, 2, 3],
    "pre_monsoon": [4, 5, 6],
    "monsoon": [7, 8, 9],
    "winter": [10, 11, 12],
}

WINDOWS = {
    "2001-2005": (2001, 2005),
    "2006-2010": (2006, 2010),
    "2011-2015": (2011, 2015),
    "2016-2021": (2016, 2021),
}

STORAGE_ABS_MAX = 1e6


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", default="Graph/reservoir_graph_nodes.geojson")
    p.add_argument("--edges", default="Graph/reservoir_graph_edges.geojson")
    p.add_argument("--reservoir_json", default="geojson/reservoir.json")
    p.add_argument("--runoff_dir", default="geojson/reservoir")
    p.add_argument("--area_dir", default="geojson/reservoir_corrected")
    p.add_argument("--storage_dir", default="geojson/reservoir_corrected")
    p.add_argument("--out_dir", default="geojson/reservoir_corrected")
    return p.parse_args()


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sid(v):
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    try:
        x = float(s)
        if x.is_integer():
            return str(int(x))
    except (ValueError, TypeError):
        pass
    return s


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def parse_ym(s):
    try:
        p = str(s).split("-")
        return int(p[0]), int(p[1])
    except Exception:
        return None, None


def r(v, n=6):
    return round(v, n) if finite(v) else None


def load_graph(nodes_path, edges_path):
    nodes_fc = load_json(nodes_path)
    edges_fc = load_json(edges_path)
    if not nodes_fc:
        raise FileNotFoundError(nodes_path)
    if not edges_fc:
        raise FileNotFoundError(edges_path)

    nodes = {}
    downstream = {}
    for f in nodes_fc.get("features", []):
        p = f.get("properties") or {}
        node_id = sid(p.get("id"))
        if not node_id:
            continue
        raw = p.get("downstream") or []
        if not isinstance(raw, list):
            raw = [raw]
        ds = []
        for x in raw:
            x = sid(x)
            if x and x != node_id and x not in ds:
                ds.append(x)
        nodes[node_id] = f
        downstream[node_id] = ds

    ancestor_index = defaultdict(list)
    for A, ds in downstream.items():
        for pos, X in enumerate(ds):
            ancestor_index[X].append({"id": A, "pos": pos, "hops": pos + 1})
    for X in ancestor_index:
        ancestor_index[X].sort(key=lambda z: (z["pos"], z["id"]))

    edges = set()
    for f in edges_fc.get("features", []):
        p = f.get("properties") or {}
        a, b = sid(p.get("from_id")), sid(p.get("to_id"))
        if a and b:
            edges.add((a, b))

    return nodes, downstream, dict(ancestor_index), edges


def validate_graph(nodes, downstream, edges):
    warnings = []
    for A, ds in downstream.items():
        if not ds:
            continue
        B = ds[0]
        if B not in nodes:
            warnings.append(f"{A}: downstream[0]={B} missing from nodes")
        if (A, B) not in edges:
            warnings.append(f"{A}: direct edge {A}->{B} missing from edges")
    return warnings


def metadata_map(path):
    obj = load_json(path) or []
    records = obj if isinstance(obj, list) else list(obj.values())
    return {sid(x.get("gww_id")): x for x in records if isinstance(x, dict) and sid(x.get("gww_id"))}


def full_area(gww, original_ts, area_dir):
    a = original_ts.get("area_km2")
    if finite(a) and a > 0:
        return float(a)

    corr = load_json(os.path.join(area_dir, f"timeseries_{gww}.json"))
    if corr:
        a = corr.get("area_km2")
        if finite(a) and a > 0:
            return float(a)
    return None


def load_full_runoff(gww, runoff_dir, area_dir):
    ts = load_json(os.path.join(runoff_dir, f"timeseries_{gww}.json"))
    if not ts:
        return None, None
    area = full_area(gww, ts, area_dir)
    if not area:
        return None, None

    dates = ts.get("dates", [])
    surf = ts.get("Surface", [])
    sub = ts.get("Sub_Surface", [])
    out = {}
    for i, d in enumerate(dates):
        y, m = parse_ym(d)
        if y is None:
            continue
        s = surf[i] if i < len(surf) else None
        b = sub[i] if i < len(sub) else None
        if s is None and b is None:
            continue
        # m/month * km2 = MCM
        out[(y, m)] = ((s or 0.0) + (b or 0.0)) * area
    return out, area


def load_storage(gww, storage_dir):
    ts = load_json(os.path.join(storage_dir, f"storage_{gww}.json"))
    if not ts:
        return None
    dates = ts.get("dates", [])
    vals = ts.get("Storage", [])
    out = {}
    for i, d in enumerate(dates):
        y, m = parse_ym(d)
        if y is None:
            continue
        v = vals[i] if i < len(vals) else None
        if not finite(v):
            continue
        if abs(v) > STORAGE_ABS_MAX:
            continue
        out[(y, m)] = float(v)
    return out



def delta_storage(storage, year, months):
    if not storage:
        return None
    vals = [(m, storage[(year, m)]) for m in months if (year, m) in storage]
    if len(vals) < 2:
        return None
    vals.sort()
    return vals[-1][1] - vals[0][1]


def seasonal_runoff(runoff, year, months):
    if not runoff:
        return None, 0
    vol = 0.0
    days = 0
    found = False
    for m in months:
        v = runoff.get((year, m))
        if finite(v):
            vol += v
            days += calendar.monthrange(year, m)[1]
            found = True
    return (vol, days) if found else (None, 0)


def impact(dS, V, days):
    if not finite(dS) or not finite(V) or V == 0:
        return None, None, None
    ratio = dS / V
    pct = 100.0 * ratio
    eq_days = ratio * days if days else None
    return ratio, pct, eq_days


def all_years(storage_by_id):
    ys = set()
    for s in storage_by_id.values():
        if s:
            ys.update(y for y, _ in s)
    return sorted(ys)


def calculate(nodes, downstream, ancestors, meta, runoff, areas, storage):
    pair_rows = []
    target_rows = []
    target_json = {}

    for target in nodes:
        if target not in runoff:
            continue

        target_json[target] = {
            "gww_id": target,
            "gdw_id": meta.get(target, {}).get("id"),
            "downstream": downstream.get(target, []),
            "annual": {},
        }

        for year in all_years(storage):
            season_json = {}

            for season, months in SEASONS.items():
                Vd, days = seasonal_runoff(runoff.get(target), year, months)
                if not finite(Vd) or Vd == 0:
                    continue

                target_dS = delta_storage(storage.get(target), year, months)
                work = []

                for a in ancestors.get(target, []):
                    source = a["id"]
                    dS = delta_storage(storage.get(source), year, months)
                    if not finite(dS):
                        continue

                    Vs, _ = seasonal_runoff(runoff.get(source), year, months)
                    ratio, pct, eq_days = impact(dS, Vd, days)
                    rc = 100.0 * Vs / Vd if finite(Vs) else None
                    ac = None
                    if finite(areas.get(source)) and finite(areas.get(target)) and areas[target] != 0:
                        ac = 100.0 * areas[source] / areas[target]

                    work.append({
                        "source": source,
                        "hops": a["hops"],
                        "direct": a["pos"] == 0,
                        "dS": dS,
                        "Vs": Vs,
                        "ratio": ratio,
                        "pct": pct,
                        "days": eq_days,
                        "runoff_cov": rc,
                        "area_cov": ac,
                    })

                upstream_dS = sum(x["dS"] for x in work) if work else 0.0
                abs_total = sum(abs(x["dS"]) for x in work)
                draw_total = sum(max(-x["dS"], 0.0) for x in work)
                fill_total = sum(max(x["dS"], 0.0) for x in work)

                source_json = []
                for x in work:
                    activity_share = 100.0 * abs(x["dS"]) / abs_total if abs_total else None
                    draw_share = 100.0 * (-x["dS"]) / draw_total if x["dS"] < 0 and draw_total else None
                    fill_share = 100.0 * x["dS"] / fill_total if x["dS"] > 0 and fill_total else None
                    signed_share = 100.0 * x["dS"] / upstream_dS if upstream_dS else None

                    pair_rows.append({
                        "source_gww_id": x["source"],
                        "source_gdw_id": meta.get(x["source"], {}).get("id"),
                        "target_gww_id": target,
                        "target_gdw_id": meta.get(target, {}).get("id"),
                        "year": year,
                        "season": season,
                        "hops": x["hops"],
                        "direct": int(x["direct"]),
                        "source_delta_storage_mcm": x["dS"],
                        "source_full_runoff_mcm": x["Vs"],
                        "target_full_runoff_mcm": Vd,
                        "pair_impact_ratio": x["ratio"],
                        "pair_impact_pct_of_target_runoff": x["pct"],
                        "pair_impact_days": x["days"],
                        "runoff_coverage_pct": x["runoff_cov"],
                        "area_coverage_pct": x["area_cov"],
                        "activity_share_pct": activity_share,
                        "drawdown_share_pct": draw_share,
                        "fill_share_pct": fill_share,
                        "signed_storage_change_share_pct": signed_share,
                    })

                    source_json.append({
                        "gww_id": x["source"],
                        "gdw_id": meta.get(x["source"], {}).get("id"),
                        "hops": x["hops"],
                        "direct": x["direct"],
                        "delta_storage_mcm": r(x["dS"]),
                        "pair_impact_pct_of_target_runoff": r(x["pct"]),
                        "pair_impact_days": r(x["days"]),
                        "runoff_coverage_pct": r(x["runoff_cov"]),
                        "area_coverage_pct": r(x["area_cov"]),
                        "activity_share_pct": r(activity_share),
                        "drawdown_share_pct": r(draw_share),
                        "fill_share_pct": r(fill_share),
                    })

                up_ratio, up_pct, up_days = impact(upstream_dS, Vd, days)
                system_dS = upstream_dS + target_dS if finite(target_dS) else None
                sys_ratio, sys_pct, sys_days = impact(system_dS, Vd, days)

                target_rows.append({
                    "target_gww_id": target,
                    "target_gdw_id": meta.get(target, {}).get("id"),
                    "year": year,
                    "season": season,
                    "upstream_count": len(work),
                    "target_full_runoff_mcm": Vd,
                    "target_delta_storage_mcm": target_dS,
                    "upstream_delta_storage_mcm": upstream_dS,
                    "system_delta_storage_mcm": system_dS,
                    "upstream_drawdown_proxy_mcm": draw_total,
                    "upstream_fill_proxy_mcm": fill_total,
                    "upstream_impact_ratio": up_ratio,
                    "upstream_impact_pct_of_target_runoff": up_pct,
                    "upstream_impact_days": up_days,
                    "system_impact_ratio": sys_ratio,
                    "system_impact_pct_of_target_runoff": sys_pct,
                    "system_impact_days": sys_days,
                    "upstream_drawdown_pct_of_target_runoff": 100.0 * draw_total / Vd,
                    "upstream_fill_pct_of_target_runoff": 100.0 * fill_total / Vd,
                })

                source_json.sort(key=lambda z: abs(z["delta_storage_mcm"] or 0), reverse=True)
                season_json[season] = {
                    "target_full_runoff_mcm": r(Vd),
                    "target_delta_storage_mcm": r(target_dS),
                    "upstream_delta_storage_mcm": r(upstream_dS),
                    "system_delta_storage_mcm": r(system_dS),
                    "upstream_impact_pct_of_target_runoff": r(up_pct),
                    "upstream_impact_days": r(up_days),
                    "system_impact_pct_of_target_runoff": r(sys_pct),
                    "system_impact_days": r(sys_days),
                    "upstream_drawdown_proxy_mcm": r(draw_total),
                    "upstream_drawdown_pct_of_target_runoff": r(100.0 * draw_total / Vd),
                    "sources": source_json,
                }

            if season_json:
                target_json[target]["annual"][str(year)] = season_json

    return pair_rows, target_rows, target_json


def win(year):
    for label, (a, b) in WINDOWS.items():
        if a <= int(year) <= b:
            return label
    return None


def average_window(rows, identity_cols):
    groups = defaultdict(list)
    for row in rows:
        w = win(row["year"])
        if w:
            groups[tuple(row.get(k) for k in identity_cols) + (w,)].append(row)

    output = []
    skip = set(identity_cols) | {"year"}
    for key, members in groups.items():
        out = {k: key[i] for i, k in enumerate(identity_cols)}
        out["window"] = key[-1]
        out["n_years"] = len({m["year"] for m in members})
        numeric_cols = set().union(*(m.keys() for m in members)) - skip
        for col in numeric_cols:
            vals = [m.get(col) for m in members if finite(m.get(col))]
            if vals:
                out[col] = sum(vals) / len(vals)
        output.append(out)
    return output


def write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    # Include any fields appearing later.
    for row in rows[1:]:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            clean = {}
            for k in fields:
                v = row.get(k)
                clean[k] = round(v, 6) if finite(v) else ("" if v is None else v)
            w.writerow(clean)


def main():
    a = args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("[1] Loading existing graph connectivity...")
    nodes, downstream, ancestors, edges = load_graph(a.nodes, a.edges)
    print(f"  nodes={len(nodes)} edges={len(edges)} targets_with_ancestors={len(ancestors)}")

    warnings = validate_graph(nodes, downstream, edges)
    if warnings:
        print(f"  [WARN] {len(warnings)} direct-link inconsistencies")
        for x in warnings[:20]:
            print("   -", x)
    else:
        print("  direct node/edge validation: OK")

    connectivity = {}
    for target in nodes:
        connectivity[target] = {
            "downstream": downstream.get(target, []),
            "ancestors": [
                {
                    "gww_id": x["id"],
                    "hops": x["hops"],
                    "direct": x["pos"] == 0,
                }
                for x in ancestors.get(target, [])
            ],
        }
    p = os.path.join(a.out_dir, "reservoir_graph_connectivity.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(connectivity, f, separators=(",", ":"))
    print("  written:", p)

    print("\n[2] Loading metadata, full runoff, full areas, and storage...")
    meta = metadata_map(a.reservoir_json)
    runoff, areas, storage = {}, {}, {}
    for gww in nodes:
        rr, ar = load_full_runoff(gww, a.runoff_dir, a.area_dir)
        if rr:
            runoff[gww] = rr
        if finite(ar):
            areas[gww] = ar
        ss = load_storage(gww, a.storage_dir)
        if ss:
            storage[gww] = ss
    print(f"  runoff={len(runoff)} full_area={len(areas)} storage={len(storage)}")

    print("\n[3] Calculating source -> target influence...")
    pair_rows, target_rows, target_json = calculate(
        nodes, downstream, ancestors, meta, runoff, areas, storage
    )
    print(f"  pair annual-season rows={len(pair_rows)}")
    print(f"  target annual-season rows={len(target_rows)}")

    print("\n[4] Averaging within 5-year windows...")
    pair_win = average_window(
        pair_rows,
        ["source_gww_id", "source_gdw_id", "target_gww_id", "target_gdw_id", "season", "hops", "direct"],
    )
    target_win = average_window(
        target_rows,
        ["target_gww_id", "target_gdw_id", "season"],
    )

    outputs = [
        ("downstream_storage_trace_annual.csv", pair_rows),
        ("downstream_storage_trace_windowed.csv", pair_win),
        ("downstream_storage_trace_target_annual.csv", target_rows),
        ("downstream_storage_trace_target_windowed.csv", target_win),
    ]
    for name, rows in outputs:
        path = os.path.join(a.out_dir, name)
        write_csv(path, rows)
        print("  written:", path)

    count = 0
    for target, obj in target_json.items():
        if not obj["annual"]:
            continue
        path = os.path.join(a.out_dir, f"downstream_storage_trace_{target}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))
        count += 1
    print(f"  per-target JSON files={count}")

    print("\nDone.")
    print("Output folder:", os.path.abspath(a.out_dir))
    print("Negative dS is retained as a net drawdown/release proxy; it is NOT treated as invalid.")


if __name__ == "__main__":
    main()