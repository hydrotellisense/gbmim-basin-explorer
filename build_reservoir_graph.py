"""
Build a downstream connectivity graph between reservoirs, using only:
  1. The coordinates JSON (reservoir.json) with all reservoir ids + lat/lng
  2. The per-reservoir downstream path files (downstream_{id}.geojson)

Logic: reservoir B is downstream of reservoir A if the START of B's
downstream path lies on A's downstream path. Path start points sit exactly
on the river network (the delineator snapped them there), unlike the raw
reservoir coordinates which are often polygon centroids several km off the
river line. Hits are ordered by distance along A's path; the first hit is
A's immediate downstream reservoir (direct edge).

Outputs:
  - reservoir_graph.json    : nodes, direct edges, and full ordered downstream
    reservoir list per reservoir
  - reservoir_graph.geojson : FeatureCollection with node Points and direct-edge
    LineStrings (edges follow the actual downstream path where possible).
    Storage impact values from the coordinates JSON are carried onto both.
  - reservoir_edges.csv     : flat edge list (from_id, to_id, direct, hops)

Usage:
  python build_reservoir_graph.py \
      --coords_json geojson/reservoir_corrected/reservoir.json \
      --downstream_dir geojson/reservoir_corrected \
      --out_dir Dataset/Reservoir_graph

Requires: shapely  (pip install shapely)
"""

import argparse
import csv
import json
import os

from shapely.geometry import LineString, Point, shape
from shapely.ops import linemerge, substring
from shapely.strtree import STRtree


def parse_args():
    p = argparse.ArgumentParser(description="Build reservoir downstream connectivity graph.")
    p.add_argument("--coords_json", default="geojson/reservoir_corrected/reservoir.json",
                   help="JSON with all reservoirs (ids + coordinates)")
    p.add_argument("--downstream_dir", default="geojson/reservoir_corrected",
                   help="Directory containing downstream_{id}.geojson files")
    p.add_argument("--downstream_pattern", default="downstream_{id}.geojson",
                   help="Filename pattern for downstream path files")
    p.add_argument("--tolerance", type=float, default=0.002,
                   help="Max distance (degrees) between B's path start and A's path "
                        "line to count B as downstream of A. Path starts lie on the "
                        "river network, so this can be small (~0.002 deg = 200 m)")
    p.add_argument("--out_dir", default="Dataset/Reservoir_graph",
                   help="Where to write reservoir_graph.json/.geojson and reservoir_edges.csv")
    return p.parse_args()


# ---------------------------------------------------------------- input parsing

ID_KEYS = ["GWW_reservoir_id", "gww_reservoir_id", "gww_id", "GWW_ID", "id", "ID"]
LAT_KEYS = ["lat", "latitude", "LAT", "Latitude"]
LNG_KEYS = ["lng", "lon", "long", "longitude", "LNG", "Longitude"]


def pick_key(record, candidates):
    for k in candidates:
        if k in record:
            return k
    return None


def load_reservoirs(coords_json):
    """
    Load reservoir records from the coordinates JSON.
    Handles: a plain list of records, a dict wrapping a list, or a
    GeoJSON FeatureCollection. Returns list of (id, lat, lng, record).
    """
    with open(coords_json) as f:
        data = json.load(f)

    if isinstance(data, dict):
        if data.get("type") == "FeatureCollection":
            records = []
            for feat in data["features"]:
                rec = dict(feat.get("properties", {}))
                geom = feat.get("geometry")
                if geom and geom.get("type") == "Point":
                    rec.setdefault("lng", geom["coordinates"][0])
                    rec.setdefault("lat", geom["coordinates"][1])
                records.append(rec)
            data = records
        else:
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break

    if not isinstance(data, list) or len(data) == 0:
        raise Exception(f"Could not find a list of reservoir records in {coords_json}")

    sample = data[0]
    id_key = pick_key(sample, ID_KEYS)
    lat_key = pick_key(sample, LAT_KEYS)
    lng_key = pick_key(sample, LNG_KEYS)

    if id_key is None or lat_key is None or lng_key is None:
        raise Exception(
            f"Could not detect id/lat/lng keys. Record keys are: {list(sample.keys())}. "
            f"Edit ID_KEYS / LAT_KEYS / LNG_KEYS at the top of this script.")

    print(f"Detected keys: id='{id_key}', lat='{lat_key}', lng='{lng_key}'")

    out = []
    for rec in data:
        rid = str(rec[id_key])
        try:
            lat = float(rec[lat_key])
            lng = float(rec[lng_key])
        except (KeyError, TypeError, ValueError):
            print(f"  (!) Skipping {rid}: missing/invalid coordinates")
            continue
        out.append((rid, lat, lng, rec))
    return out


def load_downstream_path(path_file, own_pt):
    """
    Load a downstream path geojson and return a merged shapely LineString
    oriented so the path STARTS at the reservoir (the endpoint nearest own_pt).
    Returns None if no line geometry found.
    """
    with open(path_file) as f:
        gj = json.load(f)

    geoms = []
    if gj.get("type") == "FeatureCollection":
        for feat in gj["features"]:
            if feat.get("geometry"):
                geoms.append(shape(feat["geometry"]))
    elif gj.get("type") == "Feature":
        geoms.append(shape(gj["geometry"]))
    else:
        geoms.append(shape(gj))

    lines = []
    for g in geoms:
        if g.geom_type == "LineString":
            lines.append(g)
        elif g.geom_type == "MultiLineString":
            lines.extend(g.geoms)

    if not lines:
        return None

    line = lines[0] if len(lines) == 1 else linemerge(lines)

    if line.geom_type == "MultiLineString":
        # Couldn't merge into one line (gaps between reaches).
        # Stitch parts end-to-end in order of distance from the reservoir.
        parts = sorted(line.geoms, key=lambda g: Point(g.coords[0]).distance(own_pt)
                       if Point(g.coords[0]).distance(own_pt) <
                          Point(g.coords[-1]).distance(own_pt)
                       else Point(g.coords[-1]).distance(own_pt))
        coords = []
        for part in parts:
            c = list(part.coords)
            # Orient each part away from what we already have
            if coords and Point(c[-1]).distance(Point(coords[-1])) < \
                          Point(c[0]).distance(Point(coords[-1])):
                c = c[::-1]
            coords.extend(c)
        line = LineString(coords)

    # Orient the line so it starts at the reservoir end
    start_d = Point(line.coords[0]).distance(own_pt)
    end_d = Point(line.coords[-1]).distance(own_pt)
    if end_d < start_d:
        line = LineString(list(line.coords)[::-1])

    return line


# --------------------------------------------------------------- impact values

def extract_impact(record):
    """
    Pull storage/areal impact values out of a reservoir record.
    Any key containing 'impact' (case-insensitive) is carried through,
    e.g. seasonal storage impact fields and areal impact.
    """
    return {k: v for k, v in record.items() if "impact" in k.lower()}


# ----------------------------------------------------------------------- main

def main():
    args = parse_args()

    reservoirs = load_reservoirs(args.coords_json)
    print(f"Loaded {len(reservoirs)} reservoirs from {args.coords_json}")

    impacts = {r[0]: extract_impact(r[3]) for r in reservoirs}

    # ---------------- pass 1: load every path, record its start point --------
    paths = {}        # rid -> oriented LineString
    anchor_pts = {}   # rid -> Point used for matching (path start, or raw coord)
    missing_paths = []

    for rid, lat, lng, _ in reservoirs:
        own_pt = Point(lng, lat)
        path_file = os.path.join(args.downstream_dir,
                                 args.downstream_pattern.format(id=rid))
        if not os.path.isfile(path_file):
            missing_paths.append(rid)
            anchor_pts[rid] = own_pt  # fall back to raw coordinate
            continue
        line = load_downstream_path(path_file, own_pt)
        if line is None or line.is_empty:
            anchor_pts[rid] = own_pt
            continue
        paths[rid] = line
        anchor_pts[rid] = Point(line.coords[0])

    if missing_paths:
        print(f"(!) {len(missing_paths)} reservoirs had no downstream path file "
              f"(first few: {missing_paths[:5]})")
    print(f"Loaded {len(paths)} downstream paths")

    # ---------------- pass 2: match path starts against paths ----------------
    ids = [r[0] for r in reservoirs]
    anchor_list = [anchor_pts[rid] for rid in ids]
    tree = STRtree(anchor_list)

    direct_edges = []      # (from_id, to_id, edge_coords)
    all_downstream = {}    # from_id -> ordered list of downstream reservoir ids
    near_misses = 0        # anchors within 10x tolerance but rejected

    for i, rid in enumerate(ids):
        line = paths.get(rid)
        if line is None:
            all_downstream[rid] = []
            continue

        try:
            cand_idx = tree.query(line, predicate="dwithin",
                                  distance=args.tolerance * 10)
        except TypeError:
            cand_idx = tree.query(line.buffer(args.tolerance * 10),
                                  predicate="intersects")

        own_pos = args.tolerance  # skip hits at the very start (self / co-located)

        hits = []
        for j in cand_idx:
            j = int(j)
            if ids[j] == rid:
                continue
            pt = anchor_list[j]
            d = line.distance(pt)
            if d > args.tolerance:
                if d <= args.tolerance * 10:
                    near_misses += 1
                continue
            pos = line.project(pt)
            if pos <= own_pos:
                continue
            hits.append((pos, ids[j]))

        hits.sort()
        all_downstream[rid] = [h[1] for h in hits]
        if hits:
            to_id = hits[0][1]
            seg = substring(line, 0, hits[0][0])
            if seg is None or seg.is_empty or seg.geom_type != "LineString":
                j = ids.index(to_id)
                seg_coords = [(line.coords[0][0], line.coords[0][1]),
                              (anchor_list[j].x, anchor_list[j].y)]
            else:
                seg_coords = [(round(x, 5), round(y, 5)) for x, y in seg.coords]
            direct_edges.append((rid, to_id, seg_coords))

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(ids)}")

    if near_misses:
        print(f"(i) {near_misses} candidate matches were between 1x and 10x the "
              f"tolerance ({args.tolerance} deg) - if edges seem missing, "
              f"try raising --tolerance")

    # ------------------------------------------------------------- outputs
    os.makedirs(args.out_dir, exist_ok=True)
    graph_json = os.path.join(args.out_dir, "reservoir_graph.json")
    graph_geojson = os.path.join(args.out_dir, "reservoir_graph.geojson")
    edges_csv = os.path.join(args.out_dir, "reservoir_edges.csv")

    graph = {
        "nodes": [{"id": rid, "lat": lat, "lng": lng, **impacts[rid]}
                  for rid, lat, lng, _ in reservoirs],
        "edges": [{"from": a, "to": b} for a, b, _ in direct_edges],
        "downstream": all_downstream,
    }
    with open(graph_json, "w") as f:
        json.dump(graph, f)
    print(f"Wrote {graph_json}: {len(reservoirs)} nodes, {len(direct_edges)} direct edges")

    # GeoJSON: node Points + direct-edge LineStrings, with impact values
    features = []
    for rid, lat, lng, _ in reservoirs:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lng, 5), round(lat, 5)]},
            "properties": {
                "feature_type": "node",
                "id": rid,
                "n_downstream": len(all_downstream.get(rid, [])),
                **impacts[rid],
            },
        })
    for a, b, seg_coords in direct_edges:
        props = {"feature_type": "edge", "from": a, "to": b}
        props.update({f"from_{k}": v for k, v in impacts[a].items()})
        props.update({f"to_{k}": v for k, v in impacts[b].items()})
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [list(c) for c in seg_coords]},
            "properties": props,
        })
    with open(graph_geojson, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"Wrote {graph_geojson}: {len(features)} features "
          f"({len(reservoirs)} nodes + {len(direct_edges)} edges)")

    with open(edges_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_id", "to_id", "direct", "hops"])
        n_rows = 0
        for rid, hits in all_downstream.items():
            for hop, other in enumerate(hits, start=1):
                w.writerow([rid, other, hop == 1, hop])
                n_rows += 1
    print(f"Wrote {edges_csv}: {n_rows} total downstream relations")


if __name__ == "__main__":
    main()