import os
import json
import shutil
import fiona
import geopandas as gpd
from shapely.geometry import shape
from collections import defaultdict

GEOJSON_DIR         = "geojson/reservoir"
CORRECTED_DIR       = "geojson/reservoir_corrected"
RESERVOIR_JSON      = "geojson/reservoir.json"
WATERSHED_BARRIER   = "Dataset/Merged_Catchment/GBMIM_barrier_merged_watersheds_filled.gpkg"
WATERSHED_RESERVOIR = "Dataset/Merged_Catchment/GBMIM_reservoir_merged_watersheds_filled.gpkg"
MAPPING_CSV         = "Dataset/gww_gdw_mapping.csv"
METRICS             = ["Surface", "Sub_Surface"]   # Precip not corrected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mapping(path):
    with open(path) as f:
        records = json.load(f)
    return {str(r["gww_id"]): r for r in records if "gww_id" in r}


def load_csv_gdw_ids():
    import pandas as pd
    if not os.path.exists(MAPPING_CSV):
        raise FileNotFoundError(f"Mapping CSV not found: {MAPPING_CSV}")
    df = pd.read_csv(MAPPING_CSV)
    ids = set()
    for v in df["GDW_ID"].dropna():
        s = str(v).strip()
        if s not in ("", "nan"):
            try:
                ids.add(str(int(float(s))))
            except (ValueError, TypeError):
                pass
    return ids


def align_to_dates(target_dates, source_dates, source_vals):
    src_map = dict(zip(source_dates, source_vals))
    return [src_map.get(d) for d in target_dates]


# ---------------------------------------------------------------------------
# Step 1 — Load watersheds from merged catchment GPKGs
# ---------------------------------------------------------------------------

def load_timeseries(gww_id):
    path = os.path.join(GEOJSON_DIR, f"timeseries_{gww_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_watersheds_from_gpkg(gww_map, csv_gdw_ids):
    # Build GDW_ID → (geometry_ea, area_km2) from both GPKGs
    gdw_to_data = {}   # equal-area geometry for intersection calculations
    gdw_to_geom4326 = {}  # WGS84 geometry for within() checks

    for label, path in [("Barrier", WATERSHED_BARRIER), ("Reservoir", WATERSHED_RESERVOIR)]:
        if not os.path.exists(path):
            print(f"  [WARN] {label} GPKG not found: {path}")
            continue
        layers = fiona.listlayers(path)
        gdf_4326 = gpd.read_file(path, layer=layers[0]).to_crs("EPSG:4326")
        gdf_ea   = gdf_4326.to_crs("EPSG:6933")

        if "GDW_ID" not in gdf_4326.columns:
            print(f"  [WARN] {label} GPKG has no GDW_ID column")
            continue

        added = 0
        for idx in gdf_4326.index:
            gdw_raw = gdf_4326.loc[idx, "GDW_ID"]
            if gdw_raw is None or (isinstance(gdw_raw, float) and gdw_raw != gdw_raw):
                continue
            gdw = str(int(float(gdw_raw)))
            if gdw in gdw_to_data:
                continue  # barrier already added
            if gdw not in csv_gdw_ids:
                continue  # not in CSV — skip
            geom_4326 = gdf_4326.loc[idx, "geometry"]
            geom_ea   = gdf_ea.loc[idx, "geometry"]
            area_km2  = geom_ea.area / 1e6
            gdw_to_data[gdw]     = (geom_ea, area_km2)
            gdw_to_geom4326[gdw] = geom_4326
            added += 1
        print(f"  {label} GPKG: {added} features added  (total so far: {len(gdw_to_data)})")

    # Map gww_id → data via gww_map
    result      = {}   # {gww_id: (geom_ea, area_km2, geom_4326)}
    missing_gdw = []
    for gww, rec in gww_map.items():
        gdw = str(rec.get("id", ""))
        if gdw in gdw_to_data:
            geom_ea, area_km2 = gdw_to_data[gdw]
            geom_4326         = gdw_to_geom4326[gdw]
            result[gww]       = (geom_ea, area_km2, geom_4326)
        else:
            missing_gdw.append((gww, gdw))

    if missing_gdw:
        print(f"  [WARN] {len(missing_gdw)} gww_ids not found in either GPKG:")
        for gww, gdw in missing_gdw[:10]:
            print(f"    gww={gww} gdw={gdw}")

    return result

def build_overlap_graph(ws):
    contained  = defaultdict(list)
    partial    = defaultdict(list)

    # Build GeoDataFrame for spatial join (use WGS84 for within check via sjoin)
    rows = [{"gww_id": k, "geometry": v[2]} for k, v in ws.items()]
    gdf  = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf_r = gdf.copy()

    joined = gpd.sjoin(
        gdf_r.rename(columns={"gww_id": "A"}),
        gdf_r.rename(columns={"gww_id": "B"}),
        how="left", predicate="intersects"
    )
    joined = joined[joined["A"] != joined["B"]][["A", "B"]].drop_duplicates()
    print(f"  Candidate intersecting pairs: {len(joined)}")

    for _, row in joined.iterrows():
        A, B = row["A"], row["B"]
        geom_A_4326 = ws[A][2]
        geom_B_4326 = ws[B][2]
        geom_A_ea   = ws[A][0]
        geom_B_ea   = ws[B][0]
        area_B      = ws[B][1]

        if geom_B_4326.within(geom_A_4326):
            contained[A].append((B, area_B))
        else:
            # Partial overlap: only assign if A is the LARGER watershed.
            # The larger one is the downstream/parent catchment — it subtracts
            # the overlap contribution. The smaller keeps its runoff unchanged.
            area_A = ws[A][1]
            if area_A <= area_B:
                continue
            inter      = geom_A_ea.intersection(geom_B_ea)
            inter_area = inter.area / 1e6
            if inter_area > 0.01:   # ignore slivers < 0.01 km²
                partial[A].append((B, inter_area))

    return contained, partial


def remove_circular(contained, partial, gww_map):
    circular = set()
    for A, pairs in list(contained.items()):
        for B, _ in pairs:
            if any(x == A for x, _ in contained.get(B, [])):
                circular.add((min(A, B), max(A, B)))

    if circular:
        print(f"  Duplicate polygon pairs (skipped):")
        for A, B in sorted(circular):
            gdw_A = gww_map.get(A, {}).get("id", "?")
            gdw_B = gww_map.get(B, {}).get("id", "?")
            print(f"    gww={A}(gdw={gdw_A}) ↔ gww={B}(gdw={gdw_B}) — same polygon")
        circular_ids = {x for pair in circular for x in pair}
        for A in list(contained.keys()):
            contained[A] = [(B, a) for B, a in contained[A] if B not in circular_ids]
            if not contained[A]:
                del contained[A]
        for A in list(partial.keys()):
            partial[A] = [(B, a) for B, a in partial[A] if B not in circular_ids]
            if not partial[A]:
                del partial[A]
        for A in circular_ids:
            contained.pop(A, None)
            partial.pop(A, None)

    return contained, partial


def topological_sort(all_ids, contained, partial):
    deps = {a: set() for a in all_ids}
    for A, pairs in contained.items():
        for B, _ in pairs:
            deps[A].add(B)
    for A, pairs in partial.items():
        for B, _ in pairs:
            deps[A].add(B)

    order, remaining = [], set(all_ids)
    while remaining:
        ready = [n for n in remaining if deps[n].issubset(set(order))]
        if not ready:
            print(f"  [WARN] Cycle detected among {len(remaining)} nodes — appending as-is")
            order.extend(sorted(remaining))
            break
        order.extend(sorted(ready))
        remaining -= set(ready)
    return order


def correct_all(ws, contained, partial, order, gww_map):
    from shapely.ops import unary_union

    corrected_cache = {}
    stats = {"corrected": 0, "unchanged": 0, "skipped": 0, "negative_area": 0,
             "missing_timesteps": 0}

    for gww in order:
        main_ts = load_timeseries(gww)
        if main_ts is None:
            stats["skipped"] += 1
            continue

        subs_c = contained.get(gww, [])   # [(B, area_B)]
        subs_p = partial.get(gww, [])     # [(C, inter_area)]

        if not subs_c and not subs_p:
            main_ts["area_km2"]     = round(ws[gww][1], 4)
            main_ts["net_area_km2"] = round(ws[gww][1], 4)
            corrected_cache[gww] = main_ts
            stats["unchanged"] += 1
            continue

        area_A     = ws[gww][1]
        geom_A_ea  = ws[gww][0]
        main_dates = main_ts["dates"]

        # ── Keep only MAXIMAL contained subs (drop subs nested in other subs)
        contained_ids = [B for B, _ in subs_c]
        maximal_c = []
        for B, area_B in subs_c:
            geom_B = ws[B][2]  # WGS84
            inside_another = any(
                other != B and geom_B.within(ws[other][2])
                for other in contained_ids
            )
            if not inside_another:
                maximal_c.append((B, area_B))

        # ── Geometric union of contained (equal-area geoms, clipped to A)
        contained_geoms_ea = [ws[B][0].intersection(geom_A_ea) for B, _ in maximal_c]
        contained_union = unary_union(contained_geoms_ea) if contained_geoms_ea else None

        # ── Partial pieces: (A∩C) minus contained union
        partial_pieces = []   # [(C, piece_geom_ea, piece_area_km2)]
        for C, _ in subs_p:
            piece = geom_A_ea.intersection(ws[C][0])
            if contained_union is not None:
                piece = piece.difference(contained_union)
            piece_area = piece.area / 1e6
            if piece_area > 0.01:
                partial_pieces.append((C, piece_area))

        # ── Net area from union of everything subtracted
        all_sub_geoms = list(contained_geoms_ea)
        for C, _ in subs_p:
            all_sub_geoms.append(geom_A_ea.intersection(ws[C][0]))
        total_union = unary_union(all_sub_geoms) if all_sub_geoms else None
        sub_union_area = (total_union.area / 1e6) if total_union is not None else 0.0
        net_area = area_A - sub_union_area

        if net_area <= 0:
            print(f"  [WARN] gww={gww}: net_area={net_area:.2f} km² ≤ 0 — skipping")
            main_ts["area_km2"]     = round(area_A, 4)
            main_ts["net_area_km2"] = round(area_A, 4)   # fallback: full area
            corrected_cache[gww] = main_ts
            stats["negative_area"] += 1
            continue

        new_ts = {k: v for k, v in main_ts.items()}
        new_ts["area_km2"]     = round(area_A, 4)
        new_ts["net_area_km2"] = round(net_area, 4)

        for metric in METRICS:
            main_vals = main_ts.get(metric, [])
            if not main_vals:
                continue

            # Pre-load and align ORIGINAL sub timeseries
            aligned_c = []
            for B, area_B in maximal_c:
                src = load_timeseries(B)     # ORIGINAL, not corrected
                if src is None:
                    continue
                aligned_c.append((align_to_dates(main_dates, src["dates"], src.get(metric, [])), area_B))

            aligned_p = []
            for C, piece_area in partial_pieces:
                src = load_timeseries(C)
                if src is None:
                    continue
                aligned_p.append((align_to_dates(main_dates, src["dates"], src.get(metric, [])), piece_area))

            # A sub with no value at timestep i contributes nothing to net_vol,
            # yet its area is already removed from net_area. That inflates the
            # corrected runoff for those timesteps. Detect and report it, and
            # attribute the missing sub's share of area back into net_area for
            # THAT timestep so the per-timestep normalization stays consistent.
            result_vals = []
            gww_missing = 0
            for i, mv in enumerate(main_vals):
                if mv is None:
                    result_vals.append(None)
                    continue

                net_vol = mv * area_A
                missing_area = 0.0
                for vals, area_B in aligned_c:
                    bv = vals[i]
                    if bv is not None:
                        net_vol -= bv * area_B
                    else:
                        missing_area += area_B
                        gww_missing += 1
                for vals, piece_area in aligned_p:
                    cv = vals[i]
                    if cv is not None:
                        net_vol -= cv * piece_area
                    else:
                        missing_area += piece_area
                        gww_missing += 1

                # Effective net area for this timestep: give back the area of
                # any sub that had no data (we couldn't subtract its volume,
                # so we must not subtract its area either)
                eff_net_area = net_area + missing_area
                if eff_net_area <= 0:
                    result_vals.append(None)
                    continue
                result_vals.append(round(net_vol / eff_net_area, 8))

            if gww_missing:
                stats["missing_timesteps"] += gww_missing
                print(f"  [WARN] gww={gww} metric={metric}: {gww_missing} "
                      f"sub-timestep values missing — area credited back per timestep")

            new_ts[metric] = result_vals

        corrected_cache[gww] = new_ts
        stats["corrected"] += 1

    return corrected_cache, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CORRECTED_DIR, exist_ok=True)

    print("[1] Loading reservoir.json mapping...")
    gww_map = load_mapping(RESERVOIR_JSON)
    print(f"  {len(gww_map)} reservoirs")

    print("\n[1b] Loading valid GDW_IDs from CSVs...")
    csv_gdw_ids = load_csv_gdw_ids()
    print(f"  {len(csv_gdw_ids)} unique GDW_IDs in CSVs")

    print("\n[2] Loading watershed geometries and areas from merged catchment GPKGs...")
    ws = load_watersheds_from_gpkg(gww_map, csv_gdw_ids)
    print(f"  {len(ws)} watersheds loaded with geometry + area")

    print("\n[3] Building overlap graph from GPKG geometries...")
    contained, partial = build_overlap_graph(ws)

    contained, partial = remove_circular(contained, partial, gww_map)
    print(f"  Contained-within pairs : {sum(len(v) for v in contained.values())}")
    print(f"  Partial-overlap pairs  : {sum(len(v) for v in partial.values())}")

    print("\n[4] Topological sort...")
    order = topological_sort(list(ws.keys()), contained, partial)
    print(f"  Order determined for {len(order)} reservoirs")

    print("\n[5] Applying corrections...")
    corrected_cache, stats = correct_all(ws, contained, partial, order, gww_map)
    print(f"  Corrected     : {stats['corrected']}")
    print(f"  Unchanged     : {stats['unchanged']}")
    print(f"  Skipped       : {stats['skipped']}")
    print(f"  Negative area : {stats['negative_area']}")
    print(f"  Missing sub-timesteps credited back : {stats['missing_timesteps']}")

    print(f"\n[6] Writing to {CORRECTED_DIR}/...")
    written = 0
    for gww, ts in corrected_cache.items():
        with open(os.path.join(CORRECTED_DIR, f"timeseries_{gww}.json"), "w") as f:
            json.dump(ts, f, separators=(",", ":"))
        written += 1

    copied = 0
    for fname in os.listdir(GEOJSON_DIR):
        if fname.startswith("timeseries_"):
            continue
        dst = os.path.join(CORRECTED_DIR, fname)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(GEOJSON_DIR, fname), dst)
            copied += 1

    print(f"  Timeseries written : {written}")
    print(f"  Other files copied : {copied}")

    if contained or partial:
        print(f"\n=== Corrected Reservoirs ===")
        for A in sorted(contained.keys() | partial.keys(),
                        key=lambda x: int(x) if x.isdigit() else float('inf')):
            gdw_A = gww_map.get(A, {}).get("id", "?")
            print(f"gww={A} gdw={gdw_A}")
            for B, area_B in contained.get(A, []):
                print(f"  contained : gww={B} gdw={gww_map.get(B,{}).get('id','?')}  area={area_B:.1f} km²")
            for C, inter_a in partial.get(A, []):
                print(f"  partial   : gww={C} gdw={gww_map.get(C,{}).get('id','?')}  intersection={inter_a:.1f} km²")

    print(f"\nDone. Output: {os.path.abspath(CORRECTED_DIR)}/")


if __name__ == "__main__":
    main()