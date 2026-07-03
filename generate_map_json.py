"""
generate_geojson.py
-------------------
Generates per-feature GeoJSON and timeseries files for the GBMIM atlas.

Procedure
---------
1. Scan storage dir → get authoritative GWW ids (784 files).
2. For each GWW id, find matching row in gpkg via GWW_reservoir_id column.
3a. If that row has a valid GDW_ID → use it as canonical feature id.
3b. If GDW_ID is null → parse GDW_bar_ids; use the first bar id found in a CSV.
4. Look up canonical id in reservoir CSV then barrier CSV for lat/lng + metadata.
5. Watershed polygon from the matched gpkg row's geometry.
6. Runoff timeseries: search runoff_reservoir dir then runoff_barrier dir.
7. Storage timeseries: keyed by GWW_reservoir_id.
8. Downstream path: search reservoir downstream dir then barrier dir.
9. Everything written under geojson/reservoir/.
"""

import argparse
import json
import os
import sys

import fiona
import pandas as pd
import geopandas as gpd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--gpkg",          default="Dataset/combined_reservoirs.gpkg")
    p.add_argument("--gpkg_layer",    default=None)
    p.add_argument("--gpkg_id_col",   default="GDW_ID")

    p.add_argument("--reservoirs_csv", default="Dataset/GBMIM_Reservoirs.csv")
    p.add_argument("--barriers_csv",   default="Dataset/GBMIM_Barriers.csv")

    p.add_argument("--reservoirs_downstream_dir", default="Dataset/Downstream/Reservoir")
    p.add_argument("--barriers_downstream_dir",   default="Dataset/Downstream/Barrier")

    p.add_argument("--runoff_reservoir_dir", default="Dataset/data/runoff_reservoir")
    p.add_argument("--runoff_barrier_dir",   default="Dataset/data/runoff_barrier")

    p.add_argument("--storage_dir", default="Dataset/Storage_timeseries")

    p.add_argument("--geojson_dir",   default="geojson")
    p.add_argument("--line_simplify", type=float, default=0.001)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_id(val) -> str | None:
    """Strip leading zeros; return None for null/empty."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        s = str(int(float(val))).lstrip("0")
    except (ValueError, TypeError):
        s = str(val).strip().lstrip("0")
    return s if s else None


def parse_bar_ids(raw) -> list[str]:
    """Parse GDW_bar_ids field (single or comma-separated) into list of norm ids."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    parts = str(raw).replace(";", ",").split(",")
    return [x for x in (norm_id(p.strip()) for p in parts) if x]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gpkg(gpkg_path: str, layer: str | None) -> gpd.GeoDataFrame:
    layers = fiona.listlayers(gpkg_path)
    layer = layer or layers[0]
    print(f"  Layer        : {layer}")
    gdf = gpd.read_file(gpkg_path, layer=layer).to_crs(epsg=4326)
    print(f"  Raw features : {len(gdf)}")
    return gdf


def scan_storage_dir(storage_dir: str) -> dict[str, str]:
    """Return {gww_id_stripped: filepath} for all Storage_*_apigen.csv files."""
    result = {}
    if not os.path.isdir(storage_dir):
        print(f"  [WARN] Storage dir not found: {storage_dir}")
        return result
    for fname in os.listdir(storage_dir):
        if not fname.startswith("Storage_") or not fname.endswith("_apigen.csv"):
            continue
        mid = fname[len("Storage_"):-len("_apigen.csv")]
        key = mid.lstrip("0") or "0"
        result[key] = os.path.join(storage_dir, fname)
    print(f"  Storage files found: {len(result)}")
    return result


def load_csv_lookup(csv_path: str, label: str) -> dict:
    """Return {norm_id: pd.Series} for each CSV row."""
    if not os.path.exists(csv_path):
        print(f"  [WARN] {label} CSV not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        print(f"  [WARN] No 'id' column in {csv_path}")
        return {}
    df["_id"] = df["id"].apply(norm_id)
    for col in ("lat", "lng"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return {row["_id"]: row for _, row in df.iterrows() if row["_id"]}


CSV_EXTRA = ["country", "Basin", "Areal_Impact (%)"]

def _col_key(col: str) -> str:
    return col.replace("(","").replace(")","").replace("%","").replace(" ","_").rstrip("_")


def load_downstream(dirs: list[str], fid: str) -> gpd.GeoDataFrame | None:
    for d in dirs:
        path = os.path.join(d, f"{fid}_downstream_path.gpkg")
        if os.path.exists(path):
            try:
                return gpd.read_file(path).to_crs(epsg=4326)
            except Exception as e:
                print(f"  [WARN] {path}: {e}")
    return None


RUNOFF_METRICS = ["Surface", "Sub_Surface", "Precip"]

def load_runoff(dirs: list[str], fid: str) -> dict | None:
    for d in dirs:
        path = os.path.join(d, f"GDWID_{fid}.txt")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [WARN] {path}: {e}")
            continue
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        out = {"dates": df["Date"].dt.strftime("%Y-%m").tolist()}
        for col in RUNOFF_METRICS:
            out[col] = (
                [None if pd.isna(v) else float(f"{v:.6g}") for v in df[col]]
                if col in df.columns else []
            )
        return out
    return None


def load_storage(filepath: str) -> dict | None:
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [WARN] {filepath}: {e}")
        return None
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    out = {"dates": df[date_col].dt.strftime("%Y-%m").tolist()}
    for col in df.columns:
        if col == date_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = [None if pd.isna(v) else float(f"{v:.6g}") for v in df[col]]
    return out


def geom_to_feature(geom, fid: str) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature",
                      "geometry": geom.__geo_interface__,
                      "properties": {"id": fid}}]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    res_dir = os.path.join(args.geojson_dir, "reservoir")
    os.makedirs(res_dir, exist_ok=True)
    print(f"Output:")
    print(f"  {args.geojson_dir}/reservoir.json")
    print(f"  {res_dir}/")
    if args.line_simplify > 0:
        print(f"  Downstream simplification: {args.line_simplify} deg")

    # ------------------------------------------------------------------
    # Step 1 — Scan storage directory (authoritative list of GWW ids)
    # ------------------------------------------------------------------
    print(f"\n[1] Scanning storage directory: {args.storage_dir}")
    storage_index = scan_storage_dir(args.storage_dir)  # {gww_id: filepath}

    # ------------------------------------------------------------------
    # Step 2 — Load geopackage (all rows, no filtering yet)
    # ------------------------------------------------------------------
    print(f"\n[2] Loading geopackage: {args.gpkg}")
    if not os.path.exists(args.gpkg):
        sys.exit(f"[ERROR] GeoPackage not found: {args.gpkg}")
    gdf = load_gpkg(args.gpkg, args.gpkg_layer)

    # Build GWW_reservoir_id → gpkg row lookup
    # For duplicate GWW ids, prefer the row with a valid GDW_ID
    gww_to_row: dict[str, pd.Series] = {}
    for _, row in gdf.iterrows():
        val = str(row.get("GWW_reservoir_id", "")).strip()
        if val in ("NO_MATCH", "", "nan"):
            continue
        gww = val.lstrip("0") or "0"
        existing = gww_to_row.get(gww)
        # Prefer row with non-null GDW_ID
        if existing is None:
            gww_to_row[gww] = row
        elif pd.isna(existing["GDW_ID"]) and pd.notna(row["GDW_ID"]):
            gww_to_row[gww] = row

    print(f"  Unique GWW ids in gpkg: {len(gww_to_row)}")

    # ------------------------------------------------------------------
    # Step 3 — Load metadata CSVs
    # ------------------------------------------------------------------
    print(f"\n[3] Loading metadata CSVs")
    res_lookup = load_csv_lookup(args.reservoirs_csv, "Reservoir")
    bar_lookup = load_csv_lookup(args.barriers_csv,   "Barrier")
    print(f"  Reservoir CSV: {len(res_lookup)} rows")
    print(f"  Barrier CSV  : {len(bar_lookup)} rows")

    # ------------------------------------------------------------------
    # Step 4 — Build coordinate records
    # ------------------------------------------------------------------
    print(f"\n[4] Building coordinate records")
    coord_records = []
    src_counts = {"gdw_id": 0, "bar_id": 0, "skipped": 0}

    for gww_id in sorted(storage_index.keys(), key=lambda x: int(x)):
        gpkg_row = gww_to_row.get(gww_id)
        if gpkg_row is None:
            print(f"  [WARN] GWW {gww_id} not found in geopackage — skipping")
            src_counts["skipped"] += 1
            continue

        # Determine canonical feature id
        gdw = norm_id(gpkg_row.get("GDW_ID"))

        if gdw:
            # 3a — valid GDW_ID
            fid = gdw
            csv_row = res_lookup.get(fid)
            if csv_row is None:
                csv_row = bar_lookup.get(fid)
            id_src = "gdw_id"
        else:
            # 3b — null GDW_ID: try GDW_bar_ids
            bar_ids = parse_bar_ids(gpkg_row.get("GDW_bar_ids"))
            fid = None
            csv_row = None
            for bid in bar_ids:
                if bid in res_lookup:
                    fid, csv_row = bid, res_lookup[bid]
                    break
                if bid in bar_lookup:
                    fid, csv_row = bid, bar_lookup[bid]
                    break
            if fid is None:
                print(f"  [WARN] GWW {gww_id}: null GDW_ID, no bar_id found in CSV — skipping")
                src_counts["skipped"] += 1
                continue
            id_src = "bar_id"

        # Coordinates: gpkg first, CSV as fallback
        lat = gpkg_row.get("LAT_DAM") or gpkg_row.get("LAT_RIV")
        lng = gpkg_row.get("LONG_DAM") or gpkg_row.get("LONG_RIV")
        if (not lat or pd.isna(lat) or not lng or pd.isna(lng)) and csv_row is not None:
            lat = csv_row.get("lat")
            lng = csv_row.get("lng")

        if pd.isna(lat) or pd.isna(lng):
            print(f"  [WARN] No coordinates for id={fid} (GWW {gww_id}) — skipping")
            src_counts["skipped"] += 1
            continue

        record: dict = {"id": fid, "lat": float(lat), "lng": float(lng)}

        if csv_row is not None:
            for col in CSV_EXTRA:
                if col in csv_row.index and not pd.isna(csv_row[col]):
                    record[_col_key(col)] = csv_row[col]

        # Store gww_id on record so per-feature loop can find storage file
        record["_gww_id"] = gww_id
        # Store gpkg geometry reference
        record["_geom"] = gpkg_row.get("geometry")

        coord_records.append(record)
        src_counts[id_src] += 1

    print(f"  Via GDW_ID   : {src_counts['gdw_id']}")
    print(f"  Via bar_id   : {src_counts['bar_id']}")
    print(f"  Skipped      : {src_counts['skipped']}")
    print(f"  Total records: {len(coord_records)}")

    # Write reservoir.json (strip internal keys before writing, but keep gww_id)
    public_records = [{k: v for k, v in r.items() if not k.startswith("_") or k == "_gww_id"}
                      for r in coord_records]
    # Rename _gww_id → gww_id in public output
    for rec in public_records:
        if "_gww_id" in rec:
            rec["gww_id"] = rec.pop("_gww_id")
    coords_path = os.path.join(args.geojson_dir, "reservoir.json")
    with open(coords_path, "w", encoding="utf-8") as f:
        json.dump(public_records, f, separators=(",", ":"))
    print(f"  Written: {coords_path}  ({os.path.getsize(coords_path)/1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Step 5-8 — Per-feature files
    # ------------------------------------------------------------------
    print(f"\n[5-8] Writing per-feature files → {res_dir}/")
    downstream_dirs = [args.reservoirs_downstream_dir, args.barriers_downstream_dir]
    runoff_dirs     = [args.runoff_reservoir_dir, args.runoff_barrier_dir]

    ws_ok = ws_skip = 0
    ds_ok = ds_miss = 0
    ro_ok = ro_miss = 0
    st_ok = st_miss = 0
    ws_bytes = ds_bytes = ro_bytes = st_bytes = 0

    for record in coord_records:
        fid    = record["id"]
        gww_id = record["_gww_id"]
        geom   = record["_geom"]

        # -- Watershed polygon --
        if geom is not None and not geom.is_empty:
            ws_path = os.path.join(res_dir, f"watershed_{gww_id}.geojson")
            with open(ws_path, "w", encoding="utf-8") as f:
                json.dump(geom_to_feature(geom, fid), f, separators=(",", ":"))
            ws_bytes += os.path.getsize(ws_path)
            ws_ok += 1
        else:
            ws_skip += 1

        # -- Downstream path --
        ds_gdf = load_downstream(downstream_dirs, fid)
        if ds_gdf is not None:
            if args.line_simplify > 0:
                ds_gdf = ds_gdf.copy()
                ds_gdf["geometry"] = ds_gdf["geometry"].simplify(
                    args.line_simplify, preserve_topology=True)
            ds_path = os.path.join(res_dir, f"downstream_{gww_id}.geojson")
            ds_gdf.to_file(ds_path, driver="GeoJSON")
            ds_bytes += os.path.getsize(ds_path)
            ds_ok += 1
        else:
            ds_miss += 1

        # -- Runoff timeseries --
        ro = load_runoff(runoff_dirs, fid)
        if ro is not None:
            ro_path = os.path.join(res_dir, f"timeseries_{gww_id}.json")
            with open(ro_path, "w", encoding="utf-8") as f:
                json.dump(ro, f, separators=(",", ":"))
            ro_bytes += os.path.getsize(ro_path)
            ro_ok += 1
        else:
            print(f"  [WARN] No runoff data for id={fid} (GWW {gww_id})")
            ro_miss += 1

        # -- Storage timeseries (keyed by GWW_reservoir_id) --
        st = load_storage(storage_index[gww_id])
        if st is not None:
            st_path = os.path.join(res_dir, f"storage_{gww_id}.json")
            with open(st_path, "w", encoding="utf-8") as f:
                json.dump(st, f, separators=(",", ":"))
            st_bytes += os.path.getsize(st_path)
            st_ok += 1
        else:
            st_miss += 1

    total_mb = (ws_bytes + ds_bytes + ro_bytes + st_bytes) / (1024 * 1024)
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Watershed polygons : {ws_ok} written,  {ws_skip} empty/missing")
    print(f"  Downstream paths   : {ds_ok} written,  {ds_miss} missing")
    print(f"  Runoff timeseries  : {ro_ok} written,  {ro_miss} missing")
    print(f"  Storage timeseries : {st_ok} written,  {st_miss} failed to parse")
    print(f"  Total output size  : {total_mb:.1f} MB")
    print(f"  Output directory   : {os.path.abspath(res_dir)}/")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()