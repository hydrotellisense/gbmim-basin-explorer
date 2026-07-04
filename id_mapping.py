import json

RESERVOIR_JSON = "geojson/reservoir.json"
OUTPUT         = "Dataset/gww_gdw_mapping.csv"

with open(RESERVOIR_JSON) as f:
    records = json.load(f)

with open(OUTPUT, "w") as f:
    f.write("GWW_reservoir_id,GDW_ID\n")
    for r in sorted(records, key=lambda x: int(x["gww_id"]) if str(x.get("gww_id","")).isdigit() else float('inf')):
        gww = r.get("gww_id", "")
        gdw = r.get("id", "")
        f.write(f"{gww},{gdw}\n")

print(f"Written: {OUTPUT}  ({len(records)} rows)")