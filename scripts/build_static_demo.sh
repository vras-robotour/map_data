#!/usr/bin/env bash
# Build a static, read-only scrape of the map data viewer for GitHub Pages.
#
# Boots the real viewer against the committed demo dataset, dumps the rendered
# page and every JSON endpoint the frontend needs into $OUT, and rewrites the
# page so the JS runs in static-demo mode (window.__mapdataStaticBase).
#
# Usage: bash scripts/build_static_demo.sh
#   PORT (default 5017) and OUT (default _demo_out) can be overridden via env.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-5017}"
BASE="http://127.0.0.1:$PORT"
OUT="${OUT:-_demo_out}"
DEMO_FILE="buchlovice.mapdata"

rm -rf "$OUT"
mkdir -p "$OUT/api/way_nodes" "$OUT/static"

map_data_viewer --data-dir demo --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
    curl -sf "$BASE/" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "$BASE/" >/dev/null || { echo "viewer did not start" >&2; exit 1; }

# index.html: enable static-demo mode and make asset URLs relative
curl -sf "$BASE/" \
    | sed -e 's|</head>|<script>window.__mapdataStaticBase=".";window.__mapdataDemoFile="'"$DEMO_FILE"'";</script>\n</head>|' \
          -e 's|"/static/|"./static/|g' \
    > "$OUT/index.html"

curl -sf "$BASE/api/files"                            > "$OUT/api/files.json"
curl -sf "$BASE/api/mapdata?file=$DEMO_FILE"          > "$OUT/api/mapdata.json"
curl -sf "$BASE/api/annotations?file=$DEMO_FILE"      > "$OUT/api/annotations.json"
curl -sf "$BASE/api/planner_defaults"                 > "$OUT/api/planner_defaults.json"
curl -sf "$BASE/api/export?file=$DEMO_FILE"           > "$OUT/api/export.mapdata"
curl -sf "$BASE/api/export/geojson?file=$DEMO_FILE"   > "$OUT/api/export.geojson"

# Pre-bake per-way node lists for the 'N' inspect toggle
python3 - "$BASE" "$OUT" "$DEMO_FILE" <<'PY'
import json
import pathlib
import sys
import urllib.parse
import urllib.request

base, out, demo = sys.argv[1:4]
geojson = json.loads((pathlib.Path(out) / "api/mapdata.json").read_text())
way_ids = {
    str(f["properties"]["id"])
    for f in geojson["features"]
    if f["properties"].get("category") in ("road", "footway", "barrier")
}
for way_id in sorted(way_ids):
    url = f"{base}/api/way_nodes?file={urllib.parse.quote(demo)}&way_id={urllib.parse.quote(way_id)}"
    dest = pathlib.Path(out) / "api/way_nodes" / f"{way_id.replace(':', '_')}.json"
    dest.write_bytes(urllib.request.urlopen(url).read())
print(f"scraped {len(way_ids)} way_nodes files")
PY

cp -r map_data/viewer/static/* "$OUT/static/"

echo "static demo written to $OUT/"
