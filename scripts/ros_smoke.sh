#!/usr/bin/env bash
# ROS runtime smoke test of the three executables on the demo map.
#
# Starts route_planner, osm_cloud and map_data_viewer on demo/buchlovice.mapdata in a private
# ROS domain, then checks the node list, the latched /grid cloud, one PlanRoute goal, the
# viewer's HTTP endpoints and that each process exits with code 0 on SIGINT.
# Exits non-zero on the first failed check (all checks still run).
#
# Usage (map_data and map_data_interfaces built and sourced):
#   bash scripts/ros_smoke.sh
# ROS_DOMAIN_ID and PORT (viewer) can be overridden via env.
set -u
set -m # job control: background jobs keep the default SIGINT instead of "ignored"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$((60 + RANDOM % 40))}"
PORT="${PORT:-$((5100 + RANDOM % 800))}"
MAP="$ROOT/demo/buchlovice.mapdata"
WORK="$(mktemp -d)"
LIB="$(ros2 pkg prefix map_data)/lib/map_data"
FAILED=0
PIDS=()

fail() {
	echo "FAIL: $*"
	FAILED=1
}

cleanup() {
	for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
	if [ "$FAILED" != 0 ]; then
		for log in "$WORK"/*.log; do
			echo "--- $(basename "$log")"
			tail -n 15 "$log"
		done
	fi
	rm -rf "$WORK"
}
trap cleanup EXIT

mkdir "$WORK/data" "$WORK/missions"
cp "$ROOT"/demo/buchlovice.* "$WORK/data/" # the viewer writes next to the maps

"$LIB/route_planner" --ros-args -p mapdata_file:="$MAP" -p mission_dir:="$WORK/missions" \
	>"$WORK/route_planner.log" 2>&1 &
RP=$!
"$LIB/osm_cloud" --ros-args -p mapdata_file:="$MAP" -p transform_mode:=auto \
	-p publish_intersections:=true >"$WORK/osm_cloud.log" 2>&1 &
OC=$!
"$LIB/map_data_viewer" --data-dir "$WORK/data" --port "$PORT" >"$WORK/viewer.log" 2>&1 &
VW=$!
PIDS=("$RP" "$OC" "$VW")

echo "== nodes (ROS_DOMAIN_ID=$ROS_DOMAIN_ID, viewer port $PORT)"
for _ in $(seq 1 40); do
	nodes="$(ros2 node list 2>/dev/null)"
	missing=0
	for n in /route_planner /osm_cloud /map_data_tracker; do
		grep -qx "$n" <<<"$nodes" || missing=1
	done
	[ "$missing" = 0 ] && break
	sleep 0.5
done
for n in /route_planner /osm_cloud /map_data_tracker; do
	grep -qx "$n" <<<"$nodes" && echo "ok $n" || fail "node $n is missing"
done

echo "== /grid"
width="$(timeout 20 ros2 topic echo --once /grid --field width 2>/dev/null | head -n 1)"
[ "${width:-0}" -gt 0 ] 2>/dev/null && echo "ok /grid has $width points" || fail "/grid is empty (width='$width')"

echo "== PlanRoute"
timeout 120 python3 - <<'EOF' || fail "PlanRoute goal"
import sys

import rclpy
from geographic_msgs.msg import GeoPoint
from map_data_interfaces.action import PlanRoute
from rclpy.action import ActionClient

rclpy.init()
node = rclpy.create_node("smoke_client")
client = ActionClient(node, PlanRoute, "/route_planner/plan_route")
if not client.wait_for_server(timeout_sec=20):
    sys.exit("no action server")
goal = PlanRoute.Goal()
goal.waypoints = [
    GeoPoint(latitude=49.0812857, longitude=17.3381287),
    GeoPoint(latitude=49.0814432, longitude=17.3370199),
]
goal.save_gpx = "-"
sent = client.send_goal_async(goal)
rclpy.spin_until_future_complete(node, sent, timeout_sec=60)
done = sent.result().get_result_async()
rclpy.spin_until_future_complete(node, done, timeout_sec=90)
res = done.result().result
print(f"success={res.success} length={res.length_m:.1f} m poses={len(res.route.poses)}")
sys.exit(0 if res.success and res.route.poses else f"planning failed: {res.reason} {res.message}")
EOF

echo "== viewer HTTP"
for url in /api/files "/api/mapdata?file=buchlovice.mapdata" /api/planner_defaults /api/tracker/settings; do
	code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT$url")"
	[ "$code" = 200 ] && echo "ok $url" || fail "$url -> HTTP $code"
done

echo "== SIGINT"
for entry in route_planner:$RP osm_cloud:$OC map_data_viewer:$VW; do
	name="${entry%%:*}" pid="${entry##*:}"
	kill -INT "$pid" 2>/dev/null
	for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
	if kill -0 "$pid" 2>/dev/null; then
		fail "$name still running 10 s after SIGINT"
		continue
	fi
	wait "$pid"
	code=$?
	[ "$code" = 0 ] && echo "ok $name exited 0" || fail "$name exited with code $code"
done

[ "$FAILED" = 0 ] && echo "ros smoke: OK" || echo "ros smoke: FAILED"
exit "$FAILED"
