#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
link="${ROS_WORKSPACE:-$repo/../..}/src/map_data_interfaces"
target="$repo/map_data_interfaces"

if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    exit
fi
if [ -e "$link" ]; then
    echo "$link already exists and is not the map_data_interfaces link" >&2
    exit 1
fi
mkdir -p "$(dirname "$link")"
ln -s "$target" "$link"
