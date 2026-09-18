# Moving a Shared Node: Finding & Scope

Investigation of why dragging a node that several ways share does not split it
into two independent nodes, and what it would take to make it do so.

Status: **fixed at merge time** in `apply_way_edits` (`map_data/annotations.py`), see
[What was implemented](#what-was-implemented); a moved end node is then joined to the way it
lands on, see [the follow-up](#follow-up-joining-instead-of-only-detaching). The store-level
proposal further down was not needed and is kept for the record.

---

## Symptom

Moving a node of one way in the viewer looks right on screen — that way's
geometry follows the drag, the other ways stay put — but the graph planner
plans as if nothing happened. Worse, the ways that share the node are dragged
along with it inside the graph, and the junction between them survives.

## Root cause

Node moves are recorded **per way** but resolved **per node**. The collision is
a documented shortcut, with a note sitting directly on it in
`map_data/viewer/helpers.py:436-438`:

```python
# ponytail: overrides are per way but the cache is per node, so a junction node
# moved differently in two ways keeps the last one; key the cache by (way, node) if that bites.
```

Two code paths diverge:

| Path | Function | Behaviour |
|------|----------|-----------|
| Geometry | `apply_node_position_overrides` (`helpers.py:716`) | Rebuilds `way.line` per way, from that way's overrides only. The viewer draws what you expect. |
| Topology | `edited_nodes_cache` (`helpers.py:410`) | Overlays overrides into `nodes_cache` keyed by node id alone. `way.nodes` is never touched. |

The graph planner reads `way.nodes` plus `nodes_cache`
(`map_data/pathsolver/graph_planner.py:145`, `:265-270`). It therefore sees
**one** node, at whichever way's override was written last.

### Reproduction

Node `252419265` in `robotour.mapdata` is shared by four footways. Moving it in
one of them:

```text
baseline: node 252419265, degree 5
after moving it in way 23315134 only:
  graph still has ONE node: degree 5, pos moved ~39.7 m
  ways still referencing it: [23315134, 43906829, 180998345, 229972129]
  -> the other three were dragged along in the graph; the junction is intact
```

### Live occurrence

The `robotour` annotation store already holds a collision — node `388382619`
overridden in two ways about 13 m apart:

```json
"43907854":  {"388382619": {"lat": 50.10523656, "lon": 14.42706249}},
"180216039": {"388382619": {"lat": 50.10535004, "lon": 14.42696058}}
```

The merged cache resolves it to the `180216039` position; the `43907854`
override is silently discarded.

The `robotour.exported` store holds the same collision, and there it is benign: `388382619`
is also deleted from way `43907854`, so after the merge only `180216039` uses it.

---

## What was implemented

The viewer was never wrong: every viewer endpoint resolves one way at a time from the raw map
and that way's own overrides. Only the full merge collapsed them. So the fix lives in the
merge alone, in the `node_position_overrides` block of `apply_way_edits`:

1. index which (original) ways use each node in the already deleted/split way lists;
2. for every override on a node that another way also uses, mint a fresh negative id and put
   the moved position in `nodes_cache` under it; the original node keeps its position (the
   cache the merge works on holds no moves, a node only one way uses is moved in place);
3. substitute the copy into (a copy of) the moved way's `nodes`, after its geometry is rebuilt;
4. recompute the crossroads when anything was detached.

Segments of a split way share their overrides and therefore one copy, so they stay joined.
A node moved in several ways gets one copy per way. Nothing is stored: no schema change, no
endpoint or frontend change, the viewer keeps addressing original node ids, and undoing the
move re-joins the junction. Copy ids are not stable across store edits; nothing persists them
(an exported `.mapdata` is self-consistent).

Tests: `test_moving_a_shared_node_in_one_way_detaches_that_way` and
`test_shared_node_moved_in_both_ways_honours_both` in `tests/test_annotations.py`.

### Follow-up: joining instead of only detaching

Detaching alone left the moved end loose: connectivity is by node id, and a move never
produces a shared id. What was wanted is that an end dragged onto another way connects there.
`join_ways` (end of `merge_annotations`, so both `apply_store` and the viewer's
`get_merged_mapdata` run it) does that at merge time, for moved end nodes and for drawn paths
alike:

- an end (moved, or of a drawn path) within `JOIN_DISTANCE_M` = 5 m of another way, and not
  already a node of one, gets the junction put before/after it in its node list. The junction
  is a node of the other way within `JOIN_NODE_M` = 1 m of the nearest point, else a new node
  inserted on that edge. A way of the same kind (footway/road) is preferred, since a
  footway-only plan never sees the roads;
- a drawn path gets a shared node wherever its centre line crosses another way, or another
  way ends on it;
- segments of one original way are never joined, so a detached split stays detached.

Only node lists change, no geometry, and the moved node stays where the user put it. A moved
end is recognised by its position being one the store recorded for that way, so nothing has to
be passed from `apply_way_edits`. The viewer has no snapping when a node is dragged, which is
why the reach is as generous as 5 m.

The same revision fixed three things in the detach itself: ways that move a shared node to the
same spot (`SAME_SPOT_DEG`, 0.5 m) keep sharing it; the first pass of `apply_way_edits` works on
a cache without the moves, so a way rebuilt there is not drawn through another way's move; and
a move recorded on a way that no longer uses the node is ignored.

Still open: only **end** nodes join. A moved interior node does not, deliberately: dragging a
whole way writes an override for every node, and a way dragged alongside another would be
joined to it at each of them. The viewer still does not signal that a move breaks or makes a
junction, and its way-edit view cannot show the junction (it resolves one way at a time).
`next_synthetic_node_id` looks at the store only, so a node added on an *exported* map can
take an id the export already uses.

---

## Side findings

1. **Annotation store location.** The viewer writes its store next to the
   `.mapdata` it loaded. In a colcon `symlink-install` workspace that resolves
   to `install/map_data/share/map_data/data/<stem>.annotations.json` — a real
   file, while the `.mapdata` beside it is a symlink back to `src/`. Anything
   loading the **src** path finds no store and plans on the raw OSM map. Worth
   deciding deliberately which path the `route_planner` node is given.
2. **Orphaned cache entries on export.** An exported `.mapdata` can carry
   `nodes_cache` entries for detached split nodes that no way references (a
   stale `-1` in `data/robotour.mapdata` came from an export taken while an
   earlier store was live). Harmless, but exports should drop unreferenced
   entries.

---

## Original proposal: detach-on-move in the store (not implemented)

The mechanism already exists for splits. A `detached_nodes` entry
(`{way_id, node_id, id}`) means "this way uses its own synthetic copy of that
node": `next_synthetic_node_id` mints the id, `edited_nodes_cache` materializes
its position, and `split_way` substitutes it into the segment's node list
(`helpers.py:648-651`). That is exactly "one node becomes two, in different
places" — it is simply unreachable from a move today.

**Do not** key the nodes cache by `(way, node)` as the ponytail comment
suggests. The planner assumes globally unique integer node ids throughout
(`get_points()`, `graph`, `parse_intersections`, A\*). Minting a new id keeps
that invariant, and the split path already proves it works end to end.

**Rule.** When a node is moved on a way and that node is referenced by two or
more ways in the merged map, mint a copy for *that* way and record the override
under the copy. The original id keeps its original position. If a second way
then moves the same node, it gets its own copy too — so a three-way junction
dragged apart in two of its ways becomes three nodes, and no override ever
collides.

---

## Scope

### 1. Store schema — `docs/dev/data_formats.md:139-164`

Add a discriminator to `detached_nodes` so move-detaches and split-detaches do
not cross-contaminate:

```json
{"way_id": 43907854, "node_id": 388382619, "id": -20, "reason": "move"}
```

Existing entries default to `"split"`, so old stores load unchanged.
`get_detached_node_ids` (`helpers.py:387`) grows a `reason` filter;
`next_synthetic_node_id` and `edited_nodes_cache` need no change — they already
walk the whole list.

*Small. ~30 lines plus docs.*

### 2. Shared-node detection — new helper

Needs "which ways reference node N" over the merged map. Nothing exposes this
today; `parse_intersections` (`map_data/map_data.py:507-520`) builds a similar
index internally and could be refactored, or add a standalone
`node_way_ids(md)`. The endpoint needs it before deciding whether to detach.

*Small, but it is a per-request scan of every way list unless cached alongside
`load_mapdata_cached`. Watch this on Stromovka-sized maps.*

### 3. `move_way_nodes` — `map_data/viewer/routes.py:2357-2411`

For each node in the body: if shared and not already move-detached for this
way, mint an id, append a `detached_nodes` entry, and write the override under
the **new** id. Return the id mapping in the response body (currently a bare
204) so the frontend can update its handle without a full reload.

*Medium. ~50 lines.*

### 4. Substitution in the merge — `map_data/annotations.py:132-146`

Remap `w.nodes` through the way's move-detach map, then apply overrides keyed by
the new ids.

Ordering is the delicate part. `deleted_nodes`, `added_nodes` and `split_ways`
are all keyed by *original* node ids, so substitution has to run **after** those
passes — in the `node_pos_store` block, not before it. And
`md.parse_intersections` is currently called at `annotations.py:127`, inside the
earlier block; it must move after substitution (or run twice), or the crossroads
list will still show the junction that was just broken. Way objects must be
copied, not mutated — the docstring at `annotations.py:65-74` guarantees that.

*Medium, and the highest-risk piece. ~40 lines, plus careful reading of the pass
ordering.*

### 5. Undo — `map_data/viewer/routes.py:2414-2440`

`undo_move_way_nodes` drops all overrides for a way; it must also drop that
way's `"reason": "move"` detach entries. Per-node undo does not exist today and
is now more clearly wanted, since each move is a topology change — worth adding
at the same time.

*Small. ~25 lines.*

### 6. Frontend — `draw_handlers.js:346-352`, `ui_handlers.js:299`

`get_way_nodes` (`routes.py:1556-1580`) reads ids straight off the resolved
`way.nodes`, so it returns the detached id for free once step 4 lands. The drag
handler holds `currentNodes[nodeIndex].id` in memory across drags, though —
after the first move it would address a node id that no longer belongs to the
way. Either consume the id mapping from step 3's response, or reload nodes after
a move (`_reloadWay` already runs for polygons; LineStrings deliberately skip it
for smoothness).

*Small to medium. ~40 lines of JS.*

### 7. Planner — no change

Confirmed: `_build_graph` derives everything from `way.nodes` plus
`nodes_cache`, and the annotation auto-stitching pass keys off negative **way**
ids (`graph_planner.py:198-201`), so a negative *node* id on a positive-id way
is not silently reconnected. Detached ends stay detached.

### 8. Tests — `tests/test_viewer_helpers.py`, `tests/test_annotations.py`, `tests/test_viewer_routes.py`

Existing coverage of `node_position_overrides` / `detached_nodes` is thin (4 / 4
/ 2 references). New cases:

- move a shared node → two graph nodes, junction gone, other ways unmoved;
- move a non-shared node → no new id (no id churn on every drag);
- move the same node in both ways → two copies, both honoured;
- undo restores the junction;
- old stores without `reason` still load.

The live `388382619` collision makes a good fixture.

*Medium.*

---

## Decisions to make before coding

- **Detach always, or only when shared?** Detaching unconditionally is simpler
  to reason about but mints a synthetic id on every drag of every node, and
  exported maps lose their OSM node ids wholesale. Gating on shared is
  preferred.
- **Endpoint nodes.** Two ways meeting end to end share a node the same way a
  mid-way junction does. Dragging one end apart disconnects the route there —
  correct, but the case most likely to surprise, so the viewer should probably
  signal when a move breaks a junction.
- **The reverse operation.** Nothing currently re-merges two nodes into one. If
  pulling a junction apart becomes routine, "snap these back together" will be
  wanted soon after. Not covered by the scope above.
