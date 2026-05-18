# Architecture Overview

This page describes the internal structure of the `map_data` package. The system is organised into four main layers: **data ingestion**, **representation**, **planning**, and **visualisation**.

---

## Component diagram

<div class="arch-diagram">
<style>
/* ── Semantic tokens (scoped; reference --cat-* from catppuccin.css) ── */
.arch-diagram {
  --d-ink:          var(--cat-text);
  --d-ink-3:        var(--cat-subtext0);
  --d-ink-4:        var(--cat-overlay1);
  --d-ink-faint:    var(--cat-overlay0);
  --d-box-bg:       var(--cat-base);
  --d-box-bg-soft:  var(--cat-mantle);
  --d-box-edge:     var(--cat-surface1);
  --d-dark-bg:      var(--cat-text);
  --d-dark-edge:    var(--cat-text);
  --d-dark-text:    var(--cat-base);
  --d-dark-sub:     var(--cat-overlay1);
  --d-dark-id:      var(--cat-overlay0);
  --d-input-bg:     var(--cat-surface0);
  --d-input-edge:   var(--cat-surface1);
  --d-flow-main:    var(--cat-overlay1);
  --d-flow-md:      var(--cat-text);
  --d-data-rl:      var(--cat-blue);
  --d-data-fl:      var(--cat-green);
  --d-data-bl:      var(--cat-peach);
  --d-data-cl:      var(--cat-mauve);
  --d-data-nc:      var(--cat-teal);
  --d-accent:       var(--cat-mauve);
  --d-node-shadow:  rgba(76,79,105,0.10);
  --d-group-fill:   rgba(124,127,147,0.06);
}
[data-md-color-scheme="slate"] .arch-diagram {
  --d-dark-bg:      var(--cat-crust);
  --d-dark-edge:    var(--cat-surface0);
  --d-dark-text:    var(--cat-text);
  --d-dark-sub:     var(--cat-subtext0);
  --d-dark-id:      var(--cat-overlay1);
  --d-box-bg:       var(--cat-mantle);
  --d-box-bg-soft:  var(--cat-base);
  --d-input-bg:     var(--cat-surface0);
  --d-input-edge:   var(--cat-surface1);
  --d-node-shadow:  rgba(0,0,0,0.45);
  --d-group-fill:   rgba(127,132,156,0.06);
}

/*── SVG flow lines ──*/
.arch-diagram .flow { fill: none; stroke-linecap: round; stroke-linejoin: round }
.arch-diagram .flow-main { stroke: var(--d-flow-main) }
.arch-diagram .flow-md   { stroke: var(--d-flow-md)   }
.arch-diagram .flow-rl   { stroke: var(--d-data-rl)   }
.arch-diagram .flow-fl   { stroke: var(--d-data-fl)   }
.arch-diagram .flow-bl   { stroke: var(--d-data-bl)   }
.arch-diagram .flow-cl   { stroke: var(--d-data-cl)   }
.arch-diagram .flow-nc   { stroke: var(--d-data-nc)   }

.arch-diagram .head-main { fill: var(--d-flow-main) }
.arch-diagram .head-md   { fill: var(--d-flow-md)   }
.arch-diagram .head-rl   { fill: var(--d-data-rl)   }
.arch-diagram .head-fl   { fill: var(--d-data-fl)   }
.arch-diagram .head-bl   { fill: var(--d-data-bl)   }
.arch-diagram .head-cl   { fill: var(--d-data-cl)   }
.arch-diagram .head-nc   { fill: var(--d-data-nc)   }

.arch-diagram .dot-main  { fill: var(--d-flow-main) }
.arch-diagram .dot-md    { fill: var(--d-flow-md)   }
.arch-diagram .dot-rl    { fill: var(--d-data-rl)   }
.arch-diagram .dot-fl    { fill: var(--d-data-fl)   }
.arch-diagram .dot-bl    { fill: var(--d-data-bl)   }
.arch-diagram .dot-cl    { fill: var(--d-data-cl)   }
.arch-diagram .dot-nc    { fill: var(--d-data-nc)   }

.arch-diagram .bar-rl    { fill: var(--d-data-rl)   }
.arch-diagram .bar-fl    { fill: var(--d-data-fl)   }
.arch-diagram .bar-bl    { fill: var(--d-data-bl)   }
.arch-diagram .bar-cl    { fill: var(--d-data-cl)   }
.arch-diagram .bar-nc    { fill: var(--d-data-nc)   }

/*── Groups ──*/
.arch-diagram .group-rect {
  fill: var(--d-group-fill);
  stroke: var(--cat-surface1);
  stroke-width: 1;
  stroke-dasharray: 4 4;
}
.arch-diagram .group-tag-bg {
  fill: var(--d-box-bg);
  stroke: var(--d-ink-3);
  stroke-width: 1.25;
}
.arch-diagram .group-tag-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px; font-weight: 600;
  fill: var(--d-accent);
  letter-spacing: 0.05em;
}
.arch-diagram .group-tag-text {
  font-family: 'Inter', sans-serif;
  font-size: 13.5px; font-weight: 700;
  letter-spacing: 0.22em;
  fill: var(--d-ink);
}

/*── Boxes ──*/
.arch-diagram .box-shadow     { fill: var(--d-node-shadow) }
.arch-diagram .box-rect       { fill: var(--d-box-bg);      stroke: var(--d-box-edge);   stroke-width: 1 }
.arch-diagram .box-rect-input { fill: var(--d-input-bg);    stroke: var(--d-input-edge); stroke-width: 1 }
.arch-diagram .box-rect-dark  { fill: var(--d-dark-bg);     stroke: var(--d-dark-edge);  stroke-width: 1 }
.arch-diagram .box-rect-sidecar { fill: var(--d-box-bg-soft); stroke: var(--d-box-edge); stroke-width: 1 }
.arch-diagram .box-sidecar-dashed {
  fill: none; stroke: var(--d-ink-4);
  stroke-width: 1; stroke-dasharray: 4 3;
}

/*── Labels ──*/
.arch-diagram .label-title {
  text-anchor: middle;
  font-family: 'Inter', sans-serif;
  font-size: 18px; font-weight: 600;
  fill: var(--d-ink);
}
.arch-diagram .label-title-mono {
  text-anchor: middle;
  font-family: 'JetBrains Mono', monospace;
  font-size: 16.5px; font-weight: 500;
  fill: var(--d-ink);
}
.arch-diagram .label-sub {
  text-anchor: middle;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px; font-weight: 500;
  fill: var(--d-ink-3);
}
.arch-diagram .label-title.dark,
.arch-diagram .label-title-mono.dark { fill: var(--d-dark-text) }
.arch-diagram .label-sub.dark        { fill: var(--d-dark-sub)  }
.arch-diagram .id-chip {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; font-weight: 500;
  letter-spacing: 0.14em;
  fill: var(--d-ink-faint);
}
.arch-diagram .id-chip.dark { fill: var(--d-dark-id) }
.arch-diagram .cross-mark   { stroke: var(--d-ink-faint); stroke-width: 1; opacity: 0.7 }

.arch-diagram svg.diagram { display: block; width: 100%; height: auto }
</style>

<svg class="diagram" viewBox="0 0 1400 1160" xmlns="http://www.w3.org/2000/svg" aria-label="MapData architecture diagram"></svg>

<script>
(function(){
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.currentScript.closest('.arch-diagram').querySelector('svg.diagram');
  function el(tag, attrs, text){
    const n = document.createElementNS(NS, tag);
    for(const k in attrs){ n.setAttribute(k, attrs[k]); }
    if(text != null) n.textContent = text;
    return n;
  }
  const defs = el('defs');
  function head(id, cls){
    const m = el('marker',{id,viewBox:'0 0 10 10',refX:'8.5',refY:'5',
      markerUnits:'userSpaceOnUse',markerWidth:'9',markerHeight:'9',orient:'auto-start-reverse'});
    m.appendChild(el('path',{d:'M 0 1 L 9 5 L 0 9 Z',class:cls}));
    defs.appendChild(m);
  }
  ['main','md','rl','fl','bl','cl','nc'].forEach(c=>head(`h-${c}`,`head-${c}`));
  svg.appendChild(defs);

  const boxes = {
    GPX:{x:478, y:72,  w:184,h:70,  title:'GPX / YAML file',  mono:true,                       kind:'input'},
    OA :{x:738, y:72,  w:184,h:70,  title:'Overpass API',                                        kind:'input'},
    MD :{x:540, y:244, w:320,h:104, title:'MapData',           sub:'run_queries · run_parse',    kind:'dark'},
    RL :{x:100, y:474, w:200,h:84,  title:'roads',             mono:true,accent:'rl'},
    FL :{x:350, y:474, w:200,h:84,  title:'footways',          mono:true,accent:'fl'},
    BL :{x:600, y:474, w:200,h:84,  title:'barriers',          mono:true,accent:'bl'},
    CL :{x:850, y:474, w:200,h:84,  title:'crossroads',        mono:true,accent:'cl'},
    NC :{x:1100,y:474, w:200,h:84,  title:'nodes cache',       mono:true,accent:'nc'},
    GP :{x:120, y:744, w:240,h:116, title:'GraphPlanner',      sub:'Graph A*'},
    RP :{x:392, y:744, w:240,h:116, title:'ReplanPath',        sub:'Grid A* / RRT*'},
    VW :{x:760, y:744, w:240,h:116, title:'Viewer',            sub:'Flask + Leaflet'},
    ANN:{x:1032,y:744, w:240,h:116, title:'annotations sidecar',sub:'.annotations.json',kind:'sidecar'},
    ROS:{x:430, y:990, w:540,h:116, title:'ROS2 nodes',
         sub:'osm_cloud · TrackerNode · create_mapdata',kind:'dark'},
  };
  const topAt=(b,i,n)=>{const p=22,u=b.w-2*p;return{x:b.x+p+(n<=1?u/2:u*i/(n-1)),y:b.y}};
  const bot  =b=>({x:b.x+b.w/2,y:b.y+b.h});

  const groups=[
    {x:500, y:200,w:400, h:172,label:'data ingestion'},
    {x:60,  y:422,w:1280,h:172,label:'representation'},
    {x:80,  y:682,w:592, h:222,label:'planning'},
    {x:720, y:682,w:592, h:222,label:'visualisation',tagSide:'right'},
  ];
  for(const g of groups)
    svg.appendChild(el('rect',{x:g.x,y:g.y,width:g.w,height:g.h,rx:6,ry:6,class:'group-rect'}));

  function vBezier(s,t){
    const dy=t.y-s.y,c1y=s.y+dy*0.55,c2y=t.y-dy*0.55;
    return `M ${s.x} ${s.y} C ${s.x} ${c1y}, ${t.x} ${c2y}, ${t.x} ${t.y}`;
  }
  function flow(d,cls,marker,opacity=1,dash=null,width=2.2,mstart=null){
    const a={d,class:`flow ${cls}`,'stroke-width':width,opacity};
    if(marker) a['marker-end']=`url(#${marker})`;
    if(mstart) a['marker-start']=`url(#${mstart})`;
    if(dash)   a['stroke-dasharray']=dash;
    svg.appendChild(el('path',a));
  }
  function dot(x,y,cls,r=2.5){ svg.appendChild(el('circle',{cx:x,cy:y,r,class:cls})) }

  // sources -> MapData
  {
    const md=boxes.MD;
    const e1={x:md.x+md.w*0.30,y:md.y},e2={x:md.x+md.w*0.70,y:md.y};
    flow(vBezier(bot(boxes.GPX),e1),'flow-main','h-main',0.9,null,2.4);
    flow(vBezier(bot(boxes.OA), e2),'flow-main','h-main',0.9,null,2.4);
    dot(bot(boxes.GPX).x,bot(boxes.GPX).y,'dot-main');
    dot(bot(boxes.OA ).x,bot(boxes.OA ).y,'dot-main');
  }

  // MapData -> representation bus
  {
    const md=boxes.MD,startX=md.x+md.w/2,startY=md.y+md.h,busY=408;
    flow(`M ${startX} ${startY} L ${startX} ${busY}`,'flow-md',null,1,null,2.4);
    const reps=['RL','FL','BL','CL','NC'],xs=reps.map(k=>boxes[k].x+boxes[k].w/2);
    flow(`M ${Math.min(...xs,startX)} ${busY} L ${Math.max(...xs,startX)} ${busY}`,'flow-md',null,1,null,2.4);
    for(const k of reps){
      const b=boxes[k],tx=b.x+b.w/2;
      flow(`M ${tx} ${busY} L ${tx} ${b.y-1}`,'flow-md','h-md',1,null,2.4);
    }
  }

  function orthoPath(pts,r=9){
    if(pts.length<2) return '';
    let d=`M ${pts[0][0]} ${pts[0][1]}`;
    for(let i=1;i<pts.length-1;i++){
      const[px,py]=pts[i-1],[cx,cy]=pts[i],[nx,ny]=pts[i+1];
      const dx1=Math.sign(cx-px),dy1=Math.sign(cy-py),dx2=Math.sign(nx-cx),dy2=Math.sign(ny-cy);
      d+=` L ${cx-dx1*r} ${cy-dy1*r} Q ${cx} ${cy} ${cx+dx2*r} ${cy+dy2*r}`;
    }
    const l=pts[pts.length-1];
    return d+` L ${l[0]} ${l[1]}`;
  }

  // representation -> consumers
  const repTargets={RL:['GP','RP','VW'],FL:['GP','RP','VW'],BL:['RP','VW'],CL:['VW'],NC:['GP','RP']};
  const targetInputs={GP:['RL','FL','NC'],RP:['RL','FL','BL','NC'],VW:['RL','FL','BL','CL']};
  const laneY={RL:576,FL:592,BL:608,CL:624,NC:640};
  for(const src of Object.keys(repTargets)){
    const b=boxes[src],cls=src.toLowerCase(),sx=b.x+b.w/2,sy=b.y+b.h,lane=laneY[src];
    dot(sx,sy,`dot-${cls}`);
    for(const tgtKey of repTargets[src]){
      const tgt=boxes[tgtKey],order=targetInputs[tgtKey],idx=order.indexOf(src),entry=topAt(tgt,idx,order.length);
      if(src==='CL'){ flow(`M ${sx} ${sy} L ${sx} ${tgt.y-1}`,`flow-${cls}`,`h-${cls}`,0.95,null,2.2); continue; }
      flow(orthoPath([[sx,sy],[sx,lane],[entry.x,lane],[entry.x,entry.y-1]],9),`flow-${cls}`,`h-${cls}`,0.95,null,2.2);
    }
  }

  // Viewer <-> annotations
  {
    const vw=boxes.VW,an=boxes.ANN,y=vw.y+vw.h*0.5;
    flow(`M ${vw.x+vw.w} ${y} L ${an.x} ${y}`,'flow-main','h-main',1,null,2.2,'h-main');
  }

  // GP & RP -> ROS2 ; MD -> ROS2 (right detour)
  {
    const ros=boxes.ROS;
    const eGP={x:ros.x+ros.w*0.18,y:ros.y},eRP={x:ros.x+ros.w*0.36,y:ros.y},eMD={x:ros.x+ros.w*0.78,y:ros.y};
    const gpB=bot(boxes.GP),rpB=bot(boxes.RP);
    flow(orthoPath([[gpB.x,gpB.y],[gpB.x,940],[eGP.x,940],[eGP.x,eGP.y-1]],9),'flow-main','h-main',0.95,null,2.4);
    flow(orthoPath([[rpB.x,rpB.y],[rpB.x,900],[eRP.x,900],[eRP.x,eRP.y-1]],9),'flow-main','h-main',0.95,null,2.4);
    dot(gpB.x,gpB.y,'dot-main'); dot(rpB.x,rpB.y,'dot-main');
    const md=boxes.MD,s={x:md.x+md.w,y:md.y+md.h*0.55},r=10;
    const d=[`M ${s.x} ${s.y}`,`L ${1360-r} ${s.y}`,`Q 1360 ${s.y} 1360 ${s.y+r}`,
             `L 1360 ${940-r}`,`Q 1360 940 ${1360-r} 940`,`L ${eMD.x+r} 940`,
             `Q ${eMD.x} 940 ${eMD.x} ${940+r}`,`L ${eMD.x} ${ros.y}`].join(' ');
    flow(d,'flow-md','h-md',0.8,'5 4',2.0); dot(s.x,s.y,'dot-md');
  }

  // group labels
  for(let i=0;i<groups.length;i++){
    const g=groups[i],num=String(i+1).padStart(2,'0'),label=g.label.toUpperCase();
    const tagW=52+label.length*10.2,tagH=28,tagX=(g.tagSide==='right')?(g.x+g.w-tagW-18):(g.x+18),tagY=g.y-tagH/2;
    svg.appendChild(el('rect',{x:tagX,y:tagY,width:tagW,height:tagH,rx:14,ry:14,class:'group-tag-bg'}));
    svg.appendChild(el('text',{x:tagX+14,y:tagY+tagH/2+4,class:'group-tag-num'},num));
    svg.appendChild(el('line',{x1:tagX+26,y1:tagY+7,x2:tagX+26,y2:tagY+tagH-7,stroke:'currentColor','stroke-width':'1',opacity:'0.25'}));
    svg.appendChild(el('text',{x:tagX+38,y:tagY+tagH/2+4.5,class:'group-tag-text'},label));
  }

  // boxes
  function renderBox(b){
    const g=el('g');
    g.appendChild(el('rect',{x:b.x,y:b.y+2,width:b.w,height:b.h,rx:5,ry:5,class:'box-shadow',opacity:0.6}));
    let rc='box-rect',dark=false;
    if(b.kind==='input') rc='box-rect-input';
    else if(b.kind==='dark'){rc='box-rect-dark';dark=true;}
    else if(b.kind==='sidecar') rc='box-rect-sidecar';
    g.appendChild(el('rect',{x:b.x,y:b.y,width:b.w,height:b.h,rx:5,ry:5,class:rc}));
    if(b.accent) g.appendChild(el('rect',{x:b.x,y:b.y,width:4,height:b.h,rx:2,ry:2,class:`bar-${b.accent}`}));
    if(b.kind==='sidecar') g.appendChild(el('rect',{x:b.x+.5,y:b.y+.5,width:b.w-1,height:b.h-1,rx:5,ry:5,class:'box-sidecar-dashed'}));
    const cx=b.x+b.w/2,tc=(b.mono?'label-title-mono':'label-title')+(dark?' dark':'');
    if(b.sub){ g.appendChild(el('text',{x:cx,y:b.y+b.h/2-2,  class:tc},b.title));
               g.appendChild(el('text',{x:cx,y:b.y+b.h/2+20, class:`label-sub${dark?' dark':''}`},b.sub)); }
    else       g.appendChild(el('text',{x:cx,y:b.y+b.h/2+6,  class:tc},b.title));
    svg.appendChild(g);
  }
  for(const id in boxes) renderBox(boxes[id]);

  // corner marks
  [[20,20],[1380,20],[20,1140],[1380,1140]].forEach(([x,y])=>{
    const l=6;
    svg.appendChild(el('line',{x1:x-l,y1:y,x2:x+l,y2:y,class:'cross-mark'}));
    svg.appendChild(el('line',{x1:x,y1:y-l,x2:x,y2:y+l,class:'cross-mark'}));
  });
})();
</script>
</div>

---

## MapData

`MapData` (defined in `map_data/map_data.py`) is the central data class. It accepts either a path to a `.gpx` (or `.yaml`) file or a pre-converted UTM coordinate array, queries the Overpass API for all OSM features inside the derived bounding box, and parses the responses into categorised `Way` lists.

**Overpass queries.** Three concurrent HTTP requests are fired in a thread pool:

- ways query — all OSM ways (roads, footways, barriers, buildings, etc.) plus their constituent nodes
- relations query — multipolygon relations that reference the above ways
- nodes query — all standalone point features (obstacle nodes)

The bounding box is the convex hull of the input waypoints expanded by `osm_margin + reserve_margin` metres on all sides (defaults: 100 m + 50 m).

**Core attributes after parsing:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `roads_list` | `list[Way]` | Vehicle-intended highway ways, buffered to 7 m wide polygons |
| `footways_list` | `list[Way]` | Pedestrian footways, buffered to 3 m wide polygons |
| `barriers_list` | `list[Way]` | Physical barriers, buildings, water, and obstacle nodes |
| `crossroads_list` | `list[Way]` | Footway intersection points (buffered 1.5 m circles) |
| `nodes_cache` | `dict[int, dict]` | Mapping of OSM node ID to `{lat, lon, tags}` |
| `zone_number` | `int` | UTM zone number inferred from the input waypoints |
| `zone_letter` | `str` | UTM zone letter inferred from the input waypoints |
| `waypoints` | `np.ndarray` | `(N, 2)` UTM easting/northing array of the input waypoints |
| `min_x/max_x/min_y/max_y` | `float` | UTM bounding box including margins |

**OSM caching.** After a successful Overpass query, the raw responses are saved to a `.osm_cache.json` sidecar file next to the source `.gpx`/`.yaml` file. On the next load, if the file exists and the stored bounding box matches the current one (within 1 × 10⁻⁶ °, ≈ 11 cm), the cache is used and the network round-trip is skipped. The cache is invalidated automatically whenever the bounding box changes.

**Serialisation.** `MapData.save()` writes a JSON file (`.mapdata` extension) using `json.dump`. `MapData.load(path)` reads it back. Support for the legacy pickle format has been removed for security reasons; legacy files must be re-parsed from the source GPX/YAML.

---

## Way objects

Every OSM feature is stored as a `Way` instance (`map_data/utils/way.py`). A `Way` holds:

- `id` — OSM way/node ID (positive integer) or a synthetic negative integer for merged multipolygon segments or viewer-drawn annotations
- `is_area` — `True` if the geometry is a closed polygon
- `nodes` — ordered list of OSM node IDs
- `tags` — dict of OSM tag key-value pairs
- `line` — Shapely geometry (`Polygon` for closed/buffered features, `LineString` for open features)
- `in_out` — `"outer"` or `"inner"` for multipolygon relation members

All geometry is stored in UTM coordinates (same zone as the input waypoints). Ways are buffered during parsing: roads receive a 7 m half-width buffer and footways a 3 m half-width buffer, converting `LineString` geometries to `Polygon`.

---

## Pathsolvers

The package provides two independent path-planning back-ends.

### GraphPlanner

`map_data/pathsolver/graph_planner.py`

Builds an undirected weighted graph directly from the OSM way network (`roads_list` and/or `footways_list`). Edge weights are Euclidean distances in metres. Planning uses A* with a straight-line distance heuristic.

Waypoints passed to `GraphPlanner.plan()` are first snapped to the nearest graph edge using an STRtree spatial index. Viewer-drawn annotation paths (negative-ID ways) are spliced into the graph by projecting their endpoints onto the nearest existing OSM edge and inserting synthetic junction nodes at the projection points.

This planner is well-suited for route planning on well-mapped pedestrian or road networks where staying on designated paths is required.

### ReplanPath

`map_data/pathsolver/replan.py`

Orchestrates cost-grid planning. Grid construction, smoothing, and visualization are handled by dedicated sub-modules:

| Module | Class / function | Responsibility |
|--------|-----------------|----------------|
| `pathsolver/grid_constructor.py` | `PathGrid` | Builds the cost raster at `cell_size` metre resolution; assigns per-cell costs from highway type, surface, and barrier geometry |
| `pathsolver/grid_astar.py` | `grid_astar` | Fast, optimal A* search on the discrete grid |
| `pathsolver/rrt_star.py` | `RRTStar` | Sampling-based planner; produces smoother paths in cluttered environments |
| `pathsolver/smoothing.py` | `smooth_path` | Gradient-descent path smoothing with optional collision checking |
| `pathsolver/visualizer.py` | `visualize_replan` | Matplotlib debug visualization of the grid, obstacles, and planned path |

Each `PathGrid` cell cost is determined by:

- the OSM highway type and surface material of the nearest way within `max_path_dist` metres (configured via `highway_costs` and `surface_costs` in `planner_defaults.yaml`)
- a fixed `default_off_path_cost` for cells not covered by any way
- barrier polygons inflated by `inflate_obstacles` metres, set to cost 1.0 (impassable)

Key parameters now exposed in `planner_defaults.yaml`: `grid_cost_weight`, `obstacle_radius`, and `buffer_widths` (per road type).

---

## Viewer

The viewer (`map_data/viewer/`) is a single-page web application consisting of:

- **Flask back-end** (`app.py`, `routes.py`) — serves GeoJSON representations of the `MapData` contents, handles annotation CRUD operations, and exposes a `/export` endpoint that writes a human-readable JSON export of the annotated map
- **Leaflet front-end** — renders roads, footways, and barriers as coloured polygons on an OpenStreetMap tile layer; provides drawing tools for obstacle annotations and path annotations

**Sidecar annotation files.** All viewer edits are persisted alongside the `.mapdata` file as `<stem>.annotations.json`. This design keeps the binary/JSON map data immutable while allowing iterative annotation without re-running the Overpass query. The viewer merges the sidecar at load time to produce the rendered view.

---

## ROS2 nodes

| Node | File | Purpose |
|------|------|---------|
| `create_mapdata` | `map_data/create_mapdata.py` | CLI node: reads a GPX/YAML file, runs `MapData.run_all()`, writes the `.mapdata` file |
| `osm_cloud` | `map_data/osm_cloud.py` | ROS2 node: publishes the parsed OSM features as `sensor_msgs/PointCloud2` messages for use in Nav2 or custom navigation stacks |
| `TrackerNode` | `map_data/viewer/ros_node.py` | ROS2 node embedded in the viewer: subscribes to robot pose and streams it to the Leaflet front-end via WebSocket for live position display |

---

## Data flow

A typical session follows this sequence:

1. **Create `.mapdata`** — run `create_mapdata` with a GPX waypoint file. The node queries Overpass, parses the OSM data, and writes `<name>.mapdata` to disk. Or download and parse data inside the viewer.
2. **Visualize data** — run the viewer and load parsed data in the browser. If the data were parsed through the viewer they will be shown automatically.
3. **Annotate** — use the viewer drawing tools to mark obstacles, draw alternative path segments, delete or hide erroneous ways, adjust node positions, and override OSM tags. Changes are saved automatically to `<name>.annotations.json`.
4. **Export** — click the Export button to produce `<name>.exported.mapdata`, a human-readable JSON snapshot of the annotated map suitable for downstream processing.
5. **Plan path** — instantiate `GraphPlanner` or `ReplanPath` with the loaded `MapData` object and call `plan()` with the desired waypoints.
