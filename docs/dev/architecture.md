# Architecture overview

The `map_data` package has four layers: **data ingestion**, **representation**, **planning** and **visualization**.

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
    GPX:{x:478, y:72,  w:184,h:70,  title:'GPX file',  mono:true,                       kind:'input'},
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
    ROS:{x:430, y:990, w:540,h:116, title:'ROS 2 nodes',
         sub:'osm_cloud · route_planner · TrackerNode',kind:'dark'},
  };
  const topAt=(b,i,n)=>{const p=22,u=b.w-2*p;return{x:b.x+p+(n<=1?u/2:u*i/(n-1)),y:b.y}};
  const bot  =b=>({x:b.x+b.w/2,y:b.y+b.h});

  const groups=[
    {x:500, y:200,w:400, h:172,label:'data ingestion'},
    {x:60,  y:422,w:1280,h:172,label:'representation'},
    {x:80,  y:682,w:592, h:222,label:'planning'},
    {x:720, y:682,w:592, h:222,label:'visualization',tagSide:'right'},
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

  // GP & RP -> ROS 2 ; MD -> ROS 2 (right detour)
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

`MapData` (`map_data/map_data.py`) is the central data class. It takes a `.gpx` file (waypoints or a track) or a UTM coordinate array, downloads the OSM data around it and parses it into `Way` lists.

**Bounding box.** The waypoints' axis-aligned UTM bounding box, expanded by `grid_margin` (default 150 m, `config/planner_defaults.yaml`) on every side. `MapData(..., grid_margin=...)` overrides it.

**Overpass query.** `run_queries()` sends one union query for the box: every way, its nodes, the relations that reference it, and every node. Nothing is filtered server-side; `run_parse()` classifies client-side. `OverpassClient` (`map_data/utils/overpass.py`) rotates endpoints, retries, and rejects error bodies.

**Parsing.** `run_parse()`:

1. `parse_osm_ways` builds a `Way` per OSM way (closed ways become polygons, except linear barriers such as fences and walls).
2. `parse_osm_rels` merges multipolygon relation members into rings, stamps the relation's tags onto untagged outers and subtracts inner rings.
3. `parse_osm_nodes` turns standalone obstacle nodes into circles.
4. `separate_ways` sorts ways into roads, footways and barriers (see [Tag configuration](tag_configuration.md)).
5. `parse_intersections` adds a crossroad wherever a footway/road node branches.

**Attributes after parsing:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `roads_list` | `list[Way]` | Non-footway `highway` ways, buffered to 7 m wide polygons |
| `footways_list` | `list[Way]` | Footway-class `highway` ways, buffered to 3 m wide polygons |
| `barriers_list` | `list[Way]` | Barrier polygons, buffered barrier lines and obstacle circles |
| `crossroads_list` | `list[Way]` | Junction circles of radius `CROSSROAD_RADIUS` (1.5 m) |
| `nodes_cache` | `dict[int, dict]` | OSM node ID → `{lat, lon, tags}` |
| `zone_number`, `zone_letter` | `int`, `str` | UTM zone of the input waypoints |
| `waypoints` | `np.ndarray` | `(N, 2)` UTM input waypoints |
| `min_x`, `max_x`, `min_y`, `max_y` | `float` | UTM bounding box including `grid_margin` |

**OSM cache.** The raw Overpass response is saved as `<stem>.osm_cache.json` next to the source GPX. It is reused when the hash of the query (which contains the bounding box) matches and it is younger than 30 days (`OSM_CACHE_MAX_AGE`). `run_queries()` uses it by default (the viewer's fetch and GPX upload); `run_all()` bypasses it, and `create_mapdata` uses it only with `--cache`.

**Serialization.** `MapData.save()` writes JSON (`.mapdata`) atomically; `MapData.load()` reads it and recomputes the node-based crossroads. See [Data formats](data_formats.md).

**Traversability.** `MapData.apply_traversability(rules)` removes the roads and footways the [traversability rules](../planning.md#traversability-rules) refuse and recomputes the crossroads. It runs at load time (`load_mapdata_with_annotations`), not at parse time, so the `.mapdata` file keeps the full OSM classification.

---

## Way objects

Every OSM feature is a `Way` (`map_data/utils/way.py`):

- `id`: OSM way or node ID; negative or string IDs for merged relation rings, viewer-drawn annotations and split segments;
- `is_area`: `True` when `line` is a polygon;
- `nodes`: ordered OSM node IDs;
- `tags`: OSM tags;
- `line`: Shapely geometry in UTM. Roads, footways and open barrier lines are buffered to `Polygon`s during parsing; `MapData.centre_line(way)` recovers a footway's centerline;
- `in_out`: `"outer"` or `"inner"` for multipolygon members, else `None`.

---

## Annotations and edits

Viewer edits live in `<stem>.annotations.json` next to the `.mapdata` file, which is never modified.

- `map_data/edits.py`: the annotation store (load, quarantine of an unreadable store, the locked `annotation_store()` context manager) and the way-edit pipeline that rebuilds a way's `nodes` and `line` from deleted, moved, added and split nodes.
- `map_data/annotations.py`: `apply_store(md, store)` merges a store into a `MapData` (way edits, tag overrides, drawn paths and obstacles, `join_ways`). `load_mapdata_with_annotations()` loads a map, merges its store and applies the traversability rules in one call; `edit_mapdata()` does the same for an already loaded map. The viewer, `map_data_plan`, `route_planner` and `osm_cloud` all use it, so they see the same map.

`join_ways` gives ways that meet without a shared node ID a junction node: a drawn path's loose end or a moved end node within `JOIN_DISTANCE_M` (5 m) of another way, and every crossing of a drawn path.

---

## Pathsolvers

`map_data/pathsolver/route.py` is the shared entry point: `plan_route(md, points_latlon, algorithm=...)` validates the request, runs the graph or grid planner and returns a `RouteResult`, or raises `RoutePlanningError` with a machine-readable `reason`. The viewer, `map_data_plan` and `route_planner` all call it.

### GraphPlanner

`map_data/pathsolver/graph_planner.py`

A* over the OSM way network (`footway` and/or `road`). Ways connect by shared node ID. An edge weighs `length × (1 + way cost + rule cost)` (see [Planner configuration](planner_config.md)); reported lengths stay geometric. Waypoints snap to the nearest edge through an STRtree; a waypoint farther than `max_snap_distance` fails with `snap_too_far`. Open pedestrian areas (`WalkableArea`, `walkable_area.py`) are crossed through a visibility graph instead of walked around their rim.

### ReplanPath

`map_data/pathsolver/replan.py`

Cost-grid planning over a box around the waypoints:

| Module | Class / function | Responsibility |
|--------|-----------------|----------------|
| `grid_constructor.py` | `PathGrid` | Cost raster at `cell_size`; cells near a way take its cost, obstacle cells are `inf` |
| `grid_astar.py` | `grid_astar` | 8-connected A* on the raster |
| `rrt_star.py` | `RRTStar` | Sampling planner with informed sampling and an adaptive rewiring radius |
| `smoothing.py` | `smooth_path` | Gradient-descent smoothing with collision checks |
| `way_cost.py` | `way_cost`, `load_cost_tables` | `highway`/`surface` cost shared with `GraphPlanner` |

`replan()` plans each leg between consecutive waypoints. A straight leg that hits no obstacle is kept. Otherwise blocked ends are moved to the nearest free cell within `SNAP_RADIUS_M` (2 m), and the leg fails fast when start and goal lie in disconnected free regions. RRT* falls back to A* when it finds no path. The result is then smoothed (optional) and simplified. On failure it returns `None` and sets `failure_reason` (`no_path`, `start_blocked`, `goal_blocked`, `cancelled`).

Defaults come from `grid_defaults()`; see [Planner configuration](planner_config.md).

---

## Viewer

`map_data/viewer/` is a single-page Flask + Leaflet application.

**Back-end:**

| Module | Responsibility |
|---|---|
| `app.py` | App factory, Socket.IO, CLI (`map_data_viewer`), optional ROS 2 thread |
| `routes/` | The `viewer` blueprint: `files` (page, file list, map, fetch, uploads, export), `annotations`, `ways` (way and node edits), `planning` (planner defaults, traversability, cost grid, replan), `sharing` (goal QR codes, wormhole), `common` (JSON errors, path helpers, way resolution, validators, `get_merged_mapdata`) |
| `helpers.py` | GeoJSON conversion |
| `cache.py` | Read caches keyed on file signature and annotation-store digest |
| `ros_node.py` | `TrackerNode`, the Tracker's ROS 2 subscriber |
| `tracker_routes.py`, `tracker_config.py` | Tracker settings API and its parameter file |

Every error is JSON `{"error": message}`; see [Viewer REST API](../api/viewer_routes.md).

**Front-end modes** (`static/js/`):

| Mode | Entry point | Purpose |
|------|------------|---------|
| Viewer | `ui_handlers.js`, `layers.js`, `draw_handlers.js` | Inspect, edit, annotate and search OSM features |
| Planner | `planner_mode.js` | Waypoints, route planning, GPX download, goal QR codes |
| Tracker | `tracker_mode.js` | Live robot telemetry over Socket.IO |

---

## Executables

| Executable | File | Kind | Purpose |
|------|------|------|---------|
| `create_mapdata` | `map_data/create_mapdata.py` | CLI | Build a `.mapdata` from a GPX file |
| `map_data_info` | `map_data/info.py` | CLI | Statistics and validation of a `.mapdata` file |
| `map_data_plan` | `map_data/plan_route_cli.py` | CLI | Plan a route offline (`plan_route`) |
| `map_data_viewer` | `map_data/viewer/app.py` | CLI, optional ROS 2 | Web viewer; with ROS 2 also runs `TrackerNode` |
| `osm_cloud` | `map_data/osm_cloud.py` | ROS 2 node | Publishes the way network as a cost-grid `PointCloud2`, intersections as `PoseArray` and markers |
| `route_planner` | `map_data/route_planner.py` | ROS 2 node | `PlanRoute` action server on `~/plan_route`; publishes the route latched |

The `PlanRoute` action is defined in the nested `map_data_interfaces/` package.

---

## Data flow

1. **Create the map:** `create_mapdata -f <file>.gpx -d`, or *Fetch area* / GPX upload in the viewer, writes `<name>.mapdata`.
2. **Inspect and annotate:** the viewer renders the map; edits go to `<name>.annotations.json`.
3. **Plan:** the viewer's Planner, `map_data_plan` or a `PlanRoute` goal to `route_planner` load the map with its annotations and traversability rules and call `plan_route`.
4. **Drive:** `osm_cloud` publishes the same map for the navigation stack; the viewer's Tracker shows the robot and its route.

---

## Project structure

```
map_data/                          # repository root
├── map_data/                      # Python package
│   ├── map_data.py                # MapData
│   ├── annotations.py             # apply an annotation store to a MapData
│   ├── edits.py                   # annotation store IO, way-edit pipeline
│   ├── traversability.py          # tag rules: which ways may be driven on
│   ├── create_mapdata.py          # create_mapdata CLI
│   ├── info.py                    # map_data_info CLI
│   ├── plan_route_cli.py          # map_data_plan CLI
│   ├── osm_cloud.py               # osm_cloud ROS 2 node
│   ├── route_planner.py           # route_planner ROS 2 action server
│   ├── pathsolver/
│   │   ├── route.py               # plan_route, RoutePlanningError, goal_plan_kwargs
│   │   ├── graph_planner.py       # GraphPlanner (on-network A*)
│   │   ├── walkable_area.py       # crossing open pedestrian areas
│   │   ├── astar.py               # generic A*
│   │   ├── way_cost.py            # highway/surface cost tables
│   │   ├── replan.py              # ReplanPath, grid_defaults
│   │   ├── grid_constructor.py    # PathGrid cost raster
│   │   ├── grid_astar.py          # grid A*
│   │   ├── rrt_star.py            # RRTStar
│   │   └── smoothing.py           # gradient-descent smoother
│   ├── utils/
│   │   ├── config.py              # config lookup, setup_logging
│   │   ├── densify.py             # resample ways to a max step
│   │   ├── geodesy.py             # UTM / WGS84 / ECEF
│   │   ├── gpx.py                 # GPX writing
│   │   ├── launch.py              # launch-file helpers
│   │   ├── overpass.py            # OverpassClient
│   │   ├── parsing.py             # OSM parsing and classification
│   │   ├── qr.py                  # goal QR codes
│   │   ├── serialization.py       # .mapdata JSON, atomic writes
│   │   └── way.py                 # Way
│   └── viewer/
│       ├── app.py                 # Flask app, Socket.IO, CLI
│       ├── routes/                # REST API, one module per concern
│       ├── helpers.py             # GeoJSON conversion
│       ├── cache.py               # read caches
│       ├── ros_node.py            # TrackerNode
│       ├── tracker_routes.py      # Tracker settings API
│       ├── tracker_config.py      # Tracker parameter file
│       ├── templates/index.html   # page shell
│       └── static/                # css/, js/ (one module per mode and concern)
├── map_data_interfaces/           # PlanRoute action (separate ament_cmake package)
├── config/                        # planner_defaults, traversability, route_planner, osm_grid, tracker, helhest_jr
├── parameters/                    # tag classification CSVs
├── launch/                        # osm_cloud.launch.py, route_planner.launch.py
├── demo/                          # demo map, installed to share/map_data/data
├── data/                          # local maps (untracked)
├── scripts/
│   ├── setup_ws.sh                # symlink map_data_interfaces into the workspace
│   ├── ros_smoke.sh               # ROS 2 runtime smoke test
│   └── build_static_demo.sh       # static viewer demo for GitHub Pages
├── tests/                         # pytest suite
│   └── data/                      # Stromovka .mapdata fixture (ODbL)
├── docs/                          # MkDocs source
├── pyproject.toml, uv.lock        # Python metadata, locked dev environment
├── package.xml, setup.py, setup.cfg, resource/   # ament_python packaging
└── mkdocs.yml
```
