# Known Issues & Backlog

<div class="todo-app">
<style>
.todo-app {
  font-family: inherit;
}

/*── Controls ───────────────────────────────────────*/
.todo-controls {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 24px;
  padding: 14px 16px;
  background: var(--cat-mantle);
  border: 1px solid var(--cat-surface1);
  border-radius: 8px;
}

.todo-control-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.todo-control-label {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--cat-subtext0);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  min-width: 72px;
  flex-shrink: 0;
}

.todo-control-sep {
  width: 1px;
  height: 18px;
  background: var(--cat-surface1);
  margin: 0 4px;
  flex-shrink: 0;
}

/*Group-by toggle*/
.group-btn {
  padding: 5px 16px;
  border-radius: 20px;
  border: 1.5px solid var(--cat-surface2);
  background: transparent;
  color: var(--cat-subtext1);
  cursor: pointer;
  font-size: 0.82rem;
  font-weight: 600;
  transition: all 0.15s;
  font-family: inherit;
}
.group-btn:hover { border-color: var(--cat-mauve); color: var(--cat-mauve); }
.group-btn.active {
  background: var(--cat-mauve);
  border-color: var(--cat-mauve);
  color: var(--cat-base);
}

/*Filter pills*/
.filter-btn {
  padding: 4px 12px;
  border-radius: 20px;
  border: 1.5px solid;
  background: transparent;
  cursor: pointer;
  font-size: 0.76rem;
  font-weight: 600;
  font-family: inherit;
  transition: all 0.15s;
}
.filter-btn:not(.active) { opacity: 0.35; }
.filter-btn:hover { opacity: 1 !important; }

/*Severity pill colours*/
.filter-btn[data-key="critical"]    { border-color: var(--cat-red);      color: var(--cat-red);      }
.filter-btn[data-key="important"]   { border-color: var(--cat-peach);    color: var(--cat-peach);    }
.filter-btn[data-key="minor"]       { border-color: var(--cat-blue);     color: var(--cat-blue);     }
.filter-btn[data-key="nice-to-have"]{ border-color: var(--cat-lavender); color: var(--cat-lavender); }
.filter-btn[data-key="critical"].active    { background: var(--cat-red);      color: var(--cat-base); }
.filter-btn[data-key="important"].active   { background: var(--cat-peach);    color: var(--cat-base); }
.filter-btn[data-key="minor"].active       { background: var(--cat-blue);     color: var(--cat-base); }
.filter-btn[data-key="nice-to-have"].active{ background: var(--cat-lavender); color: var(--cat-base); }

/*Type pill colours*/
.filter-btn[data-key="bug"]          { border-color: var(--cat-red);     color: var(--cat-red);    }
.filter-btn[data-key="security"]     { border-color: var(--cat-maroon);  color: var(--cat-maroon); }
.filter-btn[data-key="improvement"]  { border-color: var(--cat-green);   color: var(--cat-green);  }
.filter-btn[data-key="testing"]      { border-color: var(--cat-teal);    color: var(--cat-teal);   }
.filter-btn[data-key="documentation"]{ border-color: var(--cat-mauve);   color: var(--cat-mauve);  }
.filter-btn[data-key="bug"].active          { background: var(--cat-red);    color: var(--cat-base); }
.filter-btn[data-key="security"].active     { background: var(--cat-maroon); color: var(--cat-base); }
.filter-btn[data-key="improvement"].active  { background: var(--cat-green);  color: var(--cat-base); }
.filter-btn[data-key="testing"].active      { background: var(--cat-teal);   color: var(--cat-base); }
.filter-btn[data-key="documentation"].active{ background: var(--cat-mauve);  color: var(--cat-base); }

.todo-reset {
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--cat-subtext0);
  cursor: pointer;
  background: none;
  border: none;
  font-family: inherit;
  padding: 2px 6px;
  border-radius: 4px;
  transition: color 0.15s;
  flex-shrink: 0;
}
.todo-reset:hover { color: var(--cat-text); }
.todo-reset.hidden { visibility: hidden; }

/*── Stats bar ──────────────────────────────────────*/
.todo-stats {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
  font-size: 0.8rem;
  color: var(--cat-subtext0);
}
.todo-stat-pill {
  padding: 2px 10px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 0.73rem;
}

/*── Section headers ────────────────────────────────*/
.todo-section { margin-bottom: 28px; }

.todo-section-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.todo-section-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}
.todo-section-title {
  font-size: 1rem;
  font-weight: 700;
  margin: 0;
  padding: 0;
  border: none;
  line-height: 1.3;
}
.todo-section-count {
  font-size: 0.72rem;
  background: var(--cat-surface1);
  color: var(--cat-subtext0);
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 600;
}
.todo-section-line {
  flex: 1;
  height: 1px;
}

/*── Cards ──────────────────────────────────────────*/
.todo-cards { display: flex; flex-direction: column; gap: 7px; }

.todo-card {
  background: var(--cat-mantle);
  border: 1px solid var(--cat-surface1);
  border-left: 3px solid;
  border-radius: 6px;
  padding: 11px 14px;
}
.todo-card[data-sev="critical"]    { border-left-color: var(--cat-red);      }
.todo-card[data-sev="important"]   { border-left-color: var(--cat-peach);    }
.todo-card[data-sev="minor"]       { border-left-color: var(--cat-blue);     }
.todo-card[data-sev="nice-to-have"]{ border-left-color: var(--cat-lavender); }

.todo-card-header {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 5px;
  flex-wrap: wrap;
}
.todo-card-title {
  font-weight: 600;
  color: var(--cat-text);
  font-size: 0.9rem;
  flex: 1;
  min-width: 140px;
  line-height: 1.35;
}
.badge {
  padding: 2px 9px;
  border-radius: 10px;
  font-size: 0.68rem;
  font-weight: 700;
  white-space: nowrap;
  flex-shrink: 0;
  line-height: 1.6;
}
.badge-sev-critical    { background: var(--cat-red);      color: var(--cat-base); }
.badge-sev-important   { background: var(--cat-peach);    color: var(--cat-base); }
.badge-sev-minor       { background: var(--cat-blue);     color: var(--cat-base); }
.badge-sev-nice-to-have{ background: var(--cat-lavender); color: var(--cat-base); }
.badge-type-bug          { background: var(--cat-red);     color: var(--cat-base); }
.badge-type-security     { background: var(--cat-maroon);  color: var(--cat-base); }
.badge-type-improvement  { background: var(--cat-green);   color: var(--cat-base); }
.badge-type-testing      { background: var(--cat-teal);    color: var(--cat-base); }
.badge-type-documentation{ background: var(--cat-mauve);   color: var(--cat-base); }

.todo-card-desc {
  font-size: 0.83rem;
  color: var(--cat-subtext1);
  line-height: 1.55;
  margin-bottom: 7px;
}
.todo-card-file {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--cat-overlay1);
  background: var(--cat-surface0);
  padding: 2px 8px;
  border-radius: 4px;
  display: inline-block;
}

.todo-empty {
  color: var(--cat-subtext0);
  font-size: 0.88rem;
  padding: 32px 0;
  text-align: center;
  border: 1px dashed var(--cat-surface2);
  border-radius: 8px;
}
</style>

<div class="todo-controls">
  <div class="todo-control-row">
    <span class="todo-control-label">Group by</span>
    <button class="group-btn active" id="btn-type">Type</button>
    <button class="group-btn" id="btn-severity">Severity</button>
  </div>
  <div class="todo-control-row">
    <span class="todo-control-label">Type</span>
    <button class="filter-btn active" data-dim="type" data-key="bug">Bug</button>
    <button class="filter-btn active" data-dim="type" data-key="security">Security</button>
    <button class="filter-btn active" data-dim="type" data-key="improvement">Improvement</button>
    <button class="filter-btn active" data-dim="type" data-key="testing">Testing</button>
    <button class="filter-btn active" data-dim="type" data-key="documentation">Documentation</button>
  </div>
  <div class="todo-control-row">
    <span class="todo-control-label">Severity</span>
    <button class="filter-btn active" data-dim="sev" data-key="critical">Critical</button>
    <button class="filter-btn active" data-dim="sev" data-key="important">Important</button>
    <button class="filter-btn active" data-dim="sev" data-key="minor">Minor</button>
    <button class="filter-btn active" data-dim="sev" data-key="nice-to-have">Nice to have</button>
    <button class="todo-reset hidden" id="todo-reset">Reset filters</button>
  </div>
</div>

<div id="todo-content"></div>

<script>
(function () {
  const SEV_ORDER  = ['critical', 'important', 'minor', 'nice-to-have'];
  const TYPE_ORDER = ['bug', 'security', 'improvement', 'testing', 'documentation'];

  const SEV_LABEL = {
    'critical': 'Critical', 'important': 'Important',
    'minor': 'Minor', 'nice-to-have': 'Nice to have'
  };
  const TYPE_LABEL = {
    'bug': 'Bugs', 'security': 'Security', 'improvement': 'Improvements',
    'testing': 'Testing', 'documentation': 'Documentation'
  };
  const SEV_COLOR = {
    'critical': 'var(--cat-red)', 'important': 'var(--cat-peach)',
    'minor': 'var(--cat-blue)', 'nice-to-have': 'var(--cat-lavender)'
  };
  const TYPE_COLOR = {
    'bug': 'var(--cat-red)', 'security': 'var(--cat-maroon)',
    'improvement': 'var(--cat-green)', 'testing': 'var(--cat-teal)',
    'documentation': 'var(--cat-mauve)'
  };

  const ITEMS = [
    {
      type: 'testing', sev: 'minor',
      title: 'No unit tests for OverpassClient',
      desc: 'The Overpass client logic, including its retry and status-check mechanism, is untested. Mocking the API responses would allow for testing various error scenarios (429, 500, timeouts).',
      file: 'map_data/utils/overpass.py'
    },
    {
      type: 'improvement', sev: 'nice-to-have',
      title: 'Expose more planning parameters in the viewer',
      desc: 'The newly promoted parameters in <code>planner_defaults.yaml</code> (grid cost weight, obstacle radius, buffer widths) should be added to the viewer configuration UI to allow for easier experimentation without manual YAML edits.',
      file: 'viewer'
    },
    {
      type: 'testing', sev: 'important',
      title: 'No integration tests for the full pipeline',
      desc: 'There are no end-to-end tests covering GPX parse → Overpass query → feature classification → save → reload. Regressions in parsing are only caught during manual testing.',
      file: null
    },
    {
      type: 'testing', sev: 'important',
      title: 'Viewer has zero test coverage',
      desc: '<code>routes.py</code> and <code>helpers.py</code> contain the most complex logic in the package — annotation CRUD, way splitting, GeoJSON export — but are entirely untested.',
      file: 'map_data/viewer/routes.py, map_data/viewer/helpers.py'
    },
    {
      type: 'testing', sev: 'minor',
      title: 'RRT* tests do not verify obstacle avoidance',
      desc: '<code>test_rrt.py</code> checks that a path is returned but not that it avoids the obstacle cell. The test passes even if the planner ignores obstacles entirely.',
      file: 'tests/test_rrt.py'
    },
    {
      type: 'testing', sev: 'minor',
      title: 'No error-scenario tests',
      desc: 'No tests exercise error paths: malformed GPX input, corrupt <code>.mapdata</code> files, Overpass timeouts, or planning with an empty <code>MapData</code> object.',
      file: null
    },
    {
      type: 'improvement', sev: 'nice-to-have',
      title: 'OSM grid margin',
      desc: 'The OSM grid margin at parsing currently uses two variables. Replace them with a single <code>GRID_MARGIN</code> variable with a default in config file. Also make it changable in the viewer in Fetch and GPX parse dialogs.',
      file: null
    },
    {
      type: 'improvement', sev: 'minor',
      title: 'Annotated Objects in Edit Mode',
      desc: 'Do not show the editable parts of annotated objects before the user clicks them. The current behaviour is a bit noisy and may overwhelm new users. Also have the same editing UI for OSM objects and for user-created annotations.',
      file: 'viewer'
    },
    {
      type: 'improvement', sev: 'important',
      title: 'Rerendering of Reverted Changes',
      desc: 'When reverting an annotation edit, the viewer updates the data but does not trigger a rerender, leaving stale old geometry on the map, alongside the new reverted geometry, until the next load. This can cause confusion about whether the revert action succeeded.',
      file: 'viewer'
    },
    {
      type: 'improvement', sev: 'minor',
      title: 'Improve Node Editing',
      desc: 'Update the editing of nodes to be more responsive. Curently, clicking a node does not always trigger a drag of the node. Also true for path points in planner.',
      file: 'viewer'
    },
    {
      type: 'improvement', sev: 'nice-to-have',
      title: 'Viewer File Loading',
    desc: 'Allow for drag-and-drop loading of <code>.mapdata</code> files in the viewer, in addition to the existing file dialog. This would speed up testing of different files and be more intuitive for users. Also allow for droppning GPX/YAML files to trigger parsing. Check for file API support of YAML files.',
      file: 'viewer'
    },
    {
      type: 'improvement', sev: 'nice-to-have',
      title: 'Speed up planning hot paths with native extensions',
      desc: 'The main bottlenecks are the Python <code>_bresenham</code> generator and <code>_segment_cost</code> in RRT* (called O(max_iter × neighbors) times), and the <code>heapq</code> loop in <code>grid_astar</code>. Recommended approach: (1) try <code>@numba.njit</code> on these inner loops first — zero build overhead, easy to toggle; (2) replace <code>grid_astar</code> with <code>skimage.graph.route_through_array</code> which is C-backed; (3) Cython or pybind11 only if numba is unacceptable for deployment. Full C/C++ rewrite is not justified given that numpy, cKDTree, and Shapely/GEOS are already native.',
      file: 'map_data/pathsolver/rrt_star.py, map_data/pathsolver/grid_astar.py'
    },
    {
      type: 'security', sev: 'critical',
      title: 'Path Traversal via Unsanitized File Parameter',
      desc: 'Multiple API routes accept a <code>file</code> query parameter and join it directly with the data directory using <code>os.path.join()</code> without verifying the resolved path stays within the data directory. An attacker can supply <code>../../../etc/passwd</code> (or similar) to read arbitrary files accessible to the server process. Fix: resolve the joined path with <code>os.path.realpath()</code> and assert it starts with the real data directory.',
      file: 'map_data/viewer/routes.py'
    },
    {
      type: 'bug', sev: 'minor',
      title: 'YAML Path Parser Has No Error Handling',
      desc: '<code>parse_yaml_file()</code> has no try/except, unlike its GPX counterpart. A malformed YAML file, a missing <code>waypoints</code> key, or a waypoint with missing <code>latitude</code>/<code>longitude</code> fields will raise an uncaught exception that propagates to the caller. Wrap the body in a try/except and return <code>[]</code> with a logged error, matching the GPX parser contract.',
      file: 'map_data/utils/gpx.py'
    },
    {
      type: 'bug', sev: 'minor',
      title: 'Bounding Box Min/Max Not Validated in fetch_area',
      desc: 'The <code>/api/fetch_area</code> endpoint accepts <code>min_lat</code>, <code>max_lat</code>, <code>min_lon</code>, <code>max_lon</code> from the request body and checks they are present, but never validates that <code>min &lt; max</code>. Inverted bounds produce a valid but nonsensical UTM bounding box that silently yields an empty or incorrect Overpass query.',
      file: 'map_data/viewer/routes.py'
    }
  ];

  let groupBy = 'type';
  const activeTypes = new Set(TYPE_ORDER);
  const activeSevs  = new Set(SEV_ORDER);

  const content  = document.getElementById('todo-content');
  const resetBtn = document.getElementById('todo-reset');

  function isFiltered() {
    return activeTypes.size < TYPE_ORDER.length || activeSevs.size < SEV_ORDER.length;
  }

  function render() {
    resetBtn.classList.toggle('hidden', !isFiltered());

    const visible = ITEMS.filter(i => activeTypes.has(i.type) && activeSevs.has(i.sev));

    if (visible.length === 0) {
      content.innerHTML = '<div class="todo-empty">No items match the current filters.</div>';
      return;
    }

    const groupOrder  = groupBy === 'type' ? TYPE_ORDER  : SEV_ORDER;
    const groupLabel  = groupBy === 'type' ? TYPE_LABEL  : SEV_LABEL;
    const groupColor  = groupBy === 'type' ? TYPE_COLOR  : SEV_COLOR;
    const groupKey    = groupBy === 'type' ? 'type'      : 'sev';
    const sortOrder   = groupBy === 'type' ? SEV_ORDER   : TYPE_ORDER;
    const sortKey     = groupBy === 'type' ? 'sev'       : 'type';

    // stats
    const nCrit = visible.filter(i => i.sev === 'critical').length;
    const nImp  = visible.filter(i => i.sev === 'important').length;
    let statsHtml = `<div class="todo-stats">`;
    statsHtml += `<span>${visible.length} item${visible.length !== 1 ? 's' : ''}</span>`;
    if (nCrit) statsHtml += `<span class="todo-stat-pill" style="background:var(--cat-red);color:var(--cat-base)">${nCrit} critical</span>`;
    if (nImp)  statsHtml += `<span class="todo-stat-pill" style="background:var(--cat-peach);color:var(--cat-base)">${nImp} important</span>`;
    statsHtml += `</div>`;

    // group and sort
    const groups = {};
    groupOrder.forEach(k => { groups[k] = []; });
    visible.forEach(item => { groups[item[groupKey]].push(item); });

    let sectionsHtml = '';
    groupOrder.forEach(gk => {
      const items = groups[gk];
      if (!items.length) return;
      items.sort((a, b) => sortOrder.indexOf(a[sortKey]) - sortOrder.indexOf(b[sortKey]));

      const color = groupColor[gk];
      const label = groupLabel[gk];

      sectionsHtml += `<div class="todo-section">`;
      sectionsHtml += `<div class="todo-section-header">`;
      sectionsHtml += `<span class="todo-section-dot" style="background:${color}"></span>`;
      sectionsHtml += `<span class="todo-section-title" style="color:${color}">${label}</span>`;
      sectionsHtml += `<span class="todo-section-count">${items.length}</span>`;
      sectionsHtml += `<span class="todo-section-line" style="background:linear-gradient(to right,${color}44,transparent)"></span>`;
      sectionsHtml += `</div><div class="todo-cards">`;

      items.forEach(item => {
        // always show both badges; the grouping dimension first, the other second
        const sevBadge  = `<span class="badge badge-sev-${item.sev}">${SEV_LABEL[item.sev]}</span>`;
        const typeBadge = `<span class="badge badge-type-${item.type}">${item.type.charAt(0).toUpperCase() + item.type.slice(1)}</span>`;
        const [first, second] = groupBy === 'severity' ? [typeBadge, sevBadge] : [sevBadge, typeBadge];
        const fileHtml = item.file
          ? `<span class="todo-card-file">${item.file}</span>`
          : '';
        sectionsHtml += `
          <div class="todo-card" data-sev="${item.sev}" data-type="${item.type}">
            <div class="todo-card-header">
              <span class="todo-card-title">${item.title}</span>
              ${first}${groupBy === 'type' ? '' : ` ${second}`}
            </div>
            <div class="todo-card-desc">${item.desc}</div>
            ${fileHtml}
          </div>`;
      });

      sectionsHtml += `</div></div>`;
    });

    content.innerHTML = statsHtml + sectionsHtml;
  }

  // Group-by buttons
  document.getElementById('btn-type').addEventListener('click', () => {
    groupBy = 'type';
    document.getElementById('btn-type').classList.add('active');
    document.getElementById('btn-severity').classList.remove('active');
    render();
  });
  document.getElementById('btn-severity').addEventListener('click', () => {
    groupBy = 'severity';
    document.getElementById('btn-severity').classList.add('active');
    document.getElementById('btn-type').classList.remove('active');
    render();
  });

  // Filter pills
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const dim = btn.dataset.dim;
      const key = btn.dataset.key;
      const set = dim === 'type' ? activeTypes : activeSevs;
      if (set.has(key)) { set.delete(key); btn.classList.remove('active'); }
      else              { set.add(key);    btn.classList.add('active');    }
      render();
    });
  });

  // Reset
  resetBtn.addEventListener('click', () => {
    TYPE_ORDER.forEach(k => activeTypes.add(k));
    SEV_ORDER.forEach(k => activeSevs.add(k));
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.add('active'));
    render();
  });

  render();
})();
</script>
</div>
