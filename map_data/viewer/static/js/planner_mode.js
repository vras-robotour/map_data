/**
 * PlannerMode handles the path drawing and replanning logic.
 * Ported and adapted from mission_planner.
 */

class PlannerMode {
  constructor() {
    this.points = []; // [{lat, lon, marker}]
    this.active = false;
    this.isProcessing = false;
    this.isDragging = false;
    this.lastDragEndTime = 0;
    this.currentReplanId = null;
    this.currentWormholeId = null;
    this.pathPolyline = null;
    this.markerLayer = L.layerGroup();
    this.highwayCosts = {};
    this.surfaceCosts = {};
    this.defaults = {};
    this.rulesYaml = null; // applied, unsaved traversability rules; null = the rule file
    this.startAtRobot = false; // waypoint 0 pinned to the robot (see onRobotPosition)
    this.robotPos = null;      // last position trackerMode published
    this.dragPoint = null;     // waypoint being dragged, see _onWpDragMove

    this.init();
  }

  async init() {
    this.bindEvents();
    this.setupDragAndDrop();
    await this.fetchDefaults();
  }

  async fetchDefaults() {
    try {
      let data;
      if (STATIC_BASE) {
        data = await _staticJson('api/planner_defaults.json');
      } else {
        const res = await api('GET', '/api/planner_defaults');
        if (!res.ok) throw new Error(await res.text());
        data = await res.json();
      }
      this.defaults = data;
      this.highwayCosts = { ...data.highway_costs };
      this.surfaceCosts = { ...data.surface_costs };
      
      // Update UI fields if they exist
      if (data.cell_size) document.getElementById('planner-cell-size').value = data.cell_size;
      if (data.inflate_obstacles) document.getElementById('planner-inflate').value = data.inflate_obstacles;
      if (data.grid_cost_weight) document.getElementById('planner-grid-cost-weight').value = data.grid_cost_weight;
      if (data.simplify_path !== undefined) document.getElementById('planner-simplify').checked = data.simplify_path;
      if (data.smooth_path !== undefined) document.getElementById('planner-smooth').checked = data.smooth_path;

      // Populate advanced fields in all fetch/GPX modals with defaults
      // (a missing default keeps the field's value from the `advanced` macro)
      const advanced = {
        'grid-margin': data.grid_margin,
        'obstacle-radius': data.obstacle_radius,
        'buf-road': data.buffer_widths?.road,
        'buf-footway': data.buffer_widths?.footway,
        'buf-barrier': data.buffer_widths?.barrier,
      };
      for (const prefix of ['fetch', 'gpx']) {
        for (const [suffix, value] of Object.entries(advanced)) {
          const el = document.getElementById(`${prefix}-${suffix}`);
          if (el && value != null) el.value = value;
        }
      }
      
    } catch (err) {
      console.error('Failed to fetch planner defaults:', err);
      // Fallback
      this.resetCosts();
    }
  }

  enable() {
    this.active = true;
    this.markerLayer.addTo(map);
    this.redraw();
    if (this.startAtRobot) this._moveStartToRobot();
    this.updateUI(); // Ensure UI state is correct
    // Enable map click for adding points
    map.on('click', this.handleMapClick, this);
    this._refilter();
  }

  disable() {
    this.active = false;
    this._refilter();
    map.off('click', this.handleMapClick, this);
    this.clearMarkers();
    map.removeLayer(this.markerLayer);
    if (this.pathPolyline) {
      map.removeLayer(this.pathPolyline);
      this.pathPolyline = null;
    }
  }

  bindEvents() {
    document.getElementById('planner-clear-btn').addEventListener('click', () => this.clearAll());
    document.getElementById('planner-clear-middle-btn').addEventListener('click', () => this.clearMiddle());
    document.getElementById('import-gpx-btn').addEventListener('click', () => document.getElementById('gpx-input').click());
    document.getElementById('gpx-input').addEventListener('change', (e) => this.handleGpxImport(e));
    document.getElementById('replan-btn').addEventListener('click', () => this.replanPath());
    document.getElementById('planner-qr-btn').addEventListener('click', () => {
      if (this.points.length) this.showQr(this.points[this.points.length - 1]);
    });
    document.getElementById('qr-modal-size').addEventListener('input', (e) => {
      document.getElementById('qr-modal-img').style.height = `${e.target.value}vh`;
      // Keep the caption in proportion with the code as the slider moves.
      document.getElementById('qr-modal-caption').style.fontSize = `${e.target.value / 20}vh`;
    });
    document.getElementById('qr-modal-copy').addEventListener('click', async () => {
      const text = document.getElementById('qr-modal-text').textContent;
      try { await navigator.clipboard.writeText(text); } catch (_) { window.prompt('Copy the goal text:', text); }
    });

    document.getElementById('export-gpx-path-btn').addEventListener('click', () => this.exportToGPX());
    document.getElementById('export-wormhole-path-btn').addEventListener('click', () => this.shareViaWormhole());

    document.querySelectorAll('input[name="plan-mode"]').forEach(radio => {
      radio.addEventListener('change', () => {
        this.drawPathLine();
        this.updateUI();
      });
    });

    document.getElementById('planner-simplify').addEventListener('change', () => this.drawPathLine());
    document.getElementById('planner-smooth').addEventListener('change', () => this.drawPathLine());
    document.getElementById('planner-show-grid').addEventListener('change', (e) => this.toggleCostGrid(e.target.checked));
    document.getElementById('planner-costs-btn').addEventListener('click', () => this.showCostsModal());
    document.getElementById('planner-costs-save').addEventListener('click', () => this.saveCosts());
    document.getElementById('planner-costs-reset').addEventListener('click', () => this.resetCosts());

    document.getElementById('planner-rules-btn').addEventListener('click', () => this.showRulesModal());
    document.getElementById('planner-rules-reload').addEventListener('click', () => this.loadRulesFile());
    document.getElementById('planner-rules-apply').addEventListener('click', () => this.applyRules(false));
    document.getElementById('planner-rules-save').addEventListener('click', () => this.applyRules(true));
    for (const id of ['planner-hide-blocked', 'plan-footways', 'plan-roads']) {
      document.getElementById(id).addEventListener('change', () => this.refreshBlocked());
    }

    document.getElementById('planner-start-at-robot')
      .addEventListener('change', (e) => this.toggleStartAtRobot(e.target.checked));
    // Waypoint drag (CircleMarkers aren't draggable): mousedown on a marker sets dragPoint
    map.on('mousemove', this._onWpDragMove, this);
    map.on('mouseup', this._onWpDragEnd, this);
    // trackerMode publishes the robot position with every telemetry frame, in every mode
    document.addEventListener('robot-position', (e) => this.onRobotPosition(e.detail));
  }

  // ── Start at the robot ──────────────────────────────────────────────────────
  // The toggle pins waypoint 0 to the robot: it follows the live position while the path
  // is being drawn, and Replan Path refreshes it once more, so the route is planned from
  // where the robot is at that moment. It stops following a planned path, whose first
  // pose is a result, not a waypoint; replanning re-pins it.
  onRobotPosition(pos) {
    this.robotPos = pos;
    const toggle = document.getElementById('planner-start-at-robot');
    if (toggle.disabled) {
      toggle.disabled = false;
      toggle.parentElement.title = "Plan from the robot's current position";
    }
    if (this.active && this.startAtRobot && !this.isProcessing && !this.isDragging
        && !this.hasPlannedPath) {
      this._moveStartToRobot();
    }
  }

  /** Put waypoint 0 on the robot. Returns false when no position has arrived yet. */
  _moveStartToRobot() {
    const pos = this.robotPos;
    if (!pos) return false;
    if (!this.points.length) {
      this.addPoint(pos.lat, pos.lon);
      return true;
    }
    const start = this.points[0];
    if (start.lat === pos.lat && start.lon === pos.lon) return true;
    start.lat = pos.lat;
    start.lon = pos.lon;
    // Move the existing markers rather than redraw(): that rebuilds every marker and its
    // drag handlers, which at telemetry rate would fight with dragging a waypoint.
    start.marker?.setLatLng([pos.lat, pos.lon]);
    start.hit?.setLatLng([pos.lat, pos.lon]);
    this.hasPlannedPath = false;
    this.drawPathLine();
    this.updateUI();
    return true;
  }

  toggleStartAtRobot(on) {
    this.startAtRobot = on;
    if (!on) {
      this.redraw(); // drop the tooltip; the waypoint stays as an ordinary one
      return;
    }
    if (!this._moveStartToRobot()) {
      document.getElementById('planner-start-at-robot').checked = false;
      this.startAtRobot = false;
      setStatus('No robot position yet', 'text-warning');
      return;
    }
    this.redraw();
    setStatus('Planning from the robot position', 'text-info');
  }

  _allowedWays() {
    const allowed = [];
    if (document.getElementById('plan-footways').checked) allowed.push('footway');
    if (document.getElementById('plan-roads').checked) allowed.push('road');
    return allowed;
  }

  async showRulesModal() {
    if (STATIC_BASE) {
      setStatus('Traversability rules need the backend — run map_data_viewer locally', 'text-warning');
      return;
    }
    showError('planner-rules-error', '');
    await this.loadRulesFile();
    if (this.rulesYaml !== null) document.getElementById('planner-rules-text').value = this.rulesYaml;
    bootstrap.Modal.getOrCreateInstance(document.getElementById('planner-rules-modal')).show();
  }

  async loadRulesFile() {
    try {
      const res = await api('GET', '/api/traversability');
      if (!res.ok) throw new Error(await errorText(res));
      const data = await res.json();
      document.getElementById('planner-rules-text').value = data.yaml;
      document.getElementById('planner-rules-path').textContent = data.path;
    } catch (err) {
      showError('planner-rules-error', `Failed to load rules: ${err.message}`);
    }
  }

  async _fetchBlocked(rulesYaml) {
    const res = await api('POST', '/api/traversability/blocked', {
      body: { file: currentFile, traversability: rulesYaml ?? undefined, allowed_ways: this._allowedWays() },
    });
    if (!res.ok) throw new Error(await errorText(res));
    return (await res.json()).blocked;
  }

  async applyRules(save) {
    const text = document.getElementById('planner-rules-text').value;
    try {
      if (save) {
        const res = await api('PUT', '/api/traversability', { body: { traversability: text } });
        if (!res.ok) throw new Error(await errorText(res));
        this.rulesYaml = null;
        setStatus(`Rules saved to ${(await res.json()).path}`, 'text-success');
      } else {
        // Without a map there is nothing to evaluate against; replanning reports bad rules instead.
        if (currentFile) await this._fetchBlocked(text);
        this.rulesYaml = text;
        setStatus('Rules applied (not saved)', 'text-success');
      }
    } catch (err) {
      showError('planner-rules-error', err.message);
      return;
    }
    showError('planner-rules-error', '');
    bootstrap.Modal.getInstance(document.getElementById('planner-rules-modal')).hide();
    this.refreshBlocked();
  }

  async refreshBlocked() {
    plannerBlockedIds = new Set();
    if (document.getElementById('planner-hide-blocked').checked && currentFile && !STATIC_BASE) {
      try {
        const blocked = await this._fetchBlocked(this.rulesYaml);
        plannerBlockedIds = new Set(blocked.map(([id]) => id));
        setStatus(`Hiding ${blocked.length} ways the planner won't use`, 'text-info');
      } catch (err) {
        setStatus(`Failed to evaluate rules: ${err.message}`, 'text-danger');
      }
    }
    this._refilter();
  }

  _refilter() {
    filterLayers(document.getElementById('feature-search')?.value || '');
  }

  showCostsModal() {
    const hwContainer = document.getElementById('highway-costs-container');
    hwContainer.innerHTML = '';
    Object.entries(this.highwayCosts).sort((a,b) => a[1]-b[1]).forEach(([type, cost]) => {
      hwContainer.appendChild(this._createCostInputRow(type, cost, 'highway-cost-input'));
    });

    const surfContainer = document.getElementById('surface-costs-container');
    surfContainer.innerHTML = '';
    Object.entries(this.surfaceCosts).sort((a,b) => a[1]-b[1]).forEach(([type, cost]) => {
      surfContainer.appendChild(this._createCostInputRow(type, cost, 'surface-cost-input'));
    });
    
    const modal = new bootstrap.Modal(document.getElementById('planner-costs-modal'));
    modal.show();
  }

  _createCostInputRow(label, value, className) {
    const div = document.createElement('div');
    div.className = 'd-flex align-items-center gap-2';
    div.innerHTML = `
      <span class="text-secondary" style="font-size:0.75rem; width:80px; overflow:hidden; text-overflow:ellipsis;">${label}</span>
      <input type="number" class="form-control form-control-sm bg-dark text-light border-secondary ${className}" 
             data-type="${label}" value="${value}" step="0.05" min="0" max="1" style="font-size:0.7rem; height:24px;">
    `;
    return div;
  }

  saveCosts() {
    const hwInputs = document.querySelectorAll('.highway-cost-input');
    const surfInputs = document.querySelectorAll('.surface-cost-input');
    
    const newHwCosts = {};
    const newSurfCosts = {};
    let valid = true;
    
    const parse = (inputs, target) => {
      inputs.forEach(input => {
        const val = parseFloat(input.value);
        if (isNaN(val) || val < 0 || val > 1) {
          input.classList.add('border-danger');
          valid = false;
        } else {
          input.classList.remove('border-danger');
          target[input.dataset.type] = val;
        }
      });
    };

    parse(hwInputs, newHwCosts);
    parse(surfInputs, newSurfCosts);
    
    if (!valid) {
      alert('All costs must be between 0.0 and 1.0');
      return;
    }
    
    this.highwayCosts = newHwCosts;
    this.surfaceCosts = newSurfCosts;
    bootstrap.Modal.getInstance(document.getElementById('planner-costs-modal')).hide();
    
    // If grid is visible, refresh it
    if (document.getElementById('planner-show-grid').checked) {
      this.fetchCostGrid();
    }
    setStatus('Planner costs updated', 'text-success');
  }

  resetCosts() {
    // /api/planner_defaults is the source of truth; this.defaults is populated
    // from it in fetchDefaults(). If that fetch failed, there's nothing to
    // reset to but an empty set of costs.
    this.highwayCosts = { ...(this.defaults.highway_costs || {}) };
    this.surfaceCosts = { ...(this.defaults.surface_costs || {}) };
    this.showCostsModal();
  }

  async toggleCostGrid(show) {
    if (!show) {
      if (geoLayers.costGrid) {
        map.removeLayer(geoLayers.costGrid);
        geoLayers.costGrid = null;
      }
      return;
    }
    if (!currentFile) {
      document.getElementById('planner-show-grid').checked = false;
      setStatus('Load a map first', 'text-warning');
      return;
    }
    this.fetchCostGrid();
  }

  async fetchCostGrid() {
    if (STATIC_BASE) {
      setStatus('Cost grid needs the backend — run map_data_viewer locally', 'text-warning');
      return;
    }
    if (!currentFile) return;
    const bounds = map.getBounds();
    setStatus('Fetching cost grid...', 'text-warning');
    
    try {
      const res = await api('GET', '/api/cost_grid', {
        query: {
          file: currentFile,
          min_lat: bounds.getSouth(), min_lon: bounds.getWest(),
          max_lat: bounds.getNorth(), max_lon: bounds.getEast(),
          highway_costs: JSON.stringify(this.highwayCosts),
          surface_costs: JSON.stringify(this.surfaceCosts),
        },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      
      if (geoLayers.costGrid) map.removeLayer(geoLayers.costGrid);
      
      const offPathThreshold = this.defaults.default_off_path_cost || 0.9;
      const pathCap = this.defaults.path_cost_cap || 0.85;
      const markers = data.map(([lat, lon, cost]) => {
        let style;
        if (cost >= 1.0) {
          // Hard Obstacle style: black marker
          style = { radius: 4, fillColor: '#000', color: '#000', weight: 1, fillOpacity: 1 };
        } else if (cost >= offPathThreshold) {
          // Off-path / All-terrain style: dark gray/brown
          style = { radius: 2, fillColor: '#444', color: '#444', weight: 0, fillOpacity: 0.4 };
        } else {
          // Interpolate color from green (0) to red (pathCap)
          const normalizedCost = cost / pathCap;
          const r = Math.floor(255 * Math.min(1, normalizedCost));
          const g = Math.floor(255 * Math.max(0, 1 - normalizedCost));
          style = { radius: 3, fillColor: `rgb(${r},${g},0)`, color: '#000', weight: 0.2, fillOpacity: 0.6 };
        }
        return L.circleMarker([lat, lon], style);
      });
      
      geoLayers.costGrid = L.layerGroup(markers).addTo(map);
      
      setStatus('Cost grid loaded', 'text-success');
    } catch (err) {
      setStatus(`Failed to fetch cost grid: ${err.message}`, 'text-danger');
      document.getElementById('planner-show-grid').checked = false;
    }
  }

  handleMapClick(e) {
    if (!this.active || this.isProcessing || this.isDragging || Date.now() - this.lastDragEndTime < 200) {
      return;
    }
    this.addPoint(e.latlng.lat, e.latlng.lng);
  }

  addPoint(lat, lon) {
    const point = { lat, lon, marker: null };
    this.points.push(point);
    this.redraw();
    this.updateUI();
    // Reset path state if we just added a point and we are in graph mode
    // (the server will return a path when replan is clicked)
    this.hasPlannedPath = false;
  }

  redraw() {
    this.clearMarkers();
    if (this.pathPolyline) {
      map.removeLayer(this.pathPolyline);
      this.pathPolyline = null;
    }

    if (this.points.length === 0) return;

    const latlngs = [];
    this.points.forEach((p, i) => {
      latlngs.push([p.lat, p.lon]);

      // Create marker
      const isStart = i === 0;
      const isEnd = i === this.points.length - 1;
      const color = isStart ? '#2ecc71' : (isEnd ? '#e74c3c' : '#3498db');

      const marker = L.circleMarker([p.lat, p.lon], {
        radius: 6,
        fillColor: color,
        color: '#fff',
        weight: 2,
        fillOpacity: 1,
        draggable: true,
        bubblingMouseEvents: false
      }).addTo(map);

      // Custom drag handling for CircleMarker
      const onWpDown = (e) => {
        L.DomEvent.stopPropagation(e);
        this.dragPoint = p;
        this.isDragging = true;
        map.dragging.disable();
      };
      marker.on('mousedown', onWpDown);

      // Transparent larger hit target so waypoints are easier to grab
      const wpHit = L.circleMarker([p.lat, p.lon], {
        radius: 14, fillOpacity: 0, opacity: 0,
        bubblingMouseEvents: false, interactive: true,
      }).addTo(map);
      wpHit.on('mousedown', onWpDown);
      this.markerLayer.addLayer(wpHit);

      marker.on('mouseup', this._onWpDragEnd, this);

      // Right click to delete
      const onWpContext = (e) => {
        L.DomEvent.preventDefault(e);
        L.DomEvent.stopPropagation(e);
        this.showContextMenu(p, e.latlng);
      };
      marker.on('contextmenu', onWpContext);
      wpHit.on('contextmenu', onWpContext);

      if (isStart && this.startAtRobot) marker.bindTooltip('Robot (start)');
      this.markerLayer.addLayer(marker);
      p.marker = marker;
      p.hit = wpHit;
    });

    this.drawPathLine();
  }

  drawPathLine() {
    if (this.pathPolyline) {
      map.removeLayer(this.pathPolyline);
    }
    if (this.points.length < 2) return;

    const algorithm = document.querySelector('input[name="plan-mode"]:checked').value;
    // For graph mode, don't show the straight line if a path hasn't been returned by the server
    if (algorithm === 'graph' && !this.hasPlannedPath) {
      return;
    }

    const latlngs = this.points.map(p => [p.lat, p.lon]);
    this.pathPolyline = L.polyline(latlngs, {
      color: '#0d6efd',
      weight: 4,
      opacity: 0.7,
      dashArray: algorithm === 'rrt' && !this.hasPlannedPath ? '5, 10' : ''
    }).addTo(map);

    this.pathPolyline.on('click', (e) => {
      L.DomEvent.stopPropagation(e);
      // Find where to insert
      // Simple heuristic: find closest segment
      let bestIdx = 0;
      let minDist = Infinity;
      for (let i = 0; i < this.points.length - 1; i++) {
        const d = L.LineUtil.pointToSegmentDistance(
          map.latLngToLayerPoint(e.latlng),
          map.latLngToLayerPoint([this.points[i].lat, this.points[i].lon]),
          map.latLngToLayerPoint([this.points[i + 1].lat, this.points[i + 1].lon])
        );
        if (d < minDist) {
          minDist = d;
          bestIdx = i + 1;
        }
      }
      this.points.splice(bestIdx, 0, { lat: e.latlng.lat, lon: e.latlng.lng, marker: null });
      this.redraw();
      this.updateUI();
    });
  }

  _onWpDragMove(e) {
    const p = this.dragPoint;
    if (!p) return;
    p.marker.setLatLng(e.latlng);
    p.lat = e.latlng.lat;
    p.lon = e.latlng.lng;
    this.hasPlannedPath = false;
    this.drawPathLine();
  }

  _onWpDragEnd() {
    if (!this.dragPoint) return;
    this.dragPoint = null;
    this.isDragging = false;
    this.lastDragEndTime = Date.now();
    map.dragging.enable();
    this.redraw();
  }

  clearMarkers() {
    this.dragPoint = null; // a drag ends with its marker
    this.markerLayer.clearLayers();
  }

  updateUI() {
    const countEl = document.getElementById('point-count');
    if (this.points.length === 0) {
      countEl.textContent = 'Click map to start drawing a path';
    } else {
      let totalDist = 0;
      for (let i = 0; i < this.points.length - 1; i++) {
        const p1 = L.latLng(this.points[i].lat, this.points[i].lon);
        const p2 = L.latLng(this.points[i + 1].lat, this.points[i + 1].lon);
        totalDist += p1.distanceTo(p2);
      }

      let distStr = totalDist > 1000
        ? `${(totalDist / 1000).toFixed(2)} km`
        : `${totalDist.toFixed(1)} m`;

      // Before replanning the sum is crow-flies distance between waypoints;
      // after replanning the points are densified, so it is the actual path length.
      const distLabel = this.hasPlannedPath ? 'planned path' : 'straight-line';
      countEl.textContent = `${this.points.length} points | ${distStr} (${distLabel})`;
    }
    document.getElementById('planner-qr-btn').disabled = this.points.length < 1;
    document.getElementById('export-gpx-path-btn').disabled = this.points.length < 2;
    document.getElementById('export-wormhole-path-btn').disabled = this.points.length < 2;
    document.getElementById('planner-clear-btn').disabled = this.points.length === 0;
    document.getElementById('planner-clear-middle-btn').disabled = this.points.length <= 2;

    const isAllTerrain = document.getElementById('mode-all-terrain').checked;
    document.getElementById('sub-algorithm-select').disabled = !isAllTerrain;
  }

  clearAll() {
    if (this.isProcessing) return;
    this.clearMarkers();
    this.points = [];
    if (this.pathPolyline) map.removeLayer(this.pathPolyline);
    this.pathPolyline = null;
    this.hasPlannedPath = false;
    const seeded = this.startAtRobot && this._moveStartToRobot();
    this.updateUI();
    setStatus(seeded ? 'Path cleared; start back at the robot' : 'Path cleared', 'text-info');
  }

  clearMiddle() {
    if (this.isProcessing) return;
    if (this.points.length <= 2) return;
    const start = this.points[0];
    const end = this.points[this.points.length - 1];
    this.points = [start, end];
    this.redraw();
    this.updateUI();
    setStatus('Middle points cleared', 'text-info');
  }

  handleGpxImport(event) {
    const file = event.target.files[0];
    if (!file) return;
    this.loadGpxFile(file);
    event.target.value = '';
  }

  async loadGpxFile(file) {
    setStatus(`Importing ${file.name}...`, 'text-info');
    this.parseAndLoadGpx(await file.text());
  }

  parseAndLoadGpx(xmlText) {
    try {
      const parser = new DOMParser();
      const xml = parser.parseFromString(xmlText, 'text/xml');
      const wpts = xml.getElementsByTagName('wpt');
      const trkpts = xml.getElementsByTagName('trkpt');

      const nodes = wpts.length > 0 ? wpts : trkpts;
      const points = Array.from(nodes, n => ({
        lat: parseFloat(n.getAttribute('lat')),
        lon: parseFloat(n.getAttribute('lon'))
      }));

      if (points.length === 0) {
        setStatus('No points found in GPX', 'text-warning');
        return;
      }

      this.clearAll();
      this.points = points.map(p => ({ ...p, marker: null }));
      this.redraw();
      this.updateUI();
      setStatus(`Imported ${points.length} points`, 'text-success');

      // Zoom to fit
      const bounds = L.latLngBounds(this.points.map(p => [p.lat, p.lon]));
      map.fitBounds(bounds, { padding: [50, 50] });

    } catch (err) {
      setStatus('Failed to parse GPX', 'text-danger');
      console.error(err);
    }
  }

  setupDragAndDrop() {
    const dropzone = document.getElementById('dropzone');
    window.addEventListener('dragenter', (e) => {
      dropzone.classList.add('active');
    });
    dropzone.addEventListener('dragleave', (e) => {
      if (e.target === dropzone) dropzone.classList.remove('active');
    });
    dropzone.addEventListener('dragover', (e) => e.preventDefault());
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('active');

      const file = e.dataTransfer.files[0];
      if (!file) return;
      const ext = file.name.toLowerCase().split('.').pop();

      if (ext === 'mapdata') {
        handleMapdataUpload(file);
      } else if (ext === 'gpx') {
        if (currentAppMode === 'planner') {
          this.loadGpxFile(file);
        } else {
          handleGpxMapCreation(file);
        }
      }
    });
  }

  async replanPath() {
    if (STATIC_BASE) {
      setStatus('Path planning needs the backend — run map_data_viewer locally', 'text-warning');
      return;
    }
    if (this.isProcessing) {
      if (this.abortController) {
        this.abortController.abort();
      }
      this.cancelReplan();
      return;
    }

    // Plan from where the robot is now, not where it was when the toggle went on
    if (this.startAtRobot) this._moveStartToRobot();

    if (this.points.length < 2) {
      setStatus('At least 2 points required', 'text-warning');
      return;
    }
    if (!currentFile) {
      setStatus('Please load a map file first', 'text-warning');
      return;
    }

    this.isProcessing = true;
    this.abortController = new AbortController();
    this.currentReplanId = 'replan-' + Date.now();
    this.updateProcessingUI(true);
    setStatus('Replanning path...', 'text-warning');

    const allowedWays = this._allowedWays();

    const algorithm = document.querySelector('input[name="plan-mode"]:checked').value;
    const subAlgorithm = document.getElementById('sub-algorithm-select').value;
    const cellSize = parseFloat(document.getElementById('planner-cell-size').value) || 0.25;
    const inflate = parseFloat(document.getElementById('planner-inflate').value) || 0.25;
    const gridCostWeight = parseFloat(document.getElementById('planner-grid-cost-weight').value) || 5.0;
    const simplify = document.getElementById('planner-simplify').checked;
    const smooth = document.getElementById('planner-smooth').checked;

    try {
      const res = await api('POST', '/api/create_replan', {
        signal: this.abortController.signal,
        body: {
          points: this.points.map(p => [p.lat, p.lon]),
          file: currentFile,
          allowed_ways: allowedWays,
          algorithm: algorithm,
          sub_algorithm: subAlgorithm,
          cell_size: cellSize,
          inflate_obstacles: inflate,
          grid_cost_weight: gridCostWeight,
          simplify_path: simplify,
          smooth_path: smooth,
          highway_costs: this.highwayCosts,
          surface_costs: this.surfaceCosts,
          traversability: this.rulesYaml ?? undefined,
          transfer_id: this.currentReplanId
        }
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      if (data.retrieveNum === 1) {
        if (data.status === 'cancelled') {
          setStatus('Replanning cancelled', 'text-info');
        } else {
          setStatus('Replanning failed: path not found', 'text-danger');
        }
      } else if (data.retrieveNum === -1) {
        setStatus('Path is already optimal', 'text-success');
      } else {
        this.hasPlannedPath = true;
        this.clearMarkers();
        this.points = data.newPath.map(p => ({ lat: p[0], lon: p[1], marker: null }));
        this.redraw();
        this.updateUI();
        setStatus('Path replanned successfully', 'text-success');
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setStatus('Replanning cancelled', 'text-info');
      } else {
        setStatus(`Error: ${err.message}`, 'text-danger');
      }
    } finally {
      this.isProcessing = false;
      this.abortController = null;
      this.currentReplanId = null;
      this.updateProcessingUI(false);
    }
  }

  async cancelReplan() {
    if (STATIC_BASE) return;
    if (!this.currentReplanId) return;
    setStatus('Cancelling...', 'text-warning');
    try {
      await api('POST', '/api/cancel_replan', { body: { transfer_id: this.currentReplanId } });
    } catch (err) {
      console.error('Cancel failed:', err);
    }
  }

  updateProcessingUI(processing) {
    const btn = document.getElementById('replan-btn');
    btn.classList.toggle('btn-primary', !processing);
    btn.classList.toggle('btn-danger', processing);

    if (processing) {
      btn.innerHTML = `
        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        <span class="ms-1">Planning...</span>
        <span class="ms-auto" style="font-weight:bold; font-size:1.1rem;">&times;</span>
      `;
    } else {
      btn.textContent = 'Replan Path';
    }
  }

  exportToGPX() {
    const gpx = this.generateGPX();
    const blob = new Blob([gpx], { type: 'application/gpx+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'planned_path.gpx';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setStatus('GPX exported', 'text-success');
  }

  generateGPX() {
    const fmt = document.querySelector('input[name="gpx-format"]:checked')?.value ?? 'track';
    if (fmt === 'track') {
      const trkpts = this.points.map(p => `      <trkpt lat="${p.lat}" lon="${p.lon}"></trkpt>`).join('\n');
      return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="MapDataPlanner">
  <trk>
    <trkseg>
${trkpts}
    </trkseg>
  </trk>
</gpx>`;
    }
    const pts = this.points.map(p => `  <wpt lat="${p.lat}" lon="${p.lon}"></wpt>`).join('\n');
    return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="MapDataPlanner">
${pts}
</gpx>`;
  }

  async shareViaWormhole() {
    if (STATIC_BASE) {
      setStatus('Wormhole sharing needs the backend — run map_data_viewer locally', 'text-warning');
      return;
    }
    const gpx = this.generateGPX();
    setStatus('Creating wormhole...', 'text-warning');
    try {
      const res = await api('POST', '/api/create_wormhole', { body: { gpx } });
      const data = await res.json();
      if (data.success) {
        this.currentWormholeId = data.transfer_id;
        this.showWormholeDialog(data.code);
      } else {
        alert('Wormhole failed: ' + data.message);
      }
    } catch (err) {
      alert('Error: ' + err.message);
    }
  }

  showWormholeDialog(code) {
    document.getElementById('wormhole-code').textContent = code;
    const modalEl = document.getElementById('wormhole-modal');
    modalEl.addEventListener('hidden.bs.modal', () => {
      if (this.currentWormholeId) {
        api('POST', '/api/cancel_wormhole', { body: { transfer_id: this.currentWormholeId } }).catch(err => console.error('Wormhole cancel failed:', err));
      }
      this.currentWormholeId = null;
    }, { once: true });
    bootstrap.Modal.getOrCreateInstance(modalEl).show();
  }

  /** Robotour goal QR (geo:lat,lon) for one waypoint, full screen for the robot camera. */
  showQr(point) {
    const q = `lat=${point.lat.toFixed(7)}&lon=${point.lon.toFixed(7)}`;
    const text = `geo:${point.lat.toFixed(7)},${point.lon.toFixed(7)}`;
    document.getElementById('qr-modal-text').textContent = text;
    // `caption=` empty: the code alone, so the caption below is real text and
    // stays sharp however far the code is scaled up.
    document.getElementById('qr-modal-img').src = `/api/qr.svg?${q}&caption=`;
    document.getElementById('qr-modal-caption').textContent = text;
    document.getElementById('qr-modal-download-svg').href = `/api/qr.svg?${q}&download=1`;
    document.getElementById('qr-modal-download').href = `/api/qr?${q}&scale=20&download=1`;
    bootstrap.Modal.getOrCreateInstance(document.getElementById('qr-modal')).show();
  }

  showContextMenu(point, latlng) {
    showMenu(latlng, [
      ['🗑️ Delete Point', () => {
        this.points = this.points.filter(p => p !== point);
        this.redraw();
        this.updateUI();
      }],
      ['🔳 QR code', () => this.showQr(point)],
    ], { minWidth: 120 });
  }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
  plannerMode = new PlannerMode();
});
