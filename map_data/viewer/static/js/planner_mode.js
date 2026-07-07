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
    this._mapDragListeners = [];

    this.init();
  }

  async init() {
    this.bindEvents();
    this.setupDragAndDrop();
    await this.fetchDefaults();
  }

  async fetchDefaults() {
    try {
      const res = await fetch('/api/planner_defaults');
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
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
      const gridMarginDefault = data.grid_margin ?? 150;
      const obstacleRadiusDefault = data.obstacle_radius ?? 2.0;
      const bwRoad = data.buffer_widths?.road ?? 7.0;
      const bwFootway = data.buffer_widths?.footway ?? 3.0;
      const bwBarrier = data.buffer_widths?.barrier ?? 2.0;
      for (const prefix of ['fetch', 'gpx', 'planner-fetch']) {
        const gm = document.getElementById(`${prefix}-grid-margin`);
        const or = document.getElementById(`${prefix}-obstacle-radius`);
        const br = document.getElementById(`${prefix}-buf-road`);
        const bf = document.getElementById(`${prefix}-buf-footway`);
        const bb = document.getElementById(`${prefix}-buf-barrier`);
        if (gm) gm.value = gridMarginDefault;
        if (or) or.value = obstacleRadiusDefault;
        if (br) br.value = bwRoad;
        if (bf) bf.value = bwFootway;
        if (bb) bb.value = bwBarrier;
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
    this.updateUI(); // Ensure UI state is correct
    // Enable map click for adding points
    map.on('click', this.handleMapClick, this);
  }

  disable() {
    this.active = false;
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

    document.getElementById('planner-fetch-submit').addEventListener('click', () => this.handlePlannerAutoFetch());
    document.getElementById('planner-fetch-name-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') this.handlePlannerAutoFetch();
    });
  }

  async handlePlannerAutoFetch() {
    const name = document.getElementById('planner-fetch-name-input').value.trim();
    if (!name) {
      document.getElementById('planner-fetch-name-input').focus();
      return;
    }
    bootstrap.Modal.getInstance(document.getElementById('planner-fetch-modal')).hide();

    // Calculate BBox for the current points
    let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
    this.points.forEach(p => {
      if (p.lat < minLat) minLat = p.lat;
      if (p.lat > maxLat) maxLat = p.lat;
      if (p.lon < minLon) minLon = p.lon;
      if (p.lon > maxLon) maxLon = p.lon;
    });

    // Add a small margin (approx 50m in degrees)
    const margin = 0.0005;
    const bbox = {
      min_lat: minLat - margin,
      max_lat: maxLat + margin,
      min_lon: minLon - margin,
      max_lon: maxLon + margin,
      name: name,
      grid_margin: parseFloat(document.getElementById('planner-fetch-grid-margin')?.value) || 150,
      obstacle_radius: parseFloat(document.getElementById('planner-fetch-obstacle-radius')?.value) || 2.0,
      buffer_widths: {
        road: parseFloat(document.getElementById('planner-fetch-buf-road')?.value) || 7.0,
        footway: parseFloat(document.getElementById('planner-fetch-buf-footway')?.value) || 3.0,
        barrier: parseFloat(document.getElementById('planner-fetch-buf-barrier')?.value) || 2.0,
      },
    };

    setStatus('Fetching & parsing OSM data for the area...', 'text-warning');
    this.updateProcessingUI(true);
    this.isProcessing = true;

    try {
      const data = await fetchAreaApi(bbox);
      setStatus(`Map created: ${data.filename}. Loading and planning...`, 'text-success');

      // Add to file select if it's there
      const sel = document.getElementById('file-select');
      if (sel && ![...sel.options].some(o => o.value === data.filename)) {
        sel.appendChild(new Option(data.filename, data.filename));
      }
      if (sel) sel.value = data.filename;

      // Load the map data
      await loadMapData(data.filename, { preserveView: true });

      // After loading, proceed with replan
      this.isProcessing = false; // Reset so replanPath can proceed
      this.replanPath();
    } catch (err) {
      setStatus(`Fetch failed: ${err.message}`, 'text-danger');
      this.updateProcessingUI(false);
      this.isProcessing = false;
    }
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
    if (this.defaults && this.defaults.highway_costs) {
      this.highwayCosts = { ...this.defaults.highway_costs };
      this.surfaceCosts = { ...this.defaults.surface_costs };
    } else {
      // Hardcoded fallback if everything else fails
      this.highwayCosts = {
        "pedestrian": 0.0, "footway": 0.0, "path": 0.1, "living_street": 0.1,
        "track": 0.3, "service": 0.3, "residential": 0.5, "unclassified": 0.5,
        "tertiary": 0.7, "secondary": 0.9, "primary": 1.0,
      };
      this.surfaceCosts = {
        "asphalt": 0.0, "paving_stones": 0.0, "concrete": 0.0, "fine_gravel": 0.1,
        "gravel": 0.2, "dirt": 0.3, "grass": 0.5, "sand": 0.7,
      };
    }
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
    if (!currentFile) return;
    const bounds = map.getBounds();
    setStatus('Fetching cost grid...', 'text-warning');
    
    try {
      const costsJson = JSON.stringify(this.highwayCosts);
      const surfaceJson = JSON.stringify(this.surfaceCosts);
      const res = await fetch(`/api/cost_grid?file=${currentFile}&min_lat=${bounds.getSouth()}&min_lon=${bounds.getWest()}&max_lat=${bounds.getNorth()}&max_lon=${bounds.getEast()}&highway_costs=${encodeURIComponent(costsJson)}&surface_costs=${encodeURIComponent(surfaceJson)}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      
      if (geoLayers.costGrid) map.removeLayer(geoLayers.costGrid);
      
      geoLayers.costGrid = L.geoJSON(data, {
        pointToLayer: (feature, latlng) => {
          const cost = feature.properties.cost;
          if (cost >= 1.0) {
            // Hard Obstacle style: black marker
            return L.circleMarker(latlng, {
              radius: 4,
              fillColor: '#000',
              color: '#000',
              weight: 1,
              fillOpacity: 1
            });
          }
          const offPathThreshold = this.defaults.default_off_path_cost || 0.9;
          const pathCap = this.defaults.path_cost_cap || 0.85;

          if (cost >= offPathThreshold) {
             // Off-path / All-terrain style: dark gray/brown
             return L.circleMarker(latlng, {
                radius: 2,
                fillColor: '#444',
                color: '#444',
                weight: 0,
                fillOpacity: 0.4
             });
          }
          // Interpolate color from green (0) to red (pathCap)
          const normalizedCost = cost / pathCap;
          const r = Math.floor(255 * Math.min(1, normalizedCost));
          const g = Math.floor(255 * Math.max(0, 1 - normalizedCost));
          return L.circleMarker(latlng, {
            radius: 3,
            fillColor: `rgb(${r},${g},0)`,
            color: '#000',
            weight: 0.2,
            fillOpacity: 0.6
          });
        }
      }).addTo(map);
      
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
      let dragging = false;
      const onWpDown = (e) => {
        L.DomEvent.stopPropagation(e);
        dragging = true;
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

      const onMouseMove = (e) => {
        if (dragging) {
          marker.setLatLng(e.latlng);
          p.lat = e.latlng.lat;
          p.lon = e.latlng.lng;
          this.hasPlannedPath = false;
          this.drawPathLine();
        }
      };
      map.on('mousemove', onMouseMove);
      this._mapDragListeners.push({ event: 'mousemove', fn: onMouseMove });

      const stopDrag = () => {
        if (dragging) {
          dragging = false;
          this.isDragging = false;
          this.lastDragEndTime = Date.now();
          map.dragging.enable();
          this.redraw();
        }
      };

      map.on('mouseup', stopDrag);
      this._mapDragListeners.push({ event: 'mouseup', fn: stopDrag });
      marker.on('mouseup', stopDrag);

      // Right click to delete
      const onWpContext = (e) => {
        L.DomEvent.preventDefault(e);
        L.DomEvent.stopPropagation(e);
        this.showContextMenu(p, e.latlng);
      };
      marker.on('contextmenu', onWpContext);
      wpHit.on('contextmenu', onWpContext);

      this.markerLayer.addLayer(marker);
      p.marker = marker;
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

  clearMarkers() {
    for (const { event, fn } of this._mapDragListeners) {
      map.off(event, fn);
    }
    this._mapDragListeners = [];
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
    this.updateUI();
    setStatus('Path cleared', 'text-info');
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

  loadGpxFile(file) {
    setStatus(`Importing ${file.name}...`, 'text-info');
    const reader = new FileReader();
    reader.onload = (e) => {
      const gpxText = e.target.result;
      this.parseAndLoadGpx(gpxText);
    };
    reader.readAsText(file);
  }

  parseAndLoadGpx(xmlText) {
    try {
      const parser = new DOMParser();
      const xml = parser.parseFromString(xmlText, 'text/xml');
      const wpts = xml.getElementsByTagName('wpt');
      const trkpts = xml.getElementsByTagName('trkpt');
      const points = [];

      const nodes = wpts.length > 0 ? wpts : trkpts;
      for (let i = 0; i < nodes.length; i++) {
        points.push({
          lat: parseFloat(nodes[i].getAttribute('lat')),
          lon: parseFloat(nodes[i].getAttribute('lon'))
        });
      }

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
        if (typeof handleMapdataUpload === 'function') {
          handleMapdataUpload(file);
        }
      } else if (ext === 'gpx') {
        if (currentAppMode === 'planner') {
          this.loadGpxFile(file);
        } else {
          if (typeof handleGpxMapCreation === 'function') {
            handleGpxMapCreation(file);
          }
        }
      }
    });
  }

  async replanPath() {
    if (this.isProcessing) {
      if (this.abortController) {
        this.abortController.abort();
      }
      this.cancelReplan();
      return;
    }

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

    const allowedWays = [];
    if (document.getElementById('plan-footways').checked) allowedWays.push('footway');
    if (document.getElementById('plan-roads').checked) allowedWays.push('road');

    const algorithm = document.querySelector('input[name="plan-mode"]:checked').value;
    const subAlgorithm = document.getElementById('sub-algorithm-select').value;
    const cellSize = parseFloat(document.getElementById('planner-cell-size').value) || 0.25;
    const inflate = parseFloat(document.getElementById('planner-inflate').value) || 0.25;
    const gridCostWeight = parseFloat(document.getElementById('planner-grid-cost-weight').value) || 5.0;
    const simplify = document.getElementById('planner-simplify').checked;
    const smooth = document.getElementById('planner-smooth').checked;

    try {
      const res = await fetch('/api/create_replan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: this.abortController.signal,
        body: JSON.stringify({
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
          transfer_id: this.currentReplanId
        })
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
    if (!this.currentReplanId) return;
    setStatus('Cancelling...', 'text-warning');
    try {
      await fetch('/api/cancel_replan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transfer_id: this.currentReplanId })
      });
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
      btn.innerHTML = '<span class="btn-icon">✏️</span><span>Replan Path</span>';
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
    const gpx = this.generateGPX();
    setStatus('Creating wormhole...', 'text-warning');
    try {
      const res = await fetch('/api/create_wormhole', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gpx })
      });
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
    const overlay = document.createElement('div');
    overlay.className = 'dialog-overlay';
    overlay.innerHTML = `
      <div class="dialog-content">
        <h2>Share via Wormhole</h2>
        <p>Use this code on the receiving device:</p>
        <div class="wormhole-code">${code}</div>
        <p style="font-size:0.8rem;color:#6c7a9c;">The code will expire once the transfer is complete or after a timeout.</p>
        <button class="dialog-close-btn">Close</button>
      </div>
    `;
    document.body.appendChild(overlay);
    overlay.querySelector('.dialog-close-btn').onclick = () => {
      if (this.currentWormholeId) {
        fetch('/api/cancel_wormhole', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transfer_id: this.currentWormholeId })
        }).catch(err => console.error('Wormhole cancel failed:', err));
      }
      document.body.removeChild(overlay);
      this.currentWormholeId = null;
    };
  }

  showContextMenu(point, latlng) {
    const container = document.createElement('div');
    container.className = 'context-menu';
    const delBtn = document.createElement('button');
    delBtn.textContent = '🗑️ Delete Point';
    delBtn.onclick = () => {
      this.points = this.points.filter(p => p !== point);
      this.redraw();
      this.updateUI();
      map.closePopup();
    };
    container.appendChild(delBtn);

    L.popup({ minWidth: 120, className: 'planner-popup', offset: [0, -5], closeButton: false })
      .setLatLng(latlng)
      .setContent(container)
      .openOn(map);
  }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
  plannerMode = new PlannerMode();
});
