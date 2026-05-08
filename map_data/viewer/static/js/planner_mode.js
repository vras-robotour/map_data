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

    this.init();
  }

  init() {
    this.bindEvents();
    this.setupDragAndDrop();
  }

  enable() {
    this.active = true;
    this.redraw();
    this.updateUI();
    // Enable map click for adding points
    map.on('click', this.handleMapClick, this);
  }

  disable() {
    this.active = false;
    map.off('click', this.handleMapClick, this);
    this.clearMarkers();
    if (this.pathPolyline) {
      map.removeLayer(this.pathPolyline);
      this.pathPolyline = null;
    }
  }

  bindEvents() {
    document.getElementById('planner-clear-btn').addEventListener('click', () => this.clearAll());
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
      marker.on('mousedown', (e) => {
        L.DomEvent.stopPropagation(e);
        dragging = true;
        this.isDragging = true;
        map.dragging.disable();
      });

      map.on('mousemove', (e) => {
        if (dragging) {
          marker.setLatLng(e.latlng);
          p.lat = e.latlng.lat;
          p.lon = e.latlng.lng;
          this.hasPlannedPath = false; // Point moved, path is invalid
          this.drawPathLine();
        }
      });

      const stopDrag = () => {
        if (dragging) {
          dragging = false;
          this.isDragging = false;
          this.lastDragEndTime = Date.now();
          map.dragging.enable();
          this.redraw(); // snap or final update
        }
      };

      map.on('mouseup', stopDrag);
      marker.on('mouseup', stopDrag);

      // Right click to delete
      marker.on('contextmenu', (e) => {
        L.DomEvent.stopPropagation(e);
        this.showContextMenu(p, e.latlng);
      });

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
                map.latLngToLayerPoint([this.points[i+1].lat, this.points[i+1].lon])
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
    this.points.forEach(p => {
      if (p.marker) map.removeLayer(p.marker);
    });
  }

  updateUI() {
    const countEl = document.getElementById('point-count');
    if (this.points.length === 0) {
      countEl.textContent = 'Click map to start drawing a path';
    } else {
      let totalDist = 0;
      for (let i = 0; i < this.points.length - 1; i++) {
        const p1 = L.latLng(this.points[i].lat, this.points[i].lon);
        const p2 = L.latLng(this.points[i+1].lat, this.points[i+1].lon);
        totalDist += p1.distanceTo(p2);
      }
      
      let distStr = totalDist > 1000 
        ? `${(totalDist / 1000).toFixed(2)} km` 
        : `${totalDist.toFixed(1)} m`;

      countEl.textContent = `${this.points.length} points | ${distStr}`;
    }
    document.getElementById('planner-export-btn').disabled = this.points.length < 2;
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
        if (currentAppMode === 'planner') dropzone.classList.add('active');
    });
    dropzone.addEventListener('dragleave', (e) => {
        if (e.target === dropzone) dropzone.classList.remove('active');
    });
    dropzone.addEventListener('dragover', (e) => e.preventDefault());
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        if (currentAppMode !== 'planner') return;
        const file = e.dataTransfer.files[0];
        if (file && file.name.toLowerCase().endsWith('.gpx')) {
            this.loadGpxFile(file);
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
    const cellSize = parseFloat(document.getElementById('planner-cell-size').value) || 0.25;
    const inflate = parseFloat(document.getElementById('planner-inflate').value) || 0.25;
    const simplify = document.getElementById('planner-simplify').checked;

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
          cell_size: cellSize,
          inflate_obstacles: inflate,
          simplify_path: simplify,
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
    
    L.popup({ minWidth: 120, className: 'planner-popup' })
      .setLatLng(latlng)
      .setContent(container)
      .openOn(map);
  }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
  plannerMode = new PlannerMode();
});
