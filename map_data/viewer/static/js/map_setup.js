// ── Map setup ────────────────────────────────────────────────────────────────
const map = L.map('map', { preferCanvas: true }).setView([50.08, 14.42], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19,
}).addTo(map);

drawnItems.addTo(map);

// ── Mode switching ────────────────────────────────────────────────────────────
function setMode(newMode, commit = true) {
  const prev = currentMode;
  currentMode = newMode;

  clearNodes();
  if (currentClickedLayer) {
    currentClickedLayer.setStyle(STYLES[currentClickedLayer._osmCat]);
    currentClickedLayer = null;
  }
  currentClickedFeature = null;

  // Commit / disable previous mode
  if (commit) {
    if (prev === 'edit') disableAnnotationEditMode();
    if (prev === 'delete' && deleteHandler) {
      try { deleteHandler.save(); deleteHandler.disable(); } catch (_) {}
    }
  }
  if (prev === 'add' && drawControl) {
    try { map.removeControl(drawControl); } catch (_) {}
  }
  if (prev === 'fetch' && fetchRectDraw) {
    try { fetchRectDraw.disable(); } catch (_) {}
  }
  if (prev === 'path' && pathLineDraw) {
    try { pathLineDraw.disable(); } catch (_) {}
  }

  // Update button highlight
  document.querySelectorAll('.mode-btn').forEach(btn =>
    btn.classList.toggle('active', btn.dataset.mode === newMode)
  );

  // Cursor feedback
  const cursors = { view: '', edit: 'move', add: 'crosshair', delete: 'pointer', fetch: 'crosshair' };
  map.getContainer().style.cursor = cursors[newMode] ?? '';

  // Activate new mode
  if (newMode === 'add') {
    initDrawControl();
    drawControl.addTo(map);
  } else if (newMode === 'edit') {
    enableAnnotationEditMode();
  } else if (newMode === 'delete') {
    if (!deleteHandler) initHandlers();
    deleteHandler.enable();
  } else if (newMode === 'fetch') {
    if (!fetchRectDraw) {
      fetchRectDraw = new L.Draw.Rectangle(map, {
        shapeOptions: { color: '#00aaff', fillColor: '#00aaff', fillOpacity: 0.1 },
      });
    }
    fetchRectDraw.enable();
    setStatus('Draw a bounding box on the map to fetch OSM data…', 'text-info');
  } else if (newMode === 'path') {
    if (!pathLineDraw) {
      pathLineDraw = new L.Draw.Polyline(map, {
        shapeOptions: { ...STYLES.path },
        showLength: true,
      });
    }
    pathLineDraw.enable();
    setStatus('Click to place path nodes — double-click to finish', 'text-info');
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function initApp() {
  initHandlers();
  setupDrawEvents();
  const data = await fetchFileList();
  const sel = document.getElementById('file-select');
  if (sel) {
    data.mapdata.forEach(f => {
      if (![...sel.options].some(o => o.value === f)) {
        sel.appendChild(new Option(f, f));
      }
    });
  }

  // Event Listeners
  document.getElementById('load-btn')?.addEventListener('click', () => {
    const file = document.getElementById('file-select').value;
    if (file) loadMapData(file);
  });

  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });

  document.getElementById('export-btn')?.addEventListener('click', () => {
    if (currentFile)
      window.location = `/api/export?file=${encodeURIComponent(currentFile)}`;
  });

  document.getElementById('fetch-area-submit')?.addEventListener('click', async () => {
    const name = document.getElementById('area-name-input').value.trim();
    if (!name) {
      document.getElementById('area-name-input').focus();
      return;
    }
    bootstrap.Modal.getInstance(document.getElementById('fetch-area-modal')).hide();
    setStatus('Fetching & parsing OSM data… (may take 1–2 min)', 'text-warning');

    try {
      const data = await fetchAreaApi({ ...pendingBbox, name });
      setStatus(
        `Fetched: ${data.roads} roads, ${data.footways} footways, ${data.barriers} barriers`,
        'text-success'
      );
      const sel = document.getElementById('file-select');
      if (sel && ![...sel.options].some(o => o.value === data.filename)) {
        sel.appendChild(new Option(data.filename, data.filename));
      }
      if (sel) sel.value = data.filename;
      await loadMapData(data.filename);
    } catch (err) {
      setStatus(`Fetch failed: ${err.message}`, 'text-danger');
    }
  });

  document.getElementById('area-name-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('fetch-area-submit').click();
  });

  document.querySelectorAll('[data-layer]').forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.dataset.layer === 'annotation') {
        cb.checked ? drawnItems.addTo(map) : map.removeLayer(drawnItems);
        return;
      }
      const layer = geoLayers[cb.dataset.layer];
      if (!layer) return;
      cb.checked ? layer.addTo(map) : map.removeLayer(layer);
    });
  });

  map.on('click', () => {
    if (currentMode === 'view') {
      clearNodes();
      if (currentClickedLayer) {
        currentClickedLayer.setStyle(STYLES[currentClickedLayer._osmCat]);
        currentClickedLayer = null;
      }
      currentClickedFeature = null;
      const el = document.getElementById('props-content');
      if (el) el.innerHTML = '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
    }
  });

  document.getElementById('way-edit-add-prop-btn')?.addEventListener('click', () => {
    const div = document.createElement('div');
    div.className = 'd-flex gap-1 mb-1';
    div.innerHTML = `
      <input class="form-control form-control-sm bg-dark text-light border-secondary we-key"
             placeholder="key" style="flex:1;font-size:0.75rem;">
      <input class="form-control form-control-sm bg-dark text-light border-secondary we-val"
             placeholder="value" style="flex:1;font-size:0.75rem;">
      <button type="button" class="btn btn-sm btn-outline-danger px-1"
              onclick="this.closest('.d-flex').remove()">×</button>`;
    document.getElementById('way-edit-props').appendChild(div);
  });

  document.getElementById('way-edit-save')?.addEventListener('click', async () => {
    if (!editingWayId || !currentFile) return;
    const tags = {};
    document.querySelectorAll('#way-edit-props .d-flex').forEach(row => {
      const k = row.querySelector('.we-key').value.trim();
      const v = row.querySelector('.we-val').value.trim();
      if (k) tags[k] = v;
    });
    const savedWayId = editingWayId;
    const cat = currentClickedFeature?.properties?.category || 'unknown';
    const lbl = currentClickedFeature?.properties?.tags?.highway
             || currentClickedFeature?.properties?.tags?.barrier || '';
    bootstrap.Modal.getInstance(document.getElementById('way-edit-modal')).hide();
    const res = await updateWayTagsApi(currentFile, savedWayId, tags, cat, lbl);
    if (res.ok) {
      setStatus('Properties updated', 'text-success');
      await _reloadWay(savedWayId);
    } else {
      setStatus('Save failed', 'text-danger');
    }
  });

  document.getElementById('ann-add-prop-btn')?.addEventListener('click', () => {
    const div = document.createElement('div');
    div.className = 'd-flex gap-1 mb-1';
    div.innerHTML = `
      <input class="form-control form-control-sm bg-dark text-light border-secondary ep-key"
             placeholder="key" style="flex:1;font-size:0.75rem;">
      <input class="form-control form-control-sm bg-dark text-light border-secondary ep-val"
             placeholder="value" style="flex:1;font-size:0.75rem;">
      <button type="button" class="btn btn-sm btn-outline-danger px-1"
              onclick="this.closest('.d-flex').remove()">×</button>`;
    document.getElementById('ann-extra-props').appendChild(div);
  });

  document.getElementById('ann-detail-save')?.addEventListener('click', async () => {
    const { type, props } = _collectAnnForm();
    bootstrap.Modal.getInstance(document.getElementById('ann-detail-modal')).hide();

    if (editingAnnId) {
      const ann = annotations.find(a => a.id === editingAnnId);
      if (!ann) return;
      try {
        const updatedAnn = await updateAnnotationApi(currentFile, editingAnnId, ann.geometry, type, props);
        Object.assign(ann, updatedAnn);
        renderAnnotationLayer();
        renderAnnotationList();
        setStatus('Annotation updated', 'text-success');
      } catch (err) {
        setStatus('Update failed', 'text-danger');
      }
    } else {
      try {
        const ann = await createAnnotationApi(currentFile, type, pendingAnnGeom, props);
        annotations.push(ann);
        addAnnotationToLayer(ann);
        renderAnnotationList();
        setStatus('Annotation added', 'text-success');
      } catch (err) {
        setStatus('Add failed', 'text-danger');
      }
      pendingAnnGeom = null;
    }
  });

  document.querySelectorAll('[data-subtype-toggle]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      const cat   = btn.dataset.subtypeToggle;
      const panel = document.getElementById(`subfilter-${cat}`);
      const open  = panel.style.display !== 'block';
      panel.style.display = open ? 'block' : 'none';
      btn.textContent     = open ? '▲' : '▼';
    });
  });

  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
    if (e.target.isContentEditable) return;

    switch (e.key) {
      case 'v': case 'V':      setMode('view');   break;
      case 'e': case 'E':      setMode('edit');   break;
      case 'a': case 'A':      setMode('add');    break;
      case 'd': case 'D':      setMode('delete'); break;
      case 'f': case 'F':      setMode('fetch');  break;
      case 'p': case 'P':      setMode('path');   break;
      case 'n': case 'N':
        if (currentMode === 'view') toggleNodes();
        break;
      case 'Escape':
        if (currentMode !== 'add' && currentMode !== 'fetch') setMode('view');
        break;
      case 'Delete':
      case 'Backspace':
        if (currentMode === 'edit') { e.preventDefault(); deleteSelectedAnnotation(); }
        break;
    }
  });
}

initApp();
