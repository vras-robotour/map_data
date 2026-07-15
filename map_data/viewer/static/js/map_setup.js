// ── Map setup ────────────────────────────────────────────────────────────────
const map = L.map('map', { preferCanvas: true }).setView([50.08, 14.42], 5);

const baseLayers = {
    'OpenStreetMap': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        maxZoom: 19,
    }),
    'Satellite': L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, GIS User Community',
        maxZoom: 19,
    }),
};
(baseLayers[localStorage.getItem('baseLayer')] || baseLayers['OpenStreetMap']).addTo(map);
L.control.layers(baseLayers, null, { position: 'bottomleft' }).addTo(map);
map.on('baselayerchange', e => localStorage.setItem('baseLayer', e.name));

drawnItems.addTo(map);

// Restore the canonical draw order after any layer is toggled back on.
// bringToFront() on a GeoJSON/FeatureGroup re-queues all its sub-layers at
// the end of the canvas renderer's draw list (later = drawn on top).
// Iterating in the desired bottom→top order leaves each successive layer
// on top of the previous one, reproducing the initial load order.
function _enforceLayerOrder() {
    ['road', 'footway', 'barrier', 'crossroad', 'waypoint'].forEach(cat => {
        const layer = geoLayers[cat];
        if (layer && map.hasLayer(layer)) layer.bringToFront();
    });
    // Ensure robot is on top
    if (typeof trackerMode !== 'undefined' && trackerMode.showRobot) {
        trackerMode.showRobot(); 
    }
    if (map.hasLayer(drawnItems)) drawnItems.bringToFront();
}

// ── Mode switching ────────────────────────────────────────────────────────────
function setMode(newMode, commit = true) {
    const prev = currentMode;
    currentMode = newMode;

    deselectCurrent();

    // Commit / disable previous mode
    if (commit) {
        if (prev === 'edit') disableAnnotationEditMode();
        if (prev === 'delete' && deleteHandler) {
            try { deleteHandler.save(); deleteHandler.disable(); } catch (_) { }
        }
    }
    if (prev === 'add' && drawControl) {
        try { map.removeControl(drawControl); } catch (_) { }
    }
    if (prev === 'fetch' && fetchRectDraw) {
        try { fetchRectDraw.disable(); } catch (_) { }
    }
    if (prev === 'path' && pathLineDraw) {
        try { pathLineDraw.disable(); } catch (_) { }
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
            // Enable snapping
            const targets = getSnappableLayers();
            pathLineDraw.enable(); // needs to be enabled to have _mouseMarker
            if (pathLineDraw._mouseMarker) {
                const snap = new L.Handler.MarkerSnap(map, pathLineDraw._mouseMarker);
                targets.forEach(t => snap.addGuideLayer(t));
                snap.enable();
            }
        } else {
            pathLineDraw.enable();
        }
        setStatus('Click to place path nodes — double-click to finish (Snapping enabled)', 'text-info');
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

  document.getElementById('clear-btn')?.addEventListener('click', () => {
    clearMapData();
  });

  document.getElementById('feature-search')?.addEventListener('input', e => {
    filterLayers(e.target.value);
  });

    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    });

    document.getElementById('export-btn')?.addEventListener('click', () => {
        if (currentFile)
            window.location = `/api/export?file=${encodeURIComponent(currentFile)}`;
    });

    document.getElementById('export-geojson-btn')?.addEventListener('click', () => {
        if (currentFile)
            window.location = `/api/export/geojson?file=${encodeURIComponent(currentFile)}`;
    });

    document.getElementById('gpx-create-btn')?.addEventListener('click', () => {
        document.getElementById('gpx-create-input').click();
    });

    document.getElementById('gpx-create-input')?.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleGpxMapCreation(file);
        e.target.value = '';
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
            const data = await fetchAreaApi({
                ...pendingBbox,
                name,
                grid_margin: parseFloat(document.getElementById('fetch-grid-margin')?.value) || 150,
                obstacle_radius: parseFloat(document.getElementById('fetch-obstacle-radius')?.value) || 2.0,
                buffer_widths: {
                    road: parseFloat(document.getElementById('fetch-buf-road')?.value) || 7.0,
                    footway: parseFloat(document.getElementById('fetch-buf-footway')?.value) || 3.0,
                    barrier: parseFloat(document.getElementById('fetch-buf-barrier')?.value) || 2.0,
                },
            });
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
                if (cb.checked) { drawnItems.addTo(map); _enforceLayerOrder(); }
                else map.removeLayer(drawnItems);
                return;
            }
            if (cb.dataset.layer === 'robot') {
                if (typeof trackerMode !== 'undefined') {
                    if (cb.checked) trackerMode.showRobot();
                    else trackerMode.hideRobot();
                }
                return;
            }
            const layer = geoLayers[cb.dataset.layer];
            if (!layer) return;
            if (cb.checked) { layer.addTo(map); _enforceLayerOrder(); }
            else map.removeLayer(layer);
        });
    });

    map.on('click', () => {
        if (currentMode === 'view') {
            clearNodes();
            if (currentClickedLayer) {
                const oldCat = currentClickedLayer._osmCat;
                currentClickedLayer.setStyle(oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options._ann_id)));
                currentClickedLayer = null;
            }
            currentClickedFeature = null;
            const el = document.getElementById('props-content');
            if (el) el.innerHTML = '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
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
                annBaselineGeoms[ann.id] = JSON.parse(JSON.stringify(ann.geometry));
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
            const cat = btn.dataset.subtypeToggle;
            const panel = document.getElementById(`subfilter-${cat}`);
            const open = panel.style.display !== 'block';
            panel.style.display = open ? 'block' : 'none';
            btn.textContent = open ? '▲' : '▼';
        });
    });

    document.addEventListener('keydown', e => {
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
        if (e.target.isContentEditable) return;

        switch (e.key) {
            case 'v': case 'V': {
                if (currentAppMode !== 'viewer') return;
                const prevLayer = currentClickedLayer;
                const prevFeature = currentClickedFeature;
                setMode('view');
                if (prevLayer && prevFeature) {
                    currentClickedLayer = prevLayer;
                    currentClickedFeature = prevFeature;
                    prevLayer._osmCat = prevFeature.properties.category;
                    prevLayer.setStyle(HIGHLIGHT_STYLES[prevFeature.properties.category]);
                    showProps(prevFeature.properties, prevFeature);
                }
                break;
            }
            case 'e': case 'E': {
                if (currentAppMode !== 'viewer') return;
                const prevLayer = currentClickedLayer;
                const prevFeature = currentClickedFeature;
                setMode('edit');
                const cat = prevFeature?.properties?.category;
                if (prevLayer && prevFeature && cat && cat !== 'crossroad') {
                    currentClickedLayer = prevLayer;
                    currentClickedFeature = prevFeature;
                    prevLayer._osmCat = cat;
                    prevLayer.setStyle(HIGHLIGHT_STYLES[cat]);
                    loadNodesForEditing(prevFeature, prevLayer);
                }
                break;
            }
            case 'a': case 'A': 
                if (currentAppMode === 'viewer') setMode('add'); 
                break;
            case 'd': case 'D': 
                if (currentAppMode === 'viewer') setMode('delete'); 
                break;
            case 'f': case 'F': 
                if (currentAppMode === 'viewer') setMode('fetch'); 
                break;
            case 'p': case 'P': 
                if (currentAppMode === 'viewer') setMode('path'); 
                break;
            case 'g': case 'G':
                if (currentAppMode === 'viewer') {
                    document.getElementById('gpx-create-input')?.click();
                }
                break;
            case 'n': case 'N':
                if (currentAppMode === 'viewer' && currentMode === 'view') toggleNodes();
                break;
            case 'h': case 'H':
                if (currentAppMode === 'viewer' && currentMode === 'view' && currentClickedFeature &&
                    ['road', 'footway', 'barrier'].includes(currentClickedFeature.properties.category)) {
                    if (hiddenWayIds.has(currentClickedFeature.properties.id))
                        showWay(currentClickedFeature.properties.id);
                    else
                        hideCurrentWay();
                }
                break;
            case 'Escape':
                if (currentAppMode === 'viewer') {
                    if (currentClickedLayer || currentClickedFeature) {
                        if (currentClickedLayer) {
                            const oldCat = currentClickedLayer._osmCat;
                            currentClickedLayer.setStyle(
                                oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options._ann_id))
                            );
                            currentClickedLayer = null;
                        }
                        currentClickedFeature = null;
                        clearNodes();
                        const propsEl = document.getElementById('props-content');
                        if (propsEl) propsEl.innerHTML = '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
                    } else if (currentMode !== 'view') {
                        setMode('view');
                    }
                }
                break;
            case 'Delete':
            case 'Backspace':
                if (currentAppMode === 'viewer') {
                    if (currentMode === 'edit') {
                        e.preventDefault();
                        deleteSelectedAnnotation();
                    } else if (currentMode === 'view') {
                        if (currentClickedLayer && currentClickedLayer.options && currentClickedLayer.options._ann_id) {
                            e.preventDefault();
                            removeAnnotationById(currentClickedLayer.options._ann_id);
                            currentClickedLayer = null;
                            const propsEl = document.getElementById('props-content');
                            if (propsEl) propsEl.innerHTML = '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
                        } else if (currentClickedFeature &&
                            ['road', 'footway', 'barrier'].includes(currentClickedFeature.properties.category)) {
                            e.preventDefault();
                            if (selectedNodeIndex >= 0 && currentNodes[selectedNodeIndex]) {
                                deleteCurrentNode(currentClickedFeature.properties.id, currentNodes[selectedNodeIndex].id);
                            } else {
                                deleteCurrentWay();
                            }
                        }
                    }
                }
                break;
        }
    });
}

initApp();
