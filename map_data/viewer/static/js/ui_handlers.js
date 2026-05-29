// ── UI Handlers ───────────────────────────────────────────────────────────────

function setStatus(msg, cls = 'text-secondary') {
    const el = document.getElementById('status');
    if (!el) return;
    el.textContent = msg;
    el.className = cls + ' ms-auto';
}

// ── App Mode Switching ───────────────────────────────────────────────────────

document.querySelectorAll('.app-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        setAppMode(btn.dataset.appMode);
    });
});

function setAppMode(mode) {
    if (mode === currentAppMode) return;
    currentAppMode = mode;

    document.querySelectorAll('.app-mode-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.appMode === mode);
    });

    const sidebar = document.getElementById('sidebar');
    const propsPanel = document.getElementById('props-panel');
    const plannerPanel = document.getElementById('planner-sidebar-panel');
    const trackerPanel = document.getElementById('tracker-sidebar-panel');

    // Accessory panels
    const accessoryPanels = ['ann-panel', 'changes-panel', 'hidden-panel'];

    if (mode === 'viewer') {
        propsPanel.style.display = ''; // Revert to CSS default (block)
        plannerPanel.style.display = 'none';
        trackerPanel.style.display = 'none';

        // Enable all mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => btn.disabled = false);
        const gpxBtn = document.getElementById('gpx-create-btn');
        if (gpxBtn) gpxBtn.disabled = false;

        toggleMapInteractivity(true);

        // Robot visibility: keep if layer checked, else hide
        const robotCb = document.querySelector('[data-layer="robot"]');
        if (robotCb && !robotCb.checked && typeof trackerMode !== 'undefined') {
            trackerMode.hideRobot();
        } else if (typeof trackerMode !== 'undefined') {
            trackerMode.showRobot();
        }

        // Restore accessory panels visibility based on their content/state
        renderAnnotationList();
        renderChangesPanel();
        renderHiddenPanel();

        setStatus('Viewer Mode', 'text-info');
        if (plannerMode) plannerMode.disable();
        if (typeof trackerMode !== 'undefined' && trackerMode) trackerMode.disable();
        // Re-enable draw controls if they were active
        if (drawControl) map.addControl(drawControl);
    } else if (mode === 'planner') {
        propsPanel.style.display = 'none';
        plannerPanel.style.display = 'flex';
        trackerPanel.style.display = 'none';

        // Disable all mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => btn.disabled = true);
        const gpxBtn = document.getElementById('gpx-create-btn');
        if (gpxBtn) gpxBtn.disabled = true;

        if (currentMode !== 'view') {
            setMode('view');
        }

        toggleMapInteractivity(false);

        // Hide all accessory panels in planner mode
        accessoryPanels.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });

        setStatus('Planner Mode', 'text-warning');

        // Cleanup Viewer state
        clearNodes();
        if (currentClickedLayer && currentClickedLayer._osmCat) {
            currentClickedLayer.setStyle(STYLES[currentClickedLayer._osmCat]);
        }
        currentClickedLayer = null;
        currentClickedFeature = null;
        document.getElementById('props-content').innerHTML =
            '<span class="text-secondary" style="font-style:italic;">Click a feature to inspect</span>';

        if (drawControl) map.removeControl(drawControl);

        // Robot visibility: keep if layer checked, else hide
        const robotCb = document.querySelector('[data-layer="robot"]');
        if (robotCb && !robotCb.checked && typeof trackerMode !== 'undefined') {
            trackerMode.hideRobot();
        } else if (typeof trackerMode !== 'undefined') {
            trackerMode.showRobot();
        }

        if (typeof trackerMode !== 'undefined' && trackerMode) trackerMode.disable();
        if (plannerMode) plannerMode.enable();
    } else if (mode === 'tracker') {
        propsPanel.style.display = 'none';
        plannerPanel.style.display = 'none';
        trackerPanel.style.display = 'flex';

        // Disable all mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => btn.disabled = true);
        const gpxBtn = document.getElementById('gpx-create-btn');
        if (gpxBtn) gpxBtn.disabled = true;

        toggleMapInteractivity(false);
        accessoryPanels.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });

        setStatus('Tracker Mode', 'text-success');
        if (drawControl) map.removeControl(drawControl);
        if (plannerMode) plannerMode.disable();
        if (typeof trackerMode !== 'undefined' && trackerMode) {
            trackerMode.enable();
            trackerMode.showRobot();
        }
    }

    // Force map resize
    setTimeout(() => map.invalidateSize(), 100);
}

function renderSubtypeFilters(cat) {
    const panel = document.getElementById(`subfilter-${cat}`);
    if (!panel) return;
    const subtypes = Object.keys(subtypeLayers[cat]).sort();
    if (!subtypes.length) { panel.innerHTML = ''; return; }
    panel.innerHTML = subtypes.map(st => {
        const count = (subtypeLayers[cat][st] || []).length;
        const checked = subtypeFilters[cat][st] !== false;
        return `<label class="subtype-row">
      <input type="checkbox" class="st-check"
             data-subtype-cat="${cat}" data-subtype-val="${escHtml(st)}"
             ${checked ? 'checked' : ''}>
      <span>${escHtml(st)}</span>
      <span class="subtype-cnt">${count}</span>
    </label>`;
    }).join('');
    panel.querySelectorAll('[data-subtype-cat]').forEach(cb => {
        cb.addEventListener('change', () =>
            setSubtypeVisible(cb.dataset.subtypeCat, cb.dataset.subtypeVal, cb.checked)
        );
    });
}

function deselectCurrent() {
    if (currentClickedLayer) {
        const oldCat = currentClickedLayer._osmCat;
        currentClickedLayer.setStyle(oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options?._ann_id)));
        currentClickedLayer = null;
    }
    currentClickedFeature = null;
    document.getElementById('props-content').innerHTML = '<div class="text-secondary" style="font-style:italic;">Click a feature to inspect</div>';
    clearNodes();
}

function selectWay(feature, layer, cat) {
    if (currentClickedLayer === layer) return;
    deselectCurrent();
    layer._osmCat = cat;
    currentClickedLayer = layer;
    currentClickedFeature = feature;
    layer.setStyle(HIGHLIGHT_STYLES[cat]);
    showProps(feature.properties, feature);
}

function selectAnnotation(ann, layer) {
    if (currentClickedLayer === layer) return;
    deselectCurrent();
    const cat = ann.type === 'path' ? 'path' : 'annotation';
    layer._osmCat = cat;
    currentClickedLayer = layer;
    layer.setStyle(HIGHLIGHT_STYLES[cat]);
    showAnnProps(ann);
}

function showWayContextMenu(feature, layer, latlng, cat) {
    if (currentAppMode === 'planner') return;
    // Select it first so sidebar matches
    selectWay(feature, layer, cat);

    const props = feature.properties;
    const container = document.createElement('div');
    container.className = 'context-menu';

    if (['road', 'footway', 'barrier'].includes(props.category) && feature.geometry.type !== 'Point') {
        const nodeBtn = document.createElement('button');
        nodeBtn.innerHTML = nodeLayer ? '🙈 Hide Nodes' : '👁️ Show Nodes';
        nodeBtn.onclick = () => { map.closePopup(); toggleNodes(); };
        container.appendChild(nodeBtn);
    }

    const editBtn = document.createElement('button');
    editBtn.innerHTML = '✎ Edit Properties';
    editBtn.onclick = () => { map.closePopup(); openWayEditModal(); };
    container.appendChild(editBtn);

    if (hiddenWayIds.has(props.id)) {
        const showBtn = document.createElement('button');
        showBtn.innerHTML = '👁️ Show Object';
        showBtn.onclick = () => { map.closePopup(); showWay(props.id); };
        container.appendChild(showBtn);
    } else {
        const hideBtn = document.createElement('button');
        hideBtn.innerHTML = '🙈 Hide Object';
        hideBtn.onclick = () => { map.closePopup(); hideCurrentWay(); };
        container.appendChild(hideBtn);

        const delBtn = document.createElement('button');
        delBtn.innerHTML = '🗑️ Delete Object';
        delBtn.onclick = () => { map.closePopup(); deleteCurrentWay(); };
        container.appendChild(delBtn);
    }

    L.popup({ minWidth: 150, className: 'planner-popup', offset: [0, -5], closeButton: false })
        .setLatLng(latlng)
        .setContent(container)
        .openOn(map);
}

function showAnnotationContextMenu(ann, layer, latlng) {
    if (currentAppMode === 'planner') return;
    // Select it first so sidebar matches
    selectAnnotation(ann, layer);

    const container = document.createElement('div');
    container.className = 'context-menu';

    const editBtn = document.createElement('button');
    editBtn.innerHTML = '✎ Edit Properties';
    editBtn.onclick = () => { map.closePopup(); openAnnEditModal(ann.id); };
    container.appendChild(editBtn);

    const delBtn = document.createElement('button');
    delBtn.innerHTML = '🗑️ Delete Annotation';
    delBtn.onclick = () => { map.closePopup(); removeAnnotationById(ann.id); };
    container.appendChild(delBtn);

    L.popup({ minWidth: 150, className: 'planner-popup', offset: [0, -5], closeButton: false })
        .setLatLng(latlng)
        .setContent(container)
        .openOn(map);
}

function showProps(props, feature = null) {
    if (feature !== currentClickedFeature) {
        clearNodes();
        currentClickedFeature = feature;
    }
    const tags = props.tags || {};
    const meta = [
        ['Category', `<span class="badge bg-secondary">${props.category}</span>`],
        ['OSM ID', props.id ?? '—'],
        ['Role', props.in_out || '—'],
    ];
    const rows = [
        ...meta.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`),
        Object.keys(tags).length
            ? Object.entries(tags).map(([k, v]) =>
                `<tr><td>${escHtml(k)}</td><td>${escHtml(String(v))}</td></tr>`
            ).join('')
            : '<tr><td colspan="2" style="color:#6c7a9c;font-style:italic;">No tags</td></tr>',
    ].join('');
    const el = document.getElementById('props-content');
    el.innerHTML = `<table><tbody>${rows}</tbody></table>`;
    if (feature && ['road', 'footway', 'barrier'].includes(props.category)) {
        if (feature.geometry.type !== 'Point') {
            const label = nodeLayer ? 'Hide Nodes' : 'Show Nodes';
            el.innerHTML += `<button class="btn btn-sm btn-outline-info mt-2"
                               style="font-size:0.72rem;width:100%;"
                               onclick="toggleNodes()">${label}</button>`;
        }
        if (nodeLayer && currentNodes.length) {
            const items = currentNodes.map((n, i) => {
                const firstTag = Object.entries(n.tags || {})[0];
                const hint = firstTag
                    ? `<span class="node-tag-hint">${escHtml(firstTag[0])}: ${escHtml(String(firstTag[1]))}</span>`
                    : '';
                const sel = i === selectedNodeIndex ? ' sel' : '';
                return `<div class="node-list-item${sel}" onclick="clickNode(${i})">
          <span class="node-idx">${i + 1}</span>
          <span>${n.id}</span>${hint}
        </div>`;
            }).join('');
            el.innerHTML += `<div class="node-list">${items}</div>`;
        }
        el.innerHTML += `<button class="btn btn-sm btn-outline-secondary mt-1"
                             style="font-size:0.72rem;width:100%;"
                             onclick="openWayEditModal()">&#9998; Edit Properties</button>`;
        if (hiddenWayIds.has(props.id)) {
            el.innerHTML += `<button class="btn btn-sm btn-outline-info mt-1"
                               style="font-size:0.72rem;width:100%;"
                               onclick="showWay(${props.id})">&#128065; Show Object</button>`;
        } else {
            el.innerHTML += `<button class="btn btn-sm btn-outline-secondary mt-1"
                               style="font-size:0.72rem;width:100%;"
                               onclick="hideCurrentWay()">&#128065; Hide Object</button>`;
            el.innerHTML += `<button class="btn btn-sm btn-outline-danger mt-1"
                               style="font-size:0.72rem;width:100%;"
                               onclick="deleteCurrentWay()">&#128465; Delete Object</button>`;
        }
    }
}

function clearNodes() {
    if (nodeLayer) { map.removeLayer(nodeLayer); nodeLayer = null; }
    if (osmEditLayer) {
        osmEditLayer.off('mousedown', _onOsmWayDragDown);
        osmEditLayer = null;
    }
    if (osmDragGhost) { map.removeLayer(osmDragGhost); osmDragGhost = null; }
    nodeCount = 0;
    currentNodes = [];
    nodeMarkers = [];
    midpointMarkers = [];
    selectedNodeIndex = -1;
}

async function loadNodesForEditing(feature, layer) {
    if (!currentFile) return;
    clearNodes();
    const wayId = feature.properties.id;
    const el = document.getElementById('props-content');
    el.innerHTML = `<span class="text-secondary" style="font-size:0.8rem;">Loading nodes…</span>`;

    let data;
    try { data = await fetchWayNodes(currentFile, wayId); }
    catch (err) { setStatus('Failed to load nodes', 'text-danger'); return; }
    currentNodes = data.nodes;
    nodeCount = currentNodes.length;

    osmEditLayer = layer;
    layer.on('mousedown', _onOsmWayDragDown);

    // Ghost must be added BEFORE node markers so nodes sit on top in the SVG DOM
    // and receive mousedown events before the ghost does.
    const geomType = feature.geometry.type;
    if (geomType === 'LineString') {
        const latlngs = feature.geometry.coordinates.map(c => [c[1], c[0]]);
        osmDragGhost = L.polyline(latlngs, {
            renderer: L.svg(), color: '#000', weight: 15, opacity: 0,
            interactive: true, bubblingMouseEvents: false,
        });
    } else if (geomType === 'Polygon') {
        const ring = feature.geometry.coordinates[0].map(c => [c[1], c[0]]);
        osmDragGhost = L.polygon(ring, {
            renderer: L.svg(), color: '#000', weight: 15, opacity: 0, fillOpacity: 0,
            interactive: true, bubblingMouseEvents: false,
        });
    }
    if (osmDragGhost) {
        // Node and midpoint markers use SVG renderer and are added to the map after the ghost,
        // so they sit higher in the SVG DOM and receive their own mousedown events directly.
        // The ghost only needs to handle clicks on the way body (i.e. way drag).
        osmDragGhost.on('mousedown', _onOsmWayDragDown);
        osmDragGhost.addTo(map);
    }

    nodeLayer = L.layerGroup();
    nodeMarkers = currentNodes.map((node, i) => {
        const marker = L.circleMarker([node.lat, node.lon], {
            radius: 5, color: '#fff', weight: 2,
            fillColor: '#f0a500', fillOpacity: 0.9,
            bubblingMouseEvents: false,
            renderer: L.svg(),
        });
        const onNodeDown = e => _onOsmNodeDragDown(e, i);
        marker.on('mousedown', onNodeDown);
        marker.on('add', () => { const el = marker.getElement(); if (el) el.style.cursor = 'grab'; });
        nodeLayer.addLayer(marker);

        // Transparent larger hit target so small nodes are easier to grab
        const hit = L.circleMarker([node.lat, node.lon], {
            radius: 12, fillOpacity: 0, opacity: 0,
            bubblingMouseEvents: false, renderer: L.svg(), interactive: true,
        });
        hit.on('mousedown', onNodeDown);
        hit.on('add', () => { const el = hit.getElement(); if (el) el.style.cursor = 'grab'; });
        nodeLayer.addLayer(hit);
        return marker;
    });

    // Midpoint handles between consecutive nodes for inserting new nodes.
    // Added after node markers so they sit below nodes in the SVG DOM.
    midpointMarkers = [];
    const isClosed = currentNodes.length >= 2 &&
        currentNodes[0].id === currentNodes[currentNodes.length - 1].id;
    const mpEnd = isClosed ? currentNodes.length - 1 : currentNodes.length - 1;
    for (let i = 0; i < mpEnd; i++) {
        const a = currentNodes[i], b = currentNodes[i + 1];
        const mlat = (a.lat + b.lat) / 2, mlon = (a.lon + b.lon) / 2;
        const mp = L.circleMarker([mlat, mlon], {
            radius: 4, color: '#4af', weight: 1.5,
            fillColor: '#4af', fillOpacity: 0.7,
            bubblingMouseEvents: false,
            renderer: L.svg(),
        });
        const onMpDown = e => _onMidpointDragDown(e, i);
        mp.on('mousedown', onMpDown);
        mp.on('add', () => { const el = mp.getElement(); if (el) el.style.cursor = 'crosshair'; });
        nodeLayer.addLayer(mp);

        // Transparent larger hit target for midpoints
        const mpHit = L.circleMarker([mlat, mlon], {
            radius: 10, fillOpacity: 0, opacity: 0,
            bubblingMouseEvents: false, renderer: L.svg(), interactive: true,
        });
        mpHit.on('mousedown', onMpDown);
        mpHit.on('add', () => { const el = mpHit.getElement(); if (el) el.style.cursor = 'crosshair'; });
        nodeLayer.addLayer(mpHit);
        midpointMarkers.push(mp);
    }

    nodeLayer.addTo(map);

    const props = feature.properties;
    const tags = props.tags || {};
    const label = tags.highway || tags.barrier || '';
    el.innerHTML = `
    <div style="font-size:0.8rem;color:#aaa;margin-bottom:6px;">
      Editing <span class="badge bg-secondary">${escHtml(props.category)}</span>
      ${label ? `<span class="badge bg-dark ms-1">${escHtml(label)}</span>` : ''}
      <span class="text-secondary ms-1">#${props.id}</span>
    </div>
    <div style="font-size:0.75rem;color:#6c7a9c;">${currentNodes.length} nodes — drag node or way to move</div>`;
    setStatus(`Editing way ${wayId}`, 'text-info');
}

function clickNode(index) {
    const node = currentNodes[index];
    if (!node) return;

    if (selectedNodeIndex >= 0 && nodeMarkers[selectedNodeIndex]) {
        nodeMarkers[selectedNodeIndex].setStyle({
            radius: 4, color: '#fff', weight: 1.5, fillColor: '#f0a500', fillOpacity: 0.9,
        });
    }

    selectedNodeIndex = index;

    if (nodeMarkers[index]) {
        nodeMarkers[index].setStyle({
            radius: 6, color: '#fff', weight: 2.5, fillColor: '#ff3300', fillOpacity: 1,
        });
        nodeMarkers[index].bringToFront();
    }

    document.querySelectorAll('.node-list-item').forEach((el, i) =>
        el.classList.toggle('sel', i === index)
    );

    map.panTo([node.lat, node.lon]);
    showOsmNodeProps(node, index, currentNodes.length);
}

function showOsmNodeProps(node, index, total) {
    const wayFeature = currentClickedFeature;
    const isPath = wayFeature && ['road', 'footway'].includes(wayFeature.properties.category);
    const isOpenBarrier = wayFeature && wayFeature.properties.category === 'barrier'
        && currentNodes.length >= 2
        && currentNodes[0]?.id !== currentNodes[currentNodes.length - 1]?.id;
    const canSplit = (isPath || isOpenBarrier) && index > 0 && index < total - 1;

    const tagRows = Object.entries(node.tags || {})
        .map(([k, v]) => `<tr><td>${escHtml(k)}</td><td>${escHtml(String(v))}</td></tr>`)
        .join('') || '<tr><td colspan="2" style="color:#6c7a9c;font-style:italic;">No tags</td></tr>';

    let buttons = `
    <div class="d-flex gap-1 mt-2">
      <button class="btn btn-sm btn-outline-secondary" style="font-size:0.72rem;flex:1;"
              onclick="showProps(currentClickedFeature.properties, currentClickedFeature)">
        ← Back
      </button>`;

    if (canSplit) {
        buttons += `
      <button class="btn btn-sm btn-outline-warning" style="font-size:0.72rem;" title="Split Way"
              onclick="splitCurrentWay(currentClickedFeature.properties.id, ${node.id})">
        ✂️
      </button>`;
    }

    buttons += `
      <button class="btn btn-sm btn-outline-danger" style="font-size:0.72rem;" title="Delete Node"
              onclick="deleteCurrentNode(currentClickedFeature.properties.id, ${node.id})">
        &#128465;
      </button>
    </div>`;

    document.getElementById('props-content').innerHTML = `
    <table><tbody>
      <tr><td>Type</td><td><span class="badge bg-secondary">node</span></td></tr>
      <tr><td>OSM ID</td><td>${node.id}</td></tr>
      <tr><td>Index</td><td>${index + 1} / ${total}</td></tr>
      <tr><td>Lat</td><td>${parseFloat(node.lat).toFixed(7)}</td></tr>
      <tr><td>Lon</td><td>${parseFloat(node.lon).toFixed(7)}</td></tr>
      ${tagRows}
    </tbody></table>
    ${buttons}`;
}

async function splitCurrentWay(wayId, nodeId) {
    if (!currentFile) return;

    const res = await splitWayApi(currentFile, wayId, nodeId);
    if (res.ok) {
        const data = await res.json();
        setStatus('Way split successfully', 'text-success');

        // Surgical update
        const originalWayId = String(wayId).split(':')[0];
        const newLayer = updateWayWithSegments(originalWayId, data.segments);

        // Update selection to the first new segment if it exists
        if (newLayer && newLayer._featureRef) {
            currentClickedLayer = newLayer;
            currentClickedFeature = newLayer._featureRef;
            newLayer._osmCat = newLayer._featureRef.properties.category;
            newLayer.setStyle(HIGHLIGHT_STYLES[newLayer._osmCat]);
            showProps(newLayer._featureRef.properties, newLayer._featureRef);
            // Clear nodes but immediately reload them for the new segment
            clearNodes();
            loadNodesForEditing(newLayer._featureRef, newLayer);
        } else {
            currentClickedLayer = null;
            currentClickedFeature = null;
            clearNodes();
            document.getElementById('props-content').innerHTML =
                '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
        }

        // Refresh only metadata (changes list, etc.) without map flashing
        refreshMetadata(currentFile);
    } else {
        setStatus('Split failed', 'text-danger');
    }
}

async function toggleNodes() {
    const feature = currentClickedFeature;
    if (!feature) return;
    if (nodeLayer) { clearNodes(); showProps(feature.properties, feature); return; }

    setStatus('Loading nodes…', 'text-secondary');
    try {
        const data = await fetchWayNodes(currentFile, feature.properties.id);
        currentNodes = data.nodes;
        nodeCount = data.nodes.length;
        nodeMarkers = [];
        nodeLayer = L.layerGroup().addTo(map);
        data.nodes.forEach((node, i) => {
            const marker = L.circleMarker([node.lat, node.lon], {
                radius: 4, color: '#fff', weight: 1.5, fillColor: '#f0a500', fillOpacity: 0.9,
            }).on('click', e => {
                L.DomEvent.stopPropagation(e);
                if (currentMode === 'view') clickNode(i);
            }).on('contextmenu', e => {
                L.DomEvent.stopPropagation(e);
                L.DomEvent.preventDefault(e);
                if (currentMode === 'view') showNodeContextMenu(node, i, e.latlng);
            }).addTo(nodeLayer);
            nodeMarkers.push(marker);
        });

        setStatus('', 'text-secondary');
        showProps(feature.properties, feature);
    } catch (err) {
        setStatus(`Node load failed: ${err.message}`, 'text-danger');
    }
}

function showNodeContextMenu(node, index, latlng) {
    const wayFeature = currentClickedFeature;
    if (!wayFeature) return;
    const wayId = wayFeature.properties.id;
    const total = currentNodes.length;
    const isPath = ['road', 'footway'].includes(wayFeature.properties.category);
    const isOpenBarrier = wayFeature.properties.category === 'barrier'
        && currentNodes.length >= 2
        && currentNodes[0]?.id !== currentNodes[currentNodes.length - 1]?.id;
    const canSplit = (isPath || isOpenBarrier) && index > 0 && index < total - 1;

    const container = document.createElement('div');
    container.className = 'context-menu';

    if (canSplit) {
        const splitBtn = document.createElement('button');
        splitBtn.innerHTML = '✂️ Split Way';
        splitBtn.onclick = () => { map.closePopup(); splitCurrentWay(wayId, node.id); };
        container.appendChild(splitBtn);
    }

    const delBtn = document.createElement('button');
    delBtn.innerHTML = '&#128465; Delete Node';
    delBtn.onclick = () => { map.closePopup(); deleteCurrentNode(wayId, node.id); };
    container.appendChild(delBtn);

    L.popup({ minWidth: 150, className: 'planner-popup', offset: [0, -5], closeButton: false })
        .setLatLng(latlng)
        .setContent(container)
        .openOn(map);
}

function openWayEditModal() {
    const feature = currentClickedFeature;
    if (!feature) return;
    editingWayId = feature.properties.id;
    document.getElementById('way-edit-title').textContent = `Edit Way #${editingWayId}`;
    _renderWayEditProps(feature.properties.tags || {});
    new bootstrap.Modal(document.getElementById('way-edit-modal')).show();
}

function _renderWayEditProps(obj) {
    document.getElementById('way-edit-props').innerHTML =
        Object.entries(obj).map(([k, v]) => `
      <div class="d-flex gap-1 mb-1">
        <input class="form-control form-control-sm bg-dark text-light border-secondary we-key"
               placeholder="key" value="${escHtml(k)}" style="flex:1;font-size:0.75rem;">
        <input class="form-control form-control-sm bg-dark text-light border-secondary we-val"
               placeholder="value" value="${escHtml(String(v))}" style="flex:1;font-size:0.75rem;">
        <button type="button" class="btn btn-sm btn-outline-danger px-1"
                onclick="this.closest('.d-flex').remove()">×</button>
      </div>`).join('');
}

document.getElementById('way-edit-save')?.addEventListener('click', async () => {
    if (!currentFile || !editingWayId) return;
    const keys = document.querySelectorAll('#way-edit-props .we-key');
    const vals = document.querySelectorAll('#way-edit-props .we-val');
    const tags = {};
    keys.forEach((el, i) => {
        const k = el.value.trim();
        if (k) tags[k] = vals[i].value.trim();
    });
    const cat = currentClickedFeature?.properties?.category ?? 'unknown';
    const lbl = currentClickedFeature?.properties?.tags?.highway
        || currentClickedFeature?.properties?.tags?.barrier || '';
    const res = await updateWayTagsApi(currentFile, editingWayId, tags, cat, lbl);
    if (!res.ok) { setStatus('Save failed', 'text-danger'); return; }
    bootstrap.Modal.getInstance(document.getElementById('way-edit-modal'))?.hide();
    const existingIdx = changeLog.findIndex(c => c.type === 'tag' && String(c.id) === String(editingWayId));
    if (existingIdx >= 0) changeLog.splice(existingIdx, 1);
    changeLog.push({ type: 'tag', id: editingWayId, category: cat, label: lbl });
    await _reloadWay(editingWayId);
    setStatus(`Tags saved for way ${editingWayId}`, 'text-success');
    renderChangesPanel();
});

document.getElementById('way-edit-add-prop-btn')?.addEventListener('click', () => {
    const container = document.getElementById('way-edit-props');
    const row = document.createElement('div');
    row.className = 'd-flex gap-1 mb-1';
    row.innerHTML = `
    <input class="form-control form-control-sm bg-dark text-light border-secondary we-key"
           placeholder="key" style="flex:1;font-size:0.75rem;">
    <input class="form-control form-control-sm bg-dark text-light border-secondary we-val"
           placeholder="value" style="flex:1;font-size:0.75rem;">
    <button type="button" class="btn btn-sm btn-outline-danger px-1"
            onclick="this.closest('.d-flex').remove()">×</button>`;
    container.appendChild(row);
});

async function undoTagOverride(wayId) {
    if (!currentFile) return;
    const res = await deleteWayTagsApi(currentFile, wayId);
    if (res.ok) {
        changeLog = changeLog.filter(c => !(c.type === 'tag' && String(c.id) === String(wayId)));
        tagOverrides = tagOverrides.filter(t => String(t.id) !== String(wayId));
        await _reloadWay(wayId);
        renderChangesPanel();
    }
}

async function deleteCurrentWay() {
    const feature = currentClickedFeature;
    if (!feature || !currentFile) return;
    const wayId = feature.properties.id;
    const cat = feature.properties.category;
    const tags = feature.properties.tags || {};
    const label = tags.highway || tags.barrier || '';
    const res = await deleteWayApi(currentFile, wayId, cat, label);
    if (!res.ok) { setStatus('Delete failed', 'text-danger'); return; }
    if (currentClickedLayer && geoLayers[cat]) {
        geoLayers[cat].removeLayer(currentClickedLayer);
    }
    currentClickedLayer = null;
    currentClickedFeature = null;
    clearNodes();
    if (!deletedWays.some(d => d.id === wayId)) {
        deletedWays.push({ id: wayId, category: cat, label });
        changeLog.push({ type: 'way', id: wayId, category: cat, label });
    }
    renderChangesPanel();
    renderHiddenPanel();
    document.getElementById('props-content').innerHTML =
        '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
    setStatus(`Deleted way ${wayId}`, 'text-success');
}

async function deleteCurrentNode(wayId, nodeId) {
    if (!currentFile) return;
    const res = await deleteNodeApi(currentFile, wayId, nodeId);
    if (!res.ok) { setStatus('Delete failed', 'text-danger'); return; }
    changeLog.push({ type: 'node', way_id: wayId, node_id: nodeId });
    await _reloadWay(wayId);
    if (currentClickedFeature) await toggleNodes();
    setStatus('Node deleted', 'text-success');
}

async function hideCurrentWay() {
    const feature = currentClickedFeature;
    if (!feature || !currentFile) return;
    const wayId = feature.properties.id;
    const cat = feature.properties.category;
    const tags = feature.properties.tags || {};
    const label = tags.highway || tags.barrier || '';
    const res = await hideWayApi(currentFile, wayId, cat, label);
    if (!res.ok) { setStatus('Hide failed', 'text-danger'); return; }
    if (currentClickedLayer && geoLayers[cat]) {
        geoLayers[cat].removeLayer(currentClickedLayer);
    }
    currentClickedLayer = null;
    currentClickedFeature = null;
    clearNodes();
    if (!hiddenWays.some(d => d.id === wayId)) {
        hiddenWays.push({ id: wayId, category: cat, label });
        hiddenWayIds.add(wayId);
    }
    renderChangesPanel();
    renderHiddenPanel();
    document.getElementById('props-content').innerHTML =
        '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
    setStatus(`Hidden way ${wayId}`, 'text-success');
}

async function showWay(wayId) {
    if (!currentFile) return;
    const res = await showWayApi(currentFile, wayId);
    if (res.ok) {
        hiddenWays = hiddenWays.filter(d => d.id !== wayId);
        hiddenWayIds.delete(wayId);
        await _reloadWay(wayId);
    }
}

function focusFeatureById(wayId) {
    // Search visible layers first
    for (const cat of ['road', 'footway', 'barrier', 'crossroad']) {
        const catLayer = geoLayers[cat];
        if (!catLayer) continue;
        let found = null;
        catLayer.eachLayer(l => { if (String(l._featureId) === String(wayId)) found = l; });
        if (found) {
            if (currentClickedLayer && currentClickedLayer !== found) {
                const oldCat = currentClickedLayer._osmCat;
                currentClickedLayer.setStyle(oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options._ann_id)));
            }
            found._osmCat = cat;
            currentClickedLayer = found;
            currentClickedFeature = found._featureRef;
            found.setStyle(HIGHLIGHT_STYLES[cat]);
            showProps(found._featureRef.properties, found._featureRef);
            try { map.fitBounds(found.getBounds(), { maxZoom: 18, padding: [40, 40] }); } catch (_) { }
            return;
        }
    }
    // Search hidden layers still tracked in subtypeLayers
    for (const cat of ['road', 'footway', 'barrier']) {
        for (const layers of Object.values(subtypeLayers[cat])) {
            const found = layers.find(l => String(l._featureId) === String(wayId));
            if (found) {
                if (found._featureRef) {
                    currentClickedFeature = found._featureRef;
                    showProps(found._featureRef.properties, found._featureRef);
                }
                try { map.fitBounds(found.getBounds(), { maxZoom: 18, padding: [40, 40] }); } catch (_) { }
                return;
            }
        }
    }
    // Fall back to API for deleted ways
    if (!currentFile) return;
    fetchWayApi(currentFile, wayId).then(res => {
        if (!res.ok) return;
        res.json().then(feature => {
            showProps(feature.properties, feature);
            try { map.fitBounds(L.geoJSON(feature).getBounds(), { maxZoom: 18, padding: [40, 40] }); } catch (_) { }
        });
    });
}

function renderChangesPanel() {
    const panel = document.getElementById('changes-panel');
    const list = document.getElementById('changes-list');
    const count = document.getElementById('changes-count');
    if (!panel || !list || !count) return;
    if (!changeLog.length || currentAppMode === 'planner') { panel.style.display = 'none'; return; }
    panel.style.display = '';
    count.textContent = `(${changeLog.length})`;
    list.innerHTML = [...changeLog].reverse().map(d => {
        const wayIdJson = JSON.stringify(d.id || d.way_id);
        if (d.type === 'way') return `
      <div class="change-item" style="cursor:pointer;" onclick='focusFeatureById(${wayIdJson})'>
        <div>
          <span>del ${escHtml(d.category)}${d.label ? ' · ' + escHtml(d.label) : ''}</span>
          <br><span class="change-id">#${d.id}</span>
        </div>
        <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
                title="Undo deletion" onclick='event.stopPropagation(); undoWayDeletion(${wayIdJson})'>&#8617;</button>
      </div>`;
        if (d.type === 'node') return `
      <div class="change-item" style="cursor:pointer;" onclick='focusFeatureById(${wayIdJson})'>
        <div>
          <span>del node in way</span>
          <br><span class="change-id">#${d.node_id} &rarr; #${d.way_id}</span>
        </div>
        <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
                title="Undo deletion" onclick='event.stopPropagation(); undoNodeDeletion(${wayIdJson}, ${d.node_id})'>&#8617;</button>
      </div>`;
        if (d.type === 'tag') return `
      <div class="change-item" style="cursor:pointer;" onclick='focusFeatureById(${wayIdJson})'>
        <div>
          <span>edit ${escHtml(d.category)}${d.label ? ' · ' + escHtml(d.label) : ''}</span>
          <br><span class="change-id">#${d.id}</span>
        </div>
        <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
                title="Undo tag edit" onclick='event.stopPropagation(); undoTagOverride(${wayIdJson})'>&#8617;</button>
      </div>`;
        if (d.type === 'move') return `
      <div class="change-item" style="cursor:pointer;" onclick='focusFeatureById(${wayIdJson})'>
        <div>
          <span>move ${escHtml(d.category)}${d.label ? ' · ' + escHtml(d.label) : ''}</span>
          <br><span class="change-id">#${d.id}</span>
        </div>
        <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
                title="Undo move" onclick='event.stopPropagation(); undoWayNodeMoves(${wayIdJson})'>&#8617;</button>
      </div>`;
        if (d.type === 'split') return `
      <div class="change-item" style="cursor:pointer;" onclick='focusFeatureById(${wayIdJson})'>
        <div>
          <span>split way</span>
          <br><span class="change-id">#${d.way_id} @ node #${d.node_id}</span>
        </div>
        <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
                title="Undo split" onclick='event.stopPropagation(); undoWaySplit(${wayIdJson}, ${d.node_id})'>&#8617;</button>
      </div>`;
        return '';
    }).join('');
}

async function undoWaySplit(wayId, nodeId) {
    if (!currentFile) return;
    const res = await undoWaySplitApi(currentFile, wayId, nodeId);
    if (res.ok) {
        const data = await res.json();
        setStatus('Split reverted', 'text-success');

        // Surgical update
        const newLayer = updateWayWithSegments(wayId, data.segments);

        // If the split was focused, update focus to the merged way
        if (currentClickedFeature && (String(currentClickedFeature.properties.id) === String(wayId) || String(currentClickedFeature.properties.id).startsWith(String(wayId) + ':'))) {
            if (newLayer && newLayer._featureRef) {
                currentClickedLayer = newLayer;
                currentClickedFeature = newLayer._featureRef;
                newLayer._osmCat = newLayer._featureRef.properties.category;
                newLayer.setStyle(HIGHLIGHT_STYLES[newLayer._osmCat]);
                showProps(newLayer._featureRef.properties, newLayer._featureRef);
            } else {
                currentClickedLayer = null;
                currentClickedFeature = null;
                document.getElementById('props-content').innerHTML =
                    '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
            }
            clearNodes();
        }

        // Refresh only metadata (changes list, etc.) without map flashing
        refreshMetadata(currentFile);
    } else {
        setStatus('Undo failed', 'text-danger');
    }
}

function renderHiddenPanel() {
    const panel = document.getElementById('hidden-panel');
    const list = document.getElementById('hidden-list');
    const count = document.getElementById('hidden-count');
    if (!panel || !list || !count) return;
    if (!hiddenWays.length || currentAppMode === 'planner') { panel.style.display = 'none'; return; }
    panel.style.display = '';
    count.textContent = `(${hiddenWays.length})`;
    list.innerHTML = [...hiddenWays].reverse().map(d => {
        const wayIdJson = JSON.stringify(d.id);
        return `
    <div class="change-item" style="cursor:pointer;" onclick='focusFeatureById(${wayIdJson})'>
      <div>
        <span>${escHtml(d.category)}${d.label ? ' · ' + escHtml(d.label) : ''}</span>
        <br><span class="change-id">#${d.id}</span>
      </div>
      <button class="btn btn-sm btn-outline-info py-0 px-1" style="font-size:0.7rem;"
              title="Show object" onclick='event.stopPropagation(); showWay(${wayIdJson})'>&#128065;</button>
    </div>`;
    }).join('');
}

async function undoWayDeletion(wayId) {
    if (!currentFile) return;
    const res = await restoreWayApi(currentFile, wayId);
    if (res.ok) {
        changeLog = changeLog.filter(c => !(c.type === 'way' && c.id === wayId));
        await _reloadWay(wayId);
    }
}

async function undoNodeDeletion(wayId, nodeId) {
    if (!currentFile) return;
    const res = await restoreNodeApi(currentFile, wayId, nodeId);
    if (res.ok) {
        changeLog = changeLog.filter(c => !(c.type === 'node' && String(c.way_id) === String(wayId) && c.node_id === nodeId));
        await _reloadWay(wayId);
    }
}

async function undoNodeAddition(wayId, nodeId) {
    if (!currentFile) return;
    const res = await deleteNodeApi(currentFile, wayId, nodeId);
    if (res.ok) {
        changeLog = changeLog.filter(c => !(c.type === 'add_node' && c.way_id === wayId && c.node_id === nodeId));
        await _reloadWay(wayId);
        await refreshMetadata(currentFile);
    }
}

async function undoWayNodeMoves(wayId) {
    if (!currentFile) return;
    // Pass original way ID to backend, but use virtual ID for local state
    const originalWayId = String(wayId).split(':')[0];
    const res = await undoWayNodeMovesApi(currentFile, originalWayId);
    if (res.ok) {
        changeLog = changeLog.filter(c => !(c.type === 'move' && String(c.id) === String(wayId)));
        await _reloadWay(wayId);
    }
}

function togglePanel(name) {
    const body = document.getElementById(`${name}-body`);
    const toggle = document.getElementById(`${name}-toggle`);
    if (!body) return;
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : '';
    if (toggle) toggle.textContent = open ? '▼' : '▲';
}

function showAnnProps(ann) {
    if (currentClickedFeature) {
        clearNodes();
        currentClickedFeature = null;
    }
    const props = ann.properties || {};
    const propRows = Object.entries(props)
        .map(([k, v]) => `<tr><td>${escHtml(k)}</td><td>${escHtml(String(v))}</td></tr>`)
        .join('') || '<tr><td colspan="2" style="color:#6c7a9c;font-style:italic;">No properties</td></tr>';
    document.getElementById('props-content').innerHTML =
        `<table><tbody>
       <tr><td>Type</td><td><span class="badge bg-secondary">${escHtml(ann.type || 'obstacle')}</span></td></tr>
       ${propRows}
     </tbody></table>
     <button class="btn btn-sm btn-outline-info mt-2" style="font-size:0.72rem;width:100%;"
             onclick="openAnnEditModal('${ann.id}')">&#9998; Edit Properties</button>`;
}

function renderAnnotationList() {
    const panel = document.getElementById('ann-panel');
    const el = document.getElementById('ann-list');
    const count = document.getElementById('ann-count');
    if (!panel || !el || !count) return;
    const addedNodeEntries = changeLog.filter(c => c.type === 'add_node');
    if (!annotations.length && !addedNodeEntries.length || currentAppMode === 'planner') { panel.style.display = 'none'; return; }
    panel.style.display = '';
    count.textContent = `(${annotations.length + addedNodeEntries.length})`;
    const annHtml = annotations.map(a => `
    <div class="ann-item">
      <div>
        <span>${escHtml(a.type)}</span>
        <br><span class="ann-id">${a.id.slice(0, 8)}…</span>
      </div>
      <div class="d-flex gap-1">
        <button class="btn btn-sm btn-outline-info py-0 px-1" style="font-size:0.7rem;"
                onclick="openAnnEditModal('${a.id}')">&#9998;</button>
        <button class="btn btn-sm btn-outline-danger py-0 px-1" style="font-size:0.7rem;"
                onclick="removeAnnotationById('${a.id}')">&#10005;</button>
      </div>
    </div>
  `).join('');
    const nodeHtml = addedNodeEntries.map(d => {
        const wayIdJson = JSON.stringify(d.way_id);
        return `
    <div class="ann-item" style="cursor:pointer;" onclick='focusFeatureById(${wayIdJson})'>
      <div>
        <span>add node to way</span>
        <br><span class="ann-id">#${d.node_id} &rarr; #${d.way_id}</span>
      </div>
      <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
              title="Undo node addition" onclick='event.stopPropagation(); undoNodeAddition(${wayIdJson}, ${d.node_id})'>&#8617;</button>
    </div>`;
    }).join('');
    el.innerHTML = annHtml + nodeHtml;
}

async function deleteSelectedAnnotation() {
    if (!editSelectedLayer || !currentFile) return;
    const annId = editSelectedLayer.options._ann_id;
    if (!annId) return;
    const res = await deleteAnnotationApi(currentFile, annId);
    if (res.ok) {
        if (editSelectedLayer.editing) editSelectedLayer.editing.disable();
        drawnItems.removeLayer(editSelectedLayer);
        annotations = annotations.filter(a => a.id !== annId);
        if (currentClickedLayer === editSelectedLayer) {
            currentClickedLayer = null;
            const propsEl = document.getElementById('props-content');
            if (propsEl) propsEl.innerHTML = '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
        }
        editSelectedLayer = null;
        renderAnnotationList();
        setStatus('Annotation deleted', 'text-success');
    } else {
        setStatus('Failed to delete annotation', 'text-danger');
    }
}

async function removeAnnotationById(id) {
    if (!currentFile) return;
    const res = await deleteAnnotationApi(currentFile, id);
    if (res.ok) {
        if (currentClickedLayer && currentClickedLayer.options && currentClickedLayer.options._ann_id === id) {
            currentClickedLayer = null;
            const propsEl = document.getElementById('props-content');
            if (propsEl) propsEl.innerHTML = '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
        }
        annotations = annotations.filter(a => a.id !== id);
        renderAnnotationLayer();
        renderAnnotationList();
        setStatus('Annotation deleted', 'text-success');
    } else {
        setStatus('Failed to delete annotation', 'text-danger');
    }
}

function updateAnnTypeFields() {
    const t = document.getElementById('ann-type-sel').value;
    document.getElementById('ann-obstacle-fields').style.display = t === 'obstacle' ? '' : 'none';
    document.getElementById('ann-path-fields').style.display = t === 'path' ? '' : 'none';
}

function _renderExtraProps(obj) {
    document.getElementById('ann-extra-props').innerHTML =
        Object.entries(obj).map(([k, v]) => `
      <div class="d-flex gap-1 mb-1">
        <input class="form-control form-control-sm bg-dark text-light border-secondary ep-key"
               placeholder="key" value="${escHtml(k)}" style="flex:1;font-size:0.75rem;">
        <input class="form-control form-control-sm bg-dark text-light border-secondary ep-val"
               placeholder="value" value="${escHtml(String(v))}" style="flex:1;font-size:0.75rem;">
        <button type="button" class="btn btn-sm btn-outline-danger px-1"
                onclick="this.closest('.d-flex').remove()">×</button>
      </div>`).join('');
}

function _collectAnnForm() {
    const type = document.getElementById('ann-type-sel').value;
    const props = {};
    if (type === 'path') {
        props.highway = document.getElementById('ann-highway-sel').value;
        const w = parseFloat(document.getElementById('ann-width-inp').value);
        if (!isNaN(w) && w > 0) props.width = w;
    } else {
        props.barrier = document.getElementById('ann-barrier-sel').value;
    }
    document.querySelectorAll('#ann-extra-props .d-flex').forEach(row => {
        const k = row.querySelector('.ep-key').value.trim();
        const v = row.querySelector('.ep-val').value.trim();
        if (k) props[k] = v;
    });
    return { type, props };
}

function openAnnDetailModal(geometry, mode) {
    pendingAnnGeom = geometry;
    editingAnnId = null;
    document.getElementById('ann-detail-title').textContent = 'New Annotation';
    document.getElementById('ann-type-sel').value = mode === 'path' ? 'path' : 'obstacle';
    document.getElementById('ann-type-sel').disabled = false;
    document.getElementById('ann-barrier-sel').value = 'wall';
    document.getElementById('ann-highway-sel').value = 'footway';
    document.getElementById('ann-width-inp').value = '1.5';
    _renderExtraProps({});
    updateAnnTypeFields();
    new bootstrap.Modal(document.getElementById('ann-detail-modal')).show();
}

function openAnnEditModal(annId) {
    const ann = annotations.find(a => a.id === annId);
    if (!ann) return;
    editingAnnId = annId;
    pendingAnnGeom = null;
    document.getElementById('ann-detail-title').textContent = 'Edit Annotation';
    const type = ann.type || 'obstacle';
    document.getElementById('ann-type-sel').value = type;
    document.getElementById('ann-type-sel').disabled = false;
    const props = ann.properties || {};
    if (type === 'path') {
        document.getElementById('ann-highway-sel').value = props.highway || 'footway';
        document.getElementById('ann-width-inp').value = props.width || '1.5';
    } else {
        document.getElementById('ann-barrier-sel').value = props.barrier || 'wall';
    }
    const exclude = type === 'path' ? ['highway', 'width'] : ['barrier'];
    _renderExtraProps(Object.fromEntries(Object.entries(props).filter(([k]) => !exclude.includes(k))));
    updateAnnTypeFields();
    new bootstrap.Modal(document.getElementById('ann-detail-modal')).show();
}

async function handleMapdataUpload(file) {
    setStatus(`Uploading ${file.name}...`, 'text-info');
    const formData = new FormData();
    formData.append('file', file);
    try {
        const data = await uploadMapdataApi(formData);
        const sel = document.getElementById('file-select');
        if (sel && ![...sel.options].some(o => o.value === data.filename)) {
            sel.appendChild(new Option(data.filename, data.filename));
        }
        if (sel) sel.value = data.filename;
        await loadMapData(data.filename);
    } catch (err) {
        setStatus(`Upload failed: ${err.message}`, 'text-danger');
    }
}

let pendingGpxFile = null;

function handleGpxMapCreation(file) {
    pendingGpxFile = file;
    const nameInput = document.getElementById('gpx-name-input');
    if (nameInput) {
        nameInput.value = file.name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_\-]/g, "_");
    }
    new bootstrap.Modal(document.getElementById('gpx-upload-modal')).show();
}

document.getElementById('gpx-upload-submit')?.addEventListener('click', async () => {
    const name = document.getElementById('gpx-name-input').value.trim();
    if (!name || !pendingGpxFile) {
        document.getElementById('gpx-name-input').focus();
        return;
    }
    bootstrap.Modal.getInstance(document.getElementById('gpx-upload-modal')).hide();
    setStatus('Processing GPX and fetching OSM data... (may take 1–2 min)', 'text-warning');

    const formData = new FormData();
    formData.append('file', pendingGpxFile);
    formData.append('name', name);
    formData.append('options', JSON.stringify({
        grid_margin: parseFloat(document.getElementById('gpx-grid-margin')?.value) || 150,
        obstacle_radius: parseFloat(document.getElementById('gpx-obstacle-radius')?.value) || 2.0,
        buffer_widths: {
            road: parseFloat(document.getElementById('gpx-buf-road')?.value) || 7.0,
            footway: parseFloat(document.getElementById('gpx-buf-footway')?.value) || 3.0,
            barrier: parseFloat(document.getElementById('gpx-buf-barrier')?.value) || 2.0,
        },
    }));

    try {
        const data = await uploadGpxApi(formData);
        setStatus(
            `GPX Processed: ${data.roads} roads, ${data.footways} footways, ${data.barriers} barriers`,
            'text-success'
        );
        const sel = document.getElementById('file-select');
        if (sel && ![...sel.options].some(o => o.value === data.filename)) {
            sel.appendChild(new Option(data.filename, data.filename));
        }
        if (sel) sel.value = data.filename;
        await loadMapData(data.filename);
    } catch (err) {
        setStatus(`GPX processing failed: ${err.message}`, 'text-danger');
    } finally {
        pendingGpxFile = null;
    }
});

document.getElementById('gpx-name-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('gpx-upload-submit').click();
});
