// ── UI Handlers ───────────────────────────────────────────────────────────────

function setStatus(msg, cls = 'text-secondary') {
  const el = document.getElementById('status');
  if (!el) return;
  el.textContent = msg;
  el.className = cls + ' ms-auto';
}

function renderSubtypeFilters(cat) {
  const panel = document.getElementById(`subfilter-${cat}`);
  if (!panel) return;
  const subtypes = Object.keys(subtypeLayers[cat]).sort();
  if (!subtypes.length) { panel.innerHTML = ''; return; }
  panel.innerHTML = subtypes.map(st => {
    const count   = (subtypeLayers[cat][st] || []).length;
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

function showProps(props, feature = null) {
  if (feature !== currentClickedFeature) {
    clearNodes();
    currentClickedFeature = feature;
  }
  const tags = props.tags || {};
  const meta = [
    ['Category', `<span class="badge bg-secondary">${props.category}</span>`],
    ['OSM ID',   props.id ?? '—'],
    ['Role',     props.in_out || '—'],
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
                               onclick="deleteCurrentWay()">&#128465; Delete Way</button>`;
    }
  }
}

function clearNodes() {
  if (nodeLayer) { map.removeLayer(nodeLayer); nodeLayer = null; }
  nodeCount         = 0;
  currentNodes      = [];
  nodeMarkers       = [];
  selectedNodeIndex = -1;
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
  const tagRows = Object.entries(node.tags || {})
    .map(([k, v]) => `<tr><td>${escHtml(k)}</td><td>${escHtml(String(v))}</td></tr>`)
    .join('') || '<tr><td colspan="2" style="color:#6c7a9c;font-style:italic;">No tags</td></tr>';
  document.getElementById('props-content').innerHTML = `
    <table><tbody>
      <tr><td>Type</td><td><span class="badge bg-secondary">node</span></td></tr>
      <tr><td>OSM ID</td><td>${node.id}</td></tr>
      <tr><td>Index</td><td>${index + 1} / ${total}</td></tr>
      <tr><td>Lat</td><td>${parseFloat(node.lat).toFixed(7)}</td></tr>
      <tr><td>Lon</td><td>${parseFloat(node.lon).toFixed(7)}</td></tr>
      ${tagRows}
    </tbody></table>
    <div class="d-flex gap-1 mt-2">
      <button class="btn btn-sm btn-outline-secondary" style="font-size:0.72rem;flex:1;"
              onclick="showProps(currentClickedFeature.properties, currentClickedFeature)">
        ← Back to Way
      </button>
      <button class="btn btn-sm btn-outline-danger" style="font-size:0.72rem;"
              onclick="deleteCurrentNode(currentClickedFeature.properties.id, ${node.id})">
        &#128465;
      </button>
    </div>`;
}

async function toggleNodes() {
  const feature = currentClickedFeature;
  if (!feature) return;
  if (nodeLayer) { clearNodes(); showProps(feature.properties, feature); return; }

  setStatus('Loading nodes…', 'text-secondary');
  try {
    const data = await fetchWayNodes(currentFile, feature.properties.id);
    currentNodes = data.nodes;
    nodeCount    = data.nodes.length;
    nodeMarkers  = [];
    nodeLayer    = L.layerGroup().addTo(map);
    data.nodes.forEach((node, i) => {
      const marker = L.circleMarker([node.lat, node.lon], {
        radius: 4, color: '#fff', weight: 1.5, fillColor: '#f0a500', fillOpacity: 0.9,
      }).on('click', e => {
        L.DomEvent.stopPropagation(e);
        if (currentMode === 'view') clickNode(i);
      }).addTo(nodeLayer);
      nodeMarkers.push(marker);
    });

    setStatus('', 'text-secondary');
    showProps(feature.properties, feature);
  } catch (err) {
    setStatus(`Node load failed: ${err.message}`, 'text-danger');
  }
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

async function deleteCurrentWay() {
  const feature = currentClickedFeature;
  if (!feature || !currentFile) return;
  const wayId = feature.properties.id;
  const cat   = feature.properties.category;
  const tags  = feature.properties.tags || {};
  const label = tags.highway || tags.barrier || '';
  const res = await deleteWayApi(currentFile, wayId, cat, label);
  if (!res.ok) { setStatus('Delete failed', 'text-danger'); return; }
  if (currentClickedLayer && geoLayers[cat]) {
    geoLayers[cat].removeLayer(currentClickedLayer);
  }
  currentClickedLayer   = null;
  currentClickedFeature = null;
  clearNodes();
  if (!deletedWays.some(d => d.id === wayId))
    deletedWays.push({ id: wayId, category: cat, label });
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
  await _reloadWay(wayId);
  if (currentClickedFeature) await toggleNodes();
  setStatus('Node deleted', 'text-success');
}

async function hideCurrentWay() {
  const feature = currentClickedFeature;
  if (!feature || !currentFile) return;
  const wayId = feature.properties.id;
  const cat   = feature.properties.category;
  const tags  = feature.properties.tags || {};
  const label = tags.highway || tags.barrier || '';
  const res = await hideWayApi(currentFile, wayId, cat, label);
  if (!res.ok) { setStatus('Hide failed', 'text-danger'); return; }
  if (currentClickedLayer && geoLayers[cat]) {
    geoLayers[cat].removeLayer(currentClickedLayer);
  }
  currentClickedLayer   = null;
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
    catLayer.eachLayer(l => { if (l._featureId === wayId) found = l; });
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
      try { map.fitBounds(found.getBounds(), { maxZoom: 18, padding: [40, 40] }); } catch (_) {}
      return;
    }
  }
  // Search hidden layers still tracked in subtypeLayers
  for (const cat of ['road', 'footway', 'barrier']) {
    for (const layers of Object.values(subtypeLayers[cat])) {
      const found = layers.find(l => l._featureId === wayId);
      if (found) {
        if (found._featureRef) {
          currentClickedFeature = found._featureRef;
          showProps(found._featureRef.properties, found._featureRef);
        }
        try { map.fitBounds(found.getBounds(), { maxZoom: 18, padding: [40, 40] }); } catch (_) {}
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
      try { map.fitBounds(L.geoJSON(feature).getBounds(), { maxZoom: 18, padding: [40, 40] }); } catch (_) {}
    });
  });
}

function renderChangesPanel() {
  const total = deletedWays.length + deletedNodes.length + tagOverrides.length;
  const panel = document.getElementById('changes-panel');
  const list  = document.getElementById('changes-list');
  const count = document.getElementById('changes-count');
  if (!panel || !list || !count) return;
  if (!total) { panel.style.display = 'none'; return; }
  panel.style.display = '';
  count.textContent = `(${total})`;
  const wayItems = [...deletedWays].reverse().map(d => `
    <div class="change-item" style="cursor:pointer;" onclick="focusFeatureById(${d.id})">
      <div>
        <span>del ${escHtml(d.category)}${d.label ? ' · ' + escHtml(d.label) : ''}</span>
        <br><span class="change-id">#${d.id}</span>
      </div>
      <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
              title="Undo deletion" onclick="event.stopPropagation(); undoWayDeletion(${d.id})">&#8617;</button>
    </div>`).join('');
  const nodeItems = [...deletedNodes].reverse().map(d => `
    <div class="change-item" style="cursor:pointer;" onclick="focusFeatureById(${d.way_id})">
      <div>
        <span>del node in way</span>
        <br><span class="change-id">#${d.node_id} &rarr; #${d.way_id}</span>
      </div>
      <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
              title="Undo deletion" onclick="event.stopPropagation(); undoNodeDeletion(${d.way_id}, ${d.node_id})">&#8617;</button>
    </div>`).join('');
  const tagItems = [...tagOverrides].reverse().map(d => `
    <div class="change-item" style="cursor:pointer;" onclick="focusFeatureById(${d.id})">
      <div>
        <span>edit ${escHtml(d.category)}${d.label ? ' · ' + escHtml(d.label) : ''}</span>
        <br><span class="change-id">#${d.id}</span>
      </div>
      <button class="btn btn-sm btn-outline-warning py-0 px-1" style="font-size:0.7rem;"
              title="Undo tag edit" onclick="event.stopPropagation(); undoTagOverride(${d.id})">&#8617;</button>
    </div>`).join('');
  list.innerHTML = wayItems + nodeItems + tagItems;
}

function renderHiddenPanel() {
  const panel = document.getElementById('hidden-panel');
  const list  = document.getElementById('hidden-list');
  const count = document.getElementById('hidden-count');
  if (!panel || !list || !count) return;
  if (!hiddenWays.length) { panel.style.display = 'none'; return; }
  panel.style.display = '';
  count.textContent = `(${hiddenWays.length})`;
  list.innerHTML = hiddenWays.map(d => `
    <div class="change-item" style="cursor:pointer;" onclick="focusFeatureById(${d.id})">
      <div>
        <span>${escHtml(d.category)}${d.label ? ' · ' + escHtml(d.label) : ''}</span>
        <br><span class="change-id">#${d.id}</span>
      </div>
      <button class="btn btn-sm btn-outline-info py-0 px-1" style="font-size:0.7rem;"
              title="Show object" onclick="event.stopPropagation(); showWay(${d.id})">&#128065;</button>
    </div>`).join('');
}

async function undoWayDeletion(wayId) {
  if (!currentFile) return;
  const res = await restoreWayApi(currentFile, wayId);
  if (res.ok) await _reloadWay(wayId);
}

async function undoNodeDeletion(wayId, nodeId) {
  if (!currentFile) return;
  const res = await restoreNodeApi(currentFile, wayId, nodeId);
  if (res.ok) await _reloadWay(wayId);
}

async function undoTagOverride(wayId) {
  if (!currentFile) return;
  const res = await deleteWayTagsApi(currentFile, wayId);
  if (res.ok) await _reloadWay(wayId);
}

function togglePanel(name) {
  const body   = document.getElementById(`${name}-body`);
  const toggle = document.getElementById(`${name}-toggle`);
  if (!body) return;
  const open = body.style.display !== 'none';
  body.style.display = open ? 'none' : '';
  if (toggle) toggle.textContent = open ? '▼' : '▲';
}

function showAnnProps(ann) {
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
  const el    = document.getElementById('ann-list');
  const count = document.getElementById('ann-count');
  if (!panel || !el || !count) return;
  if (!annotations.length) { panel.style.display = 'none'; return; }
  panel.style.display = '';
  count.textContent = `(${annotations.length})`;
  el.innerHTML = annotations.map(a => `
    <div class="ann-item">
      <div>
        <span>${escHtml(a.type)}</span>
        <br><span class="ann-id">${a.id.slice(0,8)}…</span>
      </div>
      <div class="d-flex gap-1">
        <button class="btn btn-sm btn-outline-info py-0 px-1" style="font-size:0.7rem;"
                onclick="openAnnEditModal('${a.id}')">&#9998;</button>
        <button class="btn btn-sm btn-outline-danger py-0 px-1" style="font-size:0.7rem;"
                onclick="removeAnnotationById('${a.id}')">&#10005;</button>
      </div>
    </div>
  `).join('');
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
    editSelectedLayer = null;
    renderAnnotationList();
    setStatus('Annotation deleted', 'text-success');
  }
}

async function removeAnnotationById(id) {
  if (!currentFile) return;
  const res = await deleteAnnotationApi(currentFile, id);
  if (res.ok) {
    annotations = annotations.filter(a => a.id !== id);
    renderAnnotationLayer();
    renderAnnotationList();
  }
}

function updateAnnTypeFields() {
  const t = document.getElementById('ann-type-sel').value;
  document.getElementById('ann-obstacle-fields').style.display = t === 'obstacle' ? '' : 'none';
  document.getElementById('ann-path-fields').style.display    = t === 'path'     ? '' : 'none';
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
  editingAnnId   = null;
  document.getElementById('ann-detail-title').textContent = 'New Annotation';
  document.getElementById('ann-type-sel').value    = mode === 'path' ? 'path' : 'obstacle';
  document.getElementById('ann-type-sel').disabled = false;
  document.getElementById('ann-barrier-sel').value = 'wall';
  document.getElementById('ann-highway-sel').value = 'footway';
  document.getElementById('ann-width-inp').value   = '1.5';
  _renderExtraProps({});
  updateAnnTypeFields();
  new bootstrap.Modal(document.getElementById('ann-detail-modal')).show();
}

function openAnnEditModal(annId) {
  const ann = annotations.find(a => a.id === annId);
  if (!ann) return;
  editingAnnId   = annId;
  pendingAnnGeom = null;
  document.getElementById('ann-detail-title').textContent = 'Edit Annotation';
  const type = ann.type || 'obstacle';
  document.getElementById('ann-type-sel').value    = type;
  document.getElementById('ann-type-sel').disabled = false;
  const props = ann.properties || {};
  if (type === 'path') {
    document.getElementById('ann-highway-sel').value = props.highway || 'footway';
    document.getElementById('ann-width-inp').value   = props.width   || '1.5';
  } else {
    document.getElementById('ann-barrier-sel').value = props.barrier || 'wall';
  }
  const exclude = type === 'path' ? ['highway', 'width'] : ['barrier'];
  _renderExtraProps(Object.fromEntries(Object.entries(props).filter(([k]) => !exclude.includes(k))));
  updateAnnTypeFields();
  new bootstrap.Modal(document.getElementById('ann-detail-modal')).show();
}
