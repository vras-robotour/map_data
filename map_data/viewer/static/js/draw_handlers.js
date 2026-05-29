// ── Drawing and Editing Handlers ───────────────────────────────────────────

const editDrag = { active: false, layer: null, startLatLng: null, origLatLngs: null, origVertices: null };

// ── Annotation vertex editing (OSM-style markers) ─────────────────────────

const annDrag = {
    active: false,
    type: null,        // 'vertex' | 'midpoint'
    layer: null,
    vertexIndex: -1,
    origVertices: null,
    startLatLng: null,
    dragMarker: null,
};

function _getAnnVertices(layer) {
    const lls = layer.getLatLngs();
    return Array.isArray(lls[0]) ? lls[0] : lls;
}

function _setAnnVertex(layer, i, latlng) {
    const lls = layer.getLatLngs();
    if (Array.isArray(lls[0])) { lls[0][i] = latlng; } else { lls[i] = latlng; }
    layer.setLatLngs(lls);
}

function _loadAnnotationVertices(layer) {
    _clearAnnotationVertices();
    const verts = _getAnnVertices(layer);
    const isPolygon = layer instanceof L.Polygon;
    annVertexLayer = L.layerGroup();

    annVertexMarkers_ann = verts.map((v, i) => {
        const m = L.circleMarker([v.lat, v.lng], {
            radius: 5, color: '#fff', weight: 2,
            fillColor: '#f0a500', fillOpacity: 0.9,
            bubblingMouseEvents: false, renderer: L.svg(),
        });
        const onVDown = e => _onAnnVertexDragDown(e, i);
        m.on('mousedown', onVDown);
        m.on('add', () => { const el = m.getElement(); if (el) el.style.cursor = 'grab'; });
        annVertexLayer.addLayer(m);

        const vHit = L.circleMarker([v.lat, v.lng], {
            radius: 12, fillOpacity: 0, opacity: 0,
            bubblingMouseEvents: false, renderer: L.svg(), interactive: true,
        });
        vHit.on('mousedown', onVDown);
        vHit.on('add', () => { const el = vHit.getElement(); if (el) el.style.cursor = 'grab'; });
        annVertexLayer.addLayer(vHit);
        return m;
    });

    annMidpointMarkers_ann = [];
    const n = verts.length;
    const mpCount = isPolygon ? n : n - 1;
    for (let i = 0; i < mpCount; i++) {
        const a = verts[i], b = verts[(i + 1) % n];
        const midLat = (a.lat + b.lat) / 2, midLng = (a.lng + b.lng) / 2;
        const mp = L.circleMarker([midLat, midLng], {
            radius: 4, color: '#4af', weight: 1.5,
            fillColor: '#4af', fillOpacity: 0.7,
            bubblingMouseEvents: false, renderer: L.svg(),
        });
        const onMpDown = e => _onAnnMidpointDragDown(e, i);
        mp.on('mousedown', onMpDown);
        mp.on('add', () => { const el = mp.getElement(); if (el) el.style.cursor = 'crosshair'; });
        annVertexLayer.addLayer(mp);

        const mpHit = L.circleMarker([midLat, midLng], {
            radius: 10, fillOpacity: 0, opacity: 0,
            bubblingMouseEvents: false, renderer: L.svg(), interactive: true,
        });
        mpHit.on('mousedown', onMpDown);
        mpHit.on('add', () => { const el = mpHit.getElement(); if (el) el.style.cursor = 'crosshair'; });
        annVertexLayer.addLayer(mpHit);
        mp._hitMarker = mpHit;
        annMidpointMarkers_ann.push(mp);
    }

    annVertexLayer.addTo(map);
}

function _clearAnnotationVertices() {
    if (annVertexLayer) { map.removeLayer(annVertexLayer); annVertexLayer = null; }
    annVertexMarkers_ann = [];
    annMidpointMarkers_ann = [];
}

function _refreshAnnMidpointsNear(vi) {
    const verts = _getAnnVertices(annDrag.layer);
    const n = verts.length;
    const isPolygon = annDrag.layer instanceof L.Polygon;
    const mpCount = isPolygon ? n : n - 1;
    const prev = isPolygon ? (vi - 1 + n) % n : vi - 1;
    for (const mi of [prev, vi]) {
        if (mi >= 0 && mi < mpCount && annMidpointMarkers_ann[mi]) {
            const a = verts[mi], b = verts[(mi + 1) % n];
            const midPos = [(a.lat + b.lat) / 2, (a.lng + b.lng) / 2];
            annMidpointMarkers_ann[mi].setLatLng(midPos);
            annMidpointMarkers_ann[mi]._hitMarker?.setLatLng(midPos);
        }
    }
}

function _onAnnVertexDragDown(e, i) {
    L.DomEvent.stopPropagation(e);
    annDrag.active = true;
    annDrag.type = 'vertex';
    annDrag.layer = editSelectedLayer;
    annDrag.vertexIndex = i;
    annDrag.origVertices = _getAnnVertices(editSelectedLayer).map(v => L.latLng(v.lat, v.lng));
    annDrag.startLatLng = e.latlng;
    annDrag.dragMarker = annVertexMarkers_ann[i] ?? null;
    map.dragging.disable();
    map.getContainer().style.cursor = 'grabbing';
    map.on('mousemove', _onAnnDragMove);
    map.on('mouseup', _onAnnDragUp);
}

function _onAnnMidpointDragDown(e, i) {
    L.DomEvent.stopPropagation(e);
    annDrag.active = true;
    annDrag.type = 'midpoint';
    annDrag.layer = editSelectedLayer;
    annDrag.vertexIndex = i;
    annDrag.origVertices = _getAnnVertices(editSelectedLayer).map(v => L.latLng(v.lat, v.lng));
    annDrag.startLatLng = e.latlng;
    annDrag.dragMarker = annMidpointMarkers_ann[i] ?? null;
    map.dragging.disable();
    map.getContainer().style.cursor = 'crosshair';
    map.on('mousemove', _onAnnDragMove);
    map.on('mouseup', _onAnnDragUp);
}

function _onAnnDragMove(e) {
    if (!annDrag.active) return;
    if (annDrag.type === 'vertex') {
        _setAnnVertex(annDrag.layer, annDrag.vertexIndex, e.latlng);
        annDrag.dragMarker?.setLatLng(e.latlng);
        _refreshAnnMidpointsNear(annDrag.vertexIndex);
    } else if (annDrag.type === 'midpoint') {
        annDrag.dragMarker?.setLatLng(e.latlng);
    }
}

async function _onAnnDragUp(e) {
    if (!annDrag.active) return;
    map.off('mousemove', _onAnnDragMove);
    map.off('mouseup', _onAnnDragUp);
    map.dragging.enable();
    map.getContainer().style.cursor = currentMode === 'edit' ? 'move' : '';

    const { type, layer, vertexIndex } = annDrag;
    annDrag.active = false;
    annDrag.type = null;
    annDrag.layer = null;
    annDrag.dragMarker = null;
    annDrag.origVertices = null;

    if (!layer) return;

    if (type === 'vertex') {
        _setAnnVertex(layer, vertexIndex, e.latlng);
        _saveAnnotationGeometry(layer);
        _loadAnnotationVertices(layer);
    } else if (type === 'midpoint') {
        const lls = layer.getLatLngs();
        if (Array.isArray(lls[0])) { lls[0].splice(vertexIndex + 1, 0, e.latlng); }
        else { lls.splice(vertexIndex + 1, 0, e.latlng); }
        layer.setLatLngs(lls);
        _saveAnnotationGeometry(layer);
        _loadAnnotationVertices(layer);
    }
}

// ── Annotation body drag ──────────────────────────────────────────────────

function _onEditDragDown(e) {
    L.DomEvent.stopPropagation(e);
    if (editSelectedLayer && editSelectedLayer !== e.target) {
        if (editSelectedLayer.setStyle)
            editSelectedLayer.setStyle({ ..._layerBaseStyle(editSelectedLayer), cursor: 'move' });
    }
    editSelectedLayer = e.target;
    _loadAnnotationVertices(editSelectedLayer);
    if (e.target.setStyle)
        e.target.setStyle({ ..._layerBaseStyle(e.target), color: '#ff4400', weight: 3, cursor: 'grabbing' });
    editDrag.active = true;
    editDrag.layer = e.target;
    editDrag.startLatLng = e.latlng;
    editDrag.origLatLngs = _cloneLatLngs(e.target.getLatLngs());
    editDrag.origVertices = annVertexMarkers_ann.map(m => m.getLatLng());
    map.dragging.disable();
    map.getContainer().style.cursor = 'grabbing';
}

function _onEditDragMove(e) {
    if (!editDrag.active) return;
    const dlat = e.latlng.lat - editDrag.startLatLng.lat;
    const dlng = e.latlng.lng - editDrag.startLatLng.lng;
    _applyDeltaInPlace(editDrag.layer._latlngs, editDrag.origLatLngs, dlat, dlng);
    editDrag.layer.redraw();
    // Keep vertex and midpoint markers in sync with the moving layer
    if (editDrag.origVertices && annVertexMarkers_ann.length) {
        const n = annVertexMarkers_ann.length;
        const isPolygon = editDrag.layer instanceof L.Polygon;
        annVertexMarkers_ann.forEach((m, i) => {
            const orig = editDrag.origVertices[i];
            m.setLatLng([orig.lat + dlat, orig.lng + dlng]);
        });
        const mpCount = isPolygon ? n : n - 1;
        for (let i = 0; i < mpCount; i++) {
            if (annMidpointMarkers_ann[i]) {
                const a = annVertexMarkers_ann[i].getLatLng();
                const b = annVertexMarkers_ann[(i + 1) % n].getLatLng();
                annMidpointMarkers_ann[i].setLatLng([(a.lat + b.lat) / 2, (a.lng + b.lng) / 2]);
            }
        }
    }
}

function _onEditDragUp() {
    if (!editDrag.active) return;
    editDrag.active = false;
    map.dragging.enable();
    map.getContainer().style.cursor = currentMode === 'edit' ? 'move' : '';
    const layer = editDrag.layer;
    editDrag.layer = null;
    editDrag.origVertices = null;
    if (layer) {
        _saveAnnotationGeometry(layer);
        if (editSelectedLayer === layer) _loadAnnotationVertices(layer);
    }
}

async function _saveAnnotationGeometry(layer) {
    const annId = layer.options._ann_id;
    if (!annId || !currentFile) return;
    const geom = layer.toGeoJSON().geometry;
    await saveAnnotation(currentFile, annId, geom);
    const ann = annotations.find(a => a.id === annId);
    if (ann) ann.geometry = geom;
}

function getSnappableLayers() {
    const targets = [];
    ['road', 'footway', 'barrier'].forEach(cat => {
        if (geoLayers[cat]) {
            geoLayers[cat].eachLayer(l => targets.push(l));
        }
    });
    drawnItems.eachLayer(l => targets.push(l));
    return targets;
}

function enableAnnotationEditMode() {
    map.on('mousemove', _onEditDragMove);
    map.on('mouseup', _onEditDragUp);
    drawnItems.eachLayer(layer => {
        layer.on('mousedown', _onEditDragDown);
        if (layer.setStyle) layer.setStyle({ ..._layerBaseStyle(layer), cursor: 'move' });
    });
}

function disableAnnotationEditMode() {
    map.off('mousemove', _onEditDragMove);
    map.off('mouseup', _onEditDragUp);
    if (editDrag.active) { editDrag.active = false; map.dragging.enable(); }
    if (annDrag.active) {
        annDrag.active = false;
        map.off('mousemove', _onAnnDragMove);
        map.off('mouseup', _onAnnDragUp);
        map.dragging.enable();
    }
    _clearAnnotationVertices();
    editSelectedLayer = null;
    if (osmDrag.active) {
        osmDrag.active = false;
        map.off('mousemove', _onOsmDragMove);
        map.off('mouseup', _onOsmDragUp);
        map.dragging.enable();
    }
    drawnItems.eachLayer(layer => {
        layer.off('mousedown', _onEditDragDown);
        if (layer.setStyle) layer.setStyle(_layerBaseStyle(layer));
        _saveAnnotationGeometry(layer);
    });
}

// ── OSM way / node drag ────────────────────────────────────────────────────────

const osmDrag = {
    active: false,
    type: null,        // 'node' | 'way' | 'midpoint'
    wayId: null,
    nodeIndex: -1,
    afterNodeId: null,   // node ID after which to insert (for 'midpoint' type)
    dragMarker: null,    // the midpoint marker being dragged
    startLatLng: null,
    origPositions: [],    // [{lat, lon}] snapshot of currentNodes at drag start
    origLayerLatLngs: null, // snapshot of layer latlngs at drag start (for way drag)
};

function _onOsmNodeDragDown(e, nodeIndex) {
    if (currentMode !== 'edit') return;
    L.DomEvent.stopPropagation(e);
    osmDrag.active = true;
    osmDrag.type = 'node';
    osmDrag.wayId = currentClickedFeature?.properties?.id ?? null;
    osmDrag.nodeIndex = nodeIndex;
    osmDrag.startLatLng = e.latlng;
    osmDrag.origPositions = currentNodes.map(n => ({ lat: n.lat, lon: n.lon }));
    map.dragging.disable();
    map.getContainer().style.cursor = 'grabbing';
    map.on('mousemove', _onOsmDragMove);
    map.on('mouseup', _onOsmDragUp);
}

function _onOsmWayDragDown(e) {
    if (currentMode !== 'edit' || osmDrag.active) return;
    L.DomEvent.stopPropagation(e);
    osmDrag.active = true;
    osmDrag.type = 'way';
    osmDrag.wayId = currentClickedFeature?.properties?.id ?? null;
    osmDrag.nodeIndex = -1;
    osmDrag.startLatLng = e.latlng;
    osmDrag.origPositions = currentNodes.map(n => ({ lat: n.lat, lon: n.lon }));
    osmDrag.origLayerLatLngs = currentClickedLayer ? _cloneLatLngs(currentClickedLayer.getLatLngs()) : null;
    map.dragging.disable();
    map.getContainer().style.cursor = 'grabbing';
    map.on('mousemove', _onOsmDragMove);
    map.on('mouseup', _onOsmDragUp);
}

function _onMidpointDragDown(e, segmentIndex) {
    if (currentMode !== 'edit') return;
    L.DomEvent.stopPropagation(e);
    osmDrag.active = true;
    osmDrag.type = 'midpoint';
    osmDrag.wayId = currentClickedFeature?.properties?.id ?? null;
    osmDrag.nodeIndex = segmentIndex;
    osmDrag.afterNodeId = currentNodes[segmentIndex]?.id ?? null;
    osmDrag.startLatLng = e.latlng;
    osmDrag.origPositions = currentNodes.map(n => ({ lat: n.lat, lon: n.lon }));
    osmDrag.dragMarker = midpointMarkers[segmentIndex] ?? null;
    map.dragging.disable();
    map.getContainer().style.cursor = 'crosshair';
    map.on('mousemove', _onOsmDragMove);
    map.on('mouseup', _onOsmDragUp);
}

function _onOsmDragMove(e) {
    if (!osmDrag.active) return;
    const dlat = e.latlng.lat - osmDrag.startLatLng.lat;
    const dlng = e.latlng.lng - osmDrag.startLatLng.lng;
    if (osmDrag.type === 'node') {
        const ni = osmDrag.nodeIndex;
        const newLat = osmDrag.origPositions[ni].lat + dlat;
        const newLng = osmDrag.origPositions[ni].lon + dlng;
        nodeMarkers[ni]?.setLatLng([newLat, newLng]);
        _updateOsmWayVisualNode(ni, newLat, newLng);
    } else if (osmDrag.type === 'midpoint') {
        osmDrag.dragMarker?.setLatLng(e.latlng);
    } else {
        for (let i = 0; i < nodeMarkers.length; i++) {
            nodeMarkers[i]?.setLatLng([
                osmDrag.origPositions[i].lat + dlat,
                osmDrag.origPositions[i].lon + dlng,
            ]);
        }
        _updateOsmWayVisualDelta(dlat, dlng);
    }
}

async function _onOsmDragUp(e) {
    if (!osmDrag.active) return;
    map.off('mousemove', _onOsmDragMove);
    map.off('mouseup', _onOsmDragUp);
    map.dragging.enable();
    map.getContainer().style.cursor = currentMode === 'edit' ? 'move' : '';

    const { type, wayId, nodeIndex, startLatLng, origPositions, afterNodeId } = osmDrag;
    osmDrag.active = false;
    osmDrag.type = null;
    osmDrag.wayId = null;
    osmDrag.origLayerLatLngs = null;
    osmDrag.afterNodeId = null;
    osmDrag.dragMarker = null;

    if (type === 'midpoint') {
        if (!currentFile || !wayId) return;
        const res = await addWayNodeApi(currentFile, wayId, afterNodeId, e.latlng.lat, e.latlng.lng);
        if (!res.ok) { setStatus('Add node failed', 'text-danger'); return; }
        await _reloadWay(wayId);
        if (currentClickedFeature && currentClickedLayer) {
            await loadNodesForEditing(currentClickedFeature, currentClickedLayer);
        }
        await refreshMetadata(currentFile);
        setStatus('Node added', 'text-success');
        return;
    }

    const dlat = e.latlng.lat - startLatLng.lat;
    const dlng = e.latlng.lng - startLatLng.lng;

    let nodesToSave;
    if (type === 'node') {
        const newLat = origPositions[nodeIndex].lat + dlat;
        const newLon = origPositions[nodeIndex].lon + dlng;
        nodesToSave = [{ id: currentNodes[nodeIndex].id, lat: newLat, lon: newLon }];
        currentNodes[nodeIndex].lat = newLat;
        currentNodes[nodeIndex].lon = newLon;
    } else {
        nodesToSave = currentNodes.map((n, i) => ({
            id: n.id,
            lat: origPositions[i].lat + dlat,
            lon: origPositions[i].lon + dlng,
        }));
        nodesToSave.forEach((n, i) => { currentNodes[i].lat = n.lat; currentNodes[i].lon = n.lon; });
    }

    if (!currentFile || !wayId) return;
    const _mcat = currentClickedFeature?.properties?.category ?? 'unknown';
    const _mtags0 = currentClickedFeature?.properties?.tags || {};
    const _mlbl = _mtags0.highway || _mtags0.barrier || '';
    const res = await moveWayNodesApi(currentFile, wayId, nodesToSave, _mcat, _mlbl);
    if (!res.ok) { setStatus('Move failed', 'text-danger'); return; }

    const isLine = currentClickedFeature?.geometry?.type === 'LineString';
    if (!isLine) {
        await _reloadWay(wayId);
    }

    // Record move in changelog (once per way; re-drags don't add duplicates)
    if (!changeLog.some(c => c.type === 'move' && c.id === wayId)) {
        changeLog.push({ type: 'move', id: wayId, category: _mcat, label: _mlbl });
    }
    renderChangesPanel();

    if (isLine) {
        // Update ghost to match the visually-updated layer
        if (osmDragGhost && currentClickedLayer) {
            osmDragGhost.setLatLngs(currentClickedLayer.getLatLngs());
        }
    } else if (currentMode === 'edit' && currentClickedFeature && currentClickedLayer) {
        // Re-setup node editing for the reloaded polygon layer
        await loadNodesForEditing(currentClickedFeature, currentClickedLayer);
    }

    setStatus('Way updated', 'text-success');
}

function _updateOsmWayVisualNode(ni, newLat, newLng) {
    if (!currentClickedLayer || currentClickedFeature?.geometry?.type !== 'LineString') return;
    const latlngs = currentClickedLayer.getLatLngs();
    if (ni >= latlngs.length) return;
    latlngs[ni] = L.latLng(newLat, newLng);
    // Case 1: node list repeats closing node (way.nodes = [A,B,C,D,A]) — match by id
    const nodeId = currentNodes[ni]?.id;
    if (nodeId !== undefined) {
        for (let j = 0; j < Math.min(latlngs.length, currentNodes.length); j++) {
            if (j !== ni && currentNodes[j]?.id === nodeId) {
                latlngs[j] = L.latLng(newLat, newLng);
                nodeMarkers[j]?.setLatLng([newLat, newLng]);
            }
        }
    }
    // Case 2: GeoJSON has extra closing coord beyond currentNodes length — ni===0 mirrors latlngs[last]
    if (ni === 0 && latlngs.length > currentNodes.length) {
        latlngs[latlngs.length - 1] = L.latLng(newLat, newLng);
    }
    currentClickedLayer.setLatLngs(latlngs);
}

function _updateOsmWayVisualDelta(dlat, dlng) {
    if (!currentClickedLayer || !osmDrag.origLayerLatLngs) return;
    // Apply total delta to ORIGINAL latlngs captured at drag start, not to the
    // already-moved current latlngs — otherwise each mousemove frame compounds
    // the shift and the object moves too far.
    const orig = osmDrag.origLayerLatLngs;
    const move = ll => L.latLng(ll.lat + dlat, ll.lng + dlng);
    currentClickedLayer.setLatLngs(
        (orig.length && Array.isArray(orig[0]))
            ? orig.map(ring => ring.map(move))
            : orig.map(move)
    );
}

function initDrawControl() {
    if (drawControl) return;
    drawControl = new L.Control.Draw({
        draw: {
            polyline: false,
            rectangle: { shapeOptions: STYLES.annotation },
            polygon: { allowIntersection: false, shapeOptions: STYLES.annotation },
            circle: { shapeOptions: STYLES.annotation },
            circlemarker: false,
            marker: false,
        },
        edit: { featureGroup: drawnItems, edit: false, remove: false },
    });
}

function initHandlers() {
    deleteHandler = new L.EditToolbar.Delete(map, { featureGroup: drawnItems });
}

// ── Draw events ───────────────────────────────────────────────────────────────
function setupDrawEvents() {
    map.on(L.Draw.Event.CREATED, async e => {
        // Fetch-area rectangle
        if (currentMode === 'fetch') {
            const bounds = e.layer.getBounds();
            pendingBbox = {
                min_lat: bounds.getSouth(), max_lat: bounds.getNorth(),
                min_lon: bounds.getWest(), max_lon: bounds.getEast(),
            };
            document.getElementById('area-name-input').value = '';
            new bootstrap.Modal(document.getElementById('fetch-area-modal')).show();
            setMode('view', false);
            return;
        }

        if ((currentMode !== 'add' && currentMode !== 'path') || !currentFile) return;

        let geom;
        if (e.layerType === 'circle') {
            geom = circleToPolygon(e.layer.getLatLng(), e.layer.getRadius(), 48);
        } else {
            geom = e.layer.toGeoJSON().geometry;
        }

        openAnnDetailModal(geom, currentMode);

        // Re-arm the path draw tool for the next segment
        if (currentMode === 'path' && pathLineDraw)
            setTimeout(() => { if (currentMode === 'path') pathLineDraw.enable(); }, 50);
    });

    map.on(L.Draw.Event.DELETED, e => {
        const toDelete = [];
        e.layers.eachLayer(layer => {
            const annId = layer.options._ann_id;
            if (annId) toDelete.push(annId);
        });
        Promise.all(
            toDelete.map(id => deleteAnnotationApi(currentFile, id))
        ).then(() => {
            annotations = annotations.filter(a => !toDelete.includes(a.id));
            renderAnnotationList();
            setStatus(`Deleted ${toDelete.length} annotation(s)`, 'text-success');
        });
    });

    // Switch back to view when user cancels delete via Escape
    map.on(L.Draw.Event.DELETESTOP, () => {
        if (currentMode === 'delete') {
            setMode('view');
        }
    });
}
