// ── Drawing and Editing Handlers ───────────────────────────────────────────

const editDrag = { active: false, layer: null, startLatLng: null, origLatLngs: null, origVertices: null };

// ── Annotation vertex editing (OSM-style markers) ─────────────────────────

const annDrag = {
    active: false,
    type: null,        // 'vertex' | 'midpoint'
    layer: null,
    vertexIndex: -1,
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

    annVertexMarkers = verts.map((v, i) => {
        const [m, vHit] = makeHandle(
            [v.lat, v.lng],
            { radius: 5, color: '#fff', weight: 2, fillColor: '#f0a500', fillOpacity: 0.9 },
            12, 'grab', e => _onAnnVertexDragDown(e, i),
        );
        annVertexLayer.addLayer(m);
        annVertexLayer.addLayer(vHit);
        return m;
    });

    annMidpointMarkers = [];
    const n = verts.length;
    const mpCount = isPolygon ? n : n - 1;
    for (let i = 0; i < mpCount; i++) {
        const a = verts[i], b = verts[(i + 1) % n];
        const midLat = (a.lat + b.lat) / 2, midLng = (a.lng + b.lng) / 2;
        const [mp, mpHit] = makeHandle(
            [midLat, midLng],
            { radius: 4, color: '#4af', weight: 1.5, fillColor: '#4af', fillOpacity: 0.7 },
            10, 'crosshair', e => _onAnnMidpointDragDown(e, i),
        );
        annVertexLayer.addLayer(mp);
        annVertexLayer.addLayer(mpHit);
        mp._hitMarker = mpHit;
        annMidpointMarkers.push(mp);
    }

    annVertexLayer.addTo(map);
}

function _clearAnnotationVertices() {
    if (annVertexLayer) { map.removeLayer(annVertexLayer); annVertexLayer = null; }
    annVertexMarkers = [];
    annMidpointMarkers = [];
}

function _refreshAnnMidpointsNear(vi) {
    const verts = _getAnnVertices(annDrag.layer);
    const n = verts.length;
    const isPolygon = annDrag.layer instanceof L.Polygon;
    const mpCount = isPolygon ? n : n - 1;
    const prev = isPolygon ? (vi - 1 + n) % n : vi - 1;
    for (const mi of [prev, vi]) {
        if (mi >= 0 && mi < mpCount && annMidpointMarkers[mi]) {
            const a = verts[mi], b = verts[(mi + 1) % n];
            const midPos = [(a.lat + b.lat) / 2, (a.lng + b.lng) / 2];
            annMidpointMarkers[mi].setLatLng(midPos);
            annMidpointMarkers[mi]._hitMarker?.setLatLng(midPos);
        }
    }
}

function _startAnnDrag(e, type, i) {
    L.DomEvent.stopPropagation(e);
    annDrag.active = true;
    annDrag.type = type;
    annDrag.layer = editSelectedLayer;
    annDrag.vertexIndex = i;
    annDrag.startLatLng = e.latlng;
    const ann = annotations.find(a => a.id === editSelectedLayer?.options?._ann_id);
    if (ann && editSelectedLayer && !editSelectedLayer._origGeometry) {
        editSelectedLayer._origGeometry = structuredClone(ann.geometry);
    }
    annDrag.dragMarker = (type === 'vertex' ? annVertexMarkers : annMidpointMarkers)[i] ?? null;
    map.dragging.disable();
    map.getContainer().style.cursor = type === 'vertex' ? 'grabbing' : 'crosshair';
    map.on('mousemove', _onAnnDragMove);
    map.on('mouseup', _onAnnDragUp);
}

function _onAnnVertexDragDown(e, i) { _startAnnDrag(e, 'vertex', i); }
function _onAnnMidpointDragDown(e, i) { _startAnnDrag(e, 'midpoint', i); }

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

// Shared teardown of an annotation/OSM drag's map listeners.
function _endMapDrag(move, up) {
    map.off('mousemove', move);
    map.off('mouseup', up);
    map.dragging.enable();
    map.getContainer().style.cursor = currentMode === 'edit' ? 'move' : '';
}

async function _onAnnDragUp(e) {
    if (!annDrag.active) return;
    _endMapDrag(_onAnnDragMove, _onAnnDragUp);

    const { type, layer, vertexIndex, startLatLng } = annDrag;
    annDrag.active = false;
    annDrag.type = null;
    annDrag.layer = null;
    annDrag.dragMarker = null;

    if (!layer) return;

    if (type === 'vertex') {
        const dlat = e.latlng.lat - startLatLng.lat;
        const dlng = e.latlng.lng - startLatLng.lng;
        if (Math.abs(dlat) < 1e-9 && Math.abs(dlng) < 1e-9) {
            return;
        }
        _setAnnVertex(layer, vertexIndex, e.latlng);
        layer._isDirty = true;
        await _saveAnnotationGeometry(layer);
        _loadAnnotationVertices(layer);
    } else if (type === 'midpoint') {
        const lls = layer.getLatLngs();
        if (Array.isArray(lls[0])) { lls[0].splice(vertexIndex + 1, 0, e.latlng); }
        else { lls.splice(vertexIndex + 1, 0, e.latlng); }
        layer.setLatLngs(lls);
        layer._isDirty = true;
        await _saveAnnotationGeometry(layer);
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
    const ann = annotations.find(a => a.id === e.target.options._ann_id);
    if (ann && !e.target._origGeometry) {
        e.target._origGeometry = structuredClone(ann.geometry);
    }
    editDrag.active = true;
    editDrag.moved = false;
    editDrag.layer = e.target;
    editDrag.startLatLng = e.latlng;
    editDrag.origLatLngs = _cloneLatLngs(e.target.getLatLngs());
    editDrag.origVertices = annVertexMarkers.map(m => m.getLatLng());
    map.dragging.disable();
    map.getContainer().style.cursor = 'grabbing';
}

function _onEditDragMove(e) {
    if (!editDrag.active) return;
    const dlat = e.latlng.lat - editDrag.startLatLng.lat;
    const dlng = e.latlng.lng - editDrag.startLatLng.lng;
    if (Math.abs(dlat) > 1e-9 || Math.abs(dlng) > 1e-9) {
        editDrag.moved = true;
    }
    _applyDeltaInPlace(editDrag.layer._latlngs, editDrag.origLatLngs, dlat, dlng);
    editDrag.layer.redraw();
    // Keep vertex and midpoint markers in sync with the moving layer
    if (editDrag.origVertices && annVertexMarkers.length) {
        const n = annVertexMarkers.length;
        const isPolygon = editDrag.layer instanceof L.Polygon;
        annVertexMarkers.forEach((m, i) => {
            const orig = editDrag.origVertices[i];
            m.setLatLng([orig.lat + dlat, orig.lng + dlng]);
        });
        const mpCount = isPolygon ? n : n - 1;
        for (let i = 0; i < mpCount; i++) {
            if (annMidpointMarkers[i]) {
                const a = annVertexMarkers[i].getLatLng();
                const b = annVertexMarkers[(i + 1) % n].getLatLng();
                annMidpointMarkers[i].setLatLng([(a.lat + b.lat) / 2, (a.lng + b.lng) / 2]);
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
    const moved = editDrag.moved;
    editDrag.layer = null;
    editDrag.origVertices = null;
    if (layer && moved) {
        layer._isDirty = true;
        _saveAnnotationGeometry(layer);
        if (editSelectedLayer === layer) _loadAnnotationVertices(layer);
    }
}

function _restoreAnnotationLayerGeometry(layer, geom) {
    if (!geom) return;
    if (geom.type === 'LineString') {
        layer.setLatLngs(geom.coordinates.map(c => [c[1], c[0]]));
    } else if (geom.type === 'Polygon') {
        layer.setLatLngs(geom.coordinates.map(ring => ring.map(c => [c[1], c[0]])));
    }
    layer.redraw?.();
}

async function _saveAnnotationGeometry(layer) {
    const annId = layer.options._ann_id;
    if (!annId || !currentFile) return true;
    const ann = annotations.find(a => a.id === annId);
    if (!ann) return true;
    const geom = layer.toGeoJSON().geometry;
    const prevGeom = layer._origGeometry || ann.geometry;
    const res = await updateAnnotationApi(currentFile, annId, geom, ann.type, ann.properties);
    if (!res.ok) {
        setStatus('Failed to save annotation', 'text-danger');
        ann.geometry = prevGeom;
        _restoreAnnotationLayerGeometry(layer, prevGeom);
        return false;
    }
    ann.geometry = geom;
    layer._origGeometry = geom;
    layer._isDirty = false;
    return true;
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
        _endMapDrag(_onAnnDragMove, _onAnnDragUp);
    }
    _clearAnnotationVertices();
    editSelectedLayer = null;
    if (osmDrag.active) {
        osmDrag.active = false;
        _endMapDrag(_onOsmDragMove, _onOsmDragUp);
    }
    drawnItems.eachLayer(async layer => {
        layer.off('mousedown', _onEditDragDown);
        if (layer.setStyle) layer.setStyle(_layerBaseStyle(layer));
        if (layer._isDirty) {
            await _saveAnnotationGeometry(layer);
        }
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

function _startOsmDrag(e, type, idx) {
    if (currentMode !== 'edit' || (type === 'way' && osmDrag.active)) return;
    L.DomEvent.stopPropagation(e);
    osmDrag.active = true;
    osmDrag.type = type;
    osmDrag.wayId = currentClickedFeature?.properties?.id ?? null;
    osmDrag.nodeIndex = type === 'way' ? -1 : idx;
    osmDrag.startLatLng = e.latlng;
    osmDrag.origPositions = currentNodes.map(n => ({ lat: n.lat, lon: n.lon }));
    osmDrag.origLayerLatLngs = (type === 'way' && currentClickedLayer) ? _cloneLatLngs(currentClickedLayer.getLatLngs()) : null;
    osmDrag.afterNodeId = type === 'midpoint' ? (currentNodes[idx]?.id ?? null) : null;
    osmDrag.dragMarker = type === 'midpoint' ? (midpointMarkers[idx] ?? null) : null;
    map.dragging.disable();
    map.getContainer().style.cursor = type === 'midpoint' ? 'crosshair' : 'grabbing';
    map.on('mousemove', _onOsmDragMove);
    map.on('mouseup', _onOsmDragUp);
}

function _onOsmNodeDragDown(e, nodeIndex) { _startOsmDrag(e, 'node', nodeIndex); }
function _onOsmWayDragDown(e) { _startOsmDrag(e, 'way'); }
function _onMidpointDragDown(e, segmentIndex) { _startOsmDrag(e, 'midpoint', segmentIndex); }

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
    _endMapDrag(_onOsmDragMove, _onOsmDragUp);

    const { type, wayId, nodeIndex, startLatLng, origPositions, origLayerLatLngs, afterNodeId } = osmDrag;
    osmDrag.active = false;
    osmDrag.type = null;
    osmDrag.wayId = null;
    osmDrag.origLayerLatLngs = null;
    osmDrag.afterNodeId = null;
    osmDrag.dragMarker = null;

    if (type === 'midpoint') {
        if (!currentFile || !wayId) return;
        const res = await api('POST', '/api/way_node', {
            query: { file: currentFile, way_id: wayId },
            body: { after_node_id: afterNodeId, lat: e.latlng.lat, lon: e.latlng.lng },
            readOnlyMsg: 'Editing',
        });
        if (!res.ok) { setStatus('Add node failed', 'text-danger'); return; }
        redoStack = [];
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

    if (Math.abs(dlat) < 1e-9 && Math.abs(dlng) < 1e-9) {
        origPositions.forEach((pos, i) => {
            if (nodeMarkers[i]) nodeMarkers[i].setLatLng([pos.lat, pos.lon]);
        });
        if (origLayerLatLngs && currentClickedLayer) {
            currentClickedLayer.setLatLngs(origLayerLatLngs);
        }
        return;
    }

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
    const category = currentClickedFeature?.properties?.category ?? 'unknown';
    const tags = currentClickedFeature?.properties?.tags || {};
    const label = tags.highway || tags.barrier || '';
    const res = await moveWayNodesApi(currentFile, wayId, nodesToSave, category, label);
    if (!res.ok) {
        origPositions.forEach((pos, i) => {
            if (currentNodes[i]) {
                currentNodes[i].lat = pos.lat;
                currentNodes[i].lon = pos.lon;
            }
            if (nodeMarkers[i]) {
                nodeMarkers[i].setLatLng([pos.lat, pos.lon]);
            }
        });
        const isLine = currentClickedFeature?.geometry?.type === 'LineString';
        if (isLine && origLayerLatLngs && currentClickedLayer) {
            currentClickedLayer.setLatLngs(origLayerLatLngs);
            if (osmDragGhost) osmDragGhost.setLatLngs(origLayerLatLngs);
        } else {
            await _reloadWay(wayId);
        }
        if (currentClickedFeature && currentClickedLayer) {
            await loadNodesForEditing(currentClickedFeature, currentClickedLayer);
        }
        setStatus('Move failed', 'text-danger');
        return;
    }
    redoStack = [];

    const isLine = currentClickedFeature?.geometry?.type === 'LineString';
    if (!isLine) {
        await _reloadWay(wayId);
    }

    // Record move in changelog (once per way; re-drags don't add duplicates)
    if (!changeLog.some(c => c.type === 'move' && c.id === wayId)) {
        changeLog.push({ type: 'move', id: wayId, category, label });
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
    // Offset the drag-start latlngs by the total delta, so frames don't compound.
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

// ── Draw events ───────────────────────────────────────────────────────────────
function setupDrawEvents() {
    map.on(L.Draw.Event.CREATED, async e => {
        // Fetch-area rectangle
        if (currentMode === 'fetch') {
            const bounds = e.layer.getBounds();
            const widthM = bounds.getNorthWest().distanceTo(bounds.getNorthEast());
            const heightM = bounds.getNorthWest().distanceTo(bounds.getSouthWest());
            const areaKm2 = (widthM * heightM) / 1e6;
            setMode('view', false);
            if (areaKm2 > MAX_FETCH_AREA_KM2) {
                setStatus(
                    `Selected area is ${areaKm2.toFixed(1)} km², which exceeds the ${MAX_FETCH_AREA_KM2} km² limit for a single fetch. Draw a smaller rectangle.`,
                    'text-danger'
                );
                return;
            }
            pendingBbox = {
                min_lat: bounds.getSouth(), max_lat: bounds.getNorth(),
                min_lon: bounds.getWest(), max_lon: bounds.getEast(),
            };
            document.getElementById('area-name-input').value = '';
            bootstrap.Modal.getOrCreateInstance(document.getElementById('fetch-area-modal')).show();
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

    map.on(L.Draw.Event.DELETED, async e => {
        const toDelete = [];
        e.layers.eachLayer(layer => {
            const annId = layer.options._ann_id;
            if (annId) toDelete.push(annId);
        });
        if (toDelete.length === 0) return;
        try {
            const results = await Promise.all(
                toDelete.map(id => api('DELETE', `/api/annotations/${id}`, {
                    query: { file: currentFile }, readOnlyMsg: 'Editing',
                }))
            );
            const failed = results.filter(r => !r || !r.ok);
            if (failed.length > 0) {
                setStatus('Failed to delete some annotations', 'text-danger');
                loadMapData(currentFile);
                return;
            }
            annotations = annotations.filter(a => !toDelete.includes(a.id));
            renderAnnotationList();
            setStatus(`Deleted ${toDelete.length} annotation(s)`, 'text-success');
        } catch (err) {
            setStatus(`Failed to delete annotations: ${err.message}`, 'text-danger');
            loadMapData(currentFile);
        }
    });

    // Switch back to view when user cancels delete via Escape
    map.on(L.Draw.Event.DELETESTOP, () => {
        if (currentMode === 'delete') {
            setMode('view');
        }
    });
}
