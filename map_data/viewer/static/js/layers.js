// ── Layers Management ───────────────────────────────────────────────────────

function setSubtypeVisible(cat, subtype, visible) {
    subtypeFilters[cat][subtype] = visible;
    const catLayer = geoLayers[cat];
    if (!catLayer) return;
    (subtypeLayers[cat][subtype] || []).forEach(layer => {
        if (visible) {
            if (!catLayer.hasLayer(layer) && !hiddenWayIds.has(layer._featureId))
                catLayer.addLayer(layer);
        } else {
            catLayer.removeLayer(layer);
        }
    });
}

function addAnnotationToLayer(ann) {
    L.geoJSON(ann.geometry, { style: _annStyle(ann) }).eachLayer(layer => {
        layer.options._ann_id = ann.id;
        ann._layer = layer;
        const cat = ann.type === 'path' ? 'path' : 'annotation';

        layer.on('click', e => {
            if (currentAppMode === 'planner') return;
            L.DomEvent.stopPropagation(e);
            if (currentMode === 'view') {
                selectAnnotation(ann, layer);
            }
        });

        layer.on('contextmenu', e => {
            if (currentAppMode === 'planner') return;
            L.DomEvent.stopPropagation(e);
            L.DomEvent.preventDefault(e);
            if (currentMode === 'view') {
                showAnnotationContextMenu(ann, layer, e.latlng);
            }
        });

        drawnItems.addLayer(layer);
    });
}

function renderAnnotationLayer() {
    drawnItems.clearLayers();
    annotations.forEach(addAnnotationToLayer);
    const cb = document.querySelector(`[data-layer="annotation"]`);
    if (cb) {
        if (!cb.checked) map.removeLayer(drawnItems);
        else if (!map.hasLayer(drawnItems)) drawnItems.addTo(map);
    }
}

function filterLayers(query) {
    const q = query.toLowerCase().trim();

    // OSM categories: iterate subtypeLayers (persistent) so removed layers can be restored
    ['road', 'footway', 'barrier'].forEach(cat => {
        Object.values(subtypeLayers[cat] || {}).forEach(layers => {
            layers.forEach(l => {
                let visible = true;
                if (q) {
                    const id = String(l._featureId || '').toLowerCase();
                    const tags = l._featureRef?.properties?.tags || {};
                    const tagText = Object.values(tags).join(' ').toLowerCase();
                    visible = id.includes(q) || tagText.includes(q);
                }
                if (visible) {
                    const st = getSubtype(l._featureRef, cat);
                    if (subtypeFilters[cat][st] !== false && !hiddenWayIds.has(l._featureId)) {
                        if (!geoLayers[cat].hasLayer(l)) geoLayers[cat].addLayer(l);
                    } else {
                        geoLayers[cat].removeLayer(l);
                    }
                } else {
                    geoLayers[cat].removeLayer(l);
                }
            });
        });
    });

    // Annotations: iterate annotations array (persistent) using stored _layer reference
    annotations.forEach(ann => {
        const l = ann._layer;
        if (!l) return;
        let visible = true;
        if (q) {
            const id = String(ann.id || '').toLowerCase();
            const props = ann.properties || {};
            const annText = [ann.type || '', ...Object.keys(props), ...Object.values(props)]
                .join(' ').toLowerCase();
            visible = id.includes(q) || annText.includes(q);
        }
        if (visible) {
            if (!drawnItems.hasLayer(l)) drawnItems.addLayer(l);
        } else {
            drawnItems.removeLayer(l);
        }
    });
}

function toggleMapInteractivity(interactive) {
    ['road', 'footway', 'barrier', 'crossroad', 'waypoint'].forEach(cat => {
        const layer = geoLayers[cat];
        if (layer) {
            layer.eachLayer(l => {
                if (l.getElement()) {
                    l.getElement().style.pointerEvents = interactive ? 'auto' : 'none';
                }
            });
        }
    });

    drawnItems.eachLayer(l => {
        if (l.getElement()) {
            l.getElement().style.pointerEvents = interactive ? 'auto' : 'none';
        }
    });
}

function setupWayLayer(feature, layer, cat) {
    const st = getSubtype(feature, cat);
    if (!subtypeLayers[cat][st]) { subtypeLayers[cat][st] = []; subtypeFilters[cat][st] = true; }
    subtypeLayers[cat][st].push(layer);
    layer._featureId = feature.properties.id;
    layer._featureRef = feature;

    layer.on('click', e => {
        if (currentAppMode === 'planner') return;
        L.DomEvent.stopPropagation(e);
        if (currentMode === 'view') {
            selectWay(feature, layer, cat);
        } else if (currentMode === 'edit' && cat !== 'crossroad') {
            if (currentClickedLayer && currentClickedLayer !== layer) {
                const oldCat = currentClickedLayer._osmCat;
                currentClickedLayer.setStyle(oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options._ann_id)));
            }
            layer._osmCat = cat;
            currentClickedLayer = layer;
            currentClickedFeature = feature;
            layer.setStyle(HIGHLIGHT_STYLES[cat]);
            loadNodesForEditing(feature, layer);
        } else if (currentMode === 'delete' && cat !== 'crossroad') {
            layer._osmCat = cat;
            currentClickedLayer = layer;
            currentClickedFeature = feature;
            deleteCurrentWay();
        }
    });

    layer.on('contextmenu', e => {
        if (currentAppMode === 'planner') return;
        L.DomEvent.stopPropagation(e);
        L.DomEvent.preventDefault(e);
        if (currentMode === 'view' && cat !== 'crossroad') {
            showWayContextMenu(feature, layer, e.latlng, cat);
        }
    });
}

function updateWayWithSegments(originalWayId, segments) {
    const originalWayIdStr = String(originalWayId);
    // Find and remove old segments/original way from all category layers
    ['road', 'footway', 'barrier'].forEach(cat => {
        const layerGroup = geoLayers[cat];
        if (!layerGroup) return;

        const toRemove = [];
        layerGroup.eachLayer(l => {
            // Check if ID matches original or starts with "original:"
            const sid = String(l._featureId || '');
            if (sid === originalWayIdStr || sid.startsWith(originalWayIdStr + ':')) {
                toRemove.push(l);
            }
        });

        toRemove.forEach(l => {
            // Remove from subtypeLayers
            if (l._featureRef) {
                const st = getSubtype(l._featureRef, cat);
                if (subtypeLayers[cat][st]) {
                    subtypeLayers[cat][st] = subtypeLayers[cat][st].filter(item => item !== l);
                }
            }
            layerGroup.removeLayer(l);
        });
    });

    // Add new segments
    let firstLayer = null;
    segments.forEach(f => {
        const cat = f.properties.category;
        const layerGroup = geoLayers[cat];
        if (!layerGroup) return;

        L.geoJSON(f, {
            style: () => STYLES[cat],
            onEachFeature: (feature, layer) => {
                setupWayLayer(feature, layer, cat);
                if (!firstLayer) firstLayer = layer;
            }
        }).eachLayer(l => l.addTo(layerGroup)); // Add individual layers, not the group
    });
    return firstLayer;
}

async function refreshMetadata(filename, { refreshAnnotations = false } = {}) {
    try {
        const annData = await fetchAnnotations(filename);

        annotations = annData.annotations || [];
        snapshotAnnBaselines();
        deletedWays = (annData.deleted_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
        deletedNodes = (() => {
            const dn = annData.deleted_nodes || [];
            if (Array.isArray(dn)) return dn;
            return Object.entries(dn).flatMap(([wid, nids]) => nids.map(nid => ({ way_id: +wid, node_id: nid })));
        })();
        const tagOvMap = annData.tag_overrides || {};
        const tagOvMeta = annData.tag_override_meta || {};
        tagOverrides = Object.entries(tagOvMap).map(([sid, tags]) => ({
            id: +sid,
            category: tagOvMeta[sid]?.category || 'unknown',
            label: tagOvMeta[sid]?.label || '',
            tags,
        }));

        hiddenWays = (annData.hidden_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
        hiddenWayIds = new Set(hiddenWays.map(d => d.id));

        const rawChangeLog = (annData.change_log && annData.change_log.length > 0) ? annData.change_log : null;
        const addedNodes = annData.added_nodes || [];
        if (rawChangeLog) {
            const wayMap = new Map(deletedWays.map(d => [d.id, d]));
            // Use string keys: server stores tag ids as strings, tagOverrides.id is a number (+sid)
            const tagMap = new Map(tagOverrides.map(d => [String(d.id), d]));
            const nodePosOverrides = annData.node_position_overrides || {};
            const addedNodeMap = new Map(addedNodes.map(a => [a.id, a]));
            const sorted = [...rawChangeLog].sort((a, b) => (a.ts || 0) - (b.ts || 0));
            changeLog = sorted.flatMap(e => {
                if (e.type === 'way') { const d = wayMap.get(e.id); return d ? [{ type: 'way', ts: e.ts, ...d }] : []; }
                if (e.type === 'tag') { const d = tagMap.get(String(e.id)); return d ? [{ type: 'tag', ts: e.ts, ...d }] : []; }
                if (e.type === 'node') {
                    const d = deletedNodes.find(n => n.way_id === e.way_id && n.node_id === e.node_id);
                    return d ? [{ type: 'node', ts: e.ts, ...d }] : [];
                }
                if (e.type === 'move') {
                    return nodePosOverrides[String(e.id)] ? [{ type: 'move', ts: e.ts, id: e.id, category: e.category || 'unknown', label: e.label || '' }] : [];
                }
                if (e.type === 'split') {
                    return [{ type: 'split', ts: e.ts, way_id: e.way_id, node_id: e.node_id }];
                }
                if (e.type === 'add_node') {
                    return addedNodeMap.has(e.node_id) ? [{ type: 'add_node', ts: e.ts, way_id: e.way_id, node_id: e.node_id }] : [];
                }
                return [];
            });
        } else {
            const splits = annData.split_ways || {};
            const splitItems = Object.entries(splits).flatMap(([wid, nids]) => nids.map(nid => ({ type: 'split', way_id: +wid, node_id: nid })));
            changeLog = [
                ...deletedWays.map(d => ({ type: 'way', ...d })),
                ...deletedNodes.map(d => ({ type: 'node', ...d })),
                ...tagOverrides.map(d => ({ type: 'tag', ...d })),
                ...splitItems,
                ...addedNodes.map(a => ({ type: 'add_node', way_id: a.way_id, node_id: a.id })),
            ];
        }
        if (refreshAnnotations) {
            renderAnnotationLayer();
        }
        renderAnnotationList();
        renderChangesPanel();
        renderHiddenPanel();
    } catch (err) {
        console.error('Failed to refresh metadata:', err);
    }
}

async function loadMapData(filename, { preserveView = false, silent = false } = {}) {
    if (!silent) setStatus('Loading…', 'text-warning');

    try {
        const geojson = await fetchMapData(filename);
        const annData = await fetchAnnotations(filename);

        const oldGeoLayers = { ...geoLayers };
        const oldDrawnItems = drawnItems; // We'll clear it below if needed

        currentClickedLayer = null;
        currentClickedFeature = null;

        const byCategory = { road: [], footway: [], barrier: [], waypoint: [], crossroad: [] };
        geojson.features.forEach(f => {
            const cat = f.properties.category;
            if (cat in byCategory) byCategory[cat].push(f);
        });

        // Pre-compute hidden way IDs before building layers
        const _loadHiddenMeta = (annData.hidden_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
        const _loadHiddenIds = new Set(_loadHiddenMeta.map(d => d.id));

        // Reset subtype state for fresh load
        ['road', 'footway', 'barrier', 'crossroad'].forEach(cat => {
            subtypeLayers[cat] = {};
            subtypeFilters[cat] = {};
        });

        // Remove old layers before adding new ones
        Object.values(oldGeoLayers).forEach(l => l && map.removeLayer(l));
        drawnItems.clearLayers();

        ['road', 'footway', 'barrier', 'crossroad'].forEach(cat => {
            const features = cat === 'barrier'
                ? [
                    ...byCategory[cat].filter(f => !f.properties.is_node),
                    ...byCategory[cat].filter(f => f.properties.is_node),
                ]
                : byCategory[cat];

            const layerOpts = {
                style: () => STYLES[cat],
                onEachFeature: (feature, layer) => setupWayLayer(feature, layer, cat),
            };

            geoLayers[cat] = L.geoJSON({ type: 'FeatureCollection', features }, layerOpts);
            if (_loadHiddenIds.size > 0) {
                const _toHide = [];
                geoLayers[cat].eachLayer(l => { if (_loadHiddenIds.has(l._featureId)) _toHide.push(l); });
                _toHide.forEach(l => geoLayers[cat].removeLayer(l));
            }
            const cb = document.querySelector(`[data-layer="${cat}"]`);
            if (cb && cb.checked) geoLayers[cat].addTo(map);
            if (cat !== 'crossroad') renderSubtypeFilters(cat);
        });

        geoLayers.waypoint = L.geoJSON(
            { type: 'FeatureCollection', features: byCategory.waypoint },
            {
                pointToLayer: (_, latlng) => L.circleMarker(latlng, STYLES.waypoint),
                onEachFeature: (feature, layer) => {
                    layer.on('click', e => {
                        if (currentAppMode === 'planner') return;
                        L.DomEvent.stopPropagation(e);
                        if (currentMode === 'view') {
                            selectWay(feature, layer, 'waypoint');
                        }
                    });
                    layer.on('contextmenu', e => {
                        if (currentAppMode === 'planner') return;
                        L.DomEvent.stopPropagation(e);
                        L.DomEvent.preventDefault(e);
                    });
                },
            }
        );
        const wpCb = document.querySelector(`[data-layer="waypoint"]`);
        if (wpCb && wpCb.checked) geoLayers.waypoint.addTo(map);

        if (!preserveView) {
            const allFeatures = [...byCategory.road, ...byCategory.footway, ...byCategory.waypoint];
            if (allFeatures.length > 0) {
                map.fitBounds(
                    L.geoJSON({ type: 'FeatureCollection', features: allFeatures }).getBounds(),
                    { padding: [20, 20] }
                );
            }
        }

        annotations = annData.annotations || [];
        snapshotAnnBaselines();
        deletedWays = (annData.deleted_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
        deletedNodes = (() => {
            const dn = annData.deleted_nodes || [];
            if (Array.isArray(dn)) return dn;
            return Object.entries(dn).flatMap(([wid, nids]) => nids.map(nid => ({ way_id: +wid, node_id: nid })));
        })();
        const tagOvMap = annData.tag_overrides || {};
        const tagOvMeta = annData.tag_override_meta || {};
        tagOverrides = Object.entries(tagOvMap).map(([sid, tags]) => ({
            id: +sid,
            category: tagOvMeta[sid]?.category || 'unknown',
            label: tagOvMeta[sid]?.label || '',
            tags,
        }));
        hiddenWays = _loadHiddenMeta;
        hiddenWayIds = _loadHiddenIds;
        const rawChangeLog = (annData.change_log && annData.change_log.length > 0) ? annData.change_log : null;
        const addedNodes = annData.added_nodes || [];
        if (rawChangeLog) {
            const wayMap = new Map(deletedWays.map(d => [d.id, d]));
            const tagMap = new Map(tagOverrides.map(d => [String(d.id), d]));
            const nodePosOverrides = annData.node_position_overrides || {};
            const addedNodeMap = new Map(addedNodes.map(a => [a.id, a]));
            const sorted = [...rawChangeLog].sort((a, b) => (a.ts || 0) - (b.ts || 0));
            changeLog = sorted.flatMap(e => {
                if (e.type === 'way') { const d = wayMap.get(e.id); return d ? [{ type: 'way', ts: e.ts, ...d }] : []; }
                if (e.type === 'tag') { const d = tagMap.get(String(e.id)); return d ? [{ type: 'tag', ts: e.ts, ...d }] : []; }
                if (e.type === 'node') {
                    const d = deletedNodes.find(n => n.way_id === e.way_id && n.node_id === e.node_id);
                    return d ? [{ type: 'node', ts: e.ts, ...d }] : [];
                }
                if (e.type === 'move') {
                    return nodePosOverrides[String(e.id)] ? [{ type: 'move', ts: e.ts, id: e.id, category: e.category || 'unknown', label: e.label || '' }] : [];
                }
                if (e.type === 'split') {
                    return [{ type: 'split', ts: e.ts, way_id: e.way_id, node_id: e.node_id }];
                }
                if (e.type === 'add_node') {
                    return addedNodeMap.has(e.node_id) ? [{ type: 'add_node', ts: e.ts, way_id: e.way_id, node_id: e.node_id }] : [];
                }
                return [];
            });
        } else {
            const splits = annData.split_ways || {};
            const splitItems = Object.entries(splits).flatMap(([wid, nids]) => nids.map(nid => ({ type: 'split', way_id: +wid, node_id: nid })));
            changeLog = [
                ...deletedWays.map(d => ({ type: 'way', ...d })),
                ...deletedNodes.map(d => ({ type: 'node', ...d })),
                ...tagOverrides.map(d => ({ type: 'tag', ...d })),
                ...splitItems,
                ...addedNodes.map(a => ({ type: 'add_node', way_id: a.way_id, node_id: a.id })),
            ];
        }
        renderAnnotationLayer();
        renderAnnotationList();
        renderChangesPanel();
        renderHiddenPanel();

        currentFile = filename;
        document.getElementById('export-btn').disabled = false;
        setStatus(`Loaded: ${filename}`, 'text-success');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'text-danger');
        console.error(err);
    }
}

function clearMapData() {
  deselectCurrent();
  clearNodes();
  if (plannerMode) plannerMode.clearAll();

  currentFile = "";
  
  // Clear all geo layers from map and reset state
  Object.values(geoLayers).forEach(l => {
    if (l) {
      map.removeLayer(l);
    }
  });
  drawnItems.clearLayers();
  
  // Reset layers state
  ['road', 'footway', 'barrier', 'crossroad', 'waypoint'].forEach(cat => {
    geoLayers[cat] = null;
    if (cat !== 'waypoint') {
      subtypeLayers[cat] = {};
      subtypeFilters[cat] = {};
      renderSubtypeFilters(cat);
    }
  });

  // Reset data state
  annotations = [];
  annBaselineGeoms = {};
  deletedWays = [];
  deletedNodes = [];
  tagOverrides = [];
  hiddenWays = [];
  hiddenWayIds.clear();
  changeLog = [];

  // Reset UI elements
  document.getElementById('file-select').value = "";
  document.getElementById('export-btn').disabled = true;
  document.getElementById('feature-search').value = "";
  
  renderAnnotationList();
  renderChangesPanel();
  renderHiddenPanel();

  setStatus('Map cleared', 'text-secondary');
}

async function _reloadWay(wayId) {
    if (!currentFile) return;

    // Always use the segments endpoint so that split segments are removed and
    // re-added atomically — the old non-split path used strict id equality which
    // failed to remove virtual "id:n" layers when the original numeric id was given.
    const originalWayId = String(wayId).split(':')[0];

    try {
        const data = await fetchWaySegmentsApi(currentFile, originalWayId);
        updateWayWithSegments(originalWayId, data.segments);
        ['road', 'footway', 'barrier'].forEach(c => renderSubtypeFilters(c));
        _enforceLayerOrder();
        clearNodes();
        await _refreshAnnotationsState();

        // Try the exact id first, then the bare original id, then any segment
        let reselected = _reselectFeature(wayId);
        if (!reselected) reselected = _reselectFeature(originalWayId);
        if (!reselected) {
            for (let i = 0; i < 10; i++) {
                if (_reselectFeature(`${originalWayId}:${i}`)) { reselected = true; break; }
            }
        }
        if (!reselected) {
            currentClickedLayer = null;
            currentClickedFeature = null;
            document.getElementById('props-content').innerHTML =
                '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
        }
    } catch (err) {
        console.error('Failed to reload way:', err);
        currentClickedLayer = null;
        currentClickedFeature = null;
        document.getElementById('props-content').innerHTML =
            '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
    }
}

async function _refreshAnnotationsState() {
    const annData = await fetchAnnotations(currentFile);
    deletedWays = (annData.deleted_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
    deletedNodes = (() => {
        const dn = annData.deleted_nodes || [];
        if (Array.isArray(dn)) return dn;
        return Object.entries(dn).flatMap(([wid, nids]) => nids.map(nid => ({ way_id: +wid, node_id: nid })));
    })();
    const tagOvMap = annData.tag_overrides || {};
    const tagOvMeta = annData.tag_override_meta || {};
    tagOverrides = Object.entries(tagOvMap).map(([sid, tags]) => ({
        id: +sid, category: tagOvMeta[sid]?.category || 'unknown',
        label: tagOvMeta[sid]?.label || '', tags,
    }));
    hiddenWays = (annData.hidden_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
    hiddenWayIds = new Set(hiddenWays.map(d => d.id));
    renderChangesPanel();
    renderHiddenPanel();
}

function _reselectFeature(wayId) {
    for (const cat of ['road', 'footway', 'barrier']) {
        const catLayer = geoLayers[cat];
        if (!catLayer) continue;
        let found = null;
        catLayer.eachLayer(layer => { if (String(layer._featureId) === String(wayId)) found = layer; });
        if (found) {
            if (currentClickedLayer && currentClickedLayer !== found)
                currentClickedLayer.setStyle(STYLES[currentClickedLayer._osmCat]);
            found._osmCat = cat;
            currentClickedLayer = found;
            found.setStyle(HIGHLIGHT_STYLES[cat]);
            currentClickedFeature = found._featureRef;
            showProps(found._featureRef.properties, found._featureRef);
            return true;
        }
    }
    return false;
}
