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
    const cat = ann.type === 'path' ? 'path' : 'annotation';
    layer.on('click', e => {
      L.DomEvent.stopPropagation(e);
      if (currentMode === 'view') {
        if (currentClickedLayer && currentClickedLayer !== layer) {
          const oldCat = currentClickedLayer._osmCat;
          currentClickedLayer.setStyle(oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options._ann_id)));
        }
        layer._osmCat = cat;
        currentClickedLayer = layer;
        layer.setStyle(HIGHLIGHT_STYLES[cat]);
        showAnnProps(ann);
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

async function loadMapData(filename, { preserveView = false } = {}) {
  setStatus('Loading…', 'text-warning');

  currentClickedLayer  = null;
  currentClickedFeature = null;
  Object.values(geoLayers).forEach(l => l && map.removeLayer(l));
  Object.keys(geoLayers).forEach(k => geoLayers[k] = null);
  drawnItems.clearLayers();

  try {
    const geojson = await fetchMapData(filename);
    const annData = await fetchAnnotations(filename);

    const byCategory = { road: [], footway: [], barrier: [], waypoint: [] };
    geojson.features.forEach(f => {
      const cat = f.properties.category;
      if (cat in byCategory) byCategory[cat].push(f);
    });

    // Pre-compute hidden way IDs before building layers
    const _loadHiddenMeta = (annData.hidden_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
    const _loadHiddenIds  = new Set(_loadHiddenMeta.map(d => d.id));

    // Reset subtype state for fresh load
    ['road', 'footway', 'barrier'].forEach(cat => {
      subtypeLayers[cat]  = {};
      subtypeFilters[cat] = {};
    });

    ['road', 'footway', 'barrier'].forEach(cat => {
      const features = cat === 'barrier'
        ? [
            ...byCategory[cat].filter(f => !f.properties.is_node),
            ...byCategory[cat].filter(f =>  f.properties.is_node),
          ]
        : byCategory[cat];

      const layerOpts = {
        style: () => STYLES[cat],
        onEachFeature: (feature, layer) => {
          const st = getSubtype(feature, cat);
          if (!subtypeLayers[cat][st]) { subtypeLayers[cat][st] = []; subtypeFilters[cat][st] = true; }
          subtypeLayers[cat][st].push(layer);
          layer._featureId  = feature.properties.id;
          layer._featureRef = feature;
          layer.on('click', e => {
            L.DomEvent.stopPropagation(e);
            if (currentMode === 'view') {
              if (currentClickedLayer && currentClickedLayer !== layer) {
                const oldCat = currentClickedLayer._osmCat;
                currentClickedLayer.setStyle(oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options._ann_id)));
              }
              layer._osmCat = cat;
              currentClickedLayer = layer;
              layer.setStyle(HIGHLIGHT_STYLES[cat]);
              showProps(feature.properties, feature);
            }
          });
        },
      };

      geoLayers[cat] = L.geoJSON({ type: 'FeatureCollection', features }, layerOpts);
      if (_loadHiddenIds.size > 0) {
        const _toHide = [];
        geoLayers[cat].eachLayer(l => { if (_loadHiddenIds.has(l._featureId)) _toHide.push(l); });
        _toHide.forEach(l => geoLayers[cat].removeLayer(l));
      }
      const cb = document.querySelector(`[data-layer="${cat}"]`);
      if (cb && cb.checked) geoLayers[cat].addTo(map);
      renderSubtypeFilters(cat);
    });

    geoLayers.waypoint = L.geoJSON(
      { type: 'FeatureCollection', features: byCategory.waypoint },
      {
        pointToLayer: (_, latlng) =>
          L.circleMarker(latlng, {
            radius: 5, color: '#000', weight: 1, fillColor: '#50C2F6', fillOpacity: 0.9,
          }),
        onEachFeature: (feature, layer) => {
          layer.on('click', e => {
            L.DomEvent.stopPropagation(e);
            if (currentMode === 'view') {
              if (currentClickedLayer && currentClickedLayer !== layer) {
                const oldCat = currentClickedLayer._osmCat;
                currentClickedLayer.setStyle(oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options._ann_id)));
              }
              currentClickedLayer = layer;
              // Waypoints don't have a highlight style yet, but we store them
              showProps(feature.properties);
            }
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

    annotations  = annData.annotations  || [];
    deletedWays  = (annData.deleted_ways  || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
    deletedNodes = (() => {
      const dn = annData.deleted_nodes || [];
      if (Array.isArray(dn)) return dn;
      return Object.entries(dn).flatMap(([wid, nids]) => nids.map(nid => ({ way_id: +wid, node_id: nid })));
    })();
    const tagOvMap  = annData.tag_overrides      || {};
    const tagOvMeta = annData.tag_override_meta  || {};
    tagOverrides = Object.entries(tagOvMap).map(([sid, tags]) => ({
      id:       +sid,
      category: tagOvMeta[sid]?.category || 'unknown',
      label:    tagOvMeta[sid]?.label    || '',
      tags,
    }));
    hiddenWays   = _loadHiddenMeta;
    hiddenWayIds = _loadHiddenIds;
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

async function _reloadWay(wayId) {
  if (!currentFile) return;

  const affectedCats = new Set();
  for (const cat of ['road', 'footway', 'barrier']) {
    for (const st of Object.keys(subtypeLayers[cat])) {
      const before = subtypeLayers[cat][st];
      const toRemove = before.filter(l => l._featureId === wayId);
      if (!toRemove.length) continue;
      affectedCats.add(cat);
      toRemove.forEach(l => geoLayers[cat]?.removeLayer(l));
      subtypeLayers[cat][st] = before.filter(l => l._featureId !== wayId);
    }
  }

  const res = await fetchWayApi(currentFile, wayId);
  if (res.ok) {
    const feature = await res.json();
    const newCat  = feature.properties.category;
    const catLayer = geoLayers[newCat];
    if (catLayer) {
      const tmpGeo  = L.geoJSON(feature, {
        style: () => STYLES[newCat],
      });
      const newLayer = tmpGeo.getLayers()[0];
      if (newLayer) {
        tmpGeo.removeLayer(newLayer);
        const st = getSubtype(feature, newCat);
        if (!subtypeLayers[newCat][st]) { subtypeLayers[newCat][st] = []; subtypeFilters[newCat][st] = true; }
        subtypeLayers[newCat][st].push(newLayer);
        newLayer._featureId  = feature.properties.id;
        newLayer._featureRef = feature;
        newLayer.on('click', e => {
          L.DomEvent.stopPropagation(e);
          if (currentMode === 'view') {
            if (currentClickedLayer && currentClickedLayer !== newLayer) {
              const oldCat = currentClickedLayer._osmCat;
              currentClickedLayer.setStyle(oldCat ? STYLES[oldCat] : _annStyle(annotations.find(a => a.id === currentClickedLayer.options._ann_id)));
            }
            newLayer._osmCat    = newCat;
            currentClickedLayer = newLayer;
            newLayer.setStyle(HIGHLIGHT_STYLES[newCat]);
            showProps(feature.properties, feature);
          }
        });
        if (subtypeFilters[newCat][st] !== false && !hiddenWayIds.has(feature.properties.id))
          catLayer.addLayer(newLayer);
        affectedCats.add(newCat);
      }
    }
  }

  affectedCats.forEach(c => renderSubtypeFilters(c));
  clearNodes();
  await _refreshAnnotationsState();
  if (!_reselectFeature(wayId)) {
    currentClickedLayer = null; currentClickedFeature = null;
    document.getElementById('props-content').innerHTML =
      '<span class="text-secondary" style="font-size:0.8rem;font-style:italic;">Click a feature to inspect</span>';
  }
}

async function _refreshAnnotationsState() {
  const annData = await fetchAnnotations(currentFile);
  deletedWays  = (annData.deleted_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
  deletedNodes = (() => {
    const dn = annData.deleted_nodes || [];
    if (Array.isArray(dn)) return dn;
    return Object.entries(dn).flatMap(([wid, nids]) => nids.map(nid => ({ way_id: +wid, node_id: nid })));
  })();
  const tagOvMap  = annData.tag_overrides     || {};
  const tagOvMeta = annData.tag_override_meta || {};
  tagOverrides = Object.entries(tagOvMap).map(([sid, tags]) => ({
    id: +sid, category: tagOvMeta[sid]?.category || 'unknown',
    label: tagOvMeta[sid]?.label || '', tags,
  }));
  hiddenWays   = (annData.hidden_ways || []).map(d => typeof d === 'object' ? d : { id: d, category: 'unknown', label: '' });
  hiddenWayIds = new Set(hiddenWays.map(d => d.id));
  renderChangesPanel();
  renderHiddenPanel();
}

function _reselectFeature(wayId) {
  for (const cat of ['road', 'footway', 'barrier']) {
    const catLayer = geoLayers[cat];
    if (!catLayer) continue;
    let found = null;
    catLayer.eachLayer(layer => { if (layer._featureId === wayId) found = layer; });
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
