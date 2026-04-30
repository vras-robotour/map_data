// ── Drawing and Editing Handlers ───────────────────────────────────────────

const editDrag = { active: false, layer: null, startLatLng: null, origLatLngs: null };

function _onEditDragDown(e) {
  L.DomEvent.stopPropagation(e);
  // Selection highlight
  if (editSelectedLayer && editSelectedLayer !== e.target && editSelectedLayer.setStyle)
    editSelectedLayer.setStyle({ ..._layerBaseStyle(editSelectedLayer), cursor: 'move' });
  editSelectedLayer = e.target;
  if (e.target.setStyle)
    e.target.setStyle({ ..._layerBaseStyle(e.target), color: '#ff4400', weight: 3, cursor: 'grabbing' });
  editDrag.active      = true;
  editDrag.layer       = e.target;
  editDrag.startLatLng = e.latlng;
  editDrag.origLatLngs = _cloneLatLngs(e.target.getLatLngs());
  map.dragging.disable();
  map.getContainer().style.cursor = 'grabbing';
}

function _onEditDragMove(e) {
  if (!editDrag.active) return;
  const dlat = e.latlng.lat - editDrag.startLatLng.lat;
  const dlng = e.latlng.lng - editDrag.startLatLng.lng;
  _applyDeltaInPlace(editDrag.layer._latlngs, editDrag.origLatLngs, dlat, dlng);
  editDrag.layer.redraw();
  const ed = editDrag.layer.editing;
  if (ed && ed._enabled) ed.updateMarkers();
}

function _onEditDragUp() {
  if (!editDrag.active) return;
  editDrag.active = false;
  map.dragging.enable();
  map.getContainer().style.cursor = currentMode === 'edit' ? 'move' : '';
  const layer = editDrag.layer;
  editDrag.layer = null;
  if (layer) _saveAnnotationGeometry(layer);
}

async function _saveAnnotationGeometry(layer) {
  const annId = layer.options._ann_id;
  if (!annId || !currentFile) return;
  const geom = layer.toGeoJSON().geometry;
  await saveAnnotation(currentFile, annId, geom);
  const ann = annotations.find(a => a.id === annId);
  if (ann) ann.geometry = geom;
}

function enableAnnotationEditMode() {
  map.on('mousemove', _onEditDragMove);
  map.on('mouseup',   _onEditDragUp);
  drawnItems.eachLayer(layer => {
    if (layer.editing) layer.editing.enable();
    layer.on('mousedown', _onEditDragDown);
    if (layer.setStyle)  layer.setStyle({ ..._layerBaseStyle(layer), cursor: 'move' });
  });
}

function disableAnnotationEditMode() {
  map.off('mousemove', _onEditDragMove);
  map.off('mouseup',   _onEditDragUp);
  if (editDrag.active) { editDrag.active = false; map.dragging.enable(); }
  editSelectedLayer = null;
  drawnItems.eachLayer(layer => {
    if (layer.editing) layer.editing.disable();
    layer.off('mousedown', _onEditDragDown);
    if (layer.setStyle)  layer.setStyle(_layerBaseStyle(layer));
    _saveAnnotationGeometry(layer);
  });
}

function initDrawControl() {
  if (drawControl) return;
  drawControl = new L.Control.Draw({
    draw: {
      polyline:     false,
      rectangle:    { shapeOptions: STYLES.annotation },
      polygon:      { allowIntersection: false, shapeOptions: STYLES.annotation },
      circle:       { shapeOptions: STYLES.annotation },
      circlemarker: false,
      marker:       false,
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
        min_lon: bounds.getWest(),  max_lon: bounds.getEast(),
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
