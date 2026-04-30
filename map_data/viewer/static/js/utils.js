// ── Geometry helpers ──────────────────────────────────────────────────────────
function circleToPolygon(center, radiusM, numPts) {
  const R = 6371000;
  const dLat = (radiusM / R) * (180 / Math.PI);
  const dLon = dLat / Math.cos(center.lat * Math.PI / 180);
  const coords = [];
  for (let i = 0; i <= numPts; i++) {
    const a = (i * 2 * Math.PI) / numPts;
    coords.push([center.lng + dLon * Math.cos(a), center.lat + dLat * Math.sin(a)]);
  }
  return { type: 'Polygon', coordinates: [coords] };
}

function _cloneLatLngs(lls) {
  if (!lls || !lls.length) return lls;
  if (Array.isArray(lls[0])) return lls.map(_cloneLatLngs);
  return lls.map(ll => ({ lat: ll.lat, lng: ll.lng }));
}

function _applyDeltaInPlace(lls, orig, dlat, dlng) {
  for (let i = 0; i < lls.length; i++) {
    if (Array.isArray(lls[i])) {
      _applyDeltaInPlace(lls[i], orig[i], dlat, dlng);
    } else {
      lls[i].lat = orig[i].lat + dlat;
      lls[i].lng = orig[i].lng + dlng;
    }
  }
}

function escHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function _annStyle(ann) {
  return ann.type === 'path' ? STYLES.path : STYLES.annotation;
}

function _layerBaseStyle(layer) {
  const ann = annotations.find(a => a.id === layer.options._ann_id);
  return _annStyle(ann || {});
}

function getSubtype(feature, cat) {
  const tags = feature.properties.tags || {};
  if (cat === 'road' || cat === 'footway') return tags.highway || 'other';
  if (cat === 'barrier') return tags.barrier || 'other';
  return 'other';
}
