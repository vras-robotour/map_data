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
    // textContent -> innerHTML escapes &, <, > but not quotes; the two fixups
    // keep attribute values (value="${escHtml(v)}") safe as before.
    if (s === null || s === undefined) return '';
    return Object.assign(document.createElement('span'), { textContent: String(s) }).innerHTML
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Builds one "key / value / delete" row for the way-edit and annotation
// extra-properties forms. `cls` is the input class prefix ('we' or 'ep') the
// save handler later queries by (e.g. '.we-key').
function kvRow(cls, k = '', v = '') {
    return `<div class="d-flex gap-1 mb-1">
      <input class="form-control form-control-sm bg-dark text-light border-secondary ${cls}-key"
             placeholder="key" value="${escHtml(k)}" style="flex:1;font-size:0.75rem;">
      <input class="form-control form-control-sm bg-dark text-light border-secondary ${cls}-val"
             placeholder="value" value="${escHtml(v)}" style="flex:1;font-size:0.75rem;">
      <button type="button" class="btn btn-sm btn-outline-danger px-1"
              onclick="this.closest('.d-flex').remove()">×</button>
    </div>`;
}

// ── CSRF header on same-origin API calls ─────────────────────────────────────
// When the server runs with MAP_DATA_ACCESS_TOKEN set, cookie-authenticated
// state-changing requests must also carry a custom header as CSRF proof
// (cross-site pages can't set custom headers without a CORS preflight, which
// the server never grants). Wrapping fetch here covers every API call site in
// one place; cross-origin URLs (e.g. tile/CDN hosts) are left untouched so
// they don't suddenly require a preflight.
const _rawFetch = window.fetch.bind(window);
window.fetch = function (input, init) {
    if (typeof input === 'string' || input instanceof URL) {
        const url = String(input);
        const isAbsolute = /^[a-z][a-z0-9+.-]*:\/\//i.test(url) || url.startsWith('//');
        if (!isAbsolute || url.startsWith(window.location.origin + '/')) {
            init = Object.assign({}, init);
            const headers = new Headers(init.headers || {});
            if (!headers.has('X-Requested-With')) headers.set('X-Requested-With', 'XMLHttpRequest');
            init.headers = headers;
        }
    }
    return _rawFetch(input, init);
};

async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(String(text));
        setStatus(`Copied ${text}`, 'text-success');
    } catch (_) {
        setStatus('Clipboard unavailable', 'text-warning');
    }
}

function snapshotAnnBaselines() {
    annBaselineGeoms = {};
    annotations.forEach(a => {
        annBaselineGeoms[a.id] = JSON.parse(JSON.stringify(a.geometry));
    });
}

function _annStyle(ann) {
    if (!ann) return STYLES.annotation;
    return ann.type === 'path' ? STYLES.path : STYLES.annotation;
}

function _layerBaseStyle(layer) {
    const ann = annotations.find(a => a.id === layer.options._ann_id);
    return _annStyle(ann || {});
}

// Builds a visible circle-marker handle plus a larger transparent hit-target
// marker at the same point, both wired to the same mousedown handler and
// hover cursor — the pattern every draggable vertex/midpoint/node handle uses.
// Returns [visible, hit]; the caller adds both to whatever layer group it uses.
function makeHandle(latlng, style, hitRadius, cursor, onDown) {
    const visible = L.circleMarker(latlng, { bubblingMouseEvents: false, renderer: L.svg(), ...style });
    visible.on('mousedown', onDown);
    visible.on('add', () => { const el = visible.getElement(); if (el) el.style.cursor = cursor; });

    const hit = L.circleMarker(latlng, {
        radius: hitRadius, fillOpacity: 0, opacity: 0,
        bubblingMouseEvents: false, renderer: L.svg(), interactive: true,
    });
    hit.on('mousedown', onDown);
    hit.on('add', () => { const el = hit.getElement(); if (el) el.style.cursor = cursor; });

    return [visible, hit];
}

// Opens a Leaflet popup styled as a small context menu at `latlng`, with one
// button per [label, onClick] entry. The popup is closed before the handler
// runs, same as every context menu in the app already did by hand.
function showMenu(latlng, entries, { minWidth = 150 } = {}) {
    const container = document.createElement('div');
    container.className = 'context-menu';
    entries.forEach(([label, onClick]) => {
        const btn = document.createElement('button');
        btn.innerHTML = label;
        btn.onclick = () => { map.closePopup(); onClick(); };
        container.appendChild(btn);
    });
    L.popup({ minWidth, className: 'planner-popup', offset: [0, -5], closeButton: false })
        .setLatLng(latlng)
        .setContent(container)
        .openOn(map);
}

// Reads the grid_margin/obstacle_radius/buffer_widths fields shared by the
// fetch-area, GPX-upload advanced options (see the `advanced` Jinja macro).
function readFetchOptions(prefix) {
    return {
        grid_margin: parseFloat(document.getElementById(`${prefix}-grid-margin`)?.value) || 150,
        obstacle_radius: parseFloat(document.getElementById(`${prefix}-obstacle-radius`)?.value) || 2.0,
        buffer_widths: {
            road: parseFloat(document.getElementById(`${prefix}-buf-road`)?.value) || 7.0,
            footway: parseFloat(document.getElementById(`${prefix}-buf-footway`)?.value) || 3.0,
            barrier: parseFloat(document.getElementById(`${prefix}-buf-barrier`)?.value) || 2.0,
        },
    };
}

function getSubtype(feature, cat) {
    const tags = feature.properties.tags || {};
    if (cat === 'road' || cat === 'footway') return tags.highway || 'other';
    if (cat === 'barrier') return tags.barrier || 'other';
    return 'other';
}
