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
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
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

function getSubtype(feature, cat) {
    const tags = feature.properties.tags || {};
    if (cat === 'road' || cat === 'footway') return tags.highway || 'other';
    if (cat === 'barrier') return tags.barrier || 'other';
    return 'other';
}
