// ── API ──────────────────────────────────────────────────────────────────────

// Static-demo escape hatch: when window.__mapdataStaticBase is set the page is
// a flat-file scrape of the viewer (GitHub Pages demo) — read-only GETs map to
// pre-baked JSON files and every mutating call is refused with a status note.
const STATIC_BASE = window.__mapdataStaticBase || null;

async function _staticJson(relPath) {
    const res = await fetch(`${STATIC_BASE}/${relPath}`);
    if (!res.ok) throw new Error(`static demo file missing: ${relPath}`);
    return await res.json();
}

function _staticReadOnly(action) {
    setStatus(`${action} is not available in this read-only demo`, 'text-warning');
    throw new Error('Read-only static demo');
}

// Shared fetch wrapper for every non-static-demo API call: builds the query
// string, sets JSON headers/body (or leaves FormData bodies alone), and
// refuses up-front (via _staticReadOnly) when `readOnlyMsg` is given and the
// page is running as the read-only static demo.
async function api(method, path, { query, body, readOnlyMsg } = {}) {
    if (STATIC_BASE && readOnlyMsg) _staticReadOnly(readOnlyMsg);
    const qs = query
        ? '?' + Object.entries(query).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&')
        : '';
    const opts = { method };
    if (body !== undefined) {
        if (body instanceof FormData) {
            opts.body = body;
        } else {
            opts.headers = { 'Content-Type': 'application/json' };
            opts.body = JSON.stringify(body);
        }
    }
    return await fetch(`${path}${qs}`, opts);
}

async function fetchFileList() {
    if (STATIC_BASE) return await _staticJson('api/files.json');
    const res = await api('GET', '/api/files');
    return await res.json();
}

async function fetchMapData(filename) {
    if (STATIC_BASE) return await _staticJson('api/mapdata.json');
    const res = await api('GET', '/api/mapdata', { query: { file: filename } });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

async function fetchAnnotations(filename) {
    if (STATIC_BASE) {
        try { return await _staticJson('api/annotations.json'); }
        catch (_) { return { annotations: [] }; }
    }
    const res = await api('GET', '/api/annotations', { query: { file: filename } });
    if (!res.ok) return { annotations: [] };
    return await res.json();
}

async function deleteAnnotationApi(filename, annId) {
    return await api('DELETE', `/api/annotations/${annId}`, {
        query: { file: filename }, readOnlyMsg: 'Editing',
    });
}

async function createAnnotationApi(filename, type, geometry, properties) {
    const res = await api('POST', '/api/annotations', {
        query: { file: filename }, body: { type, geometry, properties }, readOnlyMsg: 'Editing',
    });
    return await res.json();
}

async function updateAnnotationApi(filename, annId, geometry, type, properties) {
    const res = await api('PUT', `/api/annotations/${annId}`, {
        query: { file: filename }, body: { geometry, type, properties }, readOnlyMsg: 'Editing',
    });
    return await res.json();
}

async function fetchWayNodes(filename, wayId) {
    if (STATIC_BASE) return await _staticJson(`api/way_nodes/${String(wayId).replace(':', '_')}.json`);
    const res = await api('GET', '/api/way_nodes', { query: { file: filename, way_id: wayId } });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

async function updateWayTagsApi(filename, wayId, tags, cat, lbl) {
    return await api('PUT', `/api/ways/${wayId}/tags`, {
        query: { file: filename }, body: { tags, category: cat, label: lbl }, readOnlyMsg: 'Editing',
    });
}

async function deleteWayTagsApi(filename, wayId) {
    return await api('DELETE', `/api/ways/${wayId}/tags`, { query: { file: filename }, readOnlyMsg: 'Editing' });
}

async function deleteWayApi(filename, wayId, cat, label) {
    return await api('DELETE', `/api/ways/${wayId}`, {
        query: { file: filename }, body: { category: cat, label }, readOnlyMsg: 'Editing',
    });
}

async function deleteNodeApi(filename, wayId, nodeId) {
    return await api('DELETE', '/api/way_node', {
        query: { file: filename, way_id: wayId, node_id: nodeId }, readOnlyMsg: 'Editing',
    });
}

async function addWayNodeApi(filename, wayId, afterNodeId, lat, lon) {
    return await api('POST', '/api/way_node', {
        query: { file: filename, way_id: wayId },
        body: { after_node_id: afterNodeId, lat, lon },
        readOnlyMsg: 'Editing',
    });
}

async function splitWayApi(filename, wayId, nodeId) {
    return await api('POST', '/api/ways/split', {
        query: { file: filename }, body: { way_id: wayId, node_id: nodeId }, readOnlyMsg: 'Editing',
    });
}

async function undoWaySplitApi(filename, wayId, nodeId) {
    return await api('DELETE', '/api/ways/split', {
        query: { file: filename, way_id: wayId, node_id: nodeId }, readOnlyMsg: 'Editing',
    });
}

async function hideWayApi(filename, wayId, cat, label) {
    return await api('PUT', `/api/ways/${wayId}/hide`, {
        query: { file: filename }, body: { category: cat, label }, readOnlyMsg: 'Editing',
    });
}

async function showWayApi(filename, wayId) {
    return await api('PUT', `/api/ways/${wayId}/show`, { query: { file: filename }, readOnlyMsg: 'Editing' });
}

async function restoreWayApi(filename, wayId) {
    return await api('PUT', `/api/ways/${wayId}/restore`, { query: { file: filename }, readOnlyMsg: 'Editing' });
}

async function restoreNodeApi(filename, wayId, nodeId) {
    return await api('PUT', '/api/way_node/restore', {
        query: { file: filename, way_id: wayId, node_id: nodeId }, readOnlyMsg: 'Editing',
    });
}

async function fetchWayApi(filename, wayId) {
    return await api('GET', `/api/ways/${wayId}`, { query: { file: filename }, readOnlyMsg: 'Way lookup' });
}

async function fetchWaySegmentsApi(filename, wayId) {
    const res = await api('GET', `/api/ways/${wayId}/segments`, { query: { file: filename }, readOnlyMsg: 'Way lookup' });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

async function moveWayNodesApi(filename, wayId, nodes, category, label) {
    return await api('PUT', '/api/way_nodes/move', {
        query: { file: filename, way_id: wayId },
        body: { nodes, category: category ?? 'unknown', label: label ?? '' },
        readOnlyMsg: 'Editing',
    });
}

async function undoWayNodeMovesApi(filename, wayId) {
    return await api('DELETE', '/api/way_nodes/move', {
        query: { file: filename, way_id: wayId }, readOnlyMsg: 'Editing',
    });
}

function formatFetchProgress(task) {
    const detail = task.detail || (task.status === 'parsing' ? 'Parsing OSM data…' : 'Fetching OSM data…');
    return `${detail} (${task.elapsedSeconds}s)`;
}

async function fetchAreaApi(params, onProgress) {
    const res = await api('POST', '/api/fetch_area', { body: params, readOnlyMsg: 'OSM fetching' });
    if (!res.ok) throw new Error(await res.text());
    const { task_id } = await res.json();
    const startedAt = Date.now();
    while (true) {
        await new Promise(r => setTimeout(r, 1500));
        const poll = await api('GET', `/api/fetch_area/${task_id}`);
        if (!poll.ok) throw new Error(await poll.text());
        const task = await poll.json();
        if (task.status === 'done') return task.result;
        if (task.status === 'failed') throw new Error(task.error || 'Fetch failed');
        onProgress?.({ ...task, elapsedSeconds: Math.round((Date.now() - startedAt) / 1000) });
    }
}

async function uploadGpxApi(formData) {
    const res = await api('POST', '/api/upload_gpx', { body: formData, readOnlyMsg: 'Uploading' });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

async function uploadMapdataApi(formData) {
    const res = await api('POST', '/api/upload_mapdata', { body: formData, readOnlyMsg: 'Uploading' });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}
