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

async function fetchFileList() {
    if (STATIC_BASE) return await _staticJson('api/files.json');
    const res = await fetch('/api/files');
    return await res.json();
}

async function fetchMapData(filename) {
    if (STATIC_BASE) return await _staticJson('api/mapdata.json');
    const geoRes = await fetch(`/api/mapdata?file=${encodeURIComponent(filename)}`);
    if (!geoRes.ok) throw new Error(await geoRes.text());
    return await geoRes.json();
}

async function fetchAnnotations(filename) {
    if (STATIC_BASE) {
        try { return await _staticJson('api/annotations.json'); }
        catch (_) { return { annotations: [] }; }
    }
    const annRes = await fetch(`/api/annotations?file=${encodeURIComponent(filename)}`);
    if (!annRes.ok) return { annotations: [] };
    return await annRes.json();
}

async function saveAnnotation(filename, annId, geometry) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    await fetch(`/api/annotations/${annId}?file=${encodeURIComponent(filename)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ geometry: geometry }),
    });
}

async function deleteAnnotationApi(filename, annId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/annotations/${annId}?file=${encodeURIComponent(filename)}`, {
        method: 'DELETE'
    });
}

async function createAnnotationApi(filename, type, geometry, properties) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    const res = await fetch(`/api/annotations?file=${encodeURIComponent(filename)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, geometry, properties })
    });
    return await res.json();
}

async function updateAnnotationApi(filename, annId, geometry, type, properties) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    const res = await fetch(`/api/annotations/${annId}?file=${encodeURIComponent(filename)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ geometry, type, properties })
    });
    return await res.json();
}

async function fetchWayNodes(filename, wayId) {
    if (STATIC_BASE) return await _staticJson(`api/way_nodes/${String(wayId).replace(':', '_')}.json`);
    const res = await fetch(`/api/way_nodes?file=${encodeURIComponent(filename)}&way_id=${wayId}`);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

async function updateWayTagsApi(filename, wayId, tags, cat, lbl) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/ways/${wayId}/tags?file=${encodeURIComponent(filename)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags, category: cat, label: lbl })
    });
}

async function deleteWayTagsApi(filename, wayId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/ways/${wayId}/tags?file=${encodeURIComponent(filename)}`, {
        method: 'DELETE'
    });
}

async function deleteWayApi(filename, wayId, cat, label) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/ways/${wayId}?file=${encodeURIComponent(filename)}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: cat, label })
    });
}

async function deleteNodeApi(filename, wayId, nodeId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    const res = await fetch(`/api/way_node?file=${encodeURIComponent(filename)}&way_id=${wayId}&node_id=${nodeId}`, {
        method: 'DELETE'
    });
    return res;
}

async function addWayNodeApi(filename, wayId, afterNodeId, lat, lon) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/way_node?file=${encodeURIComponent(filename)}&way_id=${wayId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ after_node_id: afterNodeId, lat, lon }),
    });
}

async function splitWayApi(filename, wayId, nodeId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    const res = await fetch(`/api/ways/split?file=${encodeURIComponent(filename)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ way_id: wayId, node_id: nodeId })
    });
    return res;
}

async function undoWaySplitApi(filename, wayId, nodeId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    const res = await fetch(`/api/ways/split?file=${encodeURIComponent(filename)}&way_id=${wayId}&node_id=${nodeId}`, {
        method: 'DELETE'
    });
    return res;
}
async function hideWayApi(filename, wayId, cat, label) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/ways/${wayId}/hide?file=${encodeURIComponent(filename)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: cat, label })
    });
}

async function showWayApi(filename, wayId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/ways/${wayId}/show?file=${encodeURIComponent(filename)}`, {
        method: 'PUT'
    });
}

async function restoreWayApi(filename, wayId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/ways/${wayId}/restore?file=${encodeURIComponent(filename)}`, {
        method: 'PUT'
    });
}

async function restoreNodeApi(filename, wayId, nodeId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/way_node/restore?file=${encodeURIComponent(filename)}&way_id=${wayId}&node_id=${nodeId}`, {
        method: 'PUT'
    });
}

async function fetchWayApi(filename, wayId) {
    if (STATIC_BASE) _staticReadOnly('Way lookup');
    return await fetch(`/api/ways/${wayId}?file=${encodeURIComponent(filename)}`);
}

async function fetchWaySegmentsApi(filename, wayId) {
    if (STATIC_BASE) _staticReadOnly('Way lookup');
    const res = await fetch(`/api/ways/${wayId}/segments?file=${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

async function moveWayNodesApi(filename, wayId, nodes, category, label) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/way_nodes/move?file=${encodeURIComponent(filename)}&way_id=${wayId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, category: category ?? 'unknown', label: label ?? '' }),
    });
}

async function undoWayNodeMovesApi(filename, wayId) {
    if (STATIC_BASE) _staticReadOnly('Editing');
    return await fetch(`/api/way_nodes/move?file=${encodeURIComponent(filename)}&way_id=${wayId}`, {
        method: 'DELETE',
    });
}

function formatFetchProgress(task) {
    const detail = task.detail || (task.status === 'parsing' ? 'Parsing OSM data…' : 'Fetching OSM data…');
    return `${detail} (${task.elapsedSeconds}s)`;
}

async function fetchAreaApi(params, onProgress) {
    if (STATIC_BASE) _staticReadOnly('OSM fetching');
    const res = await fetch('/api/fetch_area', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(await res.text());
    const { task_id } = await res.json();
    const startedAt = Date.now();
    while (true) {
        await new Promise(r => setTimeout(r, 1500));
        const poll = await fetch(`/api/fetch_area/${task_id}`);
        if (!poll.ok) throw new Error(await poll.text());
        const task = await poll.json();
        if (task.status === 'done') return task.result;
        if (task.status === 'failed') throw new Error(task.error || 'Fetch failed');
        onProgress?.({ ...task, elapsedSeconds: Math.round((Date.now() - startedAt) / 1000) });
    }
}

async function uploadGpxApi(formData) {
    if (STATIC_BASE) _staticReadOnly('Uploading');
    const res = await fetch('/api/upload_gpx', {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

async function uploadMapdataApi(formData) {
    if (STATIC_BASE) _staticReadOnly('Uploading');
    const res = await fetch('/api/upload_mapdata', {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}
