// ── API ──────────────────────────────────────────────────────────────────────

// When window.__mapdataStaticBase is set the page is the read-only static demo
// (GitHub Pages): a few GETs map to pre-baked JSON files, other calls are refused.
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

// Fetch wrapper for API calls; FormData bodies are sent as-is, others as JSON.
// A `readOnlyMsg` marks the call as refused in the static demo.
async function api(method, path, { query, body, readOnlyMsg, signal } = {}) {
    if (STATIC_BASE && readOnlyMsg) _staticReadOnly(readOnlyMsg);
    const qs = query ? '?' + new URLSearchParams(query) : '';
    const opts = { method, signal };
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
    if (!res.ok) throw new Error(await errorText(res));
    return await res.json();
}

async function fetchMapData(filename) {
    if (STATIC_BASE) return await _staticJson('api/mapdata.json');
    const res = await api('GET', '/api/mapdata', { query: { file: filename } });
    if (!res.ok) throw new Error(await errorText(res));
    return await res.json();
}

async function fetchAnnotations(filename) {
    if (STATIC_BASE) {
        try { return await _staticJson('api/annotations.json'); }
        catch (_) { return { annotations: [] }; }
    }
    const res = await api('GET', '/api/annotations', { query: { file: filename } });
    if (!res.ok) {
        // 409: the store was unreadable and moved aside; the map still loads without it.
        setStatus(`Annotations not loaded: ${await errorText(res)}`, 'text-danger');
        return { annotations: [] };
    }
    return await res.json();
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
    if (!res.ok) throw new Error(await errorText(res));
    return await res.json();
}

async function fetchWayApi(filename, wayId) {
    return await api('GET', `/api/ways/${wayId}`, { query: { file: filename }, readOnlyMsg: 'Way lookup' });
}

async function fetchWaySegmentsApi(filename, wayId) {
    const res = await api('GET', `/api/ways/${wayId}/segments`, { query: { file: filename }, readOnlyMsg: 'Way lookup' });
    if (!res.ok) throw new Error(await errorText(res));
    return await res.json();
}

async function moveWayNodesApi(filename, wayId, nodes, category, label) {
    return await api('PUT', '/api/way_nodes/move', {
        query: { file: filename, way_id: wayId },
        body: { nodes, category: category ?? 'unknown', label: label ?? '' },
        readOnlyMsg: 'Editing',
    });
}

function formatFetchProgress(task) {
    const detail = task.detail || (task.status === 'parsing' ? 'Parsing OSM data…' : 'Fetching OSM data…');
    return `${detail} (${task.elapsedSeconds}s)`;
}

async function fetchAreaApi(params, onProgress) {
    const res = await api('POST', '/api/fetch_area', { body: params, readOnlyMsg: 'OSM fetching' });
    if (!res.ok) throw new Error(await errorText(res));
    const { task_id } = await res.json();
    const startedAt = Date.now();
    let pollFailures = 0;
    while (true) {
        await new Promise(r => setTimeout(r, 1500));
        let poll;
        try {
            poll = await api('GET', `/api/fetch_area/${task_id}`);
        } catch (err) {
            pollFailures++;
            if (pollFailures >= 3) throw err;
            continue;
        }
        if (!poll.ok) {
            pollFailures++;
            if (pollFailures >= 3) throw new Error(await errorText(poll));
            continue;
        }
        pollFailures = 0;
        const task = await poll.json();
        if (task.status === 'done') return task.result;
        if (task.status === 'failed') throw new Error(task.error || 'Fetch failed');
        onProgress?.({ ...task, elapsedSeconds: Math.round((Date.now() - startedAt) / 1000) });
    }
}

async function uploadGpxApi(formData) {
    const res = await api('POST', '/api/upload_gpx', { body: formData, readOnlyMsg: 'Uploading' });
    if (!res.ok) throw new Error(await errorText(res));
    return await res.json();
}

async function uploadMapdataApi(formData) {
    const res = await api('POST', '/api/upload_mapdata', { body: formData, readOnlyMsg: 'Uploading' });
    if (!res.ok) throw new Error(await errorText(res));
    return await res.json();
}
