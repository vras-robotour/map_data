// ── API ──────────────────────────────────────────────────────────────────────
async function fetchFileList() {
  const res = await fetch('/api/files');
  return await res.json();
}

async function fetchMapData(filename) {
  const geoRes = await fetch(`/api/mapdata?file=${encodeURIComponent(filename)}`);
  if (!geoRes.ok) throw new Error(await geoRes.text());
  return await geoRes.json();
}

async function fetchAnnotations(filename) {
  const annRes = await fetch(`/api/annotations?file=${encodeURIComponent(filename)}`);
  if (!annRes.ok) return { annotations: [] };
  return await annRes.json();
}

async function saveAnnotation(filename, annId, geometry) {
  await fetch(`/api/annotations/${annId}?file=${encodeURIComponent(filename)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ geometry: geometry }),
  });
}

async function deleteAnnotationApi(filename, annId) {
  return await fetch(`/api/annotations/${annId}?file=${encodeURIComponent(filename)}`, {
    method: 'DELETE'
  });
}

async function createAnnotationApi(filename, type, geometry, properties) {
  const res = await fetch(`/api/annotations?file=${encodeURIComponent(filename)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type, geometry, properties })
  });
  return await res.json();
}

async function updateAnnotationApi(filename, annId, geometry, type, properties) {
  const res = await fetch(`/api/annotations/${annId}?file=${encodeURIComponent(filename)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ geometry, type, properties })
  });
  return await res.json();
}

async function fetchWayNodes(filename, wayId) {
  const res = await fetch(`/api/way_nodes?file=${encodeURIComponent(filename)}&way_id=${wayId}`);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function updateWayTagsApi(filename, wayId, tags, cat, lbl) {
  return await fetch(`/api/ways/${wayId}/tags?file=${encodeURIComponent(filename)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tags, category: cat, label: lbl })
  });
}

async function deleteWayTagsApi(filename, wayId) {
  return await fetch(`/api/ways/${wayId}/tags?file=${encodeURIComponent(filename)}`, {
    method: 'DELETE'
  });
}

async function deleteWayApi(filename, wayId, cat, label) {
  return await fetch(`/api/ways/${wayId}?file=${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ category: cat, label })
  });
}

async function deleteNodeApi(filename, wayId, nodeId) {
  return await fetch(`/api/way_node?file=${encodeURIComponent(filename)}&way_id=${wayId}&node_id=${nodeId}`, {
    method: 'DELETE'
  });
}

async function hideWayApi(filename, wayId, cat, label) {
  return await fetch(`/api/ways/${wayId}/hide?file=${encodeURIComponent(filename)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ category: cat, label })
  });
}

async function showWayApi(filename, wayId) {
  return await fetch(`/api/ways/${wayId}/show?file=${encodeURIComponent(filename)}`, {
    method: 'PUT'
  });
}

async function restoreWayApi(filename, wayId) {
  return await fetch(`/api/ways/${wayId}/restore?file=${encodeURIComponent(filename)}`, {
    method: 'PUT'
  });
}

async function restoreNodeApi(filename, wayId, nodeId) {
  return await fetch(`/api/way_node/restore?file=${encodeURIComponent(filename)}&way_id=${wayId}&node_id=${nodeId}`, {
    method: 'PUT'
  });
}

async function fetchWayApi(filename, wayId) {
  return await fetch(`/api/ways/${wayId}?file=${encodeURIComponent(filename)}`);
}

async function fetchAreaApi(params) {
  const res = await fetch('/api/fetch_area', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
