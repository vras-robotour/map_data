// ── State ────────────────────────────────────────────────────────────────────
let currentFile = null;
let currentMode = 'view';
let annotations = [];
let pendingBbox  = null;
let nodeLayer         = null;   // L.LayerGroup of node circle markers
let nodeCount         = 0;      // actual OSM node count from last fetch
let currentNodes      = [];     // fetched OSM node objects for the active way
let nodeMarkers       = [];     // L.CircleMarker refs, same order as currentNodes
let selectedNodeIndex = -1;     // currently selected node index
let currentClickedFeature = null; // last GeoJSON feature clicked in view mode
let currentClickedLayer  = null;  // Leaflet layer for the selected OSM feature
let deletedWays  = [];            // [{id, category, label}, ...] from annotations store
let deletedNodes = [];            // [{way_id, node_id}, ...]        from annotations store
let tagOverrides = [];            // [{id, category, label, tags}, ...] from annotations store
let hiddenWays   = [];            // [{id, category, label}, ...] from annotations store
let hiddenWayIds = new Set();     // Set<number> for O(1) lookup
let editSelectedLayer = null;   // annotation layer selected in edit mode
let pathLineDraw  = null;       // L.Draw.Polyline handler for path mode
let pendingAnnGeom = null;      // geometry waiting for the details modal
let editingAnnId   = null;      // annotation ID being edited in the modal
let editingWayId   = null;      // OSM way ID being edited in the way-edit modal

// ── Layers ───────────────────────────────────────────────────────────────────
const geoLayers = { road: null, footway: null, barrier: null, waypoint: null, crossroad: null };

// subtypeLayers[cat][subtype] = array of individual L.Path layers inside geoLayers[cat]
const subtypeLayers  = { road: {}, footway: {}, barrier: {}, crossroad: {} };
// subtypeFilters[cat][subtype] = boolean (true = visible)
const subtypeFilters = { road: {}, footway: {}, barrier: {}, crossroad: {} };

// drawnItems holds all annotation polygons so Leaflet.draw can edit/delete them
const drawnItems = new L.FeatureGroup(); // Will be added to map in map_setup.js

// ── Draw controls / handlers ─────────────────────────────────────────────────
let drawControl   = null;
let deleteHandler = null;
let fetchRectDraw = null;

const STYLES = {
  road:       { color: '#555', weight: 0.8, fillColor: '#333', fillOpacity: 0.5 },
  footway:    { color: '#b89900', weight: 0.5, fillColor: '#FFD700', fillOpacity: 0.35 },
  barrier:    { color: '#8b0000', weight: 1,   fillColor: '#BF0009', fillOpacity: 0.35 },
  crossroad:  { color: '#7e22ce', weight: 1,   fillColor: '#9333ea', fillOpacity: 0.85 },
  annotation: { color: '#cc5500', weight: 2,   fillColor: '#ff8c00', fillOpacity: 0.35 },
  path:       { color: '#22c55e', weight: 3,   fillOpacity: 0, opacity: 0.7 },
};

const HIGHLIGHT_STYLES = {
  road:       { color: '#fff', weight: 3,   fillColor: '#555', fillOpacity: 0.8 },
  footway:    { color: '#fff', weight: 2.5, fillColor: '#FFD700', fillOpacity: 0.7 },
  barrier:    { color: '#ff2020', weight: 3,   fillColor: '#cc0000', fillOpacity: 0.7 },
  crossroad:  { color: '#f0f',   weight: 2.5, fillColor: '#a855f7', fillOpacity: 0.95 },
  annotation: { color: '#fff', weight: 3.5, fillColor: '#ff8c00', fillOpacity: 0.7 },
  path:       { color: '#fff', weight: 4.5, fillOpacity: 0, opacity: 1.0 },
};
