const trackerMode = (() => {
    let socket = null;
    let robotMarker = null;
    let robotPathLayer = null;
    let sequenceLayer = null;
    let windowLayer = null;
    let roadPathLayer = null;
    let trailLayer = null;
    let intersectionsLayer = null;
    let _intersectionsKey = null;
    let activeEnterCircle = null;
    let activeExitCircle = null;
    let goalMarker = null;
    let enabled = false;
    let _lastPos = null; // last known robot position, for the planner's "Start at robot"

    // Cached DOM/element references — resolved lazily on first use
    let _robotCb = null;
    let _followCb = null;
    let _robotSvg = null;

    const ROBOT_ICON_HTML = `
    <div class="robot-marker-container">
      <svg viewBox="0 0 24 24" width="30" height="30" style="filter: drop-shadow(0 0 2px rgba(0,0,0,0.5));">
        <path d="M12,2L4.5,20.29L5.21,21L12,18L18.79,21L19.5,20.29L12,2Z" fill="#00ff00" stroke="#000" stroke-width="1"/>
      </svg>
    </div>
  `;

    function initSocket() {
        if (window.__mapdataStaticBase) return;
        if (socket) return;

        socket = io();

        socket.on('connect', () => {
            console.log('Tracker: Connected to server');
        });

        socket.on('telemetry', (data) => {
            // Update map (robot icon) regardless of whether 'Tracker' mode is active
            // This allows seeing the robot in Viewer and Planner modes
            updateMap(data);

            // UI status only updates if Tracker mode is actually enabled
            if (enabled) {
                updateUI(data);
            }
        });

        socket.on('disconnect', () => {
            console.log('Tracker: Disconnected from server');
        });
    }

    function updateMap(data) {
        if (!map) return;

        // The position is published before the layer check: the planner's "Start at robot"
        // needs it even when the Robot layer is switched off.
        const pos = data.position.ekf.lat ? data.position.ekf : data.position.gps;
        const havePos = !!(pos && pos.lat && pos.lon);
        if (havePos) {
            _lastPos = { lat: pos.lat, lon: pos.lon };
            document.dispatchEvent(new CustomEvent('robot-position', { detail: _lastPos }));
        }

        // Check robot layer visibility — cache the checkbox element
        if (!_robotCb) _robotCb = document.querySelector('[data-layer="robot"]');
        if (_robotCb && !_robotCb.checked) {
            hideRobot();
            return;
        }

        if (havePos) {
            const latlng = [pos.lat, pos.lon];
            const heading = pos.heading || 0;

            if (!robotMarker) {
                const icon = L.divIcon({
                    className: 'robot-marker-div',
                    html: ROBOT_ICON_HTML,
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                });
                robotMarker = L.marker(latlng, { icon: icon, zIndexOffset: 1000 }).addTo(map);

                if (enabled) {
                    map.setView(latlng, 18);
                }
            } else {
                robotMarker.setLatLng(latlng);

                // Cache the follow checkbox
                if (!_followCb) _followCb = document.getElementById('tracker-follow-robot');
                if (enabled && _followCb && _followCb.checked) {
                    map.panTo(latlng);
                }
            }

            // Rotate marker — resolve the SVG element once and reuse it
            if (!_robotSvg) {
                const el = robotMarker.getElement();
                if (el) _robotSvg = el.querySelector('svg');
            }
            if (_robotSvg) {
                _robotSvg.style.transform = `rotate(${heading}deg)`;
            }
        }

        // Planned path; a key of length and end points missed replans that keep them
        robotPathLayer = updatePolyline(robotPathLayer, data.mission && data.mission.waypoints,
            { color: '#00ff00', weight: 3, opacity: 0.6, dashArray: '5, 10' });

        // Waypoint sequence handed to the commander (blue) and the visual road path (cyan)
        sequenceLayer = updatePolyline(sequenceLayer, data.mission && data.mission.sequence,
            { color: '#3b82f6', weight: 3, opacity: 0.8, dashArray: '2, 6' });
        windowLayer = updatePolyline(windowLayer, data.mission && data.mission.sequence_window,
            { color: '#60a5fa', weight: 5, opacity: 0.9 });
        roadPathLayer = updatePolyline(roadPathLayer, data.mission && data.mission.road_path,
            { color: '#22d3ee', weight: 4, opacity: 0.9 });
        trailLayer = updatePolyline(trailLayer, data.position && data.position.trail,
            { color: '#4ade80', weight: 2, opacity: 0.5 });

        // OSM intersections as seen by road_follower (magenta rings)
        const inter = data.mission && data.mission.intersections;
        if (inter && inter.length) {
            const key = `${inter.length}:${inter[0].lat},${inter[0].lon}`;
            if (key !== _intersectionsKey) {
                if (intersectionsLayer) map.removeLayer(intersectionsLayer);
                intersectionsLayer = L.layerGroup(inter.map(p => L.circleMarker([p.lat, p.lon], {
                    radius: 4, color: '#e879f9', weight: 1.5, fill: false, opacity: 0.8
                }))).addTo(map);
                _intersectionsKey = key;
            }
        } else if (intersectionsLayer) {
            map.removeLayer(intersectionsLayer);
            intersectionsLayer = null;
            _intersectionsKey = null;
        }

        // Active intersection with the enter (red) / exit (yellow) hysteresis radii in metres
        const act = data.position && data.position.active_intersection;
        const thr = (data.position && data.position.intersection_thresholds) || { enter: 5, exit: 6 };
        if (act && act.lat) {
            const ll = [act.lat, act.lon];
            if (!activeEnterCircle) {
                activeExitCircle = L.circle(ll, { radius: thr.exit, color: '#facc15', weight: 1.5, fill: false, dashArray: '4, 4' }).addTo(map);
                activeEnterCircle = L.circle(ll, { radius: thr.enter, color: '#ef4444', weight: 2, fillColor: '#ef4444', fillOpacity: 0.1 }).addTo(map);
            } else {
                activeEnterCircle.setLatLng(ll); activeExitCircle.setLatLng(ll);
                activeEnterCircle.setRadius(thr.enter); activeExitCircle.setRadius(thr.exit);
            }
        } else if (activeEnterCircle) {
            map.removeLayer(activeEnterCircle); map.removeLayer(activeExitCircle);
            activeEnterCircle = activeExitCircle = null;
        }

        // Grey out the marker when the fix is stale
        if (_robotSvg) _robotSvg.style.opacity = (data.position && data.position.stale) ? '0.35' : '1';

        // Current navigation goal
        const goal = data.position && data.position.goal;
        if (goal && goal.lat) {
            if (!goalMarker) {
                goalMarker = L.circleMarker([goal.lat, goal.lon], {
                    radius: 7, color: '#f97316', weight: 2, fillColor: '#fb923c', fillOpacity: 0.8
                }).addTo(map).bindTooltip('Goal');
            } else {
                goalMarker.setLatLng([goal.lat, goal.lon]);
            }
        } else if (goalMarker) {
            map.removeLayer(goalMarker);
            goalMarker = null;
        }
    }

    function updatePolyline(layer, pts, style) {
        if (pts && pts.length > 1) {
            const latlngs = pts.map(p => [p.lat, p.lon]);
            if (!layer) return L.polyline(latlngs, style).addTo(map);
            layer.setLatLngs(latlngs);
            return layer;
        }
        if (layer) map.removeLayer(layer);
        return null;
    }

    function hideRobot() {
        if (robotMarker && map.hasLayer(robotMarker)) map.removeLayer(robotMarker);
        [robotPathLayer, sequenceLayer, windowLayer, roadPathLayer, trailLayer, intersectionsLayer,
         activeEnterCircle, activeExitCircle, goalMarker].forEach(l => {
            if (l && map.hasLayer(l)) map.removeLayer(l);
        });
        _robotSvg = null; // Leaflet recreates the element on the next addTo()
    }

    function showRobot() {
        initSocket();
        [robotMarker, robotPathLayer, sequenceLayer, windowLayer, roadPathLayer, trailLayer,
         intersectionsLayer, activeEnterCircle, activeExitCircle, goalMarker].forEach(l => {
            if (l && !map.hasLayer(l)) l.addTo(map);
        });
    }

    // Inline SVG swatch matching a Leaflet style (line / circle / robot marker)
    function swatch(kind, color, weight = 2, dash = '', opacity = 1, fill = 'none', fillOpacity = 0.8) {
        const common = `stroke="${color}" stroke-width="${weight}" stroke-opacity="${opacity}"` +
            (dash ? ` stroke-dasharray="${dash}"` : '');
        let body;
        if (kind === 'line') body = `<line x1="1" y1="7" x2="27" y2="7" ${common}/>`;
        else if (kind === 'circle') body = `<circle cx="14" cy="7" r="5" ${common} fill="${fill}" fill-opacity="${fillOpacity}"/>`;
        else body = `<path d="M14,1L8,13L14,10.5L20,13Z" fill="${color}" stroke="#000" stroke-width="0.8"/>`;
        return `<svg width="28" height="14" viewBox="0 0 28 14" style="vertical-align:middle;margin-right:4px">${body}</svg>`;
    }

    // The tracker status panel's markup lives in index.html (#tracker-status-content);
    // this is just a tiny lookup, not a cache — the panel updates only once per
    // telemetry frame, so there's no need to precompute/store the ~25 element refs.
    const $ = id => document.getElementById(id);

    // The legend's swatch icons are static (never change), so render them once
    // up front instead of on every telemetry update.
    function _renderLegend() {
        [
            ['tsi-leg-robot', swatch('marker', '#00ff00')],
            ['tsi-leg-trail', swatch('line', '#4ade80', 2, '', 0.6)],
            ['tsi-leg-path', swatch('line', '#00ff00', 3, '5, 4')],
            ['tsi-leg-sequence', swatch('line', '#3b82f6', 3, '2, 4')],
            ['tsi-leg-window', swatch('line', '#60a5fa', 5)],
            ['tsi-leg-road', swatch('line', '#22d3ee', 4)],
            ['tsi-leg-goal', swatch('circle', '#f97316', 2, '', 1, '#fb923c')],
            ['tsi-leg-intersections', swatch('circle', '#e879f9', 1.5)],
            ['tsi-leg-active', swatch('circle', '#ef4444', 2, '', 1, '#ef4444', 0.15)],
        ].forEach(([id, icon]) => $(id)?.insertAdjacentHTML('afterbegin', icon));
        const exitIcon = $('tsi-leg-active-exit-icon');
        if (exitIcon) exitIcon.innerHTML = swatch('circle', '#facc15', 1.5, '3, 3');
    }
    _renderLegend();

    function updateUI(data) {
        if (!$('tracker-status-content')) return;

        const s = data.status || {};
        const b = s.battery || {};
        const feat = data.enabled_features || {};

        // Hardware section
        const hasBattery = feat.battery !== false;
        const hasMotors = feat.motors !== false;
        const hasTemp = feat.temp !== false;
        const hasMotorError = feat.motor_error !== false;
        const hasEstop = feat.estop === true;

        $('tsi-row-battery').hidden = !hasBattery;
        $('tsi-row-estop').hidden = !hasEstop;
        $('tsi-row-motors').hidden = !hasMotors;
        $('tsi-row-temp').hidden = !hasTemp;
        $('tsi-section-hardware').hidden = !(hasBattery || hasMotors || hasTemp || hasMotorError || hasEstop);

        if (hasBattery) {
            const battV = b.voltage != null ? b.voltage : '—';
            const battA = b.current != null ? b.current : '—';
            const pct = b.percentage != null ? ` (${b.percentage}%)` : '';
            const battery = $('tsi-battery');
            battery.textContent = `${battV} V / ${battA} A${pct}`;
            const low = b.low_voltage != null ? b.low_voltage : 22.0;
            battery.className = (b.voltage && b.voltage < low) ? 'text-danger' : 'text-light';
        }

        if (hasEstop) {
            const estop = $('tsi-estop');
            if (s.estop_active == null) {
                estop.textContent = '—';
                estop.className = 'text-secondary';
            } else {
                estop.textContent = s.estop_active ? 'ACTIVE' : 'released';
                estop.className = s.estop_active ? 'text-danger fw-bold' : 'text-success';
            }
        }

        if (hasMotors) {
            const motors = $('tsi-motors');
            motors.textContent = s.motors_enabled ? 'ENABLED' : 'DISABLED';
            motors.className = s.motors_enabled ? 'text-success' : 'text-danger';
        }

        const motorError = $('tsi-motor-error');
        if (hasMotorError && s.motor_error) {
            motorError.textContent = `Error: 0x${s.motor_error.toString(16)}`;
            motorError.hidden = false;
        } else {
            motorError.hidden = true;
        }

        if (hasTemp) {
            const temp = $('tsi-temp');
            if (s.temp_max) {
                temp.textContent = `${s.temp_max.value} °C (${s.temp_max.name})`;
                temp.title = Object.entries(s.temperatures || {}).map(([k, v]) => `${k}: ${v} °C`).join('\n');
            } else {
                temp.textContent = s.teensy_temp != null ? `${s.teensy_temp} °C` : '—';
            }
        }

        // Localization section
        const hasGps = feat.gps_fix !== false || feat.gps_ekf !== false;
        const hasSpeed = feat.speed !== false;
        const hasSpeedLimit = feat.speed_limit !== false;

        $('tsi-row-gps').hidden = feat.gps_fix === false;
        $('tsi-row-speed').hidden = !hasSpeed;
        $('tsi-row-speed-limit').hidden = !hasSpeedLimit;
        $('tsi-section-localization').hidden = !(hasGps || hasSpeed || hasSpeedLimit);

        if (feat.gps_fix !== false) {
            let fixStr = 'No Fix', fixClass = 'text-danger';
            if (s.gps_fix === 0) { fixStr = 'Fix'; fixClass = 'text-success'; }
            else if (s.gps_fix === 1) { fixStr = 'Float'; fixClass = 'text-warning'; }
            else if (s.gps_fix === 2) { fixStr = 'Fixed'; fixClass = 'text-info'; }
            const gpsFix = $('tsi-gps-fix');
            gpsFix.textContent = fixStr;
            gpsFix.className = fixClass;
        }

        const pos = data.position || {};
        $('tsi-row-fix-age').hidden = !hasGps;
        if (hasGps) {
            const fixAge = $('tsi-fix-age');
            if (pos.fix_age == null) {
                fixAge.textContent = 'no fix yet'; fixAge.className = 'text-secondary';
            } else {
                fixAge.textContent = pos.stale ? `${pos.fix_age} s — STALE` : `${pos.fix_age} s`;
                fixAge.className = pos.stale ? 'text-danger fw-bold' : 'text-light';
            }
        }

        if (hasSpeed) {
            $('tsi-speed').textContent = s.speed != null ? `${s.speed} m/s` : '—';
        }

        if (hasSpeedLimit) {
            $('tsi-speed-limit').textContent = s.speed_limit
                ? `${s.speed_limit.value} ${s.speed_limit.percentage ? '%' : 'm/s'}`
                : '—';
        }

        // Navigation section
        const hasNavState = feat.nav_state !== false;
        const hasCollision = feat.collision !== false;
        const hasRecovery = feat.recovery !== false;
        const hasTeleop = feat.teleop !== false;

        const hasFollower = feat.follower_state === true;
        const hasDiag = feat.diagnostics === true;

        $('tsi-row-nav-state').hidden = !hasNavState;
        $('tsi-row-follower').hidden = !hasFollower;
        $('tsi-row-diag').hidden = !hasDiag;
        $('tsi-section-navigation').hidden = !(hasNavState || hasCollision || hasRecovery || hasTeleop || hasFollower || hasDiag);

        if (hasNavState) {
            const navState = $('tsi-nav-state');
            navState.textContent = s.nav_state || 'IDLE';
            navState.className = (s.nav_state === 'STUCK') ? 'text-danger fw-bold' : 'text-info';
        }
        if (hasFollower) {
            const followerState = $('tsi-follower-state');
            followerState.textContent = s.follower_state || '—';
            followerState.className = (s.follower_state || '').startsWith('GPS') ? 'text-warning' : 'text-info';
        }
        if (hasDiag) {
            const d = s.diagnostics;
            const diag = $('tsi-diag');
            if (!d) {
                diag.textContent = '—'; diag.className = 'text-secondary';
            } else if (d.errors === 0 && d.warnings === 0) {
                diag.textContent = 'OK'; diag.className = 'text-success';
            } else {
                diag.textContent = `${d.errors} err / ${d.warnings} warn` + (d.worst ? ` — ${d.worst.name}` : '');
                diag.className = d.errors ? 'text-danger' : 'text-warning';
                diag.title = d.worst ? `${d.worst.name}: ${d.worst.message}` : '';
            }
        }

        const collision = $('tsi-collision');
        if (hasCollision) {
            const showCollision = s.collision_action && s.collision_action !== 'PASSTHROUGH';
            collision.hidden = !showCollision;
            if (showCollision) $('tsi-collision-val').textContent = s.collision_action;
        } else {
            collision.hidden = true;
        }

        $('tsi-recovery').hidden = !(hasRecovery && s.recovery_active);
        $('tsi-teleop').hidden = !(hasTeleop && s.teleop_active);

        // Legend: only the layers that can actually appear
        $('tsi-leg-trail').hidden = !hasGps;
        $('tsi-leg-path').hidden = !(feat.path !== false || feat.actions !== false);
        $('tsi-leg-sequence').hidden = feat.sequence !== true;
        $('tsi-leg-window').hidden = feat.sequence_window !== true;
        $('tsi-leg-road').hidden = feat.road_path !== true;
        $('tsi-leg-goal').hidden = feat.goal !== true;
        $('tsi-leg-intersections').hidden = feat.intersections !== true;
        $('tsi-leg-active').hidden = feat.active_intersection !== true;

        // Spoken messages, newest first
        const speechBox = $('tsi-speech-box');
        const speechLog = s.speech_log || [];
        if (feat.speech !== false && speechLog.length) {
            const log = $('tsi-speech-log');
            log.textContent = '';
            speechLog.forEach((entry, i) => {
                const levelClass = entry.level === 'error' ? 'text-danger'
                    : (entry.level === 'warn' ? 'text-warning' : 'text-info');
                const row = document.createElement('div');
                // The newest message stays readable; the older ones fade into the background
                row.className = `${levelClass}${i ? ' opacity-50' : ''}`;
                row.textContent = `${entry.level}: ${entry.text}`;
                log.appendChild(row);
            });
            speechBox.hidden = false;
        } else {
            speechBox.hidden = true;
        }

        // Hide Robot layer checkbox if robot positioning is disabled
        const robotCb = document.querySelector('[data-layer="robot"]');
        const row = robotCb?.closest('.layer-row');
        if (feat.gps_fix === false && feat.gps_ekf === false) {
            if (row) row.hidden = true;
            hideRobot();
        } else if (row) {
            row.hidden = false;
        }
    }


    // ── Topics dialog: switch the tracker's topics live and save them to its config file ──
    let _settings = null; // last GET /api/tracker/settings response

    // Flask's abort() answers with an HTML page; pull the message out of it.
    async function errorText(res) {
        const doc = new DOMParser().parseFromString(await res.text(), 'text/html');
        return (doc.querySelector('p') || doc.body).textContent.trim();
    }

    function topicsError(msg) {
        const el = $('tracker-topics-error');
        el.textContent = msg || '';
        el.hidden = !msg;
    }

    const isTopic = s => s.name.endsWith('_topic');
    const listId = type => 'tts-list-' + type.replace(/[^A-Za-z0-9]/g, '_');

    function expectedType(s) {
        return s.name === 'heading_topic' ? _settings.heading_types[$('tts-heading_type').value] : s.msg_type;
    }

    // Dot left of a topic: green = published with the expected type, yellow = another type,
    // grey = nobody publishes it (yet), hollow = disabled
    function updateDot(s) {
        const topic = $(`tts-${s.name}`).value.trim();
        const type = expectedType(s);
        const dot = $(`tts-dot-${s.name}`);
        $(`tts-${s.name}`).setAttribute('list', listId(type));
        $(`tts-type-${s.name}`).textContent = type.split('/').pop();
        let color = 'transparent', title = 'disabled';
        if (topic) {
            const types = _settings.topics[topic] || _settings.topics[`/${topic}`];
            if (!types) { color = '#6b7280'; title = 'not published'; }
            else if (types.includes(type)) { color = '#22c55e'; title = `published as ${type}`; }
            else { color = '#facc15'; title = `published as ${types.join(', ')}; the tracker expects ${type}`; }
        }
        dot.style.background = color;
        dot.title = title;
    }

    function renderTopicsForm() {
        // One suggestion list per message type, from the topics on the ROS graph
        const byType = {};
        for (const [topic, types] of Object.entries(_settings.topics)) {
            for (const t of types) (byType[t] ||= []).push(topic);
        }
        const types = new Set([..._settings.settings.map(s => s.msg_type), ...Object.values(_settings.heading_types)]);
        $('tracker-topics-lists').replaceChildren(...[...types].filter(Boolean).map(type => {
            const dl = document.createElement('datalist');
            dl.id = listId(type);
            dl.append(...(byType[type] || []).sort().map(topic => new Option(topic, topic)));
            return dl;
        }));

        const form = $('tracker-topics-form');
        form.replaceChildren();
        let section = null;
        for (const s of _settings.settings) {
            if (s.section !== section) {
                section = s.section;
                const title = document.createElement('div');
                title.className = 'panel-title mt-2 mb-1';
                title.style.fontSize = '0.6rem';
                title.textContent = section.toUpperCase();
                form.appendChild(title);
            }
            const row = document.createElement('div');
            row.className = 'd-flex align-items-center gap-2 mb-1';

            const label = document.createElement('label');
            label.htmlFor = `tts-${s.name}`;
            label.className = 'text-truncate';
            label.style.flex = '0 0 45%';
            label.title = s.description + (s.default ? ` (default: ${s.default})` : '');
            const code = document.createElement('code');
            code.textContent = s.name;
            const typeHint = document.createElement('small');
            typeHint.className = 'text-secondary ms-1';
            typeHint.id = `tts-type-${s.name}`;
            label.append(code, typeHint);

            const dot = document.createElement('span');
            dot.id = `tts-dot-${s.name}`;
            dot.style.cssText = 'flex:0 0 10px; height:10px; border-radius:50%;' +
                (isTopic(s) ? ' border:1px solid #6b7280;' : '');

            let field;
            if (s.choices.length) {
                field = document.createElement('select');
                field.className = 'form-select form-select-sm bg-dark text-light border-secondary';
                field.append(...s.choices.map(c => new Option(c, c)));
                field.addEventListener('change', () => updateDot(_settings.settings.find(x => x.name === 'heading_topic')));
            } else {
                field = document.createElement('input');
                field.type = 'text';
                field.spellcheck = false;
                field.className = 'form-control form-control-sm bg-dark text-light border-secondary font-monospace';
                if (isTopic(s)) {
                    field.placeholder = 'disabled';
                    field.addEventListener('input', () => updateDot(s));
                }
            }
            field.id = `tts-${s.name}`;
            field.style.fontSize = '0.72rem';
            row.append(label, dot, field);
            form.appendChild(row);
        }
        fillTopicsForm('value');
    }

    // Fill the form with the live ('value') or the config file's ('saved') settings
    function fillTopicsForm(which) {
        for (const s of _settings.settings) $(`tts-${s.name}`).value = s[which];
        _settings.settings.filter(isTopic).forEach(updateDot);
    }

    async function showTopicsModal() {
        if (STATIC_BASE) {
            setStatus('Tracker topics need the backend — run map_data_viewer locally', 'text-warning');
            return;
        }
        try {
            const res = await api('GET', '/api/tracker/settings');
            if (!res.ok) throw new Error(await errorText(res));
            _settings = await res.json();
        } catch (err) {
            setStatus(`Failed to load tracker topics: ${err.message}`, 'text-danger');
            return;
        }
        topicsError('');
        renderTopicsForm();
        $('tracker-topics-path').textContent = _settings.path;
        const fileNote = $('tracker-topics-file-error');
        fileNote.textContent = _settings.file_error ? `Cannot read the config file: ${_settings.file_error}`
            : (_settings.exists ? '' : 'The config file does not exist yet; Save creates it.');
        fileNote.hidden = !fileNote.textContent;
        bootstrap.Modal.getOrCreateInstance($('tracker-topics-modal')).show();
    }

    async function applyTopics(save) {
        const settings = Object.fromEntries(_settings.settings.map(s => [s.name, $(`tts-${s.name}`).value.trim()]));
        try {
            const res = await api('PUT', '/api/tracker/settings', { body: { settings, save } });
            if (!res.ok) throw new Error(await errorText(res));
            const data = await res.json();
            const what = data.changed ? 'Tracker resubscribed' : 'Tracker topics unchanged';
            setStatus(save ? `${what}; saved to ${data.path}` : `${what} (not saved)`, 'text-success');
        } catch (err) {
            topicsError(err.message);
            return;
        }
        topicsError('');
        bootstrap.Modal.getInstance($('tracker-topics-modal')).hide();
    }

    $('tracker-topics-btn')?.addEventListener('click', showTopicsModal);
    $('tracker-topics-reload')?.addEventListener('click', () => _settings && fillTopicsForm('saved'));
    $('tracker-topics-apply')?.addEventListener('click', () => applyTopics(false));
    $('tracker-topics-save')?.addEventListener('click', () => applyTopics(true));

    // Auto-connect if ROS is available to show robot in other modes
    if (typeof ros_available !== 'undefined' && ros_available) {
        initSocket();
    }

    // The trail is recorded on the node, so it has to be cleared there: dropping only the
    // polyline would bring the old fixes back with the next telemetry frame.
    $('tracker-clear-trail')?.addEventListener('click', async () => {
        if (STATIC_BASE) {
            setStatus('Clearing the trail needs the backend — run map_data_viewer locally', 'text-warning');
            return;
        }
        try {
            const res = await api('DELETE', '/api/tracker/trail');
            if (!res.ok) throw new Error(await errorText(res));
            const { dropped } = await res.json();
            if (trailLayer) {
                map.removeLayer(trailLayer);
                trailLayer = null;
            }
            setStatus(`Trail cleared (${dropped} fixes)`, 'text-success');
        } catch (err) {
            setStatus(`Failed to clear the trail: ${err.message}`, 'text-danger');
        }
    });

    document.getElementById('tracker-center-robot')?.addEventListener('click', () => {
        if (robotMarker) {
            const currentZoom = map.getZoom();
            const targetZoom = currentZoom < 18 ? 18 : currentZoom;
            map.setView(robotMarker.getLatLng(), targetZoom);
        }
    });

    return {
        enable: () => {
            enabled = true;
            initSocket();
            if (robotMarker) {
                robotMarker.addTo(map);
                map.setView(robotMarker.getLatLng(), 18);
            }
            if (robotPathLayer) robotPathLayer.addTo(map);
        },
        disable: () => {
            enabled = false;
            // Note: We don't remove the layer here because it's now globally visible
        },
        showRobot: showRobot,
        hideRobot: hideRobot,
        position: () => (_lastPos ? { ..._lastPos } : null)
    };
})();
