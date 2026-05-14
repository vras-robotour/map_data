const trackerMode = (() => {
    let socket = null;
    let robotMarker = null;
    let robotPathLayer = null;
    let enabled = false;

    // Cached DOM/element references — resolved lazily on first use
    let _robotCb = null;
    let _followCb = null;
    let _robotSvg = null;
    let _lastWaypointKey = null;
    let _uiRefs = null;

    const ROBOT_ICON_HTML = `
    <div class="robot-marker-container">
      <svg viewBox="0 0 24 24" width="30" height="30" style="filter: drop-shadow(0 0 2px rgba(0,0,0,0.5));">
        <path d="M12,2L4.5,20.29L5.21,21L12,18L18.79,21L19.5,20.29L12,2Z" fill="#00ff00" stroke="#000" stroke-width="1"/>
      </svg>
    </div>
  `;

    function initSocket() {
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

        // Check robot layer visibility — cache the checkbox element
        if (!_robotCb) _robotCb = document.querySelector('[data-layer="robot"]');
        if (_robotCb && !_robotCb.checked) {
            hideRobot();
            return;
        }

        const pos = data.position.ekf.lat ? data.position.ekf : data.position.gps;
        if (pos && pos.lat && pos.lon) {
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

        // Update planned path only when the waypoint set actually changes
        if (data.mission && data.mission.waypoints && data.mission.waypoints.length > 0) {
            const wps = data.mission.waypoints;
            const key = `${wps.length}:${wps[0].lat},${wps[0].lon}:${wps[wps.length - 1].lat},${wps[wps.length - 1].lon}`;
            if (!robotPathLayer) {
                robotPathLayer = L.polyline(wps.map(w => [w.lat, w.lon]), {
                    color: '#00ff00',
                    weight: 3,
                    opacity: 0.6,
                    dashArray: '5, 10'
                }).addTo(map);
                _lastWaypointKey = key;
            } else if (key !== _lastWaypointKey) {
                robotPathLayer.setLatLngs(wps.map(w => [w.lat, w.lon]));
                _lastWaypointKey = key;
            }
        } else if (robotPathLayer) {
            map.removeLayer(robotPathLayer);
            robotPathLayer = null;
            _lastWaypointKey = null;
        }
    }

    function hideRobot() {
        if (robotMarker && map.hasLayer(robotMarker)) map.removeLayer(robotMarker);
        if (robotPathLayer && map.hasLayer(robotPathLayer)) map.removeLayer(robotPathLayer);
        _robotSvg = null; // Leaflet recreates the element on the next addTo()
    }

    function showRobot() {
        initSocket();
        if (robotMarker && !map.hasLayer(robotMarker)) robotMarker.addTo(map);
        if (robotPathLayer && !map.hasLayer(robotPathLayer)) robotPathLayer.addTo(map);
    }

    function _initUI(container) {
        container.innerHTML = `
      <div id="tsi-section-hardware" class="mb-3">
        <div id="tsi-row-battery" class="d-flex justify-content-between border-bottom border-secondary pb-1 mb-1">
          <span class="text-secondary fw-bold">BATTERY</span>
          <span id="tsi-battery" class="text-light">—</span>
        </div>
        <div id="tsi-row-motors" class="d-flex justify-content-between border-bottom border-secondary pb-1 mb-1">
          <span class="text-secondary fw-bold">MOTORS</span>
          <span id="tsi-motors">—</span>
        </div>
        <div id="tsi-motor-error" class="text-danger small" style="display:none"></div>
        <div id="tsi-row-temp" class="d-flex justify-content-between border-bottom border-secondary pb-1 mb-1">
          <span class="text-secondary fw-bold">TEMP</span>
          <span id="tsi-temp" class="text-light">—</span>
        </div>
      </div>
      <div id="tsi-section-localization" class="mb-3">
        <div id="tsi-row-gps" class="d-flex justify-content-between border-bottom border-secondary pb-1 mb-1">
          <span class="text-secondary fw-bold">LOCALIZATION</span>
          <span id="tsi-gps-fix">—</span>
        </div>
        <div id="tsi-row-speed" class="d-flex justify-content-between border-bottom border-secondary pb-1 mb-1">
          <span class="text-secondary fw-bold">SPEED</span>
          <span id="tsi-speed">—</span>
        </div>
        <div id="tsi-row-speed-limit" class="d-flex justify-content-between border-bottom border-secondary pb-1 mb-1">
          <span class="text-secondary fw-bold">LIMIT</span>
          <span id="tsi-speed-limit">—</span>
        </div>
      </div>
      <div id="tsi-section-navigation" class="mb-3">
        <div class="panel-title mb-1" style="font-size:0.6rem;">NAVIGATION</div>
        <div class="small">
          <div id="tsi-row-nav-state">State: <span id="tsi-nav-state" class="text-info">IDLE</span></div>
          <div id="tsi-collision" style="display:none">Collision: <span id="tsi-collision-val" class="text-warning"></span></div>
          <div id="tsi-recovery" class="text-danger fw-bold" style="display:none">RECOVERY ACTIVE</div>
          <div id="tsi-teleop" class="text-warning fw-bold" style="display:none">TELEOP ACTIVE</div>
        </div>
      </div>
      <div id="tsi-speech-box" class="mt-3 p-2 bg-dark border border-secondary rounded" style="display:none">
        <div class="small text-secondary mb-1">LAST SPEECH (<span id="tsi-speech-level"></span>)</div>
        <div id="tsi-speech-text"></div>
      </div>`;

        _uiRefs = {
            sectionHardware: document.getElementById('tsi-section-hardware'),
            rowBattery: document.getElementById('tsi-row-battery'),
            battery: document.getElementById('tsi-battery'),
            rowMotors: document.getElementById('tsi-row-motors'),
            motors: document.getElementById('tsi-motors'),
            motorError: document.getElementById('tsi-motor-error'),
            rowTemp: document.getElementById('tsi-row-temp'),
            temp: document.getElementById('tsi-temp'),

            sectionLocalization: document.getElementById('tsi-section-localization'),
            rowGps: document.getElementById('tsi-row-gps'),
            gpsFix: document.getElementById('tsi-gps-fix'),
            rowSpeed: document.getElementById('tsi-row-speed'),
            speed: document.getElementById('tsi-speed'),
            rowSpeedLimit: document.getElementById('tsi-row-speed-limit'),
            speedLimit: document.getElementById('tsi-speed-limit'),

            sectionNavigation: document.getElementById('tsi-section-navigation'),
            rowNavState: document.getElementById('tsi-row-nav-state'),
            navState: document.getElementById('tsi-nav-state'),
            collision: document.getElementById('tsi-collision'),
            collisionVal: document.getElementById('tsi-collision-val'),
            recovery: document.getElementById('tsi-recovery'),
            teleop: document.getElementById('tsi-teleop'),

            speechBox: document.getElementById('tsi-speech-box'),
            speechLevel: document.getElementById('tsi-speech-level'),
            speechText: document.getElementById('tsi-speech-text'),
        };
    }

    function updateUI(data) {
        const content = document.getElementById('tracker-status-content');
        if (!content) return;

        if (!_uiRefs) _initUI(content);

        const s = data.status || {};
        const b = s.battery || {};
        const feat = data.enabled_features || {};

        // Hardware section
        const hasBattery = feat.battery !== false;
        const hasMotors = feat.motors !== false;
        const hasTemp = feat.temp !== false;
        const hasMotorError = feat.motor_error !== false;

        _uiRefs.rowBattery.style.display = hasBattery ? '' : 'none';
        _uiRefs.rowMotors.style.display = hasMotors ? '' : 'none';
        _uiRefs.rowTemp.style.display = hasTemp ? '' : 'none';
        _uiRefs.sectionHardware.style.display = (hasBattery || hasMotors || hasTemp || hasMotorError) ? '' : 'none';

        if (hasBattery) {
            const battV = b.voltage != null ? b.voltage : '—';
            const battA = b.current != null ? b.current : '—';
            _uiRefs.battery.textContent = `${battV} V / ${battA} A`;
            _uiRefs.battery.className = (b.voltage && b.voltage < 22.0) ? 'text-danger' : 'text-light';
        }

        if (hasMotors) {
            _uiRefs.motors.textContent = s.motors_enabled ? 'ENABLED' : 'DISABLED';
            _uiRefs.motors.className = s.motors_enabled ? 'text-success' : 'text-danger';
        }

        if (hasMotorError && s.motor_error) {
            _uiRefs.motorError.textContent = `Error: 0x${s.motor_error.toString(16)}`;
            _uiRefs.motorError.style.display = '';
        } else {
            _uiRefs.motorError.style.display = 'none';
        }

        if (hasTemp) {
            _uiRefs.temp.textContent = s.teensy_temp != null ? `${s.teensy_temp} °C` : '—';
        }

        // Localization section
        const hasGps = feat.gps_fix !== false || feat.gps_ekf !== false;
        const hasSpeed = feat.speed !== false;
        const hasSpeedLimit = feat.speed_limit !== false;

        _uiRefs.rowGps.style.display = (feat.gps_fix !== false) ? '' : 'none';
        _uiRefs.rowSpeed.style.display = hasSpeed ? '' : 'none';
        _uiRefs.rowSpeedLimit.style.display = hasSpeedLimit ? '' : 'none';
        _uiRefs.sectionLocalization.style.display = (hasGps || hasSpeed || hasSpeedLimit) ? '' : 'none';

        if (feat.gps_fix !== false) {
            let fixStr = 'No Fix', fixClass = 'text-danger';
            if (s.gps_fix === 0) { fixStr = 'Fix'; fixClass = 'text-success'; }
            else if (s.gps_fix === 1) { fixStr = 'Float'; fixClass = 'text-warning'; }
            else if (s.gps_fix === 2) { fixStr = 'Fixed'; fixClass = 'text-info'; }
            _uiRefs.gpsFix.textContent = fixStr;
            _uiRefs.gpsFix.className = fixClass;
        }

        if (hasSpeed) {
            _uiRefs.speed.textContent = s.speed != null ? `${s.speed} m/s` : '—';
        }

        if (hasSpeedLimit) {
            _uiRefs.speedLimit.textContent = s.speed_limit
                ? `${s.speed_limit.value} ${s.speed_limit.percentage ? '%' : 'm/s'}`
                : '—';
        }

        // Navigation section
        const hasNavState = feat.nav_state !== false;
        const hasCollision = feat.collision !== false;
        const hasRecovery = feat.recovery !== false;
        const hasTeleop = feat.teleop !== false;

        _uiRefs.rowNavState.style.display = hasNavState ? '' : 'none';
        _uiRefs.sectionNavigation.style.display = (hasNavState || hasCollision || hasRecovery || hasTeleop) ? '' : 'none';

        if (hasNavState) {
            _uiRefs.navState.textContent = s.nav_state || 'IDLE';
        }

        if (hasCollision) {
            const showCollision = s.collision_action && s.collision_action !== 'PASSTHROUGH';
            _uiRefs.collision.style.display = showCollision ? '' : 'none';
            if (showCollision) _uiRefs.collisionVal.textContent = s.collision_action;
        } else {
            _uiRefs.collision.style.display = 'none';
        }

        _uiRefs.recovery.style.display = (hasRecovery && s.recovery_active) ? '' : 'none';
        _uiRefs.teleop.style.display = (hasTeleop && s.teleop_active) ? '' : 'none';

        // Last speech
        if (feat.speech !== false && s.last_speech) {
            const levelClass = s.last_speech.level === 'error' ? 'text-danger'
                : (s.last_speech.level === 'warn' ? 'text-warning' : 'text-info');
            _uiRefs.speechLevel.textContent = s.last_speech.level;
            _uiRefs.speechText.textContent = s.last_speech.text;
            _uiRefs.speechText.className = levelClass;
            _uiRefs.speechBox.style.display = '';
        } else {
            _uiRefs.speechBox.style.display = 'none';
        }

        // Hide Robot layer checkbox if robot positioning is disabled
        const robotCb = document.querySelector('[data-layer="robot"]');
        if (feat.gps_fix === false && feat.gps_ekf === false) {
            if (robotCb) {
                const row = robotCb.closest('.layer-row');
                if (row) row.style.display = 'none';
            }
            hideRobot();
        } else {
            if (robotCb) {
                const row = robotCb.closest('.layer-row');
                if (row) row.style.display = '';
            }
        }

    }


    // Auto-connect if ROS is available to show robot in other modes
    if (typeof ros_available !== 'undefined' && ros_available) {
        initSocket();
    }

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
        hideRobot: hideRobot
    };
})();
