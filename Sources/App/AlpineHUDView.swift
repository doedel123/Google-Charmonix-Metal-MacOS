import SwiftUI

public struct AlpineHUDView: View {
    @ObservedObject public var appState: AppState

    public init(appState: AppState) {
        self.appState = appState
    }

    private var activeZoneName: String {
        let zones = appState.dropZones
        guard !zones.isEmpty else { return "VALLÉE BLANCHE — COL DU MIDI" }
        let idx = max(0, min(zones.count - 1, appState.telemetry.zoneIdx))
        return zones[idx].name.uppercased()
    }

    private var activeLightName: String {
        let presets = LightingPresets.all
        let idx = max(0, min(presets.count - 1, appState.telemetry.lightIdx))
        return presets[idx].name
    }

    public var body: some View {
        ZStack {
            // 1. Full HUD (hudMode == 0)
            if appState.telemetry.hudMode == 0 && !appState.isLoading {
                fullHUDLayer
            }

            // 2. RAW Action-Cam REC Pill (hudMode == 1)
            if appState.telemetry.hudMode == 1 && !appState.isLoading {
                rawRecOverlay
            }

            // 3. Drop-In Start Overlay
            if !appState.telemetry.started && !appState.isLoading {
                startOverlay
            }

            // 4. Bottom-center DEM Attribution (visible in Full HUD)
            if appState.telemetry.hudMode == 0 && !appState.isLoading {
                VStack {
                    Spacer()
                    Text("Terrain DEM: AWS Open Data / Mapzen Terrarium (Copernicus EU-DEM)")
                        .font(.system(size: 8.5, weight: .regular))
                        .tracking(0.4)
                        .foregroundColor(Color(red: 0.85, green: 0.91, blue: 0.97).opacity(0.78))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color(red: 8/255, green: 14/255, blue: 26/255).opacity(0.55))
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.white.opacity(0.09), lineWidth: 1))
                        .cornerRadius(4)
                        .padding(.bottom, 4)
                }
                .allowsHitTesting(false)
            }

            // 5. Loading Veil
            if appState.isLoading {
                loadingVeil
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Loading Veil

    private var loadingVeil: some View {
        ZStack {
            RadialGradient(
                colors: [Color(red: 24/255, green: 42/255, blue: 69/255), Color(red: 9/255, green: 15/255, blue: 25/255)],
                center: UnitPoint(x: 0.5, y: 0.42),
                startRadius: 20,
                endRadius: 680
            )
            .ignoresSafeArea()

            VStack(spacing: 14) {
                if appState.loadError == nil {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(AlpinePalette.accent)
                        .scaleEffect(0.95)
                }
                Text("CHAMONIX · MONT BLANC")
                    .font(.system(size: 20, weight: .bold))
                    .tracking(5.6)
                    .foregroundColor(Color(red: 242/255, green: 246/255, blue: 251/255))
                Text(appState.loadingMessage)
                    .font(.system(size: 12))
                    .tracking(0.8)
                    .foregroundColor(appState.loadError != nil ? Color(red: 1, green: 0.65, blue: 0.65) : Color(red: 212/255, green: 228/255, blue: 248/255).opacity(0.72))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }
        }
    }

    // MARK: - Start Overlay

    private var startOverlay: some View {
        ZStack(alignment: .bottom) {
            LinearGradient(
                colors: [
                    Color(red: 6/255, green: 12/255, blue: 24/255).opacity(0.08),
                    Color.clear,
                    Color(red: 6/255, green: 12/255, blue: 24/255).opacity(0.56),
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)

            VStack(spacing: 6) {
                Text("CHAMONIX · MASSIF DU MONT-BLANC · STRICT 4-POINT SIM")
                    .font(.system(size: 10, weight: .bold))
                    .tracking(2.6)
                    .foregroundColor(AlpinePalette.accent)
                Text("DROP IN")
                    .font(.system(size: 24, weight: .bold))
                    .tracking(2.8)
                    .foregroundColor(.white)
                HStack(spacing: 4) {
                    Text("Click to drop in & lock")
                    Text("free head-look").fontWeight(.bold)
                    Text("·")
                    KbdPill("A")
                    Text("/")
                    KbdPill("D")
                    Text("edge & carve ·")
                    KbdPill("W")
                    Text("/")
                    KbdPill("F")
                    Text("pole / tuck ·")
                    KbdPill("Q")
                    Text("/")
                    KbdPill("E")
                    Text("alt-pole ·")
                    KbdPill("Shift")
                    Text("stop ·")
                    KbdPill("Space")
                    Text("pop")
                }
                .font(.system(size: 11.5))
                .foregroundColor(Color(red: 232/255, green: 241/255, blue: 252/255).opacity(0.86))
            }
            .padding(.horizontal, 34)
            .padding(.vertical, 18)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(red: 10/255, green: 16/255, blue: 28/255).opacity(0.70))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.45), radius: 24, y: 12)
            .padding(.bottom, 76)
            .onTapGesture {
                appState.dropIn()
            }
        }
    }

    // MARK: - RAW REC Pill

    private var rawRecOverlay: some View {
        let totalSec = max(0, Int(appState.telemetry.simTime))
        let mm = String(format: "%02d", totalSec / 60)
        let ss = String(format: "%02d", totalSec % 60)
        let camTag = appState.telemetry.camMode == .chesty ? "CHESTY" : "4K"
        let alt = Int(roundf(appState.telemetry.altitude))

        return VStack {
            HStack {
                HStack(spacing: 9) {
                    PulsingRecDot()
                    Text("REC \(mm):\(ss)")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .tracking(1.4)
                        .foregroundColor(.white.opacity(0.95))
                    Text("\(camTag) · 60 · \(alt)m")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundColor(.white.opacity(0.72))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(AlpinePalette.glassDark.opacity(0.60))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.16), lineWidth: 1))
                .cornerRadius(6)
                .padding(.top, 18)
                .padding(.leading, 22)
                Spacer()
            }
            Spacer()
        }
        .allowsHitTesting(false)
    }

    // MARK: - Full HUD Layer

    private var fullHUDLayer: some View {
        ZStack {
            // Top bar: Zone selector (left), Compass ribbon (center), Quick controls (right)
            VStack(spacing: 0) {
                ZStack(alignment: .top) {
                    // Peak Compass Ribbon centered in top gap
                    compassRibbonView
                        .padding(.top, 14)
                        .allowsHitTesting(false)

                    HStack(alignment: .top) {
                        topLeftZoneSelector
                        Spacer()
                        topRightControls
                    }
                    .padding(.top, 16)
                    .padding(.horizontal, 18)
                }

                // Mode / Avalanche Banners + Air Badge
                VStack(spacing: 8) {
                    if appState.telemetry.scoutActive || appState.telemetry.replayActive {
                        modeBannerView
                    }
                    if appState.telemetry.avyActive && !appState.telemetry.scoutActive && !appState.telemetry.replayActive {
                        avyBannerView
                    }
                    if !appState.telemetry.replayActive && !appState.telemetry.scoutActive &&
                        !appState.telemetry.grounded && appState.telemetry.airTime > 0.16 && !appState.telemetry.crashed {
                        airBadgeView
                    }
                }
                .padding(.top, 6)
                .allowsHitTesting(false)

                Spacer()

                // Lower center overlays (Trick toast, Pop bar, Replay bar)
                VStack(spacing: 8) {
                    if let toast = activeTrickToast {
                        trickToastView(text: toast.text, quality: toast.quality)
                    }
                    if !appState.telemetry.replayActive && !appState.telemetry.scoutActive &&
                        appState.telemetry.popCharge > 0.04 && appState.telemetry.grounded {
                        popBarView
                    }
                    if appState.telemetry.replayActive {
                        replayBarView
                    }
                }
                .padding(.bottom, 14)
                .allowsHitTesting(false)

                // Bottom row: Telemetry (left) + Controls Hint (right)
                HStack(alignment: .bottom) {
                    telemetryCluster
                        .allowsHitTesting(false)
                    Spacer()
                    if appState.controlsExpanded {
                        controlsHintCard
                            .allowsHitTesting(false)
                    }
                }
                .padding(.horizontal, 18)
                .padding(.bottom, 26)
            }

            // Center Crash Banner
            if !appState.telemetry.replayActive && !appState.telemetry.scoutActive && appState.telemetry.crashed {
                crashBannerView
                    .allowsHitTesting(false)
            }

            // Settings Popover (top-right below buttons)
            if appState.settingsOpen {
                VStack {
                    HStack {
                        Spacer()
                        settingsPopover
                            .padding(.top, 94)
                            .padding(.trailing, 18)
                    }
                    Spacer()
                }
            }
        }
    }

    // MARK: - Compass Ribbon

    private var compassRibbonView: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                Rectangle()
                    .fill(Color.white.opacity(0.16))
                    .frame(height: 1)

                ForEach(appState.compassPins) { pin in
                    VStack(spacing: 1) {
                        Rectangle()
                            .fill(Color.white.opacity(0.75))
                            .frame(width: 1, height: 7)
                        Text(pin.name.uppercased())
                            .font(.system(size: 10, weight: .semibold))
                            .tracking(0.9)
                            .foregroundColor(.white.opacity(0.94))
                        Text("\(pin.altLabel) · \(pin.distLabel)")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(Color(red: 210/255, green: 228/255, blue: 252/255).opacity(0.80))
                    }
                    .shadow(color: .black.opacity(0.68), radius: 3, y: 1)
                    .fixedSize()
                    .position(x: geo.size.width * (pin.pct / 100.0), y: 18)
                }
            }
        }
        .frame(width: 280, height: 36)
        .mask(
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0.0),
                    .init(color: .black, location: 0.16),
                    .init(color: .black, location: 0.84),
                    .init(color: .clear, location: 1.0),
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
        )
    }

    // MARK: - Top-Left Zone Selector

    private var topLeftZoneSelector: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                appState.dropMenuOpen.toggle()
            } label: {
                HStack(spacing: 8) {
                    PulsingRecDot()
                    Text(activeZoneName)
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(1.1)
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 11)
                .padding(.vertical, 6)
                .background(AlpinePalette.glassDark)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.14), lineWidth: 1))
                .cornerRadius(6)
            }
            .buttonStyle(.plain)

            if appState.dropMenuOpen {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(appState.dropZones.enumerated()), id: \.element.id) { idx, dz in
                        let isActive = idx == appState.telemetry.zoneIdx
                        Button {
                            appState.selectZone(idx)
                        } label: {
                            HStack(spacing: 7) {
                                Text("\(idx + 1)")
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .foregroundColor(AlpinePalette.accent)
                                    .frame(width: 14)
                                Text(dz.name)
                                    .font(.system(size: 10))
                                    .foregroundColor(isActive ? .white : Color(red: 236/255, green: 244/255, blue: 255/255).opacity(0.86))
                                Spacer(minLength: 10)
                                Text("\(Int(roundf(dz.elev)))m · \(Int(roundf(dz.slopeDeg)))°")
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.72))
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .frame(minWidth: 270)
                            .background(isActive ? AlpinePalette.accent.opacity(0.24) : AlpinePalette.glassDark.opacity(0.82))
                            .overlay(
                                RoundedRectangle(cornerRadius: 5)
                                    .stroke(isActive ? AlpinePalette.accent.opacity(0.85) : Color.white.opacity(0.10), lineWidth: 1)
                            )
                            .cornerRadius(5)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    // MARK: - Top-Right Quick Buttons

    private var topRightControls: some View {
        let fps = Int(roundf(appState.telemetry.fps))
        let isLowFps = fps < 32
        let isAdapted = appState.settings.quality == .auto && appState.activeTier > 0

        return VStack(alignment: .trailing, spacing: 5) {
            HStack(spacing: 5) {
                HudButton(title: activeLightName) {
                    appState.cycleLight()
                }
                HudButton(title: appState.telemetry.camMode.label) {
                    appState.cycleCamera()
                }
                HudButton(title: "RESET") {
                    appState.resetRun()
                }
                HudButton(title: "AUDIO", muted: appState.settings.muted) {
                    appState.toggleMute()
                }
                HudButton(title: "⚙", active: appState.settingsOpen) {
                    appState.settingsOpen.toggle()
                }
                HudButton(title: "?", active: appState.controlsExpanded) {
                    appState.controlsExpanded.toggle()
                }
                Text("\(fps) FPS")
                    .font(.system(size: 9.5, weight: .medium, design: .monospaced))
                    .foregroundColor(
                        isLowFps ? Color(red: 253/255, green: 230/255, blue: 138/255) :
                            (isAdapted ? Color(red: 186/255, green: 230/255, blue: 253/255) : Color(red: 200/255, green: 224/255, blue: 250/255).opacity(0.85))
                    )
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(AlpinePalette.glassDark)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(isLowFps ? Color.yellow.opacity(0.45) : (isAdapted ? Color.cyan.opacity(0.45) : Color.white.opacity(0.10)), lineWidth: 1)
                    )
                    .cornerRadius(6)
            }

            HStack(spacing: 5) {
                HudButton(title: "SCOUT · V", active: appState.telemetry.scoutActive, sub: true) {
                    appState.toggleScout()
                }
                HudButton(title: "REPLAY · P", active: appState.telemetry.replayActive, sub: true) {
                    appState.toggleReplay()
                }
                HudButton(
                    title: appState.settings.avalanchesEnabled ? "AVY: ON" : "AVY: OFF",
                    sub: true,
                    tintColor: appState.settings.avalanchesEnabled ? Color(red: 254/255, green: 202/255, blue: 202/255) : Color.gray.opacity(0.8),
                    borderColor: appState.settings.avalanchesEnabled ? Color.red.opacity(0.45) : Color.gray.opacity(0.35)
                ) {
                    appState.toggleAvalanches()
                }
                HudButton(
                    title: "SLAB AVY · G",
                    active: appState.telemetry.avyActive,
                    muted: !appState.settings.avalanchesEnabled,
                    sub: true,
                    activeColor: Color(red: 220/255, green: 38/255, blue: 38/255).opacity(0.40),
                    borderColor: appState.telemetry.avyActive ? Color.red.opacity(0.9) : nil
                ) {
                    appState.triggerAvalanche()
                }
            }

            Text(appState.pointerLocked
                 ? "HEAD LOOK LOCKED · L / MMB = CENTER · ESC = CURSOR"
                 : "MOVE MOUSE TO LOOK AROUND · CLICK SNOW TO LOCK VIEW")
                .font(.system(size: 8.5, weight: .medium))
                .tracking(1.0)
                .foregroundColor(appState.pointerLocked
                                 ? Color(red: 187/255, green: 247/255, blue: 208/255)
                                 : Color(red: 210/255, green: 228/255, blue: 250/255).opacity(0.75))
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(appState.pointerLocked
                            ? Color(red: 16/255, green: 48/255, blue: 28/255).opacity(0.65)
                            : Color(red: 8/255, green: 14/255, blue: 26/255).opacity(0.45))
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(appState.pointerLocked ? Color.green.opacity(0.45) : Color.white.opacity(0.09), lineWidth: 1)
                )
                .cornerRadius(4)
                .allowsHitTesting(false)
        }
    }

    // MARK: - Settings Popover

    private var settingsPopover: some View {
        let tierNames = ["HIGH", "MED", "LOW"]
        let w = Int(max(1, appState.viewportSizePx.width))
        let h = Int(max(1, appState.viewportSizePx.height))
        let msaa = appState.renderer?.subsystems?.postOptics.msaaSampleCount ?? 4
        let aaTxt = msaa > 1 ? "\(msaa)× MSAA" : "FXAA"
        let fxTxt = appState.activeTier == 2 ? "Fast Optics" : "God-Rays + Bloom"
        let qualLabel = appState.settings.quality == .auto
            ? "AUTO (\(tierNames[appState.activeTier]))"
            : tierNames[appState.activeTier]

        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("SETTINGS · PERFORMANCE")
                    .font(.system(size: 10, weight: .bold))
                    .tracking(1.6)
                    .foregroundColor(Color(red: 232/255, green: 242/255, blue: 255/255).opacity(0.92))
                Spacer()
                Button("×") {
                    appState.settingsOpen = false
                }
                .buttonStyle(.plain)
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(Color.white.opacity(0.75))
            }
            .padding(.bottom, 4)
            .overlay(Rectangle().frame(height: 1).foregroundColor(.white.opacity(0.10)), alignment: .bottom)

            // Graphics Quality
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text("GRAPHICS QUALITY")
                    Spacer()
                    Text(qualLabel).font(.system(size: 9.5, weight: .semibold, design: .monospaced)).foregroundColor(.white)
                }
                .font(.system(size: 9.5, weight: .medium))
                .foregroundColor(Color(red: 218/255, green: 232/255, blue: 250/255).opacity(0.78))

                HStack(spacing: 3) {
                    ForEach(QualityMode.allCases, id: \.self) { q in
                        let isSel = appState.settings.quality == q
                        Button {
                            appState.setQualityMode(q)
                        } label: {
                            Text(q.rawValue.uppercased())
                                .font(.system(size: 9, weight: .bold))
                                .tracking(0.8)
                                .foregroundColor(isSel ? .white : .white.opacity(0.72))
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 4)
                                .background(isSel ? AlpinePalette.accent.opacity(0.85) : Color.clear)
                                .cornerRadius(4)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(2)
                .background(Color.white.opacity(0.06))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.09), lineWidth: 1))
                .cornerRadius(6)

                Text("\(w)×\(h) · \(aaTxt) · \(fxTxt)")
                    .font(.system(size: 8.5, design: .monospaced))
                    .foregroundColor(Color(red: 188/255, green: 212/255, blue: 240/255).opacity(0.72))
            }

            // Audio Volume
            VStack(spacing: 4) {
                HStack {
                    Text("AUDIO VOLUME")
                    Spacer()
                    Text("\(Int(roundf(appState.settings.volume * 100)))%")
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .foregroundColor(.white)
                }
                .font(.system(size: 9.5))
                .foregroundColor(Color(red: 218/255, green: 232/255, blue: 250/255).opacity(0.78))

                Slider(value: Binding(
                    get: { Double(appState.settings.volume) },
                    set: { newVal in
                        var s = appState.settings
                        s.volume = Float(newVal)
                        s.muted = s.volume <= 0.001
                        appState.renderer?.subsystems?.audio.unlock()
                        appState.applySettings(s)
                    }
                ), in: 0.0...1.0)
                .tint(AlpinePalette.accent)
            }

            // Look Sensitivity
            VStack(spacing: 4) {
                HStack {
                    Text("LOOK SENSITIVITY")
                    Spacer()
                    Text(String(format: "%.1f×", appState.settings.lookSens))
                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                        .foregroundColor(.white)
                }
                .font(.system(size: 9.5))
                .foregroundColor(Color(red: 218/255, green: 232/255, blue: 250/255).opacity(0.78))

                Slider(value: Binding(
                    get: { Double(appState.settings.lookSens) },
                    set: { newVal in
                        var s = appState.settings
                        s.lookSens = Float(newVal)
                        appState.applySettings(s)
                    }
                ), in: 0.4...2.2, step: 0.05)
                .tint(AlpinePalette.accent)
            }

            // Inline Toggles
            SettingsToggleRow(label: "INVERT LOOK Y", title: appState.settings.invertY ? "ON" : "OFF", isOn: appState.settings.invertY) {
                var s = appState.settings
                s.invertY.toggle()
                appState.applySettings(s)
            }
            SettingsToggleRow(label: "MOTION BLUR", title: appState.settings.motionBlur ? "ON" : "OFF", isOn: appState.settings.motionBlur) {
                var s = appState.settings
                s.motionBlur.toggle()
                appState.applySettings(s)
            }
            SettingsToggleRow(label: "KEYBOARD CARVE", title: appState.settings.smoothCarve ? "SMOOTH" : "DIRECT", isOn: appState.settings.smoothCarve) {
                var s = appState.settings
                s.smoothCarve.toggle()
                appState.applySettings(s)
            }
            SettingsToggleRow(label: "AVALANCHES", title: appState.settings.avalanchesEnabled ? "ON" : "OFF", isOn: appState.settings.avalanchesEnabled) {
                appState.toggleAvalanches()
            }

            // Gamepad Status
            Text(appState.gamepadName.map { "GAMEPAD: \($0)" } ?? "GAMEPAD: NONE DETECTED")
                .font(.system(size: 8.5, design: .monospaced))
                .tracking(0.6)
                .foregroundColor(appState.gamepadName != nil
                                 ? Color(red: 187/255, green: 247/255, blue: 208/255)
                                 : Color(red: 196/255, green: 214/255, blue: 238/255).opacity(0.68))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(appState.gamepadName != nil
                            ? Color(red: 22/255, green: 101/255, blue: 52/255).opacity(0.42)
                            : Color.white.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: 5)
                        .stroke(appState.gamepadName != nil ? Color.green.opacity(0.55) : Color.white.opacity(0.09), lineWidth: 1)
                )
                .cornerRadius(5)

            // Footer
            HStack {
                Text("O toggle · Gamepad ready")
                    .font(.system(size: 8.5))
                    .foregroundColor(Color(red: 196/255, green: 214/255, blue: 238/255).opacity(0.62))
                Spacer()
                Button("RESET") {
                    appState.resetSettings()
                }
                .buttonStyle(.plain)
                .font(.system(size: 8.5, weight: .semibold))
                .foregroundColor(.white.opacity(0.85))
                .padding(.horizontal, 7)
                .padding(.vertical, 2)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.white.opacity(0.18), lineWidth: 1))
            }
            .padding(.top, 4)
            .overlay(Rectangle().frame(height: 1).foregroundColor(.white.opacity(0.09)), alignment: .top)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(width: 268)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(red: 8/255, green: 14/255, blue: 26/255).opacity(0.86))
        )
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.16), lineWidth: 1))
        .shadow(color: .black.opacity(0.45), radius: 20, y: 10)
    }

    // MARK: - Mode / Avalanche / Air / Trick / Crash Banners

    private var modeBannerView: some View {
        let isScout = appState.telemetry.scoutActive
        return VStack(spacing: 2) {
            Text(isScout ? "LINE SCOUT · HELICOPTER PREVIEW" : "HELICOPTER REPLAY · 30 HZ")
                .font(.system(size: 10.5, weight: .bold))
                .tracking(1.6)
                .foregroundColor(Color(red: 219/255, green: 238/255, blue: 255/255))
            Text(isScout
                 ? "Press V or ESC to drop into helmet POV"
                 : "\(appState.telemetry.replayFrameCount) frames recorded · Press P or ESC to exit")
                .font(.system(size: 9))
                .tracking(0.8)
                .foregroundColor(Color(red: 196/255, green: 224/255, blue: 250/255).opacity(0.82))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Color(red: 8/255, green: 16/255, blue: 30/255).opacity(0.72))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(red: 110/255, green: 195/255, blue: 255/255).opacity(0.48), lineWidth: 1))
        .cornerRadius(8)
    }

    private var avyBannerView: some View {
        let spKmh = Int(roundf(appState.telemetry.avySpeedMps * 3.6))
        let gap = Int(roundf(appState.telemetry.avyGapMeters))
        let detail = gap > 0
            ? "SLAB FRACTURE · \(spKmh) KM/H · \(gap)m BEHIND"
            : "SLAB FRACTURE · \(spKmh) KM/H · SLAB PASSING — SKI SIDEWAYS!"

        return VStack(spacing: 2) {
            Text("⚠ AVALANCHE WARNING")
                .font(.system(size: 11, weight: .heavy))
                .tracking(1.8)
                .foregroundColor(Color(red: 1, green: 235/255, blue: 233/255))
            Text(detail)
                .font(.system(size: 9.5, design: .monospaced))
                .tracking(0.8)
                .foregroundColor(Color(red: 254/255, green: 202/255, blue: 202/255))
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 7)
        .background(Color(red: 92/255, green: 16/255, blue: 16/255).opacity(0.80))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(red: 252/255, green: 165/255, blue: 165/255).opacity(0.72), lineWidth: 1))
        .cornerRadius(8)
        .shadow(color: Color.red.opacity(0.35), radius: 14, y: 4)
    }

    private var airBadgeView: some View {
        let t = appState.telemetry
        let flips = abs(t.airPitch) / (Float.pi * 2.0)
        let spinDeg = Int(roundf(abs(radToDeg(t.airYaw))))
        let flipStr = flips > 0.4 ? " · FLIP" : ""
        let grabStr = t.grabbing ? " · MUTE GRAB" : ""
        return Text(String(format: "AIR %.1fs · %.1fm · %d°%@%@", t.airTime, t.peakAirHeight, spinDeg, flipStr, grabStr))
            .font(.system(size: 11, weight: .bold, design: .monospaced))
            .tracking(1.0)
            .foregroundColor(Color(red: 1, green: 234/255, blue: 173/255))
            .padding(.horizontal, 14)
            .padding(.vertical, 5)
            .background(Color(red: 8/255, green: 14/255, blue: 26/255).opacity(0.68))
            .overlay(Capsule().stroke(Color(red: 1, green: 208/255, blue: 118/255).opacity(0.45), lineWidth: 1))
            .clipShape(Capsule())
    }

    private var popBarView: some View {
        let pct = CGFloat(min(1, max(0, appState.telemetry.popCharge)))
        return ZStack(alignment: .leading) {
            Capsule()
                .fill(Color(red: 8/255, green: 14/255, blue: 26/255).opacity(0.60))
            Capsule()
                .fill(LinearGradient(colors: [Color(red: 77/255, green: 184/255, blue: 1), AlpinePalette.accent], startPoint: .leading, endPoint: .trailing))
                .frame(width: 140 * pct)
        }
        .frame(width: 140, height: 5)
        .overlay(Capsule().stroke(Color.white.opacity(0.22), lineWidth: 1))
    }

    private var replayBarView: some View {
        let pct = CGFloat(min(1, max(0, appState.telemetry.replayProgress)))
        return ZStack(alignment: .leading) {
            Capsule()
                .fill(Color(red: 8/255, green: 14/255, blue: 26/255).opacity(0.60))
            Capsule()
                .fill(LinearGradient(colors: [AlpinePalette.cyan, AlpinePalette.accent], startPoint: .leading, endPoint: .trailing))
                .frame(width: 300 * pct)
        }
        .frame(width: 300, height: 4)
        .overlay(Capsule().stroke(Color.white.opacity(0.20), lineWidth: 1))
    }

    private var activeTrickToast: (text: String, quality: String)? {
        let t = appState.telemetry
        if t.customToastActive && !t.customToast.isEmpty {
            return (t.customToast, "info")
        }
        let age = t.simTime - t.lastTrickTime
        if !t.lastTrickText.isEmpty && age >= 0 && age < 3.6 && !t.crashed {
            return (t.lastTrickText, t.lastLandingQuality.lowercased())
        }
        return nil
    }

    private func trickToastView(text: String, quality: String) -> some View {
        let border: Color
        let fg: Color
        switch quality {
        case "stomped":
            border = Color(red: 74/255, green: 222/255, blue: 128/255).opacity(0.7)
            fg = Color(red: 214/255, green: 1, blue: 228/255)
        case "sketchy":
            border = Color(red: 251/255, green: 191/255, blue: 36/255).opacity(0.75)
            fg = Color(red: 254/255, green: 240/255, blue: 199/255)
        case "wipeout":
            border = Color(red: 248/255, green: 113/255, blue: 113/255).opacity(0.8)
            fg = Color(red: 1, green: 224/255, blue: 224/255)
        default:
            border = Color(red: 110/255, green: 195/255, blue: 255/255).opacity(0.75)
            fg = Color(red: 226/255, green: 242/255, blue: 1)
        }
        return Text(text)
            .font(.system(size: 12, weight: .bold))
            .tracking(1.4)
            .foregroundColor(fg)
            .padding(.horizontal, 16)
            .padding(.vertical, 7)
            .background(Color(red: 8/255, green: 14/255, blue: 26/255).opacity(0.72))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(border, lineWidth: 1))
            .cornerRadius(8)
    }

    private var crashBannerView: some View {
        VStack(spacing: 4) {
            Text("WIPEOUT")
                .font(.system(size: 20, weight: .heavy))
                .tracking(4.4)
                .foregroundColor(Color(red: 1, green: 138/255, blue: 128/255))
            Text(appState.telemetry.crashReason.isEmpty ? "EDGE CATCH" : appState.telemetry.crashReason)
                .font(.system(size: 11))
                .tracking(1.5)
                .foregroundColor(Color(red: 1, green: 218/255, blue: 214/255))
            Text("Auto-recovering... or press R to stand up")
                .font(.system(size: 10))
                .tracking(0.8)
                .foregroundColor(.white.opacity(0.72))
                .padding(.top, 2)
        }
        .padding(.horizontal, 26)
        .padding(.vertical, 14)
        .background(Color(red: 24/255, green: 8/255, blue: 10/255).opacity(0.76))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(red: 248/255, green: 113/255, blue: 113/255).opacity(0.55), lineWidth: 1))
        .cornerRadius(10)
    }

    // MARK: - Bottom-Left Telemetry Cluster

    private var telemetryCluster: some View {
        let t = appState.telemetry
        let kmh = Int(roundf(t.speedMps * 3.6))
        let ed = Int(roundf(t.edgeDeg))
        let edgeTxt = abs(ed) < 4 ? "FLAT" : "\(abs(ed))°\(ed > 0 ? "R" : "L")"

        return HStack(alignment: .bottom, spacing: 11) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(String(format: "%02d", kmh))
                    .font(.system(size: 34, weight: .bold, design: .monospaced))
                    .monospacedDigit()
                    .foregroundColor(.white)
                Text("KM/H")
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(1.0)
                    .foregroundColor(.white.opacity(0.72))
            }

            HStack(spacing: 10) {
                TelemetryStat(label: "ALT", value: "\(Int(roundf(t.altitude)))")
                TelemetryStat(label: "SLOPE", value: "\(Int(roundf(t.slopeDeg)))°")
                TelemetryStat(label: "EDGE", value: edgeTxt)
                TelemetryStat(label: "DROP", value: "\(Int(roundf(t.verticalDropped)))m")
            }
            .padding(.bottom, 2)

            FallLineDial(slopeDeg: t.slopeDeg, fallAngleDeg: t.fallAngleDeg)
                .padding(.bottom, 1)

            HStack(spacing: 6) {
                SurfaceChip(label: t.surfaceLabel, surfId: t.grounded ? t.surfaceId : "air")
                StanceChip(mode: t.grounded ? t.stanceMode : "AIR")
                if t.isSwitch {
                    Text("SWITCH")
                        .font(.system(size: 9.5, weight: .heavy))
                        .tracking(1.1)
                        .foregroundColor(Color(red: 254/255, green: 243/255, blue: 199/255))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color(red: 217/255, green: 119/255, blue: 6/255).opacity(0.35))
                        .overlay(RoundedRectangle(cornerRadius: 5).stroke(Color(red: 251/255, green: 191/255, blue: 36/255).opacity(0.72), lineWidth: 1))
                        .cornerRadius(5)
                }
            }
            .padding(.bottom, 2)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(AlpinePalette.glassDark)
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.12), lineWidth: 1))
        .cornerRadius(10)
        .shadow(color: .black.opacity(0.45), radius: 6, y: 2)
    }

    // MARK: - Bottom-Right Collapsible Controls Hint

    private var controlsHintCard: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 3) {
                Text("LOOK:").fontWeight(.semibold).foregroundColor(AlpinePalette.warmLabel)
                Text("Move Mouse · Click Snow = Lock ·")
                KbdPill("L")
                Text("/MMB Center ·")
                KbdPill("Esc")
                Text("Cursor")
            }
            HStack(spacing: 3) {
                Text("SKI:").fontWeight(.semibold).foregroundColor(AlpinePalette.warmLabel)
                KbdPill("A")
                Text("/")
                KbdPill("D")
                Text("Edge/Carve ·")
                KbdPill("W")
                Text("/")
                KbdPill("F")
                Text("Pole/Tuck ·")
                KbdPill("Q")
                Text("/")
                KbdPill("E")
                Text("Alt-Pole ·")
                KbdPill("S")
                Text("Float ·")
                KbdPill("Shift")
                Text("Stop ·")
                KbdPill("Space")
                Text("Pop")
            }
            HStack(spacing: 3) {
                Text("AIR / WORLD:").fontWeight(.semibold).foregroundColor(AlpinePalette.warmLabel)
                KbdPill("W")
                Text("/")
                KbdPill("S")
                Text("Flip ·")
                KbdPill("Q")
                Text("/")
                KbdPill("E")
                Text("Cork ·")
                KbdPill("F")
                Text("Grab ·")
                KbdPill("X")
                Text("Switch ·")
                KbdPill("1")
                Text("-")
                KbdPill("6")
                Text("Lines ·")
                KbdPill("C")
                Text("/")
                KbdPill("V")
                Text("/")
                KbdPill("P")
                Text("/")
                KbdPill("G")
                Text("/")
                KbdPill("T")
                Text("/")
                KbdPill("H")
                Text("/")
                KbdPill("O")
            }
        }
        .font(.system(size: 9))
        .foregroundColor(Color(red: 226/255, green: 238/255, blue: 252/255).opacity(0.86))
        .padding(.horizontal, 11)
        .padding(.vertical, 8)
        .background(AlpinePalette.glassDark)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))
        .cornerRadius(8)
    }
}

// MARK: - HUD Subcomponents & Styling

private enum AlpinePalette {
    static let glassDark = Color(red: 8/255, green: 14/255, blue: 26/255).opacity(0.56)
    static let accent = Color(red: 242/255, green: 108/255, blue: 66/255)
    static let cyan = Color(red: 86/255, green: 194/255, blue: 255/255)
    static let warmLabel = Color(red: 244/255, green: 140/255, blue: 106/255)
}

private struct PulsingRecDot: View {
    @State private var dim = false

    var body: some View {
        Circle()
            .fill(Color(red: 239/255, green: 68/255, blue: 68/255))
            .frame(width: 7, height: 7)
            .shadow(color: .red.opacity(0.8), radius: 4)
            .opacity(dim ? 0.35 : 1.0)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                    dim = true
                }
            }
    }
}

private struct KbdPill: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .foregroundColor(.white)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(Color.white.opacity(0.15))
            .cornerRadius(3)
    }
}

private struct HudButton: View {
    let title: String
    var active: Bool = false
    var muted: Bool = false
    var sub: Bool = false
    var tintColor: Color? = nil
    var activeColor: Color? = nil
    var borderColor: Color? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: sub ? 9 : 9.5, weight: .semibold))
                .tracking(0.8)
                .foregroundColor(tintColor ?? (active ? .white : Color(red: 244/255, green: 248/255, blue: 255/255).opacity(0.92)))
                .padding(.horizontal, sub ? 8 : 9)
                .padding(.vertical, sub ? 3.5 : 5)
                .background(active ? (activeColor ?? AlpinePalette.accent.opacity(0.28)) : AlpinePalette.glassDark)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(borderColor ?? (active ? AlpinePalette.accent.opacity(0.90) : Color.white.opacity(0.13)), lineWidth: 1)
                )
                .cornerRadius(6)
                .opacity(muted ? 0.48 : 1.0)
        }
        .buttonStyle(.plain)
    }
}

private struct SettingsToggleRow: View {
    let label: String
    let title: String
    let isOn: Bool
    let action: () -> Void

    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 9.5))
                .tracking(1.0)
                .foregroundColor(Color(red: 218/255, green: 232/255, blue: 250/255).opacity(0.78))
            Spacer()
            Button(action: action) {
                Text(title)
                    .font(.system(size: 9, weight: .bold))
                    .tracking(0.9)
                    .foregroundColor(isOn ? .white : Color(red: 214/255, green: 228/255, blue: 248/255).opacity(0.78))
                    .frame(minWidth: 64)
                    .padding(.vertical, 3)
                    .background(isOn ? AlpinePalette.accent.opacity(0.28) : Color.white.opacity(0.08))
                    .overlay(
                        RoundedRectangle(cornerRadius: 5)
                            .stroke(isOn ? AlpinePalette.accent.opacity(0.88) : Color.white.opacity(0.16), lineWidth: 1)
                    )
                    .cornerRadius(5)
            }
            .buttonStyle(.plain)
        }
    }
}

private struct TelemetryStat: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.system(size: 9))
                .tracking(1.2)
                .foregroundColor(.white.opacity(0.62))
            Text(value)
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .monospacedDigit()
                .foregroundColor(.white)
        }
    }
}

private struct FallLineDial: View {
    let slopeDeg: Float
    let fallAngleDeg: Float

    var body: some View {
        ZStack {
            Circle()
                .fill(Color.white.opacity(0.06))
                .overlay(Circle().stroke(Color.white.opacity(0.16), lineWidth: 1))

            // Twin ski rails
            HStack(spacing: 4) {
                RoundedRectangle(cornerRadius: 1)
                    .fill(Color.white.opacity(0.38))
                    .frame(width: 1.5, height: 12)
                RoundedRectangle(cornerRadius: 1)
                    .fill(Color.white.opacity(0.38))
                    .frame(width: 1.5, height: 12)
            }

            // Rotating fall-line arrow
            Canvas { ctx, size in
                let cx = size.width * 0.5
                var shaft = Path()
                shaft.move(to: CGPoint(x: cx, y: size.height - 5))
                shaft.addLine(to: CGPoint(x: cx, y: 5))
                ctx.stroke(shaft, with: .color(AlpinePalette.cyan), style: StrokeStyle(lineWidth: 1.8, lineCap: .round))

                var head = Path()
                head.move(to: CGPoint(x: cx, y: 3))
                head.addLine(to: CGPoint(x: cx - 3, y: 7.5))
                head.addLine(to: CGPoint(x: cx + 3, y: 7.5))
                head.closeSubpath()
                ctx.fill(head, with: .color(AlpinePalette.cyan))
            }
            .opacity(slopeDeg < 1.5 ? 0.28 : 1.0)
            .rotationEffect(.degrees(slopeDeg < 1.5 ? 0 : Double(fallAngleDeg)))
        }
        .frame(width: 28, height: 28)
    }
}

private struct SurfaceChip: View {
    let label: String
    let surfId: String

    var body: some View {
        let (bg, border, fg): (Color, Color, Color) = {
            switch surfId {
            case "ice":
                return (Color(red: 66/255, green: 178/255, blue: 1).opacity(0.30),
                        Color(red: 120/255, green: 214/255, blue: 1).opacity(0.55),
                        Color(red: 200/255, green: 241/255, blue: 1))
            case "granite":
                return (Color(red: 235/255, green: 92/255, blue: 62/255).opacity(0.32),
                        Color(red: 248/255, green: 125/255, blue: 96/255).opacity(0.60),
                        Color(red: 1, green: 224/255, blue: 214/255))
            case "air":
                return (Color(red: 245/255, green: 168/255, blue: 52/255).opacity(0.32),
                        Color(red: 1, green: 204/255, blue: 112/255).opacity(0.60),
                        Color(red: 1, green: 244/255, blue: 214/255))
            default:
                return (Color(red: 140/255, green: 200/255, blue: 1).opacity(0.18),
                        Color.white.opacity(0.16),
                        Color(red: 226/255, green: 242/255, blue: 1))
            }
        }()

        return Text(label)
            .font(.system(size: 10, weight: .bold))
            .tracking(1.1)
            .foregroundColor(fg)
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(bg)
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(border, lineWidth: 1))
            .cornerRadius(5)
    }
}

private struct StanceChip: View {
    let mode: String

    var body: some View {
        let (bg, border, fg): (Color, Color, Color) = {
            switch mode {
            case "CARVE":
                return (Color(red: 16/255, green: 185/255, blue: 129/255).opacity(0.24),
                        Color(red: 52/255, green: 211/255, blue: 153/255).opacity(0.58),
                        Color(red: 209/255, green: 250/255, blue: 229/255))
            case "POLE":
                return (Color(red: 56/255, green: 189/255, blue: 248/255).opacity(0.24),
                        Color(red: 125/255, green: 211/255, blue: 252/255).opacity(0.58),
                        Color(red: 224/255, green: 242/255, blue: 254/255))
            case "SLIP":
                return (Color(red: 245/255, green: 158/255, blue: 11/255).opacity(0.24),
                        Color(red: 251/255, green: 191/255, blue: 36/255).opacity(0.58),
                        Color(red: 254/255, green: 243/255, blue: 199/255))
            case "SKID":
                return (AlpinePalette.accent.opacity(0.28),
                        Color(red: 248/255, green: 134/255, blue: 96/255).opacity(0.62),
                        Color(red: 1, green: 237/255, blue: 213/255))
            case "UPHILL":
                return (Color(red: 168/255, green: 85/255, blue: 247/255).opacity(0.22),
                        Color(red: 196/255, green: 181/255, blue: 253/255).opacity(0.50),
                        Color(red: 237/255, green: 233/255, blue: 254/255))
            case "HOLD":
                return (AlpinePalette.accent.opacity(0.22),
                        Color(red: 251/255, green: 146/255, blue: 60/255).opacity(0.55),
                        Color(red: 254/255, green: 215/255, blue: 170/255))
            default:
                return (Color.white.opacity(0.08),
                        Color.white.opacity(0.14),
                        Color(red: 226/255, green: 238/255, blue: 252/255).opacity(0.88))
            }
        }()

        return Text(mode)
            .font(.system(size: 9.5, weight: .bold))
            .tracking(1.1)
            .foregroundColor(fg)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(bg)
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(border, lineWidth: 1))
            .cornerRadius(5)
    }
}
