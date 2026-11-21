import Combine
import Foundation
import GameController
import Metal
import MetalKit
import simd

public struct CLIOptions {
    public var captureFramePath: String?
    public var quitAfterCapture: Bool = true
    public var zoneId: String?
    public var lightId: String?
    public var camId: String?
    public var hudModeArg: String?
    public var qualityArg: String?
    public var autostart: Bool = false
    public var pauseSim: Bool = false
    public var inspectMode: String?
    public var windowWidth: CGFloat = 1440
    public var windowHeight: CGFloat = 900

    public static func parse(_ args: [String] = CommandLine.arguments) -> CLIOptions {
        var opts = CLIOptions()
        var i = 1
        while i < args.count {
            let arg = args[i]
            func nextVal() -> String? {
                guard i + 1 < args.count else { return nil }
                i += 1
                return args[i]
            }
            switch arg {
            case "--capture-frame":
                opts.captureFramePath = nextVal()
            case "--no-quit":
                opts.quitAfterCapture = false
            case "--zone":
                opts.zoneId = nextVal()
            case "--light":
                opts.lightId = nextVal()
            case "--cam":
                opts.camId = nextVal()
            case "--hud":
                opts.hudModeArg = nextVal()
            case "--quality":
                opts.qualityArg = nextVal()
            case "--autostart":
                opts.autostart = true
            case "--pause":
                opts.pauseSim = true
            case "--demo":
                opts.inspectMode = nextVal()
            case "--width":
                if let s = nextVal(), let v = Double(s), v >= 320 { opts.windowWidth = CGFloat(v) }
            case "--height":
                if let s = nextVal(), let v = Double(s), v >= 240 { opts.windowHeight = CGFloat(v) }
            default:
                break
            }
            i += 1
        }
        return opts
    }
}

public struct CompassPin: Identifiable {
    public let id: String
    public let name: String
    public let altLabel: String
    public let distLabel: String
    public let pct: Double
}

@MainActor
public final class AppState: ObservableObject {
    private static let settingsKey = "chamonix_ski_settings_v1"

    public let cli: CLIOptions

    @Published public var isLoading: Bool = true
    @Published public var loadingMessage: String = "Loading real EU-DEM alpine terrain..."
    @Published public var loadError: String? = nil

    @Published public var telemetry = HUDTelemetry()
    @Published public var settings: SimulationSettings
    @Published public var dropZones: [DropZone] = []
    @Published public var peaks: [PeakInfo] = []
    @Published public var compassPins: [CompassPin] = []

    @Published public var settingsOpen: Bool = false
    @Published public var controlsExpanded: Bool = true
    @Published public var dropMenuOpen: Bool = false
    @Published public var pointerLocked: Bool = false

    @Published public var activeTier: Int = 0 // 0=HIGH, 1=MED, 2=LOW
    @Published public var gamepadName: String? = nil
    @Published public var viewportSizePx: CGSize = CGSize(width: 1440, height: 900)

    public var renderer: MetalRenderer?
    private weak var mtkView: MTKView?

    private var gpTimer: Timer?
    private var gpPrevButtons: [Int: Bool] = [:]
    private var autoFpsAccum: Float = 0
    private var autoFpsFrames: Int = 0
    private var autoLowStreak: Int = 0
    private var autoHighStreak: Int = 0
    private var lastAutoEvalTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    public init(cli: CLIOptions = CLIOptions.parse()) {
        self.cli = cli
        self.settings = Self.loadSettings()
        if let qStr = cli.qualityArg?.lowercased(), let q = QualityMode(rawValue: qStr) {
            self.settings.quality = q
        }
        setupGameControllerObservers()
    }

    deinit {
        gpTimer?.invalidate()
    }

    // MARK: - Setup & Subsystem Boot

    public func attachRenderer(_ renderer: MetalRenderer, mtkView: MTKView) {
        self.renderer = renderer
        self.mtkView = mtkView

        renderer.onTelemetryUpdate = { [weak self] tel in
            guard let self else { return }
            self.handleTelemetry(tel)
        }

        if let subs = renderer.subsystems {
            finishBoot(subsystems: subs, renderer: renderer)
            return
        }

        guard let builder = MetalRenderer.buildSubsystems else {
            self.loadingMessage = "Waiting for subsystem integration..."
            return
        }

        let w = max(1, Int(mtkView.drawableSize.width > 0 ? mtkView.drawableSize.width : cli.windowWidth))
        let h = max(1, Int(mtkView.drawableSize.height > 0 ? mtkView.drawableSize.height : cli.windowHeight))

        let dev = renderer.device
        let queue = renderer.commandQueue
        self.loadingMessage = "Decoding real Chamonix / Mont Blanc EU-DEM topography & textures..."
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let subs = try builder(dev, queue, w, h)
                DispatchQueue.main.async {
                    guard let self, let renderer = self.renderer else { return }
                    self.finishBoot(subsystems: subs, renderer: renderer)
                }
            } catch {
                DispatchQueue.main.async {
                    self?.loadError = error.localizedDescription
                    self?.loadingMessage = "Load error: \(error.localizedDescription)"
                }
            }
        }
    }

    private func finishBoot(subsystems: SubsystemContainer, renderer: MetalRenderer) {
        self.dropZones = subsystems.terrainData.meta.dropZones
        self.peaks = subsystems.terrainData.meta.peaks

        var initialZoneIdx = 0
        if let zId = cli.zoneId, let idx = dropZones.firstIndex(where: { $0.id == zId }) {
            initialZoneIdx = idx
        }
        var initialLightIdx = 0
        if let lId = cli.lightId, let idx = LightingPresets.index(forId: lId) {
            initialLightIdx = idx
        }

        renderer.attachSubsystems(subsystems, initialZoneIdx: initialZoneIdx, initialLightIdx: initialLightIdx)

        if let cId = cli.camId, let cIdx = CameraMode.allCases.firstIndex(where: { $0.rawValue == cId }) {
            renderer.setCameraMode(cIdx)
        }
        if let hudArg = cli.hudModeArg?.lowercased() {
            if hudArg == "0" || hudArg == "hidden" { renderer.setHudMode(2) }
            else if hudArg == "raw" || hudArg == "1" { renderer.setHudMode(1) }
            else { renderer.setHudMode(0) }
        }
        if let demo = cli.inspectMode {
            if let gameSim = subsystems.simulation as? GameSimulation {
                let preset = LightingPresets.all[initialLightIdx]
                let demoRes = gameSim.runDeterministicDemo(
                    mode: demo,
                    trackBuffer: subsystems.trackBuffer,
                    particles: subsystems.particles,
                    preset: preset,
                    initialSimTime: renderer.simTime,
                    viewportHeightPx: Float(cli.windowHeight),
                    fovDeg: renderer.camera.fovDeg
                )
                if demoRes.activateScout { renderer.toggleScout(true) }
                if demoRes.activateReplay { renderer.toggleReplay(true) }
                if let bodyView = demoRes.inspectBodyView {
                    renderer.inspectBodyView = bodyView
                }
                if let msg = demoRes.toastMessage, !msg.isEmpty {
                    renderer.showToast(msg, duration: demoRes.toastDuration)
                }
            } else if demo == "body_close" || demo == "body_back" {
                renderer.inspectBodyView = demo
            }
        }

        applySettings(settings)

        if cli.autostart || cli.zoneId != nil || cli.lightId != nil || cli.camId != nil || cli.inspectMode != nil {
            renderer.setStarted(true)
        }
        if cli.pauseSim {
            renderer.setPaused(true)
        }
        if let capPath = cli.captureFramePath, !capPath.isEmpty {
            renderer.requestCapture(to: capPath, afterFrames: 8, quitAfter: cli.quitAfterCapture)
        }

        self.isLoading = false
        startGamepadPolling()
    }

    // MARK: - Telemetry & Auto-Quality

    private func handleTelemetry(_ tel: HUDTelemetry) {
        self.telemetry = tel
        updateCompass(tel: tel)

        if let mtkView {
            let ds = mtkView.drawableSize
            if ds.width > 0 && ds.height > 0 && ds != viewportSizePx {
                viewportSizePx = ds
            }
        }

        // Auto-adaptive resolution/MSAA governor
        if settings.quality == .auto && cli.captureFramePath == nil && !tel.pausedSim && tel.started {
            let now = CFAbsoluteTimeGetCurrent()
            let elapsed = Float(now - lastAutoEvalTime)
            autoFpsFrames += 1
            autoFpsAccum += tel.fps
            if elapsed >= 2.4 && autoFpsFrames > 0 {
                let avgFps = autoFpsAccum / Float(autoFpsFrames)
                autoFpsFrames = 0
                autoFpsAccum = 0
                lastAutoEvalTime = now

                if avgFps < 32 && activeTier < 2 {
                    autoLowStreak += 1
                    autoHighStreak = 0
                    if autoLowStreak >= 2 {
                        autoLowStreak = 0
                        applyQualityTier(activeTier + 1, showToastMsg: true)
                    }
                } else if avgFps > 54 && activeTier > 0 {
                    autoHighStreak += 1
                    autoLowStreak = 0
                    if autoHighStreak >= 4 {
                        autoHighStreak = 0
                        applyQualityTier(activeTier - 1, showToastMsg: true)
                    }
                } else {
                    autoLowStreak = 0
                    autoHighStreak = 0
                }
            }
        }
    }

    private func updateCompass(tel: HUDTelemetry) {
        guard tel.hudMode == 0, !peaks.isEmpty else {
            if !compassPins.isEmpty { compassPins = [] }
            return
        }
        let camYaw = -eulerYXZFromQuat(tel.cameraQuaternion).yaw
        let halfFovRad = degToRad(tel.cameraFovDeg * 0.55 * max(0.5, tel.cameraAspect) * 0.58)
        guard halfFovRad > 0.01 else { return }

        struct Candidate {
            let pk: PeakInfo
            let dist: Float
            let pct: Double
        }

        var candidates: [Candidate] = []
        let refX = tel.cameraPosition.x
        let refZ = tel.cameraPosition.z
        for pk in peaks {
            let dx = pk.x - refX
            let dz = pk.z - refZ
            let dist = hypotf(dx, dz)
            if dist < 180 { continue }
            let bearing = atan2f(dx, -dz)
            let rel = wrapAngle(bearing - camYaw)
            if abs(rel) > halfFovRad { continue }
            let pct = 50.0 + Double(rel / halfFovRad) * 44.0
            candidates.append(Candidate(pk: pk, dist: dist, pct: pct))
        }
        candidates.sort { $0.pk.elev > $1.pk.elev }

        var placed: [Double] = []
        var pins: [CompassPin] = []
        for c in candidates {
            if c.pct < 16.0 || c.pct > 84.0 { continue }
            if placed.contains(where: { abs($0 - c.pct) < 36.0 }) { continue }
            placed.append(c.pct)
            let distStr = c.dist >= 1000.0
                ? String(format: "%.1fkm", c.dist / 1000.0)
                : "\(Int(roundf(c.dist)))m"
            pins.append(CompassPin(
                id: c.pk.name,
                name: c.pk.name,
                altLabel: c.pk.altLabel,
                distLabel: distStr,
                pct: c.pct
            ))
        }
        self.compassPins = pins
    }

    // MARK: - Actions & Settings

    public func dropIn() {
        guard let renderer else { return }
        renderer.subsystems?.audio.unlock()
        if !telemetry.started {
            renderer.setStarted(true)
            renderer.subsystems?.simulation.triggerPoleStroke(side: 0)
        }
    }

    public func selectZone(_ index: Int) {
        guard let renderer else { return }
        renderer.spawnAtZone(index)
        renderer.subsystems?.audio.unlock()
        if !telemetry.started {
            renderer.setStarted(true)
        }
        dropMenuOpen = false
    }

    public func cycleLight() {
        renderer?.cycleLightPreset()
    }

    public func cycleCamera() {
        renderer?.exitSpecialModes()
        renderer?.cycleCameraMode()
    }

    public func resetRun() {
        guard let renderer else { return }
        if telemetry.scoutActive || telemetry.replayActive {
            renderer.exitSpecialModes()
        } else if telemetry.crashed {
            renderer.subsystems?.simulation.standUp()
        } else {
            renderer.spawnAtZone(telemetry.zoneIdx)
        }
    }

    public func toggleMute() {
        settings.muted.toggle()
        if !settings.muted && settings.volume <= 0.01 {
            settings.volume = 0.75
        }
        renderer?.subsystems?.audio.unlock()
        applySettings(settings)
    }

    public func toggleScout() {
        guard let renderer else { return }
        if !telemetry.started { renderer.setStarted(true) }
        renderer.toggleScout()
    }

    public func toggleReplay() {
        guard let renderer else { return }
        if !telemetry.replayActive && (renderer.subsystems?.simulation.replayFrames.count ?? 0) < 10 {
            renderer.showToast("SKI A LINE FIRST TO RECORD REPLAY", duration: 2.4)
            return
        }
        if !telemetry.started { renderer.setStarted(true) }
        renderer.toggleReplay()
    }

    public func triggerAvalanche() {
        guard let renderer else { return }
        if !settings.avalanchesEnabled {
            renderer.showToast("AVALANCHES ARE OFF — CLICK AVY: OFF TO ENABLE", duration: 2.4)
            return
        }
        renderer.exitSpecialModes()
        renderer.subsystems?.simulation.triggerAvalanche()
        renderer.showToast("SLAB AVALANCHE RELEASED — SKI SIDEWAYS!", duration: 3.2)
    }

    public func toggleAvalanches() {
        settings.avalanchesEnabled.toggle()
        if !settings.avalanchesEnabled {
            renderer?.subsystems?.particles.clearAvalanchePuffs()
        }
        applySettings(settings)
        renderer?.showToast(settings.avalanchesEnabled ? "AVALANCHES ENABLED" : "AVALANCHES OFF", duration: 2.0)
    }

    public func toggleSwitchStance() {
        guard let renderer, telemetry.grounded, !telemetry.crashed else { return }
        renderer.subsystems?.simulation.toggleSwitch()
        if let st = renderer.subsystems?.simulation.state {
            renderer.showToast(st.isSwitch ? "SWITCH STANCE (FAKIE · VIEW FLIPPED 180°)" : "FORWARD STANCE", duration: 1.8)
        }
    }

    public func recenterHead() {
        renderer?.subsystems?.simulation.resetHeadLook()
        renderer?.showToast("HEAD VIEW CENTERED", duration: 1.4)
    }

    public func setQualityMode(_ mode: QualityMode) {
        settings.quality = mode
        autoLowStreak = 0
        autoHighStreak = 0
        switch mode {
        case .high, .auto: applyQualityTier(0, showToastMsg: false)
        case .med:         applyQualityTier(1, showToastMsg: false)
        case .low:         applyQualityTier(2, showToastMsg: false)
        }
        Self.saveSettings(settings)
    }

    private func applyQualityTier(_ tier: Int, showToastMsg: Bool) {
        activeTier = max(0, min(2, tier))
        var s = settings
        if settings.quality == .auto {
            switch activeTier {
            case 0: s.quality = .high
            case 1: s.quality = .med
            default: s.quality = .low
            }
        }
        let samples = activeTier == 0 ? 4 : (activeTier == 1 ? 2 : 1)
        renderer?.subsystems?.postOptics.setMSAA(samples: samples)
        renderer?.settings = settings
        if showToastMsg {
            let names = ["HIGH", "MEDIUM", "LOW"]
            renderer?.showToast("AUTO QUALITY → \(names[activeTier])", duration: 2.6)
        }
    }

    public func applySettings(_ newSettings: SimulationSettings) {
        settings = newSettings
        renderer?.applySettings(newSettings)
        switch newSettings.quality {
        case .auto, .high: activeTier = 0
        case .med:         activeTier = 1
        case .low:         activeTier = 2
        }
        Self.saveSettings(newSettings)
    }

    public func resetSettings() {
        applySettings(SimulationSettings())
    }

    // MARK: - Persistence

    private static func loadSettings() -> SimulationSettings {
        guard let data = UserDefaults.standard.data(forKey: settingsKey),
              let decoded = try? JSONDecoder().decode(SimulationSettings.self, from: data) else {
            return SimulationSettings()
        }
        return decoded
    }

    private static func saveSettings(_ s: SimulationSettings) {
        if let data = try? JSONEncoder().encode(s) {
            UserDefaults.standard.set(data, forKey: settingsKey)
        }
    }

    // MARK: - Gamepad (GCController)

    private func setupGameControllerObservers() {
        NotificationCenter.default.addObserver(forName: .GCControllerDidConnect, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.refreshGamepadLabel()
                self?.renderer?.showToast("GAMEPAD CONNECTED · STICKS = CARVE / LOOK", duration: 2.8)
            }
        }
        NotificationCenter.default.addObserver(forName: .GCControllerDidDisconnect, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.refreshGamepadLabel()
            }
        }
        refreshGamepadLabel()
    }

    private func refreshGamepadLabel() {
        if let pad = GCController.controllers().first {
            gamepadName = (pad.vendorName ?? "Controller").uppercased()
        } else {
            gamepadName = nil
        }
    }

    private func startGamepadPolling() {
        gpTimer?.invalidate()
        gpTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.pollGamepad(dt: 1.0 / 60.0)
            }
        }
    }

    private func applyStickCurve(_ val: Float, dead: Float = 0.12) -> Float {
        let a = abs(val)
        if a <= dead { return 0 }
        let n = (a - dead) / (1.0 - dead)
        let s: Float = val >= 0 ? 1.0 : -1.0
        return s * (n * 0.45 + n * n * n * 0.55)
    }

    private func pollGamepad(dt: Float) {
        guard let renderer else { return }
        guard let pad = GCController.controllers().first?.extendedGamepad else {
            renderer.input.gamepadSteer = 0
            renderer.input.gamepadFore = 0
            renderer.input.gamepadBrake = 0
            renderer.input.gamepadCork = 0
            return
        }

        let lx = applyStickCurve(pad.leftThumbstick.xAxis.value, dead: 0.10)
        let ly = applyStickCurve(pad.leftThumbstick.yAxis.value, dead: 0.12)
        let rx = applyStickCurve(pad.rightThumbstick.xAxis.value, dead: 0.13)
        let ry = applyStickCurve(pad.rightThumbstick.yAxis.value, dead: 0.13)

        if !telemetry.started && (abs(lx) > 0.25 || abs(ly) > 0.25 || pad.buttonA.isPressed || pad.rightTrigger.value > 0.35) {
            dropIn()
        }

        renderer.input.gamepadSteer = lx
        renderer.input.gamepadFore = ly
        renderer.input.gamepadBrake = pad.leftTrigger.value > 0.08 ? pad.leftTrigger.value : 0
        renderer.input.jumpHeld = pad.buttonA.isPressed || pad.rightTrigger.value > 0.35 || renderer.input.activeKeys.contains("Space")
        renderer.input.grabHeld = pad.buttonB.isPressed || renderer.input.activeKeys.contains("KeyF")
        renderer.input.gamepadCork = (pad.rightShoulder.isPressed ? 1 : 0) - (pad.leftShoulder.isPressed ? 1 : 0)

        if (abs(rx) > 0 || abs(ry) > 0) && telemetry.started && !telemetry.scoutActive && !telemetry.replayActive && !telemetry.pausedSim {
            let ySign: Float = settings.invertY ? 1.0 : -1.0
            renderer.subsystems?.simulation.addHeadLookDelta(
                dYaw: rx * 2.5 * settings.lookSens * dt,
                dPitch: ry * 1.9 * settings.lookSens * ySign * dt
            )
        }

        func justPressed(_ id: Int, _ pressed: Bool) -> Bool {
            let prev = gpPrevButtons[id] ?? false
            gpPrevButtons[id] = pressed
            return pressed && !prev
        }

        if justPressed(3, pad.buttonY.isPressed) { cycleCamera() }
        if justPressed(2, pad.buttonX.isPressed) { toggleSwitchStance() }
        if justPressed(12, pad.dpad.up.isPressed) { cycleLight() }
        if justPressed(14, pad.dpad.left.isPressed) { selectZone(telemetry.zoneIdx - 1) }
        if justPressed(15, pad.dpad.right.isPressed) { selectZone(telemetry.zoneIdx + 1) }
        if justPressed(9, pad.buttonMenu.isPressed) { resetRun() }
        if justPressed(8, pad.buttonOptions?.isPressed ?? false) { settingsOpen.toggle() }
    }
}
