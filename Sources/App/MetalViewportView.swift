import AppKit
import CoreGraphics
import MetalKit
import SwiftUI

public final class InteractiveMTKView: MTKView {
    public weak var appState: AppState?
    public var renderer: MetalRenderer?

    private var trackingArea: NSTrackingArea?
    private var lastHoverPoint: NSPoint?
    private var hoverNormX: Float = 0
    private var hoverInside: Bool = false
    private var isPointerLocked: Bool = false
    private var edgeTurnTimer: Timer?

    private static let keyCodeMap: [UInt16: String] = [
        0:   "KeyA",
        1:   "KeyS",
        2:   "KeyD",
        3:   "KeyF",
        4:   "KeyH",
        5:   "KeyG",
        7:   "KeyX",
        8:   "KeyC",
        9:   "KeyV",
        12:  "KeyQ",
        13:  "KeyW",
        14:  "KeyE",
        15:  "KeyR",
        17:  "KeyT",
        18:  "Digit1",
        19:  "Digit2",
        20:  "Digit3",
        21:  "Digit4",
        22:  "Digit6",
        23:  "Digit5",
        31:  "KeyO",
        35:  "KeyP",
        36:  "Enter",
        37:  "KeyL",
        46:  "KeyM",
        49:  "Space",
        53:  "Escape",
        123: "ArrowLeft",
        124: "ArrowRight",
        125: "ArrowDown",
        126: "ArrowUp",
    ]

    public override var acceptsFirstResponder: Bool { true }

    deinit {
        edgeTurnTimer?.invalidate()
        setPointerLocked(false)
    }

    public override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.acceptsMouseMovedEvents = true
        window?.makeFirstResponder(self)

        NotificationCenter.default.removeObserver(self, name: NSWindow.didResignKeyNotification, object: nil)
        if let win = window {
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(windowDidResignKey),
                name: NSWindow.didResignKeyNotification,
                object: win
            )
        }

        if edgeTurnTimer == nil {
            edgeTurnTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.applyEdgeHoverPan(dt: 1.0 / 60.0)
                }
            }
        }
    }

    @objc private func windowDidResignKey() {
        setPointerLocked(false)
        clearAllHeldInputs()
    }

    public override func resetCursorRects() {
        discardCursorRects()
        if !isPointerLocked {
            addCursorRect(bounds, cursor: .crosshair)
        }
    }

    public override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let existing = trackingArea {
            removeTrackingArea(existing)
        }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.activeInKeyWindow, .mouseMoved, .mouseEnteredAndExited, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(area)
        trackingArea = area
    }

    public func setPointerLocked(_ locked: Bool) {
        guard isPointerLocked != locked else { return }
        isPointerLocked = locked
        if locked {
            CGAssociateMouseAndMouseCursorPosition(boolean_t(0))
            NSCursor.hide()
        } else {
            CGAssociateMouseAndMouseCursorPosition(boolean_t(1))
            NSCursor.unhide()
        }
        lastHoverPoint = nil
        window?.invalidateCursorRects(for: self)
        Task { @MainActor [weak self] in
            self?.appState?.pointerLocked = locked
        }
    }

    // MARK: - Mouse Input

    public override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        guard let appState, let renderer else { return }
        if appState.settingsOpen {
            appState.settingsOpen = false
            return
        }
        appState.dropIn()
        setPointerLocked(true)
        _ = renderer
    }

    public override func rightMouseDown(with event: NSEvent) {
        renderer?.input.grabHeld = true
    }

    public override func rightMouseUp(with event: NSEvent) {
        renderer?.input.grabHeld = false
    }

    public override func otherMouseDown(with event: NSEvent) {
        if event.buttonNumber == 2 {
            Task { @MainActor [weak self] in
                self?.appState?.recenterHead()
            }
        }
    }

    public override func mouseEntered(with event: NSEvent) {
        hoverInside = true
        lastHoverPoint = convert(event.locationInWindow, from: nil)
    }

    public override func mouseExited(with event: NSEvent) {
        hoverInside = false
        lastHoverPoint = nil
    }

    public override func mouseMoved(with event: NSEvent) {
        handlePointerMove(event)
    }

    public override func mouseDragged(with event: NSEvent) {
        handlePointerMove(event)
    }

    public override func rightMouseDragged(with event: NSEvent) {
        handlePointerMove(event)
    }

    private func handlePointerMove(_ event: NSEvent) {
        guard let appState, let renderer else { return }
        let tel = appState.telemetry
        guard tel.started, !tel.scoutActive, !tel.replayActive, !tel.pausedSim, !appState.settingsOpen else {
            lastHoverPoint = nil
            return
        }

        let sens = appState.settings.lookSens
        let ySign: Float = appState.settings.invertY ? 1.0 : -1.0

        if isPointerLocked {
            let dx = abs(event.deltaX) < 220 ? Float(event.deltaX) : 0
            let dy = abs(event.deltaY) < 220 ? Float(event.deltaY) : 0
            renderer.subsystems?.simulation.addHeadLookDelta(
                dYaw: dx * 0.0024 * sens,
                dPitch: dy * 0.0020 * sens * ySign
            )
            return
        }

        let pt = convert(event.locationInWindow, from: nil)
        hoverInside = bounds.contains(pt)
        if !hoverInside {
            lastHoverPoint = nil
            return
        }
        hoverNormX = Float(pt.x / max(1.0, bounds.width)) * 2.0 - 1.0
        if let prev = lastHoverPoint {
            let dx = Float(pt.x - prev.x)
            let dy = Float(-(pt.y - prev.y)) // top-down Y matching web clientY
            if abs(dx) < 160 && abs(dy) < 160 {
                renderer.subsystems?.simulation.addHeadLookDelta(
                    dYaw: dx * 0.0032 * sens,
                    dPitch: dy * 0.0025 * sens * ySign
                )
            }
        }
        lastHoverPoint = pt
    }

    private func applyEdgeHoverPan(dt: Float) {
        guard !isPointerLocked, hoverInside, abs(hoverNormX) > 0.80,
              let appState, let renderer else { return }
        let tel = appState.telemetry
        guard tel.started, !tel.scoutActive, !tel.replayActive, !tel.pausedSim, !appState.settingsOpen else { return }
        let edgePush = (abs(hoverNormX) - 0.80) / 0.20
        let sign: Float = hoverNormX >= 0 ? 1.0 : -1.0
        renderer.subsystems?.simulation.addHeadLookDelta(dYaw: sign * edgePush * 1.15 * dt, dPitch: 0)
    }

    // MARK: - Keyboard Input

    public override func keyDown(with event: NSEvent) {
        guard let code = Self.keyCodeMap[event.keyCode] else {
            super.keyDown(with: event)
            return
        }
        guard let appState, let renderer else { return }

        if !appState.telemetry.started &&
            ["KeyW", "ArrowUp", "Space", "Enter", "KeyF", "KeyQ", "KeyE"].contains(code) {
            appState.dropIn()
        }

        applyKeyState(code: code, isDown: true)

        if event.isARepeat { return }

        switch code {
        case "Digit1": appState.selectZone(0)
        case "Digit2": appState.selectZone(1)
        case "Digit3": appState.selectZone(2)
        case "Digit4": appState.selectZone(3)
        case "Digit5": appState.selectZone(4)
        case "Digit6": appState.selectZone(5)
        case "KeyT":   appState.cycleLight()
        case "KeyC":   appState.cycleCamera()
        case "KeyV":   appState.toggleScout()
        case "KeyP":   appState.toggleReplay()
        case "KeyL":   appState.recenterHead()
        case "KeyG":   appState.triggerAvalanche()
        case "KeyX":   appState.toggleSwitchStance()
        case "KeyO":
            appState.settingsOpen.toggle()
            if appState.settingsOpen { setPointerLocked(false) }
        case "KeyR":   appState.resetRun()
        case "KeyM":   appState.toggleMute()
        case "KeyH":   renderer.cycleHudMode()
        case "Escape":
            if isPointerLocked {
                setPointerLocked(false)
            } else if appState.settingsOpen {
                appState.settingsOpen = false
            } else {
                renderer.exitSpecialModes()
            }
        default:
            break
        }
    }

    public override func keyUp(with event: NSEvent) {
        guard let code = Self.keyCodeMap[event.keyCode] else {
            super.keyUp(with: event)
            return
        }
        applyKeyState(code: code, isDown: false)
    }

    public override func flagsChanged(with event: NSEvent) {
        let shiftDown = event.modifierFlags.contains(.shift)
        applyKeyState(code: "ShiftLeft", isDown: shiftDown)
        super.flagsChanged(with: event)
    }

    private func applyKeyState(code: String, isDown: Bool) {
        guard let renderer else { return }
        if isDown {
            renderer.input.activeKeys.insert(code)
        } else {
            renderer.input.activeKeys.remove(code)
        }
        let keys = renderer.input.activeKeys
        renderer.input.left = keys.contains("KeyA") || keys.contains("ArrowLeft")
        renderer.input.right = keys.contains("KeyD") || keys.contains("ArrowRight")
        renderer.input.tuck = keys.contains("KeyW") || keys.contains("ArrowUp")
        renderer.input.brake = keys.contains("KeyS") || keys.contains("ArrowDown") || keys.contains("ShiftLeft")
        renderer.input.poleLeft = keys.contains("KeyQ")
        renderer.input.poleRight = keys.contains("KeyE")
        renderer.input.poleBoth = keys.contains("KeyF")
        renderer.input.jumpHeld = keys.contains("Space")
        renderer.input.grabHeld = keys.contains("KeyF")
        renderer.input.cork = keys.contains("KeyQ") || keys.contains("KeyE")
    }

    private func clearAllHeldInputs() {
        guard let renderer else { return }
        renderer.input.activeKeys.removeAll()
        renderer.input.left = false
        renderer.input.right = false
        renderer.input.tuck = false
        renderer.input.brake = false
        renderer.input.poleLeft = false
        renderer.input.poleRight = false
        renderer.input.poleBoth = false
        renderer.input.jumpHeld = false
        renderer.input.grabHeld = false
        renderer.input.cork = false
    }
}

public struct MetalViewportView: NSViewRepresentable {
    @ObservedObject public var appState: AppState

    public init(appState: AppState) {
        self.appState = appState
    }

    public func makeNSView(context: Context) -> InteractiveMTKView {
        let mtkView = InteractiveMTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.clearColor = MTLClearColor(red: 0.04, green: 0.06, blue: 0.10, alpha: 1.0)
        mtkView.appState = appState

        if let renderer = MetalRenderer(mtkView: mtkView, subsystems: nil, settings: appState.settings) {
            mtkView.delegate = renderer
            mtkView.renderer = renderer
            appState.attachRenderer(renderer, mtkView: mtkView)
        } else {
            appState.loadError = "Metal device unavailable on this Mac."
            appState.loadingMessage = "Unable to initialize Metal 3D renderer."
        }
        return mtkView
    }

    public func updateNSView(_ nsView: InteractiveMTKView, context: Context) {
        nsView.appState = appState
        if appState.settingsOpen {
            nsView.setPointerLocked(false)
        }
    }
}
