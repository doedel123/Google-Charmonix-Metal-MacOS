import Foundation
import simd

private let GRAVITY: Float = 9.81

/// Strict 120 Hz Off-Piste Freeride Skiing Physics Engine.
/// 1:1 port of `src/physics.js`:
///  - Real slope-normal gravity decomposition on EU-DEM Chamonix slopes
///  - 4-point terrain contact sampling (left ski, right ski, tip, tail)
///  - Carving vs. skidding with L2 kinetic energy conservation & twin-ski track events
///  - Switch (fakie) skiing on 180°/540° landings or `X` hop-spin
///  - Dynamic Slab Avalanche trigger & propagation
///  - Wind-lip / cornice kicker ramp launch detection
///  - 30 Hz Run Recorder (`replayFrames`) for Helicopter Replay (`P`)
///  - Pre-load crouch + ollie/pop, independent 3-axis mid-air flips/spins/corks + Mute Grab
///  - Strict landing alignment evaluation & tumble/tomahawk wipeouts
public final class SkiPhysics: SimulationSubsystem {
    public let terrainData: any TerrainQueryProtocol
    public var colliders: [PropCollider]
    public var kickers: [WindLipKicker]

    public private(set) var state = SkierState()
    public private(set) var avalanche = AvalancheState()
    public var events: [PhysicsEvent] = []
    public private(set) var replayFrames: [ReplayFrame] = []

    public var avalanchesEnabled: Bool = true {
        didSet {
            if !avalanchesEnabled {
                if avalanche.active {
                    state.camShake = 0
                    events.removeAll {
                        if case .avalancheStart = $0 { return true }
                        return false
                    }
                }
                resetAvalanche()
            }
        }
    }

    public var lookYawOff: Float {
        get { cameraController.lookYawOff }
        set { cameraController.lookYawOff = newValue }
    }

    public var lookPitchOff: Float {
        get { cameraController.lookPitchOff }
        set { cameraController.lookPitchOff = newValue }
    }

    public var headSteerEnabled: Bool = false
    public var smoothCarve: Bool = false

    public let cameraController = CameraController()

    // Internal simulation state
    private var currentTime: Float = 0
    private var recAccum: Float = 0
    private var kickerCooldown: Float = 0
    private var sprayAccum: Float = 0
    private var sluffAccum: Float = 0
    private var prevEdgeSign: Float = 0
    private var poleCooldown: Float = 0
    private var bootCreakCD: Float = 0
    private var prevCrouch: Float = 0
    private var prevSkateSin: Float = 0
    private var smoothKeySteer: Float = 0
    private var airPitchHoldMask: Int = 0
    private var whooshCD: [Int: Float] = [:]
    private var prevJumpHeld: Bool = false

    // Discrete pole-stroke impulse queue
    private var poleDriveTimer: Float = 0
    private var poleDriveDur: Float = 0.28
    private var poleDriveImpulse: Float = 0
    private var poleDriveSide: Int = 0
    private var poleStrokeCD: Float = 0
    private var lastPoleHand: Int = 0
    private var lastPoleTime: Float = -99
    private var prevPoleF: Bool = false
    private var prevPoleQ: Bool = false
    private var prevPoleE: Bool = false
    private var prevWPoling: Bool = false
    private var poleBobY: Float = 0
    private var stepPhase: Float = 0
    private var reboundUnweight: Float = 0
    private var carveOmega: Float = 0

    // Ski track previous positions
    private var prevLeftSki: SIMD2<Float>?
    private var prevRightSki: SIMD2<Float>?

    // Critically-damped landing suspension & switch hop-spin state
    public private(set) var landDisp: Float = 0
    private var landVel: Float = 0
    private var landPitch: Float = 0
    private var landPitchVel: Float = 0
    private var landRoll: Float = 0
    private var landRollVel: Float = 0
    private var scissorTarget: Float = 0
    public private(set) var switchSpinRem: Float = 0
    private var switchSpinDir: Float = 1
    private var switchHopT: Float = 0
    private var landingQuatBlend: Float = 0
    private var landingStartQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    public init(
        terrainData: any TerrainQueryProtocol,
        colliders: [PropCollider] = [],
        kickers: [WindLipKicker] = []
    ) {
        self.terrainData = terrainData
        self.colliders = colliders
        self.kickers = kickers
    }

    // MARK: - Protocol Entrypoints

    public func spawn(zone: DropZone) {
        let t = terrainData.sampleTerrain(x: zone.x, z: zone.z)
        var s = SkierState()
        s.x = zone.x
        s.z = zone.z
        s.y = t.y
        s.terrainY = t.y
        s.yaw = zone.yaw
        s.surface = t.surface
        s.slopeDeg = t.slopeDeg
        s.normal = t.normal
        s.startElev = t.y

        let gHoriz = hypotf(t.gx, t.gz)
        if gHoriz > 0.025 {
            let fYaw = atan2f(-t.gx, t.gz)
            s.fallAngleDeg = radToDeg(wrapAngle(fYaw - s.yaw))
        } else {
            s.fallAngleDeg = 0
        }
        s.stanceMode = t.slopeDeg < 5 ? "FLAT" : "GLIDE"

        camControllerReset(yaw: s.yaw, normal: t.normal, gx: t.gx, gz: t.gz)
        s.bodyQuat = cameraController.airBodyQuat
        state = s
        updateGroundedQuaternion(dt: 0)

        resetAvalanche()
        replayFrames.removeAll(keepingCapacity: true)
        currentTime = 0
        recAccum = 0
        kickerCooldown = 0
        sprayAccum = 0
        sluffAccum = 0
        prevEdgeSign = 0
        poleCooldown = 0
        bootCreakCD = 0
        prevCrouch = 0
        prevSkateSin = 0
        smoothKeySteer = 0
        airPitchHoldMask = 0
        prevJumpHeld = false
        poleDriveTimer = 0
        poleDriveImpulse = 0
        poleDriveSide = 0
        poleStrokeCD = 0
        lastPoleHand = 0
        lastPoleTime = -99
        prevPoleF = false
        prevPoleQ = false
        prevPoleE = false
        prevWPoling = false
        poleBobY = 0
        stepPhase = 0
        reboundUnweight = 0
        carveOmega = 0
        prevLeftSki = nil
        prevRightSki = nil

        landDisp = 0
        landVel = 0
        landPitch = 0
        landPitchVel = 0
        landRoll = 0
        landRollVel = 0
        scissorTarget = 0
        switchSpinRem = 0
        switchSpinDir = 1
        switchHopT = 0
        landingQuatBlend = 0
        whooshCD.removeAll(keepingCapacity: true)
        events.removeAll(keepingCapacity: true)
    }

    private func camControllerReset(yaw: Float, normal: SIMD3<Float>, gx: Float, gz: Float) {
        cameraController.resetForSpawn(yaw: yaw, normal: normal, gx: gx, gz: gz)
    }

    public func standUp() {
        let t = terrainData.sampleTerrain(x: state.x, z: state.z)
        state.y = t.y
        state.vx *= 0.1
        state.vy = 0
        state.vz *= 0.1
        state.crashed = false
        state.crashTimer = 0
        state.grounded = true
        state.isSwitch = false
        state.edgeAngle = 0
        state.crouch = 0
        state.polePlantL = 0
        state.polePlantR = 0
        state.skatePush = 0
        state.insideUnweight = 0
        state.scissorSpread = 0
        state.bowWave = 0
        state.edgeDeg = 0
        state.sideslip = 0
        state.washOut = 0
        state.carveLoadG = 1.0
        state.poleStrokePulse = 0
        state.stepBounce = 0
        poleDriveTimer = 0
        poleDriveImpulse = 0
        poleBobY = 0
        reboundUnweight = 0
        carveOmega = 0
        landDisp = 0
        landVel = 0
        landPitch = 0
        landPitchVel = 0
        landRoll = 0
        landRollVel = 0
        scissorTarget = 0
        switchSpinRem = 0
        switchSpinDir = 1
        switchHopT = 0
        landingQuatBlend = 0
        state.landImpact = 0
        state.angPitch = 0
        state.angYaw = 0
        state.angRoll = 0
        state.airPitch = 0
        state.airYaw = 0
        state.airRoll = 0
        prevLeftSki = nil
        prevRightSki = nil
        if hypotf(t.gx, t.gz) > 0.08 {
            state.yaw = atan2f(-t.gx, t.gz)
        }
        cameraController.resetHeadLook()
        updateGroundedQuaternion(dt: 0)
    }

    public func toggleSwitch() {
        _ = toggleSwitch(time: currentTime, instant: false)
    }

    @discardableResult
    public func toggleSwitch(time: Float, instant: Bool) -> Bool {
        if !state.grounded || state.crashed { return false }
        state.isSwitch.toggle()
        cameraController.switchYawBlend = 0
        if instant {
            let rem: Float = switchSpinRem > 0.001 ? switchSpinDir * switchSpinRem : 0
            state.yaw = wrapAngle(state.yaw + .pi - rem)
            switchSpinRem = 0
            switchHopT = 0
            cameraController.lookYawOff = 0
            let fwdX = sinf(state.yaw), fwdZ = -cosf(state.yaw)
            let slopeAlong = -(terrainData.sampleHeight(x: state.x + fwdX * 8, z: state.z + fwdZ * 8) - state.y) / 8.0
            let pitchBias: Float = cameraController.camMode == .chesty ? -0.22 : -0.14
            cameraController.smoothSlopePitch = clamp(-atanf(slopeAlong) * 0.48 + pitchBias, -0.68, 0.46)
            landingQuatBlend = 0
            updateGroundedQuaternion(dt: 0)
        } else {
            let dir: Float = abs(state.edgeAngle) > 0.08 ? (state.edgeAngle >= 0 ? 1 : -1) : 1
            if switchSpinRem > 0.01 {
                switchSpinRem = .pi - switchSpinRem
                switchSpinDir = -switchSpinDir
            } else {
                switchSpinRem = .pi
                switchSpinDir = dir
            }
            switchHopT = 1.0
            landVel = max(landVel, 0.45)
        }
        return true
    }

    public func triggerPoleStroke(side: Int) {
        _ = triggerPoleStroke(side: side, time: currentTime, strengthScale: 1.0)
    }

    @discardableResult
    public func triggerPoleStroke(side: Int, time: Float, strengthScale: Float = 1.0) -> Bool {
        if !state.grounded || state.crashed { return false }
        let isDouble = (side == 0)
        let minCD: Float = isDouble ? 0.44 : 0.21
        if poleStrokeCD > 0 && isDouble { return false }
        if !isDouble && poleStrokeCD > 0.16 && lastPoleHand == side { return false }

        let t = terrainData.sampleTerrain(x: state.x, z: state.z)
        let n = t.normal
        state.normal = n
        let rawFwdX = sinf(state.yaw), rawFwdZ = -cosf(state.yaw)
        let fwdDotN = rawFwdX * n.x + rawFwdZ * n.z
        var fwd = SIMD3<Float>(rawFwdX - fwdDotN * n.x, -fwdDotN * n.y, rawFwdZ - fwdDotN * n.z)
        let fwdLen = simd_length(fwd)
        fwd = fwdLen > 1e-6 ? (fwd / fwdLen) : SIMD3<Float>(0, 0, -1)
        var right = simd_cross(fwd, n)
        let rLen = simd_length(right)
        right = rLen > 1e-6 ? (right / rLen) : SIMD3<Float>(1, 0, 0)

        let pushSign: Float = state.isSwitch ? -1 : 1
        let vAlong = (state.vx * fwd.x + state.vy * fwd.y + state.vz * fwd.z) * pushSign

        let maxV: Float = isDouble ? 8.0 : 7.4
        if vAlong >= maxV { return false }
        let taper = clamp(1.0 - max(0, vAlong) / maxV, 0.08, 1.0)
        let surf = t.surface
        let surfGrip: Float
        switch surf {
        case .packed, .corn: surfGrip = 1.04
        case .crust:         surfGrip = 0.96
        case .powder:        surfGrip = 0.86
        case .ice, .granite: surfGrip = 0.60
        }

        let dtPrev = time - lastPoleTime
        let altBonus: Float = (!isDouble && lastPoleHand != 0 && lastPoleHand != side && dtPrev > 0.14 && dtPrev < 1.05) ? 1.15 : 1.0

        let baseImp = (isDouble ? 1.80 : 1.28) * surfGrip * altBonus * strengthScale
        let impulse = baseImp * powf(taper, 0.78)

        poleDriveDur = isDouble ? 0.28 : 0.22
        poleDriveTimer = poleDriveDur
        poleDriveImpulse = impulse
        poleDriveSide = side
        poleStrokeCD = minCD
        lastPoleHand = side
        lastPoleTime = time
        state.poleStrokePulse = 1.0

        let isPow = (surf == .powder)
        if isDouble {
            state.polePlantL = max(state.polePlantL, 0.95)
            state.polePlantR = max(state.polePlantR, 0.95)
            state.skatePush = max(state.skatePush, 0.85)
            let pxL = state.x - right.x * 0.44 + fwd.x * 0.46
            let pzL = state.z - right.z * 0.44 + fwd.z * 0.46
            let pxR = state.x + right.x * 0.44 + fwd.x * 0.46
            let pzR = state.z + right.z * 0.44 + fwd.z * 0.46
            events.append(.polePlant(side: -1, x: pxL, y: terrainData.sampleHeight(x: pxL, z: pzL), z: pzL, strength: 0.56 * strengthScale, powder: isPow))
            events.append(.polePlant(side: 1, x: pxR, y: terrainData.sampleHeight(x: pxR, z: pzR), z: pzR, strength: 0.56 * strengthScale, powder: isPow))
        } else {
            if side < 0 { state.polePlantL = 1.0 }
            else { state.polePlantR = 1.0 }
            let sf = Float(side)
            let px = state.x + right.x * sf * 0.48 + fwd.x * 0.52
            let pz = state.z + right.z * sf * 0.48 + fwd.z * 0.52
            events.append(.polePlant(side: side, x: px, y: terrainData.sampleHeight(x: px, z: pz), z: pz, strength: 0.62 * strengthScale, powder: isPow))
        }
        return true
    }

    public func triggerAvalanche() {
        _ = triggerAvalanche(uphillOffset: -22, customWidth: 30, force: false)
    }

    @discardableResult
    public func triggerAvalanche(uphillOffset: Float = -22, customWidth: Float = 30, force: Bool = false) -> Bool {
        if !avalanchesEnabled && !force { return false }
        let t = terrainData.sampleTerrain(x: state.x, z: state.z)
        var dx = -t.gx
        var dz = -t.gz
        let mag = hypotf(dx, dz)
        if mag > 0.04 {
            dx /= mag
            dz /= mag
        } else {
            dx = sinf(state.yaw)
            dz = -cosf(state.yaw)
        }
        avalanche.active = true
        avalanche.strength = 1.0
        avalanche.originX = state.x + dx * uphillOffset
        avalanche.originZ = state.z + dz * uphillOffset
        avalanche.dirX = dx
        avalanche.dirZ = dz
        avalanche.frontDist = 4.0
        avalanche.width = customWidth
        avalanche.speed = 16.5 + clamp((t.slopeDeg - 20.0) * 0.45, 0.0, 10.0)
        avalanche.age = 0
        avalanche.proximity = 0.65
        avalanche.engulfed = false
        avalanche.cooldown = 42.0

        state.camShake = max(state.camShake, 0.72)
        events.append(.avalancheStart(
            x: avalanche.originX,
            y: terrainData.sampleHeight(x: avalanche.originX, z: avalanche.originZ),
            z: avalanche.originZ,
            width: avalanche.width
        ))
        return true
    }

    public func resetAvalanche() {
        avalanche.active = false
        avalanche.strength = 0
        avalanche.frontDist = 0
        avalanche.age = 0
        avalanche.proximity = 0
        avalanche.engulfed = false
        avalanche.cooldown = 0
    }

    public func addHeadLookDelta(dYaw: Float, dPitch: Float) {
        cameraController.addHeadLookDelta(dYaw: dYaw, dPitch: dPitch, state: state)
    }

    public func resetHeadLook() {
        cameraController.resetHeadLook()
    }

    public func clampLookOffsets(mode: CameraMode? = nil) {
        cameraController.clampLookOffsets(mode: mode, state: state)
    }

    public func setCamMode(_ mode: CameraMode) {
        cameraController.setCamMode(mode, state: state)
    }

    public func applyCamera(camera: inout CameraState, mode: CameraMode, time: Float, dt: Float) {
        cameraController.applyCamera(
            camera: &camera,
            mode: mode,
            time: time,
            dt: dt,
            state: state,
            terrainData: terrainData,
            colliders: colliders,
            poleBobY: poleBobY,
            switchHopT: switchHopT,
            landDisp: landDisp,
            landPitch: landPitch,
            landRoll: landRoll
        )
    }

    public func enforceCameraClearance(camera: inout CameraState, minClearance: Float, lockRoll: Bool) {
        cameraController.enforceCameraClearance(
            camera: &camera,
            terrainData: terrainData,
            colliders: colliders,
            minClearance: minClearance,
            checkProps: lockRoll
        )
    }

    public func clearEvents() {
        events.removeAll(keepingCapacity: true)
    }

    // MARK: - Helpers for Deterministic Demos & Direct Overrides

    public func setPosition(x: Float, z: Float, yawDeg: Float? = nil) {
        let t = terrainData.sampleTerrain(x: x, z: z)
        state.x = x
        state.z = z
        state.y = t.y
        state.terrainY = t.y
        state.vx = 0; state.vy = 0; state.vz = 0; state.speed = 0
        state.grounded = true
        state.crashed = false
        state.isSwitch = false
        state.edgeAngle = 0
        state.edgeDeg = 0
        state.slipAngle = 0
        state.sideslip = 0
        state.washOut = 0
        prevLeftSki = nil
        prevRightSki = nil
        state.surface = t.surface
        state.slopeDeg = t.slopeDeg
        state.normal = t.normal
        cameraController.smoothNormal = t.normal
        if let yawDeg {
            state.yaw = degToRad(yawDeg)
        }
        updateGroundedQuaternion(dt: 0)
    }

    public func mutateState(_ block: (inout SkierState) -> Void) {
        block(&state)
    }

    public func setSmoothNormal(_ n: SIMD3<Float>) {
        cameraController.smoothNormal = n
    }

    public func syncGroundedQuaternion(dt: Float = 0) {
        updateGroundedQuaternion(dt: dt)
    }

    public func applyLandingStompState(landDisp disp: Float, landPitch pitch: Float, landRoll roll: Float, scissorTarget scissor: Float) {
        self.landDisp = disp
        self.landVel = 0
        self.landPitch = pitch
        self.landRoll = roll
        self.scissorTarget = scissor
    }

    public func triggerJumpPreset(charge: Float = 0.85) {
        state.popCharge = charge
        prevJumpHeld = true
    }

    public func advanceAvalancheDemoStep(dt: Float) {
        avalanche.age += dt
        avalanche.frontDist += 20.0 * dt
        avalanche.width = min(46.0, avalanche.width + dt * 1.4)
    }

    // MARK: - 120 Hz Fixed Physics Step

    public func step(dt: Float, time: Float, input: InputState) {
        currentTime = time
        let useSmoothCarve = smoothCarve || input.smoothCarve
        let useHeadSteer = headSteerEnabled || input.headSteerEnabled

        if input.lookDeltaYaw != 0 || input.lookDeltaPitch != 0 {
            addHeadLookDelta(dYaw: input.lookDeltaYaw, dPitch: input.lookDeltaPitch)
        }

        let keys = input.activeKeys
        let steerL: Float = (input.left || keys.contains("KeyA") || keys.contains("ArrowLeft")) ? 1 : 0
        let steerR: Float = (input.right || keys.contains("KeyD") || keys.contains("ArrowRight")) ? 1 : 0
        let fwdIn: Float  = (input.tuck || keys.contains("KeyW") || keys.contains("ArrowUp")) ? 1 : 0
        let backIn: Float = (keys.contains("KeyS") || keys.contains("ArrowDown")) ? 1 : 0
        let corkL: Float  = (input.poleLeft || keys.contains("KeyQ")) ? 1 : 0
        let corkR: Float  = (input.poleRight || keys.contains("KeyE")) ? 1 : 0
        let brakeKey: Float = (input.brake || keys.contains("ShiftLeft") || keys.contains("ShiftRight")) ? 1 : 0
        let brake = min(1.0, max(brakeKey, input.gamepadBrake))

        let rawKeySteer = steerR - steerL
        if useSmoothCarve {
            let rate: Float = rawKeySteer != 0 ? 6.2 : 9.5
            smoothKeySteer += (rawKeySteer - smoothKeySteer) * min(1.0, dt * rate)
            if abs(smoothKeySteer) < 0.005 && rawKeySteer == 0 { smoothKeySteer = 0 }
        } else {
            smoothKeySteer = rawKeySteer
        }

        let steer = clamp(smoothKeySteer + input.mouseSteer + input.gamepadSteer, -1.35, 1.35)
        let foreIn = clamp((fwdIn - backIn) + input.gamepadFore, -1.0, 1.0)
        let corkIn = clamp((corkR - corkL) + input.gamepadCork, -1.0, 1.0)
        let jumpNow = input.jumpHeld || keys.contains("Space")
        let grabNow = input.grabHeld || keys.contains("KeyF")
        state.grabbing = grabNow

        if kickerCooldown > 0 { kickerCooldown = max(0, kickerCooldown - dt) }

        let t = terrainData.sampleTerrain(x: state.x, z: state.z)
        state.terrainY = t.y
        state.surface = t.surface
        state.slopeDeg = t.slopeDeg
        state.normal = t.normal
        cameraController.smoothNormal = simd_normalize(lerp(cameraController.smoothNormal, state.normal, min(1.0, dt * 16.0)))

        stepAvalanche(dt: dt, time: time)

        // Decay transient visual effects & integrate critically-damped landing suspension ODE
        state.faceShotIntensity = max(0, state.faceShotIntensity - dt * 2.40)
        state.camShake = max(0, state.camShake - dt * 3.2)
        state.sluffTrigger = max(0, state.sluffTrigger - dt * 1.5)

        let fDisp = -28.0 * landDisp - 8.9 * landVel
        landVel += fDisp * dt
        landDisp = clamp(landDisp + landVel * dt, -0.05, 0.44)
        if abs(landDisp) < 0.0004 && abs(landVel) < 0.002 {
            landDisp = 0
            landVel = 0
        }
        state.landImpact = clamp(max(0, landDisp) / 0.28, 0, 1)

        let fPitch = -36.0 * landPitch - 10.2 * landPitchVel
        landPitchVel += fPitch * dt
        landPitch = clamp(landPitch + landPitchVel * dt, -0.26, 0.14)
        let fRoll = -42.0 * landRoll - 9.8 * landRollVel
        landRollVel += fRoll * dt
        landRoll = clamp(landRoll + landRollVel * dt, -0.24, 0.24)

        if switchSpinRem > 0.0005 && state.grounded && !state.crashed {
            let dSpin = min(switchSpinRem, max(dt * 7.2, switchSpinRem * min(1.0, dt * 13.5)))
            state.yaw = wrapAngle(state.yaw + switchSpinDir * dSpin)
            switchSpinRem = max(0, switchSpinRem - dSpin)
            cameraController.lookYawOff *= expf(-dt * 12.0)
        } else if !state.grounded || state.crashed {
            switchSpinRem = 0
        }
        if switchHopT > 0 {
            switchHopT = max(0, switchHopT - dt * 3.85)
        }

        // Wipeout tumble
        if state.crashed {
            state.crashTimer += dt
            let n = state.normal
            let gDotN = -GRAVITY * n.y
            state.vx += (-n.x * gDotN) * dt * 0.45
            state.vz += (-n.z * gDotN) * dt * 0.45
            let drag = expf(-dt * 2.15)
            state.vx *= drag
            state.vz *= drag
            state.x += state.vx * dt
            state.z += state.vz * dt
            let nt = terrainData.sampleTerrain(x: state.x, z: state.z)
            state.y = nt.y + 0.35
            state.speed = hypotf(state.vx, state.vz)
            let spinScale = min(1.0, state.speed / 6.0)
            let qStep = quatFromEulerXYZ(
                x: dt * 4.8 * spinScale,
                y: dt * 2.6 * spinScale,
                z: dt * 5.2 * spinScale
            )
            state.quat = simd_normalize(state.quat * qStep)
            state.sprayRate = spinScale * 1.4
            if spinScale > 0.15 {
                events.append(.crashTrail(x: state.x, y: nt.y, z: state.z, speed: state.speed))
            }
            if state.crashTimer > 2.35 && state.speed < 2.2 {
                standUp()
            }
            recAccum += dt
            if recAccum >= 1.0 / 30.0 {
                recAccum -= 1.0 / 30.0
                recordFrame()
            }
            prevJumpHeld = jumpNow
            return
        }

        // Clamp to playable core massif margin (±5850 m)
        let bound: Float = 5850.0
        if abs(state.x) > bound || abs(state.z) > bound {
            state.x = clamp(state.x, -bound, bound)
            state.z = clamp(state.z, -bound, bound)
            state.vx *= -0.25
            state.vz *= -0.25
        }

        let poleL = (corkL > 0 || input.gamepadCork < -0.5)
        let poleR = (corkR > 0 || input.gamepadCork > 0.5)
        let altHeld = input.freeLookHeld || keys.contains("AltLeft") || keys.contains("AltRight")

        if state.grounded {
            stepGrounded(
                dt: dt,
                t: t,
                steer: steer,
                foreIn: foreIn,
                brake: brake,
                jumpNow: jumpNow,
                poleLIn: poleL,
                poleRIn: poleR,
                poleFIn: grabNow || input.poleBoth,
                altHeld: altHeld,
                useHeadSteer: useHeadSteer,
                time: time
            )
        } else {
            stepAirborne(
                dt: dt,
                t: t,
                steer: steer,
                foreIn: foreIn,
                corkIn: corkIn,
                grabNow: grabNow,
                altHeld: altHeld,
                useHeadSteer: useHeadSteer,
                time: time
            )
        }

        checkCollisions(time: time)

        state.verticalDropped = max(state.verticalDropped, state.startElev - state.y)
        let kmh = state.speed * 3.6
        if kmh > state.maxSpeedKmh { state.maxSpeedKmh = kmh }
        if kmh > 32 { state.breathing = min(1.0, state.breathing + dt * 0.16) }
        else { state.breathing = max(0, state.breathing - dt * 0.055) }

        if state.speed > 0.25 || !state.grounded || !replayFrames.isEmpty {
            recAccum += dt
            if recAccum >= 1.0 / 30.0 {
                recAccum -= 1.0 / 30.0
                recordFrame()
            }
        }

        prevJumpHeld = jumpNow
    }

    // MARK: - Grounded Physics

    private func stepGrounded(
        dt: Float,
        t: TerrainSample,
        steer: Float,
        foreIn: Float,
        brake: Float,
        jumpNow: Bool,
        poleLIn: Bool,
        poleRIn: Bool,
        poleFIn: Bool,
        altHeld: Bool,
        useHeadSteer: Bool,
        time: Float
    ) {
        let surf = t.surface
        let n = state.normal
        let keySteer = steer

        let fp = sampleFourPoint(x: state.x, z: state.z, yaw: state.yaw)
        state.rollContact = fp.rollContact
        state.pitchContact = fp.pitchContact

        state.foreAft += (foreIn - state.foreAft) * min(1.0, dt * 8.0)
        if jumpNow {
            state.popCharge = min(1.0, state.popCharge + dt * 2.2)
        }

        var relHead = wrapAngle(cameraController.lookYawOff)
        let curSpeed = hypotf(hypotf(state.vx, state.vy), state.vz)

        if useHeadSteer && !altHeld && curSpeed > 6.5 && abs(relHead) > 2.05 {
            cameraController.lookYawOff = wrapAngle(cameraController.lookYawOff * expf(-dt * 4.5))
            relHead = wrapAngle(cameraController.lookYawOff)
        }

        var headSteer: Float = 0
        if useHeadSteer && !altHeld && switchSpinRem <= 0.05 {
            if foreIn > 0.1 && curSpeed < 5.0 && abs(relHead) > 0.025 {
                let stepTurn = (relHead >= 0 ? 1 : -1) * min(abs(relHead), 3.6 * dt)
                state.yaw += stepTurn
                cameraController.lookYawOff = wrapAngle(cameraController.lookYawOff - stepTurn)
                relHead = wrapAngle(cameraController.lookYawOff)
            }
            if curSpeed >= 0.85 || foreIn > 0.1 {
                let dead: Float = 0.025
                if abs(relHead) > dead {
                    let effRel = relHead - (relHead >= 0 ? 1 : -1) * dead
                    headSteer = clamp(effRel / 0.52, -1.10, 1.10)
                }
            }
        }

        // --- 1. Exact Fall-Line, Steepness & Ski-Heading Geometry ---
        let grav = SIMD3<Float>(0, -GRAVITY, 0)
        let gNormal = simd_dot(grav, n)
        let normalWeight = max(0.5, -gNormal)
        let gravTan = grav - n * gNormal
        let slopeMagTan = simd_length(gravTan)
        let sinSlope = min(1.0, slopeMagTan / GRAVITY)

        let rawFwdX = sinf(state.yaw)
        let rawFwdZ = -cosf(state.yaw)
        let fwdDotN = rawFwdX * n.x + rawFwdZ * n.z
        var fwd = SIMD3<Float>(rawFwdX - fwdDotN * n.x, -fwdDotN * n.y, rawFwdZ - fwdDotN * n.z)
        let fwdLen = simd_length(fwd)
        fwd = fwdLen > 1e-6 ? (fwd / fwdLen) : SIMD3<Float>(0, 0, -1)
        var right = simd_cross(fwd, n)
        let rLen = simd_length(right)
        right = rLen > 1e-6 ? (right / rLen) : SIMD3<Float>(1, 0, 0)

        let gFwd = simd_dot(gravTan, fwd)
        let gLat = simd_dot(gravTan, right)
        let hasSlope = slopeMagTan > 0.16
        let fallAngle: Float = hasSlope ? atan2f(gLat, gFwd) : 0.0
        let sinFall: Float = hasSlope ? (gLat / slopeMagTan) : 0.0
        let cosFall: Float = hasSlope ? (gFwd / slopeMagTan) : 1.0
        let uphillSign: Float = abs(gLat) > 0.18 ? (gLat > 0 ? -1 : 1) : 0
        state.fallAngleDeg = radToDeg(fallAngle)

        let vFwdPre = state.vx * fwd.x + state.vy * fwd.y + state.vz * fwd.z
        _ = state.vx * right.x + state.vy * right.y + state.vz * right.z

        // --- 2. Poling on Flat Ground, Gentle Slopes & Short Climbs ---
        if poleStrokeCD > 0 { poleStrokeCD = max(0, poleStrokeCD - dt) }
        state.poleStrokePulse = max(0, state.poleStrokePulse - dt * 2.8)

        let qJust = poleLIn && !prevPoleQ
        let eJust = poleRIn && !prevPoleE
        let fJust = poleFIn && !prevPoleF
        prevPoleQ = poleLIn
        prevPoleE = poleRIn
        prevPoleF = poleFIn

        if brake <= 0 && !jumpNow {
            if fJust || (poleFIn && poleStrokeCD <= 0 && curSpeed < 7.8) {
                _ = triggerPoleStroke(side: 0, time: time, strengthScale: 1.06)
            } else if qJust || (poleLIn && !poleRIn && poleStrokeCD <= 0 && curSpeed < 7.2) {
                _ = triggerPoleStroke(side: -1, time: time, strengthScale: 1.0)
            } else if eJust || (poleRIn && !poleLIn && poleStrokeCD <= 0 && curSpeed < 7.2) {
                _ = triggerPoleStroke(side: 1, time: time, strengthScale: 1.0)
            }
        }

        let isFlatOrGentle = gFwd < 2.25
        let wPoleSpeedCap: Float = isFlatOrGentle ? 7.6 : 5.0
        let wPolingActive = foreIn > 0.14 && brake <= 0 && !jumpNow && (vFwdPre * (state.isSwitch ? -1 : 1) < wPoleSpeedCap)
        if wPolingActive {
            state.skatePush = min(1.0, state.skatePush + dt * 5.8)
            if !prevWPoling {
                state.skatePhase = 0.18
                _ = triggerPoleStroke(side: 0, time: time, strengthScale: 0.96 * foreIn)
            }
            let cycleRate = 9.4 + min(2.2, curSpeed * 0.24)
            state.skatePhase += dt * cycleRate
            let sinP = sinf(state.skatePhase)
            if prevSkateSin <= 0 && sinP > 0 && poleStrokeCD <= 0 {
                _ = triggerPoleStroke(side: 0, time: time, strengthScale: 0.96 * foreIn)
            }
            prevSkateSin = sinP
        } else if poleDriveTimer > 0 && poleDriveSide == 0 {
            state.skatePush = min(1.0, state.skatePush + dt * 6.5)
            state.skatePhase += dt * 10.2
            prevSkateSin = sinf(state.skatePhase)
        } else {
            state.skatePush = max(0, state.skatePush - dt * 4.8)
            prevSkateSin = 0
        }
        prevWPoling = wPolingActive

        var poleBobTarget: Float = 0
        if poleDriveTimer > 0 {
            let dur = max(0.05, poleDriveDur)
            let u = 1.0 - (poleDriveTimer / dur)
            let weight = sinf(u * .pi) * (.pi * 0.5)
            let dv = (poleDriveImpulse / dur) * weight * dt
            let pushSign: Float = state.isSwitch ? -1 : 1
            state.vx += fwd.x * pushSign * dv
            state.vy += fwd.y * pushSign * dv
            state.vz += fwd.z * pushSign * dv
            if poleDriveSide != 0 {
                let yawNudge = Float(-poleDriveSide) * 0.032 * (weight / dur) * dt
                state.yaw += yawNudge
            }
            poleBobTarget = (poleDriveSide == 0 ? -0.044 : -0.022) * sinf(u * .pi)
            poleDriveTimer = max(0, poleDriveTimer - dt)
        }
        poleBobY += (poleBobTarget - poleBobY) * min(1.0, dt * 18.0)

        let baseTuck: Float = jumpNow ? 0.92 : (wPolingActive ? 0.24 : (foreIn > 0.2 ? 0.62 : (brake > 0 ? 0.36 : 0.12)))
        let tuckTarget = max(baseTuck, state.landImpact * 0.36)
        let risingDuringLand = tuckTarget < state.crouch && (landVel > 0.05 || state.landImpact > 0.14)
        state.crouch += (tuckTarget - state.crouch) * min(1.0, dt * (risingDuringLand ? 2.2 : 11.0))

        if bootCreakCD > 0 { bootCreakCD = max(0, bootCreakCD - dt) }
        let crouchDelta = abs(state.crouch - prevCrouch)
        if curSpeed < 1.3 && crouchDelta > 0.0045 && bootCreakCD <= 0 {
            events.append(.bootCreak(strength: min(1.0, crouchDelta * 75.0)))
            bootCreakCD = 0.52
        }
        prevCrouch = state.crouch

        // --- 3. Authentic Edging Dynamics (Fall Line * Steepness * Speed) ---
        let combinedSteer = clamp(keySteer + headSteer, -1.35, 1.35)
        let edgeVisualSteer = combinedSteer
        let rawSteerInput = clamp(edgeVisualSteer, -1.0, 1.0)

        let gentleUphillBias: Float = hasSlope
            ? (-sinFall * clamp(sinSlope * 0.64, 0.0, 0.22) * (1.0 - min(1.0, brake)))
            : 0.0

        let speedEdgeCap = clamp(0.30 + curSpeed * 0.048 + sinSlope * 0.44, 0.30, 1.00)
        let pushingUphill = hasSlope && (rawSteerInput * gLat < -0.15)
        let steepAngBonus: Float = pushingUphill ? sinSlope * 0.18 : 0.0
        let maxEdge = min(1.04, speedEdgeCap + steepAngBonus)

        let targetEdge: Float
        if brake > 0 {
            let stopSide: Float = abs(rawSteerInput) > 0.12
                ? (rawSteerInput >= 0 ? 1 : -1)
                : (uphillSign != 0 ? uphillSign : (abs(state.edgeAngle) > 0.08 ? (state.edgeAngle >= 0 ? 1 : -1) : 1))
            targetEdge = stopSide * min(1.02, (curSpeed > 0.8 ? 0.78 : 0.62) + sinSlope * 0.34)
        } else if abs(rawSteerInput) > 0.04 {
            targetEdge = clamp(rawSteerInput * maxEdge, -1.04, 1.04)
        } else {
            targetEdge = gentleUphillBias
        }

        let prevEdge = state.edgeAngle
        let isReleasing = abs(targetEdge) < abs(state.edgeAngle)
        let edgeRate: Float = brake > 0 ? 9.2 : (isReleasing ? 4.8 : (6.6 + min(2.4, curSpeed * 0.10)))
        state.edgeAngle += (targetEdge - state.edgeAngle) * min(1.0, dt * edgeRate)
        if prevEdge * state.edgeAngle < -0.002 && abs(prevEdge - state.edgeAngle) > 0.01 && curSpeed > 5.0 {
            reboundUnweight = min(1.0, curSpeed / 16.0)
        }
        reboundUnweight = max(0, reboundUnweight - dt * 3.8)

        state.edgeDeg = state.edgeAngle * 64.0
        if !hasSlope || abs(state.edgeAngle) < 0.08 {
            state.uphillEdge = 0
        } else {
            let eSign: Float = state.edgeAngle >= 0 ? 1 : -1
            state.uphillEdge = (eSign == uphillSign) ? 1 : -1
        }
        let edgeSignVal: Float = state.edgeAngle > 0 ? 1 : (state.edgeAngle < 0 ? -1 : 0)
        state.turnLead += (edgeSignVal * min(1.0, abs(state.edgeAngle)) - state.turnLead) * min(1.0, dt * 7.5)

        let targetUnweight = clamp(state.edgeAngle * min(1.0, 0.25 + curSpeed / 12.0), -1.0, 1.0)
        state.insideUnweight += (targetUnweight - state.insideUnweight) * min(1.0, dt * 10.0)
        scissorTarget *= expf(-dt * 2.6)
        state.scissorSpread += (scissorTarget - state.scissorSpread) * min(1.0, dt * 14.0)

        state.polePlantL = max(0, state.polePlantL - dt * 3.0)
        state.polePlantR = max(0, state.polePlantR - dt * 3.0)
        if poleCooldown > 0 { poleCooldown = max(0, poleCooldown - dt) }
        if abs(state.edgeAngle) < 0.08 && abs(targetEdge) < 0.10 {
            prevEdgeSign = 0
        }
        if curSpeed > 4.0 && brake <= 0 && state.skatePush < 0.25 && poleDriveTimer <= 0 && abs(targetEdge) > 0.28 {
            let edgeSign: Float = targetEdge >= 0 ? 1 : -1
            if edgeSign != prevEdgeSign && poleCooldown <= 0 {
                if edgeSign > 0 { state.polePlantR = 1.0 }
                else { state.polePlantL = 1.0 }
                prevEdgeSign = edgeSign
                poleCooldown = 0.44
                let px = state.x + right.x * edgeSign * 0.56 + fwd.x * 0.65
                let pz = state.z + right.z * edgeSign * 0.56 + fwd.z * 0.65
                let py = terrainData.sampleHeight(x: px, z: pz)
                events.append(.polePlant(
                    side: Int(edgeSign),
                    x: px, y: py, z: pz,
                    strength: min(1.0, 0.45 + curSpeed / 22.0),
                    powder: surf == .powder
                ))
            }
        }

        // --- 4. True Parabolic Sidecut Carving + Fall-Line Pull + Rotary Skid/Step ---
        let absEdge = abs(state.edgeAngle)
        let edgeRad = state.edgeAngle * 1.12
        let absSinEdge = abs(sinf(edgeRad))

        let radiusMult = 1.0 - state.foreAft * (state.foreAft >= 0 ? 0.20 : 0.26)
        let rEff = 15.5 * clamp(radiusMult, 0.76, 1.32)

        let absSlipPre = abs(state.slipAngle)
        let railClean = 1.0 - smoothstep(0.22, 1.08, absSlipPre) * 0.56

        let maxLatAccel = normalWeight * (0.42 + 1.88 * surf.grip * powf(absSinEdge, 0.85))
        let rawCurv = (sinf(edgeRad) * (1.0 + 0.44 * absSinEdge)) / rEff
        let vCarve = clamp(vFwdPre, -24, 36)
        let desiredOmegaCarve = vCarve * rawCurv * railClean
        let desiredCentrip = abs(vCarve * desiredOmegaCarve)
        let washOut: Float = (desiredCentrip > maxLatAccel && curSpeed > 5.0)
            ? clamp((desiredCentrip - maxLatAccel) / max(3.5, maxLatAccel), 0, 1)
            : 0.0
        state.washOut += (washOut - state.washOut) * min(1.0, dt * 10.0)
        let satScale: Float = (desiredCentrip > maxLatAccel && desiredCentrip > 0.01) ? (maxLatAccel / desiredCentrip) : 1.0
        let surfCarveBite = (0.36 + 0.64 * surf.grip) * (0.45 + 0.55 * satScale)
        let omegaCarve = desiredOmegaCarve * surfCarveBite
        carveOmega = omegaCarve

        let lowSpdMask = 1.0 - smoothstep(0.8, 4.8, curSpeed)
        let stepSteer = keySteer
        let omegaStep: Float = switchSpinRem > 0.05 ? 0 : (stepSteer * 1.12 * lowSpdMask)
        if lowSpdMask > 0.25 && abs(stepSteer) > 0.15 {
            stepPhase += dt * 11.5
            state.stepBounce = abs(sinf(stepPhase)) * lowSpdMask
        } else {
            state.stepBounce *= expf(-dt * 12.0)
        }

        var omegaRotary: Float = 0
        if brake > 0 && curSpeed > 0.8 {
            let slashDir: Float = abs(rawSteerInput) > 0.10
                ? (rawSteerInput >= 0 ? 1 : -1)
                : (uphillSign != 0 ? uphillSign : (abs(state.edgeAngle) > 0.08 ? (state.edgeAngle >= 0 ? 1 : -1) : 1))
            let alreadyPerp = abs(rawSteerInput) <= 0.10 && hasSlope && abs(abs(fallAngle) - .pi * 0.5) < 0.20
            let brakeTurnGain: Float = alreadyPerp ? 0.18 : 1.0
            omegaRotary = slashDir * (1.85 + 0.45 * surf.grip) * brakeTurnGain * min(1.0, curSpeed / 3.0)
        } else if curSpeed > 1.2 && abs(rawSteerInput) > 0.05 {
            let surfSmear: Float = surf == .powder ? 0.34 : (surf == .corn ? 0.24 : 0.18)
            let dropAssist: Float = (hasSlope && rawSteerInput * gLat > 0.2) ? (1.0 + sinSlope * 0.45) : 1.0
            omegaRotary = rawSteerInput * surfSmear * dropAssist * min(1.0, curSpeed / 5.0)
        }

        let flatMask = 1.0 - smoothstep(0.12, 0.52, absEdge)
        let slopePullMask = smoothstep(5.5, 22.0, t.slopeDeg)
        let movePullMask = smoothstep(0.5, 4.0, curSpeed)
        let fallDir = state.isSwitch ? -sinFall : sinFall
        let omegaFallLine: Float = (hasSlope && brake <= 0 && switchSpinRem <= 0.05)
            ? fallDir * sinSlope * flatMask * slopePullMask * movePullMask * (0.34 + 0.22 * min(1.0, curSpeed / 14.0))
            : 0.0

        let crossCatch: Float = (surf == .ice || surf == .crust)
            ? clamp(fp.rollContact * 0.24, -0.35, 0.35) * min(1.0, curSpeed / 16.0)
            : 0.0

        var dYawHead: Float = 0
        if headSteer != 0 {
            let headRate: Float = brake > 0 ? 2.2 : (0.9 + 0.85 * min(1.0, curSpeed / 14.0))
            dYawHead = headSteer * headRate * (0.6 + 0.4 * surf.grip) * dt
            if abs(dYawHead) > abs(relHead) { dYawHead = relHead }
        }

        let totalOmega = omegaCarve + omegaStep + omegaRotary + omegaFallLine + crossCatch * 0.38
        state.yaw += totalOmega * dt + dYawHead
        cameraController.lookYawOff = wrapAngle(cameraController.lookYawOff - dYawHead)
        let steerEff = edgeVisualSteer

        // Recompute tangent basis after yaw update
        let nRawFwdX = sinf(state.yaw), nRawFwdZ = -cosf(state.yaw)
        let nFwdDotN = nRawFwdX * n.x + nRawFwdZ * n.z
        fwd = SIMD3<Float>(nRawFwdX - nFwdDotN * n.x, -nFwdDotN * n.y, nRawFwdZ - nFwdDotN * n.z)
        let nFwdLen = simd_length(fwd)
        fwd = nFwdLen > 1e-6 ? (fwd / nFwdLen) : SIMD3<Float>(0, 0, -1)
        right = simd_cross(fwd, n)
        let nrLen = simd_length(right)
        right = nrLen > 1e-6 ? (right / nrLen) : SIMD3<Float>(1, 0, 0)

        // --- 5. Gravity Projection, Lateral Edge Bite, Sideslip & L2 Carve-vs-Skid Energy ---
        var vel = SIMD3<Float>(state.vx, state.vy, state.vz)
        vel += gravTan * dt
        vel -= n * simd_dot(vel, n)

        let vFwd = simd_dot(vel, fwd)
        let vLat = simd_dot(vel, right)
        let speed = hypotf(vFwd, vLat)

        if switchSpinRem <= 0.05 && speed > 2.4 {
            if state.isSwitch && vFwd > 2.2 { state.isSwitch = false }
            else if !state.isSwitch && vFwd < -5.8 && abs(vLat) < 1.8 { state.isSwitch = true }
        }

        state.slipAngle = speed > 0.55 ? atan2f(vLat, max(0.28, abs(vFwd))) : 0
        let absSlip = abs(state.slipAngle)

        let centripAccel = abs(vFwd * omegaCarve)
        let totalNormal = sqrtf(normalWeight * normalWeight + centripAccel * centripAccel)
        state.carveLoadG += (clamp(totalNormal / GRAVITY, 0.65, 2.8) - state.carveLoadG) * min(1.0, dt * 8.0)

        let slideSign: Float = abs(vLat) > 0.10 ? (vLat >= 0 ? 1 : -1) : (gLat > 0 ? 1 : (gLat < 0 ? -1 : 0))
        let edgeAgainstSlide = max(0.0, -state.edgeAngle * slideSign)
        let edgeWithSlide = max(0.0, state.edgeAngle * slideSign)
        let sinBite = sinf(min(1.16, edgeAgainstSlide * 1.12))

        let baseLatMu: Float = surf == .ice ? 0.09 : (surf == .crust ? 0.15 : (surf == .powder ? 0.22 : 0.19))
        let dynPressure = 1.0 + min(0.55, (state.carveLoadG - 1.0) * 0.42)
        let snowBankPile: Float = surf == .powder ? (0.20 * min(1.2, abs(vLat) / 4.2)) : 0.0
        let maxEdgeBite = normalWeight * dynPressure * surf.grip * (
            baseLatMu
            + 1.82 * powf(sinBite, 0.88)
            + (brake > 0 ? 0.92 * surf.grip : 0.0)
            + snowBankPile
        ) * (1.0 - reboundUnweight * 0.25)

        if (surf == .ice || surf == .crust) && ((abs(fp.rollContact) > 0.32 && absSlip > 0.42 && speed > 10) || state.washOut > 0.25 || (edgeWithSlide > 0.35 && abs(vLat) > 3.5)) {
            state.camShake = max(state.camShake, min(0.45, 0.16 + state.washOut * 0.30))
        }

        let absLat = abs(vLat)
        let latLoss = min(absLat, maxEdgeBite * dt)
        let vLatSign: Float = vLat > 0 ? 1 : (vLat < 0 ? -1 : 0)
        let newVLat = vLat - vLatSign * latLoss

        let slipRate = max(0, abs(newVLat) - 0.65) / 5.5
        let targetSideslip = clamp(slipRate * (0.5 + 0.5 * sinf(absSlip)), 0, 1)
        state.sideslip += (targetSideslip - state.sideslip) * min(1.0, dt * 8.0)

        var tipDiveDrag: Float = 0
        if surf == .powder && !state.isSwitch && state.foreAft > 0.55 && fp.pitchContact > 0.12 && speed > 10.0 {
            let dive = min(1.0, (fp.pitchContact - 0.12) * 3.0) * (state.foreAft - 0.45)
            tipDiveDrag = dive * 4.2
            state.faceShotIntensity = min(0.42, state.faceShotIntensity + dive * dt * 0.55)
            state.camShake = max(state.camShake, dive * 0.30)
        }

        let muBase: Float
        switch surf {
        case .powder:  muBase = 0.066
        case .corn:    muBase = 0.044
        case .packed:  muBase = 0.036
        case .crust:   muBase = 0.030
        case .ice:     muBase = 0.016
        case .granite: muBase = 0.095
        }
        let planeFactor: Float = surf == .powder
            ? (1.0 + 0.34 * (1.0 - clamp(speed / 9.0, 0, 1)) - max(0, -state.foreAft) * 0.28 - clamp((speed - 9.0) / 18.0, 0, 0.22))
            : 1.0
        let bowCompaction: Float = surf == .powder
            ? (0.00135 * speed * speed * (1.0 - max(0, -state.foreAft) * 0.35))
            : 0.0
        let edgeCutDrag = absEdge * absEdge * (surf == .powder ? 0.36 : 0.16) * normalWeight * 0.06
        let aeroK = 0.0030 - max(0, state.foreAft) * 0.0011 - state.crouch * 0.00055
        let aeroDrag = max(0.00125, aeroK) * speed * speed

        let sinSlip = sinf(absSlip)
        let plowDrag: Float = (sinSlip > 0.18)
            ? powf((sinSlip - 0.18) / 0.82, 1.55) * normalWeight * (0.28 + surf.sinkDepth * 2.15 + (brake > 0 ? 0.58 : 0.0)) * (0.35 + 0.65 * min(1.6, speed / 12.0))
            : 0.0

        let glideDrag = (muBase * normalWeight * planeFactor) + edgeCutDrag + bowCompaction + aeroDrag + plowDrag + tipDiveDrag

        var newVFwd = vFwd
        if abs(newVFwd) > 0.01 {
            let decel = min(abs(newVFwd), glideDrag * dt)
            newVFwd -= (newVFwd >= 0 ? 1 : -1) * decel
        }

        if brake > 0 && speed > 0.35 {
            let brakeAlign = 0.42 + 0.58 * sinSlip
            let brakeForce = (4.4 + 5.8 * surf.grip) * brakeAlign * (0.48 + 0.52 * min(1.0, speed / 12.0)) * (normalWeight / GRAVITY)
            let cut = min(abs(newVFwd), brakeForce * dt)
            newVFwd -= (newVFwd >= 0 ? 1 : -1) * cut
        } else if state.foreAft < -0.3 && speed > 4.0 && surf != .powder {
            newVFwd *= expf(-dt * 0.18 * (-state.foreAft))
        }

        // Exact L2 kinetic energy conservation during carving vs skidding
        let cleanCarveMask = (1.0 - smoothstep(0.18, 0.72, absSlip)) * smoothstep(0.10, 0.42, edgeAgainstSlide)
        let baseEff: Float
        switch surf {
        case .packed, .corn: baseEff = 0.90
        case .powder:        baseEff = 0.78
        case .ice:           baseEff = 0.55
        case .crust, .granite: baseEff = 0.68
        }
        let carveEfficiency: Float = brake > 0 ? 0.04 : (0.12 + (baseEff - 0.12) * cleanCarveMask)
        let dELat = max(0.0, vLat * vLat - newVLat * newVLat)
        if dELat > 1e-6 && (abs(newVFwd) > 0.25 || cleanCarveMask > 0.25) {
            let fwdSign: Float = abs(newVFwd) > 0.05 ? (newVFwd >= 0 ? 1 : -1) : (vFwd >= 0 ? 1 : -1)
            newVFwd = fwdSign * sqrtf(newVFwd * newVFwd + carveEfficiency * dELat)
        }

        if brake > 0 && hypotf(newVFwd, newVLat) < 0.45 && abs(gFwd) < 4.5 {
            newVFwd *= expf(-dt * 12.0)
        }

        if poleDriveTimer <= 0 && !wPolingActive && t.slopeDeg < 5.0 && hypotf(newVFwd, newVLat) < 0.18 && abs(gFwd) < muBase * normalWeight * 1.15 {
            newVFwd = 0
        }
        if t.slopeDeg < 6.5 && abs(newVLat) < 0.08 && abs(gLat) < maxEdgeBite * 0.9 {
            vel = fwd * newVFwd
        } else {
            vel = fwd * newVFwd + right * newVLat
        }

        state.vx = vel.x
        state.vy = vel.y
        state.vz = vel.z
        state.speed = simd_length(vel)

        if poleDriveTimer > 0 || wPolingActive { state.stanceMode = "POLE" }
        else if brake > 0 { state.stanceMode = state.speed > 1.2 ? "SKID" : "HOLD" }
        else if hasSlope && t.slopeDeg > 3.5 && cosFall < -0.25 && state.speed > 0.6 { state.stanceMode = "UPHILL" }
        else if state.washOut > 0.28 { state.stanceMode = "SLIP" }
        else if state.sideslip > 0.26 && edgeAgainstSlide < 0.35 { state.stanceMode = "SLIP" }
        else if cleanCarveMask > 0.32 && absEdge > 0.26 && state.speed > 4.0 { state.stanceMode = "CARVE" }
        else if absSlip > 0.34 && state.speed > 3.5 { state.stanceMode = "SKID" }
        else if t.slopeDeg < 4.5 && state.speed < 0.8 { state.stanceMode = "FLAT" }
        else { state.stanceMode = "GLIDE" }

        let nextX = state.x + state.vx * dt
        let nextZ = state.z + state.vz * dt
        let nextT = terrainData.sampleTerrain(x: nextX, z: nextZ)

        // Check natural wind-lip / cornice kicker ramp launch
        var kickerHit: WindLipKicker?
        if kickerCooldown <= 0 && state.speed > 7.5 && !kickers.isEmpty {
            for k in kickers {
                let dx = nextX - k.x
                if abs(dx) > k.radius { continue }
                let dz = nextZ - k.z
                if abs(dz) > k.radius { continue }
                let along = dx * k.fwdX + dz * k.fwdZ
                let across = abs(dx * (-k.fwdZ) + dz * k.fwdX)
                let velAlign = (state.vx * k.fwdX + state.vz * k.fwdZ) / max(0.1, hypotf(state.vx, state.vz))
                if along > 1.2 && along < k.radius && across < k.width && velAlign > 0.35 {
                    kickerHit = k
                    break
                }
            }
        }

        let popped = !jumpNow && prevJumpHeld && state.popCharge > 0.04
        let expectedY = state.y + state.vy * dt
        let slopeDropGap = expectedY - nextT.y
        let detachSpeed = hypotf(state.vx, state.vz)
        let naturalLaunch = (slopeDropGap > 0.14 && detachSpeed > 13.0 && (state.slopeDeg > 20 || nextT.slopeDeg > state.slopeDeg + 6))

        if popped || naturalLaunch || kickerHit != nil {
            state.grounded = false
            state.stanceMode = "AIR"
            state.airTime = 0
            state.launchY = state.y
            state.peakAirHeight = 0
            state.airPitch = 0
            state.airYaw = 0
            state.airRoll = 0
            state.grabDuration = 0
            prevLeftSki = nil
            prevRightSki = nil
            poleDriveTimer = 0
            cameraController.airBodyQuat = simd_quatf(angle: -state.yaw, axis: SIMD3<Float>(0, 1, 0))
            state.bodyQuat = cameraController.airBodyQuat
            airPitchHoldMask = foreIn > 0.20 ? 1 : 0

            if let kh = kickerHit {
                kickerCooldown = 1.6
                let rampSlope = kh.rampSlope > 0 ? kh.rampSlope : Float(0.24)
                let speedBoost = min(4.8, state.speed * rampSlope)
                let ollieAdd = state.popCharge * 1.10
                let kickV = clamp(speedBoost + ollieAdd, 1.8, 5.6)
                let lx = n.x * 0.55, ly = n.y * 0.78 + 0.22, lz = n.z * 0.55
                let lMag = max(1e-5, sqrtf(lx * lx + ly * ly + lz * lz))
                state.vx += (lx / lMag) * kickV
                state.vy += (ly / lMag) * kickV
                state.vz += (lz / lMag) * kickV
                state.angYaw = steerEff * 1.3
                state.angPitch = foreIn < -0.20 ? foreIn * 1.4 : 0
                state.angRoll = 0
                events.append(.pop(x: state.x, y: state.y, z: state.z, charge: max(0.6, state.popCharge)))
            } else if popped {
                let surfPopMult: Float = surf == .powder ? 0.80 : (surf == .corn ? 0.90 : 1.0)
                let flexBonus = clamp((curSpeed - 6.0) / 18.0, 0, 1) * 0.35 * state.popCharge
                let popStrength = ((1.40 + 1.65 * powf(state.popCharge, 0.85)) * surfPopMult) + flexBonus
                let lx = n.x * 0.82, ly = n.y * 0.82 + 0.18, lz = n.z * 0.82
                let lMag = max(1e-5, sqrtf(lx * lx + ly * ly + lz * lz))
                state.vx += (lx / lMag) * popStrength
                state.vy += (ly / lMag) * popStrength
                state.vz += (lz / lMag) * popStrength
                state.angYaw = steerEff * (1.1 + 1.9 * state.popCharge)
                state.angPitch = foreIn < -0.20 ? foreIn * (1.2 + 1.8 * state.popCharge) : 0
                state.angRoll = 0
                events.append(.pop(x: state.x, y: state.y, z: state.z, charge: state.popCharge))
            } else {
                state.angYaw = steerEff * 0.55
                state.angPitch = 0
                state.angRoll = 0
            }
            state.popCharge = 0
            landingQuatBlend = 0
            if popped {
                landVel = min(0, landVel) - 0.35
            }
            state.x = nextX
            state.z = nextZ
            state.y = max(expectedY, nextT.y + 0.035)
            return
        }

        state.x = nextX
        state.z = nextZ
        state.y = nextT.y

        // Powder spray, bow-wave snow wash & localized sluff streams
        let edgeEnergy = state.speed * (abs(sinf(state.slipAngle)) * 1.45 + absEdge * 0.55 + state.sideslip * 0.65 + (brake > 0 ? 1.15 : 0))
        state.sprayRate = clamp((edgeEnergy / 11.5) * surf.sprayMult, 0, 2.5)
        var targetBow: Float = 0
        if surf == .powder {
            let spdBow = clamp((state.speed - 4.8) / 13.0, 0, 1)
            targetBow = clamp(
                spdBow * (0.44 + 0.50 * abs(state.edgeAngle) + 0.52 * abs(sinf(state.slipAngle)) + max(0, state.foreAft) * 0.25),
                0, 1.0
            )
        } else if (surf == .packed || surf == .corn) && brake > 0 && state.speed > 7.0 {
            targetBow = clamp((state.speed - 7.0) / 18.0, 0, 0.35)
        }
        state.bowWave += (targetBow - state.bowWave) * min(1.0, dt * 6.5)

        if state.sprayRate > 1.35 && surf == .powder && state.speed > 13.5 {
            state.faceShotIntensity = min(0.55, state.faceShotIntensity + dt * (state.sprayRate - 1.25) * 0.42)
        }
        if state.slopeDeg > 33 && state.sprayRate > 0.52 && surf == .powder {
            state.sluffTrigger = min(1.0, state.sluffTrigger + dt * 1.25)
        }

        if state.slopeDeg > 27.0 && surf == .powder && state.sprayRate > 0.40 && state.speed > 5.0 {
            let gm = hypotf(t.gx, t.gz)
            if gm > 0.15 {
                sluffAccum += dt * (4.5 + state.sprayRate * 5.0 * min(1.5, (state.slopeDeg - 24.0) / 16.0))
                if sluffAccum >= 1.0 {
                    sluffAccum = min(2.0, sluffAccum - 1.0)
                    let dirX = -t.gx / gm, dirZ = -t.gz / gm
                    let sideX = -dirZ, sideZ = dirX
                    let lat = (Float.random(in: 0..<1) - 0.5) * 2.2
                    let sxPos = state.x + sideX * lat + dirX * (0.8 + Float.random(in: 0..<1) * 1.4)
                    let szPos = state.z + sideZ * lat + dirZ * (0.8 + Float.random(in: 0..<1) * 1.4)
                    events.append(.sluffStream(
                        x: sxPos,
                        y: terrainData.sampleHeight(x: sxPos, z: szPos) + 0.08,
                        z: szPos,
                        dirX: dirX,
                        dirZ: dirZ,
                        slopeDeg: state.slopeDeg,
                        strength: clamp((state.slopeDeg - 26.0) / 18.0 + state.sprayRate * 0.35, 0.25, 1.0)
                    ))
                }
            }
        } else {
            sluffAccum = 0
        }

        // Rare automatic slab avalanche trigger on steep >32° wind-loaded powder
        if avalanchesEnabled && !avalanche.active && avalanche.cooldown <= 0 && state.slopeDeg > 32.0 && t.powder > 0.50 && state.sprayRate > 1.45 && state.speed > 13.0 {
            if Float.random(in: 0..<1) < dt * 0.16 {
                _ = triggerAvalanche(uphillOffset: -22, customWidth: 30)
            }
        }

        // Parallel twin ski track stamping
        let sep: Float = 0.31
        let rx = cosf(state.yaw), rz = sinf(state.yaw)
        let lx = state.x - rx * sep, lz = state.z - rz * sep
        let rxp = state.x + rx * sep, rzp = state.z + rz * sep
        let leftPt = SIMD2<Float>(lx, lz)
        let rightPt = SIMD2<Float>(rxp, rzp)
        if surf.sinkDepth > 0.01 && state.speed > 0.35 {
            if let pL = prevLeftSki, let pR = prevRightSki {
                let dMoved = hypotf(lx - pL.x, lz - pL.y)
                if dMoved > 0.14 && dMoved < 6.0 {
                    let depth = clamp((surf.sinkDepth * 3.15) * (0.48 + 0.52 * min(1.0, state.speed / 12.0) + abs(state.edgeAngle) * 0.32), 0.14, 0.94)
                    let berm = clamp(depth * (0.62 + abs(sinf(state.slipAngle)) * 0.68 + (brake > 0 ? 0.5 : 0)), 0.18, 0.96)
                    let rad = 0.088 + abs(sinf(state.slipAngle)) * 0.036 + (brake > 0 ? 0.068 : 0)
                    let skewSrc: Float = state.edgeAngle != 0 ? state.edgeAngle : (state.slipAngle != 0 ? state.slipAngle : 1)
                    let skewSign: Float = skewSrc >= 0 ? 1 : -1
                    events.append(.track(
                        lA: pL, lB: leftPt,
                        rA: pR, rB: rightPt,
                        radius: rad,
                        depth: depth,
                        berm: berm,
                        iceGlint: surf == .powder ? 0.45 : 0.88,
                        skew: skewSign * min(1.0, abs(state.slipAngle) * 1.6)
                    ))
                    prevLeftSki = leftPt
                    prevRightSki = rightPt
                }
            } else {
                prevLeftSki = leftPt
                prevRightSki = rightPt
            }
        }

        // Rate-limited carving coldsmoke rooster-tails
        if state.sprayRate > 0.16 && state.speed > 2.8 {
            sprayAccum += dt * (18.0 + state.sprayRate * 16.0)
            if sprayAccum >= 1.0 {
                sprayAccum = min(2.0, sprayAccum - 1.0)
                let side: Float = state.edgeAngle >= 0 ? -1 : 1
                events.append(.spray(
                    x: state.x + rx * side * 0.35 - sinf(state.yaw) * 0.45,
                    y: state.y + 0.08,
                    z: state.z + rz * side * 0.35 + cosf(state.yaw) * 0.45,
                    vx: state.vx * 0.28 + rx * side * (1.8 + state.sprayRate * 2.6),
                    vy: 0.72 + state.sprayRate * 1.05,
                    vz: state.vz * 0.28 + rz * side * (1.8 + state.sprayRate * 2.6),
                    rate: state.sprayRate,
                    sluff: state.sluffTrigger > 0.3,
                    slopeX: -t.gx,
                    slopeZ: -t.gz
                ))
            }
        } else {
            sprayAccum = 0
        }

        updateGroundedQuaternion(dt: dt)
    }

    // MARK: - Airborne Physics

    private func stepAirborne(
        dt: Float,
        t: TerrainSample,
        steer: Float,
        foreIn: Float,
        corkIn: Float,
        grabNow: Bool,
        altHeld: Bool,
        useHeadSteer: Bool,
        time: Float
    ) {
        state.airTime += dt
        state.stanceMode = "AIR"
        state.sideslip = 0
        state.washOut = 0
        state.carveLoadG = 0.0
        state.sprayRate = 0
        state.bowWave = max(0, state.bowWave - dt * 6.0)
        state.polePlantL = max(0, state.polePlantL - dt * 4.5)
        state.polePlantR = max(0, state.polePlantR - dt * 4.5)
        state.skatePush = max(0, state.skatePush - dt * 5.0)
        state.insideUnweight *= expf(-dt * 4.0)

        state.vy -= GRAVITY * dt
        let airDrag = expf(-dt * 0.045)
        state.vx *= airDrag
        state.vz *= airDrag
        state.x += state.vx * dt
        state.y += state.vy * dt
        state.z += state.vz * dt
        state.speed = sqrtf(state.vx * state.vx + state.vy * state.vy + state.vz * state.vz)

        let groundT = terrainData.sampleTerrain(x: state.x, z: state.z)
        state.terrainY = groundT.y
        state.normal = groundT.normal
        let clearance = state.y - groundT.y
        if clearance > state.peakAirHeight { state.peakAirHeight = clearance }

        let grabBoost: Float = grabNow ? 1.24 : 1.0
        if grabNow { state.grabDuration += dt }
        state.crouch += ((grabNow ? 0.85 : 0.32) - state.crouch) * min(1.0, dt * 9.0)

        let maxSpin: Float = 6.2 * grabBoost
        let maxFlip: Float = 5.4 * grabBoost
        let maxCork: Float = 4.6 * grabBoost

        let headSpin: Float = (useHeadSteer && !altHeld && abs(cameraController.lookYawOff) > 0.08)
            ? clamp((cameraController.lookYawOff - (cameraController.lookYawOff >= 0 ? 1 : -1) * 0.08) * 0.72, -0.58, 0.58)
            : 0
        let totalAirSteer = clamp(steer + headSpin, -1.25, 1.25)

        if abs(totalAirSteer) > 0.05 {
            state.angYaw = clamp(state.angYaw + totalAirSteer * 11.5 * grabBoost * dt, -maxSpin, maxSpin)
        } else {
            state.angYaw *= expf(-dt * 2.2)
        }
        if abs(headSpin) > 0.01 {
            let bleed = (cameraController.lookYawOff >= 0 ? Float(1) : Float(-1)) * min(abs(cameraController.lookYawOff), abs(state.angYaw) * dt * 0.45)
            cameraController.lookYawOff = wrapAngle(cameraController.lookYawOff - bleed)
        }
        if airPitchHoldMask != 0 && foreIn <= 0.15 {
            airPitchHoldMask = 0
        }
        let effForeIn: Float = (airPitchHoldMask != 0 && foreIn > 0) ? 0 : foreIn
        if abs(effForeIn) > 0.05 {
            state.angPitch = clamp(state.angPitch - effForeIn * 10.2 * grabBoost * dt, -maxFlip, maxFlip)
        } else {
            state.angPitch *= expf(-dt * 3.2)
        }
        if abs(corkIn) > 0.05 {
            state.angRoll = clamp(state.angRoll + corkIn * 9.5 * grabBoost * dt, -maxCork, maxCork)
        } else {
            state.angRoll *= expf(-dt * 2.6)
        }

        state.airYaw += state.angYaw * dt
        state.airPitch += state.angPitch * dt
        state.airRoll += state.angRoll * dt

        let dq = quatFromEulerYXZ(pitch: state.angPitch * dt, yaw: -state.angYaw * dt, roll: -state.angRoll * dt)
        state.quat = simd_normalize(state.quat * dq)
        cameraController.airBodyQuat = simd_normalize(cameraController.airBodyQuat * dq)
        state.bodyQuat = cameraController.airBodyQuat

        let fwdWorld = state.quat.act(SIMD3<Float>(0, 0, -1))
        if hypotf(fwdWorld.x, fwdWorld.z) > 0.25 {
            state.yaw = atan2f(fwdWorld.x, -fwdWorld.z)
        }

        if state.y <= groundT.y {
            state.y = groundT.y
            state.grounded = true
            let n = groundT.normal
            let vel = SIMD3<Float>(state.vx, state.vy, state.vz)
            let impactNormalSpeed = max(0, -simd_dot(vel, n))

            let skiUp = state.quat.act(SIMD3<Float>(0, 1, 0))
            let skiFwd = state.quat.act(SIMD3<Float>(0, 0, -1))
            let upAlign = simd_dot(skiUp, n)
            let tiltErrDeg = radToDeg(acosf(clamp(upAlign, -1, 1)))

            let hVel = vel - n * simd_dot(vel, n)
            let hSpd = simd_length(hVel)
            var crossErrDeg: Float = 0
            var signedTravelDot: Float = 1.0
            if hSpd > 3.0 {
                let travelDir = hVel / hSpd
                let skiFlat = skiFwd - n * simd_dot(skiFwd, n)
                let flatLen = simd_length(skiFlat)
                if flatLen > 0.1 {
                    let skiFlatNorm = skiFlat / flatLen
                    signedTravelDot = simd_dot(skiFlatNorm, travelDir)
                    crossErrDeg = radToDeg(acosf(clamp(abs(signedTravelDot), 0, 1)))
                }
            }

            let surf = groundT.surface
            let maxImpact: Float = surf == .powder ? 21.5 : ((surf == .packed || surf == .corn) ? 16.5 : 13.0)
            let maxTilt: Float = surf == .powder ? 58.0 : ((surf == .ice || surf == .granite) ? 38.0 : 48.0)
            let maxCross: Float = surf == .powder ? 68.0 : (surf == .ice ? 42.0 : 54.0)

            let bigAir = state.airTime > 0.42 && state.peakAirHeight > 0.72

            if tiltErrDeg > maxTilt || impactNormalSpeed > maxImpact || (bigAir && crossErrDeg > maxCross && hSpd > 11) {
                var reason = "OVER-ROTATED LANDING"
                if impactNormalSpeed > maxImpact { reason = String(format: "FLAT IMPACT (%.0f m/s)", impactNormalSpeed) }
                else if crossErrDeg > maxCross { reason = String(format: "CAUGHT EDGE (%.0f° CROSS-LINE)", crossErrDeg) }
                else if upAlign < 0 { reason = "INVERTED IMPACT" }
                triggerCrash(reason: reason, groundT: groundT, impact: impactNormalSpeed, time: time)
                return
            }

            let wasSwitch = state.isSwitch
            let landedSwitch = signedTravelDot < -0.35
            state.isSwitch = landedSwitch

            let sketchy = tiltErrDeg > maxTilt * 0.65 || impactNormalSpeed > maxImpact * 0.72 || crossErrDeg > maxCross * 0.72
            state.lastLandingQuality = sketchy ? "SKETCHY" : "STOMPED"
            if sketchy {
                state.vx *= 0.72
                state.vz *= 0.72
                state.camShake = 0.85
            } else {
                let damp: Float = surf == .powder ? 0.90 : 0.96
                state.vx *= damp
                state.vz *= damp
                state.camShake = min(0.55, impactNormalSpeed * 0.04)
            }

            var newVel = SIMD3<Float>(state.vx, state.vy, state.vz)
            newVel -= n * simd_dot(newVel, n)
            state.vx = newVel.x
            state.vy = newVel.y
            state.vz = newVel.z

            let softFactor: Float = surf == .powder ? 0.88 : (surf == .ice ? 1.12 : 1.02)
            let impDrive = 1.25 + min(3.3, impactNormalSpeed * 0.36)
            landVel += min(4.40, impDrive * softFactor)
            landPitchVel -= min(2.6, (0.55 + impactNormalSpeed * 0.19) * softFactor)
            if sketchy {
                landRollVel += (state.edgeAngle >= 0 ? 1 : -1) * min(2.4, impactNormalSpeed * 0.16)
            } else {
                landRollVel += (state.edgeAngle >= 0 ? -1 : 1) * min(0.45, impactNormalSpeed * 0.03)
            }
            scissorTarget = (sketchy ? 0.86 : min(0.60, impactNormalSpeed * 0.050)) * (state.edgeAngle >= 0 ? 1 : -1)
            if surf == .powder {
                state.bowWave = min(1.0, max(state.bowWave, impactNormalSpeed * 0.085))
            }
            state.faceShotIntensity = min(0.65, state.faceShotIntensity + (surf == .powder ? min(0.38, impactNormalSpeed * 0.030) : 0.06))

            if avalanchesEnabled && !avalanche.active && avalanche.cooldown <= 0 && groundT.slopeDeg > 32.0 && groundT.powder > 0.50 && impactNormalSpeed > 9.5 {
                if Float.random(in: 0..<1) < 0.45 {
                    _ = triggerAvalanche(uphillOffset: -18, customWidth: 32)
                }
            }

            if bigAir {
                var parts: [String] = []
                if wasSwitch { parts.append("SWITCH") }
                let flips = Int(roundf(abs(state.airPitch) / (.pi * 1.65)))
                let spins = Int(roundf(abs(state.airYaw) / (.pi * 0.85))) * 180
                let corks = abs(state.airRoll) > 1.25
                if flips >= 1 {
                    parts.append(state.airPitch > 0 ? (flips > 1 ? "DOUBLE BACKFLIP" : "BACKFLIP") : (flips > 1 ? "DOUBLE FRONTFLIP" : "FRONTFLIP"))
                }
                if corks && spins >= 360 { parts.append("CORK \(spins)") }
                else if spins >= 180 { parts.append("\(spins)°") }
                if state.grabDuration > 0.18 { parts.append("MUTE GRAB") }
                if landedSwitch && !wasSwitch { parts.append("TO SWITCH") }
                let label = parts.isEmpty ? "CLIFF DROP" : parts.joined(separator: " · ")
                state.lastTrickText = String(format: "%@ · %@ (%.1fm AIR)", state.lastLandingQuality, label, state.peakAirHeight)
                state.lastTrickTime = time
            }

            events.append(.land(
                x: state.x, y: groundT.y, z: state.z,
                impact: impactNormalSpeed,
                powder: surf == .powder,
                sketchy: sketchy
            ))

            state.angPitch = 0
            state.angYaw = 0
            state.angRoll = 0
            clampLookOffsets(mode: cameraController.camMode)
            landingStartQuat = state.quat
            landingQuatBlend = 1.0
            updateGroundedQuaternion(dt: dt)
        }
    }

    // MARK: - Sub-Routines (Avalanche, Collisions, Quaternion, 4-Point, Recorder)

    private func stepAvalanche(dt: Float, time: Float) {
        if avalanche.cooldown > 0 { avalanche.cooldown = max(0, avalanche.cooldown - dt) }
        if !avalanche.active {
            avalanche.proximity = 0
            return
        }
        avalanche.age += dt
        let fx = avalanche.originX + avalanche.dirX * avalanche.frontDist
        let fz = avalanche.originZ + avalanche.dirZ * avalanche.frontDist
        let ft = terrainData.sampleTerrain(x: fx, z: fz)
        let targetSpd = clamp(9.0 + max(0, ft.slopeDeg - 10.0) * 0.65, 10.0, 29.0)
        avalanche.speed += (targetSpd - avalanche.speed) * min(1.0, dt * 1.4)
        avalanche.frontDist += avalanche.speed * dt
        avalanche.width = min(54.0, avalanche.width + dt * 0.55)

        if avalanche.age > 28 || avalanche.frontDist > avalanche.maxDist {
            avalanche.strength = max(0, avalanche.strength - dt * 0.16)
            if avalanche.strength <= 0.01 {
                avalanche.active = false
                avalanche.proximity = 0
                return
            }
        }

        let relX = state.x - avalanche.originX
        let relZ = state.z - avalanche.originZ
        let along = relX * avalanche.dirX + relZ * avalanche.dirZ
        let across = relX * (-avalanche.dirZ) + relZ * avalanche.dirX
        let distToFront = hypotf(along - avalanche.frontDist, max(0, abs(across) - avalanche.width * 0.72))
        avalanche.proximity = clamp(1.0 - distToFront / 135.0, 0, 1) * avalanche.strength

        if avalanche.proximity > 0.25 {
            state.camShake = max(state.camShake, avalanche.proximity * 0.36)
        }

        let inCorridor = abs(across) < avalanche.width * 0.85 && along > 5.0
        let overtaken = avalanche.frontDist >= along - 3.5 && avalanche.frontDist <= along + 42.0
        let clearance = state.y - state.terrainY
        if inCorridor && overtaken && clearance < 6.5 && !state.crashed && avalanche.age > 1.4 {
            state.faceShotIntensity = min(1.0, state.faceShotIntensity + dt * 3.8)
            if avalanche.frontDist >= along + 3.0 && !avalanche.engulfed {
                avalanche.engulfed = true
                triggerCrash(
                    reason: "ENGULFED BY SLAB AVALANCHE",
                    groundT: terrainData.sampleTerrain(x: state.x, z: state.z),
                    impact: 18.0,
                    time: time
                )
            }
        }
    }

    private func triggerCrash(reason: String, groundT: TerrainSample, impact: Float, time: Float) {
        state.crashed = true
        state.crashTimer = 0
        state.crashReason = reason
        state.lastLandingQuality = "WIPEOUT"
        state.lastTrickText = "WIPEOUT · \(reason)"
        state.lastTrickTime = time
        state.faceShotIntensity = 0.72
        state.camShake = 1.0
        state.vx *= 0.55
        state.vz *= 0.55
        state.vy = 0
        events.append(.crash(x: state.x, y: groundT.y, z: state.z, impact: max(8.0, impact), reason: reason))
    }

    private func checkCollisions(time: Float) {
        if state.crashed { return }
        let fastTreeCheck = state.speed > 9.0
        let invSpd: Float = fastTreeCheck ? (1.0 / max(0.1, hypotf(state.vx, state.vz))) : 0
        let vDirX = fastTreeCheck ? state.vx * invSpd : 0
        let vDirZ = fastTreeCheck ? state.vz * invSpd : 0
        let rgtX = cosf(state.yaw + cameraController.lookYawOff)
        let rgtZ = sinf(state.yaw + cameraController.lookYawOff)

        for i in 0..<colliders.count {
            let c = colliders[i]
            let maxBox: Float = (fastTreeCheck && c.kind == .tree) ? 9.8 : (c.r + 1.5)
            let dx = state.x - c.x
            if abs(dx) > maxBox { continue }
            let dz = state.z - c.z
            if abs(dz) > maxBox { continue }
            if state.y > c.y + c.h { continue }
            let d = hypotf(dx, dz)
            let minD = c.r + 0.55
            if d < minD && d > 0.001 {
                let nx = dx / d, nz = dz / d
                state.x = c.x + nx * minD
                state.z = c.z + nz * minD
                let vn = state.vx * nx + state.vz * nz
                if vn < -10.5 {
                    triggerCrash(
                        reason: "STRUCK \(c.kind.rawValue.uppercased())",
                        groundT: terrainData.sampleTerrain(x: state.x, z: state.z),
                        impact: -vn,
                        time: time
                    )
                    return
                } else if vn < 0 {
                    state.vx -= nx * vn * 1.35
                    state.vz -= nz * vn * 1.35
                    state.camShake = max(state.camShake, 0.55)
                }
            } else if fastTreeCheck && c.kind == .tree && d < 9.5 {
                let tx = c.x - state.x
                let tz = c.z - state.z
                let along = tx * vDirX + tz * vDirZ
                if along > -2.2 && along < 3.5 {
                    let lastT = whooshCD[i] ?? -99
                    if time - lastT > 1.8 {
                        whooshCD[i] = time
                        let lat = tx * rgtX + tz * rgtZ
                        events.append(.treeWhoosh(
                            pan: clamp(lat / 5.5, -1, 1),
                            closeness: clamp(1.0 - d / 9.5, 0.12, 1.0),
                            speed: state.speed
                        ))
                    }
                }
            }
        }
    }

    private func updateGroundedQuaternion(dt: Float) {
        let up = cameraController.smoothNormal
        let rawFwd = SIMD3<Float>(sinf(state.yaw), 0, -cosf(state.yaw))
        var fwd = rawFwd - up * simd_dot(rawFwd, up)
        let fLen = simd_length(fwd)
        fwd = fLen > 1e-6 ? (fwd / fLen) : SIMD3<Float>(0, 0, -1)
        var right = simd_cross(fwd, up)
        let rLen = simd_length(right)
        right = rLen > 1e-6 ? (right / rLen) : SIMD3<Float>(1, 0, 0)
        var trueUp = simd_cross(right, fwd)
        let uLen = simd_length(trueUp)
        trueUp = uLen > 1e-6 ? (trueUp / uLen) : SIMD3<Float>(0, 1, 0)

        var targetQ = quatFromBasis(right: right, up: trueUp, forwardZ: -fwd)
        if landingQuatBlend > 0.001 && dt > 0 {
            landingQuatBlend = max(0, landingQuatBlend - dt * 7.2)
            let t = 1.0 - landingQuatBlend * landingQuatBlend
            if simd_dot(landingStartQuat, targetQ) < 0 {
                targetQ = -targetQ
            }
            state.quat = simd_normalize(simd_slerp(landingStartQuat, targetQ, t))
        } else {
            state.quat = targetQ
        }
    }

    private func sampleFourPoint(x: Float, z: Float, yaw: Float) -> (rollContact: Float, pitchContact: Float) {
        let fx = sinf(yaw), fz = -cosf(yaw)
        let rx = cosf(yaw), rz = sinf(yaw)
        let hLeft = terrainData.sampleHeight(x: x - rx * 0.28, z: z - rz * 0.28)
        let hRight = terrainData.sampleHeight(x: x + rx * 0.28, z: z + rz * 0.28)
        let hTip = terrainData.sampleHeight(x: x + fx * 0.96, z: z + fz * 0.96)
        let hTail = terrainData.sampleHeight(x: x - fx * 0.86, z: z - fz * 0.86)
        return (
            rollContact: (hRight - hLeft) / 0.56,
            pitchContact: (hTip - hTail) / 1.82
        )
    }

    private func recordFrame() {
        if replayFrames.count >= 3600 {
            replayFrames.removeFirst()
        }
        replayFrames.append(ReplayFrame(from: state))
    }
}
