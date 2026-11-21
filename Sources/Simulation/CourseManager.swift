import Foundation
import simd

/// Result returned by running a deterministic demo script from CLI `--demo <mode>`.
public struct DemoScriptResult: Sendable {
    public var simTime: Float
    public var activateScout: Bool = false
    public var scoutPhase: Float = 0
    public var activateReplay: Bool = false
    public var replayCursor: Int = 0
    public var inspectBodyView: String? = nil
    public var toastMessage: String? = nil
    public var toastDuration: Float = 2.6

    public init(simTime: Float) {
        self.simTime = simTime
    }
}

/// Manages Chamonix drop-zone courses, optional gate runs, and deterministic demo scripts
/// ported from `src/main.js`.
public final class CourseManager {
    public private(set) var activeZoneIndex: Int = 0
    public private(set) var activeCourse: CourseDefinition?
    public private(set) var runState = RunState()

    public init() {}

    public func selectDropZone(
        index: Int,
        terrainData: any TerrainQueryProtocol
    ) -> DropZone? {
        let zones = terrainData.meta.dropZones
        guard !zones.isEmpty else { return nil }
        let clampedIdx = ((index % zones.count) + zones.count) % zones.count
        activeZoneIndex = clampedIdx
        let dz = zones[clampedIdx]
        activeCourse = CourseDefinition(dropZone: dz, gates: [])
        runState = RunState()
        runState.totalGates = 0
        return dz
    }

    public func selectDropZone(
        id: String,
        terrainData: any TerrainQueryProtocol
    ) -> Int? {
        guard let idx = terrainData.meta.dropZones.firstIndex(where: { $0.id == id }) else {
            return nil
        }
        _ = selectDropZone(index: idx, terrainData: terrainData)
        return idx
    }

    public func setCourse(_ course: CourseDefinition) {
        activeCourse = course
        runState = RunState()
        runState.totalGates = course.gates.count
    }

    public func updateRunState(state: SkierState, dt: Float, started: Bool, paused: Bool) {
        runState.started = started
        runState.pausedSim = paused
        if started && !paused {
            runState.elapsedTime += dt
        }
        runState.verticalDropped = state.verticalDropped
        runState.maxSpeedKmh = state.maxSpeedKmh

        if var course = activeCourse, !course.gates.isEmpty {
            var passedCount = 0
            for i in 0..<course.gates.count {
                if !course.gates[i].passed {
                    let gPos = course.gates[i].position
                    let distXZ = hypotf(state.x - gPos.x, state.z - gPos.z)
                    if distXZ < 6.0 && abs(state.y - gPos.y) < 5.0 {
                        course.gates[i].passed = true
                    }
                }
                if course.gates[i].passed {
                    passedCount += 1
                }
            }
            activeCourse = course
            runState.gatesPassed = passedCount
            runState.totalGates = course.gates.count
        }
    }

    /// Consumes queued physics events into track buffer and particle emitters (mirrors `main.js` `consumePhysicsEvents`).
    public static func flushPhysicsEvents(
        physics: SkiPhysics,
        trackBuffer: any SnowTracksSubsystem,
        particles: any ParticlesSubsystem
    ) {
        for ev in physics.events {
            switch ev {
            case let .track(lA, lB, rA, rB, radius, depth, berm, iceGlint, skew):
                trackBuffer.stampTwinSkis(
                    lA: lA, lB: lB, rA: rA, rB: rB,
                    radius: radius, depth: depth, berm: berm, iceGlint: iceGlint, skew: skew
                )
            case let .spray(x, y, z, vx, vy, vz, rate, sluff, slopeX, slopeZ):
                particles.emitCarveRoosterTail(
                    x: x, y: y, z: z, vx: vx, vy: vy, vz: vz,
                    rate: rate, sluff: sluff, slopeX: slopeX, slopeZ: slopeZ
                )
                if sluff && Float.random(in: 0..<1) < 0.38 {
                    let sn = max(0.001, hypotf(slopeX, slopeZ))
                    particles.emitPuff(
                        x: x, y: y + 0.18, z: z,
                        vx: (slopeX / sn) * 7.2, vy: -1.0, vz: (slopeZ / sn) * 7.2,
                        size: 1.8, expand: 2.4, peakAlpha: 0.42, life: 1.8, kind: 0
                    )
                }
            case let .polePlant(_, x, y, z, strength, powder):
                trackBuffer.stampPoleDimple(x: x, z: z, radius: powder ? 0.15 : 0.11, depth: powder ? 0.56 : 0.34)
                particles.emitBurst(
                    x: x, y: y + 0.06, z: z,
                    count: powder ? 5 : 3,
                    speed: 1.15 + strength * 0.75,
                    sizeMin: 0.18, sizeMax: 0.36,
                    expand: 0.55 + strength * 0.25,
                    peakAlpha: powder ? 0.56 : 0.34,
                    life: 0.65,
                    kind: 0
                )
            case let .sluffStream(x, y, z, dirX, dirZ, _, strength):
                let spd = 6.2 + strength * 5.4
                let sx = -dirZ, sz = dirX
                for k in 0..<2 {
                    let lat = (Float.random(in: 0..<1) - 0.5) * 0.65
                    particles.emitPuff(
                        x: x + sx * lat, y: y + 0.05 + Float(k) * 0.04, z: z + sz * lat,
                        vx: dirX * spd + (Float.random(in: 0..<1) - 0.5) * 0.9,
                        vy: -0.65,
                        vz: dirZ * spd + (Float.random(in: 0..<1) - 0.5) * 0.9,
                        size: 0.42 + strength * 0.42,
                        expand: 1.35 + Float.random(in: 0..<1) * 0.55,
                        peakAlpha: 0.36 + strength * 0.22,
                        life: 0.72,
                        kind: 0
                    )
                }
            case let .pop(x, y, z, _):
                particles.emitBurst(
                    x: x, y: y + 0.1, z: z,
                    count: 12, speed: 3.2, sizeMin: 0.65, sizeMax: 1.35,
                    expand: 1.05, peakAlpha: 0.54, life: 1.8, kind: 0
                )
            case let .land(x, y, z, impact, _, _):
                let rad = 0.48 + min(0.85, impact * 0.055)
                trackBuffer.stampCapsule(
                    ax: x - 0.24, az: z, bx: x + 0.24, bz: z,
                    radius: rad, depth: 0.95, berm: 0.92, iceGlint: 0.6, skew: 0
                )
                particles.emitBurst(
                    x: x, y: y + 0.2, z: z,
                    count: min(22, 8 + Int(roundf(impact * 1.2))),
                    speed: 3.8 + impact * 0.35,
                    sizeMin: 0.85, sizeMax: 2.0,
                    expand: 1.5, peakAlpha: 0.50, life: 1.9, kind: 0
                )
            case let .crash(x, y, z, _, _):
                trackBuffer.stampCapsule(
                    ax: x - 0.5, az: z - 0.5, bx: x + 0.5, bz: z + 0.5,
                    radius: 1.35, depth: 1.0, berm: 1.0, iceGlint: 0.5, skew: 0
                )
                particles.emitBurst(
                    x: x, y: y + 0.25, z: z,
                    count: 26, speed: 6.2, sizeMin: 1.0, sizeMax: 2.6,
                    expand: 1.9, peakAlpha: 0.56, life: 2.1, kind: 0
                )
            case let .crashTrail(x, y, z, _):
                trackBuffer.stampCapsule(
                    ax: x - 0.2, az: z, bx: x + 0.2, bz: z,
                    radius: 0.80, depth: 0.9, berm: 0.85, iceGlint: 0.3, skew: 0
                )
                particles.emitPuff(
                    x: x, y: y + 0.2, z: z,
                    vx: (Float.random(in: 0..<1) - 0.5) * 4, vy: 2.0, vz: (Float.random(in: 0..<1) - 0.5) * 4,
                    size: 1.4, expand: 1.6, peakAlpha: 0.60, life: 2.0, kind: 0
                )
            case .avalancheStart:
                let a = physics.avalanche
                let sx = -a.dirZ, sz = a.dirX
                for k in -8...8 {
                    let across = (Float(k) / 8.0) * a.width * 0.85
                    let wx = a.originX + sx * across
                    let wz = a.originZ + sz * across
                    let wy = physics.terrainData.sampleHeight(x: wx, z: wz)
                    particles.emitPuff(
                        x: wx, y: wy + 1.2, z: wz,
                        vx: a.dirX * 10 + (Float.random(in: 0..<1) - 0.5) * 3,
                        vy: 2.8 + Float.random(in: 0..<1) * 2.8,
                        vz: a.dirZ * 10 + (Float.random(in: 0..<1) - 0.5) * 3,
                        size: 3.4 + Float.random(in: 0..<1) * 2.8,
                        expand: 2.8, peakAlpha: 0.78, life: 2.6, kind: 1
                    )
                }
            case .treeWhoosh, .bootCreak:
                break
            }
        }
        physics.clearEvents()
    }

    private static func sampleSunLOS(
        terrainData: any TerrainQueryProtocol,
        ox: Float, oy: Float, oz: Float,
        sunDir: SIMD3<Float>
    ) -> Float {
        if sunDir.y <= 0.02 { return 0.0 }
        var minClear: Float = 1e9
        let dists: [Float] = [45, 110, 240, 480, 860, 1450, 2250, 3250, 4450, 5800]
        for d in dists {
            let px = ox + sunDir.x * d
            let pz = oz + sunDir.z * d
            if abs(px) > 6300 || abs(pz) > 6300 { break }
            let rayY = oy + sunDir.y * d
            let terrY = terrainData.sampleHeight(x: px, z: pz)
            let angleDiff = (rayY - terrY) / d
            if angleDiff < minClear { minClear = angleDiff }
            if minClear < -0.028 { return 0.0 }
        }
        return smoothstep(-0.022, 0.036, minClear)
    }

    /// Runs deterministic verification/showcase scenarios ported 1:1 from `runDeterministicScript` in `src/main.js`.
    public static func runDeterministicScript(
        mode: String,
        physics: SkiPhysics,
        terrainData: any TerrainQueryProtocol,
        trackBuffer: any SnowTracksSubsystem,
        particles: any ParticlesSubsystem,
        colliders: [PropCollider],
        preset: LightingPreset,
        initialSimTime: Float = 0,
        cameraPos: SIMD3<Float> = .zero,
        viewportHeightPx: Float = 1080,
        fovDeg: Float = 86
    ) -> DemoScriptResult {
        let dt = ChamonixConstants.fixedPhysicsDt
        let sunDir = preset.sunDirection
        var simTime = initialSimTime
        var res = DemoScriptResult(simTime: simTime)

        func stepLoop(steps: Int, keysForStep: (Int, Float) -> Set<String>) {
            for stepIdx in 0..<steps {
                let sec = Float(stepIdx) * dt
                var input = InputState()
                input.activeKeys = keysForStep(stepIdx, sec)
                simTime += dt
                physics.step(dt: dt, time: simTime, input: input)
                trackBuffer.recenterIfNeeded(skierX: physics.state.x, skierZ: physics.state.z, decayAmt: 0)
                flushPhysicsEvents(physics: physics, trackBuffer: trackBuffer, particles: particles)
                particles.update(
                    dt: dt,
                    cameraPos: cameraPos,
                    viewportHeightPx: viewportHeightPx,
                    fovDeg: fovDeg,
                    preset: preset,
                    sunDir: sunDir,
                    time: simTime
                )
            }
        }

        switch mode {
        case "tracks":
            let totalSteps = Int(roundf(120 * 4.4))
            physics.mutateState { s in
                s.vx = sinf(s.yaw) * 4.0
                s.vz = -cosf(s.yaw) * 4.0
            }
            stepLoop(steps: totalSteps) { _, sec in
                if sec < 1.1 { return ["KeyD"] }
                else if sec >= 1.3 && sec < 2.8 { return ["KeyA"] }
                else if sec >= 3.0 && sec < 4.2 { return ["KeyD"] }
                return []
            }
            physics.lookYawOff = .pi * 0.94
            physics.lookPitchOff = 0.05
            physics.mutateState { s in
                s.vx = 0; s.vy = 0; s.vz = 0; s.speed = 0
            }

        case "carve", "replay":
            let totalSteps = Int(roundf(120 * 4.2))
            physics.mutateState { s in
                s.vx = sinf(s.yaw) * 6.0
                s.vz = -cosf(s.yaw) * 6.0
            }
            stepLoop(steps: totalSteps) { _, sec in
                if sec < 1.1 { return ["KeyD"] }
                else if sec >= 1.25 && sec < 2.65 { return ["KeyA"] }
                else if sec >= 2.8 { return ["KeyD"] }
                return []
            }
            physics.mutateState { s in
                s.isSwitch = false
                s.edgeAngle = 0.78
            }
            let s = physics.state
            let fwdX = sinf(s.yaw), fwdZ = -cosf(s.yaw)
            let rx = cosf(s.yaw), rz = sinf(s.yaw)
            let outSide: Float = -1
            for k in 0..<34 {
                let f = Float(k) / 33.0
                let alongBack = 0.8 + f * 9.8
                let out = 0.55 + powf(f, 0.78) * 3.8
                let lift = 0.35 + sinf(min(1, f * 1.15) * .pi) * 2.55 + (Float.random(in: 0..<1) - 0.5) * 0.4
                let px = s.x - fwdX * alongBack + rx * outSide * out + (Float.random(in: 0..<1) - 0.5) * 0.45
                let pz = s.z - fwdZ * alongBack + rz * outSide * out + (Float.random(in: 0..<1) - 0.5) * 0.45
                let gy = terrainData.sampleHeight(x: px, z: pz)
                let sz = 0.85 + sinf(min(1, f * 1.2) * .pi) * 1.45 + f * 0.75
                particles.emitPuff(
                    x: px, y: gy + lift, z: pz,
                    vx: rx * outSide * 2.5, vy: 1.4, vz: rz * outSide * 2.5,
                    size: sz, expand: 2.2, peakAlpha: 0.86 - f * 0.30, life: 0.95, kind: 0
                )
            }
            if mode == "replay" && physics.replayFrames.count > 10 {
                res.activateReplay = true
                res.replayCursor = max(0, physics.replayFrames.count - 1)
            } else {
                physics.lookYawOff = -0.34
            }

        case "air":
            let totalSteps = Int(roundf(120 * 3.72))
            physics.mutateState { s in
                s.vx = sinf(s.yaw) * 7.5
                s.vz = -cosf(s.yaw) * 7.5
            }
            stepLoop(steps: totalSteps) { _, sec in
                var k: Set<String> = ["KeyW"]
                if sec > 2.15 && sec < 3.40 { k.insert("Space") }
                if sec >= 3.42 {
                    k.insert("KeyF")
                    k.insert("KeyD")
                }
                return k
            }
            physics.lookPitchOff = -0.32

        case "avalanche":
            let yaw = physics.state.yaw
            let initFwdX = sinf(yaw), initFwdZ = -cosf(yaw)
            let nx = physics.state.x + initFwdX * 36
            let nz = physics.state.z + initFwdZ * 36
            let nt = terrainData.sampleTerrain(x: nx, z: nz)
            physics.mutateState { s in
                s.x = nx; s.z = nz
                s.y = nt.y; s.terrainY = nt.y; s.slopeDeg = nt.slopeDeg
                s.normal = nt.normal
                s.vx = 0; s.vy = 0; s.vz = 0; s.speed = 0
                s.isSwitch = false
            }
            _ = physics.triggerAvalanche(uphillOffset: 15, customWidth: 34, force: true)
            let advSteps = Int(roundf(120 * 2.1))
            for _ in 0..<advSteps {
                simTime += dt
                physics.advanceAvalancheDemoStep(dt: dt)
                particles.emitAvalancheWave(avalanche: physics.avalanche, terrainData: terrainData, dt: dt)
                particles.update(
                    dt: dt, cameraPos: cameraPos, viewportHeightPx: viewportHeightPx,
                    fovDeg: fovDeg, preset: preset, sunDir: sunDir, time: simTime
                )
            }
            let avy = physics.avalanche
            let sx = -avy.dirZ, sz = avy.dirX
            for k in -10...10 {
                let fk = Float(k)
                let across = (fk / 10.0) * avy.width * 0.85
                let along = avy.frontDist - 4.0 + sinf(fk * 0.9) * 3.5
                let wx = avy.originX + avy.dirX * along + sx * across
                let wz = avy.originZ + avy.dirZ * along + sz * across
                let wy = terrainData.sampleHeight(x: wx, z: wz)
                particles.emitPuff(
                    x: wx, y: wy + 2.8 + abs(cosf(fk * 0.7)) * 3.2, z: wz,
                    vx: 0, vy: 2.0, vz: 0,
                    size: 5.2 + Float.random(in: 0..<1) * 3.8,
                    expand: 3.0, peakAlpha: 0.92, life: 1.5, kind: 1
                )
            }
            physics.mutateState { s in
                s.yaw = atan2f(avy.dirX, -avy.dirZ)
                s.faceShotIntensity = 0
                s.camShake = 0
            }
            physics.syncGroundedQuaternion(dt: 0)
            physics.lookYawOff = 0
            physics.lookPitchOff = -0.26

        case "scout":
            res.activateScout = true
            res.scoutPhase = 0.34

        case "trees_closeup":
            let s = physics.state
            let fx = sinf(s.yaw), fz = -cosf(s.yaw)
            let rx = cosf(s.yaw), rz = sinf(s.yaw)
            let nx = s.x + fx * 18 - rx * 4
            let nz = s.z + fz * 18 - rz * 4
            let t = terrainData.sampleTerrain(x: nx, z: nz)
            physics.mutateState { st in
                st.x = nx; st.z = nz; st.y = t.y; st.terrainY = t.y; st.slopeDeg = t.slopeDeg
                st.normal = t.normal
            }
            physics.setSmoothNormal(t.normal)
            physics.syncGroundedQuaternion(dt: 0)
            physics.lookYawOff = degToRad(-24)
            physics.lookPitchOff = 0.08

        case "headturn":
            let startYaw = physics.state.yaw
            physics.mutateState { s in
                s.vx = sinf(s.yaw) * 6.5
                s.vz = -cosf(s.yaw) * 6.5
            }
            physics.addHeadLookDelta(dYaw: degToRad(48), dPitch: degToRad(-12))
            stepLoop(steps: Int(roundf(120 * 2.4))) { _, _ in [] }
            let turnedDeg = radToDeg(physics.state.yaw - startYaw)
            let headDeg = radToDeg(physics.lookYawOff)
            res.toastMessage = String(format: "FREE HEAD LOOK · HEAD %.1f° · SKI YAW Δ %.1f° (INDEPENDENT)", headDeg, turnedDeg)
            res.toastDuration = 9.9

        case "speedblur":
            physics.mutateState { s in
                s.vx = sinf(s.yaw) * 14.0
                s.vz = -cosf(s.yaw) * 14.0
            }
            stepLoop(steps: Int(roundf(120 * 2.1))) { stepIdx, _ in
                if stepIdx > 140 { return ["KeyW", "KeyD"] }
                return ["KeyW"]
            }
            let yaw = physics.state.yaw
            let fx = sinf(yaw), fz = -cosf(yaw)
            physics.mutateState { s in
                s.vx = fx * 21.8; s.vz = fz * 21.8; s.speed = 21.8
                s.edgeAngle = 0.42
            }
            physics.lookPitchOff = -0.34

        case "godrays":
            let sunH = max(1e-5, hypotf(sunDir.x, sunDir.z))
            let shx = sunDir.x / sunH, shz = sunDir.z / sunH
            let srx = -shz, srz = shx
            let tanEl = sunDir.y / sunH
            var bestX = physics.state.x, bestZ = physics.state.z
            var bestErr: Float = 1e9
            var found = false
            for c in colliders where c.kind == .tree {
                let hScale = c.h / 10.5
                let sc = c.r / 0.75
                if hScale < 0.78 { continue }
                let crownTargetY = c.y + (14.4 * hScale) * 0.58
                var dist: Float = 7.5
                while dist <= 24.0 {
                    let cx = c.x - shx * dist + srx * (0.98 * sc)
                    let cz = c.z - shz * dist + srz * (0.98 * sc)
                    let gy = terrainData.sampleHeight(x: cx, z: cz)
                    let rayYAtTree = (gy + 1.63) + tanEl * dist
                    let heightErr = abs(rayYAtTree - crownTargetY)
                    let score = heightErr + abs(dist - 13.5) * 0.14
                    if score < bestErr {
                        bestErr = score
                        bestX = cx
                        bestZ = cz
                        found = true
                    }
                    dist += 1.25
                }
            }
            if found {
                let t = terrainData.sampleTerrain(x: bestX, z: bestZ)
                physics.mutateState { s in
                    s.x = bestX; s.z = bestZ; s.y = t.y; s.terrainY = t.y; s.slopeDeg = t.slopeDeg
                    s.normal = t.normal
                    s.yaw = atan2f(sunDir.x, -sunDir.z)
                    s.vx = 0; s.vy = 0; s.vz = 0; s.speed = 0
                }
                physics.setSmoothNormal(t.normal)
                physics.syncGroundedQuaternion(dt: 0)
                physics.lookYawOff = degToRad(-2.5)
                physics.lookPitchOff = atan2f(sunDir.y, sunH) - 0.12
            }

        case "valley_lights":
            let vx: Float = -2600, vz: Float = -1680
            let targetX: Float = -2980, targetZ: Float = -4850
            let aimYaw = atan2f(targetX - vx, -(targetZ - vz))
            let t = terrainData.sampleTerrain(x: vx, z: vz)
            physics.mutateState { s in
                s.x = vx; s.z = vz; s.yaw = aimYaw
                s.y = t.y; s.terrainY = t.y; s.slopeDeg = t.slopeDeg; s.normal = t.normal
                s.vx = 0; s.vy = 0; s.vz = 0; s.speed = 0
            }
            physics.setSmoothNormal(t.normal)
            physics.syncGroundedQuaternion(dt: 0)
            physics.lookYawOff = 0
            physics.lookPitchOff = degToRad(-12.5)

        case "boulder", "serac":
            let targetKind: PropColliderKind = mode == "serac" ? .serac : .rock
            let list = colliders.filter { $0.kind == targetKind }
            var chosen: PropCollider?
            if mode == "serac" {
                chosen = list.first
            } else {
                var bestD: Float = 1e9
                for item in list {
                    let sc = item.r / 0.78
                    if sc < 3.2 { continue }
                    let los = sampleSunLOS(terrainData: terrainData, ox: item.x, oy: item.y + sc * 1.1, oz: item.z, sunDir: sunDir)
                    if los < 0.5 { continue }
                    let d = hypotf(item.x - physics.state.x, item.z - physics.state.z)
                    if d < bestD { bestD = d; chosen = item }
                }
                if chosen == nil { chosen = list.first }
            }
            if let best = chosen {
                let sc = best.r / (mode == "serac" ? 1.45 : 0.78)
                let offX: Float
                let offZ: Float
                let standDist: Float
                if mode == "serac" {
                    let ang: Float = 0.52
                    offX = sinf(ang)
                    offZ = cosf(ang)
                    standDist = sc * 3.6 + 6.5
                } else {
                    let sunH = max(1e-5, hypotf(sunDir.x, sunDir.z))
                    offX = (sunDir.x / sunH) * 0.72 - (sunDir.z / sunH) * 0.69
                    offZ = (sunDir.z / sunH) * 0.72 + (sunDir.x / sunH) * 0.69
                    standDist = sc * 2.55 + 4.4
                }
                let nx = best.x + offX * standDist
                let nz = best.z + offZ * standDist
                let t = terrainData.sampleTerrain(x: nx, z: nz)
                physics.mutateState { s in
                    s.x = nx; s.z = nz; s.y = t.y; s.terrainY = t.y; s.slopeDeg = t.slopeDeg
                    s.normal = t.normal
                    s.yaw = atan2f(best.x - nx, -(best.z - nz))
                    s.vx = 0; s.vy = 0; s.vz = 0; s.speed = 0
                }
                physics.setSmoothNormal(t.normal)
                physics.syncGroundedQuaternion(dt: 0)
                let targetY = best.y + sc * (mode == "serac" ? 0.68 : 0.45)
                physics.lookYawOff = 0
                physics.lookPitchOff = atan2f(targetY - (t.y + 1.63), standDist)
            }

        case "midi_glow":
            let spire = terrainData.meta.aiguilleSpire
            let nx = spire.x + 510, nz = spire.z + 185
            let t = terrainData.sampleTerrain(x: nx, z: nz)
            let spireY = terrainData.sampleHeight(x: spire.x, z: spire.z) + 56
            let dist = hypotf(spire.x - nx, spire.z - nz)
            physics.mutateState { s in
                s.x = nx; s.z = nz; s.y = t.y; s.terrainY = t.y; s.slopeDeg = t.slopeDeg
                s.normal = t.normal
                s.yaw = atan2f(spire.x - nx, -(spire.z - nz))
                s.vx = 0; s.vy = 0; s.vz = 0; s.speed = 0
            }
            physics.setSmoothNormal(t.normal)
            physics.syncGroundedQuaternion(dt: 0)
            physics.lookYawOff = 0
            physics.lookPitchOff = atan2f(spireY - (t.y + 1.63), dist)

        case "chesty_carve":
            let totalSteps = Int(roundf(120 * 2.6))
            physics.mutateState { s in
                s.vx = sinf(s.yaw) * 9.0
                s.vz = -cosf(s.yaw) * 9.0
            }
            stepLoop(steps: totalSteps) { _, sec in
                if sec < 0.72 { return ["KeyA"] }
                else if sec >= 0.90 && sec < 1.72 { return ["KeyD"] }
                else if sec >= 1.95 && sec < 2.25 { return ["KeyA"] }
                else if sec >= 2.30 { return ["KeyD"] }
                return []
            }
            let s = physics.state
            let tNow = terrainData.sampleTerrain(x: s.x, z: s.z)
            let fallYaw = atan2f(-tNow.gx, tNow.gz)
            let newYaw = fallYaw + 0.28
            physics.setSmoothNormal(tNow.normal)
            physics.mutateState { st in
                st.yaw = newYaw
                st.isSwitch = false
                st.edgeAngle = 0.56
                st.edgeDeg = 0.56 * 64.0
                st.fallAngleDeg = radToDeg(-0.28)
                st.stanceMode = "CARVE"
                st.insideUnweight = 0.64
                st.polePlantR = 0.52
                st.polePlantL = 0
                st.skatePush = 0
                st.bowWave = 0.94
                st.crouch = 0.26
                let fx = sinf(newYaw), fz = -cosf(newYaw)
                st.vx = fx * 17.5; st.vz = fz * 17.5; st.speed = 17.5
            }
            physics.syncGroundedQuaternion(dt: 0)
            let fx = sinf(newYaw), fz = -cosf(newYaw)
            let rx = cosf(newYaw), rz = sinf(newYaw)
            for k in 0..<26 {
                let f = Float(k) / 25.0
                let alongBack = 0.85 + f * 7.8
                let out = 0.45 + powf(f, 0.8) * 2.9
                let lift = 0.25 + sinf(min(1, f * 1.15) * .pi) * 1.9
                let qx = s.x - fx * alongBack - rx * out + (Float.random(in: 0..<1) - 0.5) * 0.35
                let qz = s.z - fz * alongBack - rz * out + (Float.random(in: 0..<1) - 0.5) * 0.35
                let qy = terrainData.sampleHeight(x: qx, z: qz)
                particles.emitPuff(
                    x: qx, y: qy + lift, z: qz,
                    vx: -rx * 2.2, vy: 1.2, vz: -rz * 2.2,
                    size: 0.72 + f * 1.25, expand: 2.0, peakAlpha: 0.78 - f * 0.26, life: 0.85, kind: 0
                )
            }
            physics.lookYawOff = 0
            physics.lookPitchOff = -0.14
            simTime = max(simTime, 42.0)

        case "skate_push":
            let s = physics.state
            let fx = sinf(s.yaw), fz = -cosf(s.yaw)
            let rx = cosf(s.yaw), rz = sinf(s.yaw)
            physics.mutateState { st in
                st.vx = fx * 3.8; st.vz = fz * 3.8; st.speed = 3.8
                st.skatePush = 1.0; st.skatePhase = 0.22
                st.polePlantL = 0; st.polePlantR = 0
                st.edgeAngle = 0; st.insideUnweight = 0
                st.bowWave = 0.14; st.crouch = 0.22
            }
            for side: Float in [-1, 1] {
                let bx = s.x + rx * side * 0.44 + fx * 0.92
                let bz = s.z + rz * side * 0.44 + fz * 0.92
                let by = terrainData.sampleHeight(x: bx, z: bz)
                for k in 0..<4 {
                    particles.emitPuff(
                        x: bx + (Float(k) - 1.5) * 0.06, y: by + 0.08, z: bz + Float(k % 2) * 0.06,
                        vx: 0, vy: 0.3, vz: 0,
                        size: 0.20, expand: 0.8, peakAlpha: 0.48, life: 0.25, kind: 0
                    )
                }
            }
            physics.lookYawOff = 0
            physics.lookPitchOff = -0.20
            simTime = max(simTime, 18.0)

        case "sluff_carve":
            let totalSteps = Int(roundf(120 * 2.0))
            physics.mutateState { s in
                s.vx = sinf(s.yaw) * 7.5
                s.vz = -cosf(s.yaw) * 7.5
            }
            stepLoop(steps: totalSteps) { _, sec in
                if sec < 0.8 { return ["KeyD"] }
                else if sec > 1.0 && sec < 1.7 { return ["KeyA"] }
                return []
            }
            let s = physics.state
            let t = terrainData.sampleTerrain(x: s.x, z: s.z)
            let gm = max(1e-5, hypotf(t.gx, t.gz))
            let dirX = -t.gx / gm, dirZ = -t.gz / gm
            let sideX = -dirZ, sideZ = dirX
            let newYaw = atan2f(dirX, -dirZ) - 0.24
            physics.setSmoothNormal(t.normal)
            physics.mutateState { st in
                st.yaw = newYaw
                st.edgeAngle = -0.52
                st.insideUnweight = -0.58
                st.polePlantL = 0.50; st.polePlantR = 0
                st.bowWave = 0.88; st.scissorSpread = 0.35; st.crouch = 0.24
                let fx = sinf(newYaw), fz = -cosf(newYaw)
                st.vx = fx * 15.8; st.vz = fz * 15.8; st.speed = 15.8
            }
            physics.syncGroundedQuaternion(dt: 0)
            for ribbon in 0..<2 {
                let sideSign: Float = ribbon == 0 ? -1 : 1
                let baseLat = sideSign * 1.15
                for k in 0..<28 {
                    let f = Float(k) / 27.0
                    let along = 2.0 + f * 15.5
                    let snake = baseLat + sinf(f * 4.8 + Float(ribbon) * 2.1) * 0.48 + sideSign * f * 1.35
                    let px = s.x + dirX * along + sideX * snake + (Float.random(in: 0..<1) - 0.5) * 0.22
                    let pz = s.z + dirZ * along + sideZ * snake + (Float.random(in: 0..<1) - 0.5) * 0.22
                    let py = terrainData.sampleHeight(x: px, z: pz)
                    let sz = 0.85 + sinf(f * .pi) * 0.95 + f * 0.65
                    let alpha = (0.34 - f * 0.14) * (0.45 + 0.55 * sinf(min(1, f * 3.0) * .pi * 0.5))
                    particles.emitPuff(
                        x: px, y: py + 0.18 + f * 0.20, z: pz,
                        vx: dirX * 8.0, vy: -1.0, vz: dirZ * 8.0,
                        size: sz, expand: 1.8, peakAlpha: alpha, life: 0.65, kind: 0
                    )
                }
            }
            physics.lookYawOff = 0
            physics.lookPitchOff = -0.14
            simTime = max(simTime, 64.0)

        case "switch_fakie":
            let totalSteps = Int(roundf(120 * 1.95))
            physics.mutateState { s in
                s.vx = sinf(s.yaw) * 8.5
                s.vz = -cosf(s.yaw) * 8.5
            }
            stepLoop(steps: totalSteps) { _, sec in
                if sec < 0.65 { return ["KeyA"] }
                else if sec >= 0.80 && sec < 1.50 { return ["KeyD"] }
                return []
            }
            _ = physics.toggleSwitch(time: simTime, instant: true)
            physics.mutateState { s in
                let fx = sinf(s.yaw), fz = -cosf(s.yaw)
                s.vx = -fx * 10.6; s.vz = -fz * 10.6; s.speed = 10.6
                s.edgeAngle = -0.26
                s.edgeDeg = -0.26 * 64.0
                s.stanceMode = "SWITCH"
                s.bowWave = 0.42
                s.crouch = 0.22
            }
            physics.lookYawOff = 0
            physics.lookPitchOff = -0.10
            res.toastMessage = "SWITCH STANCE (FAKIE · VIEW FLIPPED 180°)"
            res.toastDuration = 6.0
            simTime = max(simTime, 28.0)

        case "landing_stomp":
            let s = physics.state
            let fx = sinf(s.yaw), fz = -cosf(s.yaw)
            let rx = cosf(s.yaw), rz = sinf(s.yaw)
            physics.mutateState { st in
                st.vx = fx * 14.5; st.vz = fz * 14.5; st.speed = 14.5
                st.grounded = true
                st.crouch = 0.34
                st.landImpact = 0.72
                st.scissorSpread = 0.44
                st.edgeAngle = 0.16
                st.edgeDeg = 0.16 * 64.0
                st.bowWave = 0.88
                st.stanceMode = "STOMP"
            }
            physics.applyLandingStompState(landDisp: 0.215, landPitch: -0.065, landRoll: 0.032, scissorTarget: 0.44)
            for k in 0..<18 {
                let a = (Float(k) / 18.0) * .pi * 2
                let sideX = cosf(a) * (0.75 + Float(k % 3) * 0.22)
                let alongZ = sinf(a) * 0.85 + 0.55
                if abs(sideX) < 0.36 && alongZ > 0.2 { continue }
                let px = s.x + rx * sideX + fx * alongZ
                let pz = s.z + rz * sideX + fz * alongZ
                let py = terrainData.sampleHeight(x: px, z: pz)
                particles.emitPuff(
                    x: px, y: py + 0.16 + Float(k % 2) * 0.14, z: pz,
                    vx: rx * sideX * 1.4, vy: 0.7, vz: rz * sideX * 1.4,
                    size: 0.48 + Float(k % 3) * 0.14, expand: 1.4, peakAlpha: 0.44, life: 0.55, kind: 0
                )
            }
            physics.lookYawOff = 0
            physics.lookPitchOff = -0.18
            res.toastMessage = "STOMPED LANDING · SPRING-DAMPER ABSORB"
            res.toastDuration = 6.0
            simTime = max(simTime, 35.0)

        case "body_close", "body_back":
            res.inspectBodyView = mode
            physics.mutateState { s in
                s.edgeAngle = 0.22
                s.edgeDeg = 0.22 * 64.0
                s.polePlantR = 0.35
                s.crouch = 0.18
                s.speed = 6.5
                s.vx = sinf(s.yaw) * 6.5
                s.vz = -cosf(s.yaw) * 6.5
            }

        default:
            break
        }

        res.simTime = simTime
        return res
    }
}
