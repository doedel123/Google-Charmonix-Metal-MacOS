import Foundation
import simd

/// First-person / action-cam / clearance camera controller.
/// 1:1 port of camera kinematics, anatomical neck limits, landing smoothing,
/// and multi-point anti-clipping clearance from `src/physics.js`.
public final class CameraController {
    public var camMode: CameraMode = .gopro
    public var lookYawOff: Float = 0.0
    public var lookPitchOff: Float = -0.14

    public var smoothNormal = SIMD3<Float>(0, 1, 0)
    public var smoothCamRoll: Float = 0.0
    public var smoothSlopePitch: Float = 0.0
    public var switchYawBlend: Float = 0.0

    public var airBodyQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    public var landCamDelta = SIMD3<Float>.zero
    public var prevCamRel = SIMD3<Float>.zero
    public var prevCamWasAir: Bool = false
    public var hasPrevCamRel: Bool = false

    public init() {}

    public func resetForSpawn(
        yaw: Float,
        normal: SIMD3<Float>,
        gx: Float,
        gz: Float
    ) {
        lookYawOff = 0.0
        lookPitchOff = -0.14
        smoothNormal = normal
        smoothCamRoll = 0.0
        switchYawBlend = 0.0
        airBodyQuat = simd_quatf(angle: -yaw, axis: SIMD3<Float>(0, 1, 0))
        landCamDelta = .zero
        prevCamRel = .zero
        prevCamWasAir = false
        hasPrevCamRel = false

        let fwdX = sinf(yaw)
        let fwdZ = -cosf(yaw)
        smoothSlopePitch = -atan2f(-gx * fwdX - gz * fwdZ, 1.0)
    }

    public func resetHeadLook() {
        lookYawOff = 0.0
        lookPitchOff = -0.14
    }

    public func getLookBounds(mode: CameraMode, state: SkierState) -> (maxYaw: Float, minPitch: Float, maxPitch: Float) {
        let isChesty = (mode == .chesty)
        let maxYawDeg: Float = isChesty ? 58.0 : ((state.isSwitch && state.grounded) ? 118.0 : 82.0)
        let maxYawBase = degToRad(maxYawDeg)
        let maxYaw = !state.grounded ? min(maxYawBase, degToRad(72.0)) : maxYawBase
        let minPitch = isChesty ? degToRad(-42.0) : degToRad(-38.0)
        let maxPitch = isChesty ? degToRad(42.0) : degToRad(46.0)
        return (maxYaw, minPitch, maxPitch)
    }

    public func setCamMode(_ mode: CameraMode, state: SkierState) {
        camMode = mode
        clampLookOffsets(mode: mode, state: state)
    }

    public func clampLookOffsets(mode: CameraMode? = nil, state: SkierState) {
        let activeMode = mode ?? camMode
        let bounds = getLookBounds(mode: activeMode, state: state)
        lookYawOff = clamp(wrapAngle(lookYawOff), -bounds.maxYaw, bounds.maxYaw)
        lookPitchOff = clamp(lookPitchOff, bounds.minPitch, bounds.maxPitch)
    }

    public func addHeadLookDelta(dYaw: Float, dPitch: Float, state: SkierState) {
        if state.crashed { return }
        let bounds = getLookBounds(mode: camMode, state: state)
        lookYawOff = clamp(lookYawOff + dYaw, -bounds.maxYaw, bounds.maxYaw)
        lookPitchOff = clamp(lookPitchOff + dPitch, bounds.minPitch, bounds.maxPitch)
    }

    /// Multi-point near-frustum anti-mountain & boulder/serac clearance guard.
    public func enforceCameraClearance(
        camera: inout CameraState,
        terrainData: any TerrainQueryProtocol,
        colliders: [PropCollider],
        minClearance: Float = 0.42,
        checkProps: Bool = true
    ) {
        var p = camera.position
        let camFwd = camera.quaternion.act(SIMD3<Float>(0, 0, -1))
        let camRgt = camera.quaternion.act(SIMD3<Float>(1, 0, 0))

        let baseT = terrainData.sampleTerrain(x: p.x, z: p.z)
        let slopeTan = min(1.15, hypotf(baseT.gx, baseT.gz))
        let slopePad = slopeTan * 0.24

        var reqY = baseT.y + minClearance + slopePad
        let ringR = max(0.38, min(2.2, minClearance * 0.65))
        let offsets: [(Float, Float)] = [
            (ringR, 0), (-ringR, 0), (0, ringR), (0, -ringR),
            (camFwd.x * 0.34, camFwd.z * 0.34),
            (camFwd.x * 0.72, camFwd.z * 0.72),
            (camFwd.x * 0.34 + camRgt.x * 0.28, camFwd.z * 0.34 + camRgt.z * 0.28),
            (camFwd.x * 0.34 - camRgt.x * 0.28, camFwd.z * 0.34 - camRgt.z * 0.28)
        ]

        for i in 0..<offsets.count {
            let (ox, oz) = offsets[i]
            let gy = terrainData.sampleHeight(x: p.x + ox, z: p.z + oz)
            let isFwd = i >= 4
            let pad = isFwd ? max(0.26, minClearance * 0.58) : (minClearance * 0.72)
            let rayDrop = isFwd ? max(0.0, -camFwd.y * (i == 5 ? 0.72 : 0.34)) : 0.0
            let need = gy + pad + rayDrop
            if need > reqY { reqY = need }
        }

        if checkProps && !colliders.isEmpty {
            for c in colliders {
                if c.kind == .tree { continue }
                let dx = p.x - c.x
                let maxD = c.r + 0.85
                if abs(dx) > maxD { continue }
                let dz = p.z - c.z
                if abs(dz) > maxD { continue }
                if p.y > c.y + c.h + 0.55 { continue }
                let d = hypotf(dx, dz)
                if d < maxD {
                    if p.y > c.y + c.h * 0.65 {
                        reqY = max(reqY, c.y + c.h + 0.45)
                    } else if d > 0.001 {
                        p.x = c.x + (dx / d) * maxD
                        p.z = c.z + (dz / d) * maxD
                        let pushedY = terrainData.sampleHeight(x: p.x, z: p.z) + minClearance
                        if pushedY > reqY { reqY = pushedY }
                    } else {
                        reqY = max(reqY, c.y + c.h + 0.45)
                    }
                }
            }
        }

        if p.y < reqY {
            p.y = reqY
        }
        camera.position = p
    }

    public func applyCamera(
        camera: inout CameraState,
        mode: CameraMode,
        time: Float,
        dt: Float,
        state s: SkierState,
        terrainData: any TerrainQueryProtocol,
        colliders: [PropCollider],
        poleBobY: Float,
        switchHopT: Float,
        landDisp: Float,
        landPitch: Float,
        landRoll: Float
    ) {
        camMode = mode
        camera.mode = mode
        clampLookOffsets(mode: mode, state: s)

        let isChesty = (mode == .chesty)
        let poleBob: Float = s.grounded ? (poleBobY + s.stepBounce * 0.018) : 0.0
        let switchHopLift: Float = (s.grounded && switchHopT > 0) ? sinf(switchHopT * .pi) * 0.085 : 0.0
        let landDrop = max(-0.04, landDisp) * (isChesty ? 0.78 : 0.96)
        let eyeHeight = (isChesty
            ? (1.20 - s.crouch * 0.24 - landDrop)
            : (1.63 - s.crouch * 0.36 - landDrop)) + poleBob + switchHopLift

        if s.crashed {
            let localEye = s.quat.act(SIMD3<Float>(0, 0.56, 0))
            camera.position = SIMD3<Float>(
                s.x + localEye.x,
                max(s.terrainY + 0.52, s.y + localEye.y),
                s.z + localEye.z
            )
            camera.quaternion = s.quat
            enforceCameraClearance(camera: &camera, terrainData: terrainData, colliders: colliders, minClearance: 0.48, checkProps: true)
            camera.fovDeg += (94.0 - camera.fovDeg) * min(1.0, dt * 8.0)
            camera.lookYawOff = lookYawOff
            camera.lookPitchOff = lookPitchOff
            return
        }

        if s.grounded {
            let chatterAmp = s.surface.chatter * min(1.0, s.speed / 22.0) * (isChesty ? 0.025 : 0.022)
            let shakeEnv = s.camShake * 0.032
            let vibX = sinf(time * 57.0) * chatterAmp + (sinf(time * 31.0) * 0.65 + cosf(time * 49.0) * 0.35) * shakeEnv
            let vibY = cosf(time * 43.0) * chatterAmp * 0.8 + (cosf(time * 37.0) * 0.65 + sinf(time * 53.0) * 0.35) * shakeEnv

            switchYawBlend = 0.0
            let viewYaw = s.yaw + lookYawOff
            let fwdX = sinf(viewYaw)
            let fwdZ = -cosf(viewYaw)

            let skiFwdX = sinf(s.yaw), skiFwdZ = -cosf(s.yaw)
            let skiRgtX = cosf(s.yaw), skiRgtZ = sinf(s.yaw)

            let slopeAlong = -(terrainData.sampleHeight(x: s.x + fwdX * 8.0, z: s.z + fwdZ * 8.0) - s.y) / 8.0
            let pitchBias: Float = isChesty ? -0.22 : -0.14
            let targetSlopePitch = clamp(-atanf(slopeAlong) * 0.48 + pitchBias, -0.68, 0.46)
            smoothSlopePitch += (targetSlopePitch - smoothSlopePitch) * min(1.0, dt * 5.5)

            let minTotalPitch: Float = isChesty ? -0.78 : -0.74
            let maxTotalPitch: Float = isChesty ? 0.70 : 0.84
            let finalPitch = clamp(smoothSlopePitch + lookPitchOff + landPitch, minTotalPitch, maxTotalPitch)

            if isChesty {
                let latShift = s.edgeAngle * 0.16
                let chestPush = 0.12 + s.skatePush * sinf(s.skatePhase) * 0.025
                let baseX = s.x + skiFwdX * chestPush + skiRgtX * latShift
                let baseZ = s.z + skiFwdZ * chestPush + skiRgtZ * latShift
                camera.position = SIMD3<Float>(
                    baseX + fwdX * 0.13 + vibX,
                    s.y + eyeHeight + vibY,
                    baseZ + fwdZ * 0.13
                )
            } else {
                let latCamShift = s.edgeAngle * 0.18
                let neckBack = 0.03 + s.crouch * 0.02
                let downPush = smoothstep(0.28, 0.74, -finalPitch)
                let eyeOrbit = 0.22 + downPush * 0.14
                let sideAbs = min(1.0, abs(lookYawOff) / 1.4)
                let eyeLift = downPush * 0.07 + sideAbs * 0.04
                let neckX = s.x - skiFwdX * neckBack + skiRgtX * latCamShift
                let neckZ = s.z - skiFwdZ * neckBack + skiRgtZ * latCamShift
                camera.position = SIMD3<Float>(
                    neckX + fwdX * eyeOrbit + vibX,
                    s.y + eyeHeight + eyeLift + vibY,
                    neckZ + fwdZ * eyeOrbit
                )
            }

            let rawRel = SIMD3<Float>(camera.position.x - s.x, camera.position.y - s.y, camera.position.z - s.z)
            if prevCamWasAir && hasPrevCamRel {
                landCamDelta = SIMD3<Float>(
                    clamp(prevCamRel.x - rawRel.x, -0.55, 0.55),
                    clamp(prevCamRel.y - rawRel.y, -0.45, 0.45),
                    clamp(prevCamRel.z - rawRel.z, -0.55, 0.55)
                )
            }
            landCamDelta *= expf(-dt * 15.0)
            if simd_length_squared(landCamDelta) > 1e-6 {
                camera.position += landCamDelta
            }
            prevCamRel = SIMD3<Float>(camera.position.x - s.x, camera.position.y - s.y, camera.position.z - s.z)
            prevCamWasAir = false
            hasPrevCamRel = true

            let targetRoll = clamp(-s.edgeAngle * (isChesty ? 0.36 : 0.32), -0.38, 0.34)
            smoothCamRoll += (targetRoll - smoothCamRoll) * min(1.0, dt * 8.5)

            camera.quaternion = quatFromEulerYXZ(
                pitch: finalPitch,
                yaw: -viewYaw,
                roll: smoothCamRoll + landRoll + sinf(time * 31.0) * chatterAmp * 0.55
            )
            enforceCameraClearance(camera: &camera, terrainData: terrainData, colliders: colliders, minClearance: isChesty ? 0.56 : 0.62, checkProps: true)
        } else {
            let lFwdX = sinf(lookYawOff)
            let lFwdZ = -cosf(lookYawOff)
            let viewPitch = clamp(smoothSlopePitch + lookPitchOff, isChesty ? -0.78 : -0.74, isChesty ? 0.70 : 0.84)
            var localEye: SIMD3<Float>
            if isChesty {
                let latShift = s.edgeAngle * 0.16
                localEye = SIMD3<Float>(latShift + lFwdX * 0.13, eyeHeight, -0.12 + lFwdZ * 0.13)
            } else {
                let latCamShift = s.edgeAngle * 0.18
                let neckBack = 0.03 + s.crouch * 0.02
                let downPush = smoothstep(0.28, 0.74, -viewPitch)
                let eyeOrbit = 0.22 + downPush * 0.14
                let sideAbs = min(1.0, abs(lookYawOff) / 1.4)
                let eyeLift = downPush * 0.07 + sideAbs * 0.04
                localEye = SIMD3<Float>(latCamShift + lFwdX * eyeOrbit, eyeHeight + eyeLift, neckBack + lFwdZ * eyeOrbit)
            }
            let worldEyeOffset = airBodyQuat.act(localEye)
            camera.position = SIMD3<Float>(
                s.x + worldEyeOffset.x,
                max(s.terrainY + 0.62, s.y + worldEyeOffset.y),
                s.z + worldEyeOffset.z
            )
            prevCamRel = SIMD3<Float>(camera.position.x - s.x, camera.position.y - s.y, camera.position.z - s.z)
            prevCamWasAir = true
            hasPrevCamRel = true
            landCamDelta = .zero

            let fwdAirX = sinf(s.yaw + lookYawOff), fwdAirZ = -cosf(s.yaw + lookYawOff)
            let slopeAhead = -(terrainData.sampleHeight(x: s.x + fwdAirX * 8.0, z: s.z + fwdAirZ * 8.0) - s.terrainY) / 8.0
            let pitchBias: Float = isChesty ? -0.22 : -0.14
            let targetAirSlopePitch = clamp(-atanf(slopeAhead) * 0.48 + pitchBias, -0.68, 0.46)
            smoothSlopePitch += (targetAirSlopePitch - smoothSlopePitch) * min(1.0, dt * 3.0)
            smoothCamRoll *= expf(-dt * 4.5)

            let qPitchTilt = quatFromEulerYXZ(pitch: viewPitch, yaw: -lookYawOff, roll: smoothCamRoll)
            camera.quaternion = simd_normalize(airBodyQuat * qPitchTilt)
            enforceCameraClearance(camera: &camera, terrainData: terrainData, colliders: colliders, minClearance: 0.60, checkProps: true)
        }

        let baseFov: Float = isChesty ? 92.0 : (mode == .gopro ? 86.0 : 74.0)
        let targetFov = baseFov + min(16.0, (s.speed * 3.6) * 0.145) + (s.grounded ? 0.0 : 4.0)
        camera.fovDeg += (targetFov - camera.fovDeg) * min(1.0, dt * 6.0)
        camera.lookYawOff = lookYawOff
        camera.lookPitchOff = lookPitchOff
    }
}
