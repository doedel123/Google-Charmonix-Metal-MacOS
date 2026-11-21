import Foundation
import Metal
import simd

public struct SkiRigDiagnostics: Sendable {
    public var minClearLCm: Float = 0
    public var minClearRCm: Float = 0
    public var minVertLCm: Float = 0
    public var minVertRCm: Float = 0
    public var powSinkL: Float = 0
    public var powSinkR: Float = 0
    public var concaveFlexL: Float = 0
    public var concaveFlexR: Float = 0
    public var dyL: Float = 0
    public var dyR: Float = 0
    public var dpLDeg: Float = 0
    public var dpRDeg: Float = 0
    public var surface: String = "powder"
}

final class SkiRigAnimator {
    private let rig: SkierRigScene
    private weak var terrainData: (any TerrainQueryProtocol)?

    private var snowDustLevel: Float = 0.0
    private var grabBlend: Float = 0.0
    private var airRigBlend: Float = 0.0
    private var smoothFlexL: Float = 0.0
    private var smoothFlexR: Float = 0.0
    private var flowDist: Float = 0.0
    private var smoothPowSinkL: Float = 0.0
    private var smoothPowSinkR: Float = 0.0
    private var lastPosX: Float = .nan
    private var lastPosZ: Float = .nan
    private var bodyQSmooth = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var bodyQInit: Bool = false

    // Shared foreground bounce/rim/frost parameters
    var fgBounceCol = SIMD3<Float>(0.68, 0.78, 0.94)
    var fgRimCol = SIMD3<Float>(0.86, 0.92, 1.00)
    var fgBounceAmt: Float = 0.24
    var fgFrostGlobal: Float = 0.0

    private(set) var lastDiag = SkiRigDiagnostics()

    private let fitT: [Float] = [0.06, 0.18, 0.44, 0.64, 0.82, 0.96]
    private let verifyT: [Float] = [0.06, 0.18, 0.32, 0.44, 0.64, 0.82, 0.96]

    init(rig: SkierRigScene, terrainData: (any TerrainQueryProtocol)?) {
        self.rig = rig
        self.terrainData = terrainData
    }

    private func evalFlexAt(_ t: Float, _ f: Float) -> Float {
        if t > 0.56 {
            let u = (t - 0.56) / 0.44
            return u * u * f
        }
        if t < 0.24 {
            let u = (0.24 - t) / 0.24
            return u * u * f * 0.35
        }
        return 0
    }

    private func updateSkiFlex(_ ski: SkiAssembly, flexAmt: Float, chatterPhase: Float) {
        let count = ski.bladeVertices.count
        for i in 0..<count {
            let t = ski.tVals[i]
            var dy: Float = 0
            if t > 0.56 {
                let u = (t - 0.56) / 0.44
                dy = u * u * flexAmt + u * u * chatterPhase
            } else if t < 0.24 {
                let u = (0.24 - t) / 0.24
                dy = u * u * flexAmt * 0.35
            }
            ski.bladeVertices[i].position.y = ski.baseY[i] + dy
        }
        ski.bladeVertices.withUnsafeBytes { ptr in
            if let base = ptr.baseAddress {
                memcpy(ski.bladePart.mesh.vertexBuffer.contents(), base, count * MemoryLayout<StandardVertex>.stride)
            }
        }

        if ski.dustPart.material.opacity > 0.01 {
            let dCount = ski.dustVertices.count
            for i in 0..<dCount {
                let t = ski.dTVals[i]
                let u = max(0, (t - 0.56) / 0.44)
                ski.dustVertices[i].position.y = ski.dBaseY[i] + u * u * flexAmt + u * u * chatterPhase
            }
            ski.dustVertices.withUnsafeBytes { ptr in
                if let base = ptr.baseAddress {
                    memcpy(ski.dustPart.mesh.vertexBuffer.contents(), base, dCount * MemoryLayout<StandardVertex>.stride)
                }
            }
        }
    }

    private func placeLimbSegment(
        _ node: RigNode,
        pA: SIMD3<Float>,
        pB: SIMD3<Float>,
        rScaleXZ: Float = 1.0,
        hintX: Float = 1,
        hintY: Float = 0,
        hintZ: Float = 0
    ) {
        let dirRaw = pB - pA
        let len = simd_length(dirRaw)
        if len < 0.001 { return }
        let mid = (pA + pB) * 0.5
        let dir = dirRaw / len
        var hint = SIMD3<Float>(hintX, hintY, hintZ)
        if abs(simd_dot(dir, hint)) > 0.94 {
            hint = SIMD3<Float>(0, 0, -1)
        }
        let segZ = simd_normalize(simd_cross(hint, dir))
        let segX = simd_normalize(simd_cross(dir, segZ))
        node.position = mid
        node.setQuaternion(quatFromBasis(right: segX, up: dir, forwardZ: segZ))
        node.scale = SIMD3<Float>(rScaleXZ, len, rScaleXZ)
    }

    func update(
        state: SkierState,
        dt: Float,
        isThirdPerson: Bool,
        time: Float,
        sunDir: SIMD3<Float>,
        isStorm: Bool,
        lookYawOff: Float,
        mtnSunVis: Float,
        isAlpenglow: Bool,
        isChesty: Bool,
        preset: LightingPreset
    ) {
        rig.fpGroup.position = SIMD3<Float>(state.x, state.y, state.z)
        rig.fpGroup.setQuaternion(state.quat)

        let teleported = lastPosX.isNaN || hypotf(state.x - lastPosX, state.z - lastPosZ) > 3.5 || dt >= 0.45
        lastPosX = state.x
        lastPosZ = state.z

        if state.grounded && !state.crashed {
            flowDist += state.speed * dt
        }

        let surfId = state.surface.id
        let isPowder = surfId == "powder"
        let isCorn = surfId == "corn"

        if state.grounded && isPowder && state.speed > 5 {
            snowDustLevel = min(0.18, snowDustLevel + dt * (0.06 + abs(state.slipAngle) * 0.16))
        } else {
            snowDustLevel = max(0.0, snowDustLevel - dt * (state.grounded ? 0.08 : 0.25))
        }
        rig.leftSki.dustPart.material.opacity = snowDustLevel
        rig.rightSki.dustPart.material.opacity = snowDustLevel
        rig.leftSki.dustPart.visible = snowDustLevel > 0.01
        rig.rightSki.dustPart.visible = snowDustLevel > 0.01

        // CPU cloudShadowAt() matching terrain_renderer.js 1:1
        let fract: (Float) -> Float = { v in v - floorf(v) }
        let hash21: (Float, Float) -> Float = { px, py in
            var x = fract(px * 123.34)
            var y = fract(py * 456.21)
            let d = x * (x + 45.32) + y * (y + 45.32)
            x += d; y += d
            return fract(x * y)
        }
        let vnoise2: (Float, Float) -> Float = { px, py in
            let ix = floorf(px), iy = floorf(py)
            let fx = px - ix, fy = py - iy
            let ux = fx * fx * (3.0 - 2.0 * fx)
            let uy = fy * fy * (3.0 - 2.0 * fy)
            let a = hash21(ix, iy)
            let b = hash21(ix + 1, iy)
            let c = hash21(ix, iy + 1)
            let d = hash21(ix + 1, iy + 1)
            return (a * (1 - ux) + b * ux) * (1 - uy) + (c * (1 - ux) + d * ux) * uy
        }
        let cpX = (state.x - time * 22.0) * 0.00052
        let cpZ = (state.z - time * 8.5) * 0.00052
        let cloudN = vnoise2(cpX, cpZ) * 0.62 + vnoise2(cpX * 2.15 + 7.3, cpZ * 2.15 - 4.1) * 0.38
        let cloudMask = smoothstep(0.50, 0.78, cloudN) * preset.cloudShadowStrength * (isStorm ? 0.2 : 1.0)

        let directVis = (isStorm ? 0.30 : mtnSunVis) * (1.0 - cloudMask)
        let nVec = simd_length_squared(state.normal) > 1e-6 ? simd_normalize(state.normal) : SIMD3<Float>(0, 1, 0)
        let dotNL = simd_dot(nVec, sunDir)
        let ndl = max(0, dotNL)
        let wrap = max(0, (dotNL + 0.06) / 1.06)
        let diffFactor = wrap * 0.14 + ndl * 0.86
        let skyFactor = clamp(0.42 + 0.58 * nVec.y, 0.16, 1.0)
        let bounceScaled = preset.groundBounceColor * 0.64
        let skyFillCol = lerp(bounceScaled, preset.shadowSkyColor, skyFactor) * (preset.shadowFillStrength * 0.84)
        let dirLightCol = preset.sunColor * (preset.sunIntensity * diffFactor * directVis)

        let snowAlb = SIMD3<Float>(0.77, 0.825, 0.90)
        let snowLit = (skyFillCol * 1.00 + dirLightCol * 1.02) * snowAlb
        let snowShad = (skyFillCol * 0.90 + dirLightCol * 0.52) * snowAlb
        let sssGain = (0.10 + 0.32 * directVis) * (isAlpenglow ? 0.45 : 0.72)
        let snowSSS = preset.sssColor * sssGain

        // Dynamic ski edging, outside-ski weighting & scissor-spread
        let edge = state.edgeAngle
        let unweight = state.insideUnweight
        let scissor = state.scissorSpread
        let rollL = -edge * (edge > 0 ? 0.96 : 0.86)
        let rollR = -edge * (edge < 0 ? 0.96 : 0.86)

        let airTarget: Float = state.grounded ? 0.0 : 1.0
        airRigBlend = teleported ? airTarget : (airRigBlend + (airTarget - airRigBlend) * (1.0 - expf(-dt * (state.grounded ? 14.0 : 18.0))))
        let crouchOffset = -state.crouch * 0.14
        let airPull = airRigBlend * (0.12 + grabBlend * 0.40)
        grabBlend += ((state.grabbing && !state.grounded ? 1.0 : 0.0) - grabBlend) * min(1.0, dt * 14.0)

        rig.skiPivot.position = SIMD3<Float>(0, airRigBlend * (crouchOffset + airPull), -airRigBlend * grabBlend * 0.22)
        rig.skiPivot.setEuler(
            airRigBlend * (grabBlend * 0.42 + state.foreAft * 0.18),
            airRigBlend * (grabBlend * 0.46 + state.shiftyAngle),
            0
        )

        let fwdX = sinf(state.yaw), fwdZ = -cosf(state.yaw)
        let rgtX = cosf(state.yaw), rgtZ = sinf(state.yaw)
        let vibL: Float = state.grounded ? (sinf(time * 75.0) * 0.0040 * state.surface.chatter * min(1, state.speed / 18.0)) : 0
        let vibR: Float = state.grounded ? (sinf(time * 69.0 + 1.9) * 0.0040 * state.surface.chatter * min(1, state.speed / 18.0)) : 0

        let leadL = state.turnLead * 0.09 + scissor * 0.11
        let leadR = -state.turnLead * 0.09 - scissor * 0.11

        let rigRight = state.quat.act(SIMD3<Float>(1, 0, 0))
        let rigUp = state.quat.act(SIMD3<Float>(0, 1, 0))
        let rigFwd = state.quat.act(SIMD3<Float>(0, 0, -1))
        let upY = max(0.32, rigUp.y)

        let cushion: Float = isPowder ? 0.020 : 0.028
        var pitchErrL: Float = 0, pitchErrR: Float = 0, dyL: Float = 0, dyR: Float = 0
        var concaveL: Float = 0, concaveR: Float = 0

        struct Pass1Result {
            var dp: Float
            var concaveFlex: Float
            var errCopy: [Float]
            var cosR: Float
            var absSinR: Float
        }

        let solveSkiPass1: (Float, Float, Float, any TerrainQueryProtocol) -> Pass1Result = { sx, leadZ, roll, td in
            let cosR = cosf(roll)
            let absSinR = abs(sinf(roll))
            let edgeSide: Float = roll < -0.015 ? 1 : (roll > 0.015 ? -1 : 0)
            var errBuf = [Float](repeating: 0, count: self.fitT.count)
            for k in 0..<self.fitT.count {
                let t = self.fitT[k]
                let zk = (SkiRigConstants.bootT - t) * SkiRigConstants.skiLen
                let along = -(zk + leadZ)
                let wx = state.x + rigRight.x * sx + rigFwd.x * along
                let wy = state.y + rigRight.y * sx + rigFwd.y * along
                let wz = state.z + rigRight.z * sx + rigFwd.z * along
                var maxVertErr = td.sampleHeight(x: wx, z: wz) - wy
                let hw = evalSkiHalfWidth(t)
                if edgeSide != 0 {
                    let eShift = edgeSide * hw * cosR
                    let errE = td.sampleHeight(x: wx + rigRight.x * eShift, z: wz + rigRight.z * eShift) - (wy + rigRight.y * eShift)
                    if errE > maxVertErr { maxVertErr = errE }
                } else {
                    let errL = td.sampleHeight(x: wx - rigRight.x * hw, z: wz - rigRight.z * hw) - (wy - rigRight.y * hw)
                    let errR = td.sampleHeight(x: wx + rigRight.x * hw, z: wz + rigRight.z * hw) - (wy + rigRight.y * hw)
                    if errL > maxVertErr { maxVertErr = errL }
                    if errR > maxVertErr { maxVertErr = errR }
                }
                errBuf[k] = maxVertErr * upY
            }
            let span = (self.fitT[4] - self.fitT[1]) * SkiRigConstants.skiLen
            let dp = clamp(atan2f(errBuf[4] - errBuf[1], span), -0.44, 0.44)
            let concave = max(0, 0.5 * (errBuf[4] + errBuf[1]) - errBuf[2])
            let concaveFlex = min(0.085, concave * 1.05)
            return Pass1Result(dp: dp, concaveFlex: concaveFlex, errCopy: errBuf, cosR: cosR, absSinR: absSinR)
        }

        var p1L: Pass1Result?
        var p1R: Pass1Result?
        if state.grounded, let td = terrainData {
            let l = solveSkiPass1(-SkiRigConstants.skiSep, leadL, rollL, td)
            let r = solveSkiPass1(SkiRigConstants.skiSep, leadR, rollR, td)
            p1L = l
            p1R = r
            pitchErrL = l.dp
            pitchErrR = r.dp
            concaveL = l.concaveFlex
            concaveR = r.concaveFlex
        }

        let baseFlex: Float = state.grounded
            ? (state.landImpact * 0.095 + abs(edge) * 0.036 + min(0.030, state.speed * 0.0011))
            : (grabBlend * 0.026)
        let loadBiasL: Float = state.grounded ? (1.0 + clamp(unweight * 0.38, -0.30, 0.42)) : 1.0
        let loadBiasR: Float = state.grounded ? (1.0 - clamp(unweight * 0.38, -0.42, 0.30)) : 1.0
        let targetFlexL = baseFlex * loadBiasL + concaveL
        let targetFlexR = baseFlex * loadBiasR + concaveR
        if teleported {
            smoothFlexL = targetFlexL
            smoothFlexR = targetFlexR
        } else {
            let flexAlpha = min(1.0, dt * 16.0)
            smoothFlexL += (targetFlexL - smoothFlexL) * flexAlpha
            smoothFlexR += (targetFlexR - smoothFlexR) * flexAlpha
        }
        let chatterL: Float = state.grounded ? sinf(time * 72.0) * 0.010 * state.surface.chatter * min(1, state.speed / 16.0) * (0.85 + 0.3 * loadBiasL) : 0
        let chatterR: Float = state.grounded ? sinf(time * 67.0 + 2.1) * 0.010 * state.surface.chatter * min(1, state.speed / 16.0) * (0.85 + 0.3 * loadBiasR) : 0
        updateSkiFlex(rig.leftSki, flexAmt: smoothFlexL, chatterPhase: chatterL)
        updateSkiFlex(rig.rightSki, flexAmt: smoothFlexR, chatterPhase: chatterR)

        let computeNeedDy: (Pass1Result, Float) -> Float = { p1, flexVal in
            let sinDp = sinf(p1.dp)
            let cosDp = cosf(p1.dp)
            var needDy: Float = -1e9
            for k in 0..<self.fitT.count {
                let t = self.fitT[k]
                let zk = (SkiRigConstants.bootT - t) * SkiRigConstants.skiLen
                let hw = evalSkiHalfWidth(t)
                let coreThick = 0.012 + 0.007 * sinf(t * .pi)
                let y0 = evalRockerY(t) + self.evalFlexAt(t, flexVal) + coreThick * 0.50
                let yRotZ = -hw * p1.absSinR + y0 * p1.cosR
                let edgeLocalY = yRotZ * cosDp - zk * sinDp
                let req = (p1.errCopy[k] + cushion) - edgeLocalY
                if req > needDy { needDy = req }
            }
            return clamp(needDy, -0.38, 0.48)
        }

        if state.grounded, terrainData != nil, let p1L, let p1R {
            dyL = computeNeedDy(p1L, smoothFlexL)
            dyR = computeNeedDy(p1R, smoothFlexR)
        }

        let unweightLiftL: Float = (state.grounded && unweight < 0) ? abs(unweight) * 0.036 : 0
        let unweightLiftR: Float = (state.grounded && unweight > 0) ? abs(unweight) * 0.036 : 0
        let unweightPitchL: Float = (state.grounded && unweight < 0) ? abs(unweight) * 0.045 : 0
        let unweightPitchR: Float = (state.grounded && unweight > 0) ? abs(unweight) * 0.045 : 0

        rig.leftSki.node.setEuler(pitchErrL + unweightPitchL, 0, rollL)
        rig.rightSki.node.setEuler(pitchErrR + unweightPitchR, 0, rollR)
        rig.leftSki.node.position = SIMD3<Float>(-SkiRigConstants.skiSep, dyL + unweightLiftL + vibL, leadL)
        rig.rightSki.node.position = SIMD3<Float>(SkiRigConstants.skiSep, dyR + unweightLiftR + vibR, leadR)

        rig.fpGroup.updateWorldMatrices()

        var minClearL: Float = 0
        var minClearR: Float = 0
        if state.grounded, let td = terrainData {
            let targetVertClear = cushion / upY
            let enforceClearance: (SkiAssembly, Float) -> Float = { skiObj, flexVal in
                for _ in 0..<2 {
                    var maxDeficit: Float = 0
                    let mw = skiObj.node.worldMatrix
                    for t in self.verifyT {
                        let zk = (SkiRigConstants.bootT - t) * SkiRigConstants.skiLen
                        let hw = evalSkiHalfWidth(t)
                        let coreThick = 0.012 + 0.007 * sinf(t * .pi)
                        let yBevel = evalRockerY(t) + self.evalFlexAt(t, flexVal) + coreThick * 0.50
                        for s in -1...1 {
                            let localPt = SIMD4<Float>(Float(s) * hw, yBevel, zk, 1.0)
                            let wp = mw * localPt
                            let gy = td.sampleHeight(x: wp.x, z: wp.z)
                            let clear = wp.y - gy
                            let deficit = targetVertClear - clear
                            if deficit > maxDeficit { maxDeficit = deficit }
                        }
                    }
                    if maxDeficit > 0.001 {
                        skiObj.node.position.y += min(0.45, maxDeficit * upY)
                        self.rig.fpGroup.updateWorldMatrices()
                    } else {
                        break
                    }
                }
                var finalMin: Float = 1e9
                let mw = skiObj.node.worldMatrix
                for t in self.verifyT {
                    let zk = (SkiRigConstants.bootT - t) * SkiRigConstants.skiLen
                    let hw = evalSkiHalfWidth(t)
                    let coreThick = 0.012 + 0.007 * sinf(t * .pi)
                    let yBevel = evalRockerY(t) + self.evalFlexAt(t, flexVal) + coreThick * 0.50
                    for s in -1...1 {
                        let wp = mw * SIMD4<Float>(Float(s) * hw, yBevel, zk, 1.0)
                        let clear = wp.y - td.sampleHeight(x: wp.x, z: wp.z)
                        if clear < finalMin { finalMin = clear }
                    }
                }
                return finalMin
            }
            minClearL = enforceClearance(rig.leftSki, smoothFlexL)
            minClearR = enforceClearance(rig.rightSki, smoothFlexR)
        }

        // Dynamic deep-powder immersion per ski
        var targetPowL: Float = 0.0
        var targetPowR: Float = 0.0
        if state.grounded && !state.crashed {
            let fore = state.foreAft
            let absEdge = abs(edge)
            let slipFactor = abs(sinf(state.slipAngle))
            if isPowder {
                let speedRelief = clamp((state.speed - 5.0) / 16.0, 0, 1) * 0.25
                let backFloat = max(0, -fore) * 0.24
                let noseDive = max(0, fore) * 0.18
                let stomp = state.landImpact * 0.30
                let basePow = 0.62 - speedRelief - backFloat + noseDive + stomp + absEdge * 0.14 + slipFactor * 0.16
                targetPowL = clamp(basePow + unweight * 0.14 + clamp(concaveL * 3.5, 0, 0.20), 0.22, 0.94)
                targetPowR = clamp(basePow - unweight * 0.14 + clamp(concaveR * 3.5, 0, 0.20), 0.22, 0.94)
            } else if isCorn {
                let cornSlush = clamp((absEdge * 0.26 + slipFactor * 0.24) * min(1, state.speed / 8.0), 0, 0.28)
                targetPowL = cornSlush
                targetPowR = cornSlush
            }
        }
        if teleported {
            smoothPowSinkL = targetPowL
            smoothPowSinkR = targetPowR
        } else {
            let powAlpha = 1.0 - expf(-dt * (state.grounded ? 8.5 : 5.5))
            smoothPowSinkL += (targetPowL - smoothPowSinkL) * powAlpha
            smoothPowSinkR += (targetPowR - smoothPowSinkR) * powAlpha
            if smoothPowSinkL < 0.002 && targetPowL == 0 { smoothPowSinkL = 0 }
            if smoothPowSinkR < 0.002 && targetPowR == 0 { smoothPowSinkR = 0 }
        }

        // Shared foreground bounce & frost
        let powAvg = 0.5 * (smoothPowSinkL + smoothPowSinkR)
        fgFrostGlobal = clamp(snowDustLevel * 2.1 + powAvg * 0.42, 0.0, 0.72)
        let sunFill = isStorm ? 0.30 : (0.24 + 0.24 * mtnSunVis)
        fgBounceAmt = isAlpenglow ? sunFill * (0.48 + 0.38 * mtnSunVis) : sunFill
        fgBounceCol = lerp(preset.groundBounceColor, preset.sunColor, isStorm ? 0.12 : 0.42)
        fgRimCol = lerp(preset.shadowSkyColor, preset.sunColor, isStorm ? 0.22 : 0.48)

        let syncPowder: (SkiAssembly, Float, Float) -> Void = { skiObj, sinkVal, edgeVal in
            skiObj.powSink = sinkVal
            skiObj.edgeRoll = edgeVal
            skiObj.flowDist = self.flowDist
            skiObj.snowLitCol = snowLit
            skiObj.snowShadCol = snowShad
            skiObj.snowSSSCol = snowSSS
            skiObj.powSunVis = mtnSunVis
            skiObj.powStorm = isStorm ? 1.0 : 0.0
            skiObj.bladePart.material.depthWrite = sinkVal < 0.02
            skiObj.bladePart.castShadow = !(state.grounded && sinkVal > 0.25)
        }
        syncPowder(rig.leftSki, smoothPowSinkL, -rollL / 0.96)
        syncPowder(rig.rightSki, smoothPowSinkR, -rollR / 0.96)

        // Bow-wave snow wash
        let rawBow: Float = (state.grounded && !state.crashed && state.speed > 3.5) ? state.bowWave : 0.0
        let syncBow: (SkiAssembly, Float, Float) -> Void = { skiObj, scaleBias, skiRoll in
            let val = clamp(rawBow * scaleBias, 0, 1.15)
            skiObj.bowVisible = val > 0.035
            if skiObj.bowVisible {
                skiObj.bowIntensity = val
                skiObj.bowEdgeBias = edge
                skiObj.bowSkiRoll = skiRoll
                skiObj.bowSunVis = directVis
            }
        }
        syncBow(rig.leftSki, edge > 0.08 ? 1.08 : (edge < -0.08 ? 0.44 : 0.92), rollL)
        syncBow(rig.rightSki, edge < -0.08 ? 1.08 : (edge > 0.08 ? 0.44 : 0.92), rollR)

        lastDiag = SkiRigDiagnostics(
            minClearLCm: (minClearL * upY * 100.0 * 100.0).rounded() / 100.0,
            minClearRCm: (minClearR * upY * 100.0 * 100.0).rounded() / 100.0,
            minVertLCm: (minClearL * 100.0 * 100.0).rounded() / 100.0,
            minVertRCm: (minClearR * 100.0 * 100.0).rounded() / 100.0,
            powSinkL: (smoothPowSinkL * 1000.0).rounded() / 1000.0,
            powSinkR: (smoothPowSinkR * 1000.0).rounded() / 1000.0,
            concaveFlexL: (concaveL * 100.0 * 100.0).rounded() / 100.0,
            concaveFlexR: (concaveR * 100.0 * 100.0).rounded() / 100.0,
            dyL: (rig.leftSki.node.position.y * 100.0 * 100.0).rounded() / 100.0,
            dyR: (rig.rightSki.node.position.y * 100.0 * 100.0).rounded() / 100.0,
            dpLDeg: (radToDeg(pitchErrL) * 10.0).rounded() / 10.0,
            dpRDeg: (radToDeg(pitchErrR) * 10.0).rounded() / 10.0,
            surface: surfId
        )

        // Update World-Space Ground-Snapped Hourglass Shadow Ribbons
        let clearance = max(0, state.y - state.terrainY)
        let showSkiRibbons = !state.crashed && terrainData != nil && clearance < 0.95
        rig.leftShadowRib.visible = showSkiRibbons
        rig.rightShadowRib.visible = showSkiRibbons
        if showSkiRibbons, let td = terrainData {
            let sunY = max(0.28, sunDir.y)
            let castX = -sunDir.x / sunY
            let castZ = -sunDir.z / sunY
            let castLen = hypotf(castX, castZ)
            let maxCast: Float = 2.4
            let cX = castLen > maxCast ? (castX / castLen) * maxCast : castX
            let cZ = castLen > maxCast ? (castZ / castLen) * maxCast : castZ

            let updateRibbon: (ShadowRibbonMesh, Float, Float, Float, Float, Float) -> Void = { rib, latX, leadZ, skiFlex, extraLift, powSinkVal in
                let useRig = state.grounded
                let rX = useRig ? rigRight.x : rgtX
                let rZ = useRig ? rigRight.z : rgtZ
                let fX = useRig ? rigFwd.x : fwdX
                let fZ = useRig ? rigFwd.z : fwdZ
                let wLen = max(1e-5, hypotf(rX, rZ))
                let rnX = rX / wLen, rnZ = rZ / wLen

                for i in 0...rib.segs {
                    let t = Float(i) / Float(rib.segs)
                    let localForward = (t - SkiRigConstants.bootT) * SkiRigConstants.skiLen - leadZ
                    let rockLift = evalRockerY(t)
                    let flexLift: Float = skiFlex * (t > 0.56 ? powf((t - 0.56) / 0.44, 2) : 0)
                    let lift = max(0.008, clearance + extraLift + rockLift + flexLift)
                    let shOffX: Float = isStorm ? 0 : cX * lift * 0.85
                    let shOffZ: Float = isStorm ? 0 : cZ * lift * 0.85
                    let cx = state.x + rX * latX + fX * localForward + shOffX
                    let cz = state.z + rZ * latX + fZ * localForward + shOffZ
                    let skiHW = evalSkiHalfWidth(t)
                    let halfW = skiHW + 0.036 + lift * 0.34
                    let lx = cx - rnX * halfW, lz = cz - rnZ * halfW
                    let rx = cx + rnX * halfW, rz = cz + rnZ * halfW
                    let ly = td.sampleHeight(x: lx, z: lz) + 0.012
                    let ry = td.sampleHeight(x: rx, z: rz) + 0.012
                    let vi = i * 2
                    rib.vertices[vi].position = SIMD3<Float>(lx, ly, lz)
                    rib.vertices[vi + 1].position = SIMD3<Float>(rx, ry, rz)
                    rib.vertices[vi].custom0.x = lift
                    rib.vertices[vi + 1].custom0.x = lift
                }
                rib.upload()

                let fade = max(0, 1.0 - clearance / 0.95)
                let catcherActive = !state.crashed && !isStorm && mtnSunVis > 0.08 && clearance < 2.8
                rib.emergeT = (powSinkVal > 0.02 && state.grounded) ? lerp(0.28, 0.80, clamp(powSinkVal, 0, 1)) : -1.0
                let powSoft = 1.0 - clamp(powSinkVal * 0.34, 0, 0.34)
                rib.opacity = (isStorm ? 0.56 : (catcherActive ? 0.68 : 0.88)) * fade * powSoft
                if isStorm {
                    rib.tint = SIMD3<Float>(0.50, 0.55, 0.62)
                } else if isAlpenglow {
                    rib.tint = SIMD3<Float>(0.20, 0.25, 0.48)
                } else {
                    rib.tint = SIMD3<Float>(0.22, 0.32, 0.52)
                }
            }
            updateRibbon(rig.leftShadowRib, -SkiRigConstants.skiSep, leadL, smoothFlexL, unweightLiftL, smoothPowSinkL)
            updateRibbon(rig.rightShadowRib, SkiRigConstants.skiSep, leadR, smoothFlexR, unweightLiftR, smoothPowSinkR)
        }

        // Update Full Sculpted Articulated Body
        let showBody = !state.crashed || isThirdPerson
        rig.bodyGroup.visible = showBody
        if showBody {
            rig.bodyGroup.position = SIMD3<Float>(state.x, state.y, state.z)
            let bodyQTarget: simd_quatf = state.grounded
                ? simd_quatf(angle: -state.yaw, axis: SIMD3<Float>(0, 1, 0))
                : state.bodyQuat
            if !bodyQInit || teleported {
                bodyQSmooth = bodyQTarget
                bodyQInit = true
            } else {
                let qRate: Float = state.grounded ? (airRigBlend > 0.02 ? 14.0 : 28.0) : 26.0
                bodyQSmooth = simd_slerp(bodyQSmooth, bodyQTarget, 1.0 - expf(-dt * qRate))
            }
            rig.bodyGroup.setQuaternion(bodyQSmooth)
            rig.bodyGroup.updateWorldMatrices()
            let invBodyMat = simd_inverse(rig.bodyGroup.worldMatrix)

            rig.headGroup.visible = isThirdPerson
            rig.headShadowProxy.visible = !isThirdPerson
            rig.chestFrontGroup.visible = !isChesty
            rig.packGroup.visible = !isChesty
            rig.chestShadowProxy.visible = isChesty
            rig.neckBuffNode.visible = isThirdPerson
            rig.yokeCapNode.visible = isThirdPerson
            rig.packLidNode.visible = isThirdPerson || (!isChesty && abs(lookYawOff) < 0.85)
            rig.armL.shoulderBall.visible = !isChesty
            rig.armR.shoulderBall.visible = !isChesty

            let ankL4 = invBodyMat * (rig.leftSki.ankleAnchor.worldMatrix * SIMD4<Float>(0, 0, 0, 1))
            let ankR4 = invBodyMat * (rig.rightSki.ankleAnchor.worldMatrix * SIMD4<Float>(0, 0, 0, 1))
            let ankleL = SIMD3<Float>(ankL4.x, ankL4.y - 0.045, ankL4.z)
            let ankleR = SIMD3<Float>(ankR4.x, ankR4.y - 0.045, ankR4.z)

            let cr = state.crouch
            let imp = state.landImpact
            let absEdge = abs(edge)

            let airKneePull = airRigBlend * (0.10 + grabBlend * 0.24)
            let avgAnkleY = (ankleL.y + ankleR.y) * 0.5
            let kneeBendFwd = (isThirdPerson ? 0.11 : 0.04) + cr * 0.10 + imp * 0.085
            let kneeHeight = 0.36 - cr * 0.10 - imp * 0.075 - absEdge * 0.04 + airKneePull * 0.28
            let kneeLatShift = edge * 0.15

            let kneeL = SIMD3<Float>(ankleL.x + kneeLatShift * 0.85, ankleL.y + kneeHeight + scissor * 0.025, ankleL.z - kneeBendFwd)
            let kneeR = SIMD3<Float>(ankleR.x + kneeLatShift * 0.85, ankleR.y + kneeHeight - scissor * 0.025, ankleR.z - kneeBendFwd)

            let hipShiftX = edge * 0.22
            let pelvisY = avgAnkleY + 0.65 - cr * 0.20 - imp * 0.17 - absEdge * 0.07 + airKneePull * 0.26
            let pelvisZ: Float = 0.13 + cr * 0.06 + imp * 0.05

            rig.pelvisNode.position = SIMD3<Float>(hipShiftX, pelvisY, pelvisZ)
            rig.pelvisNode.setEuler(-0.05 - cr * 0.12 - imp * 0.10, 0, -edge * 0.42)

            let hipL = SIMD3<Float>(hipShiftX - 0.130, pelvisY - 0.03, pelvisZ - 0.01)
            let hipR = SIMD3<Float>(hipShiftX + 0.130, pelvisY - 0.03, pelvisZ - 0.01)

            placeLimbSegment(rig.legL.shinSeg, pA: ankleL, pB: kneeL)
            placeLimbSegment(rig.legR.shinSeg, pA: ankleR, pB: kneeR)
            rig.legL.kneeBall.position = kneeL
            rig.legR.kneeBall.position = kneeR
            placeLimbSegment(rig.legL.thighSeg, pA: kneeL, pB: hipL)
            placeLimbSegment(rig.legR.thighSeg, pA: kneeR, pB: hipR)

            let skatePush: Float = state.grounded ? max(state.skatePush, state.poleStrokePulse * 0.65) : 0
            let skatePhase = state.skatePhase
            let skateReach = cosf(skatePhase)
            let skateDrive = max(0, sinf(skatePhase))
            let plantL: Float = state.grounded ? sinf(min(1, state.polePlantL) * .pi) : 0
            let plantR: Float = state.grounded ? sinf(min(1, state.polePlantR) * .pi) : 0

            let headTwist: Float = (isThirdPerson || isChesty) ? 0.0 : sinf(lookYawOff) * 0.36
            let torsoLeanFwd = -0.11 - cr * 0.16 - imp * 0.14 - state.foreAft * 0.08 - skatePush * skateDrive * 0.20 - max(plantL, plantR) * 0.06
            let plantTwist = (plantL - plantR) * 0.15
            rig.torsoNode.setEuler(torsoLeanFwd, -edge * 0.26 - headTwist + plantTwist, edge * 0.30)
            rig.headNode.setEuler(0.18, edge * 0.20, 0)

            rig.bodyGroup.updateWorldMatrices()
            let shL4 = invBodyMat * (rig.torsoNode.worldMatrix * SIMD4<Float>(-0.194, 0.418, 0.0, 1.0))
            let shR4 = invBodyMat * (rig.torsoNode.worldMatrix * SIMD4<Float>( 0.194, 0.418, 0.0, 1.0))
            let shL = SIMD3<Float>(shL4.x, shL4.y, shL4.z)
            let shR = SIMD3<Float>(shR4.x, shR4.y, shR4.z)

            let breathe = sinf(time * 2.4) * 0.008
            let poleSwing = sinf(time * 3.1) * 0.020 * min(1, state.speed / 12.0)
            let tuckIn = cr * 0.06
            let braceSpread = imp * 0.085
            let braceDip = imp * 0.055
            let braceFwd = imp * 0.06

            let cTw = cosf(headTwist), sTw = sinf(headTwist)
            let handFwd: Float = isThirdPerson ? 0.40 : (isChesty ? 0.54 : 0.40)
            let handYOff: Float = isThirdPerson ? 0.06 : (isChesty ? 0.055 : -0.05)
            let handSpread: Float = isThirdPerson ? 0.34 : (isChesty ? 0.36 : 0.36)

            let skFwd = skatePush * skateReach * 0.24
            let skUp = skatePush * (skateReach * 0.15 - skateDrive * 0.08)
            let skElbFwd = skatePush * skateReach * 0.12
            let skElbUp = skatePush * (skateReach * 0.08)

            let relLX = -handSpread - braceSpread + tuckIn - edge * 0.05 - plantL * 0.02
            let relLZ = -handFwd - braceFwd - cr * 0.06 - plantL * 0.15 - skFwd
            var handL = SIMD3<Float>(
                hipShiftX + relLX * cTw - relLZ * sTw,
                pelvisY + handYOff - braceDip - cr * 0.05 + breathe + poleSwing + plantL * 0.08 + skUp,
                pelvisZ + relLX * sTw + relLZ * cTw
            )
            let elbLX = -handSpread - 0.04 - braceSpread * 0.65 - plantL * 0.02
            let elbLZ = 0.00 - plantL * 0.09 - skElbFwd
            let elbL = SIMD3<Float>(
                hipShiftX + elbLX * cTw - elbLZ * sTw,
                pelvisY + 0.11 - braceDip * 0.5 + plantL * 0.06 + skElbUp,
                pelvisZ + elbLX * sTw + elbLZ * cTw
            )

            let relRX = handSpread + braceSpread - tuckIn - edge * 0.05 + plantR * 0.02
            let relRZ = -handFwd - braceFwd - cr * 0.06 - plantR * 0.15 - skFwd
            let baseHandRX = hipShiftX + relRX * cTw - relRZ * sTw
            let baseHandRY = pelvisY + handYOff - braceDip - cr * 0.05 + breathe - poleSwing + plantR * 0.08 + skUp
            let baseHandRZ = pelvisZ + relRX * sTw + relRZ * cTw

            var handR: SIMD3<Float>
            var elbR: SIMD3<Float>
            if grabBlend > 0.02 && airRigBlend > 0.05 {
                let gW = grabBlend * airRigBlend
                let grabTarget4 = invBodyMat * (rig.rightSki.node.worldMatrix * SIMD4<Float>(-0.04, 0.04, -0.36, 1.0))
                let grabTarget = SIMD3<Float>(grabTarget4.x, grabTarget4.y, grabTarget4.z)
                handR = SIMD3<Float>(
                    lerp(baseHandRX, grabTarget.x, gW),
                    lerp(baseHandRY, grabTarget.y + 0.03, gW),
                    lerp(baseHandRZ, grabTarget.z, gW)
                )
                elbR = SIMD3<Float>(
                    lerp(hipShiftX + handSpread + 0.04, (shR.x + handR.x) * 0.5 + 0.12, gW),
                    lerp(pelvisY + 0.11, (shR.y + handR.y) * 0.5 - 0.05, gW),
                    lerp(pelvisZ + 0.00, (shR.z + handR.z) * 0.5 + 0.06, gW)
                )
            } else {
                handR = SIMD3<Float>(baseHandRX, baseHandRY, baseHandRZ)
                let elbRX = handSpread + 0.04 + braceSpread * 0.65 + plantR * 0.02
                let elbRZ = 0.00 - plantR * 0.09 - skElbFwd
                elbR = SIMD3<Float>(
                    hipShiftX + elbRX * cTw - elbRZ * sTw,
                    pelvisY + 0.11 - braceDip * 0.5 + plantR * 0.06 + skElbUp,
                    pelvisZ + elbRX * sTw + elbRZ * cTw
                )
            }

            if !isThirdPerson && airRigBlend > 0.02 {
                handL.x -= 0.06 * airRigBlend
                handL.y -= 0.05 * airRigBlend
                handL.z += 0.06 * airRigBlend
            }

            let skatePolePitch = skatePush * (skateReach * 0.72 - 0.08)
            let pitchL = -0.42 - cr * 0.18 + imp * 0.16 + plantL * 0.64 + skatePolePitch
            let pitchR = -0.42 - cr * 0.18 + imp * 0.16 + grabBlend * airRigBlend * 0.38 + plantR * 0.64 + skatePolePitch

            rig.armL.handNode.position = handL
            rig.armL.handNode.setEuler(pitchL, -0.22 - headTwist + plantL * 0.08, -0.44 + plantL * 0.08 - braceSpread * 0.6)
            rig.armL.pole.setEuler(0, 0, 0)
            rig.armL.handNode.updateLocalMatrix()
            let wristL4 = rig.armL.handNode.localMatrix * SIMD4<Float>(rig.armL.wristAnchor.position, 1.0)
            let wristL = SIMD3<Float>(wristL4.x, wristL4.y, wristL4.z)

            rig.armR.handNode.position = handR
            rig.armR.handNode.setEuler(pitchR, 0.22 - headTwist - plantR * 0.08, 0.44 - grabBlend * airRigBlend * 0.30 - plantR * 0.08 + braceSpread * 0.6)
            rig.armR.pole.setEuler(0, 0, 0)
            rig.armR.handNode.updateLocalMatrix()
            let wristR4 = rig.armR.handNode.localMatrix * SIMD4<Float>(rig.armR.wristAnchor.position, 1.0)
            let wristR = SIMD3<Float>(wristR4.x, wristR4.y, wristR4.z)

            rig.armL.shoulderBall.position = shL
            rig.armL.shoulderBall.scale = SIMD3<Float>(repeating: 1.0)
            placeLimbSegment(rig.armL.upperSeg, pA: elbL, pB: shL)
            rig.armL.elbowBall.position = elbL
            rig.armL.elbowBall.scale = SIMD3<Float>(repeating: 1.0)
            placeLimbSegment(rig.armL.lowerSeg, pA: wristL, pB: elbL)
            placeLimbSegment(rig.armL.cuffNode, pA: wristL, pB: elbL)
            rig.armL.cuffNode.position = wristL
            rig.armL.cuffNode.scale = SIMD3<Float>(repeating: 1.0)

            rig.armR.shoulderBall.position = shR
            rig.armR.shoulderBall.scale = SIMD3<Float>(repeating: 1.0)
            placeLimbSegment(rig.armR.upperSeg, pA: elbR, pB: shR)
            rig.armR.elbowBall.position = elbR
            rig.armR.elbowBall.scale = SIMD3<Float>(repeating: 1.0)
            placeLimbSegment(rig.armR.lowerSeg, pA: wristR, pB: elbR)
            placeLimbSegment(rig.armR.cuffNode, pA: wristR, pB: elbR)
            rig.armR.cuffNode.position = wristR
            rig.armR.cuffNode.scale = SIMD3<Float>(repeating: 1.0)
        }

        // Dynamic sun-ray-aligned DEM-snapped multiplicative HDR PCF Shadow Catcher
        let showCatcher = !state.crashed && terrainData != nil && !isStorm && mtnSunVis > 0.08 && clearance < 2.8
        rig.groundCatcher.visible = showCatcher
        if showCatcher, let td = terrainData {
            let sunHorizLen = hypotf(sunDir.x, sunDir.z)
            let castX: Float = sunHorizLen > 0.05 ? (-sunDir.x / sunHorizLen) : 0.0
            let castZ: Float = sunHorizLen > 0.05 ? (-sunDir.z / sunHorizLen) : 1.0
            let rx = castZ
            let rz = -castX

            let slopeRay = sunDir.y / max(0.05, sunHorizLen)
            var headHitDist: Float = 28.0
            for stepD in 1...32 {
                let d = Float(stepD)
                let rayY = state.y + 1.98 - d * slopeRay
                let gy = td.sampleHeight(x: state.x + castX * d, z: state.z + castZ * d)
                if rayY <= gy {
                    headHitDist = d
                    break
                }
            }
            let backDist: Float = 2.6
            let frontDist = clamp(headHitDist * 1.32 + 5.0, 9.0, 38.0)
            let spanLen = backDist + frontDist

            var vi = 0
            for iz in 0...rig.groundCatcher.segsZ {
                let u = Float(iz) / Float(rig.groundCatcher.segsZ)
                let s = 0.52 * u + 0.48 * u * u
                let along = -backDist + s * spanLen
                let bx = state.x + castX * along
                let bz = state.z + castZ * along
                let lift = 0.032 + max(0, along) * 0.004
                for ix in 0...rig.groundCatcher.segsX {
                    let across = ((Float(ix) / Float(rig.groundCatcher.segsX)) * 2.0 - 1.0) * rig.groundCatcher.halfW
                    let wx = bx + rx * across
                    let wz = bz + rz * across
                    let wy = td.sampleHeight(x: wx, z: wz) + lift
                    rig.groundCatcher.vertices[vi].position = SIMD3<Float>(wx, wy, wz)
                    vi += 1
                }
            }
            rig.groundCatcher.recomputeNormalsAndUpload()
            let airFade = max(0, 1.0 - clearance / 2.8)
            rig.groundCatcher.strength = 0.90 * mtnSunVis * airFade
            rig.groundCatcher.tint = isAlpenglow
                ? SIMD3<Float>(0.16, 0.23, 0.48)
                : SIMD3<Float>(0.18, 0.29, 0.50)
        }

        // World-space airborne drop shadow
        rig.groundShadowNode.visible = !state.crashed && clearance < 38.0 && ((!state.grounded && clearance > 0.45) || (isThirdPerson && (isStorm || mtnSunVis <= 0.15)))
        if rig.groundShadowNode.visible {
            let sunY = max(0.24, sunDir.y)
            let offsetScale = clearance / sunY
            let sx = state.x - sunDir.x * min(18.0, offsetScale * 0.65)
            let sz = state.z - sunDir.z * min(18.0, offsetScale * 0.65)
            let sy = state.terrainY + 0.10 + min(0.35, clearance * 0.015)
            rig.groundShadowNode.position = SIMD3<Float>(sx, sy, sz)

            var castVec = SIMD3<Float>(-sunDir.x, 0, -sunDir.z)
            if simd_length_squared(castVec) < 0.001 {
                castVec = SIMD3<Float>(0, 0, 1)
            }
            castVec = simd_normalize(castVec)
            let along = simd_normalize(castVec - nVec * simd_dot(castVec, nVec))
            let across = simd_normalize(simd_cross(along, nVec))
            let trueUp = simd_normalize(simd_cross(across, along))
            rig.groundShadowNode.setQuaternion(quatFromBasis(right: across, up: trueUp, forwardZ: along))

            let spread = 1.0 + clearance * 0.14
            let stretch: Float = isStorm ? 1.0 : min(2.6, 1.0 + (0.65 / sunY))
            rig.groundShadowNode.scale = SIMD3<Float>(spread * 1.05, 1.0, spread * stretch)

            let heightFade = max(0.16, 1.0 - clearance / 32.0)
            let modeMult: Float = (!state.grounded || isThirdPerson) ? 0.85 : 0.48
            rig.groundShadowMat.opacity = heightFade * modeMult * (isStorm ? 0.36 : 0.72)
        }

        // Update all world matrices in the scene graph
        rig.rootNode.updateWorldMatrices()
    }
}
