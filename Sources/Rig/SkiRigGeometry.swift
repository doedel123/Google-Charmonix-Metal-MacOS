import Foundation
import Metal
import simd

// MARK: - Ski Dimensions & Profile Math (1:1 port of src/ski_rig.js)

public enum SkiRigConstants {
    public static let skiLen: Float = 1.84
    public static let bootT: Float = 0.44
    public static let skiSep: Float = 0.192
}

private func rawSkiHalfWidth(_ t: Float) -> Float {
    let halfWaist: Float = 0.044 // 88 mm pinched modern freestyle waist
    let halfTip: Float = 0.086   // 172 mm wide flared powder shovel
    let halfTail: Float = 0.078  // 156 mm wide twin-tip switch tail

    if t >= 0.84 {
        let s = clamp((t - 0.84) / 0.16, 0, 1)
        let cap = powf(max(0.0, 1.0 - powf(s, 3.0)), 0.38)
        return max(0.020, halfTip * cap)
    }
    if t <= 0.14 {
        let s = clamp((0.14 - t) / 0.14, 0, 1)
        let cap = powf(max(0.0, 1.0 - powf(s, 2.85)), 0.38)
        return max(0.018, halfTail * cap)
    }
    if t >= SkiRigConstants.bootT {
        let u = (t - SkiRigConstants.bootT) / (0.84 - SkiRigConstants.bootT)
        return halfWaist + (halfTip - halfWaist) * powf(u, 1.52)
    }
    let u = (SkiRigConstants.bootT - t) / (SkiRigConstants.bootT - 0.14)
    return halfWaist + (halfTail - halfWaist) * powf(u, 1.52)
}

public func evalSkiHalfWidth(_ t: Float) -> Float {
    let tc = clamp(t, 0, 1)
    if tc < 0.015 || tc > 0.985 { return rawSkiHalfWidth(tc) }
    let dt: Float = 0.022
    let w0 = rawSkiHalfWidth(max(0, tc - dt * 2))
    let w1 = rawSkiHalfWidth(max(0, tc - dt))
    let w2 = rawSkiHalfWidth(tc)
    let w3 = rawSkiHalfWidth(min(1, tc + dt))
    let w4 = rawSkiHalfWidth(min(1, tc + dt * 2))
    let capMask: Float = (tc > 0.92 || tc < 0.08) ? 0.25 : 1.0
    let smooth = w0 * 0.08 + w1 * 0.22 + w2 * 0.40 + w3 * 0.22 + w4 * 0.08
    return lerp(w2, smooth, capMask)
}

public func evalRockerY(_ t: Float) -> Float {
    let tipCurve: Float = t > 0.72 ? powf((t - 0.72) / 0.28, 2.18) * 0.092 : 0
    let tailCurve: Float = t < 0.19 ? powf((0.19 - t) / 0.19, 2.08) * 0.052 : 0
    let camber = sinf(t * .pi) * 0.006
    return tipCurve + tailCurve + camber
}

private func buildSkiTSamples() -> [Float] {
    var ts: [Float] = []
    let nTail = 12
    let nMid = 28
    let nNose = 14
    ts.reserveCapacity(nTail + nMid + nNose + 1)
    for i in 0..<nTail {
        let f = Float(i) / Float(nTail)
        ts.append(0.14 * (1.0 - cosf(f * .pi * 0.5)))
    }
    for i in 0..<nMid {
        ts.append(0.14 + (0.84 - 0.14) * (Float(i) / Float(nMid)))
    }
    for i in 0...nNose {
        let f = Float(i) / Float(nNose)
        ts.append(0.84 + (1.0 - 0.84) * sinf(f * .pi * 0.5))
    }
    ts[0] = 0.0
    ts[ts.count - 1] = 1.0
    return ts
}

// MARK: - Material & Scene Node Types

final class RigMaterial {
    var baseColor: SIMD3<Float>
    var roughness: Float
    var metalness: Float
    var emissive: Float = 0.0
    var frostScale: Float = 0.0
    var sheenAmt: Float = 0.16
    var uvScale: SIMD2<Float> = SIMD2<Float>(1, 1)
    var bumpScale: Float = 0.0
    var albedoTex: MTLTexture?
    var bumpTex: MTLTexture?
    var roughTex: MTLTexture?
    var isSki: Bool = false
    var isDoubleSided: Bool = false
    var isDropShadow: Bool = false
    var opacity: Float = 1.0
    var depthWrite: Bool = true

    init(
        hex: UInt32 = 0xffffff,
        roughness: Float = 0.5,
        metalness: Float = 0.0,
        frostScale: Float = 0.0,
        sheenAmt: Float = 0.16
    ) {
        let r = Float((hex >> 16) & 0xff) / 255.0
        let g = Float((hex >> 8) & 0xff) / 255.0
        let b = Float(hex & 0xff) / 255.0
        // Convert sRGB hex color to linear space to match Three.js Color.setHex()
        self.baseColor = SIMD3<Float>(
            powf(r, 2.2),
            powf(g, 2.2),
            powf(b, 2.2)
        )
        self.roughness = roughness
        self.metalness = metalness
        self.frostScale = frostScale
        self.sheenAmt = sheenAmt
    }
}

final class RigMeshPart {
    let mesh: ProceduralMesh
    let material: RigMaterial
    var castShadow: Bool
    var shadowOnly: Bool
    var visible: Bool = true

    init(mesh: ProceduralMesh, material: RigMaterial, castShadow: Bool = true, shadowOnly: Bool = false) {
        self.mesh = mesh
        self.material = material
        self.castShadow = castShadow
        self.shadowOnly = shadowOnly
    }
}

enum RigEulerOrder {
    case xyz
    case yxz
}

final class RigNode {
    var position: SIMD3<Float> = .zero
    var scale: SIMD3<Float> = SIMD3<Float>(repeating: 1)
    var euler: SIMD3<Float> = .zero
    var eulerOrder: RigEulerOrder = .xyz
    var explicitQuat: simd_quatf? = nil
    var visible: Bool = true
    var parts: [RigMeshPart] = []
    var children: [RigNode] = []
    private(set) var localMatrix: simd_float4x4 = matrix_identity_float4x4
    private(set) var worldMatrix: simd_float4x4 = matrix_identity_float4x4

    init(part: RigMeshPart? = nil) {
        if let part { self.parts.append(part) }
    }

    func add(_ child: RigNode) {
        children.append(child)
    }

    func addPart(_ part: RigMeshPart) {
        parts.append(part)
    }

    func setEuler(_ x: Float, _ y: Float, _ z: Float) {
        euler = SIMD3<Float>(x, y, z)
        explicitQuat = nil
    }

    func setQuaternion(_ q: simd_quatf) {
        explicitQuat = q
    }

    var effectiveQuaternion: simd_quatf {
        if let q = explicitQuat { return q }
        switch eulerOrder {
        case .xyz: return quatFromEulerXYZ(x: euler.x, y: euler.y, z: euler.z)
        case .yxz: return quatFromEulerYXZ(pitch: euler.x, yaw: euler.y, roll: euler.z)
        }
    }

    func updateLocalMatrix() {
        localMatrix = matrixTRS(translation: position, rotation: effectiveQuaternion, scale: scale)
    }

    func updateWorldMatrices(parentWorld: simd_float4x4 = matrix_identity_float4x4) {
        updateLocalMatrix()
        worldMatrix = parentWorld * localMatrix
        for child in children {
            child.updateWorldMatrices(parentWorld: worldMatrix)
        }
    }
}

// MARK: - Procedural Textures (Ripstop, Goat Leather, Webbing, Carbon, Drop Shadow)

struct RigPBRTextures {
    let map: MTLTexture
    let bump: MTLTexture
    let rough: MTLTexture
    let repeatScale: SIMD2<Float>
}

struct RigTextureBank {
    let ripJacket: RigPBRTextures
    let ripDark: RigPBRTextures
    let ripPants: RigPBRTextures
    let leatherTan: RigPBRTextures
    let leatherDark: RigPBRTextures
    let webMap: MTLTexture
    let webBump: MTLTexture
    let webRepeat: SIMD2<Float>
    let carbonBump: MTLTexture
    let carbonRepeat: SIMD2<Float>
    let dropShadowTex: MTLTexture
    let fallbackWhite: MTLTexture
}

private func uploadProceduralTexture(
    device: MTLDevice,
    commandQueue: MTLCommandQueue,
    rgba: [UInt8],
    width: Int,
    height: Int,
    srgb: Bool,
    mipmaps: Bool = true,
    label: String
) -> MTLTexture? {
    let format: MTLPixelFormat = srgb ? .rgba8Unorm_srgb : .rgba8Unorm
    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: format,
        width: width,
        height: height,
        mipmapped: mipmaps
    )
    desc.usage = [.shaderRead]
    desc.storageMode = .shared
    guard let tex = device.makeTexture(descriptor: desc) else { return nil }
    tex.label = label
    rgba.withUnsafeBytes { ptr in
        if let base = ptr.baseAddress {
            tex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: base,
                bytesPerRow: width * 4
            )
        }
    }
    if mipmaps, tex.mipmapLevelCount > 1,
       let cmd = commandQueue.makeCommandBuffer(),
       let blit = cmd.makeBlitCommandEncoder() {
        blit.generateMipmaps(for: tex)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    return tex
}

private func buildRipstopTextures(
    device: MTLDevice,
    commandQueue: MTLCommandQueue,
    baseRgb: SIMD3<Float>,
    frostBias: Float,
    repeatN: Float,
    label: String
) -> RigPBRTextures? {
    let S = 512
    var alb = [UInt8](repeating: 255, count: S * S * 4)
    var bmp = [UInt8](repeating: 255, count: S * S * 4)
    var rgh = [UInt8](repeating: 255, count: S * S * 4)

    let tau = Float.pi * 2.0 / Float(S)
    for y in 0..<S {
        let ty = Float(y) * tau
        let ripY: Float = (y % 16 == 0 || y % 16 == 1) ? 1.0 : ((y % 8 == 0) ? 0.45 : 0.0)
        for x in 0..<S {
            let tx = Float(x) * tau
            let ripX: Float = (x % 16 == 0 || x % 16 == 1) ? 1.0 : ((x % 8 == 0) ? 0.45 : 0.0)
            let rip = max(ripX, ripY)
            let weave: Float = ((((x >> 1) + (y >> 1)) & 1) != 0) ? 0.42 : -0.36
            let twill: Float = ((x + y) % 4 < 2) ? 0.30 : -0.25
            let fold = sinf(tx * 2.0 + ty * 1.0 + cosf(ty * 2.0) * 0.6) * 0.48
                + cosf(tx * 1.0 - ty * 3.0 + 1.4) * 0.34
                + sinf(tx * 4.0 + ty * 3.0 - 0.9) * 0.18
            let rawHash = sinf(Float(x) * 12.9898 + Float(y) * 78.233) * 43758.5453
            let micro = rawHash - floorf(rawHash)
            let hVal = clamp(roundf(128.0 + (ripX + ripY) * 26.0 + weave * 11.0 + twill * 7.0 + fold * 38.0 + (micro - 0.5) * 12.0), 0, 255)
            let frostRaw = sinf(tx * 2.0 + 0.8) * cosf(ty * 2.0 - 0.4) + sinf(tx * 5.0 - ty * 4.0) * 0.35
            let frost = frostBias > 0 ? max(Float(0), frostRaw - 0.44) * frostBias : 0.0
            let shade = 0.94 + fold * 0.075 + (rip > 0.6 ? -0.055 : 0.015) + weave * 0.018 + (micro - 0.5) * 0.02
            let rgVal = clamp(roundf(182.0 - (ripX + ripY) * 16.0 + fold * 16.0 - weave * 6.0 + frost * 40.0), 105, 248)

            let k = (y * S + x) * 4
            alb[k]     = UInt8(clamp(roundf(baseRgb.x * shade * (1.0 - frost * 0.40) + 235.0 * frost * 0.40), 0, 255))
            alb[k + 1] = UInt8(clamp(roundf(baseRgb.y * shade * (1.0 - frost * 0.40) + 242.0 * frost * 0.40), 0, 255))
            alb[k + 2] = UInt8(clamp(roundf(baseRgb.z * shade * (1.0 - frost * 0.40) + 252.0 * frost * 0.40), 0, 255))
            alb[k + 3] = 255

            let hb = UInt8(hVal)
            bmp[k] = hb; bmp[k + 1] = hb; bmp[k + 2] = hb; bmp[k + 3] = 255

            let rb = UInt8(rgVal)
            rgh[k] = rb; rgh[k + 1] = rb; rgh[k + 2] = rb; rgh[k + 3] = 255
        }
    }

    guard let m = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: alb, width: S, height: S, srgb: true, label: "\(label)_alb"),
          let b = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: bmp, width: S, height: S, srgb: false, label: "\(label)_bmp"),
          let r = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: rgh, width: S, height: S, srgb: false, label: "\(label)_rgh") else {
        return nil
    }
    return RigPBRTextures(map: m, bump: b, rough: r, repeatScale: SIMD2<Float>(repeatN, repeatN))
}

private func buildLeatherTextures(
    device: MTLDevice,
    commandQueue: MTLCommandQueue,
    baseRgb: SIMD3<Float>,
    wornRgb: SIMD3<Float>,
    repeatN: Float,
    label: String
) -> RigPBRTextures? {
    let S = 256
    let grid = 32
    var jx = [Float](repeating: 0, count: grid * grid)
    var jy = [Float](repeating: 0, count: grid * grid)
    for i in 0..<(grid * grid) {
        let s1 = sinf(Float(i) * 127.1 + 311.7) * 43758.5453
        let s2 = sinf(Float(i) * 269.5 + 183.3) * 43758.5453
        jx[i] = 0.18 + 0.64 * (s1 - floorf(s1))
        jy[i] = 0.18 + 0.64 * (s2 - floorf(s2))
    }

    var alb = [UInt8](repeating: 255, count: S * S * 4)
    var bmp = [UInt8](repeating: 255, count: S * S * 4)
    var rgh = [UInt8](repeating: 255, count: S * S * 4)
    let tau = Float.pi * 2.0 / Float(S)

    for y in 0..<S {
        let gy = (Float(y) / Float(S)) * Float(grid)
        let iy = Int(floorf(gy))
        let ty = Float(y) * tau
        for x in 0..<S {
            let gx = (Float(x) / Float(S)) * Float(grid)
            let ix = Int(floorf(gx))
            let tx = Float(x) * tau
            var d1: Float = 9.0
            var d2: Float = 9.0
            for oy in -1...1 {
                let ny = (iy + oy + grid) % grid
                for ox in -1...1 {
                    let nx = (ix + ox + grid) % grid
                    let idx = ny * grid + nx
                    let dx = (Float(ix + ox) + jx[idx]) - gx
                    let dy = (Float(iy + oy) + jy[idx]) - gy
                    let d = dx * dx + dy * dy
                    if d < d1 { d2 = d1; d1 = d }
                    else if d < d2 { d2 = d }
                }
            }
            let edgeD = sqrtf(d2) - sqrtf(d1)
            let pebble = min(1.0, edgeD * 2.6)
            let crease1 = abs(sinf(tx * 3.0 + ty * 2.0 + sinf(ty * 4.0) * 0.45))
            let crease2 = abs(sinf(tx * 5.0 - ty * 3.0 + cosf(tx * 3.0) * 0.35))
            let creaseDip = powf(crease1, 0.35) * 0.65 + powf(crease2, 0.40) * 0.35
            let patina = 0.5 + 0.30 * sinf(tx * 1.0 + ty * 2.0) + 0.20 * cosf(tx * 2.0 - ty * 1.0 + 0.7)
            let rawHash = sinf(Float(x) * 12.9898 + Float(y) * 78.233) * 43758.5453
            let micro = rawHash - floorf(rawHash)

            let grainHeight = pebble * 0.72 + creaseDip * 0.28
            let hVal = clamp(roundf(70.0 + grainHeight * 150.0 + (micro - 0.5) * 14.0), 0, 255)
            let tCol = clamp(grainHeight * 0.78 + patina * 0.22, 0, 1)
            let rgVal = clamp(roundf(212.0 - pebble * 78.0 - patina * 32.0 + (micro - 0.5) * 10.0), 92, 228)

            let k = (y * S + x) * 4
            let mult = 0.94 + micro * 0.08
            alb[k]     = UInt8(clamp(roundf(lerp(wornRgb.x, baseRgb.x, tCol) * mult), 0, 255))
            alb[k + 1] = UInt8(clamp(roundf(lerp(wornRgb.y, baseRgb.y, tCol) * mult), 0, 255))
            alb[k + 2] = UInt8(clamp(roundf(lerp(wornRgb.z, baseRgb.z, tCol) * mult), 0, 255))
            alb[k + 3] = 255

            let hb = UInt8(hVal)
            bmp[k] = hb; bmp[k + 1] = hb; bmp[k + 2] = hb; bmp[k + 3] = 255

            let rb = UInt8(rgVal)
            rgh[k] = rb; rgh[k + 1] = rb; rgh[k + 2] = rb; rgh[k + 3] = 255
        }
    }

    guard let m = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: alb, width: S, height: S, srgb: true, label: "\(label)_alb"),
          let b = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: bmp, width: S, height: S, srgb: false, label: "\(label)_bmp"),
          let r = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: rgh, width: S, height: S, srgb: false, label: "\(label)_rgh") else {
        return nil
    }
    return RigPBRTextures(map: m, bump: b, rough: r, repeatScale: SIMD2<Float>(repeatN, repeatN))
}

private func buildWebbingTextures(
    device: MTLDevice,
    commandQueue: MTLCommandQueue,
    baseRgb: SIMD3<Float>,
    accentRgb: SIMD3<Float>
) -> (map: MTLTexture, bump: MTLTexture)? {
    let S = 128
    var alb = [UInt8](repeating: 255, count: S * S * 4)
    var bmp = [UInt8](repeating: 255, count: S * S * 4)

    for y in 0..<S {
        for x in 0..<S {
            let col = (x / 16) % 2
            let phase = col == 0 ? Float(x + y) : Float(x - y + 256)
            let rib = sinf(phase * 0.785) * 0.5 + 0.5
            let border: Float = (x < 7 || x > S - 8) ? 1.0 : 0.0
            let t = min(1.0, rib * 0.75 + border * 0.25)
            let k = (y * S + x) * 4
            alb[k]     = UInt8(clamp(roundf(baseRgb.x * (1.0 - t * 0.45) + accentRgb.x * (t * 0.45)), 0, 255))
            alb[k + 1] = UInt8(clamp(roundf(baseRgb.y * (1.0 - t * 0.45) + accentRgb.y * (t * 0.45)), 0, 255))
            alb[k + 2] = UInt8(clamp(roundf(baseRgb.z * (1.0 - t * 0.45) + accentRgb.z * (t * 0.45)), 0, 255))
            alb[k + 3] = 255
            let bh = UInt8(min(255, Int(roundf(85.0 + rib * 130.0 + border * 25.0))))
            bmp[k] = bh; bmp[k + 1] = bh; bmp[k + 2] = bh; bmp[k + 3] = 255
        }
    }
    guard let m = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: alb, width: S, height: S, srgb: true, label: "web_alb"),
          let b = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: bmp, width: S, height: S, srgb: false, label: "web_bmp") else {
        return nil
    }
    return (m, b)
}

private func buildCarbonBump(device: MTLDevice, commandQueue: MTLCommandQueue) -> MTLTexture? {
    let S = 64
    var bmp = [UInt8](repeating: 255, count: S * S * 4)
    for y in 0..<S {
        for x in 0..<S {
            let w: UInt8 = (((x + y) >> 1) & 1) != 0 ? 165 : 95
            let k = (y * S + x) * 4
            bmp[k] = w; bmp[k + 1] = w; bmp[k + 2] = w; bmp[k + 3] = 255
        }
    }
    return uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: bmp, width: S, height: S, srgb: false, label: "carbon_bump")
}

private func buildSkierDropShadowTexture(device: MTLDevice, commandQueue: MTLCommandQueue) -> MTLTexture? {
    let S = 128
    var layer1 = [SIMD4<Float>](repeating: .zero, count: S * S)
    for y in 0..<S {
        for x in 0..<S {
            let fx = Float(x) - 64.0
            let fy = Float(y) - 64.0
            // Crisp ellipse + twin ski silhouettes before 4px Gaussian blur
            let ex = fx / 18.0
            let ey = fy / 28.0
            let inBody = (ex * ex + ey * ey) <= 1.0
            let inSkiL = (x >= 46 && x < 54 && y >= 16 && y < 112)
            let inSkiR = (x >= 74 && x < 82 && y >= 16 && y < 112)
            if inBody || inSkiL || inSkiR {
                layer1[y * S + x] = SIMD4<Float>(20.0 / 255.0, 38.0 / 255.0, 68.0 / 255.0, 0.52)
            }
        }
    }

    // Separable Gaussian blur (~4px radius)
    let kernel: [Float] = [0.028, 0.066, 0.123, 0.180, 0.206, 0.180, 0.123, 0.066, 0.028]
    var temp = [SIMD4<Float>](repeating: .zero, count: S * S)
    for y in 0..<S {
        for x in 0..<S {
            var acc = SIMD4<Float>.zero
            for k in -4...4 {
                let sx = max(0, min(S - 1, x + k))
                acc += layer1[y * S + sx] * kernel[k + 4]
            }
            temp[y * S + x] = acc
        }
    }
    var blurred = [SIMD4<Float>](repeating: .zero, count: S * S)
    for y in 0..<S {
        for x in 0..<S {
            var acc = SIMD4<Float>.zero
            for k in -4...4 {
                let sy = max(0, min(S - 1, y + k))
                acc += temp[sy * S + x] * kernel[k + 4]
            }
            blurred[y * S + x] = acc
        }
    }

    var rgba = [UInt8](repeating: 0, count: S * S * 4)
    for y in 0..<S {
        for x in 0..<S {
            let d = hypotf(Float(x) - 64.0, Float(y) - 64.0)
            var r0: Float = 0, g0: Float = 0, b0: Float = 0, a0: Float = 0
            if d < 58.0 {
                if d <= 8.0 {
                    r0 = 26.0 / 255.0; g0 = 46.0 / 255.0; b0 = 78.0 / 255.0; a0 = 0.46
                } else {
                    let t = (d - 8.0) / 50.0
                    if t < 0.55 {
                        let u = t / 0.55
                        r0 = lerp(26.0 / 255.0, 30.0 / 255.0, u)
                        g0 = lerp(46.0 / 255.0, 52.0 / 255.0, u)
                        b0 = lerp(78.0 / 255.0, 86.0 / 255.0, u)
                        a0 = lerp(0.46, 0.20, u)
                    } else {
                        let u = (t - 0.55) / 0.45
                        r0 = 30.0 / 255.0; g0 = 52.0 / 255.0; b0 = 86.0 / 255.0
                        a0 = lerp(0.20, 0.0, u)
                    }
                }
            }
            let b = blurred[y * S + x]
            let aOut = b.w + a0 * (1.0 - b.w)
            let k = (y * S + x) * 4
            if aOut > 1e-4 {
                let rOut = (b.x * b.w + r0 * a0 * (1.0 - b.w)) / aOut
                let gOut = (b.y * b.w + g0 * a0 * (1.0 - b.w)) / aOut
                let bOut = (b.z * b.w + b0 * a0 * (1.0 - b.w)) / aOut
                rgba[k]     = UInt8(clamp(roundf(rOut * 255.0), 0, 255))
                rgba[k + 1] = UInt8(clamp(roundf(gOut * 255.0), 0, 255))
                rgba[k + 2] = UInt8(clamp(roundf(bOut * 255.0), 0, 255))
                rgba[k + 3] = UInt8(clamp(roundf(aOut * 255.0), 0, 255))
            }
        }
    }
    return uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: rgba, width: S, height: S, srgb: false, mipmaps: false, label: "drop_shadow")
}

func buildRigTextureBank(device: MTLDevice, commandQueue: MTLCommandQueue) -> RigTextureBank? {
    guard let ripJacket = buildRipstopTextures(device: device, commandQueue: commandQueue, baseRgb: SIMD3<Float>(216, 66, 38), frostBias: 0.12, repeatN: 4, label: "ripJacket"),
          let ripDark = buildRipstopTextures(device: device, commandQueue: commandQueue, baseRgb: SIMD3<Float>(48, 58, 74), frostBias: 0.16, repeatN: 4, label: "ripDark"),
          let ripPants = buildRipstopTextures(device: device, commandQueue: commandQueue, baseRgb: SIMD3<Float>(46, 56, 72), frostBias: 0.28, repeatN: 4, label: "ripPants"),
          let leatherTan = buildLeatherTextures(device: device, commandQueue: commandQueue, baseRgb: SIMD3<Float>(198, 138, 82), wornRgb: SIMD3<Float>(134, 82, 44), repeatN: 2.4, label: "leatherTan"),
          let leatherDark = buildLeatherTextures(device: device, commandQueue: commandQueue, baseRgb: SIMD3<Float>(76, 58, 48), wornRgb: SIMD3<Float>(42, 32, 27), repeatN: 2.8, label: "leatherDark"),
          let web = buildWebbingTextures(device: device, commandQueue: commandQueue, baseRgb: SIMD3<Float>(38, 44, 54), accentRgb: SIMD3<Float>(82, 92, 106)),
          let carbon = buildCarbonBump(device: device, commandQueue: commandQueue),
          let dropShadow = buildSkierDropShadowTexture(device: device, commandQueue: commandQueue),
          let white = uploadProceduralTexture(device: device, commandQueue: commandQueue, rgba: [255, 255, 255, 255], width: 1, height: 1, srgb: false, mipmaps: false, label: "rig_white") else {
        return nil
    }
    return RigTextureBank(
        ripJacket: ripJacket,
        ripDark: ripDark,
        ripPants: ripPants,
        leatherTan: leatherTan,
        leatherDark: leatherDark,
        webMap: web.map,
        webBump: web.bump,
        webRepeat: SIMD2<Float>(1, 6),
        carbonBump: carbon,
        carbonRepeat: SIMD2<Float>(3, 24),
        dropShadowTex: dropShadow,
        fallbackWhite: white
    )
}

// MARK: - Catmull-Rom Curve, Frenet Frames, Tube & Lathe Builders

final class RigCatmullRomCurve {
    enum CurveType { case catmullrom, centripetal }

    private let points: [SIMD3<Float>]
    private let type: CurveType

    init(_ points: [SIMD3<Float>], type: CurveType = .catmullrom) {
        self.points = points
        self.type = type
    }

    func getPoint(_ t: Float) -> SIMD3<Float> {
        let l = points.count
        guard l >= 2 else { return points.first ?? .zero }
        let p = Float(l - 1) * clamp(t, 0, 1)
        var intPoint = Int(floorf(p))
        var weight = p - Float(intPoint)
        if weight == 0 && intPoint == l - 1 {
            intPoint = l - 2
            weight = 1
        }

        let p0: SIMD3<Float> = intPoint > 0 ? points[intPoint - 1] : (2.0 * points[0] - points[1])
        let p1: SIMD3<Float> = points[intPoint]
        let p2: SIMD3<Float> = points[intPoint + 1]
        let p3: SIMD3<Float> = intPoint + 2 < l ? points[intPoint + 2] : (2.0 * points[l - 1] - points[l - 2])

        switch type {
        case .catmullrom:
            return evalCatmullRomUniform(p0, p1, p2, p3, t: weight)
        case .centripetal:
            return evalCentripetal(p0, p1, p2, p3, t: weight)
        }
    }

    func getTangent(_ t: Float) -> SIMD3<Float> {
        let delta: Float = 0.0001
        let t1 = max(0, t - delta)
        let t2 = min(1, t + delta)
        let diff = getPoint(t2) - getPoint(t1)
        let len = simd_length(diff)
        return len > 1e-7 ? (diff / len) : SIMD3<Float>(0, 1, 0)
    }

    func computeFrenetFrames(segments: Int) -> (tangents: [SIMD3<Float>], normals: [SIMD3<Float>], binormals: [SIMD3<Float>]) {
        var tangents = [SIMD3<Float>](repeating: .zero, count: segments + 1)
        var normals = [SIMD3<Float>](repeating: .zero, count: segments + 1)
        var binormals = [SIMD3<Float>](repeating: .zero, count: segments + 1)

        for i in 0...segments {
            tangents[i] = getTangent(Float(i) / Float(segments))
        }

        var normal = SIMD3<Float>.zero
        var minC: Float = Float.greatestFiniteMagnitude
        let t0 = tangents[0]
        if abs(t0.x) <= minC { minC = abs(t0.x); normal = SIMD3<Float>(1, 0, 0) }
        if abs(t0.y) <= minC { minC = abs(t0.y); normal = SIMD3<Float>(0, 1, 0) }
        if abs(t0.z) <= minC { normal = SIMD3<Float>(0, 0, 1) }

        let vec0 = simd_normalize(simd_cross(t0, normal))
        normals[0] = simd_normalize(simd_cross(t0, vec0))

        for i in 1...segments {
            normals[i] = normals[i - 1]
            let axis = simd_cross(tangents[i - 1], tangents[i])
            let axisLen = simd_length(axis)
            if axisLen > 1e-4 {
                let nAxis = axis / axisLen
                let theta = acosf(clamp(simd_dot(tangents[i - 1], tangents[i]), -1, 1))
                let q = simd_quatf(angle: theta, axis: nAxis)
                normals[i] = simd_normalize(q.act(normals[i]))
            }
        }

        for i in 0...segments {
            binormals[i] = simd_normalize(simd_cross(tangents[i], normals[i]))
        }
        return (tangents, normals, binormals)
    }

    private func evalCatmullRomUniform(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>, t: Float) -> SIMD3<Float> {
        let t2 = t * t
        let t3 = t2 * t
        return 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * t +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
    }

    private func evalCentripetal(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>, t: Float) -> SIMD3<Float> {
        let powHalf: (SIMD3<Float>, SIMD3<Float>) -> Float = { a, b in
            max(1e-4, powf(simd_length_squared(a - b), 0.25))
        }
        let dt0 = powHalf(p0, p1)
        let dt1 = powHalf(p1, p2)
        let dt2 = powHalf(p2, p3)
        let t1 = ((p1 - p0) / dt0 - (p2 - p0) / (dt0 + dt1) + (p2 - p1) / dt1) * dt1
        let t2 = ((p2 - p1) / dt1 - (p3 - p1) / (dt1 + dt2) + (p3 - p2) / dt2) * dt1
        let c0 = p1
        let c1 = t1
        let c2 = -3.0 * p1 + 3.0 * p2 - 2.0 * t1 - t2
        let c3 = 2.0 * p1 - 2.0 * p2 + t1 + t2
        return c0 + c1 * t + c2 * (t * t) + c3 * (t * t * t)
    }
}

private func buildTubeGeometry(
    curve: RigCatmullRomCurve,
    tubularSegments: Int,
    radius: Float,
    radialSegments: Int
) -> ([StandardVertex], [UInt32]) {
    let frames = curve.computeFrenetFrames(segments: tubularSegments)
    var verts: [StandardVertex] = []
    var indices: [UInt32] = []
    let color = SIMD4<Float>(1, 1, 1, 1)

    for i in 0...tubularSegments {
        let u = Float(i) / Float(tubularSegments)
        let P = curve.getPoint(u)
        let N = frames.normals[i]
        let B = frames.binormals[i]
        for j in 0...radialSegments {
            let v = Float(j) / Float(radialSegments)
            let theta = v * .pi * 2.0
            let sinT = sinf(theta)
            let cosT = -cosf(theta)
            let normal = simd_normalize(cosT * N + sinT * B)
            let pos = P + radius * normal
            verts.append(StandardVertex(position: pos, normal: normal, uv: SIMD2<Float>(u, v), color: color))
        }
    }

    let row = radialSegments + 1
    for i in 0..<tubularSegments {
        for j in 0..<radialSegments {
            let a = UInt32(i * row + j)
            let b = UInt32((i + 1) * row + j)
            let c = UInt32((i + 1) * row + j + 1)
            let d = UInt32(i * row + j + 1)
            indices.append(contentsOf: [a, b, d, b, c, d])
        }
    }
    return (verts, indices)
}

private func buildLatheGeometry(points: [SIMD2<Float>], segments: Int) -> ([StandardVertex], [UInt32]) {
    var verts: [StandardVertex] = []
    var indices: [UInt32] = []
    let color = SIMD4<Float>(1, 1, 1, 1)
    let pCount = points.count

    for i in 0...segments {
        let u = Float(i) / Float(segments)
        let phi = u * .pi * 2.0
        let sinP = sinf(phi)
        let cosP = cosf(phi)
        for j in 0..<pCount {
            let v = Float(j) / Float(max(1, pCount - 1))
            let pt = points[j]
            let pos = SIMD3<Float>(pt.x * sinP, pt.y, pt.x * cosP)
            verts.append(StandardVertex(position: pos, normal: SIMD3<Float>(0, 1, 0), uv: SIMD2<Float>(u, v), color: color))
        }
    }

    for i in 0..<segments {
        for j in 0..<(pCount - 1) {
            let base = UInt32(j + pCount * i)
            let a = base
            let b = base + UInt32(pCount)
            let c = base + UInt32(pCount) + 1
            let d = base + 1
            indices.append(contentsOf: [a, b, d, b, c, d])
        }
    }
    MeshPrimitives.computeSmoothNormals(vertices: &verts, indices: indices)
    return (verts, indices)
}

// MARK: - Procedural Skier & Equipment Geometry Builders

struct LimbOptions {
    var radSegs: Int = 24
    var lenSegs: Int = 20
    var scaleX: Float = 1.0
    var scaleZ: Float = 1.0
    var bulgeMid: Float = 0.0
    var foldAmp: Float = 0.0052
    var foldFreq: Float = 5.5
    var phase: Float = 0.0
    var gatherBot: Float = 0.0
    var gatherTop: Float = 0.0
    var sideSeam: Bool = true
}

private func makeSculptedLimbMesh(
    device: MTLDevice,
    rBottom: Float,
    rTop: Float,
    opts: LimbOptions,
    label: String
) -> ProceduralMesh? {
    let radSegs = opts.radSegs
    let lenSegs = opts.lenSegs
    let rowCols = radSegs + 1
    var verts: [StandardVertex] = []
    var indices: [UInt32] = []
    let color = SIMD4<Float>(1, 1, 1, 1)

    for i in 0...lenSegs {
        let v = Float(i) / Float(lenSegs)
        let y = v - 0.5
        let baseR = rBottom + (rTop - rBottom) * v + opts.bulgeMid * sinf(v * .pi) * (0.65 + 0.35 * v)
        let botGate = powf(max(0, 1.0 - v * 3.2), 1.6)
        let topGate = powf(max(0, (v - 0.68) / 0.32), 1.6)
        for j in 0...radSegs {
            let u = Float(j) / Float(radSegs)
            let th = u * .pi * 2.0
            let c = cosf(th)
            let s = sinf(th)
            let diag1 = sinf(v * opts.foldFreq * .pi + s * 1.45 + c * 0.75 + opts.phase)
            let diag2 = cosf(v * (opts.foldFreq * 1.65) * .pi - c * 1.8 + s * 0.9 + opts.phase * 1.7)
            let jointWave = (botGate * opts.gatherBot + topGate * opts.gatherTop) * sinf(v * 26.0 + c * 2.4 + opts.phase)
            let microPleat = (diag1 * 0.62 + diag2 * 0.38) * opts.foldAmp * (0.45 + 0.55 * (botGate + topGate + 0.45))
            let seamRidge: Float = opts.sideSeam ? expf(-s * s * 42.0) * 0.0014 : 0.0
            let flare = 1.0 + botGate * opts.gatherBot * 0.14 - topGate * opts.gatherTop * 0.04
            let r = baseR * flare + microPleat + jointWave * 0.0038 + seamRidge
            let ex = lerp(opts.scaleX, 1.0, v * 0.35)
            let ez = lerp(opts.scaleZ, 1.0, v * 0.35)
            verts.append(StandardVertex(
                position: SIMD3<Float>(c * r * ex, y, s * r * ez),
                normal: SIMD3<Float>(0, 1, 0),
                uv: SIMD2<Float>(u * 1.5, v * 1.5),
                color: color
            ))
        }
    }

    for i in 0..<lenSegs {
        let r0 = UInt32(i * rowCols)
        let r1 = UInt32((i + 1) * rowCols)
        for j in 0..<radSegs {
            let jj = UInt32(j)
            indices.append(contentsOf: [r0 + jj, r1 + jj, r0 + jj + 1, r0 + jj + 1, r1 + jj, r1 + jj + 1])
        }
    }

    MeshPrimitives.computeSmoothNormals(vertices: &verts, indices: indices)
    return MeshPrimitives.upload(device: device, vertices: verts, indices: indices, label: label)
}

private func buildSculptedFingerMesh(
    device: MTLDevice,
    curve: RigCatmullRomCurve,
    radFn: (Float) -> Float,
    segs: Int = 20,
    radSegs: Int = 12,
    flatAspect: Float = 0.88,
    label: String
) -> ProceduralMesh? {
    let frames = curve.computeFrenetFrames(segments: segs)
    struct Ring {
        var p: SIMD3<Float>
        var t: SIMD3<Float>
        var n: SIMD3<Float>
        var b: SIMD3<Float>
        var u: Float
        var r: Float
    }
    var rings: [Ring] = []
    for i in 0...segs {
        let u = Float(i) / Float(segs)
        let p = curve.getPoint(u)
        let T = frames.tangents[i]
        let N = frames.normals[i]
        let B = simd_normalize(simd_cross(T, N))
        let N2 = simd_normalize(simd_cross(B, T))
        rings.append(Ring(p: p, t: T, n: N2, b: B, u: u, r: radFn(u)))
    }
    if let last = rings.last {
        let tipAngles: [Float] = [0.42, 0.85, 1.25, .pi * 0.5]
        for (k, a) in tipAngles.enumerated() {
            let pTip = last.p + last.t * (sinf(a) * last.r * 0.95)
            let rTip = max(0.0004, last.r * cosf(a))
            rings.append(Ring(p: pTip, t: last.t, n: last.n, b: last.b, u: 1.0 + Float(k + 1) * 0.04, r: rTip))
        }
    }

    var verts: [StandardVertex] = []
    var indices: [UInt32] = []
    let rowCols = radSegs + 1
    let color = SIMD4<Float>(1, 1, 1, 1)

    for ring in rings {
        for j in 0...radSegs {
            let v = Float(j) / Float(radSegs)
            let th = v * .pi * 2.0
            let c = cosf(th)
            let s = sinf(th)
            let knucklePleat: Float = (ring.u > 0.25 && ring.u < 0.82)
                ? (sinf(ring.u * 42.0) * 0.00055 * max(0, -c))
                : 0
            let rx = (ring.r + knucklePleat) * flatAspect
            let ry = ring.r + knucklePleat * 0.5
            let pos = ring.p + ring.n * (c * rx) + ring.b * (s * ry)
            verts.append(StandardVertex(position: pos, normal: SIMD3<Float>(0, 1, 0), uv: SIMD2<Float>(v * 0.85, ring.u * 1.4), color: color))
        }
    }

    for i in 0..<(rings.count - 1) {
        let r0 = UInt32(i * rowCols)
        let r1 = UInt32((i + 1) * rowCols)
        for j in 0..<radSegs {
            let jj = UInt32(j)
            indices.append(contentsOf: [r0 + jj, r0 + jj + 1, r1 + jj, r0 + jj + 1, r1 + jj + 1, r1 + jj])
        }
    }

    MeshPrimitives.computeSmoothNormals(vertices: &verts, indices: indices)
    return MeshPrimitives.upload(device: device, vertices: verts, indices: indices, label: label)
}

private func buildJacketTorsoMesh(device: MTLDevice) -> ProceduralMesh? {
    let radSegs = 32
    let lenSegs = 22
    let cols = radSegs + 1
    var verts: [StandardVertex] = []
    var indices: [UInt32] = []
    let color = SIMD4<Float>(1, 1, 1, 1)

    for i in 0...lenSegs {
        let v = Float(i) / Float(lenSegs)
        let y = lerp(-0.025, 0.485, v)
        let waistDip = -0.018 * expf(-powf((v - 0.28) / 0.16, 2))
        let chestFull = 0.030 * expf(-powf((v - 0.74) / 0.22, 2))
        let hemFlare = 0.012 * powf(max(0, 1.0 - v * 4.0), 1.6)
        let neckTap: Float = v > 0.86 ? (1.0 - powf((v - 0.86) / 0.14, 1.6) * 0.24) : 1.0
        let rx = (0.194 + waistDip + chestFull + hemFlare) * 1.08 * neckTap
        let rz = (0.124 + waistDip * 0.6 + chestFull * 0.72 + hemFlare * 0.5) * neckTap

        for j in 0...radSegs {
            let u = Float(j) / Float(radSegs)
            let th = u * .pi * 2.0
            let c = cosf(th)
            let s = sinf(th)
            let gather = expf(-powf((v - 0.24) / 0.14, 2)) * sinf(v * 34.0 + c * 1.8) * 0.0042
            let chestPec = (s < 0 ? abs(s) : 0) * expf(-powf((v - 0.72) / 0.18, 2)) * (0.008 * powf(abs(c), 0.7))
            let fold = sinf(v * 16.0 + c * 2.2 - s * 1.5) * 0.0028
            let sx = (c >= 0 ? Float(1) : Float(-1)) * powf(abs(c), 0.90)
            let sz = (s >= 0 ? Float(1) : Float(-1)) * powf(abs(s), 0.90)
            verts.append(StandardVertex(
                position: SIMD3<Float>(sx * (rx + gather + fold), y, sz * (rz + gather * 0.7 + fold * 0.7 + chestPec) - 0.004),
                normal: SIMD3<Float>(0, 1, 0),
                uv: SIMD2<Float>(u * 2.2, v * 2.0),
                color: color
            ))
        }
    }

    for i in 0..<lenSegs {
        let r0 = UInt32(i * cols)
        let r1 = UInt32((i + 1) * cols)
        for j in 0..<radSegs {
            let jj = UInt32(j)
            indices.append(contentsOf: [r0 + jj, r1 + jj, r0 + jj + 1, r0 + jj + 1, r1 + jj, r1 + jj + 1])
        }
    }
    MeshPrimitives.computeSmoothNormals(vertices: &verts, indices: indices)
    return MeshPrimitives.upload(device: device, vertices: verts, indices: indices, label: "JacketTorso")
}

private func buildPackMesh(device: MTLDevice) -> ProceduralMesh? {
    let radSegs = 28
    let lenSegs = 18
    let cols = radSegs + 1
    var verts: [StandardVertex] = []
    var indices: [UInt32] = []
    let color = SIMD4<Float>(1, 1, 1, 1)

    for i in 0...lenSegs {
        let v = Float(i) / Float(lenSegs)
        let y = lerp(0.03, 0.48, v)
        let capEnv = sinf(powf(v, 0.85) * .pi)
        let profile = powf(max(0.04, capEnv), 0.25)
        let rx = (0.172 - v * 0.018) * profile
        let rz = (0.088 - abs(v - 0.5) * 0.020) * profile
        for j in 0...radSegs {
            let u = Float(j) / Float(radSegs)
            let th = u * .pi * 2.0
            let c = cosf(th)
            let s = sinf(th)
            let sx = (c >= 0 ? Float(1) : Float(-1)) * powf(abs(c), 0.72)
            let sz = (s >= 0 ? Float(1) : Float(-1)) * powf(abs(s), 0.78)
            let zFlat = s < 0 ? sz * rz * 0.52 : sz * rz * 1.04
            verts.append(StandardVertex(
                position: SIMD3<Float>(sx * rx, y, 0.152 + zFlat),
                normal: SIMD3<Float>(0, 1, 0),
                uv: SIMD2<Float>(u * 1.5, v * 1.5),
                color: color
            ))
        }
    }
    for i in 0..<lenSegs {
        let r0 = UInt32(i * cols)
        let r1 = UInt32((i + 1) * cols)
        for j in 0..<radSegs {
            let jj = UInt32(j)
            indices.append(contentsOf: [r0 + jj, r1 + jj, r0 + jj + 1, r0 + jj + 1, r1 + jj, r1 + jj + 1])
        }
    }
    MeshPrimitives.computeSmoothNormals(vertices: &verts, indices: indices)
    return MeshPrimitives.upload(device: device, vertices: verts, indices: indices, label: "AvyPack")
}

// MARK: - Bow Wave & Ski Structures

struct BowWaveVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var uv: SIMD2<Float>
    var bowData: SIMD4<Float> // xyz = aBow, w = aBaseY
}

struct ShadowRibbonVertex {
    var position: SIMD3<Float>
    var uv: SIMD2<Float>
    var custom0: SIMD4<Float> // x = aLift
}

struct ShadowCatcherVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var catchUv: SIMD2<Float>
}

final class SkiAssembly {
    let node = RigNode()
    let isLeft: Bool
    let bladePart: RigMeshPart
    var bladeVertices: [StandardVertex]
    let bladeIndices: [UInt32]
    let baseY: [Float]
    let tVals: [Float]

    let dustPart: RigMeshPart
    var dustVertices: [StandardVertex]
    let dustIndices: [UInt32]
    let dBaseY: [Float]
    let dTVals: [Float]

    let bowVertexBuffer: MTLBuffer
    let bowIndexBuffer: MTLBuffer
    let bowIndexCount: Int
    var bowVisible: Bool = false

    let ankleAnchor = RigNode()
    var lastFlex: Float = -99

    // Per-frame bow wave & powder uniforms
    var powSink: Float = 0
    var edgeRoll: Float = 0
    var flowDist: Float = 0
    var snowLitCol: SIMD3<Float> = SIMD3<Float>(1, 1, 1)
    var snowShadCol: SIMD3<Float> = SIMD3<Float>(0.32, 0.45, 0.68)
    var snowSSSCol: SIMD3<Float> = SIMD3<Float>(0.14, 0.48, 0.90)
    var powSunVis: Float = 1.0
    var powStorm: Float = 0.0

    var bowIntensity: Float = 0.0
    var bowEdgeBias: Float = 0.0
    var bowSkiRoll: Float = 0.0
    var bowSunVis: Float = 1.0

    init(
        isLeft: Bool,
        bladePart: RigMeshPart,
        bladeVertices: [StandardVertex],
        bladeIndices: [UInt32],
        baseY: [Float],
        tVals: [Float],
        dustPart: RigMeshPart,
        dustVertices: [StandardVertex],
        dustIndices: [UInt32],
        dBaseY: [Float],
        dTVals: [Float],
        bowVertexBuffer: MTLBuffer,
        bowIndexBuffer: MTLBuffer,
        bowIndexCount: Int
    ) {
        self.isLeft = isLeft
        self.bladePart = bladePart
        self.bladeVertices = bladeVertices
        self.bladeIndices = bladeIndices
        self.baseY = baseY
        self.tVals = tVals
        self.dustPart = dustPart
        self.dustVertices = dustVertices
        self.dustIndices = dustIndices
        self.dBaseY = dBaseY
        self.dTVals = dTVals
        self.bowVertexBuffer = bowVertexBuffer
        self.bowIndexBuffer = bowIndexBuffer
        self.bowIndexCount = bowIndexCount
    }
}

final class ShadowRibbonMesh {
    let segs: Int
    var vertices: [ShadowRibbonVertex]
    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let indexCount: Int
    var visible: Bool = false
    var opacity: Float = 0.90
    var tint: SIMD3<Float> = SIMD3<Float>(0.38, 0.48, 0.66)
    var emergeT: Float = -1.0

    init?(device: MTLDevice, segs: Int = 20) {
        self.segs = segs
        let vCount = (segs + 1) * 2
        var verts = [ShadowRibbonVertex](
            repeating: ShadowRibbonVertex(position: .zero, uv: .zero, custom0: .zero),
            count: vCount
        )
        var indices: [UInt32] = []
        indices.reserveCapacity(segs * 6)
        for i in 0...segs {
            let t = Float(i) / Float(segs)
            verts[i * 2].uv = SIMD2<Float>(-1, t)
            verts[i * 2 + 1].uv = SIMD2<Float>(1, t)
            if i < segs {
                let r0 = UInt32(i * 2)
                let r1 = r0 + 2
                indices.append(contentsOf: [r0, r1, r0 + 1, r0 + 1, r1, r1 + 1])
            }
        }
        guard let vb = device.makeBuffer(length: vCount * MemoryLayout<ShadowRibbonVertex>.stride, options: .storageModeShared),
              let ib = indices.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: indices.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) }) else {
            return nil
        }
        vb.label = "SkiShadowRibbon_VB"
        ib.label = "SkiShadowRibbon_IB"
        self.vertices = verts
        self.vertexBuffer = vb
        self.indexBuffer = ib
        self.indexCount = indices.count
    }

    func upload() {
        vertices.withUnsafeBytes { ptr in
            if let base = ptr.baseAddress {
                memcpy(vertexBuffer.contents(), base, vertices.count * MemoryLayout<ShadowRibbonVertex>.stride)
            }
        }
    }
}

final class ShadowCatcherMesh {
    let segsX = 14
    let segsZ = 34
    let halfW: Float = 3.8
    var vertices: [ShadowCatcherVertex]
    let indices: [UInt32]
    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    var visible: Bool = false
    var tint: SIMD3<Float> = SIMD3<Float>(0.18, 0.29, 0.50)
    var strength: Float = 0.90

    init?(device: MTLDevice) {
        let vCount = (segsX + 1) * (segsZ + 1)
        var verts = [ShadowCatcherVertex](
            repeating: ShadowCatcherVertex(position: .zero, normal: SIMD3<Float>(0, 1, 0), catchUv: .zero),
            count: vCount
        )
        var vi = 0
        for iz in 0...segsZ {
            let v = Float(iz) / Float(segsZ)
            for ix in 0...segsX {
                verts[vi].catchUv = SIMD2<Float>(Float(ix) / Float(segsX), v)
                vi += 1
            }
        }
        var idx: [UInt32] = []
        idx.reserveCapacity(segsX * segsZ * 6)
        let row = segsX + 1
        for iz in 0..<segsZ {
            for ix in 0..<segsX {
                let a = UInt32(iz * row + ix)
                let b = UInt32((iz + 1) * row + ix)
                let c = UInt32((iz + 1) * row + ix + 1)
                let d = UInt32(iz * row + ix + 1)
                // Upward +Y CCW winding when rx = castZ, rz = -castX: (+castDir) x (+rightDir) = (0, +1, 0)
                idx.append(contentsOf: [a, b, d, b, c, d])
            }
        }
        guard let vb = device.makeBuffer(length: vCount * MemoryLayout<ShadowCatcherVertex>.stride, options: .storageModeShared),
              let ib = idx.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: idx.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) }) else {
            return nil
        }
        vb.label = "ShadowCatcher_VB"
        ib.label = "ShadowCatcher_IB"
        self.vertices = verts
        self.indices = idx
        self.vertexBuffer = vb
        self.indexBuffer = ib
    }

    func recomputeNormalsAndUpload() {
        for i in vertices.indices { vertices[i].normal = .zero }
        var t = 0
        while t + 2 < indices.count {
            let i0 = Int(indices[t])
            let i1 = Int(indices[t + 1])
            let i2 = Int(indices[t + 2])
            let p0 = vertices[i0].position
            let p1 = vertices[i1].position
            let p2 = vertices[i2].position
            let n = simd_cross(p1 - p0, p2 - p0)
            vertices[i0].normal += n
            vertices[i1].normal += n
            vertices[i2].normal += n
            t += 3
        }
        for i in vertices.indices {
            let l = simd_length(vertices[i].normal)
            vertices[i].normal = l > 1e-7 ? (vertices[i].normal / l) : SIMD3<Float>(0, 1, 0)
        }
        vertices.withUnsafeBytes { ptr in
            if let base = ptr.baseAddress {
                memcpy(vertexBuffer.contents(), base, vertices.count * MemoryLayout<ShadowCatcherVertex>.stride)
            }
        }
    }
}

// MARK: - Complete Skier Rig Builder

struct LegRig {
    let shinSeg: RigNode
    let kneeBall: RigNode
    let thighSeg: RigNode
}

struct ArmRig {
    let shoulderBall: RigNode
    let upperSeg: RigNode
    let elbowBall: RigNode
    let lowerSeg: RigNode
    let cuffNode: RigNode
    let handNode: RigNode
    let pole: RigNode
    let wristAnchor: RigNode
}

final class SkierRigScene {
    let rootNode = RigNode()
    let fpGroup = RigNode()
    let skiPivot = RigNode()
    let leftSki: SkiAssembly
    let rightSki: SkiAssembly

    let groundShadowNode = RigNode()
    let groundShadowMat: RigMaterial
    let leftShadowRib: ShadowRibbonMesh
    let rightShadowRib: ShadowRibbonMesh
    let groundCatcher: ShadowCatcherMesh

    let bodyGroup = RigNode()
    let pelvisNode = RigNode()
    let torsoNode = RigNode()
    let chestFrontGroup = RigNode()
    let packGroup = RigNode()
    let headGroup = RigNode()
    let headNode = RigNode()
    let neckBuffNode = RigNode()
    let yokeCapNode = RigNode()
    let packLidNode = RigNode()
    let headShadowProxy = RigNode()
    let chestShadowProxy = RigNode()

    let legL: LegRig
    let legR: LegRig
    let armL: ArmRig
    let armR: ArmRig

    let allForegroundMaterials: [RigMaterial]
    let allMaterials: [RigMaterial]

    init?(device: MTLDevice, skiTopsheetTex: MTLTexture, textures: RigTextureBank) {
        let makeMat: (UInt32, Float, Float, Float, Float) -> RigMaterial = { hex, r, m, fs, sh in
            RigMaterial(hex: hex, roughness: r, metalness: m, frostScale: fs, sheenAmt: sh)
        }

        // Shared materials
        let bindMat = makeMat(0x1a1e24, 0.36, 0.56, 0.0, 0.16)
        let gunmetalMat = makeMat(0x343c47, 0.28, 0.76, 0.0, 0.16)
        let steelMat = makeMat(0xb0bac8, 0.20, 0.90, 0.0, 0.16)
        let coralMetalMat = makeMat(0xdf4a26, 0.28, 0.60, 0.0, 0.16)
        let rubberMat = makeMat(0x14171b, 0.86, 0.04, 0.0, 0.16)

        let bootShellMat = makeMat(0x2c3542, 0.36, 0.20, 0.55, 0.16)
        let bootTongueMat = makeMat(0x202732, 0.68, 0.08, 0.45, 0.16)

        let pantsMat = makeMat(0xffffff, 0.84, 0.04, 0.40, 0.22)
        pantsMat.albedoTex = textures.ripPants.map
        pantsMat.bumpTex = textures.ripPants.bump
        pantsMat.roughTex = textures.ripPants.rough
        pantsMat.uvScale = textures.ripPants.repeatScale
        pantsMat.bumpScale = 0.017

        let corduraMat = makeMat(0x262e3a, 0.86, 0.05, 0.34, 0.18)
        corduraMat.bumpTex = textures.ripPants.bump
        corduraMat.uvScale = textures.ripPants.repeatScale
        corduraMat.bumpScale = 0.018

        let seamMat = makeMat(0x3b485c, 0.72, 0.06, 0.22, 0.20)
        seamMat.bumpTex = textures.ripPants.bump
        seamMat.uvScale = textures.ripPants.repeatScale
        seamMat.bumpScale = 0.012

        let jacketMat = makeMat(0xffffff, 0.72, 0.04, 0.22, 0.28)
        jacketMat.albedoTex = textures.ripJacket.map
        jacketMat.bumpTex = textures.ripJacket.bump
        jacketMat.roughTex = textures.ripJacket.rough
        jacketMat.uvScale = textures.ripJacket.repeatScale
        jacketMat.bumpScale = 0.019

        let jacketDarkMat = makeMat(0xffffff, 0.74, 0.06, 0.26, 0.24)
        jacketDarkMat.albedoTex = textures.ripDark.map
        jacketDarkMat.bumpTex = textures.ripDark.bump
        jacketDarkMat.roughTex = textures.ripDark.rough
        jacketDarkMat.uvScale = textures.ripDark.repeatScale
        jacketDarkMat.bumpScale = 0.017

        let buckleMat = makeMat(0xb6c0ce, 0.20, 0.90, 0.05, 0.10)
        let orangePullMat = makeMat(0xf46824, 0.36, 0.10, 0.08, 0.20)

        let packMat = makeMat(0xd0d8e4, 0.78, 0.05, 0.24, 0.22)
        packMat.albedoTex = textures.ripDark.map
        packMat.bumpTex = textures.ripPants.bump
        packMat.uvScale = textures.ripDark.repeatScale
        packMat.bumpScale = 0.015

        let helmetMat = makeMat(0x222936, 0.22, 0.30, 0.14, 0.14)
        let goggleMat = makeMat(0xee8c26, 0.05, 0.94, 0.0, 0.08)

        let gloveTanMat = makeMat(0xffffff, 0.64, 0.03, 0.30, 0.30)
        gloveTanMat.albedoTex = textures.leatherTan.map
        gloveTanMat.bumpTex = textures.leatherTan.bump
        gloveTanMat.roughTex = textures.leatherTan.rough
        gloveTanMat.uvScale = textures.leatherTan.repeatScale
        gloveTanMat.bumpScale = 0.025

        let glovePalmMat = makeMat(0xffffff, 0.70, 0.05, 0.28, 0.22)
        glovePalmMat.albedoTex = textures.leatherDark.map
        glovePalmMat.bumpTex = textures.leatherDark.bump
        glovePalmMat.roughTex = textures.leatherDark.rough
        glovePalmMat.uvScale = textures.leatherDark.repeatScale
        glovePalmMat.bumpScale = 0.020

        let gloveShellMat = makeMat(0xc8d2e0, 0.72, 0.05, 0.34, 0.26)
        gloveShellMat.albedoTex = textures.ripDark.map
        gloveShellMat.bumpTex = textures.ripDark.bump
        gloveShellMat.roughTex = textures.ripDark.rough
        gloveShellMat.uvScale = textures.ripDark.repeatScale
        gloveShellMat.bumpScale = 0.016

        let stitchMat = makeMat(0xe6c391, 0.85, 0.02, 0.12, 0.20)

        let webbingMat = makeMat(0xffffff, 0.80, 0.04, 0.22, 0.24)
        webbingMat.albedoTex = textures.webMap
        webbingMat.bumpTex = textures.webBump
        webbingMat.uvScale = textures.webRepeat
        webbingMat.bumpScale = 0.011

        let rubberTabMat = makeMat(0x1c2028, 0.60, 0.08, 0.16, 0.16)
        let badgeMat = makeMat(0xdce4ee, 0.42, 0.25, 0.05, 0.14)

        // Pole materials
        let poleShaftMat = makeMat(0x202630, 0.28, 0.62, 0.15, 0.14)
        poleShaftMat.bumpTex = textures.carbonBump
        poleShaftMat.uvScale = textures.carbonRepeat
        poleShaftMat.bumpScale = 0.009

        let poleGripMat = makeMat(0x26292f, 0.86, 0.04, 0.20, 0.18)
        poleGripMat.bumpTex = textures.carbonBump
        poleGripMat.uvScale = textures.carbonRepeat
        poleGripMat.bumpScale = 0.005

        let poleCorkMat = makeMat(0x3b3530, 0.90, 0.02, 0.15, 0.22)
        poleCorkMat.bumpTex = textures.carbonBump
        poleCorkMat.uvScale = textures.carbonRepeat
        poleCorkMat.bumpScale = 0.007

        let poleBasketMat = makeMat(0x1c222a, 0.70, 0.12, 0.40, 0.12)
        let poleAccentMat = makeMat(0xe64e24, 0.30, 0.56, 0.10, 0.18)
        let poleMetalMat = makeMat(0xb6c0ce, 0.20, 0.90, 0.05, 0.10)
        let poleStrapMat = makeMat(0xffffff, 0.82, 0.04, 0.28, 0.24)
        poleStrapMat.albedoTex = textures.webMap
        poleStrapMat.bumpTex = textures.webBump
        poleStrapMat.uvScale = textures.webRepeat
        poleStrapMat.bumpScale = 0.010

        let shadowOnlyMat = makeMat(0x000000, 1.0, 0.0, 0.0, 0.0)

        let dsMat = RigMaterial(hex: 0xffffff, roughness: 1.0, metalness: 0.0)
        dsMat.albedoTex = textures.dropShadowTex
        dsMat.isDropShadow = true
        dsMat.isDoubleSided = true
        dsMat.depthWrite = false
        dsMat.opacity = 0.72
        self.groundShadowMat = dsMat

        // Helper to upload mesh tuple quickly
        func uploadNode(
            _ meshTuple: ([StandardVertex], [UInt32]),
            mat: RigMaterial,
            pos: SIMD3<Float> = .zero,
            rot: SIMD3<Float> = .zero,
            scale: SIMD3<Float> = SIMD3<Float>(repeating: 1),
            castShadow: Bool = true,
            shadowOnly: Bool = false,
            label: String
        ) -> RigNode {
            let n = RigNode()
            n.position = pos
            n.setEuler(rot.x, rot.y, rot.z)
            n.scale = scale
            if let pm = MeshPrimitives.upload(device: device, vertices: meshTuple.0, indices: meshTuple.1, label: label) {
                n.addPart(RigMeshPart(mesh: pm, material: mat, castShadow: castShadow, shadowOnly: shadowOnly))
            }
            return n
        }

        // Build left & right skis
        func buildSkiAssembly(isLeft: Bool) -> SkiAssembly? {
            let tSamples = buildSkiTSamples()
            let nRows = tSamples.count
            var verts: [StandardVertex] = []
            var indices: [UInt32] = []
            var baseY: [Float] = []
            var tVals: [Float] = []

            for i in 0..<nRows {
                let t = tSamples[i]
                let z = (SkiRigConstants.bootT - t) * SkiRigConstants.skiLen
                let y = evalRockerY(t)
                let w = evalSkiHalfWidth(t)
                let tipBevel: Float = t > 0.78 ? powf((t - 0.78) / 0.22, 1.6) * 0.005 : 0
                let tailBevel: Float = t < 0.16 ? powf((0.16 - t) / 0.16, 1.6) * 0.003 : 0
                let spoonEdge = tipBevel + tailBevel
                let waistMask: Float = (t >= 0.16 && t <= 0.74)
                    ? 1.0
                    : (t > 0.74 ? max(0, 1.0 - (t - 0.74) / 0.11) : max(0, (t - 0.05) / 0.11))
                let wallBot = -0.006 - waistMask * 0.038
                let coreThick = 0.012 + 0.007 * sinf(t * .pi)
                let bevelTop = coreThick * 0.50

                let xs: [Float] = [-w, -w, -w * 0.87, w * 0.87, w, w]
                let us: [Float] = [0.0, 0.0, 0.056, 0.944, 1.0, 1.0]
                let thick: [Float] = [wallBot + spoonEdge, bevelTop + spoonEdge, coreThick, coreThick, bevelTop + spoonEdge, wallBot + spoonEdge]

                for j in 0..<6 {
                    let vy = y + thick[j]
                    let col: SIMD4<Float>
                    if j == 0 || j == 5 {
                        col = SIMD4<Float>(0.12, 0.14, 0.18, 1.0)
                    } else if j == 1 || j == 4 {
                        col = SIMD4<Float>(0.86, 0.90, 0.96, 1.0)
                    } else {
                        col = SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
                    }
                    verts.append(StandardVertex(
                        position: SIMD3<Float>(xs[j], vy, z),
                        normal: SIMD3<Float>(0, 1, 0),
                        uv: SIMD2<Float>(us[j], t),
                        color: col
                    ))
                    baseY.append(vy)
                    tVals.append(t)
                }
                if i < nRows - 1 {
                    let r0 = UInt32(i * 6)
                    let r1 = UInt32((i + 1) * 6)
                    for j in 0..<5 {
                        let jj = UInt32(j)
                        indices.append(contentsOf: [r0 + jj, r0 + jj + 1, r1 + jj, r0 + jj + 1, r1 + jj + 1, r1 + jj])
                    }
                }
            }
            let capTail = UInt32(0)
            indices.append(contentsOf: [
                capTail + 0, capTail + 5, capTail + 1,
                capTail + 1, capTail + 5, capTail + 4,
                capTail + 1, capTail + 4, capTail + 2,
                capTail + 2, capTail + 4, capTail + 3,
            ])
            let capNose = UInt32((nRows - 1) * 6)
            indices.append(contentsOf: [
                capNose + 0, capNose + 1, capNose + 5,
                capNose + 1, capNose + 4, capNose + 5,
                capNose + 1, capNose + 2, capNose + 4,
                capNose + 2, capNose + 3, capNose + 4,
            ])
            MeshPrimitives.computeSmoothNormals(vertices: &verts, indices: indices)

            let skiMat = makeMat(0xffffff, 0.25, 0.28, 0.0, 0.16)
            skiMat.albedoTex = skiTopsheetTex
            skiMat.isSki = true
            guard let bladePM = MeshPrimitives.upload(device: device, vertices: verts, indices: indices, label: isLeft ? "SkiL" : "SkiR") else {
                return nil
            }
            let bladePart = RigMeshPart(mesh: bladePM, material: skiMat, castShadow: true)

            // Shovel dust clump
            let dSegs = 12
            var dVerts: [StandardVertex] = []
            var dIdx: [UInt32] = []
            var dBaseY: [Float] = []
            var dTVals: [Float] = []
            for i in 0...dSegs {
                let f = Float(i) / Float(dSegs)
                let t = 0.56 + f * 0.34
                let z = (SkiRigConstants.bootT - t) * SkiRigConstants.skiLen
                let w = evalSkiHalfWidth(t) * (0.72 * sinf(f * .pi) + 0.14)
                let yBase = evalRockerY(t) + 0.019
                let dome = sinf(f * .pi) * 0.011
                let xs: [Float] = [-w, 0, w]
                let ys: [Float] = [yBase, yBase + dome, yBase]
                for j in 0..<3 {
                    dVerts.append(StandardVertex(position: SIMD3<Float>(xs[j], ys[j], z), normal: SIMD3<Float>(0, 1, 0), uv: SIMD2<Float>(Float(j) * 0.5, f), color: SIMD4<Float>(1, 1, 1, 1)))
                    dBaseY.append(ys[j])
                    dTVals.append(t)
                }
                if i < dSegs {
                    let r0 = UInt32(i * 3)
                    let r1 = UInt32((i + 1) * 3)
                    dIdx.append(contentsOf: [r0, r0 + 1, r1, r0 + 1, r1 + 1, r1])
                    dIdx.append(contentsOf: [r0 + 1, r0 + 2, r1 + 1, r0 + 2, r1 + 2, r1 + 1])
                }
            }
            MeshPrimitives.computeSmoothNormals(vertices: &dVerts, indices: dIdx)
            let dustMat = makeMat(0xf4f8fc, 0.92, 0.0, 0.0, 0.0)
            dustMat.opacity = 0.0
            dustMat.depthWrite = false
            guard let dustPM = MeshPrimitives.upload(device: device, vertices: dVerts, indices: dIdx, label: isLeft ? "SkiDustL" : "SkiDustR") else {
                return nil
            }
            let dustPart = RigMeshPart(mesh: dustPM, material: dustMat, castShadow: false)
            dustPart.visible = false

            // 3D Dynamic Bow-Wave Snow Wash
            let bRows = 16
            let bCols = 8
            var bVerts: [BowWaveVertex] = []
            var bIdx: [UInt32] = []
            let appendBowSheet: (Float, Float, Float, Float) -> Void = { layerIdx, hScale, wExtra, yLiftBase in
                let baseV = UInt32(bVerts.count)
                for i in 0...bRows {
                    let f = Float(i) / Float(bRows)
                    let t = 0.32 + f * 0.70
                    let z = (SkiRigConstants.bootT - t) * SkiRigConstants.skiLen
                    let tClamp = clamp(t, 0, 1)
                    let rockY = evalRockerY(tClamp)
                    let skiHW = evalSkiHalfWidth(tClamp)
                    let env = sinf(powf(f, 0.76) * .pi)
                    let flareOut = skiHW * (1.12 + env * 1.48) + wExtra * env
                    for j in 0...bCols {
                        let s = (Float(j) / Float(bCols)) * 2.0 - 1.0
                        let absS = abs(s)
                        let x = s * flareOut
                        let curlRise = powf(absS, 1.55) * (0.14 + 0.13 * env) * hScale
                        let centerDome = (1.0 - absS * absS) * (0.028 + 0.045 * env) * hScale
                        let prowLift = powf(f, 2.2) * 0.042 * hScale
                        let y = rockY + yLiftBase + centerDome + curlRise + prowLift
                        let zSweep = z + absS * absS * env * 0.18 * (1.0 + layerIdx * 0.35)
                        bVerts.append(BowWaveVertex(
                            position: SIMD3<Float>(x, y, zSweep),
                            normal: SIMD3<Float>(0, 1, 0),
                            uv: SIMD2<Float>(Float(j) / Float(bCols), f),
                            bowData: SIMD4<Float>(layerIdx, absS, env, rockY + 0.008)
                        ))
                    }
                    if i < bRows {
                        let r0 = baseV + UInt32(i * (bCols + 1))
                        let r1 = baseV + UInt32((i + 1) * (bCols + 1))
                        for j in 0..<bCols {
                            let jj = UInt32(j)
                            bIdx.append(contentsOf: [r0 + jj, r0 + jj + 1, r1 + jj, r0 + jj + 1, r1 + jj + 1, r1 + jj])
                        }
                    }
                }
            }
            appendBowSheet(0.0, 0.85, 0.09, 0.018)
            appendBowSheet(1.0, 1.42, 0.21, 0.036)

            // Smooth normals for bow wave
            for i in bVerts.indices { bVerts[i].normal = .zero }
            var bt = 0
            while bt + 2 < bIdx.count {
                let i0 = Int(bIdx[bt]), i1 = Int(bIdx[bt + 1]), i2 = Int(bIdx[bt + 2])
                let n = simd_cross(bVerts[i1].position - bVerts[i0].position, bVerts[i2].position - bVerts[i0].position)
                bVerts[i0].normal += n; bVerts[i1].normal += n; bVerts[i2].normal += n
                bt += 3
            }
            for i in bVerts.indices {
                let len = simd_length(bVerts[i].normal)
                bVerts[i].normal = len > 1e-6 ? (bVerts[i].normal / len) : SIMD3<Float>(0, 1, 0)
            }

            guard let bowVB = bVerts.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: bVerts.count * MemoryLayout<BowWaveVertex>.stride, options: .storageModeShared) }),
                  let bowIB = bIdx.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: bIdx.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) }) else {
                return nil
            }
            bowVB.label = isLeft ? "BowWaveL_VB" : "BowWaveR_VB"
            bowIB.label = isLeft ? "BowWaveL_IB" : "BowWaveR_IB"

            let assembly = SkiAssembly(
                isLeft: isLeft,
                bladePart: bladePart,
                bladeVertices: verts,
                bladeIndices: indices,
                baseY: baseY,
                tVals: tVals,
                dustPart: dustPart,
                dustVertices: dVerts,
                dustIndices: dIdx,
                dBaseY: dBaseY,
                dTVals: dTVals,
                bowVertexBuffer: bowVB,
                bowIndexBuffer: bowIB,
                bowIndexCount: bIdx.count
            )
            assembly.node.addPart(bladePart)
            assembly.node.addPart(dustPart)

            // Bindings
            assembly.node.add(uploadNode(MeshPrimitives.makeBox(width: 0.066, height: 0.015, depth: 0.125), mat: bindMat, pos: SIMD3<Float>(0, 0.025, -0.152), label: "ToePlate"))
            assembly.node.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.028, radiusBottom: 0.034, height: 0.034, radialSegs: 10), mat: gunmetalMat, pos: SIMD3<Float>(0, 0.044, -0.180), label: "ToeTurret"))
            for s: Float in [-1, 1] {
                assembly.node.add(uploadNode(MeshPrimitives.makeBox(width: 0.013, height: 0.024, depth: 0.062), mat: steelMat, pos: SIMD3<Float>(s * 0.033, 0.044, -0.152), rot: SIMD3<Float>(0, s * 0.28, 0), label: "ToeWing"))
            }
            assembly.node.add(uploadNode(MeshPrimitives.makeBox(width: 0.040, height: 0.013, depth: 0.038), mat: coralMetalMat, pos: SIMD3<Float>(0, 0.054, -0.204), rot: SIMD3<Float>(0.32, 0, 0), label: "ToeLever"))
            assembly.node.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.036, radiusBottom: 0.040, height: 0.022, radialSegs: 12), mat: bindMat, pos: SIMD3<Float>(0, 0.028, 0.165), label: "HeelBase"))
            assembly.node.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.030, radiusBottom: 0.033, height: 0.056, radialSegs: 12), mat: gunmetalMat, pos: SIMD3<Float>(0, 0.062, 0.175), rot: SIMD3<Float>(-0.18, 0, 0), label: "HeelTurret"))
            assembly.node.add(uploadNode(MeshPrimitives.makeBox(width: 0.042, height: 0.016, depth: 0.055), mat: coralMetalMat, pos: SIMD3<Float>(0, 0.062, 0.215), rot: SIMD3<Float>(-0.45, 0, 0), label: "HeelLever"))

            for s: Float in [-1, 1] {
                assembly.node.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.004, radiusBottom: 0.004, height: 0.11, radialSegs: 6), mat: steelMat, pos: SIMD3<Float>(s * 0.050, 0.046, 0.105), rot: SIMD3<Float>(.pi * 0.44, 0, s * -0.12), label: "BrakeArm"))
                assembly.node.add(uploadNode(MeshPrimitives.makeBox(width: 0.012, height: 0.016, depth: 0.022), mat: rubberMat, pos: SIMD3<Float>(s * 0.053, 0.062, 0.155), label: "BrakePad"))
            }

            // Sculpted 3-Buckle Freeride Ski Boot
            let bootGroup = RigNode()
            bootGroup.add(uploadNode(MeshPrimitives.makeBox(width: 0.088, height: 0.024, depth: 0.295), mat: rubberMat, pos: SIMD3<Float>(0, 0.036, 0.0), label: "BootSole"))
            bootGroup.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.044, radiusBottom: 0.047, height: 0.21, radialSegs: 12), mat: bootShellMat, pos: SIMD3<Float>(0, 0.068, -0.005), rot: SIMD3<Float>(.pi * 0.5, 0, 0), scale: SIMD3<Float>(1.02, 1.0, 0.86), label: "BootClog"))
            bootGroup.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.044, widthSegs: 12, heightSegs: 10), mat: bootShellMat, pos: SIMD3<Float>(0, 0.064, -0.108), scale: SIMD3<Float>(1.02, 0.76, 1.15), label: "BootToeCap"))
            bootGroup.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.046, widthSegs: 12, heightSegs: 10), mat: bootShellMat, pos: SIMD3<Float>(0, 0.072, 0.100), scale: SIMD3<Float>(0.98, 0.92, 0.95), label: "BootHeelCup"))
            bootGroup.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.055, radiusBottom: 0.048, height: 0.175, radialSegs: 14), mat: bootShellMat, pos: SIMD3<Float>(0, 0.145, 0.042), rot: SIMD3<Float>(-0.24, 0, 0), label: "BootCuff"))
            bootGroup.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.042, radiusBottom: 0.038, height: 0.165, radialSegs: 10), mat: bootTongueMat, pos: SIMD3<Float>(0, 0.152, 0.012), rot: SIMD3<Float>(-0.24, 0, 0), label: "BootTongue"))
            bootGroup.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.046, tube: 0.007, radialSegs: 6, tubularSegs: 14, arc: .pi * 1.15), mat: coralMetalMat, pos: SIMD3<Float>(0, 0.088, -0.052), rot: SIMD3<Float>(.pi * 0.42, 0, .pi * -0.07), label: "BootBuckle1"))
            bootGroup.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.050, tube: 0.007, radialSegs: 6, tubularSegs: 14, arc: .pi * 1.1), mat: coralMetalMat, pos: SIMD3<Float>(0, 0.125, 0.018), rot: SIMD3<Float>(.pi * 0.44, 0, .pi * -0.05), label: "BootBuckle2"))
            bootGroup.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.058, radiusBottom: 0.056, height: 0.028, radialSegs: 14), mat: coralMetalMat, pos: SIMD3<Float>(0, 0.205, 0.025), rot: SIMD3<Float>(-0.24, 0, 0), label: "BootStrap"))

            assembly.ankleAnchor.position = SIMD3<Float>(0, 0.215, 0.020)
            bootGroup.add(assembly.ankleAnchor)
            assembly.node.add(bootGroup)
            return assembly
        }

        guard let leftSki = buildSkiAssembly(isLeft: true),
              let rightSki = buildSkiAssembly(isLeft: false),
              let leftRib = ShadowRibbonMesh(device: device, segs: 20),
              let rightRib = ShadowRibbonMesh(device: device, segs: 20),
              let catcher = ShadowCatcherMesh(device: device) else {
            return nil
        }

        self.leftSki = leftSki
        self.rightSki = rightSki
        self.leftShadowRib = leftRib
        self.rightShadowRib = rightRib
        self.groundCatcher = catcher

        leftSki.node.position = SIMD3<Float>(-SkiRigConstants.skiSep, 0.02, 0)
        rightSki.node.position = SIMD3<Float>(SkiRigConstants.skiSep, 0.02, 0)
        skiPivot.add(leftSki.node)
        skiPivot.add(rightSki.node)
        fpGroup.add(skiPivot)
        rootNode.add(fpGroup)

        // Drop shadow plane
        let dsGeo = MeshPrimitives.makePlaneXZ(width: 2.6, depth: 3.6)
        if let dsPM = MeshPrimitives.upload(device: device, vertices: dsGeo.0, indices: dsGeo.1, label: "DropShadow") {
            groundShadowNode.addPart(RigMeshPart(mesh: dsPM, material: dsMat, castShadow: false))
        }
        rootNode.add(groundShadowNode)

        // Body Group
        rootNode.add(bodyGroup)
        let bodyGroupNode = self.bodyGroup

        // Legs
        func buildLeg(_ side: Float) -> LegRig {
            let shinSeg = RigNode()
            if let sm = makeSculptedLimbMesh(
                device: device, rBottom: 0.074, rTop: 0.078,
                opts: LimbOptions(radSegs: 24, lenSegs: 18, scaleX: 0.96, scaleZ: 1.06, bulgeMid: 0.009, foldAmp: 0.0052, foldFreq: 5.0, phase: side * 1.1, gatherBot: 0.65, gatherTop: 0.45),
                label: "Shin"
            ) {
                shinSeg.addPart(RigMeshPart(mesh: sm, material: pantsMat))
            }
            shinSeg.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.076, radiusBottom: 0.088, height: 0.25, radialSegs: 24), mat: corduraMat, pos: SIMD3<Float>(0, -0.39, 0), label: "Gaiter"))
            let kickStart: Float = side > 0 ? .pi * 0.68 : -.pi * 0.32
            let kickNode = uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.080, radiusBottom: 0.090, height: 0.30, radialSegs: 16, heightSegs: 1, openEnded: true, thetaStart: kickStart, thetaLength: .pi * 0.64), mat: corduraMat, pos: SIMD3<Float>(-side * 0.004, -0.34, 0.0), label: "KickPatch")
            // Kick patch is thin single-sheet arc; make double-sided
            let kickMat = makeMat(0x262e3a, 0.86, 0.05, 0.34, 0.18)
            kickMat.bumpTex = textures.ripPants.bump
            kickMat.uvScale = textures.ripPants.repeatScale
            kickMat.bumpScale = 0.018
            kickMat.isDoubleSided = true
            if let pm = kickNode.parts.first?.mesh {
                kickNode.parts = [RigMeshPart(mesh: pm, material: kickMat)]
            }
            shinSeg.add(kickNode)
            bodyGroupNode.add(shinSeg)

            let kneeBall = RigNode()
            kneeBall.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.079, widthSegs: 22, heightSegs: 16), mat: pantsMat, scale: SIMD3<Float>(0.98, 1.06, 1.08), label: "KneeCore"))
            kneeBall.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.078, widthSegs: 20, heightSegs: 12, phiStart: .pi * 1.5 - 0.78, phiLength: 1.56, thetaStart: 0.95, thetaLength: 1.22), mat: seamMat, scale: SIMD3<Float>(1.03, 1.05, 1.09), label: "KneePatch"))
            for dy: Float in [-0.022, 0.024] {
                kneeBall.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.076, tube: 0.0024, radialSegs: 6, tubularSegs: 16, arc: .pi * 0.55), mat: corduraMat, pos: SIMD3<Float>(0, dy, -0.012), rot: SIMD3<Float>(.pi * 0.5, 0, .pi * 0.22), label: "KneeDart"))
            }
            bodyGroupNode.add(kneeBall)

            let thighSeg = RigNode()
            if let tm = makeSculptedLimbMesh(
                device: device, rBottom: 0.079, rTop: 0.099,
                opts: LimbOptions(radSegs: 24, lenSegs: 18, scaleX: 0.96, scaleZ: 1.08, bulgeMid: 0.012, foldAmp: 0.0056, foldFreq: 4.5, phase: side * 2.3, gatherBot: 0.55, gatherTop: 0.30),
                label: "Thigh"
            ) {
                thighSeg.addPart(RigMeshPart(mesh: tm, material: pantsMat))
            }
            thighSeg.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.056, widthSegs: 14, heightSegs: 12), mat: seamMat, pos: SIMD3<Float>(side * 0.071, -0.01, -0.018), scale: SIMD3<Float>(0.32, 1.85, 0.96), label: "PocketBody"))
            let zipCurve = RigCatmullRomCurve([
                SIMD3<Float>(side * 0.078, -0.22, -0.014),
                SIMD3<Float>(side * 0.087,  0.00, -0.018),
                SIMD3<Float>(side * 0.092,  0.22, -0.014),
            ])
            thighSeg.add(uploadNode(buildTubeGeometry(curve: zipCurve, tubularSegments: 12, radius: 0.0055, radialSegments: 6), mat: corduraMat, label: "ThighZip"))
            let flapCurve = RigCatmullRomCurve([
                SIMD3<Float>(side * 0.077, 0.095, -0.064),
                SIMD3<Float>(side * 0.089, 0.098, -0.018),
                SIMD3<Float>(side * 0.078, 0.095,  0.026),
            ])
            thighSeg.add(uploadNode(buildTubeGeometry(curve: flapCurve, tubularSegments: 10, radius: 0.0095, radialSegments: 8), mat: seamMat, label: "ThighFlap"))
            thighSeg.add(uploadNode(MeshPrimitives.makeCapsule(radius: 0.0052, length: 0.024, capSegs: 4, radialSegs: 8), mat: orangePullMat, pos: SIMD3<Float>(side * 0.091, 0.19, -0.018), label: "ThighZipPull"))
            bodyGroupNode.add(thighSeg)

            return LegRig(shinSeg: shinSeg, kneeBall: kneeBall, thighSeg: thighSeg)
        }
        self.legL = buildLeg(-1)
        self.legR = buildLeg(1)

        // Pelvis & Belt
        bodyGroup.add(pelvisNode)
        pelvisNode.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.198, radiusBottom: 0.194, height: 0.25, radialSegs: 32), mat: pantsMat, scale: SIMD3<Float>(1.09, 1.0, 0.72), label: "HipCore"))
        for s: Float in [-1, 1] {
            pelvisNode.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.092, widthSegs: 20, heightSegs: 14), mat: pantsMat, pos: SIMD3<Float>(s * 0.096, -0.050, -0.004), scale: SIMD3<Float>(0.96, 0.90, 0.94), label: "HipCap"))
        }
        pelvisNode.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.196, radiusBottom: 0.194, height: 0.038, radialSegs: 28), mat: webbingMat, pos: SIMD3<Float>(0, 0.088, 0), scale: SIMD3<Float>(1.07, 1.0, 0.70), label: "Belt"))
        pelvisNode.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.026, widthSegs: 12, heightSegs: 10), mat: buckleMat, pos: SIMD3<Float>(0, 0.088, -0.132), scale: SIMD3<Float>(1.15, 0.78, 0.45), label: "BeltBuckle"))

        // Torso & Jacket
        torsoNode.position = SIMD3<Float>(0, 0.08, 0)
        pelvisNode.add(torsoNode)
        torsoNode.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.198, radiusBottom: 0.224, height: 0.22, radialSegs: 32), mat: jacketMat, pos: SIMD3<Float>(0, 0.008, 0.004), scale: SIMD3<Float>(1.08, 1.0, 0.71), label: "JacketHem"))

        torsoNode.add(chestFrontGroup)
        if let torsoPM = buildJacketTorsoMesh(device: device) {
            chestFrontGroup.addPart(RigMeshPart(mesh: torsoPM, material: jacketMat))
        }
        chestFrontGroup.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.146, radiusBottom: 0.210, height: 0.082, radialSegs: 32), mat: jacketDarkMat, pos: SIMD3<Float>(0, 0.448, 0.002), scale: SIMD3<Float>(1.07, 1.0, 0.63), label: "Yoke"))

        yokeCapNode.position = SIMD3<Float>(0, 0.486, 0.002)
        yokeCapNode.scale = SIMD3<Float>(1.06, 0.26, 0.62)
        let ycGeo = MeshPrimitives.makeSphere(radius: 0.150, widthSegs: 32, heightSegs: 14, phiStart: 0, phiLength: .pi * 2, thetaStart: 0, thetaLength: .pi * 0.5)
        if let ycPM = MeshPrimitives.upload(device: device, vertices: ycGeo.0, indices: ycGeo.1, label: "YokeCap") {
            yokeCapNode.addPart(RigMeshPart(mesh: ycPM, material: jacketDarkMat))
        }
        chestFrontGroup.add(yokeCapNode)

        neckBuffNode.position = SIMD3<Float>(0, 0.520, 0.004)
        neckBuffNode.scale = SIMD3<Float>(1.0, 0.72, 0.96)
        let nbGeo = MeshPrimitives.makeSphere(radius: 0.080, widthSegs: 24, heightSegs: 14)
        if let nbPM = MeshPrimitives.upload(device: device, vertices: nbGeo.0, indices: nbGeo.1, label: "NeckBuff") {
            neckBuffNode.addPart(RigMeshPart(mesh: nbPM, material: corduraMat))
        }
        chestFrontGroup.add(neckBuffNode)

        let fZipCurve = RigCatmullRomCurve([
            SIMD3<Float>(0, 0.00, -0.130),
            SIMD3<Float>(0, 0.14, -0.120),
            SIMD3<Float>(0, 0.34, -0.146),
            SIMD3<Float>(0, 0.46, -0.128),
        ])
        chestFrontGroup.add(uploadNode(buildTubeGeometry(curve: fZipCurve, tubularSegments: 16, radius: 0.0075, radialSegments: 8), mat: jacketDarkMat, label: "FrontZip"))
        chestFrontGroup.add(uploadNode(MeshPrimitives.makeCapsule(radius: 0.0055, length: 0.018, capSegs: 4, radialSegs: 8), mat: orangePullMat, pos: SIMD3<Float>(0, 0.415, -0.142), rot: SIMD3<Float>(0.24, 0, 0), label: "FrontZipTab"))

        let sternCurve = RigCatmullRomCurve([
            SIMD3<Float>(-0.126, 0.28, -0.118),
            SIMD3<Float>(-0.062, 0.28, -0.142),
            SIMD3<Float>( 0.000, 0.28, -0.148),
            SIMD3<Float>( 0.062, 0.28, -0.142),
            SIMD3<Float>( 0.126, 0.28, -0.118),
        ])
        chestFrontGroup.add(uploadNode(buildTubeGeometry(curve: sternCurve, tubularSegments: 16, radius: 0.0095, radialSegments: 8), mat: webbingMat, label: "SternumStrap"))
        chestFrontGroup.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.017, widthSegs: 12, heightSegs: 10), mat: orangePullMat, pos: SIMD3<Float>(0, 0.28, -0.150), scale: SIMD3<Float>(1.25, 0.78, 0.62), label: "SternumClip"))

        for s: Float in [-1, 1] {
            let pktCurve = RigCatmullRomCurve([
                SIMD3<Float>(s * 0.068, 0.39, -0.132),
                SIMD3<Float>(s * 0.088, 0.34, -0.144),
                SIMD3<Float>(s * 0.100, 0.28, -0.136),
            ])
            chestFrontGroup.add(uploadNode(buildTubeGeometry(curve: pktCurve, tubularSegments: 10, radius: 0.0052, radialSegments: 6), mat: jacketDarkMat, label: "ChestPocket"))
            let hsCurve = RigCatmullRomCurve([
                SIMD3<Float>(s * 0.122, 0.06, -0.106),
                SIMD3<Float>(s * 0.124, 0.20, -0.126),
                SIMD3<Float>(s * 0.132, 0.36, -0.134),
                SIMD3<Float>(s * 0.126, 0.495, -0.020),
                SIMD3<Float>(s * 0.112, 0.455,  0.115),
            ])
            chestFrontGroup.add(uploadNode(buildTubeGeometry(curve: hsCurve, tubularSegments: 20, radius: 0.017, radialSegments: 8), mat: webbingMat, scale: SIMD3<Float>(1.0, 1.0, 0.96), label: "HarnessStrap"))
            chestFrontGroup.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.078, widthSegs: 20, heightSegs: 14), mat: jacketMat, pos: SIMD3<Float>(s * 0.194, 0.418, 0.0), scale: SIMD3<Float>(0.90, 0.94, 0.96), label: "RaglanShoulder"))
        }

        chestFrontGroup.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.014, radiusBottom: 0.014, height: 0.088, radialSegs: 12), mat: orangePullMat, pos: SIMD3<Float>(-0.130, 0.33, -0.146), rot: SIMD3<Float>(0, 0, 0.12), label: "DeploySleeve"))
        chestFrontGroup.add(uploadNode(MeshPrimitives.makeCapsule(radius: 0.0085, length: 0.042, capSegs: 4, radialSegs: 8), mat: orangePullMat, pos: SIMD3<Float>(-0.134, 0.285, -0.150), rot: SIMD3<Float>(0, 0, .pi * 0.5), label: "DeployGrip"))

        // Avalanche Airbag Backpack
        torsoNode.add(packGroup)
        if let packPM = buildPackMesh(device: device) {
            packGroup.addPart(RigMeshPart(mesh: packPM, material: packMat))
        }
        packLidNode.position = SIMD3<Float>(0, 0.456, 0.155)
        packLidNode.scale = SIMD3<Float>(1.04, 0.28, 0.52)
        let plGeo = MeshPrimitives.makeSphere(radius: 0.144, widthSegs: 28, heightSegs: 14)
        if let plPM = MeshPrimitives.upload(device: device, vertices: plGeo.0, indices: plGeo.1, label: "PackLid") {
            packLidNode.addPart(RigMeshPart(mesh: plPM, material: corduraMat))
        }
        packGroup.add(packLidNode)
        packGroup.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.207, radiusBottom: 0.209, height: 0.044, radialSegs: 32), mat: webbingMat, pos: SIMD3<Float>(0, 0.085, 0.002), scale: SIMD3<Float>(1.08, 1.0, 0.71), label: "HipBelt"))
        let diagStrap = RigCatmullRomCurve([
            SIMD3<Float>(-0.105, 0.12, 0.214),
            SIMD3<Float>( 0.000, 0.255, 0.241),
            SIMD3<Float>( 0.105, 0.39, 0.212),
        ])
        packGroup.add(uploadNode(buildTubeGeometry(curve: diagStrap, tubularSegments: 16, radius: 0.0085, radialSegments: 8), mat: webbingMat, label: "PackDiagStrap"))
        let packAccent = RigCatmullRomCurve([
            SIMD3<Float>(0, 0.13, 0.233),
            SIMD3<Float>(0, 0.255, 0.242),
            SIMD3<Float>(0, 0.38, 0.230),
        ])
        packGroup.add(uploadNode(buildTubeGeometry(curve: packAccent, tubularSegments: 14, radius: 0.0068, radialSegments: 8), mat: orangePullMat, label: "PackAccent"))

        // Head, Helmet, Goggles, Hood
        torsoNode.add(headGroup)
        headGroup.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.108, radiusBottom: 0.140, height: 0.095, radialSegs: 24), mat: jacketMat, pos: SIMD3<Float>(0, 0.525, 0.010), label: "StormCollar"))
        headGroup.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.088, tube: 0.022, radialSegs: 10, tubularSegs: 20, arc: .pi * 0.85), mat: jacketMat, pos: SIMD3<Float>(0, 0.542, 0.068), rot: SIMD3<Float>(.pi * 0.44, 0, .pi * 0.075), label: "HoodFold"))

        headNode.position = SIMD3<Float>(0, 0.655, -0.015)
        headGroup.add(headNode)
        headNode.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.116, widthSegs: 22, heightSegs: 16), mat: corduraMat, pos: SIMD3<Float>(0, -0.022, -0.012), scale: SIMD3<Float>(0.90, 1.06, 0.98), label: "FaceMask"))
        headNode.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.135, widthSegs: 26, heightSegs: 20, phiStart: 0, phiLength: .pi * 2, thetaStart: 0, thetaLength: .pi * 0.62), mat: helmetMat, pos: SIMD3<Float>(0, 0.012, 0.005), scale: SIMD3<Float>(0.97, 1.02, 1.09), label: "Helmet"))
        headNode.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.133, widthSegs: 22, heightSegs: 14, phiStart: .pi * 0.14, phiLength: .pi * 0.72, thetaStart: .pi * 0.48, thetaLength: .pi * 0.34), mat: helmetMat, pos: SIMD3<Float>(0, 0.012, 0.005), scale: SIMD3<Float>(0.97, 1.02, 1.09), label: "HelmetRear"))
        let brimNode = uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.130, radiusBottom: 0.146, height: 0.020, radialSegs: 20, heightSegs: 1, openEnded: true, thetaStart: .pi * 0.72, thetaLength: .pi * 0.56), mat: helmetMat, pos: SIMD3<Float>(0, 0.036, -0.024), rot: SIMD3<Float>(0.28, 0, 0), label: "HelmetBrim")
        headNode.add(brimNode)
        headNode.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.139, widthSegs: 26, heightSegs: 14, phiStart: .pi * 1.21, phiLength: .pi * 0.58, thetaStart: .pi * 0.39, thetaLength: .pi * 0.27), mat: goggleMat, pos: SIMD3<Float>(0, -0.006, -0.016), label: "Goggles"))
        headNode.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.137, widthSegs: 26, heightSegs: 14, phiStart: .pi * 1.19, phiLength: .pi * 0.62, thetaStart: .pi * 0.37, thetaLength: .pi * 0.31), mat: rubberTabMat, pos: SIMD3<Float>(0, -0.006, -0.012), label: "GoggleRim"))
        headNode.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.133, radiusBottom: 0.133, height: 0.044, radialSegs: 24, heightSegs: 1, openEnded: true), mat: webbingMat, pos: SIMD3<Float>(0, -0.004, 0.006), label: "GoggleStrap"))
        headNode.add(uploadNode(MeshPrimitives.makeCapsule(radius: 0.011, length: 0.034, capSegs: 4, radialSegs: 8), mat: orangePullMat, pos: SIMD3<Float>(0, -0.004, 0.140), label: "GoggleRearClip"))

        // Shadow-caster-only proxies (FP Helmet & Chesty Torso)
        torsoNode.add(headShadowProxy)
        headShadowProxy.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.140, widthSegs: 12, heightSegs: 10), mat: shadowOnlyMat, pos: SIMD3<Float>(0, 0.68, -0.02), scale: SIMD3<Float>(0.96, 1.02, 1.08), castShadow: true, shadowOnly: true, label: "HeadShadowSphere"))
        headShadowProxy.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.11, radiusBottom: 0.145, height: 0.16, radialSegs: 10), mat: shadowOnlyMat, pos: SIMD3<Float>(0, 0.56, 0.01), castShadow: true, shadowOnly: true, label: "HeadShadowNeck"))

        torsoNode.add(chestShadowProxy)
        chestShadowProxy.visible = false
        chestShadowProxy.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.21, radiusBottom: 0.19, height: 0.44, radialSegs: 12), mat: shadowOnlyMat, pos: SIMD3<Float>(0, 0.28, 0.0), scale: SIMD3<Float>(1.08, 1.0, 0.64), castShadow: true, shadowOnly: true, label: "ChestShadowTorso"))
        chestShadowProxy.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.15, radiusBottom: 0.16, height: 0.43, radialSegs: 12), mat: shadowOnlyMat, pos: SIMD3<Float>(0, 0.26, 0.165), scale: SIMD3<Float>(1.06, 1.0, 0.60), castShadow: true, shadowOnly: true, label: "ChestShadowPack"))

        // Hero Multi-Panel Freeride Leather Glove + Carbon Ski Pole
        func buildFreerideGlove(_ side: Float) -> RigNode {
            let g = RigNode()
            // 1. Metacarpal hand body
            let mLen = 14, mRad = 24
            let rowCols = mRad + 1
            var mVerts: [StandardVertex] = []
            var mIdx: [UInt32] = []
            for i in 0...mLen {
                let f = Float(i) / Float(mLen)
                let z = lerp(0.100, -0.016, f)
                let widthEnv = sinf(powf(f, 0.72) * .pi * 0.78)
                let ry = 0.032 + 0.014 * widthEnv + (f > 0.75 ? -0.002 * ((f - 0.75) / 0.25) : 0)
                let rx = 0.0235 - 0.0042 * f
                let cx = side * (0.006 + 0.0085 * f)
                let cy = 0.001 - 0.003 * (1.0 - f)
                for j in 0...mRad {
                    let u = Float(j) / Float(mRad)
                    let th = u * .pi * 2.0
                    let cs = cosf(th), sn = sinf(th)
                    let sxSign = (cs >= 0 ? Float(1) : Float(-1)) * powf(abs(cs), 0.78)
                    let sySign = (sn >= 0 ? Float(1) : Float(-1)) * powf(abs(sn), 0.82)
                    var px = cx + sxSign * rx
                    var py = cy + sySign * ry
                    let isPalm = max(0, -cs * side)
                    px += side * isPalm * 0.0040 * sinf(f * .pi)
                    let isThumbWeb = max(0, -cs * side) * max(0, sn + 0.18)
                    let webEnv = sinf(min(1, max(0, (f - 0.14) / 0.72)) * .pi)
                    px -= side * isThumbWeb * 0.0058 * webEnv
                    py += isThumbWeb * 0.0048 * webEnv
                    let isDorsal = max(0, cs * side)
                    let knuckRidge = isDorsal * powf(f, 2.2) * (0.0028 + 0.0020 * cosf(sn * .pi * 3.8))
                    let leatherCrease = sinf(f * 20.0 + sn * 2.2) * 0.0012 * (1.0 - f * 0.35)
                    px += side * (knuckRidge + leatherCrease * (0.35 + 0.65 * isDorsal))
                    let closeCap: Float = f > 0.96 ? (1.0 - ((f - 0.96) / 0.04) * 0.14) : 1.0
                    mVerts.append(StandardVertex(
                        position: SIMD3<Float>(cx + (px - cx) * closeCap, cy + (py - cy) * closeCap, z),
                        normal: SIMD3<Float>(0, 1, 0),
                        uv: SIMD2<Float>(u * 1.4, f * 1.1),
                        color: SIMD4<Float>(1, 1, 1, 1)
                    ))
                }
            }
            for i in 0..<mLen {
                let r0 = UInt32(i * rowCols), r1 = UInt32((i + 1) * rowCols)
                for j in 0..<mRad {
                    let jj = UInt32(j)
                    mIdx.append(contentsOf: [r0 + jj, r1 + jj, r0 + jj + 1, r0 + jj + 1, r1 + jj, r1 + jj + 1])
                }
            }
            let capCIdx = UInt32(mVerts.count)
            mVerts.append(StandardVertex(position: SIMD3<Float>(side * 0.014, 0.001, -0.020), normal: SIMD3<Float>(0, 0, -1), uv: SIMD2<Float>(0.5, 1.0), color: SIMD4<Float>(1, 1, 1, 1)))
            let lastRow = UInt32(mLen * rowCols)
            for j in 0..<mRad {
                let jj = UInt32(j)
                mIdx.append(contentsOf: [lastRow + jj, capCIdx, lastRow + jj + 1])
            }
            MeshPrimitives.computeSmoothNormals(vertices: &mVerts, indices: mIdx)
            if let pm = MeshPrimitives.upload(device: device, vertices: mVerts, indices: mIdx, label: "GloveMeta") {
                g.addPart(RigMeshPart(mesh: pm, material: gloveTanMat))
            }

            // 2. Softshell Backhand Armor Overlay
            let bhLen = 10, bhRad = 14, bhCols = bhRad + 1
            var bhVerts: [StandardVertex] = []
            var bhIdx: [UInt32] = []
            for i in 0...bhLen {
                let f = 0.08 + (Float(i) / Float(bhLen)) * 0.72
                let z = lerp(0.100, -0.016, f)
                let widthEnv = sinf(powf(f, 0.72) * .pi * 0.78)
                let ry = 0.032 + 0.014 * widthEnv + (f > 0.75 ? -0.002 * ((f - 0.75) / 0.25) : 0)
                let rx = 0.0235 - 0.0042 * f
                let cx = side * (0.006 + 0.0085 * f)
                let cy = 0.001 - 0.003 * (1.0 - f)
                let quilt = sinf((Float(i) / Float(bhLen)) * .pi * 4.0) * 0.0010
                let webEnv = sinf(min(1, max(0, (f - 0.14) / 0.72)) * .pi)
                for j in 0...bhRad {
                    let aNorm = (Float(j) / Float(bhRad)) * 2.0 - 1.0
                    let a = 0.22 + aNorm * 0.96
                    let th = side > 0 ? a : .pi - a
                    let cs = cosf(th), sn = sinf(th)
                    let sxSign = (cs >= 0 ? Float(1) : Float(-1)) * powf(abs(cs), 0.78)
                    let sySign = (sn >= 0 ? Float(1) : Float(-1)) * powf(abs(sn), 0.82)
                    let isPalm = max(0, -cs * side)
                    let isDorsal = max(0, cs * side)
                    let isThumbWeb = max(0, -cs * side) * max(0, sn + 0.18)
                    let knuckRidge = isDorsal * powf(f, 2.2) * (0.0028 + 0.0020 * cosf(sn * .pi * 3.8))
                    let leatherCrease = sinf(f * 20.0 + sn * 2.2) * 0.0012 * (1.0 - f * 0.35)
                    let basePx = cx + sxSign * rx + side * (isPalm * 0.0040 * sinf(f * .pi) - isThumbWeb * 0.0058 * webEnv + knuckRidge + leatherCrease * (0.35 + 0.65 * isDorsal))
                    let basePy = cy + sySign * ry + isThumbWeb * 0.0048 * webEnv
                    let lift = 0.0028 + quilt * cosf(aNorm * 1.2)
                    bhVerts.append(StandardVertex(
                        position: SIMD3<Float>(basePx + cs * lift, basePy + sn * lift, z),
                        normal: SIMD3<Float>(0, 1, 0),
                        uv: SIMD2<Float>(Float(j) / Float(bhRad), Float(i) / Float(bhLen)),
                        color: SIMD4<Float>(1, 1, 1, 1)
                    ))
                }
            }
            for i in 0..<bhLen {
                let r0 = UInt32(i * bhCols), r1 = UInt32((i + 1) * bhCols)
                for j in 0..<bhRad {
                    let jj = UInt32(j)
                    if side > 0 {
                        bhIdx.append(contentsOf: [r0 + jj, r1 + jj, r0 + jj + 1, r0 + jj + 1, r1 + jj, r1 + jj + 1])
                    } else {
                        bhIdx.append(contentsOf: [r0 + jj, r0 + jj + 1, r1 + jj, r0 + jj + 1, r1 + jj + 1, r1 + jj])
                    }
                }
            }
            MeshPrimitives.computeSmoothNormals(vertices: &bhVerts, indices: bhIdx)
            if let pm = MeshPrimitives.upload(device: device, vertices: bhVerts, indices: bhIdx, label: "GloveBackhand") {
                g.addPart(RigMeshPart(mesh: pm, material: gloveShellMat))
            }

            // Stitch welt
            var stitchPts: [SIMD3<Float>] = []
            for k in 0...12 {
                let aNorm = (Float(k) / 12.0) * 2.0 - 1.0
                let a = 0.36 + aNorm * 1.12
                let th = side > 0 ? a : .pi - a
                let cs = cosf(th), sn = sinf(th)
                let sxSign = (cs >= 0 ? Float(1) : Float(-1)) * powf(abs(cs), 0.78)
                let sySign = (sn >= 0 ? Float(1) : Float(-1)) * powf(abs(sn), 0.82)
                stitchPts.append(SIMD3<Float>(side * 0.013 + sxSign * 0.0218, sySign * 0.0425, 0.008 + abs(aNorm) * 0.008))
            }
            g.add(uploadNode(buildTubeGeometry(curve: RigCatmullRomCurve(stitchPts), tubularSegments: 18, radius: 0.0016, radialSegments: 6), mat: stitchMat, label: "GloveStitch"))

            // 3. Ribbed Accordion Knuckle Flex Guard
            for rib in 0..<3 {
                let rz = -0.001 - Float(rib) * 0.0068
                var rPts: [SIMD3<Float>] = []
                for k in 0...10 {
                    let aNorm = (Float(k) / 10.0) * 2.0 - 1.0
                    let a = 0.32 + aNorm * 1.04
                    let th = side > 0 ? a : .pi - a
                    let cs = cosf(th), sn = sinf(th)
                    let sxSign = (cs >= 0 ? Float(1) : Float(-1)) * powf(abs(cs), 0.78)
                    let sySign = (sn >= 0 ? Float(1) : Float(-1)) * powf(abs(sn), 0.82)
                    rPts.append(SIMD3<Float>(side * 0.0135 + sxSign * (0.0205 - Float(rib) * 0.0008), sySign * 0.041, rz))
                }
                let rMat = rib == 1 ? glovePalmMat : gloveTanMat
                g.add(uploadNode(buildTubeGeometry(curve: RigCatmullRomCurve(rPts), tubularSegments: 16, radius: 0.0029, radialSegments: 8), mat: rMat, label: "GloveKnuckleRib"))
            }

            // 4. Palm Pad
            var palmPts: [SIMD3<Float>] = []
            for k in 0...8 {
                let f = Float(k) / 8.0
                palmPts.append(SIMD3<Float>(
                    -side * (0.010 + sinf(f * .pi) * 0.008),
                    lerp(-0.028, 0.026, f),
                    lerp(0.056, 0.012, sinf(f * .pi * 0.7))
                ))
            }
            g.add(uploadNode(buildTubeGeometry(curve: RigCatmullRomCurve(palmPts), tubularSegments: 12, radius: 0.0095, radialSegments: 8), mat: glovePalmMat, scale: SIMD3<Float>(0.55, 1.0, 1.35), label: "GlovePalmPad"))

            // 5. Four Articulated Curled Fingers
            let px = -side * 0.012, pz: Float = -0.024, gripR: Float = 0.0165
            let fingerSpecs: [(y: Float, r: Float, reach: Float, tuck: Float)] = [
                ( 0.0305, 0.0118, 1.02, 0.96),
                ( 0.0095, 0.0124, 1.06, 1.00),
                (-0.0115, 0.0115, 1.02, 0.98),
                (-0.0305, 0.0101, 0.96, 0.92),
            ]
            for (fi, fs) in fingerSpecs.enumerated() {
                let dist = gripR + fs.r * 0.86
                let fCurve = RigCatmullRomCurve([
                    SIMD3<Float>(side * 0.0145, fs.y + 0.001, -0.002),
                    SIMD3<Float>(side * (0.0185 + fs.r * 0.16), fs.y, pz - 0.006 * fs.reach),
                    SIMD3<Float>(-side * 0.0015, fs.y - 0.002, pz - dist * 1.04 * fs.reach),
                    SIMD3<Float>(px - side * dist * 0.92 * fs.tuck, fs.y - 0.001, pz - dist * 0.14),
                    SIMD3<Float>(px - side * dist * 0.56 * fs.tuck, fs.y + 0.001, pz + dist * 0.64),
                ], type: .centripetal)
                if let fPM = buildSculptedFingerMesh(device: device, curve: fCurve, radFn: { u in
                    let taper = 1.0 - 0.16 * u
                    let pip = expf(-powf((u - 0.48) / 0.11, 2)) * 0.11
                    let dip = expf(-powf((u - 0.76) / 0.09, 2)) * 0.07
                    return fs.r * (taper + pip + dip)
                }, segs: 18, radSegs: 12, flatAspect: 0.90, label: "GloveFinger_\(fi)") {
                    g.addPart(RigMeshPart(mesh: fPM, material: gloveTanMat))
                }
                if fi < 3 {
                    let capCurve = RigCatmullRomCurve([
                        fCurve.getPoint(0.60),
                        fCurve.getPoint(0.75),
                        fCurve.getPoint(0.88),
                        fCurve.getPoint(0.98),
                    ])
                    g.add(uploadNode(buildTubeGeometry(curve: capCurve, tubularSegments: 8, radius: fs.r * 0.87, radialSegments: 8), mat: glovePalmMat, pos: SIMD3<Float>(side * 0.0012, 0, 0.0012), label: "FingerWearCap"))
                }
            }

            // 6. Opposable Thumb
            let thumbCurve = RigCatmullRomCurve([
                SIMD3<Float>( side * 0.005, 0.015,  0.056),
                SIMD3<Float>(-side * 0.006, 0.024,  0.026),
                SIMD3<Float>(-side * 0.022, 0.034,  0.000),
                SIMD3<Float>(-side * 0.028, 0.039, -0.021),
                SIMD3<Float>(-side * 0.008, 0.041, -0.043),
            ], type: .centripetal)
            if let thPM = buildSculptedFingerMesh(device: device, curve: thumbCurve, radFn: { u in
                let rootFade: Float = u < 0.25 ? lerp(0.0092, 0.0138, u / 0.25) : lerp(0.0138, 0.0108, (u - 0.25) / 0.75)
                let ipBulge = expf(-powf((u - 0.62) / 0.13, 2)) * 0.0014
                return rootFade + ipBulge
            }, segs: 18, radSegs: 14, flatAspect: 0.86, label: "GloveThumb") {
                g.addPart(RigMeshPart(mesh: thPM, material: gloveTanMat))
            }
            let crotchCurve = RigCatmullRomCurve([
                SIMD3<Float>( side * 0.001, 0.029,  0.040),
                SIMD3<Float>(-side * 0.007, 0.033,  0.014),
                SIMD3<Float>(-side * 0.016, 0.036, -0.010),
            ])
            g.add(uploadNode(buildTubeGeometry(curve: crotchCurve, tubularSegments: 10, radius: 0.0018, radialSegments: 6), mat: stitchMat, label: "ThumbStitch"))

            // 7. Wrist Gauntlet + Velcro Cinch Strap + Hang Loop
            let gLen = 8, gRad = 20, gCols = gRad + 1
            var gVerts: [StandardVertex] = []
            var gIdx: [UInt32] = []
            for i in 0...gLen {
                let f = Float(i) / Float(gLen)
                let z = lerp(0.082, 0.138, f)
                let flare = 1.0 + sinf(f * .pi * 0.65) * 0.14
                let rx = 0.0255 * flare + sinf(f * 14.0) * 0.0012
                let ry = 0.0345 * flare + sinf(f * 14.0) * 0.0012
                for j in 0...gRad {
                    let u = Float(j) / Float(gRad)
                    let th = u * .pi * 2.0
                    gVerts.append(StandardVertex(
                        position: SIMD3<Float>(side * 0.004 + cosf(th) * rx, -0.001 + sinf(th) * ry, z),
                        normal: SIMD3<Float>(0, 1, 0),
                        uv: SIMD2<Float>(u * 1.4, f * 0.6),
                        color: SIMD4<Float>(1, 1, 1, 1)
                    ))
                }
            }
            for i in 0..<gLen {
                let r0 = UInt32(i * gCols), r1 = UInt32((i + 1) * gCols)
                for j in 0..<gRad {
                    let jj = UInt32(j)
                    gIdx.append(contentsOf: [r0 + jj, r0 + jj + 1, r1 + jj, r0 + jj + 1, r1 + jj + 1, r1 + jj])
                }
            }
            MeshPrimitives.computeSmoothNormals(vertices: &gVerts, indices: gIdx)
            if let gPM = MeshPrimitives.upload(device: device, vertices: gVerts, indices: gIdx, label: "GloveGauntlet") {
                g.addPart(RigMeshPart(mesh: gPM, material: gloveShellMat))
            }

            var cinchPts: [SIMD3<Float>] = []
            for k in 0...16 {
                let a = (Float(k) / 16.0) * .pi * 1.55 - .pi * 0.72
                let th = side > 0 ? a : .pi - a
                cinchPts.append(SIMD3<Float>(side * 0.004 + cosf(th) * 0.0285, -0.001 + sinf(th) * 0.0372, 0.092))
            }
            g.add(uploadNode(buildTubeGeometry(curve: RigCatmullRomCurve(cinchPts), tubularSegments: 18, radius: 0.0052, radialSegments: 8), mat: webbingMat, label: "GloveCinch"))
            g.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.011, tube: 0.0026, radialSegs: 6, tubularSegs: 12, arc: .pi * 1.3), mat: orangePullMat, pos: SIMD3<Float>(side * 0.006, -0.036, 0.098), rot: SIMD3<Float>(0.45, .pi * 0.5, 0), label: "GloveHangLoop"))
            g.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.0068, tube: 0.0016, radialSegs: 6, tubularSegs: 10), mat: buckleMat, pos: SIMD3<Float>(side * 0.018, -0.030, 0.088), rot: SIMD3<Float>(0, .pi * 0.5, 0), label: "GloveDRing"))
            return g
        }

        func buildPole(_ side: Float) -> RigNode {
            let g = RigNode()
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.0094, radiusBottom: 0.0058, height: 1.24, radialSegs: 16), mat: poleShaftMat, pos: SIMD3<Float>(0, -0.60, 0), label: "PoleShaft"))
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.0124, radiusBottom: 0.0114, height: 0.048, radialSegs: 16), mat: poleAccentMat, pos: SIMD3<Float>(0, -0.34, 0), label: "PoleLockBody"))
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.0132, radiusBottom: 0.0126, height: 0.012, radialSegs: 16), mat: poleMetalMat, pos: SIMD3<Float>(0, -0.318, 0), label: "PoleLockCap"))
            g.add(uploadNode(MeshPrimitives.makeCapsule(radius: 0.0038, length: 0.022, capSegs: 4, radialSegs: 8), mat: poleMetalMat, pos: SIMD3<Float>(side * 0.013, -0.34, -0.004), rot: SIMD3<Float>(0, 0, side * 0.14), label: "PoleLockLever"))
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.0102, radiusBottom: 0.0099, height: 0.055, radialSegs: 16), mat: poleAccentMat, pos: SIMD3<Float>(0, -0.23, 0), label: "PoleUpperBand"))
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.0080, radiusBottom: 0.0076, height: 0.036, radialSegs: 16), mat: poleAccentMat, pos: SIMD3<Float>(0, -0.64, 0), label: "PoleLowerBand"))

            let gripPts: [SIMD2<Float>] = [
                SIMD2<Float>(0.0098, -0.195),
                SIMD2<Float>(0.0138, -0.185),
                SIMD2<Float>(0.0142, -0.110),
                SIMD2<Float>(0.0134, -0.092),
                SIMD2<Float>(0.0168, -0.072),
                SIMD2<Float>(0.0156, -0.048),
                SIMD2<Float>(0.0166, -0.030),
                SIMD2<Float>(0.0158, -0.014),
                SIMD2<Float>(0.0168,  0.004),
                SIMD2<Float>(0.0160,  0.020),
                SIMD2<Float>(0.0170,  0.036),
                SIMD2<Float>(0.0162,  0.048),
                SIMD2<Float>(0.0218,  0.064),
                SIMD2<Float>(0.0226,  0.075),
                SIMD2<Float>(0.0195,  0.084),
                SIMD2<Float>(0.0110,  0.089),
                SIMD2<Float>(0.0005,  0.090),
            ]
            g.add(uploadNode(buildLatheGeometry(points: gripPts, segments: 24), mat: poleGripMat, label: "PoleGrip"))
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.0105, radiusBottom: 0.0122, height: 0.006, radialSegs: 18), mat: poleAccentMat, pos: SIMD3<Float>(0, 0.089, 0), label: "PommelRivet"))
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.0055, radiusBottom: 0.0055, height: 0.007, radialSegs: 14), mat: poleMetalMat, pos: SIMD3<Float>(0, 0.090, 0), label: "PommelNut"))
            g.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.0142, tube: 0.0022, radialSegs: 8, tubularSegs: 20), mat: poleCorkMat, pos: SIMD3<Float>(0, -0.092, 0), rot: SIMD3<Float>(.pi * 0.5, 0, 0), label: "ChokeRing"))

            let strapCurve = RigCatmullRomCurve([
                SIMD3<Float>(0.0,           0.065, 0.014),
                SIMD3<Float>(side * 0.028,  0.044, 0.046),
                SIMD3<Float>(side * 0.045,  0.012, 0.088),
                SIMD3<Float>(side * 0.028, -0.024, 0.112),
                SIMD3<Float>(-side * 0.004, -0.016, 0.098),
                SIMD3<Float>(-side * 0.012,  0.014, 0.058),
                SIMD3<Float>(-side * 0.005,  0.047, 0.025),
                SIMD3<Float>(0.0,           0.062, 0.014),
            ], type: .centripetal)
            g.add(uploadNode(buildTubeGeometry(curve: strapCurve, tubularSegments: 32, radius: 0.0036, radialSegments: 8), mat: poleStrapMat, label: "PoleStrap"))
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.007, radiusBottom: 0.007, height: 0.016, radialSegs: 10), mat: poleMetalMat, pos: SIMD3<Float>(0, 0.066, 0.014), rot: SIMD3<Float>(0, 0, .pi * 0.5), label: "PoleStrapSlot"))

            g.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.047, tube: 0.0055, radialSegs: 8, tubularSegs: 20), mat: poleBasketMat, pos: SIMD3<Float>(0, -1.14, 0), rot: SIMD3<Float>(.pi * 0.5, 0, 0), label: "BasketRing"))
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.019, radiusBottom: 0.014, height: 0.022, radialSegs: 14), mat: poleBasketMat, pos: SIMD3<Float>(0, -1.14, 0), label: "BasketHub"))
            let basketWebMat = makeMat(0x1c222a, 0.70, 0.12, 0.40, 0.12)
            basketWebMat.isDoubleSided = true
            g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.046, radiusBottom: 0.018, height: 0.009, radialSegs: 18, heightSegs: 1, openEnded: true), mat: basketWebMat, pos: SIMD3<Float>(0, -1.139, 0), label: "BasketWeb"))
            for k in 0..<6 {
                let a = (Float(k) / 6.0) * .pi * 2.0
                g.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.0024, radiusBottom: 0.0024, height: 0.046, radialSegs: 6), mat: poleBasketMat, pos: SIMD3<Float>(cosf(a) * 0.024, -1.14, -sinf(a) * 0.024), rot: SIMD3<Float>(0, a, .pi * 0.5), label: "BasketSpoke"))
            }
            g.add(uploadNode(MeshPrimitives.makeCone(radius: 0.0058, height: 0.048, radialSegs: 10), mat: poleMetalMat, pos: SIMD3<Float>(0, -1.21, 0), rot: SIMD3<Float>(.pi, 0, 0), label: "PoleTipSpike"))
            return g
        }

        func buildArm(_ side: Float) -> ArmRig {
            let shoulderBall = uploadNode(MeshPrimitives.makeSphere(radius: 0.076, widthSegs: 20, heightSegs: 14), mat: jacketMat, scale: SIMD3<Float>(0.94, 1.0, 0.96), label: "ShoulderBall")
            bodyGroupNode.add(shoulderBall)

            let upperSeg = RigNode()
            if let um = makeSculptedLimbMesh(
                device: device, rBottom: 0.064, rTop: 0.074,
                opts: LimbOptions(radSegs: 24, lenSegs: 16, scaleX: 0.96, scaleZ: 1.04, bulgeMid: 0.008, foldAmp: 0.0050, foldFreq: 4.5, phase: side * 0.9, gatherBot: 0.55, gatherTop: 0.35),
                label: "UpperArm"
            ) {
                upperSeg.addPart(RigMeshPart(mesh: um, material: jacketMat))
            }
            if side < 0 {
                upperSeg.add(uploadNode(MeshPrimitives.makeCylinder(radiusTop: 0.022, radiusBottom: 0.022, height: 0.006, radialSegs: 16), mat: jacketDarkMat, pos: SIMD3<Float>(-0.073, 0.12, 0.0), rot: SIMD3<Float>(0, 0, .pi * 0.5), label: "ArmPatch"))
                upperSeg.add(uploadNode(MeshPrimitives.makeCone(radius: 0.013, height: 0.018, radialSegs: 3), mat: badgeMat, pos: SIMD3<Float>(-0.076, 0.12, 0.0), rot: SIMD3<Float>(0, 0, .pi * 0.5), label: "ArmPatchTri"))
            }
            bodyGroupNode.add(upperSeg)

            let elbowBall = RigNode()
            elbowBall.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.066, widthSegs: 20, heightSegs: 14), mat: jacketMat, scale: SIMD3<Float>(0.95, 1.05, 1.02), label: "ElbowCore"))
            elbowBall.add(uploadNode(MeshPrimitives.makeSphere(radius: 0.0668, widthSegs: 16, heightSegs: 10, phiStart: .pi * 0.5 - 0.65, phiLength: 1.3, thetaStart: 0.95, thetaLength: 1.15), mat: jacketDarkMat, scale: SIMD3<Float>(1.01, 1.04, 1.04), label: "ElbowGusset"))
            bodyGroupNode.add(elbowBall)

            let lowerSeg = RigNode()
            if let lm = makeSculptedLimbMesh(
                device: device, rBottom: 0.041, rTop: 0.063,
                opts: LimbOptions(radSegs: 24, lenSegs: 20, scaleX: 0.92, scaleZ: 1.08, bulgeMid: 0.006, foldAmp: 0.0038, foldFreq: 6.0, phase: side * 1.7 + 0.5, gatherBot: 0.0, gatherTop: 0.50),
                label: "Forearm"
            ) {
                lowerSeg.addPart(RigMeshPart(mesh: lm, material: jacketMat))
            }
            bodyGroupNode.add(lowerSeg)

            let cuffNode = RigNode()
            let cuffCollar = RigNode()
            cuffCollar.scale = SIMD3<Float>(1.0, 0.058, 1.0)
            cuffCollar.position = SIMD3<Float>(0, 0.019, 0)
            if let cm = makeSculptedLimbMesh(
                device: device, rBottom: 0.050, rTop: 0.056,
                opts: LimbOptions(radSegs: 24, lenSegs: 8, scaleX: 0.93, scaleZ: 1.09, foldAmp: 0.0012, foldFreq: 8.0, phase: side * 0.6, gatherBot: 0.0, gatherTop: 0.0),
                label: "CuffCollar"
            ) {
                cuffCollar.addPart(RigMeshPart(mesh: cm, material: jacketDarkMat))
            }
            cuffNode.add(cuffCollar)
            cuffNode.add(uploadNode(MeshPrimitives.makeTorus(radius: 0.050, tube: 0.0038, radialSegs: 8, tubularSegs: 24), mat: rubberTabMat, pos: SIMD3<Float>(0, -0.010, 0), rot: SIMD3<Float>(.pi * 0.5, 0, 0), scale: SIMD3<Float>(0.93, 1.09, 1.0), label: "CuffTrim"))

            var vPts: [SIMD3<Float>] = []
            for k in 0...14 {
                let a = (Float(k) / 14.0) * .pi * 1.15 - .pi * 0.35
                let th = side > 0 ? a : .pi - a
                vPts.append(SIMD3<Float>(cosf(th) * 0.054 * 0.93, 0.014, sinf(th) * 0.054 * 1.09))
            }
            cuffNode.add(uploadNode(buildTubeGeometry(curve: RigCatmullRomCurve(vPts), tubularSegments: 16, radius: 0.0068, radialSegments: 8), mat: rubberTabMat, scale: SIMD3<Float>(1.0, 0.65, 1.0), label: "VelcroStrap"))
            cuffNode.add(uploadNode(MeshPrimitives.makeCapsule(radius: 0.0032, length: 0.014, capSegs: 4, radialSegs: 8), mat: orangePullMat, pos: SIMD3<Float>(side * 0.042, 0.014, -0.032), rot: SIMD3<Float>(0, side * 0.4, .pi * 0.5), label: "VelcroTip"))
            if side < 0 {
                let passZip = RigCatmullRomCurve([
                    SIMD3<Float>(-0.048, 0.085, -0.022),
                    SIMD3<Float>(-0.053, 0.130, -0.024),
                    SIMD3<Float>(-0.057, 0.175, -0.022),
                ])
                cuffNode.add(uploadNode(buildTubeGeometry(curve: passZip, tubularSegments: 10, radius: 0.0030, radialSegments: 6), mat: rubberTabMat, label: "PassZip"))
                cuffNode.add(uploadNode(MeshPrimitives.makeCapsule(radius: 0.0028, length: 0.012, capSegs: 4, radialSegs: 6), mat: orangePullMat, pos: SIMD3<Float>(-0.058, 0.165, -0.025), rot: SIMD3<Float>(0, 0, 0.25), label: "PassPull"))
            }
            bodyGroupNode.add(cuffNode)

            let handNode = RigNode()
            handNode.eulerOrder = .yxz
            handNode.add(buildFreerideGlove(side))
            let pole = buildPole(side)
            pole.position = SIMD3<Float>(-side * 0.012, 0.0, -0.024)
            handNode.add(pole)
            let wristAnchor = RigNode()
            wristAnchor.position = SIMD3<Float>(side * 0.004, -0.001, 0.098)
            handNode.add(wristAnchor)
            bodyGroupNode.add(handNode)

            return ArmRig(
                shoulderBall: shoulderBall,
                upperSeg: upperSeg,
                elbowBall: elbowBall,
                lowerSeg: lowerSeg,
                cuffNode: cuffNode,
                handNode: handNode,
                pole: pole,
                wristAnchor: wristAnchor
            )
        }
        self.armL = buildArm(-1)
        self.armR = buildArm(1)

        // Collect all foreground materials so bounce/rim/frost uniforms sync each frame
        self.allForegroundMaterials = [
            bindMat, gunmetalMat, steelMat, coralMetalMat, rubberMat,
            bootShellMat, bootTongueMat,
            leftSki.bladePart.material, rightSki.bladePart.material,
            pantsMat, corduraMat, seamMat, jacketMat, jacketDarkMat,
            buckleMat, orangePullMat, packMat, helmetMat, goggleMat,
            gloveTanMat, glovePalmMat, gloveShellMat, stitchMat, webbingMat,
            rubberTabMat, badgeMat,
            poleShaftMat, poleGripMat, poleCorkMat, poleBasketMat, poleAccentMat, poleMetalMat, poleStrapMat
        ]
        self.allMaterials = self.allForegroundMaterials + [
            leftSki.dustPart.material, rightSki.dustPart.material,
            dsMat, shadowOnlyMat
        ]
    }
}
