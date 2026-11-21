import Foundation
import Metal
import simd

// MARK: - Shared Deterministic Micro-Relief & Ridge Sharpening (matches Terrain.metal 1:1)

/// Deterministic wind-blown snow pillows, rolling terrain lips, sastrugi waves, and granite ribs
/// at 4m–38m scale, matching `microRelief()` in `Terrain.metal` 1:1.
@inlinable
public func microRelief(x: Float, z: Float, slopeMag: Float, powder: Float) -> Float {
    let w1 = sinf(x * 0.047 + z * 0.029) * cosf(x * 0.023 - z * 0.051)
    let w2 = sinf(x * 0.115 - z * 0.088 + 1.3) * cosf(x * 0.072 + z * 0.131 - 0.7)
    let w3 = sinf(x * 0.240 + z * 0.195 - 2.1) * sinf(x * 0.180 - z * 0.270 + 0.5)
    let roller = sinf((x * 0.74 + z * 0.67) * 0.082 + w1 * 1.4) * 0.68

    let midSlope = min(1.0, max(0.0, (slopeMag - 0.08) / 0.45))
    let steepRock = min(1.0, max(0.0, (slopeMag - 0.78) / 0.45))
    let snowAmp = (0.55 + 0.45 * powder) * (0.45 + 0.55 * midSlope) * (1.0 - steepRock * 0.45)
    let rockRib = steepRock * (abs(w2) * 1.65 + w3 * 0.55)
    return (w1 * 1.35 + roller * 0.85 + w2 * 0.42 + w3 * 0.14) * snowAmp + rockRib
}

/// Steep-face granite ridge sharpening matching `Terrain.metal` `evalCoreHeight()` 1:1.
@inlinable
public func ridgeSharpen(x: Float, z: Float, slopeMag: Float) -> Float {
    if slopeMag <= 0.72 { return 0.0 }
    let sharpM = min(1.0, max(0.0, (slopeMag - 0.72) / 0.55))
    let ridgeS = 1.0 - abs(sinf(x * 0.019 + z * 0.014) * cosf(x * 0.012 - z * 0.023))
    return (ridgeS - 0.5) * 4.5 * sharpM
}

// MARK: - Packed DEM Manifest Decoding

private struct DemPackedEntry: Decodable {
    let step: Int
    let res: String
    let ao: String
    let esc: [Int]
}

private struct DemPackedManifest: Decodable {
    let core_dem: DemPackedEntry
    let outer_dem: DemPackedEntry
}

private func decodeMedDem(
    resRGBA: [UInt8],
    aoRGBA: [UInt8],
    width: Int,
    height: Int,
    step: Int,
    escapes: [Int]
) -> [UInt8] {
    let n = width * height
    var hq = [Int32](repeating: 0, count: n)
    var escPtr = 0
    var nextEscIdx = escapes.isEmpty ? -1 : escapes[0]

    for y in 0..<height {
        let row = y * width
        for x in 0..<width {
            let i = row + x
            let pred: Int32
            if i == 0 {
                pred = 0
            } else if y == 0 {
                pred = hq[i - 1]
            } else if x == 0 {
                pred = hq[i - width]
            } else {
                let left = hq[i - 1]
                let up = hq[i - width]
                let ul = hq[i - width - 1]
                let mn = left < up ? left : up
                let mx = left > up ? left : up
                if ul >= mx {
                    pred = mn
                } else if ul <= mn {
                    pred = mx
                } else {
                    pred = left + up - ul
                }
            }

            let val: Int32
            if i == nextEscIdx {
                val = Int32(escapes[escPtr + 1])
                escPtr += 2
                nextEscIdx = escPtr < escapes.count ? escapes[escPtr] : -1
            } else {
                val = Int32(resRGBA[i * 4]) - 128
            }
            hq[i] = pred + val
        }
    }

    var out = [UInt8](repeating: 255, count: n * 4)
    let step32 = Int32(step)
    for i in 0..<n {
        var q = hq[i] * step32
        if q < 0 { q = 0 }
        else if q > 65535 { q = 65535 }
        let k = i * 4
        out[k] = UInt8((q >> 8) & 0xFF)
        out[k + 1] = UInt8(q & 0xFF)
        out[k + 2] = aoRGBA[k]
        out[k + 3] = 255
    }
    return out
}

// MARK: - ChamonixTerrainData

/// Loads baked Chamonix / Mont Blanc EU-DEM textures and exposes both GPU `MTLTexture`s
/// and exact CPU bilinear height/normal/surface sampling matching `Terrain.metal` 1:1.
public final class ChamonixTerrainData: TerrainQueryProtocol {
    public let meta: ChamonixMeta
    public let coreSize: Int
    public let outerSize: Int
    public let textures: TerrainTextures

    private let coreElev: [Float]
    private let coreAO: [Float]
    private let corePowder: [Float]
    private let coreCurv: [Float]
    private let outerElev: [Float]

    public init(device: MTLDevice, commandQueue: MTLCommandQueue, assetSearchPaths: [URL] = []) throws {
        let metaURL = try TextureLoader.assetURL(for: "chamonix_meta.json", searchPaths: assetSearchPaths)
        let packedURL = try TextureLoader.assetURL(for: "dem_packed.json", searchPaths: assetSearchPaths)
        let metaData = try Data(contentsOf: metaURL)
        let packedData = try Data(contentsOf: packedURL)

        self.meta = try JSONDecoder().decode(ChamonixMeta.self, from: metaData)
        let demPacked = try JSONDecoder().decode(DemPackedManifest.self, from: packedData)

        let coreResImg = try TextureLoader.loadRawRGBA8(url: TextureLoader.assetURL(for: demPacked.core_dem.res, searchPaths: assetSearchPaths))
        let coreAoImg = try TextureLoader.loadRawRGBA8(url: TextureLoader.assetURL(for: demPacked.core_dem.ao, searchPaths: assetSearchPaths))
        let coreAux1Img = try TextureLoader.loadRawRGBA8(url: TextureLoader.assetURL(for: "core_aux1.webp", searchPaths: assetSearchPaths))
        let coreAux2Img = try TextureLoader.loadRawRGBA8(url: TextureLoader.assetURL(for: "core_aux2.webp", searchPaths: assetSearchPaths))

        let outerResImg = try TextureLoader.loadRawRGBA8(url: TextureLoader.assetURL(for: demPacked.outer_dem.res, searchPaths: assetSearchPaths))
        let outerAoImg = try TextureLoader.loadRawRGBA8(url: TextureLoader.assetURL(for: demPacked.outer_dem.ao, searchPaths: assetSearchPaths))
        let outerAux1Img = try TextureLoader.loadRawRGBA8(url: TextureLoader.assetURL(for: "outer_aux1.webp", searchPaths: assetSearchPaths))
        let outerAux2Img = try TextureLoader.loadRawRGBA8(url: TextureLoader.assetURL(for: "outer_aux2.webp", searchPaths: assetSearchPaths))

        let coreDemData = decodeMedDem(
            resRGBA: coreResImg.data,
            aoRGBA: coreAoImg.data,
            width: coreResImg.width,
            height: coreResImg.height,
            step: demPacked.core_dem.step,
            escapes: demPacked.core_dem.esc
        )
        let outerDemData = decodeMedDem(
            resRGBA: outerResImg.data,
            aoRGBA: outerAoImg.data,
            width: outerResImg.width,
            height: outerResImg.height,
            step: demPacked.outer_dem.step,
            escapes: demPacked.outer_dem.esc
        )

        self.coreSize = coreResImg.width
        self.outerSize = outerResImg.width

        let maxElev = ChamonixConstants.maxElev
        let nCount = coreSize * coreSize
        var elevArr = [Float](repeating: 0, count: nCount)
        var aoArr = [Float](repeating: 0, count: nCount)
        var powArr = [Float](repeating: 0, count: nCount)
        var curvArr = [Float](repeating: 0, count: nCount)

        for i in 0..<nCount {
            let k = i * 4
            let hi = Int(coreDemData[k])
            let lo = Int(coreDemData[k + 1])
            elevArr[i] = (Float(hi * 256 + lo) / 65535.0) * maxElev
            aoArr[i] = Float(coreDemData[k + 2]) / 255.0
            powArr[i] = Float(coreAux1Img.data[k]) / 255.0
            curvArr[i] = Float(coreAux2Img.data[k + 2]) / 255.0
        }
        self.coreElev = elevArr
        self.coreAO = aoArr
        self.corePowder = powArr
        self.coreCurv = curvArr

        let mCount = outerSize * outerSize
        var outerArr = [Float](repeating: 0, count: mCount)
        for i in 0..<mCount {
            let k = i * 4
            let hi = Int(outerDemData[k])
            let lo = Int(outerDemData[k + 1])
            outerArr[i] = (Float(hi * 256 + lo) / 65535.0) * maxElev
        }
        self.outerElev = outerArr

        let coreDemTex = try TextureLoader.makeDataTexture(device: device, data: coreDemData, width: coreSize, height: coreSize, pixelFormat: .rgba8Unorm, label: "coreDem")
        let coreAux1Tex = try TextureLoader.makeDataTexture(device: device, data: coreAux1Img.data, width: coreAux1Img.width, height: coreAux1Img.height, pixelFormat: .rgba8Unorm, label: "coreAux1")
        let coreAux2Tex = try TextureLoader.makeDataTexture(device: device, data: coreAux2Img.data, width: coreAux2Img.width, height: coreAux2Img.height, pixelFormat: .rgba8Unorm, label: "coreAux2")

        let outerDemTex = try TextureLoader.makeDataTexture(device: device, data: outerDemData, width: outerSize, height: outerSize, pixelFormat: .rgba8Unorm, label: "outerDem")
        let outerAux1Tex = try TextureLoader.makeDataTexture(device: device, data: outerAux1Img.data, width: outerAux1Img.width, height: outerAux1Img.height, pixelFormat: .rgba8Unorm, label: "outerAux1")
        let outerAux2Tex = try TextureLoader.makeDataTexture(device: device, data: outerAux2Img.data, width: outerAux2Img.width, height: outerAux2Img.height, pixelFormat: .rgba8Unorm, label: "outerAux2")

        let graniteAlbedo = try TextureLoader.loadMTLTexture(device: device, commandQueue: commandQueue, url: TextureLoader.assetURL(for: "granite_albedo.webp", searchPaths: assetSearchPaths), srgb: true, mipmaps: true)
        let graniteNormal = try TextureLoader.loadMTLTexture(device: device, commandQueue: commandQueue, url: TextureLoader.assetURL(for: "granite_normal.webp", searchPaths: assetSearchPaths), srgb: false, mipmaps: true)
        let snowNormal = try TextureLoader.loadMTLTexture(device: device, commandQueue: commandQueue, url: TextureLoader.assetURL(for: "snow_normal.webp", searchPaths: assetSearchPaths), srgb: false, mipmaps: true)
        let snowCavity = try TextureLoader.loadMTLTexture(device: device, commandQueue: commandQueue, url: TextureLoader.assetURL(for: "snow_cavity.webp", searchPaths: assetSearchPaths), srgb: false, mipmaps: true)
        let iceNormal = try TextureLoader.loadMTLTexture(device: device, commandQueue: commandQueue, url: TextureLoader.assetURL(for: "ice_normal.webp", searchPaths: assetSearchPaths), srgb: false, mipmaps: true)
        let powderPuff = try TextureLoader.loadMTLTexture(device: device, commandQueue: commandQueue, url: TextureLoader.assetURL(for: "powder_puff.webp", searchPaths: assetSearchPaths), srgb: false, mipmaps: true)
        let spruceBough = try TextureLoader.loadMTLTexture(device: device, commandQueue: commandQueue, url: TextureLoader.assetURL(for: "spruce_bough.webp", searchPaths: assetSearchPaths), srgb: true, mipmaps: true)
        let skiTopsheet = try TextureLoader.loadMTLTexture(device: device, commandQueue: commandQueue, url: TextureLoader.assetURL(for: "ski_topsheet.webp", searchPaths: assetSearchPaths), srgb: true, mipmaps: true)

        self.textures = TerrainTextures(
            coreDem: coreDemTex,
            coreAux1: coreAux1Tex,
            coreAux2: coreAux2Tex,
            outerDem: outerDemTex,
            outerAux1: outerAux1Tex,
            outerAux2: outerAux2Tex,
            graniteAlbedo: graniteAlbedo,
            graniteNormal: graniteNormal,
            snowNormal: snowNormal,
            snowCavity: snowCavity,
            iceNormal: iceNormal,
            powderPuff: powderPuff,
            spruceBough: spruceBough,
            skiTopsheet: skiTopsheet
        )
    }

    public func sampleRawDEM(x: Float, z: Float) -> TerrainRawSample {
        let n = coreSize
        let coreHalf = ChamonixConstants.coreHalf
        let u = ((x + coreHalf) / (2.0 * coreHalf)) * Float(n - 1)
        let v = ((z + coreHalf) / (2.0 * coreHalf)) * Float(n - 1)
        if u >= 0 && u <= Float(n - 1) && v >= 0 && v <= Float(n - 1) {
            let ix = min(n - 2, max(0, Int(floorf(u))))
            let iz = min(n - 2, max(0, Int(floorf(v))))
            let fx = u - Float(ix)
            let fz = v - Float(iz)
            let r0 = iz * n + ix
            let r1 = (iz + 1) * n + ix

            let h00 = coreElev[r0], h10 = coreElev[r0 + 1]
            let h01 = coreElev[r1], h11 = coreElev[r1 + 1]
            let elev = (h00 * (1.0 - fx) + h10 * fx) * (1.0 - fz) + (h01 * (1.0 - fx) + h11 * fx) * fz

            let ao = (coreAO[r0] * (1.0 - fx) + coreAO[r0 + 1] * fx) * (1.0 - fz)
                   + (coreAO[r1] * (1.0 - fx) + coreAO[r1 + 1] * fx) * fz
            let powder = (corePowder[r0] * (1.0 - fx) + corePowder[r0 + 1] * fx) * (1.0 - fz)
                       + (corePowder[r1] * (1.0 - fx) + corePowder[r1 + 1] * fx) * fz
            let curv = (coreCurv[r0] * (1.0 - fx) + coreCurv[r0 + 1] * fx) * (1.0 - fz)
                     + (coreCurv[r1] * (1.0 - fx) + coreCurv[r1 + 1] * fx) * fz
            return TerrainRawSample(elev: elev, ao: ao, powder: powder, curv: curv, inCore: true)
        }

        let m = outerSize
        let outerHalf = ChamonixConstants.outerHalf
        let ou = min(Float(m) - 1.001, max(0.0, ((x + outerHalf) / (2.0 * outerHalf)) * Float(m - 1)))
        let ov = min(Float(m) - 1.001, max(0.0, ((z + outerHalf) / (2.0 * outerHalf)) * Float(m - 1)))
        let ix = Int(floorf(ou))
        let iz = Int(floorf(ov))
        let fx = ou - Float(ix)
        let fz = ov - Float(iz)
        let r0 = iz * m + ix
        let r1 = (iz + 1) * m + ix
        let elev = (outerElev[r0] * (1.0 - fx) + outerElev[r0 + 1] * fx) * (1.0 - fz)
                 + (outerElev[r1] * (1.0 - fx) + outerElev[r1 + 1] * fx) * fz
        return TerrainRawSample(elev: elev, ao: 0.7, powder: 0.5, curv: 0.5, inCore: false)
    }

    public func sampleHeight(x: Float, z: Float) -> Float {
        let s = sampleRawDEM(x: x, z: z)
        if !s.inCore { return s.elev }
        let e: Float = 8.0
        let gx = (sampleRawDEM(x: x + e, z: z).elev - sampleRawDEM(x: x - e, z: z).elev) / (2.0 * e)
        let gz = (sampleRawDEM(x: x, z: z + e).elev - sampleRawDEM(x: x, z: z - e).elev) / (2.0 * e)
        let slopeMag = hypotf(gx, gz)
        return s.elev + microRelief(x: x, z: z, slopeMag: slopeMag, powder: s.powder) + ridgeSharpen(x: x, z: z, slopeMag: slopeMag)
    }

    public func sampleTerrain(x: Float, z: Float) -> TerrainSample {
        let raw = sampleRawDEM(x: x, z: z)
        let eRaw: Float = 8.0
        let rgx = (sampleRawDEM(x: x + eRaw, z: z).elev - sampleRawDEM(x: x - eRaw, z: z).elev) / (2.0 * eRaw)
        let rgz = (sampleRawDEM(x: x, z: z + eRaw).elev - sampleRawDEM(x: x, z: z - eRaw).elev) / (2.0 * eRaw)
        let rawSlope = hypotf(rgx, rgz)

        let h = raw.elev + (raw.inCore ? (microRelief(x: x, z: z, slopeMag: rawSlope, powder: raw.powder) + ridgeSharpen(x: x, z: z, slopeMag: rawSlope)) : 0.0)
        let e: Float = 1.1
        let hL = sampleHeight(x: x - e, z: z)
        let hR = sampleHeight(x: x + e, z: z)
        let hN = sampleHeight(x: x, z: z - e)
        let hS = sampleHeight(x: x, z: z + e)
        let gx = (hR - hL) / (2.0 * e)
        let gz = (hS - hN) / (2.0 * e)
        let inv = 1.0 / sqrtf(gx * gx + 1.0 + gz * gz)
        let normal = SIMD3<Float>(-gx * inv, inv, -gz * inv)
        let slopeRad = acosf(min(1.0, max(0.0, normal.y)))
        let slopeDeg = slopeRad * (180.0 / .pi)

        let surf: SurfaceType
        let southFacing = normal.z > 0.25
        if slopeDeg > 48.0 || (slopeDeg > 43.0 && raw.curv > 0.62) {
            surf = .granite
        } else if slopeDeg > 37.5 && raw.curv > 0.58 && h > 2350.0 {
            surf = .ice
        } else if raw.curv > 0.63 && raw.powder < 0.42 {
            surf = .crust
        } else if h < 1850.0 && southFacing {
            surf = .corn
        } else if raw.powder > 0.48 && h > 1950.0 {
            surf = .powder
        } else {
            surf = .packed
        }

        return TerrainSample(
            y: h,
            rawY: raw.elev,
            normal: normal,
            gx: gx,
            gz: gz,
            slopeRad: slopeRad,
            slopeDeg: slopeDeg,
            ao: raw.ao,
            powder: raw.powder,
            curv: raw.curv,
            surface: surf
        )
    }
}
