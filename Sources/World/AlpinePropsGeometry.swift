import Foundation
import Metal
import simd

public struct PropVertex {
    public var position: SIMD4<Float>
    public var normal: SIMD4<Float>
    public var prop: SIMD4<Float>

    public init(position: SIMD3<Float>, normal: SIMD3<Float> = .zero, prop: SIMD3<Float> = .zero) {
        self.position = SIMD4<Float>(position, 1.0)
        self.normal = SIMD4<Float>(normal, 0.0)
        self.prop = SIMD4<Float>(prop, 0.0)
    }
}

public struct PropMesh {
    public let vertexBuffer: MTLBuffer
    public let indexBuffer: MTLBuffer?
    public let vertexCount: Int
    public let indexCount: Int

    public static func vertexDescriptor() -> MTLVertexDescriptor {
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float4
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = BufferIndex.vertices.rawValue

        vd.attributes[1].format = .float4
        vd.attributes[1].offset = MemoryLayout<SIMD4<Float>>.stride
        vd.attributes[1].bufferIndex = BufferIndex.vertices.rawValue

        vd.attributes[2].format = .float4
        vd.attributes[2].offset = MemoryLayout<SIMD4<Float>>.stride * 2
        vd.attributes[2].bufferIndex = BufferIndex.vertices.rawValue

        vd.layouts[BufferIndex.vertices.rawValue].stride = MemoryLayout<PropVertex>.stride
        vd.layouts[BufferIndex.vertices.rawValue].stepFunction = .perVertex
        vd.layouts[BufferIndex.vertices.rawValue].stepRate = 1
        return vd
    }

    public static func upload(
        device: MTLDevice,
        vertices: [PropVertex],
        indices: [UInt32]? = nil,
        label: String? = nil
    ) -> PropMesh? {
        guard !vertices.isEmpty else { return nil }
        let vLen = vertices.count * MemoryLayout<PropVertex>.stride
        guard let vb = vertices.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: vLen, options: .storageModeShared)
        }) else { return nil }
        vb.label = label.map { "\($0)_VB" }

        var ib: MTLBuffer? = nil
        var idxCount = 0
        if let indices, !indices.isEmpty {
            let iLen = indices.count * MemoryLayout<UInt32>.stride
            ib = indices.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: iLen, options: .storageModeShared)
            }
            ib?.label = label.map { "\($0)_IB" }
            idxCount = indices.count
        }
        return PropMesh(vertexBuffer: vb, indexBuffer: ib, vertexCount: vertices.count, indexCount: idxCount)
    }
}

public enum SpruceStyle {
    case close
    case tall
    case broad
    case far
}

public enum AlpinePropsGeometry {

    // MARK: - Normal Helpers

    public static func computeIndexedNormals(positions: [SIMD3<Float>], indices: [UInt32]) -> [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
        var t = 0
        while t + 2 < indices.count {
            let i0 = Int(indices[t])
            let i1 = Int(indices[t + 1])
            let i2 = Int(indices[t + 2])
            let p0 = positions[i0]
            let p1 = positions[i1]
            let p2 = positions[i2]
            let fn = simd_cross(p1 - p0, p2 - p0)
            normals[i0] += fn
            normals[i1] += fn
            normals[i2] += fn
            t += 3
        }
        for i in normals.indices {
            let l = simd_length(normals[i])
            normals[i] = l > 1e-8 ? (normals[i] / l) : SIMD3<Float>(0, 1, 0)
        }
        return normals
    }

    public static func computeWeldedSmoothNormals(positions: [SIMD3<Float>], indices: [UInt32]) -> [SIMD3<Float>] {
        var normals = computeIndexedNormals(positions: positions, indices: indices)
        struct BucketKey: Hashable {
            let x: Int
            let y: Int
            let z: Int
        }
        var buckets: [BucketKey: (sum: SIMD3<Float>, indices: [Int])] = [:]
        buckets.reserveCapacity(positions.count)
        for i in positions.indices {
            let p = positions[i]
            let key = BucketKey(
                x: Int((p.x * 500.0).rounded()),
                y: Int((p.y * 500.0).rounded()),
                z: Int((p.z * 500.0).rounded())
            )
            if var acc = buckets[key] {
                acc.sum += normals[i]
                acc.indices.append(i)
                buckets[key] = acc
            } else {
                buckets[key] = (normals[i], [i])
            }
        }
        for (_, acc) in buckets {
            let len = simd_length(acc.sum)
            let n = len > 1e-8 ? (acc.sum / len) : SIMD3<Float>(0, 1, 0)
            for idx in acc.indices {
                normals[idx] = n
            }
        }
        return normals
    }

    // MARK: - Subdivided Box (6 faces, distinct seam vertices for crisp 90-degree edges)

    private static func makeSubdividedBox(
        width: Float, height: Float, depth: Float,
        segX: Int, segY: Int, segZ: Int
    ) -> (positions: [SIMD3<Float>], isTopFace: [Bool], indices: [UInt32]) {
        var positions: [SIMD3<Float>] = []
        var isTop: [Bool] = []
        var indices: [UInt32] = []

        func buildPlane(
            uAxis: Int, vAxis: Int, wAxis: Int,
            uDir: Float, vDir: Float,
            uSize: Float, vSize: Float, wOffset: Float,
            gridX: Int, gridY: Int,
            topFlag: Bool
        ) {
            let baseIdx = UInt32(positions.count)
            let halfU = uSize * 0.5
            let halfV = vSize * 0.5
            for iy in 0...gridY {
                let fy = Float(iy) / Float(gridY)
                let v = (fy * vSize - halfV) * vDir
                for ix in 0...gridX {
                    let fx = Float(ix) / Float(gridX)
                    let u = (fx * uSize - halfU) * uDir
                    var p = SIMD3<Float>.zero
                    p[uAxis] = u
                    p[vAxis] = v
                    p[wAxis] = wOffset
                    positions.append(p)
                    isTop.append(topFlag)
                }
            }
            let row = gridX + 1
            for iy in 0..<gridY {
                for ix in 0..<gridX {
                    let a = baseIdx + UInt32(ix + row * iy)
                    let b = baseIdx + UInt32(ix + row * (iy + 1))
                    let c = baseIdx + UInt32((ix + 1) + row * (iy + 1))
                    let d = baseIdx + UInt32((ix + 1) + row * iy)
                    indices.append(contentsOf: [a, b, d, b, c, d])
                }
            }
        }

        // +X, -X
        buildPlane(uAxis: 2, vAxis: 1, wAxis: 0, uDir: -1, vDir: -1, uSize: depth, vSize: height, wOffset:  width * 0.5, gridX: segZ, gridY: segY, topFlag: false)
        buildPlane(uAxis: 2, vAxis: 1, wAxis: 0, uDir:  1, vDir: -1, uSize: depth, vSize: height, wOffset: -width * 0.5, gridX: segZ, gridY: segY, topFlag: false)
        // +Y (top), -Y (bottom)
        buildPlane(uAxis: 0, vAxis: 2, wAxis: 1, uDir:  1, vDir:  1, uSize: width, vSize: depth, wOffset:  height * 0.5, gridX: segX, gridY: segZ, topFlag: true)
        buildPlane(uAxis: 0, vAxis: 2, wAxis: 1, uDir:  1, vDir: -1, uSize: width, vSize: depth, wOffset: -height * 0.5, gridX: segX, gridY: segZ, topFlag: false)
        // +Z, -Z
        buildPlane(uAxis: 0, vAxis: 1, wAxis: 2, uDir:  1, vDir: -1, uSize: width, vSize: height, wOffset:  depth * 0.5, gridX: segX, gridY: segY, topFlag: false)
        buildPlane(uAxis: 0, vAxis: 1, wAxis: 2, uDir: -1, vDir: -1, uSize: width, vSize: height, wOffset: -depth * 0.5, gridX: segX, gridY: segY, topFlag: false)

        return (positions, isTop, indices)
    }

    // MARK: - 1. Stratified Protogine Granite Piton (Aiguille du Midi)

    public static func buildStratifiedPitonMesh(
        device: MTLDevice,
        radius: Float,
        height: Float,
        seed: Float = 11,
        label: String = "StratifiedPiton"
    ) -> PropMesh? {
        let (stdVerts, indices) = MeshPrimitives.makeCylinder(
            radiusTop: radius * 0.48,
            radiusBottom: radius * 1.08,
            height: height,
            radialSegs: 36,
            heightSegs: 18,
            openEnded: false
        )
        var positions = stdVerts.map { $0.position }
        for i in positions.indices {
            let x = positions[i].x
            let y = positions[i].y
            let z = positions[i].z
            let r = hypotf(x, z)
            let yn = clamp((y + height * 0.5) / max(0.01, height), 0.0, 1.0)
            if r > 0.01 {
                let ang = atan2f(z, x)
                let ridge = 1.0
                    + 0.20 * sinf(ang * 3.0 + seed * 0.7) * cosf(yn * 4.2)
                    + 0.11 * sinf(ang * 5.0 - yn * 6.5 + seed)
                    + 0.06 * cosf(ang * 8.0 + yn * 9.0 - seed * 0.4)
                let strata = 1.0 + 0.055 * sinf(yn * 18.0 + cosf(ang * 2.0 + seed) * 1.4)
                let crag = 1.0 + 0.035 * sinf(ang * 11.0 + yn * 14.0)
                let scale = ridge * strata * crag
                let yCrag = (sinf(ang * 4.0 + yn * 10.0 + seed) * 0.9 + cosf(ang * 7.0 - yn * 8.0) * 0.45) * (1.0 - yn * 0.65)
                positions[i] = SIMD3<Float>(x * scale, y + yCrag, z * scale)
            }
        }
        let normals = computeWeldedSmoothNormals(positions: positions, indices: indices)
        var outVerts: [PropVertex] = []
        outVerts.reserveCapacity(positions.count)
        let seedMod = (seed * 0.17).truncatingRemainder(dividingBy: 1.0)
        for i in positions.indices {
            let p = positions[i]
            let n = normals[i]
            let yn = clamp((p.y + height * 0.5) / max(0.01, height), 0.0, 1.0)
            let snow = smoothstep(0.48, 0.78, n.y)
            let expose = smoothstep(0.03, 0.16, yn) * (1.0 - snow * 0.48)
            let ao = clamp(0.56 + yn * 0.38 + n.y * 0.14, 0.42, 1.0)
            outVerts.append(PropVertex(position: p, normal: n, prop: SIMD3<Float>(expose, ao, seedMod)))
        }
        return PropMesh.upload(device: device, vertices: outVerts, indices: indices, label: label)
    }

    // MARK: - 2. Volumetric 3D Snow-Laden Norway Spruce

    public static func buildVolumetricSpruceMesh(
        device: MTLDevice,
        seed: UInt32,
        style: SpruceStyle,
        label: String
    ) -> PropMesh? {
        var rand = Mulberry32(seed: seed)
        var vertices: [PropVertex] = []
        vertices.reserveCapacity(style == .far ? 2800 : (style == .close ? 9500 : 6200))

        let totalH: Float = (style == .broad) ? 11.8 : ((style == .far) ? 13.4 : 14.4)
        let yBuried: Float = -2.4

        func pushTri(
            _ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>,
            _ matType: Float, _ aoA: Float, _ aoB: Float, _ aoC: Float,
            _ radialBlend: Float = 0.25
        ) {
            let u = b - a
            let v = c - a
            var fn = simd_cross(u, v)
            let fl = simd_length(fn)
            if fl > 1e-8 { fn /= fl } else { fn = SIMD3<Float>(0, 1, 0) }

            let pts = [(a, aoA), (b, aoB), (c, aoC)]
            for (p, ao) in pts {
                let rl = max(1e-5, simd_length(SIMD3<Float>(p.x, max(0.4, p.y * 0.25), p.z)))
                let rnx = p.x / rl
                let rny: Float = matType > 1.5 ? 0.82 : (p.y > 0 ? 0.35 : -0.2)
                let rnz = p.z / rl
                var n = fn * (1.0 - radialBlend) + SIMD3<Float>(rnx, rny, rnz) * radialBlend
                let nl = simd_length(n)
                n = nl > 1e-8 ? (n / nl) : SIMD3<Float>(0, 1, 0)
                let hNorm = clamp(p.y / totalH, 0.0, 1.0)
                vertices.append(PropVertex(position: p, normal: n, prop: SIMD3<Float>(matType, ao, hNorm)))
            }
        }

        func pushQuad(
            _ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>,
            _ matType: Float, _ ao0: Float, _ ao1: Float, _ ao2: Float, _ ao3: Float,
            _ rb: Float = 0.25
        ) {
            pushTri(p0, p1, p2, matType, ao0, ao1, ao2, rb)
            pushTri(p0, p2, p3, matType, ao0, ao2, ao3, rb)
        }

        // 1. Dark bark trunk
        let trunkSegs = style == .far ? 3 : 5
        let trunkSides = 6
        let crookX = (rand.next() - 0.5) * 0.44
        let crookZ = (rand.next() - 0.5) * 0.44
        var trunkRings: [(ring: [SIMD3<Float>], t: Float)] = []
        for s in 0...trunkSegs {
            let t = Float(s) / Float(trunkSegs)
            let y = yBuried + (totalH * 0.96 - yBuried) * t
            let baseRad: Float = style == .broad ? 0.52 : 0.44
            let rad = baseRad * powf(max(0.0, 1.0 - t * 0.88), 1.15) * (t < 0.18 ? 1.25 : 1.0)
            let ox = sinf(t * .pi) * crookX
            let oz = sinf(t * .pi) * crookZ
            var ring: [SIMD3<Float>] = []
            for k in 0..<trunkSides {
                let a = (Float(k) / Float(trunkSides)) * .pi * 2.0
                ring.append(SIMD3<Float>(ox + cosf(a) * rad, y, oz + sinf(a) * rad))
            }
            trunkRings.append((ring, t))
        }
        for s in 0..<trunkSegs {
            let r0 = trunkRings[s].ring
            let r1 = trunkRings[s + 1].ring
            let ao0 = 0.30 + trunkRings[s].t * 0.25
            let ao1 = 0.30 + trunkRings[s + 1].t * 0.25
            for k in 0..<trunkSides {
                let kn = (k + 1) % trunkSides
                pushQuad(r0[k], r1[k], r1[kn], r0[kn], 0.0, ao0, ao1, ao1, ao0, 0.35)
            }
        }

        // Helper: 3D curved drooping bough + snow pillow
        func emitBoughFrond(
            ox: Float, oy: Float, oz: Float,
            az: Float, len: Float, halfW: Float, droop: Float, thick: Float, snowH: Float,
            isSubTwig: Bool
        ) {
            let dx = cosf(az), dz = sinf(az)
            let sx = -sinf(az), sz = cosf(az)

            func evalSpine(_ u: Float) -> SIMD3<Float> {
                let r = len * u
                let dropY = -droop * powf(max(0.0, u), 1.38) + powf(max(0.0, u), 2.8) * (droop * 0.22)
                let jitter: Float = (u > 0.1 && u < 0.95) ? (rand.next() - 0.5) * 0.09 : 0.0
                return SIMD3<Float>(ox + dx * r, oy + dropY + jitter, oz + dz * r)
            }

            let p0 = evalSpine(0.0)
            let p1 = evalSpine(0.46)
            let p2 = evalSpine(1.0)

            let w0 = halfW * 0.34
            let w1 = halfW * (isSubTwig ? 0.92 : 1.0)
            let t0 = thick * 0.95
            let t1 = thick * 0.72

            let bT = SIMD3<Float>(p0.x, p0.y + t0 * 0.45, p0.z)
            let bL = SIMD3<Float>(p0.x + sx * w0, p0.y - t0 * 0.18, p0.z + sz * w0)
            let bB = SIMD3<Float>(p0.x, p0.y - t0 * 0.65, p0.z)
            let bR = SIMD3<Float>(p0.x - sx * w0, p0.y - t0 * 0.18, p0.z - sz * w0)

            let mT = SIMD3<Float>(p1.x, p1.y + t1 * 0.42, p1.z)
            let mL = SIMD3<Float>(p1.x + sx * w1, p1.y - t1 * 0.28, p1.z + sz * w1)
            let mB = SIMD3<Float>(p1.x, p1.y - t1 * 0.78, p1.z)
            let mR = SIMD3<Float>(p1.x - sx * w1, p1.y - t1 * 0.28, p1.z - sz * w1)

            let tip = SIMD3<Float>(p2.x, p2.y - t1 * 0.12, p2.z)

            let baseAO: Float = isSubTwig ? 0.52 : 0.22
            let midAO: Float = isSubTwig ? 0.78 : 0.72
            let tipAO: Float = 0.98

            pushQuad(bT, bL, mL, mT, 1.0, baseAO * 1.1, baseAO * 0.85, midAO * 0.92, midAO * 1.05, 0.22)
            pushQuad(bT, mT, mR, bR, 1.0, baseAO * 1.1, midAO * 1.05, midAO * 0.92, baseAO * 0.85, 0.22)
            pushQuad(bL, bB, mB, mL, 1.0, baseAO * 0.65, baseAO * 0.40, midAO * 0.44, midAO * 0.74, 0.18)
            pushQuad(bB, bR, mR, mB, 1.0, baseAO * 0.40, baseAO * 0.65, midAO * 0.74, midAO * 0.44, 0.18)

            pushTri(mT, mL, tip, 1.0, midAO * 1.05, midAO * 0.92, tipAO, 0.25)
            pushTri(mT, tip, mR, 1.0, midAO * 1.05, tipAO, midAO * 0.92, 0.25)
            pushTri(mL, mB, tip, 1.0, midAO * 0.78, midAO * 0.46, tipAO * 0.82, 0.20)
            pushTri(mB, mR, tip, 1.0, midAO * 0.46, midAO * 0.78, tipAO * 0.82, 0.20)

            if snowH > 0.04 {
                let s0 = evalSpine(0.10)
                let sm = evalSpine(0.42)
                let s2 = evalSpine(0.76)
                let sw0 = halfW * 0.28
                let swm = halfW * 0.66

                let sbT = SIMD3<Float>(s0.x, s0.y + t0 * 0.34 + snowH * 0.65, s0.z)
                let sbL = SIMD3<Float>(s0.x + sx * sw0, s0.y + t0 * 0.12, s0.z + sz * sw0)
                let sbR = SIMD3<Float>(s0.x - sx * sw0, s0.y + t0 * 0.12, s0.z - sz * sw0)

                let smT = SIMD3<Float>(sm.x, sm.y + t1 * 0.34 + snowH * 0.88, sm.z)
                let smL = SIMD3<Float>(sm.x + sx * swm, sm.y + t1 * 0.08, sm.z + sz * swm)
                let smR = SIMD3<Float>(sm.x - sx * swm, sm.y + t1 * 0.08, sm.z - sz * swm)

                let sTip = SIMD3<Float>(s2.x, s2.y + t1 * 0.18 + snowH * 0.28, s2.z)
                let sTipUnder = SIMD3<Float>(sTip.x, sTip.y - snowH * 0.24, sTip.z)

                pushQuad(sbT, sbL, smL, smT, 2.0, 0.78, 0.62, 0.82, 0.98, 0.48)
                pushQuad(sbT, smT, smR, sbR, 2.0, 0.78, 0.98, 0.82, 0.62, 0.48)
                pushTri(smT, smL, sTip, 2.0, 0.98, 0.82, 0.94, 0.48)
                pushTri(smT, sTip, smR, 2.0, 0.98, 0.94, 0.82, 0.48)
                pushTri(smL, sTipUnder, sTip, 2.0, 0.65, 0.72, 0.90, 0.35)
                pushTri(smR, sTip, sTipUnder, 2.0, 0.65, 0.90, 0.72, 0.35)
            }
        }

        // 2. Golden-angle whorls
        let numWhorls = style == .far ? 8 : 12
        let maxRad: Float = style == .broad ? 4.35 : (style == .far ? 3.85 : 3.95)
        let startY: Float = style == .broad ? 0.55 : 0.85
        let endY = totalH - 0.55
        var goldenAz = rand.next() * .pi * 2.0

        for w in 0..<numWhorls {
            let wt = Float(w) / Float(max(1, numWhorls - 1))
            let baseH = startY + (endY - startY) * powf(wt, 0.92)
            let trunkFrac = (baseH - yBuried) / (totalH - yBuried)
            let trunkOx = sinf(trunkFrac * .pi) * crookX
            let trunkOz = sinf(trunkFrac * .pi) * crookZ

            let envExp: Float = style == .broad ? 0.82 : 0.94
            let env = powf(max(0.0, 1.0 - wt * 0.82), envExp)
            let branchCount: Int
            if style == .far {
                branchCount = max(4, Int((6.0 - wt * 2.2).rounded()))
            } else {
                let baseCount: Float = style == .broad ? 7.5 : 7.0
                branchCount = max(4, Int((baseCount - wt * 2.8).rounded()))
            }

            goldenAz += 2.399963

            for b in 0..<branchCount {
                if wt > 0.22 && wt < 0.86 && branchCount >= 5 && rand.next() < 0.14 { continue }
                let az = goldenAz + (Float(b) / Float(branchCount)) * .pi * 2.0 + (rand.next() - 0.5) * (0.58 / Float(branchCount) * 4.5)
                let len = maxRad * env * (0.64 + rand.next() * 0.72)
                let yOff = (rand.next() - 0.5) * 0.68 * (1.0 - wt * 0.35)
                let by = baseH + yOff
                let halfW = (0.46 + len * 0.27) * (0.78 + rand.next() * 0.42)
                let droop = len * (0.36 + (1.0 - wt) * 0.19) * (0.74 + rand.next() * 0.46)
                let thick = (0.28 + len * 0.12) * (0.82 + rand.next() * 0.34)
                let snowChance = rand.next()
                let snowH: Float = snowChance > 0.34
                    ? (0.14 + env * 0.26 + rand.next() * 0.14) * (style == .broad ? 1.15 : 1.0)
                    : 0.0

                emitBoughFrond(ox: trunkOx, oy: by, oz: trunkOz, az: az, len: len, halfW: halfW, droop: droop, thick: thick, snowH: snowH, isSubTwig: false)

                if style != .far && len > 1.05 {
                    let dx = cosf(az), dz = sinf(az)
                    let startR = len * (0.32 + rand.next() * 0.16)
                    let subY = by - droop * powf(startR / len, 1.38) + 0.03
                    let subOx = trunkOx + dx * startR
                    let subOz = trunkOz + dz * startR
                    for side: Float in [-1.0, 1.0] {
                        if wt > 0.72 && rand.next() < 0.40 { continue }
                        let subAz = az + side * (0.50 + rand.next() * 0.26)
                        let subLen = len * (0.44 + rand.next() * 0.18)
                        let subW = halfW * 0.56
                        let subDroop = subLen * 0.40
                        let subSnow = rand.next() > 0.56 ? snowH * 0.48 : 0.0
                        emitBoughFrond(ox: subOx, oy: subY, oz: subOz, az: subAz, len: subLen, halfW: subW, droop: subDroop, thick: thick * 0.64, snowH: subSnow, isSubTwig: true)
                    }
                    if style == .close && wt < 0.64 && len > 1.50 {
                        let tipR = len * (0.60 + rand.next() * 0.12)
                        let uT = tipR / len
                        let tipY = by - droop * powf(uT, 1.38) + powf(uT, 2.8) * (droop * 0.22)
                        let tipOx = trunkOx + dx * tipR
                        let tipOz = trunkOz + dz * tipR
                        let side: Float = (b % 2 == 0) ? 1.0 : -1.0
                        for sgn in [side, -side] {
                            if sgn != side && rand.next() < 0.45 { continue }
                            let fanAz = az + sgn * (0.58 + rand.next() * 0.30)
                            let fanLen = len * (0.30 + rand.next() * 0.14)
                            let fanW = halfW * 0.36
                            let fanDroop = fanLen * 0.48
                            emitBoughFrond(ox: tipOx, oy: tipY - 0.04, oz: tipOz, az: fanAz, len: fanLen, halfW: fanW, droop: fanDroop, thick: thick * 0.44, snowH: 0.0, isSubTwig: true)
                        }
                    }
                }
            }
        }

        // 3. Apex leader shoots
        let apexY = totalH - 0.55
        let ax = sinf(0.95 * .pi) * crookX
        let azO = sinf(0.95 * .pi) * crookZ
        for k in 0..<5 {
            let a = (Float(k) / 5.0) * .pi * 2.0 + rand.next() * 0.45
            emitBoughFrond(ox: ax, oy: apexY - 0.18, oz: azO, az: a, len: 0.85 + rand.next() * 0.25, halfW: 0.34, droop: 0.26, thick: 0.26, snowH: rand.next() > 0.4 ? 0.14 : 0.0, isSubTwig: true)
        }
        let tipH = totalH + 0.25
        let rCap: Float = 0.22
        for k in 0..<5 {
            let a0 = (Float(k) / 5.0) * .pi * 2.0
            let a1 = (Float(k + 1) / 5.0) * .pi * 2.0
            pushTri(
                SIMD3<Float>(ax + cosf(a0) * rCap, apexY - 0.15, azO + sinf(a0) * rCap),
                SIMD3<Float>(ax, tipH, azO),
                SIMD3<Float>(ax + cosf(a1) * rCap, apexY - 0.15, azO + sinf(a1) * rCap),
                1.0, 0.82, 1.0, 0.82, 0.4
            )
        }

        return PropMesh.upload(device: device, vertices: vertices, indices: nil, label: label)
    }

    // MARK: - 3. Protogine Granite Outcrop

    public static func buildRockMesh(device: MTLDevice, seed: Float = 77) -> PropMesh? {
        var vertices: [PropVertex] = []
        var indices: [UInt32] = []

        func appendSubRock(
            ox: Float, oy: Float, oz: Float,
            sx: Float, sy: Float, sz: Float,
            rotY: Float, subSeed: Float, isSnowCollar: Bool
        ) {
            let (stdVerts, subIdx) = MeshPrimitives.makeSphere(
                radius: 1.0,
                widthSegs: isSnowCollar ? 20 : 22,
                heightSegs: isSnowCollar ? 12 : 16
            )
            var subPos: [SIMD3<Float>] = []
            subPos.reserveCapacity(stdVerts.count)
            let cY = cosf(rotY), sY = sinf(rotY)
            for v in stdVerts {
                let ux = v.position.x, uy = v.position.y, uz = v.position.z
                var x: Float, y: Float, z: Float
                if isSnowCollar {
                    let wave = 1.0 + 0.14 * sinf(atan2f(uz, ux) * 3.0 + subSeed)
                    x = ux * sx * wave
                    z = uz * sz * wave
                    y = uy > 0 ? powf(uy, 0.72) * sy * (0.85 + 0.20 * cosf(ux * 2.5 + subSeed)) : (uy * 1.15)
                } else {
                    let dip1 = uy * 0.44 + ux * 0.89
                    let dip2 = ux * 0.62 - uz * 0.78 + uy * 0.35
                    let plane1 = 1.0 - abs(sinf(dip1 * 2.0 + uz * 1.2 + subSeed * 0.4))
                    let plane2 = 1.0 - abs(cosf(dip2 * 2.4 + subSeed * 0.9))
                    let crag = 0.5 + 0.5 * sinf(ux * 3.9 + uz * 3.1 - uy * 3.2)
                    let strata = 0.68 + 0.36 * plane1 + 0.20 * plane2 + 0.08 * crag
                    let normP = powf(powf(abs(ux), 2.5) + powf(abs(uy), 2.5) + powf(abs(uz), 2.5) + 1e-4, 1.0 / 2.5)
                    let mixU: Float = 0.58
                    let bx = ux * (1.0 - mixU) + (ux / normP) * mixU
                    let by = uy * (1.0 - mixU) + (uy / normP) * mixU
                    let bz = uz * (1.0 - mixU) + (uz / normP) * mixU
                    let flare = uy < 0 ? (1.0 + (-uy) * 0.42) : (1.0 - uy * 0.14)
                    x = bx * sx * strata * flare + (uy > 0 ? uy * 0.18 : 0.0)
                    z = bz * sz * strata * flare - (uy > 0 ? uy * 0.12 : 0.0)
                    y = by * sy * (uy > 0 ? (0.82 + 0.32 * plane1) : 1.05)
                }
                let rx = x * cY - z * sY + ox
                let rz = x * sY + z * cY + oz
                subPos.append(SIMD3<Float>(rx, y + oy, rz))
            }
            let subNorms = computeWeldedSmoothNormals(positions: subPos, indices: subIdx)
            let baseV = UInt32(vertices.count)
            for i in subPos.indices {
                let p = subPos[i]
                let n = subNorms[i]
                let prop: SIMD3<Float>
                if isSnowCollar {
                    prop = SIMD3<Float>(0.0, 0.86 + min(0.14, max(0.0, p.y * 0.25)), subSeed)
                } else {
                    let expose = smoothstep(-0.25, 0.14, p.y)
                    let seamDist = hypotf(p.x - ox * 0.5, p.z - oz * 0.5)
                    let ao = clamp(0.48 + p.y * 0.42 + seamDist * 0.14, 0.36, 1.0)
                    prop = SIMD3<Float>(expose, ao, subSeed)
                }
                vertices.append(PropVertex(position: p, normal: n, prop: prop))
            }
            for idxVal in subIdx {
                indices.append(baseV + idxVal)
            }
        }

        appendSubRock(ox:  0.00, oy:  0.00, oz:  0.00, sx: 1.24, sy: 0.92, sz: 0.88, rotY:  0.00, subSeed: seed,      isSnowCollar: false)
        appendSubRock(ox:  0.95, oy: -0.26, oz:  0.34, sx: 0.66, sy: 0.52, sz: 0.58, rotY:  0.72, subSeed: seed + 17, isSnowCollar: false)
        appendSubRock(ox: -0.82, oy: -0.34, oz: -0.28, sx: 0.52, sy: 0.40, sz: 0.48, rotY: -0.95, subSeed: seed + 41, isSnowCollar: false)
        appendSubRock(ox:  0.08, oy: -0.30, oz:  0.06, sx: 1.72, sy: 0.32, sz: 1.35, rotY:  0.20, subSeed: seed + 5,  isSnowCollar: true)

        return PropMesh.upload(device: device, vertices: vertices, indices: indices, label: "GraniteRock")
    }

    // MARK: - 4. Clustered Multi-Slab Glacial Serac Ridge

    public static func buildSeracClusterMesh(device: MTLDevice, seed: Float = 311) -> PropMesh? {
        var vertices: [PropVertex] = []
        var indices: [UInt32] = []

        func emitIceSlab(
            cx: Float, cy: Float, cz: Float,
            hw: Float, hh: Float, hd: Float,
            yaw: Float, tiltX: Float, tiltZ: Float,
            subSeed: Float, pinchAO: Float = 0.0
        ) {
            let baseDepth: Float = -1.55
            let totalH = hh - baseDepth
            var (boxPos, isTopFace, boxIdx) = makeSubdividedBox(
                width: hw * 2.0, height: totalH, depth: hd * 2.0,
                segX: 6, segY: 9, segZ: 6
            )
            let cY = cosf(yaw), sY = sinf(yaw)
            let strataSeed = (subSeed * 0.137).truncatingRemainder(dividingBy: 1.0)
            let topSlopeX = sinf(subSeed * 1.7) * 0.24 * hh
            let topSlopeZ = cosf(subSeed * 2.3) * 0.22 * hh
            var vNormArr = [Float](repeating: 0, count: boxPos.count)

            for i in boxPos.indices {
                let ox = boxPos[i].x
                let oy = boxPos[i].y
                let oz = boxPos[i].z
                let nx = ox / max(0.01, hw)
                let nz = oz / max(0.01, hd)
                let v = clamp((oy + totalH * 0.5) / max(0.01, totalH), 0.0, 1.0)
                vNormArr[i] = v

                let tap = 1.0 - powf(v, 1.35) * 0.24
                let skirtFlare: Float = v < 0.28 ? powf((0.28 - v) / 0.28, 1.5) * 0.26 : 0.0
                let corniceLip: Float = v > 0.85 ? sinf(((v - 0.85) / 0.15) * .pi * 0.85) * 0.075 : 0.0
                let rawStep = sinf(v * 15.5 + subSeed * 2.1 + nx * 0.6)
                let stepSign: Float = rawStep > 0 ? 1.0 : (rawStep < 0 ? -1.0 : 0.0)
                let strataStep = 0.068 * stepSign * powf(abs(rawStep), 0.62) * sinf(v * .pi)
                let cornerChisel = 0.065 * cosf(v * 9.0 + nx * 2.4 - nz * 2.1 + subSeed) * sinf(v * .pi)
                let radScale = tap + skirtFlare + corniceLip + strataStep + cornerChisel

                let edgeEnvX = max(0.0, 1.0 - nx * nx)
                let edgeEnvZ = max(0.0, 1.0 - nz * nz)
                let vEnv = sinf(v * .pi)
                let fluteX = abs(sinf(nz * 3.8 + subSeed + v * 1.4)) - 0.55
                let fluteZ = abs(cosf(nx * 3.8 - subSeed + v * 1.2)) - 0.55
                let scoopX = -0.15 * edgeEnvZ * abs(nx) * vEnv * (0.45 + 0.55 * fluteX)
                let scoopZ = -0.15 * edgeEnvX * abs(nz) * vEnv * (0.45 + 0.55 * fluteZ)

                let signOx: Float = (ox == 0 ? 1.0 : (ox > 0 ? 1.0 : -1.0))
                let signOz: Float = (oz == 0 ? 1.0 : (oz > 0 ? 1.0 : -1.0))
                let lx = ox * radScale + scoopX * signOx + v * tiltX
                let lz = oz * radScale + scoopZ * signOz + v * tiltZ

                let ridgeCrest = 0.06 * abs(sinf(nx * 2.6 + subSeed)) * cosf(nz * 1.8 - subSeed)
                let topY = hh + nx * topSlopeX + nz * topSlopeZ + ridgeCrest
                var ly = baseDepth + (topY - baseDepth) * v
                if v > 0.99 {
                    let thinCrust = max(0.0, 1.0 - nx * nx) * max(0.0, 1.0 - nz * nz) * 0.055 * hh
                    ly += thinCrust
                }

                let wx = lx * cY - lz * sY + cx
                let wz = lx * sY + lz * cY + cz
                boxPos[i] = SIMD3<Float>(wx, ly + cy, wz)
            }

            let slabNorms = computeIndexedNormals(positions: boxPos, indices: boxIdx)
            let baseV = UInt32(vertices.count)
            for i in boxPos.indices {
                let p = boxPos[i]
                let n = slabNorms[i]
                let v = vNormArr[i]
                let prop: SIMD3<Float>
                if isTopFace[i] {
                    let dCenter = hypotf(p.x - cx, p.z - cz) / max(0.5, hypotf(hw, hd))
                    prop = SIMD3<Float>(0.0, clamp(1.03 - dCenter * 0.14, 0.86, 1.04), strataSeed)
                } else {
                    let botBlend = smoothstep(0.22, 0.36, v)
                    let topBlend = 1.0 - smoothstep(0.90, 0.985, v)
                    let expose = botBlend * topBlend
                    let underLip = 1.0 - 0.36 * expf(-powf((v - 0.86) / 0.09, 2.0))
                    let baseAO = lerp(0.36, 1.0, smoothstep(0.14, 0.58, v))
                    let clusterDist = hypotf(p.x, p.z)
                    let innerPinch = lerp(1.0 - pinchAO, 1.0, clamp(clusterDist / 1.55, 0.0, 1.0))
                    prop = SIMD3<Float>(expose, clamp(baseAO * underLip * innerPinch, 0.20, 1.04), strataSeed)
                }
                vertices.append(PropVertex(position: p, normal: n, prop: prop))
            }
            for idxVal in boxIdx {
                indices.append(baseV + idxVal)
            }
        }

        emitIceSlab(cx:  0.00, cy:  0.00, cz:  0.00, hw: 0.86, hh: 1.24, hd: 0.50, yaw:  0.04, tiltX:  0.14, tiltZ:  0.20, subSeed: seed + 1,   pinchAO: 0.20)
        emitIceSlab(cx: -0.90, cy: -0.10, cz:  0.14, hw: 0.66, hh: 0.94, hd: 0.42, yaw: -0.26, tiltX: -0.18, tiltZ:  0.14, subSeed: seed + 19,  pinchAO: 0.30)
        emitIceSlab(cx:  0.92, cy: -0.14, cz: -0.12, hw: 0.70, hh: 0.84, hd: 0.44, yaw:  0.29, tiltX:  0.16, tiltZ:  0.24, subSeed: seed + 37,  pinchAO: 0.30)
        emitIceSlab(cx: -0.24, cy: -0.22, cz:  0.54, hw: 0.60, hh: 0.56, hd: 0.32, yaw:  0.15, tiltX: -0.08, tiltZ:  0.26, subSeed: seed + 53,  pinchAO: 0.38)
        emitIceSlab(cx:  0.38, cy: -0.24, cz: -0.46, hw: 0.62, hh: 0.66, hd: 0.34, yaw: -0.18, tiltX:  0.10, tiltZ: -0.10, subSeed: seed + 71,  pinchAO: 0.32)
        emitIceSlab(cx: -1.42, cy: -0.42, cz:  0.58, hw: 0.32, hh: 0.34, hd: 0.24, yaw:  0.52, tiltX: -0.06, tiltZ:  0.08, subSeed: seed + 89,  pinchAO: 0.15)
        emitIceSlab(cx:  1.38, cy: -0.44, cz:  0.48, hw: 0.28, hh: 0.28, hd: 0.22, yaw: -0.48, tiltX:  0.07, tiltZ:  0.05, subSeed: seed + 103, pinchAO: 0.15)
        emitIceSlab(cx:  0.18, cy: -0.46, cz:  0.98, hw: 0.30, hh: 0.25, hd: 0.20, yaw:  0.24, tiltX: -0.04, tiltZ:  0.06, subSeed: seed + 127, pinchAO: 0.12)

        // Snow ramp collar
        let (cVerts, cIdx) = MeshPrimitives.makeSphere(radius: 1.0, widthSegs: 26, heightSegs: 12)
        var cPos: [SIMD3<Float>] = []
        cPos.reserveCapacity(cVerts.count)
        for v in cVerts {
            let ux = v.position.x, uy = v.position.y, uz = v.position.z
            let ang = atan2f(uz, ux)
            let wave = 1.0 + 0.16 * sinf(ang * 2.0 + seed) + 0.10 * cosf(ang * 3.0 - seed)
            let x = ux * 1.56 * wave
            let z = uz * 1.04 * wave
            let y: Float = uy > 0
                ? powf(uy, 0.96) * 0.24 * (0.76 + 0.24 * sinf(ux * 2.4 + uz * 2.1)) - 0.24
                : (uy * 1.85 - 0.24)
            cPos.append(SIMD3<Float>(x, y, z))
        }
        let cNorms = computeWeldedSmoothNormals(positions: cPos, indices: cIdx)
        let baseV = UInt32(vertices.count)
        for i in cPos.indices {
            vertices.append(PropVertex(position: cPos[i], normal: cNorms[i], prop: SIMD3<Float>(0.0, 0.85, 0.5)))
        }
        for idxVal in cIdx {
            indices.append(baseV + idxVal)
        }

        return PropMesh.upload(device: device, vertices: vertices, indices: indices, label: "SeracCluster")
    }

    // MARK: - 5. Wind-Lip Kicker Mesh

    public static func buildWindLipKickerMesh(device: MTLDevice) -> PropMesh? {
        let segX = 14, segZ = 16
        let W: Float = 8.4, L: Float = 11.6
        var positions: [SIMD3<Float>] = []
        var indices: [UInt32] = []

        for iz in 0...segZ {
            let tz = Float(iz) / Float(segZ)
            let z = (0.5 - tz) * L
            let prof: Float
            if tz < 0.78 {
                let u = tz / 0.78
                prof = powf(u, 1.85) * smoothstep(0.0, 0.22, tz)
            } else {
                let u = (tz - 0.78) / 0.22
                prof = cosf(u * .pi * 0.5)
            }
            for ix in 0...segX {
                let tx = Float(ix) / Float(segX)
                let x = (tx - 0.5) * W
                let edgeFade = powf(max(0.0, cosf((tx - 0.5) * .pi)), 1.35)
                let edgeMin = min(min(tx, 1.0 - tx), min(tz, 1.0 - tz)) * 2.0
                let skirt = -0.95 * powf(max(0.0, 1.0 - smoothstep(0.0, 0.24, edgeMin)), 1.6)
                let y = prof * edgeFade * 2.15 + skirt
                positions.append(SIMD3<Float>(x, y, z))
            }
        }

        let stride = segX + 1
        for iz in 0..<segZ {
            for ix in 0..<segX {
                let i00 = UInt32(iz * stride + ix)
                let i10 = i00 + 1
                let i01 = i00 + UInt32(stride)
                let i11 = i01 + 1
                // CCW winding looking from +Y
                indices.append(contentsOf: [i00, i10, i01, i10, i11, i01])
            }
        }

        let normals = computeIndexedNormals(positions: positions, indices: indices)
        var verts: [PropVertex] = []
        verts.reserveCapacity(positions.count)
        for i in positions.indices {
            verts.append(PropVertex(position: positions[i], normal: normals[i], prop: SIMD3<Float>(0.0, 1.0, 0.5)))
        }
        return PropMesh.upload(device: device, vertices: verts, indices: indices, label: "WindLipKicker")
    }

    // MARK: - 6. Stake Base Snow Mound

    public static func buildStakeSnowMoundMesh(device: MTLDevice) -> PropMesh? {
        let (sVerts, indices) = MeshPrimitives.makeSphere(radius: 0.18, widthSegs: 12, heightSegs: 8)
        let positions = sVerts.map { v -> SIMD3<Float> in
            let my = v.position.y
            return SIMD3<Float>(v.position.x, my > 0 ? my * 0.16 : my * 0.85, v.position.z)
        }
        let normals = computeIndexedNormals(positions: positions, indices: indices)
        let verts = positions.enumerated().map { idx, p in
            PropVertex(position: p, normal: normals[idx], prop: SIMD3<Float>(0.0, 0.80, 0.5))
        }
        return PropMesh.upload(device: device, vertices: verts, indices: indices, label: "StakeSnowMound")
    }
}
