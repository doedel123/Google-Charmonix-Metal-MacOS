import Foundation
import Metal
import simd

public struct ProceduralMesh {
    public let vertexBuffer: MTLBuffer
    public let indexBuffer: MTLBuffer
    public let vertexCount: Int
    public let indexCount: Int

    public init(vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, vertexCount: Int, indexCount: Int) {
        self.vertexBuffer = vertexBuffer
        self.indexBuffer = indexBuffer
        self.vertexCount = vertexCount
        self.indexCount = indexCount
    }
}

public enum MeshPrimitives {

    public static func standardVertexDescriptor() -> MTLVertexDescriptor {
        let vd = MTLVertexDescriptor()
        vd.attributes[VertexAttribute.position.rawValue].format = .float3
        vd.attributes[VertexAttribute.position.rawValue].offset = MemoryLayout<StandardVertex>.offset(of: \.position) ?? 0
        vd.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.vertices.rawValue

        vd.attributes[VertexAttribute.normal.rawValue].format = .float3
        vd.attributes[VertexAttribute.normal.rawValue].offset = MemoryLayout<StandardVertex>.offset(of: \.normal) ?? 16
        vd.attributes[VertexAttribute.normal.rawValue].bufferIndex = BufferIndex.vertices.rawValue

        vd.attributes[VertexAttribute.UV.rawValue].format = .float2
        vd.attributes[VertexAttribute.UV.rawValue].offset = MemoryLayout<StandardVertex>.offset(of: \.uv) ?? 32
        vd.attributes[VertexAttribute.UV.rawValue].bufferIndex = BufferIndex.vertices.rawValue

        vd.attributes[VertexAttribute.color.rawValue].format = .float4
        vd.attributes[VertexAttribute.color.rawValue].offset = MemoryLayout<StandardVertex>.offset(of: \.color) ?? 48
        vd.attributes[VertexAttribute.color.rawValue].bufferIndex = BufferIndex.vertices.rawValue

        vd.layouts[BufferIndex.vertices.rawValue].stride = MemoryLayout<StandardVertex>.stride
        vd.layouts[BufferIndex.vertices.rawValue].stepFunction = .perVertex
        vd.layouts[BufferIndex.vertices.rawValue].stepRate = 1
        return vd
    }

    public static func upload(
        device: MTLDevice,
        vertices: [StandardVertex],
        indices: [UInt32],
        label: String? = nil
    ) -> ProceduralMesh? {
        guard !vertices.isEmpty, !indices.isEmpty else { return nil }
        let vLen = vertices.count * MemoryLayout<StandardVertex>.stride
        let iLen = indices.count * MemoryLayout<UInt32>.stride
        guard let vb = vertices.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: vLen, options: .storageModeShared) }),
              let ib = indices.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: iLen, options: .storageModeShared) }) else {
            return nil
        }
        vb.label = label.map { "\($0)_VB" }
        ib.label = label.map { "\($0)_IB" }
        return ProceduralMesh(vertexBuffer: vb, indexBuffer: ib, vertexCount: vertices.count, indexCount: indices.count)
    }

    public static func computeSmoothNormals(vertices: inout [StandardVertex], indices: [UInt32]) {
        for i in vertices.indices {
            vertices[i].normal = .zero
        }
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
            let len = simd_length(vertices[i].normal)
            vertices[i].normal = len > 1e-8 ? (vertices[i].normal / len) : SIMD3<Float>(0, 1, 0)
        }
    }

    // MARK: - Fullscreen Quad (Metal NDC -> UV (0,0) top-left to (1,1) bottom-right)

    public static func makeFullscreenQuad(device: MTLDevice) -> ProceduralMesh? {
        let verts: [StandardVertex] = [
            StandardVertex(position: SIMD3<Float>(-1,  1, 0), normal: SIMD3<Float>(0, 0, 1), uv: SIMD2<Float>(0, 0), color: SIMD4<Float>(1, 1, 1, 1)),
            StandardVertex(position: SIMD3<Float>(-1, -1, 0), normal: SIMD3<Float>(0, 0, 1), uv: SIMD2<Float>(0, 1), color: SIMD4<Float>(1, 1, 1, 1)),
            StandardVertex(position: SIMD3<Float>( 1, -1, 0), normal: SIMD3<Float>(0, 0, 1), uv: SIMD2<Float>(1, 1), color: SIMD4<Float>(1, 1, 1, 1)),
            StandardVertex(position: SIMD3<Float>( 1,  1, 0), normal: SIMD3<Float>(0, 0, 1), uv: SIMD2<Float>(1, 0), color: SIMD4<Float>(1, 1, 1, 1)),
        ]
        let indices: [UInt32] = [0, 1, 2, 0, 2, 3]
        return upload(device: device, vertices: verts, indices: indices, label: "FullscreenQuad")
    }

    // MARK: - Box (CCW, Right-Handed Y-up)

    public static func makeBox(
        width: Float,
        height: Float,
        depth: Float,
        color: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 1)
    ) -> ([StandardVertex], [UInt32]) {
        let hw = width * 0.5, hh = height * 0.5, hd = depth * 0.5
        var verts: [StandardVertex] = []
        var indices: [UInt32] = []

        func addFace(_ n: SIMD3<Float>, _ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>, _ p3: SIMD3<Float>) {
            let base = UInt32(verts.count)
            verts.append(StandardVertex(position: p0, normal: n, uv: SIMD2<Float>(0, 0), color: color))
            verts.append(StandardVertex(position: p1, normal: n, uv: SIMD2<Float>(0, 1), color: color))
            verts.append(StandardVertex(position: p2, normal: n, uv: SIMD2<Float>(1, 1), color: color))
            verts.append(StandardVertex(position: p3, normal: n, uv: SIMD2<Float>(1, 0), color: color))
            indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }

        // +Z front
        addFace(SIMD3<Float>(0, 0, 1), SIMD3<Float>(-hw, hh, hd), SIMD3<Float>(-hw, -hh, hd), SIMD3<Float>(hw, -hh, hd), SIMD3<Float>(hw, hh, hd))
        // -Z back
        addFace(SIMD3<Float>(0, 0, -1), SIMD3<Float>(hw, hh, -hd), SIMD3<Float>(hw, -hh, -hd), SIMD3<Float>(-hw, -hh, -hd), SIMD3<Float>(-hw, hh, -hd))
        // +X right
        addFace(SIMD3<Float>(1, 0, 0), SIMD3<Float>(hw, hh, hd), SIMD3<Float>(hw, -hh, hd), SIMD3<Float>(hw, -hh, -hd), SIMD3<Float>(hw, hh, -hd))
        // -X left
        addFace(SIMD3<Float>(-1, 0, 0), SIMD3<Float>(-hw, hh, -hd), SIMD3<Float>(-hw, -hh, -hd), SIMD3<Float>(-hw, -hh, hd), SIMD3<Float>(-hw, hh, hd))
        // +Y top
        addFace(SIMD3<Float>(0, 1, 0), SIMD3<Float>(-hw, hh, -hd), SIMD3<Float>(-hw, hh, hd), SIMD3<Float>(hw, hh, hd), SIMD3<Float>(hw, hh, -hd))
        // -Y bottom
        addFace(SIMD3<Float>(0, -1, 0), SIMD3<Float>(-hw, -hh, hd), SIMD3<Float>(-hw, -hh, -hd), SIMD3<Float>(hw, -hh, -hd), SIMD3<Float>(hw, -hh, hd))

        return (verts, indices)
    }

    // MARK: - Plane (XZ horizontal, normal +Y, CCW)

    public static func makePlaneXZ(
        width: Float,
        depth: Float,
        segX: Int = 1,
        segZ: Int = 1,
        color: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 1)
    ) -> ([StandardVertex], [UInt32]) {
        let sx = max(1, segX), sz = max(1, segZ)
        var verts: [StandardVertex] = []
        var indices: [UInt32] = []
        let n = SIMD3<Float>(0, 1, 0)
        for iz in 0...sz {
            let v = Float(iz) / Float(sz)
            let z = (v - 0.5) * depth
            for ix in 0...sx {
                let u = Float(ix) / Float(sx)
                let x = (u - 0.5) * width
                verts.append(StandardVertex(position: SIMD3<Float>(x, 0, z), normal: n, uv: SIMD2<Float>(u, v), color: color))
            }
        }
        let row = sx + 1
        for iz in 0..<sz {
            for ix in 0..<sx {
                let a = UInt32(iz * row + ix)
                let b = UInt32((iz + 1) * row + ix)
                let c = UInt32((iz + 1) * row + ix + 1)
                let d = UInt32(iz * row + ix + 1)
                indices.append(contentsOf: [a, b, c, a, c, d])
            }
        }
        return (verts, indices)
    }

    // MARK: - Cylinder / Cone (CCW, Y-up)

    public static func makeCylinder(
        radiusTop: Float,
        radiusBottom: Float,
        height: Float,
        radialSegs: Int = 16,
        heightSegs: Int = 1,
        openEnded: Bool = false,
        thetaStart: Float = 0,
        thetaLength: Float = .pi * 2,
        color: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 1)
    ) -> ([StandardVertex], [UInt32]) {
        let rSeg = max(3, radialSegs)
        let hSeg = max(1, heightSegs)
        var verts: [StandardVertex] = []
        var indices: [UInt32] = []

        let halfH = height * 0.5
        let slope = (radiusBottom - radiusTop) / max(height, 1e-5)

        for iy in 0...hSeg {
            let v = Float(iy) / Float(hSeg)
            let y = halfH - v * height
            let r = lerp(radiusTop, radiusBottom, v)
            for ix in 0...rSeg {
                let u = Float(ix) / Float(rSeg)
                let theta = u * thetaLength + thetaStart
                let sinT = sinf(theta)
                let cosT = cosf(theta)
                let pos = SIMD3<Float>(r * sinT, y, r * cosT)
                let nor = simd_normalize(SIMD3<Float>(sinT, slope, cosT))
                verts.append(StandardVertex(position: pos, normal: nor, uv: SIMD2<Float>(u, v), color: color))
            }
        }

        let row = rSeg + 1
        for iy in 0..<hSeg {
            for ix in 0..<rSeg {
                let a = UInt32(iy * row + ix)
                let b = UInt32((iy + 1) * row + ix)
                let c = UInt32((iy + 1) * row + ix + 1)
                let d = UInt32(iy * row + ix + 1)
                indices.append(contentsOf: [a, b, c, a, c, d])
            }
        }

        if !openEnded {
            if radiusTop > 1e-5 {
                let centerIdx = UInt32(verts.count)
                let n = SIMD3<Float>(0, 1, 0)
                verts.append(StandardVertex(position: SIMD3<Float>(0, halfH, 0), normal: n, uv: SIMD2<Float>(0.5, 0.5), color: color))
                let ringStart = UInt32(verts.count)
                for ix in 0...rSeg {
                    let u = Float(ix) / Float(rSeg)
                    let theta = u * thetaLength + thetaStart
                    let x = radiusTop * sinf(theta)
                    let z = radiusTop * cosf(theta)
                    verts.append(StandardVertex(position: SIMD3<Float>(x, halfH, z), normal: n, uv: SIMD2<Float>(sinf(theta) * 0.5 + 0.5, cosf(theta) * 0.5 + 0.5), color: color))
                }
                for ix in 0..<UInt32(rSeg) {
                    indices.append(contentsOf: [centerIdx, ringStart + ix, ringStart + ix + 1])
                }
            }
            if radiusBottom > 1e-5 {
                let centerIdx = UInt32(verts.count)
                let n = SIMD3<Float>(0, -1, 0)
                verts.append(StandardVertex(position: SIMD3<Float>(0, -halfH, 0), normal: n, uv: SIMD2<Float>(0.5, 0.5), color: color))
                let ringStart = UInt32(verts.count)
                for ix in 0...rSeg {
                    let u = Float(ix) / Float(rSeg)
                    let theta = u * thetaLength + thetaStart
                    let x = radiusBottom * sinf(theta)
                    let z = radiusBottom * cosf(theta)
                    verts.append(StandardVertex(position: SIMD3<Float>(x, -halfH, z), normal: n, uv: SIMD2<Float>(sinf(theta) * 0.5 + 0.5, cosf(theta) * 0.5 + 0.5), color: color))
                }
                for ix in 0..<UInt32(rSeg) {
                    indices.append(contentsOf: [centerIdx, ringStart + ix + 1, ringStart + ix])
                }
            }
        }
        return (verts, indices)
    }

    public static func makeCone(
        radius: Float,
        height: Float,
        radialSegs: Int = 16,
        heightSegs: Int = 1,
        openEnded: Bool = false,
        color: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 1)
    ) -> ([StandardVertex], [UInt32]) {
        makeCylinder(radiusTop: 0, radiusBottom: radius, height: height, radialSegs: radialSegs, heightSegs: heightSegs, openEnded: openEnded, color: color)
    }

    // MARK: - Sphere (CCW, Y-up, outward normals)

    public static func makeSphere(
        radius: Float,
        widthSegs: Int = 24,
        heightSegs: Int = 16,
        phiStart: Float = 0,
        phiLength: Float = .pi * 2,
        thetaStart: Float = 0,
        thetaLength: Float = .pi,
        color: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 1)
    ) -> ([StandardVertex], [UInt32]) {
        let wSeg = max(3, widthSegs)
        let hSeg = max(2, heightSegs)
        var verts: [StandardVertex] = []
        var indices: [UInt32] = []

        for iy in 0...hSeg {
            let v = Float(iy) / Float(hSeg)
            let theta = thetaStart + v * thetaLength
            let sinT = sinf(theta)
            let cosT = cosf(theta)
            for ix in 0...wSeg {
                let u = Float(ix) / Float(wSeg)
                let phi = phiStart + u * phiLength
                let nx = -cosf(phi) * sinT
                let ny = cosT
                let nz = sinf(phi) * sinT
                let nor = simd_normalize(SIMD3<Float>(nx, ny, nz))
                verts.append(StandardVertex(position: nor * radius, normal: nor, uv: SIMD2<Float>(u, v), color: color))
            }
        }

        let row = wSeg + 1
        for iy in 0..<hSeg {
            for ix in 0..<wSeg {
                let a = UInt32(iy * row + ix)
                let b = UInt32((iy + 1) * row + ix)
                let c = UInt32((iy + 1) * row + ix + 1)
                let d = UInt32(iy * row + ix + 1)
                if iy != 0 || thetaStart > 0 {
                    indices.append(contentsOf: [a, b, d])
                }
                if iy != hSeg - 1 || (thetaStart + thetaLength) < .pi {
                    indices.append(contentsOf: [b, c, d])
                }
            }
        }
        return (verts, indices)
    }

    // MARK: - Capsule (Y-aligned, total height = length + 2 * radius)

    public static func makeCapsule(
        radius: Float,
        length: Float,
        capSegs: Int = 8,
        radialSegs: Int = 16,
        color: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 1)
    ) -> ([StandardVertex], [UInt32]) {
        let rSeg = max(4, radialSegs)
        let cSeg = max(2, capSegs)
        let halfL = length * 0.5
        var verts: [StandardVertex] = []
        var indices: [UInt32] = []

        let totalRings = cSeg * 2 + 1
        for ir in 0...totalRings {
            let yOff: Float
            let ringR: Float
            let nyBase: Float
            let v = Float(ir) / Float(totalRings)

            if ir <= cSeg {
                let t = Float(ir) / Float(cSeg) * (.pi * 0.5)
                ringR = radius * sinf(t)
                yOff = halfL + radius * cosf(t)
                nyBase = cosf(t)
            } else {
                let t = Float(ir - cSeg - 1) / Float(cSeg) * (.pi * 0.5)
                ringR = radius * cosf(t)
                yOff = -halfL - radius * sinf(t)
                nyBase = -sinf(t)
            }

            let horizScale = ringR / max(radius, 1e-6)
            for ix in 0...rSeg {
                let u = Float(ix) / Float(rSeg)
                let theta = u * .pi * 2.0
                let sinT = sinf(theta)
                let cosT = cosf(theta)
                let pos = SIMD3<Float>(ringR * sinT, yOff, ringR * cosT)
                let nor = simd_normalize(SIMD3<Float>(horizScale * sinT, nyBase, horizScale * cosT))
                verts.append(StandardVertex(position: pos, normal: nor, uv: SIMD2<Float>(u, v), color: color))
            }
        }

        let row = rSeg + 1
        for ir in 0..<totalRings {
            for ix in 0..<rSeg {
                let a = UInt32(ir * row + ix)
                let b = UInt32((ir + 1) * row + ix)
                let c = UInt32((ir + 1) * row + ix + 1)
                let d = UInt32(ir * row + ix + 1)
                indices.append(contentsOf: [a, b, c, a, c, d])
            }
        }
        return (verts, indices)
    }

    // MARK: - Torus (XY plane, normal Z axis)

    public static func makeTorus(
        radius: Float,
        tube: Float,
        radialSegs: Int = 12,
        tubularSegs: Int = 24,
        arc: Float = .pi * 2,
        color: SIMD4<Float> = SIMD4<Float>(1, 1, 1, 1)
    ) -> ([StandardVertex], [UInt32]) {
        let rSeg = max(3, radialSegs)
        let tSeg = max(3, tubularSegs)
        var verts: [StandardVertex] = []
        var indices: [UInt32] = []

        for j in 0...rSeg {
            let v = Float(j) / Float(rSeg) * .pi * 2.0
            let cosV = cosf(v), sinV = sinf(v)
            for i in 0...tSeg {
                let u = Float(i) / Float(tSeg) * arc
                let cosU = cosf(u), sinU = sinf(u)
                let cx = radius * cosU
                let cy = radius * sinU
                let pos = SIMD3<Float>(
                    (radius + tube * cosV) * cosU,
                    (radius + tube * cosV) * sinU,
                    tube * sinV
                )
                let nor = simd_normalize(pos - SIMD3<Float>(cx, cy, 0))
                verts.append(StandardVertex(position: pos, normal: nor, uv: SIMD2<Float>(Float(i) / Float(tSeg), Float(j) / Float(rSeg)), color: color))
            }
        }

        let row = tSeg + 1
        for j in 0..<rSeg {
            for i in 0..<tSeg {
                let a = UInt32(j * row + i)
                let b = UInt32((j + 1) * row + i)
                let c = UInt32((j + 1) * row + i + 1)
                let d = UInt32(j * row + i + 1)
                indices.append(contentsOf: [a, d, b, b, d, c])
            }
        }
        return (verts, indices)
    }
}
