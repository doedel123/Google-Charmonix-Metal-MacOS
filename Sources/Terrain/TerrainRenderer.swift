import Foundation
import Metal
import simd

/// Photorealistic Chamonix / Mont Blanc GPU Geometry Clipmap Terrain Renderer.
/// 11 nested power-of-two clipmap levels (N = 64, base step = 0.45 m) + 48x48 km outer Alps mesh.
public final class TerrainRenderer: TerrainSubsystem {
    private static let gridN: Int = 64
    private static let baseStep: Float = 0.45
    private static let numLevels: Int = 11

    private struct ClipmapLevel {
        let mesh: ProceduralMesh
        let step: Float
        let level: Int
        var snapCenter: SIMD2<Float> = .zero
        var childCenter: SIMD2<Float> = .zero
        var childHalf: Float = -1.0
    }

    private let device: MTLDevice
    private let terrainData: any TerrainQueryProtocol
    private var levels: [ClipmapLevel]
    private let outerMesh: ProceduralMesh

    private let vertexFn: MTLFunction
    private let fragmentFn: MTLFunction
    private let depthState: MTLDepthStencilState
    private var pipelines: [Int: MTLRenderPipelineState] = [:]

    public init(
        device: MTLDevice,
        terrainData: any TerrainQueryProtocol
    ) throws {
        self.device = device
        self.terrainData = terrainData

        var builtLevels: [ClipmapLevel] = []
        builtLevels.reserveCapacity(Self.numLevels)
        for lv in 0..<Self.numLevels {
            let step = Self.baseStep * powf(2.0, Float(lv))
            let mesh = try Self.buildClipmapLevelGeometry(device: device, gridN: Self.gridN, isHollow: lv > 0, level: lv)
            builtLevels.append(ClipmapLevel(mesh: mesh, step: step, level: lv))
        }
        self.levels = builtLevels
        self.outerMesh = try Self.buildOuterAlpsGeometry(device: device, gridN: 176)

        let lib = try ShaderLibrary.loadLibrary(device: device, named: "Terrain")
        guard let vf = lib.makeFunction(name: "terrainVertex"),
              let ff = lib.makeFunction(name: "terrainFragment") else {
            throw ShaderLibraryError.compileFailed("Terrain.metal", "Missing Terrain shader entrypoints")
        }
        self.vertexFn = vf
        self.fragmentFn = ff

        let dsDesc = MTLDepthStencilDescriptor()
        dsDesc.depthCompareFunction = .lessEqual
        dsDesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: dsDesc) else {
            throw TextureLoaderError.textureCreationFailed("TerrainDepthState")
        }
        self.depthState = ds
    }

    private static func buildClipmapLevelGeometry(
        device: MTLDevice,
        gridN: Int,
        isHollow: Bool,
        level: Int
    ) throws -> ProceduralMesh {
        let half = Float(gridN) * 0.5
        let innerHalf = Float(gridN) * 0.25 - 2.0
        var verts: [StandardVertex] = []
        var indices: [UInt32] = []
        verts.reserveCapacity((gridN + 1) * (gridN + 1))
        indices.reserveCapacity(gridN * gridN * 6)

        let n = SIMD3<Float>(0, 1, 0)
        let c = SIMD4<Float>(1, 1, 1, 1)
        for iz in 0...gridN {
            for ix in 0...gridN {
                let lx = Float(ix) - half
                let lz = Float(iz) - half
                verts.append(StandardVertex(position: SIMD3<Float>(lx, 0, lz), normal: n, uv: .zero, color: c))
            }
        }

        let row = gridN + 1
        for iz in 0..<gridN {
            for ix in 0..<gridN {
                let cx = Float(ix) - half + 0.5
                let cz = Float(iz) - half + 0.5
                if isHollow && abs(cx) < innerHalf && abs(cz) < innerHalf { continue }
                let i00 = UInt32(iz * row + ix)
                let i10 = i00 + 1
                let i01 = i00 + UInt32(row)
                let i11 = i01 + 1
                indices.append(contentsOf: [i00, i01, i10, i10, i01, i11])
            }
        }

        guard let mesh = MeshPrimitives.upload(device: device, vertices: verts, indices: indices, label: "ClipmapLevel_\(level)") else {
            throw TextureLoaderError.textureCreationFailed("ClipmapLevel_\(level)")
        }
        return mesh
    }

    private static func buildOuterAlpsGeometry(device: MTLDevice, gridN: Int) throws -> ProceduralMesh {
        let outerHalf = ChamonixConstants.outerHalf
        let coreHalf = ChamonixConstants.coreHalf
        let step = (2.0 * outerHalf) / Float(gridN)
        let cutout = coreHalf - 420.0

        var verts: [StandardVertex] = []
        var indices: [UInt32] = []
        verts.reserveCapacity((gridN + 1) * (gridN + 1))
        indices.reserveCapacity(gridN * gridN * 6)

        let n = SIMD3<Float>(0, 1, 0)
        let c = SIMD4<Float>(1, 1, 1, 1)
        for iz in 0...gridN {
            for ix in 0...gridN {
                let x = -outerHalf + Float(ix) * step
                let z = -outerHalf + Float(iz) * step
                verts.append(StandardVertex(position: SIMD3<Float>(x, 0, z), normal: n, uv: .zero, color: c))
            }
        }

        let row = gridN + 1
        for iz in 0..<gridN {
            for ix in 0..<gridN {
                let cx = -outerHalf + (Float(ix) + 0.5) * step
                let cz = -outerHalf + (Float(iz) + 0.5) * step
                if abs(cx) < cutout - step && abs(cz) < cutout - step { continue }
                let i00 = UInt32(iz * row + ix)
                let i10 = i00 + 1
                let i01 = i00 + UInt32(row)
                let i11 = i01 + 1
                indices.append(contentsOf: [i00, i01, i10, i10, i01, i11])
            }
        }

        guard let mesh = MeshPrimitives.upload(device: device, vertices: verts, indices: indices, label: "OuterAlps") else {
            throw TextureLoaderError.textureCreationFailed("OuterAlps")
        }
        return mesh
    }

    private func pipeline(sampleCount: Int) throws -> MTLRenderPipelineState {
        if let existing = pipelines[sampleCount] { return existing }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "TerrainHDR_MSAA\(sampleCount)"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        desc.rasterSampleCount = sampleCount
        desc.colorAttachments[0].pixelFormat = .rgba16Float
        desc.depthAttachmentPixelFormat = .depth32Float
        let pso = try device.makeRenderPipelineState(descriptor: desc)
        pipelines[sampleCount] = pso
        return pso
    }

    public func update(
        cameraPos: SIMD3<Float>,
        focusXZ: SIMD2<Float>,
        preset: LightingPreset,
        sunDir: SIMD3<Float>,
        time: Float,
        trackBuffer: any SnowTracksSubsystem,
        avalanche: AvalancheState
    ) {
        _ = (cameraPos, preset, sunDir, time, trackBuffer, avalanche)
        let halfGrid = Float(Self.gridN) * 0.5
        for i in 0..<levels.count {
            let snap = levels[i].step * 2.0
            let sx = roundf(focusXZ.x / snap) * snap
            let sz = roundf(focusXZ.y / snap) * snap
            levels[i].snapCenter = SIMD2<Float>(sx, sz)
            if i > 0 {
                let prev = levels[i - 1]
                levels[i].childCenter = prev.snapCenter
                levels[i].childHalf = halfGrid * prev.step - prev.step * 1.25
            } else {
                levels[i].childCenter = .zero
                levels[i].childHalf = -1.0
            }
        }
    }

    public func renderHDRPass(
        encoder: MTLRenderCommandEncoder,
        context: RenderPassContext,
        trackBuffer: any SnowTracksSubsystem
    ) {
        let samples = max(1, context.msaaSampleCount)
        guard let pso = try? pipeline(sampleCount: samples) else { return }

        let tex = terrainData.textures

        encoder.pushDebugGroup("TerrainClipmap")
        encoder.setRenderPipelineState(pso)
        encoder.setDepthStencilState(depthState)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setCullMode(.back)

        encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)

        // Vertex textures
        encoder.setVertexTexture(tex.coreDem, index: TextureIndex.coreDem.rawValue)
        encoder.setVertexTexture(tex.coreAux1, index: TextureIndex.coreAux1.rawValue)
        encoder.setVertexTexture(tex.outerDem, index: TextureIndex.outerDem.rawValue)
        encoder.setVertexTexture(trackBuffer.trackTexture, index: TextureIndex.trackTex.rawValue)

        // Fragment textures
        encoder.setFragmentTexture(tex.coreDem, index: TextureIndex.coreDem.rawValue)
        encoder.setFragmentTexture(tex.coreAux1, index: TextureIndex.coreAux1.rawValue)
        encoder.setFragmentTexture(tex.coreAux2, index: TextureIndex.coreAux2.rawValue)
        encoder.setFragmentTexture(tex.outerDem, index: TextureIndex.outerDem.rawValue)
        encoder.setFragmentTexture(tex.outerAux1, index: TextureIndex.outerAux1.rawValue)
        encoder.setFragmentTexture(tex.outerAux2, index: TextureIndex.outerAux2.rawValue)
        encoder.setFragmentTexture(tex.graniteAlbedo, index: TextureIndex.graniteAlbedo.rawValue)
        encoder.setFragmentTexture(tex.graniteNormal, index: TextureIndex.graniteNormal.rawValue)
        encoder.setFragmentTexture(tex.snowNormal, index: TextureIndex.snowNormal.rawValue)
        encoder.setFragmentTexture(tex.snowCavity, index: TextureIndex.snowCavity.rawValue)
        encoder.setFragmentTexture(tex.iceNormal, index: TextureIndex.iceNormal.rawValue)
        encoder.setFragmentTexture(trackBuffer.trackTexture, index: TextureIndex.trackTex.rawValue)
        encoder.setFragmentTexture(trackBuffer.macroTrackTexture, index: TextureIndex.macroTrackTex.rawValue)

        // Draw clipmap levels 0..10
        for lv in levels {
            var lu = TerrainLevelUniforms()
            lu.snapCenter = lv.snapCenter
            lu.childCenter = lv.childCenter
            lu.gridStep = lv.step
            lu.gridN = Float(Self.gridN)
            lu.childHalf = lv.childHalf
            lu.isOuterMesh = 0.0
            lu.level = Float(lv.level)

            encoder.setVertexBuffer(lv.mesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
            encoder.setVertexBytes(&lu, length: MemoryLayout<TerrainLevelUniforms>.stride, index: BufferIndex.subsystemUniforms.rawValue)
            encoder.setFragmentBytes(&lu, length: MemoryLayout<TerrainLevelUniforms>.stride, index: BufferIndex.subsystemUniforms.rawValue)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: lv.mesh.indexCount,
                indexType: .uint32,
                indexBuffer: lv.mesh.indexBuffer,
                indexBufferOffset: 0
            )
        }

        // Draw static 48x48 km outer Alps mesh
        var outerU = TerrainLevelUniforms()
        outerU.snapCenter = .zero
        outerU.childCenter = .zero
        outerU.gridStep = 1.0
        outerU.gridN = Float(Self.gridN)
        outerU.childHalf = ChamonixConstants.coreHalf - 55.0
        outerU.isOuterMesh = 1.0
        outerU.level = Float(Self.numLevels)

        encoder.setVertexBuffer(outerMesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
        encoder.setVertexBytes(&outerU, length: MemoryLayout<TerrainLevelUniforms>.stride, index: BufferIndex.subsystemUniforms.rawValue)
        encoder.setFragmentBytes(&outerU, length: MemoryLayout<TerrainLevelUniforms>.stride, index: BufferIndex.subsystemUniforms.rawValue)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: outerMesh.indexCount,
            indexType: .uint32,
            indexBuffer: outerMesh.indexBuffer,
            indexBufferOffset: 0
        )

        encoder.popDebugGroup()
    }
}
