import Foundation
import Metal
import simd

/// High-Resolution Rolling World-Locked GPU Snow-Track Deformation Buffer
/// (1536x1536 covering 64x64 m + 1024x1024 macro cascade covering 360x360 m).
public final class SnowTrackBuffer: SnowTracksSubsystem {
    public let worldSpan: Float
    public let resolution: Int
    public private(set) var originX: Float = 0
    public private(set) var originZ: Float = 0

    public let macroSpan: Float
    public let macroResolution: Int
    public private(set) var macroOriginX: Float = 0
    public private(set) var macroOriginZ: Float = 0

    public var trackTexture: MTLTexture { rtA }
    public var macroTrackTexture: MTLTexture { macroA }

    private var rtA: MTLTexture
    private var rtB: MTLTexture
    private var macroA: MTLTexture
    private var macroB: MTLTexture

    private let quadMesh: ProceduralMesh
    private let copyPipeline: MTLRenderPipelineState
    private let stampPipeline: MTLRenderPipelineState
    private var initialized: Bool = false

    private enum TrackOp {
        case clearAll
        case copyNear(shiftUV: SIMD2<Float>, decay: Float)
        case copyMacro(shiftUV: SIMD2<Float>, decay: Float)
        case stampNear(SnowTrackUniforms)
        case stampMacro(SnowTrackUniforms)
    }
    private var pendingOps: [TrackOp] = []

    public init(
        device: MTLDevice,
        worldSpan: Float = 64.0,
        resolution: Int = 1536,
        macroSpan: Float = 360.0,
        macroResolution: Int = 1024
    ) throws {
        self.worldSpan = worldSpan
        self.resolution = resolution
        self.macroSpan = macroSpan
        self.macroResolution = macroResolution

        func makeRT(_ size: Int, label: String) throws -> MTLTexture {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm,
                width: size,
                height: size,
                mipmapped: false
            )
            desc.usage = [.renderTarget, .shaderRead]
            desc.storageMode = .private
            guard let tex = device.makeTexture(descriptor: desc) else {
                throw TextureLoaderError.textureCreationFailed(label)
            }
            tex.label = label
            return tex
        }

        self.rtA = try makeRT(resolution, label: "SnowTracksNearA")
        self.rtB = try makeRT(resolution, label: "SnowTracksNearB")
        self.macroA = try makeRT(macroResolution, label: "SnowTracksMacroA")
        self.macroB = try makeRT(macroResolution, label: "SnowTracksMacroB")

        guard let quad = MeshPrimitives.makeFullscreenQuad(device: device) else {
            throw TextureLoaderError.textureCreationFailed("SnowTracksQuad")
        }
        self.quadMesh = quad

        let lib = try ShaderLibrary.loadLibrary(device: device, named: "SnowTracks")
        guard let vFn = lib.makeFunction(name: "snowTracksVertex"),
              let copyFn = lib.makeFunction(name: "snowTracksCopyFragment"),
              let stampFn = lib.makeFunction(name: "snowTracksStampFragment") else {
            throw ShaderLibraryError.compileFailed("SnowTracks.metal", "Missing SnowTracks shader entrypoints")
        }

        let copyDesc = MTLRenderPipelineDescriptor()
        copyDesc.label = "SnowTracksCopy"
        copyDesc.vertexFunction = vFn
        copyDesc.fragmentFunction = copyFn
        copyDesc.colorAttachments[0].pixelFormat = .rgba8Unorm
        self.copyPipeline = try device.makeRenderPipelineState(descriptor: copyDesc)

        let stampDesc = MTLRenderPipelineDescriptor()
        stampDesc.label = "SnowTracksStamp"
        stampDesc.vertexFunction = vFn
        stampDesc.fragmentFunction = stampFn
        stampDesc.colorAttachments[0].pixelFormat = .rgba8Unorm
        self.stampPipeline = try device.makeRenderPipelineState(descriptor: stampDesc)

        self.pendingOps.append(.clearAll)
    }

    public func reset(x: Float, z: Float) {
        originX = x
        originZ = z
        macroOriginX = x
        macroOriginZ = z
        initialized = true
        pendingOps.removeAll(keepingCapacity: true)
        pendingOps.append(.clearAll)
    }

    @inline(__always)
    private func worldToUV(_ wx: Float, _ wz: Float) -> SIMD2<Float> {
        SIMD2<Float>((wx - originX) / worldSpan + 0.5, (wz - originZ) / worldSpan + 0.5)
    }

    @inline(__always)
    private func worldToMacroUV(_ wx: Float, _ wz: Float) -> SIMD2<Float> {
        SIMD2<Float>((wx - macroOriginX) / macroSpan + 0.5, (wz - macroOriginZ) / macroSpan + 0.5)
    }

    public func recenterIfNeeded(skierX: Float, skierZ: Float, decayAmt: Float) {
        if !initialized {
            reset(x: skierX, z: skierZ)
            return
        }

        // 1. High-res near-field cascade (64m @ 4.1cm/px)
        let dx = skierX - originX
        let dz = skierZ - originZ
        if hypotf(dx, dz) >= 7.0 || decayAmt > 0.00001 {
            let texelM = worldSpan / Float(resolution)
            let stepX = roundf(dx / texelM) * texelM
            let stepZ = roundf(dz / texelM) * texelM
            if stepX != 0 || stepZ != 0 || decayAmt > 0.00001 {
                originX += stepX
                originZ += stepZ
                pendingOps.append(.copyNear(
                    shiftUV: SIMD2<Float>(stepX / worldSpan, stepZ / worldSpan),
                    decay: decayAmt
                ))
            }
        }

        // 2. Macro persistent trail cascade (360m @ 35cm/px)
        let mdx = skierX - macroOriginX
        let mdz = skierZ - macroOriginZ
        if hypotf(mdx, mdz) >= 48.0 || decayAmt > 0.00001 {
            let mTexel = macroSpan / Float(macroResolution)
            let mStepX = roundf(mdx / mTexel) * mTexel
            let mStepZ = roundf(mdz / mTexel) * mTexel
            if mStepX != 0 || mStepZ != 0 || decayAmt > 0.00001 {
                macroOriginX += mStepX
                macroOriginZ += mStepZ
                pendingOps.append(.copyMacro(
                    shiftUV: SIMD2<Float>(mStepX / macroSpan, mStepZ / macroSpan),
                    decay: decayAmt * 0.5
                ))
            }
        }
    }

    public func stampTwinSkis(
        lA: SIMD2<Float>, lB: SIMD2<Float>,
        rA: SIMD2<Float>, rB: SIMD2<Float>,
        radius: Float, depth: Float, berm: Float, iceGlint: Float, skew: Float
    ) {
        guard initialized else { return }

        let laUV = worldToUV(lA.x, lA.y)
        let lbUV = worldToUV(lB.x, lB.y)
        let raUV = worldToUV(rA.x, rA.y)
        let rbUV = worldToUV(rB.x, rB.y)

        let maxU = max(max(laUV.x, lbUV.x), max(raUV.x, rbUV.x))
        let minU = min(min(laUV.x, lbUV.x), min(raUV.x, rbUV.x))
        if !(maxU < -0.05 || minU > 1.05) {
            var u = SnowTrackUniforms()
            u.singleMode = 0.0
            u.lA = laUV
            u.lB = lbUV
            u.rA = raUV
            u.rB = rbUV
            u.radiusUV = radius / worldSpan
            u.depth = depth
            u.berm = berm
            u.iceGlint = iceGlint
            u.skewSide = skew
            pendingOps.append(.stampNear(u))
        }

        let mlaUV = worldToMacroUV(lA.x, lA.y)
        let mlbUV = worldToMacroUV(lB.x, lB.y)
        let mraUV = worldToMacroUV(rA.x, rA.y)
        let mrbUV = worldToMacroUV(rB.x, rB.y)

        let mMaxU = max(max(mlaUV.x, mlbUV.x), max(mraUV.x, mrbUV.x))
        let mMinU = min(min(mlaUV.x, mlbUV.x), min(mraUV.x, mrbUV.x))
        if !(mMaxU < -0.05 || mMinU > 1.05) {
            var u = SnowTrackUniforms()
            u.singleMode = 0.0
            u.lA = mlaUV
            u.lB = mlbUV
            u.rA = mraUV
            u.rB = mrbUV
            u.radiusUV = max(radius * 1.4, 0.36) / macroSpan
            u.depth = depth
            u.berm = berm * 0.92
            u.iceGlint = iceGlint * 0.85
            u.skewSide = skew
            pendingOps.append(.stampMacro(u))
        }
    }

    public func stampPoleDimple(x: Float, z: Float, radius: Float = 0.14, depth: Float = 0.58) {
        guard initialized else { return }
        stampCapsuleImpl(
            ax: x, az: z, bx: x + 0.02, bz: z + 0.02,
            radius: radius, depth: depth, berm: depth * 0.72, iceGlint: 0.25, skew: 0.0,
            includeMacro: false
        )
    }

    public func stampCapsule(
        ax: Float, az: Float, bx: Float, bz: Float,
        radius: Float, depth: Float, berm: Float, iceGlint: Float, skew: Float
    ) {
        guard initialized else { return }
        stampCapsuleImpl(
            ax: ax, az: az, bx: bx, bz: bz,
            radius: radius, depth: depth, berm: berm, iceGlint: iceGlint, skew: skew,
            includeMacro: true
        )
    }

    private func stampCapsuleImpl(
        ax: Float, az: Float, bx: Float, bz: Float,
        radius: Float, depth: Float, berm: Float, iceGlint: Float, skew: Float,
        includeMacro: Bool
    ) {
        let aUV = worldToUV(ax, az)
        let bUV = worldToUV(bx, bz)
        if !(max(aUV.x, bUV.x) < -0.05 || min(aUV.x, bUV.x) > 1.05 || max(aUV.y, bUV.y) < -0.05 || min(aUV.y, bUV.y) > 1.05) {
            var u = SnowTrackUniforms()
            u.singleMode = 1.0
            u.lA = aUV
            u.lB = bUV
            u.radiusUV = radius / worldSpan
            u.depth = depth
            u.berm = berm
            u.iceGlint = iceGlint
            u.skewSide = skew
            pendingOps.append(.stampNear(u))
        }

        if includeMacro {
            let maUV = worldToMacroUV(ax, az)
            let mbUV = worldToMacroUV(bx, bz)
            if !(max(maUV.x, mbUV.x) < -0.05 || min(maUV.x, mbUV.x) > 1.05 || max(maUV.y, mbUV.y) < -0.05 || min(maUV.y, mbUV.y) > 1.05) {
                var u = SnowTrackUniforms()
                u.singleMode = 1.0
                u.lA = maUV
                u.lB = mbUV
                u.radiusUV = max(radius, 0.42) / macroSpan
                u.depth = depth
                u.berm = berm * 0.85
                u.iceGlint = iceGlint
                u.skewSide = skew
                pendingOps.append(.stampMacro(u))
            }
        }
    }

    public func encodeRenderPasses(commandBuffer: MTLCommandBuffer) {
        guard !pendingOps.isEmpty else { return }

        for op in pendingOps {
            switch op {
            case .clearAll:
                for tex in [rtA, rtB, macroA, macroB] {
                    let rp = MTLRenderPassDescriptor()
                    rp.colorAttachments[0].texture = tex
                    rp.colorAttachments[0].loadAction = .clear
                    rp.colorAttachments[0].storeAction = .store
                    rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
                    if let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rp) {
                        enc.label = "SnowTracksClear"
                        enc.endEncoding()
                    }
                }

            case let .copyNear(shiftUV, decay):
                var u = SnowTrackUniforms()
                u.shiftUV = shiftUV
                u.decay = decay
                encodeQuadPass(
                    commandBuffer: commandBuffer,
                    pipeline: copyPipeline,
                    srcTex: rtA,
                    dstTex: rtB,
                    uniforms: &u,
                    label: "SnowTracksCopyNear"
                )
                swap(&rtA, &rtB)

            case let .copyMacro(shiftUV, decay):
                var u = SnowTrackUniforms()
                u.shiftUV = shiftUV
                u.decay = decay
                encodeQuadPass(
                    commandBuffer: commandBuffer,
                    pipeline: copyPipeline,
                    srcTex: macroA,
                    dstTex: macroB,
                    uniforms: &u,
                    label: "SnowTracksCopyMacro"
                )
                swap(&macroA, &macroB)

            case .stampNear(var u):
                encodeQuadPass(
                    commandBuffer: commandBuffer,
                    pipeline: stampPipeline,
                    srcTex: rtA,
                    dstTex: rtB,
                    uniforms: &u,
                    label: "SnowTracksStampNear"
                )
                swap(&rtA, &rtB)

            case .stampMacro(var u):
                encodeQuadPass(
                    commandBuffer: commandBuffer,
                    pipeline: stampPipeline,
                    srcTex: macroA,
                    dstTex: macroB,
                    uniforms: &u,
                    label: "SnowTracksStampMacro"
                )
                swap(&macroA, &macroB)
            }
        }
        pendingOps.removeAll(keepingCapacity: true)
    }

    private func encodeQuadPass(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLRenderPipelineState,
        srcTex: MTLTexture,
        dstTex: MTLTexture,
        uniforms: inout SnowTrackUniforms,
        label: String
    ) {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = dstTex
        rp.colorAttachments[0].loadAction = .dontCare
        rp.colorAttachments[0].storeAction = .store
        guard let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rp) else { return }
        enc.label = label
        enc.setRenderPipelineState(pipeline)
        enc.setVertexBuffer(quadMesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
        enc.setFragmentTexture(srcTex, index: 0)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<SnowTrackUniforms>.stride, index: BufferIndex.subsystemUniforms.rawValue)
        enc.drawIndexedPrimitives(
            type: .triangle,
            indexCount: quadMesh.indexCount,
            indexType: .uint32,
            indexBuffer: quadMesh.indexBuffer,
            indexBufferOffset: 0
        )
        enc.endEncoding()
    }
}
