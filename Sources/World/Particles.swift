import Foundation
import Metal
import simd

public final class AlpineParticleSystem: ParticlesSubsystem {
    private let device: MTLDevice
    private let puffTexture: MTLTexture
    private let library: MTLLibrary

    private let maxPuffs: Int
    private let maxFlakes: Int
    private var cursor: Int = 0

    // SoA puff storage matching particles.js 1:1
    private var posX: [Float]
    private var posY: [Float]
    private var posZ: [Float]
    private var velX: [Float]
    private var velY: [Float]
    private var velZ: [Float]
    private var size: [Float]
    private var expand: [Float]
    private var age: [Float]
    private var life: [Float]
    private var rot: [Float]
    private var peakAlpha: [Float]
    private var opacity: [Float]
    private var kind: [Float]

    // Triple-buffered GPU puff buffers
    private let maxInFlight = 3
    private var puffBuffers: [MTLBuffer] = []
    private var puffBufferIndex: Int = 0
    private var activePuffCount: Int = 0

    // Ambient flakes GPU seed buffer (static)
    private let flakeBuffer: MTLBuffer?

    // Ridge snow banners
    private var bannerMesh: ProceduralMesh?
    private var bannerInstanceBuffer: MTLBuffer?
    private var bannerInstanceCount: Int = 0

    // Pipelines keyed by MSAA sample count
    private var puffPipelines: [Int: MTLRenderPipelineState] = [:]
    private var flakePipelines: [Int: MTLRenderPipelineState] = [:]
    private var bannerPipelines: [Int: MTLRenderPipelineState] = [:]
    private let depthReadOnlyState: MTLDepthStencilState

    // Per-frame uniforms
    private var avyAccum: Float = 0
    private var currentFocalScale: Float = 650.0
    private var currentWind: SIMD3<Float> = SIMD3<Float>(4.2, -1.5, 1.8)

    public init(
        device: MTLDevice,
        puffTexture: MTLTexture,
        peaks: [PeakInfo] = [],
        maxPuffs: Int = 1800,
        maxFlakes: Int = 2600
    ) throws {
        self.device = device
        self.puffTexture = puffTexture
        self.maxPuffs = maxPuffs
        self.maxFlakes = maxFlakes
        self.library = try ShaderLibrary.loadLibrary(device: device, named: "Particles")

        self.posX = [Float](repeating: 0, count: maxPuffs)
        self.posY = [Float](repeating: 0, count: maxPuffs)
        self.posZ = [Float](repeating: 0, count: maxPuffs)
        self.velX = [Float](repeating: 0, count: maxPuffs)
        self.velY = [Float](repeating: 0, count: maxPuffs)
        self.velZ = [Float](repeating: 0, count: maxPuffs)
        self.size = [Float](repeating: 0, count: maxPuffs)
        self.expand = [Float](repeating: 0, count: maxPuffs)
        self.age = [Float](repeating: 0, count: maxPuffs)
        self.life = [Float](repeating: -1, count: maxPuffs)
        self.rot = [Float](repeating: 0, count: maxPuffs)
        self.peakAlpha = [Float](repeating: 0, count: maxPuffs)
        self.opacity = [Float](repeating: 0, count: maxPuffs)
        self.kind = [Float](repeating: 0, count: maxPuffs)

        let dsDesc = MTLDepthStencilDescriptor()
        dsDesc.depthCompareFunction = .lessEqual
        dsDesc.isDepthWriteEnabled = false
        self.depthReadOnlyState = device.makeDepthStencilState(descriptor: dsDesc)!

        let puffBufLen = maxPuffs * MemoryLayout<ParticleVertexData>.stride
        for i in 0..<maxInFlight {
            if let buf = device.makeBuffer(length: puffBufLen, options: .storageModeShared) {
                buf.label = "ParticlePuffs_\(i)"
                puffBuffers.append(buf)
            }
        }

        var flakes: [SIMD4<Float>] = []
        flakes.reserveCapacity(maxFlakes)
        for _ in 0..<maxFlakes {
            let bx = (Float.random(in: 0..<1) - 0.5) * 44.0
            let by = (Float.random(in: 0..<1) - 0.5) * 26.0
            let bz = (Float.random(in: 0..<1) - 0.5) * 44.0
            let seed = Float.random(in: 0..<1)
            flakes.append(SIMD4<Float>(bx, by, bz, seed))
        }
        let flakeLen = flakes.count * MemoryLayout<SIMD4<Float>>.stride
        self.flakeBuffer = flakes.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: flakeLen, options: .storageModeShared)
        }
        self.flakeBuffer?.label = "AmbientFlakes"

        if !peaks.isEmpty {
            addRidgeSnowBanners(peaks: peaks)
        }
    }

    // MARK: - Summit Ridge Snow Banners

    public func addRidgeSnowBanners(peaks: [PeakInfo]) {
        let bannerPeaks = peaks.filter { $0.elev >= 3650 }
        guard !bannerPeaks.isEmpty else { return }

        // PlaneGeometry(1, 1, 6, 1) translated (+0.5, 0, 0) in XY plane
        let segX = 6
        var verts: [StandardVertex] = []
        var indices: [UInt32] = []
        for iy in 0...1 {
            let v = Float(iy)
            let y = v - 0.5
            for ix in 0...segX {
                let u = Float(ix) / Float(segX)
                let x = u
                verts.append(StandardVertex(
                    position: SIMD3<Float>(x, y, 0),
                    normal: SIMD3<Float>(0, 0, 1),
                    uv: SIMD2<Float>(u, v),
                    color: SIMD4<Float>(1, 1, 1, 1)
                ))
            }
        }
        let row = segX + 1
        for ix in 0..<segX {
            let a = UInt32(ix)
            let b = UInt32(ix + 1)
            let c = UInt32(row + ix)
            let d = UInt32(row + ix + 1)
            indices.append(contentsOf: [a, b, c, b, d, c])
        }
        self.bannerMesh = MeshPrimitives.upload(device: device, vertices: verts, indices: indices, label: "RidgeBannerPlane")

        let windDir = simd_normalize(SIMD3<Float>(0.86, 0.04, 0.51))
        var matrices: [simd_float4x4] = []
        for pk in bannerPeaks {
            let len: Float = pk.elev > 4400 ? 1150 : (pk.elev > 4000 ? 880 : 620)
            let height = len * 0.32
            for layer in 0..<2 {
                let pos = SIMD3<Float>(pk.x, pk.elev + 15 + Float(layer) * 18, pk.z)
                let yaw = -atan2f(windDir.z, windDir.x) + (layer == 1 ? 0.34 : 0.0)
                let pitch: Float = layer == 1 ? 0.24 : -0.10
                let q = quatFromEulerXYZ(x: pitch, y: yaw, z: 0.04)
                let mat = matrixTRS(translation: pos, rotation: q, scale: SIMD3<Float>(len, height, 1.0))
                matrices.append(mat)
            }
        }
        self.bannerInstanceCount = matrices.count
        let mLen = matrices.count * MemoryLayout<simd_float4x4>.stride
        self.bannerInstanceBuffer = matrices.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: mLen, options: .storageModeShared)
        }
        self.bannerInstanceBuffer?.label = "RidgeBannerInstances"
    }

    // MARK: - ParticlesSubsystem Emission

    public func emitPuff(
        x: Float, y: Float, z: Float,
        vx: Float, vy: Float, vz: Float,
        size: Float, expand: Float, peakAlpha: Float, life: Float, kind: Float
    ) {
        let i = cursor
        cursor = (cursor + 1) % maxPuffs
        posX[i] = x
        posY[i] = y
        posZ[i] = z
        velX[i] = vx
        velY[i] = vy
        velZ[i] = vz
        self.size[i] = size
        self.expand[i] = expand
        self.age[i] = 0
        self.life[i] = life
        self.rot[i] = Float.random(in: 0..<(.pi * 2))
        self.peakAlpha[i] = peakAlpha
        self.opacity[i] = peakAlpha * 0.25
        self.kind[i] = kind
    }

    public func emitBurst(
        x: Float, y: Float, z: Float,
        count: Int, speed: Float, sizeMin: Float, sizeMax: Float,
        expand: Float, peakAlpha: Float, life: Float, kind: Float
    ) {
        for _ in 0..<count {
            let a = Float.random(in: 0..<(.pi * 2))
            let r = (0.3 + Float.random(in: 0..<1) * 0.7) * speed
            emitPuff(
                x: x + (Float.random(in: 0..<1) - 0.5) * 1.1,
                y: y + Float.random(in: 0..<1) * 0.45,
                z: z + (Float.random(in: 0..<1) - 0.5) * 1.1,
                vx: cosf(a) * r,
                vy: 0.35 + Float.random(in: 0..<1) * min(2.2, speed * 0.30),
                vz: sinf(a) * r,
                size: sizeMin + Float.random(in: 0..<1) * (sizeMax - sizeMin),
                expand: expand,
                peakAlpha: peakAlpha,
                life: life * (0.72 + Float.random(in: 0..<1) * 0.55),
                kind: kind
            )
        }
    }

    public func emitCarveRoosterTail(
        x: Float, y: Float, z: Float,
        vx: Float, vy: Float, vz: Float,
        rate: Float, sluff: Bool, slopeX: Float, slopeZ: Float
    ) {
        _ = sluff
        _ = slopeX
        _ = slopeZ
        let intensity = min(2.2, rate)
        let n = intensity > 1.25 ? 2 : 1
        for _ in 0..<n {
            let puffSz = (0.55 + Float.random(in: 0..<1) * 0.55) * (0.76 + 0.32 * intensity)
            let puffLife = 0.95 + Float.random(in: 0..<1) * 0.72
            emitPuff(
                x: x + (Float.random(in: 0..<1) - 0.5) * 0.42,
                y: y + 0.08 + Float.random(in: 0..<1) * 0.18,
                z: z + (Float.random(in: 0..<1) - 0.5) * 0.42,
                vx: vx + (Float.random(in: 0..<1) - 0.5) * 1.8,
                vy: vy * 0.62 + Float.random(in: 0..<1) * 0.45,
                vz: vz + (Float.random(in: 0..<1) - 0.5) * 1.8,
                size: puffSz,
                expand: 0.82 + Float.random(in: 0..<1) * 0.35,
                peakAlpha: 0.30 + min(0.20, intensity * 0.10),
                life: puffLife,
                kind: 0
            )
        }
    }

    public func emitAvalancheWave(avalanche: AvalancheState, terrainData: any TerrainQueryProtocol, dt: Float) {
        guard avalanche.active, avalanche.strength >= 0.05 else { return }
        avyAccum += dt * (38.0 + avalanche.speed * 1.5)
        let count = min(18, Int(floorf(avyAccum)))
        guard count > 0 else { return }
        avyAccum -= Float(count)

        let sideX = -avalanche.dirZ
        let sideZ = avalanche.dirX
        for _ in 0..<count {
            let isFront = Float.random(in: 0..<1) < 0.72
            let along = isFront
                ? avalanche.frontDist - Float.random(in: 0..<1) * 10.0
                : max(2.0, avalanche.frontDist - 8.0 - Float.random(in: 0..<1) * min(avalanche.frontDist, 42.0))
            let across = (Float.random(in: 0..<1) - 0.5) * 2.0 * avalanche.width * (isFront ? 0.92 : 0.78)
            let wx = avalanche.originX + avalanche.dirX * along + sideX * across
            let wz = avalanche.originZ + avalanche.dirZ * along + sideZ * across
            let gy = terrainData.sampleHeight(x: wx, z: wz)

            let sz: Float = isFront ? (4.2 + Float.random(in: 0..<1) * 5.2) : (3.4 + Float.random(in: 0..<1) * 4.4)
            let spd = avalanche.speed * (isFront ? 0.75 : 0.35)
            emitPuff(
                x: wx,
                y: gy + (isFront ? 1.4 + Float.random(in: 0..<1) * 2.6 : 1.0 + Float.random(in: 0..<1) * 2.2),
                z: wz,
                vx: avalanche.dirX * spd + (Float.random(in: 0..<1) - 0.5) * 4.5,
                vy: (isFront ? 3.2 : 1.4) + Float.random(in: 0..<1) * 3.8,
                vz: avalanche.dirZ * spd + (Float.random(in: 0..<1) - 0.5) * 4.5,
                size: sz,
                expand: isFront ? 3.2 : 2.1,
                peakAlpha: (isFront ? 0.84 : 0.62) * avalanche.strength,
                life: isFront ? (2.2 + Float.random(in: 0..<1) * 1.6) : (2.6 + Float.random(in: 0..<1) * 1.6),
                kind: 1
            )
        }
    }

    public func clearAvalanchePuffs() {
        for i in 0..<maxPuffs {
            if kind[i] == 1 {
                life[i] = -1
                opacity[i] = 0
            }
        }
        avyAccum = 0
    }

    // MARK: - Update & Upload

    public func update(
        dt: Float,
        cameraPos: SIMD3<Float>,
        viewportHeightPx: Float,
        fovDeg: Float,
        preset: LightingPreset,
        sunDir: SIMD3<Float>,
        time: Float
    ) {
        _ = cameraPos
        _ = sunDir
        _ = time
        let halfFov = max(0.05, degToRad(fovDeg * 0.5))
        currentFocalScale = (viewportHeightPx * 0.5) / tanf(halfFov)
        currentWind = preset.isStorm
            ? SIMD3<Float>(16.5, -5.4, 7.8)
            : SIMD3<Float>(4.2, -1.5, 1.8)

        let windX: Float = preset.isStorm ? 9.5 : 3.4
        let windZ: Float = preset.isStorm ? 4.2 : 1.5

        for i in 0..<maxPuffs {
            if life[i] <= 0 {
                opacity[i] = 0
                continue
            }
            if dt > 0 {
                age[i] += dt
                let u = age[i] / life[i]
                if u >= 1.0 {
                    life[i] = -1
                    opacity[i] = 0
                    continue
                }
                let isAvy = kind[i] > 0.5
                let drag = expf(-dt * (isAvy ? 1.35 : 1.85))
                velX[i] = velX[i] * drag + windX * (1.0 - drag) * 0.45
                velY[i] = velY[i] * drag - (isAvy ? 1.4 : 2.3) * dt
                velZ[i] = velZ[i] * drag + windZ * (1.0 - drag) * 0.45
                posX[i] += velX[i] * dt
                posY[i] += velY[i] * dt
                posZ[i] += velZ[i] * dt
                size[i] += dt * expand[i] * (0.55 + (1.0 - u) * 0.85)
                let spinSign: Float = (i % 2 == 1) ? 1.0 : -1.0
                rot[i] += dt * (isAvy ? 0.72 : 0.45) * spinSign
                let env = sinf(min(1.0, u * 5.2) * .pi * 0.5) * powf(max(0.0, 1.0 - u), 1.38)
                opacity[i] = env * peakAlpha[i]
            } else {
                let u = min(0.99, age[i] / max(0.001, life[i]))
                let env = sinf(min(1.0, max(0.25, u * 5.2)) * .pi * 0.5) * powf(max(0.0, 1.0 - u), 1.38)
                opacity[i] = env * peakAlpha[i]
            }
        }

        // Pack active puffs into triple-buffered GPU staging buffer
        if !puffBuffers.isEmpty {
            puffBufferIndex = (puffBufferIndex + 1) % puffBuffers.count
            let buf = puffBuffers[puffBufferIndex]
            let ptr = buf.contents().bindMemory(to: ParticleVertexData.self, capacity: maxPuffs)
            var count = 0
            for i in 0..<maxPuffs {
                let op = opacity[i]
                if life[i] > 0 && op > 0.004 {
                    ptr[count] = ParticleVertexData(
                        posSize: SIMD4<Float>(posX[i], posY[i], posZ[i], size[i]),
                        colorAlpha: SIMD4<Float>(op, rot[i], kind[i], 0)
                    )
                    count += 1
                }
            }
            activePuffCount = count
        }
    }

    // MARK: - Pipelines

    private func configureTransparentAttachment(_ att: MTLRenderPipelineColorAttachmentDescriptor) {
        att.pixelFormat = .rgba16Float
        att.isBlendingEnabled = true
        att.rgbBlendOperation = .add
        att.sourceRGBBlendFactor = .sourceAlpha
        att.destinationRGBBlendFactor = .oneMinusSourceAlpha
        att.alphaBlendOperation = .add
        att.sourceAlphaBlendFactor = .zero
        att.destinationAlphaBlendFactor = .one
    }

    private func ensurePuffPipeline(msaa: Int) -> MTLRenderPipelineState? {
        if let p = puffPipelines[msaa] { return p }
        guard let vf = library.makeFunction(name: "puffVertex"),
              let ff = library.makeFunction(name: "puffFragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "ParticlePuffPipeline_\(msaa)"
        desc.vertexFunction = vf
        desc.fragmentFunction = ff
        desc.rasterSampleCount = msaa
        desc.depthAttachmentPixelFormat = .depth32Float
        configureTransparentAttachment(desc.colorAttachments[0]!)
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        if let pso { puffPipelines[msaa] = pso }
        return pso
    }

    private func ensureFlakePipeline(msaa: Int) -> MTLRenderPipelineState? {
        if let p = flakePipelines[msaa] { return p }
        guard let vf = library.makeFunction(name: "flakeVertex"),
              let ff = library.makeFunction(name: "flakeFragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "ParticleFlakePipeline_\(msaa)"
        desc.vertexFunction = vf
        desc.fragmentFunction = ff
        desc.rasterSampleCount = msaa
        desc.depthAttachmentPixelFormat = .depth32Float
        configureTransparentAttachment(desc.colorAttachments[0]!)
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        if let pso { flakePipelines[msaa] = pso }
        return pso
    }

    private func ensureBannerPipeline(msaa: Int) -> MTLRenderPipelineState? {
        if let p = bannerPipelines[msaa] { return p }
        guard let vf = library.makeFunction(name: "bannerVertex"),
              let ff = library.makeFunction(name: "bannerFragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "RidgeBannerPipeline_\(msaa)"
        desc.vertexFunction = vf
        desc.fragmentFunction = ff
        desc.vertexDescriptor = MeshPrimitives.standardVertexDescriptor()
        desc.rasterSampleCount = msaa
        desc.depthAttachmentPixelFormat = .depth32Float
        configureTransparentAttachment(desc.colorAttachments[0]!)
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        if let pso { bannerPipelines[msaa] = pso }
        return pso
    }

    // MARK: - HDR Pass

    public func renderHDRPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        let msaa = context.msaaSampleCount
        encoder.setDepthStencilState(depthReadOnlyState)
        encoder.setCullMode(.none)
        encoder.setFrontFacing(.counterClockwise)

        var pu = SIMD4<Float>(currentWind.x, currentWind.y, currentWind.z, currentFocalScale)

        // 1. Summit Ridge Snow Banners (renderOrder = 4)
        if bannerInstanceCount > 0,
           let bMesh = bannerMesh,
           let bInst = bannerInstanceBuffer,
           let bPso = ensureBannerPipeline(msaa: msaa) {
            encoder.setRenderPipelineState(bPso)
            encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setVertexBuffer(bMesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
            encoder.setVertexBuffer(bInst, offset: 0, index: BufferIndex.instances.rawValue)
            encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: bMesh.indexCount,
                indexType: .uint32,
                indexBuffer: bMesh.indexBuffer,
                indexBufferOffset: 0,
                instanceCount: bannerInstanceCount
            )
        }

        // 2. Volumetric Coldsmoke / Powder Puffs / Avalanche Clouds (renderOrder = 5)
        if activePuffCount > 0,
           !puffBuffers.isEmpty,
           let pPso = ensurePuffPipeline(msaa: msaa) {
            let buf = puffBuffers[puffBufferIndex]
            encoder.setRenderPipelineState(pPso)
            encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setVertexBuffer(buf, offset: 0, index: BufferIndex.instances.rawValue)
            encoder.setVertexBytes(&pu, length: MemoryLayout<SIMD4<Float>>.stride, index: BufferIndex.subsystemUniforms.rawValue)
            encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setFragmentTexture(puffTexture, index: TextureIndex.albedo.rawValue)
            encoder.drawPrimitives(
                type: .triangleStrip,
                vertexStart: 0,
                vertexCount: 4,
                instanceCount: activePuffCount
            )
        }

        // 3. Ambient Alpine Snow & Prismatic Diamond Dust (renderOrder = 6)
        if let fBuf = flakeBuffer,
           maxFlakes > 0,
           let fPso = ensureFlakePipeline(msaa: msaa) {
            encoder.setRenderPipelineState(fPso)
            encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setVertexBuffer(fBuf, offset: 0, index: BufferIndex.instances.rawValue)
            encoder.setVertexBytes(&pu, length: MemoryLayout<SIMD4<Float>>.stride, index: BufferIndex.subsystemUniforms.rawValue)
            encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.drawPrimitives(
                type: .triangleStrip,
                vertexStart: 0,
                vertexCount: 4,
                instanceCount: maxFlakes
            )
        }
    }
}
