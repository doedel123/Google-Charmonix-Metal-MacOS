import Foundation
import Metal
import simd

// MARK: - Swift Mirrors of Rig-Specific Metal Structs

private struct RigMaterialParamsCPU {
    var uvScale: SIMD2<Float>
    var bumpScale: Float
    var hasRoughnessMap: Float
    var opacity: Float
    var isDropShadow: Float
    var pad0: Float = 0
    var pad1: Float = 0
}

private struct BowWaveUniformsCPU {
    var modelMatrix: simd_float4x4
    var normalMatrix: simd_float4x4
    var params0: SIMD4<Float>     // x = flowDist, y = intensity, z = nestAmt, w = edgeBias
    var params1: SIMD4<Float>     // x = skiRoll, y = sideSign, z = storm, w = sunVis
    var snowLitCol: SIMD4<Float>
    var snowShadCol: SIMD4<Float>
    var snowSSSCol: SIMD4<Float>
}

private struct ShadowRibbonUniformsCPU {
    var tintOpacity: SIMD4<Float>
    var emergeT: Float
    var pad0: Float = 0
    var pad1: Float = 0
    var pad2: Float = 0
}

private struct ShadowCatcherUniformsCPU {
    var tintStrength: SIMD4<Float>
}

private struct RigHDRPipelines {
    let pbrOpaque: MTLRenderPipelineState
    let pbrSkiPowder: MTLRenderPipelineState
    let pbrDust: MTLRenderPipelineState
    let dropShadow: MTLRenderPipelineState
    let bowWave: MTLRenderPipelineState
    let shadowRibbon: MTLRenderPipelineState
    let shadowCatcher: MTLRenderPipelineState
}

// MARK: - SkiRigRenderer (implements RigSubsystem)

public final class SkiRigRenderer: RigSubsystem {
    private let device: MTLDevice
    private let shaderLibrary: MTLLibrary
    private let textures: RigTextureBank
    private let rigScene: SkierRigScene
    private let animator: SkiRigAnimator

    private let shadowPipeline: MTLRenderPipelineState
    private var hdrPipelinesBySamples: [Int: RigHDRPipelines] = [:]

    private let depthReadWriteState: MTLDepthStencilState
    private let depthReadOnlyState: MTLDepthStencilState
    private let shadowDepthState: MTLDepthStencilState

    public var lastDiag: SkiRigDiagnostics { animator.lastDiag }

    public init(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        terrainData: any TerrainQueryProtocol
    ) throws {
        self.device = device
        self.shaderLibrary = try ShaderLibrary.loadLibrary(device: device, named: "StandardPBR")

        guard let texBank = buildRigTextureBank(device: device, commandQueue: commandQueue) else {
            throw TextureLoaderError.textureCreationFailed("RigTextureBank")
        }
        self.textures = texBank

        guard let scene = SkierRigScene(device: device, skiTopsheetTex: terrainData.textures.skiTopsheet, textures: texBank) else {
            throw TextureLoaderError.textureCreationFailed("SkierRigScene")
        }
        self.rigScene = scene
        self.animator = SkiRigAnimator(rig: scene, terrainData: terrainData)

        // Depth stencil states
        let rwDesc = MTLDepthStencilDescriptor()
        rwDesc.depthCompareFunction = .lessEqual
        rwDesc.isDepthWriteEnabled = true
        guard let rwState = device.makeDepthStencilState(descriptor: rwDesc) else {
            throw ShaderLibraryError.compileFailed("StandardPBR.metal", "Failed to create depthReadWriteState")
        }
        self.depthReadWriteState = rwState

        let roDesc = MTLDepthStencilDescriptor()
        roDesc.depthCompareFunction = .lessEqual
        roDesc.isDepthWriteEnabled = false
        guard let roState = device.makeDepthStencilState(descriptor: roDesc) else {
            throw ShaderLibraryError.compileFailed("StandardPBR.metal", "Failed to create depthReadOnlyState")
        }
        self.depthReadOnlyState = roState

        let shDesc = MTLDepthStencilDescriptor()
        shDesc.depthCompareFunction = .lessEqual
        shDesc.isDepthWriteEnabled = true
        guard let shState = device.makeDepthStencilState(descriptor: shDesc) else {
            throw ShaderLibraryError.compileFailed("StandardPBR.metal", "Failed to create shadowDepthState")
        }
        self.shadowDepthState = shState

        // Directional Shadow Map Pipeline
        guard let shVert = shaderLibrary.makeFunction(name: "rig_shadow_vertex"),
              let shFrag = shaderLibrary.makeFunction(name: "rig_shadow_fragment") else {
            throw ShaderLibraryError.compileFailed("StandardPBR.metal", "Missing shadow functions")
        }
        let shPipeDesc = MTLRenderPipelineDescriptor()
        shPipeDesc.label = "RigShadowPipeline"
        shPipeDesc.vertexFunction = shVert
        shPipeDesc.fragmentFunction = shFrag
        shPipeDesc.vertexDescriptor = MeshPrimitives.standardVertexDescriptor()
        shPipeDesc.colorAttachments[0].pixelFormat = .invalid
        shPipeDesc.depthAttachmentPixelFormat = .depth32Float
        shPipeDesc.rasterSampleCount = 1
        self.shadowPipeline = try device.makeRenderPipelineState(descriptor: shPipeDesc)

        // Pre-warm default 4x MSAA HDR pipelines
        _ = try pipelines(forSampleCount: 4)
    }

    // MARK: - RigSubsystem Protocol

    public func update(
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
        animator.update(
            state: state,
            dt: dt,
            isThirdPerson: isThirdPerson,
            time: time,
            sunDir: sunDir,
            isStorm: isStorm,
            lookYawOff: lookYawOff,
            mtnSunVis: mtnSunVis,
            isAlpenglow: isAlpenglow,
            isChesty: isChesty,
            preset: preset
        )
    }

    public func renderShadowPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        encoder.pushDebugGroup("SkierRig_ShadowPass")
        encoder.setRenderPipelineState(shadowPipeline)
        encoder.setDepthStencilState(shadowDepthState)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)

        drawShadowNode(rigScene.rootNode, encoder: encoder)
        encoder.popDebugGroup()
    }

    public func renderGroundShadowsPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        guard let pipes = try? pipelines(forSampleCount: context.msaaSampleCount) else { return }
        encoder.pushDebugGroup("SkierRig_GroundShadows")
        encoder.setFrontFacing(.counterClockwise)
        encoder.setCullMode(.none)
        encoder.setDepthStencilState(depthReadOnlyState)
        encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)

        // 1. DEM-snapped multiplicative PCF Shadow Catcher
        if rigScene.groundCatcher.visible, let shadowTex = context.shadowMapTexture {
            encoder.setRenderPipelineState(pipes.shadowCatcher)
            encoder.setDepthBias(-3.0, slopeScale: -4.0, clamp: 0.0)
            encoder.setVertexBuffer(rigScene.groundCatcher.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
            var u = ShadowCatcherUniformsCPU(
                tintStrength: SIMD4<Float>(rigScene.groundCatcher.tint, rigScene.groundCatcher.strength)
            )
            encoder.setFragmentBytes(&u, length: MemoryLayout<ShadowCatcherUniformsCPU>.stride, index: BufferIndex.modelUniforms.rawValue)
            encoder.setFragmentTexture(shadowTex, index: TextureIndex.shadowMap.rawValue)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: rigScene.groundCatcher.indices.count,
                indexType: .uint32,
                indexBuffer: rigScene.groundCatcher.indexBuffer,
                indexBufferOffset: 0
            )
        }

        // 2. Per-Ski Multiplicative Hourglass Contact Shadow Ribbons
        for rib in [rigScene.leftShadowRib, rigScene.rightShadowRib] where rib.visible {
            encoder.setRenderPipelineState(pipes.shadowRibbon)
            encoder.setDepthBias(-3.0, slopeScale: -3.0, clamp: 0.0)
            encoder.setVertexBuffer(rib.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
            var u = ShadowRibbonUniformsCPU(
                tintOpacity: SIMD4<Float>(rib.tint, rib.opacity),
                emergeT: rib.emergeT
            )
            encoder.setFragmentBytes(&u, length: MemoryLayout<ShadowRibbonUniformsCPU>.stride, index: BufferIndex.modelUniforms.rawValue)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: rib.indexCount,
                indexType: .uint32,
                indexBuffer: rib.indexBuffer,
                indexBufferOffset: 0
            )
        }

        // 3. Airborne Drop Shadow Blob
        if rigScene.groundShadowNode.visible, let part = rigScene.groundShadowNode.parts.first {
            encoder.setRenderPipelineState(pipes.dropShadow)
            encoder.setDepthBias(-3.0, slopeScale: -3.0, clamp: 0.0)
            drawPBRPart(part, worldMatrix: rigScene.groundShadowNode.worldMatrix, encoder: encoder, context: context)
        }

        encoder.setDepthBias(0, slopeScale: 0, clamp: 0)
        encoder.popDebugGroup()
    }

    public func renderHDRPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        guard let pipes = try? pipelines(forSampleCount: context.msaaSampleCount) else { return }
        encoder.pushDebugGroup("SkierRig_HDR")
        encoder.setFrontFacing(.counterClockwise)
        encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        if let shadowTex = context.shadowMapTexture {
            encoder.setFragmentTexture(shadowTex, index: TextureIndex.shadowMap.rawValue)
        }

        // 1. Opaque foreground gear & skier body (writes alpha = 0.0)
        encoder.setRenderPipelineState(pipes.pbrOpaque)
        encoder.setDepthStencilState(depthReadWriteState)
        drawOpaqueNodes(rigScene.rootNode, encoder: encoder, context: context)

        // 2. Parabolic Skis (markSkiSurface powder optical dissolve)
        encoder.setRenderPipelineState(pipes.pbrSkiPowder)
        for ski in [rigScene.leftSki, rigScene.rightSki] where ski.bladePart.visible {
            encoder.setDepthStencilState(ski.bladePart.material.depthWrite ? depthReadWriteState : depthReadOnlyState)
            drawPBRPart(ski.bladePart, worldMatrix: ski.node.worldMatrix, encoder: encoder, context: context, ski: ski)
        }

        // 3. Front Shovel Snow Dust Clumps
        encoder.setRenderPipelineState(pipes.pbrDust)
        encoder.setDepthStencilState(depthReadOnlyState)
        for ski in [rigScene.leftSki, rigScene.rightSki] where ski.dustPart.visible {
            drawPBRPart(ski.dustPart, worldMatrix: ski.node.worldMatrix, encoder: encoder, context: context)
        }

        // 4. 3D Dynamic Powder Bow-Wave Snow Wash
        encoder.setRenderPipelineState(pipes.bowWave)
        encoder.setDepthStencilState(depthReadOnlyState)
        encoder.setCullMode(.none)
        for ski in [rigScene.leftSki, rigScene.rightSki] where ski.bowVisible {
            var bu = BowWaveUniformsCPU(
                modelMatrix: ski.node.worldMatrix,
                normalMatrix: normalMatrix(from: ski.node.worldMatrix),
                params0: SIMD4<Float>(ski.flowDist, ski.bowIntensity, 0.0, ski.bowEdgeBias),
                params1: SIMD4<Float>(ski.bowSkiRoll, ski.isLeft ? -1.0 : 1.0, ski.powStorm, ski.bowSunVis),
                snowLitCol: SIMD4<Float>(ski.snowLitCol, 1.0),
                snowShadCol: SIMD4<Float>(ski.snowShadCol, 1.0),
                snowSSSCol: SIMD4<Float>(ski.snowSSSCol, 1.0)
            )
            encoder.setVertexBuffer(ski.bowVertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
            encoder.setVertexBytes(&bu, length: MemoryLayout<BowWaveUniformsCPU>.stride, index: BufferIndex.modelUniforms.rawValue)
            encoder.setFragmentBytes(&bu, length: MemoryLayout<BowWaveUniformsCPU>.stride, index: BufferIndex.modelUniforms.rawValue)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: ski.bowIndexCount,
                indexType: .uint32,
                indexBuffer: ski.bowIndexBuffer,
                indexBufferOffset: 0
            )
        }

        encoder.popDebugGroup()
    }

    // MARK: - Traversal Helpers

    private func drawShadowNode(_ node: RigNode, encoder: MTLRenderCommandEncoder) {
        guard node.visible else { return }
        for part in node.parts where part.visible && part.castShadow {
            encoder.setCullMode(part.material.isDoubleSided ? .none : .back)
            var mu = ModelUniforms()
            mu.modelMatrix = node.worldMatrix
            mu.normalMatrix = normalMatrix(from: node.worldMatrix)
            encoder.setVertexBuffer(part.mesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
            encoder.setVertexBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: part.mesh.indexCount,
                indexType: .uint32,
                indexBuffer: part.mesh.indexBuffer,
                indexBufferOffset: 0
            )
        }
        for child in node.children {
            drawShadowNode(child, encoder: encoder)
        }
    }

    private func drawOpaqueNodes(_ node: RigNode, encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        guard node.visible else { return }
        for part in node.parts where part.visible && !part.shadowOnly && !part.material.isSki && !part.material.isDropShadow && part.material.opacity > 0.99 {
            drawPBRPart(part, worldMatrix: node.worldMatrix, encoder: encoder, context: context)
        }
        for child in node.children {
            drawOpaqueNodes(child, encoder: encoder, context: context)
        }
    }

    private func drawPBRPart(
        _ part: RigMeshPart,
        worldMatrix: simd_float4x4,
        encoder: MTLRenderCommandEncoder,
        context: RenderPassContext,
        ski: SkiAssembly? = nil
    ) {
        let mat = part.material
        encoder.setCullMode(mat.isDoubleSided ? .none : .back)

        var mu = ModelUniforms()
        mu.modelMatrix = worldMatrix
        mu.normalMatrix = normalMatrix(from: worldMatrix)
        mu.baseColor = SIMD4<Float>(mat.baseColor, 1.0)
        mu.pbrParams = SIMD4<Float>(mat.roughness, mat.metalness, mat.emissive, 0.0) // alphaMask = 0.0 (sharp foreground!)
        mu.fgParams = SIMD4<Float>(mat.frostScale, mat.sheenAmt, animator.fgBounceAmt, animator.fgFrostGlobal)
        mu.fgBounceColor = SIMD4<Float>(animator.fgBounceCol, mat.isSki ? 1.0 : 0.0)
        mu.fgRimColor = SIMD4<Float>(animator.fgRimCol, (ski?.isLeft ?? true) ? -1.0 : 1.0)
        if let ski {
            mu.skiPowderParams = SIMD4<Float>(ski.powSink, ski.edgeRoll, ski.flowDist, ski.powSunVis)
            mu.snowLitColor = SIMD4<Float>(ski.snowLitCol, ski.powStorm)
            mu.snowShadowColor = SIMD4<Float>(ski.snowShadCol, mat.albedoTex != nil ? 1.0 : 0.0)
            mu.snowSSSColor = SIMD4<Float>(ski.snowSSSCol, mat.bumpTex != nil ? 1.0 : 0.0)
        } else {
            mu.snowShadowColor = SIMD4<Float>(0, 0, 0, mat.albedoTex != nil ? 1.0 : 0.0)
            mu.snowSSSColor = SIMD4<Float>(0, 0, 0, mat.bumpTex != nil ? 1.0 : 0.0)
        }

        var mp = RigMaterialParamsCPU(
            uvScale: mat.uvScale,
            bumpScale: mat.bumpScale,
            hasRoughnessMap: mat.roughTex != nil ? 1.0 : 0.0,
            opacity: mat.opacity,
            isDropShadow: mat.isDropShadow ? 1.0 : 0.0
        )

        encoder.setVertexBuffer(part.mesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
        encoder.setVertexBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
        encoder.setFragmentBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
        encoder.setFragmentBytes(&mp, length: MemoryLayout<RigMaterialParamsCPU>.stride, index: BufferIndex.materialUniforms.rawValue)

        encoder.setFragmentTexture(mat.albedoTex ?? textures.fallbackWhite, index: TextureIndex.albedo.rawValue)
        encoder.setFragmentTexture(mat.bumpTex ?? textures.fallbackWhite, index: TextureIndex.normal.rawValue)
        encoder.setFragmentTexture(mat.roughTex ?? textures.fallbackWhite, index: TextureIndex.roughness.rawValue)

        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: part.mesh.indexCount,
            indexType: .uint32,
            indexBuffer: part.mesh.indexBuffer,
            indexBufferOffset: 0
        )
    }

    // MARK: - Pipeline Cache per MSAA Sample Count

    private func pipelines(forSampleCount sampleCount: Int) throws -> RigHDRPipelines {
        let count = max(1, sampleCount)
        if let existing = hdrPipelinesBySamples[count] {
            return existing
        }

        let pbrVert = try makeFn("rig_pbr_vertex")
        let pbrFrag = try makeFn("rig_pbr_fragment")
        let bowVert = try makeFn("bow_wave_vertex")
        let bowFrag = try makeFn("bow_wave_fragment")
        let ribVert = try makeFn("shadow_ribbon_vertex")
        let ribFrag = try makeFn("shadow_ribbon_fragment")
        let catchVert = try makeFn("shadow_catcher_vertex")
        let catchFrag = try makeFn("shadow_catcher_fragment")

        let stdVD = MeshPrimitives.standardVertexDescriptor()

        let bowVD = MTLVertexDescriptor()
        bowVD.attributes[VertexAttribute.position.rawValue].format = .float3
        bowVD.attributes[VertexAttribute.position.rawValue].offset = MemoryLayout<BowWaveVertex>.offset(of: \.position) ?? 0
        bowVD.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        bowVD.attributes[VertexAttribute.normal.rawValue].format = .float3
        bowVD.attributes[VertexAttribute.normal.rawValue].offset = MemoryLayout<BowWaveVertex>.offset(of: \.normal) ?? 16
        bowVD.attributes[VertexAttribute.normal.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        bowVD.attributes[VertexAttribute.UV.rawValue].format = .float2
        bowVD.attributes[VertexAttribute.UV.rawValue].offset = MemoryLayout<BowWaveVertex>.offset(of: \.uv) ?? 32
        bowVD.attributes[VertexAttribute.UV.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        bowVD.attributes[VertexAttribute.custom0.rawValue].format = .float4
        bowVD.attributes[VertexAttribute.custom0.rawValue].offset = MemoryLayout<BowWaveVertex>.offset(of: \.bowData) ?? 48
        bowVD.attributes[VertexAttribute.custom0.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        bowVD.layouts[BufferIndex.vertices.rawValue].stride = MemoryLayout<BowWaveVertex>.stride

        let ribVD = MTLVertexDescriptor()
        ribVD.attributes[VertexAttribute.position.rawValue].format = .float3
        ribVD.attributes[VertexAttribute.position.rawValue].offset = MemoryLayout<ShadowRibbonVertex>.offset(of: \.position) ?? 0
        ribVD.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        ribVD.attributes[VertexAttribute.UV.rawValue].format = .float2
        ribVD.attributes[VertexAttribute.UV.rawValue].offset = MemoryLayout<ShadowRibbonVertex>.offset(of: \.uv) ?? 16
        ribVD.attributes[VertexAttribute.UV.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        ribVD.attributes[VertexAttribute.custom0.rawValue].format = .float4
        ribVD.attributes[VertexAttribute.custom0.rawValue].offset = MemoryLayout<ShadowRibbonVertex>.offset(of: \.custom0) ?? 32
        ribVD.attributes[VertexAttribute.custom0.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        ribVD.layouts[BufferIndex.vertices.rawValue].stride = MemoryLayout<ShadowRibbonVertex>.stride

        let catchVD = MTLVertexDescriptor()
        catchVD.attributes[VertexAttribute.position.rawValue].format = .float3
        catchVD.attributes[VertexAttribute.position.rawValue].offset = MemoryLayout<ShadowCatcherVertex>.offset(of: \.position) ?? 0
        catchVD.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        catchVD.attributes[VertexAttribute.normal.rawValue].format = .float3
        catchVD.attributes[VertexAttribute.normal.rawValue].offset = MemoryLayout<ShadowCatcherVertex>.offset(of: \.normal) ?? 16
        catchVD.attributes[VertexAttribute.normal.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        catchVD.attributes[VertexAttribute.UV.rawValue].format = .float2
        catchVD.attributes[VertexAttribute.UV.rawValue].offset = MemoryLayout<ShadowCatcherVertex>.offset(of: \.catchUv) ?? 32
        catchVD.attributes[VertexAttribute.UV.rawValue].bufferIndex = BufferIndex.vertices.rawValue
        catchVD.layouts[BufferIndex.vertices.rawValue].stride = MemoryLayout<ShadowCatcherVertex>.stride

        func makePipe(
            label: String,
            vert: MTLFunction,
            frag: MTLFunction,
            vd: MTLVertexDescriptor,
            blend: (MTLRenderPipelineColorAttachmentDescriptor) -> Void
        ) throws -> MTLRenderPipelineState {
            let desc = MTLRenderPipelineDescriptor()
            desc.label = "\(label)_MSAA\(count)"
            desc.vertexFunction = vert
            desc.fragmentFunction = frag
            desc.vertexDescriptor = vd
            desc.colorAttachments[0].pixelFormat = .rgba16Float
            desc.depthAttachmentPixelFormat = .depth32Float
            desc.rasterSampleCount = count
            blend(desc.colorAttachments[0])
            return try device.makeRenderPipelineState(descriptor: desc)
        }

        let pbrOpaque = try makePipe(label: "RigPBROpaque", vert: pbrVert, frag: pbrFrag, vd: stdVD) { att in
            att.isBlendingEnabled = false
        }
        let pbrSkiPowder = try makePipe(label: "RigSkiPowder", vert: pbrVert, frag: pbrFrag, vd: stdVD) { att in
            att.isBlendingEnabled = true
            att.rgbBlendOperation = .add
            att.sourceRGBBlendFactor = .sourceAlpha
            att.destinationRGBBlendFactor = .oneMinusSourceAlpha
            att.alphaBlendOperation = .add
            att.sourceAlphaBlendFactor = .zero
            att.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        }
        let alphaPreserveBlend: (MTLRenderPipelineColorAttachmentDescriptor) -> Void = { att in
            att.isBlendingEnabled = true
            att.rgbBlendOperation = .add
            att.sourceRGBBlendFactor = .sourceAlpha
            att.destinationRGBBlendFactor = .oneMinusSourceAlpha
            att.alphaBlendOperation = .add
            att.sourceAlphaBlendFactor = .zero
            att.destinationAlphaBlendFactor = .one
        }
        let multPreserveBlend: (MTLRenderPipelineColorAttachmentDescriptor) -> Void = { att in
            att.isBlendingEnabled = true
            att.rgbBlendOperation = .add
            att.sourceRGBBlendFactor = .destinationColor
            att.destinationRGBBlendFactor = .zero
            att.alphaBlendOperation = .add
            att.sourceAlphaBlendFactor = .zero
            att.destinationAlphaBlendFactor = .one
        }

        let pbrDust = try makePipe(label: "RigSkiDust", vert: pbrVert, frag: pbrFrag, vd: stdVD, blend: alphaPreserveBlend)
        let dropShadow = try makePipe(label: "RigDropShadow", vert: pbrVert, frag: pbrFrag, vd: stdVD, blend: alphaPreserveBlend)
        let bowWave = try makePipe(label: "RigBowWave", vert: bowVert, frag: bowFrag, vd: bowVD, blend: alphaPreserveBlend)
        let shadowRibbon = try makePipe(label: "RigShadowRibbon", vert: ribVert, frag: ribFrag, vd: ribVD, blend: multPreserveBlend)
        let shadowCatcher = try makePipe(label: "RigShadowCatcher", vert: catchVert, frag: catchFrag, vd: catchVD, blend: multPreserveBlend)

        let bundle = RigHDRPipelines(
            pbrOpaque: pbrOpaque,
            pbrSkiPowder: pbrSkiPowder,
            pbrDust: pbrDust,
            dropShadow: dropShadow,
            bowWave: bowWave,
            shadowRibbon: shadowRibbon,
            shadowCatcher: shadowCatcher
        )
        hdrPipelinesBySamples[count] = bundle
        return bundle
    }

    private func makeFn(_ name: String) throws -> MTLFunction {
        guard let fn = shaderLibrary.makeFunction(name: name) else {
            throw ShaderLibraryError.compileFailed("StandardPBR.metal", "Missing function '\(name)'")
        }
        return fn
    }
}
