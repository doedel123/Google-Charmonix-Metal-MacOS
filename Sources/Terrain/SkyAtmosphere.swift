import Foundation
import Metal
import simd

/// Analytic high-altitude Rayleigh + Mie sky dome, solar disc & corona,
/// orographic lenticular/cirrus clouds, and Chamonix valley sea-of-clouds (mer de nuages).
public final class SkyAtmosphere: SkySubsystem {
    private let device: MTLDevice
    private let skyMesh: ProceduralMesh
    private let cloudPlaneMesh: ProceduralMesh
    private let skyVertexFn: MTLFunction
    private let skyFragmentFn: MTLFunction
    private let cloudVertexFn: MTLFunction
    private let cloudFragmentFn: MTLFunction
    private let depthState: MTLDepthStencilState

    private var skyPipelines: [Int: MTLRenderPipelineState] = [:]
    private var cloudPipelines: [Int: MTLRenderPipelineState] = [:]

    public init(device: MTLDevice) throws {
        self.device = device

        let (skyVerts, skyIdx) = MeshPrimitives.makeSphere(radius: 42000.0, widthSegs: 48, heightSegs: 32)
        guard let skyM = MeshPrimitives.upload(device: device, vertices: skyVerts, indices: skyIdx, label: "SkyDome") else {
            throw TextureLoaderError.textureCreationFailed("SkyDome")
        }
        self.skyMesh = skyM

        let (cloudVerts, cloudIdx) = MeshPrimitives.makePlaneXZ(width: 52000.0, depth: 52000.0, segX: 1, segZ: 1)
        guard let cloudM = MeshPrimitives.upload(device: device, vertices: cloudVerts, indices: cloudIdx, label: "ValleyCloudPlane") else {
            throw TextureLoaderError.textureCreationFailed("ValleyCloudPlane")
        }
        self.cloudPlaneMesh = cloudM

        let lib = try ShaderLibrary.loadLibrary(device: device, named: "Sky")
        guard let sv = lib.makeFunction(name: "skyVertex"),
              let sf = lib.makeFunction(name: "skyFragment"),
              let cv = lib.makeFunction(name: "valleyCloudVertex"),
              let cf = lib.makeFunction(name: "valleyCloudFragment") else {
            throw ShaderLibraryError.compileFailed("Sky.metal", "Missing Sky shader entrypoints")
        }
        self.skyVertexFn = sv
        self.skyFragmentFn = sf
        self.cloudVertexFn = cv
        self.cloudFragmentFn = cf

        let dsDesc = MTLDepthStencilDescriptor()
        dsDesc.depthCompareFunction = .lessEqual
        dsDesc.isDepthWriteEnabled = false
        guard let ds = device.makeDepthStencilState(descriptor: dsDesc) else {
            throw TextureLoaderError.textureCreationFailed("SkyDepthState")
        }
        self.depthState = ds
    }

    public func update(cameraPos: SIMD3<Float>, preset: LightingPreset, sunDir: SIMD3<Float>, time: Float) {
        // Uniforms are populated in FrameUniforms by MetalRenderer.
        _ = (cameraPos, preset, sunDir, time)
    }

    private func skyPipeline(sampleCount: Int) throws -> MTLRenderPipelineState {
        if let existing = skyPipelines[sampleCount] { return existing }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "SkyDome_MSAA\(sampleCount)"
        desc.vertexFunction = skyVertexFn
        desc.fragmentFunction = skyFragmentFn
        desc.rasterSampleCount = sampleCount
        desc.colorAttachments[0].pixelFormat = .rgba16Float
        desc.depthAttachmentPixelFormat = .depth32Float
        let pso = try device.makeRenderPipelineState(descriptor: desc)
        skyPipelines[sampleCount] = pso
        return pso
    }

    private func cloudPipeline(sampleCount: Int) throws -> MTLRenderPipelineState {
        if let existing = cloudPipelines[sampleCount] { return existing }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "ValleyClouds_MSAA\(sampleCount)"
        desc.vertexFunction = cloudVertexFn
        desc.fragmentFunction = cloudFragmentFn
        desc.rasterSampleCount = sampleCount
        let ca = desc.colorAttachments[0]!
        ca.pixelFormat = .rgba16Float
        ca.isBlendingEnabled = true
        ca.rgbBlendOperation = .add
        ca.sourceRGBBlendFactor = .sourceAlpha
        ca.destinationRGBBlendFactor = .oneMinusSourceAlpha
        ca.alphaBlendOperation = .add
        ca.sourceAlphaBlendFactor = .zero
        ca.destinationAlphaBlendFactor = .one
        desc.depthAttachmentPixelFormat = .depth32Float
        let pso = try device.makeRenderPipelineState(descriptor: desc)
        cloudPipelines[sampleCount] = pso
        return pso
    }

    public func renderSkyPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        let samples = max(1, context.msaaSampleCount)
        guard let pso = try? skyPipeline(sampleCount: samples) else { return }

        encoder.pushDebugGroup("SkyDome")
        encoder.setRenderPipelineState(pso)
        encoder.setDepthStencilState(depthState)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setCullMode(.front) // Inside of sphere (matches THREE.BackSide)
        encoder.setVertexBuffer(skyMesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
        encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: skyMesh.indexCount,
            indexType: .uint32,
            indexBuffer: skyMesh.indexBuffer,
            indexBufferOffset: 0
        )
        encoder.popDebugGroup()
    }

    public func renderValleyCloudsPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        let samples = max(1, context.msaaSampleCount)
        guard let pso = try? cloudPipeline(sampleCount: samples) else { return }

        encoder.pushDebugGroup("ValleyClouds")
        encoder.setRenderPipelineState(pso)
        encoder.setDepthStencilState(depthState)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setCullMode(.none) // DoubleSide
        encoder.setVertexBuffer(cloudPlaneMesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
        encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: cloudPlaneMesh.indexCount,
            indexType: .uint32,
            indexBuffer: cloudPlaneMesh.indexBuffer,
            indexBufferOffset: 0
        )
        encoder.popDebugGroup()
    }
}
