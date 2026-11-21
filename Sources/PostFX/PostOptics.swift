import Foundation
import Metal
import simd

/// HDR Photorealistic Optical Post-Processing Pipeline:
///  - 4x MSAA on RGBA16Float HDR render target
///  - Half-res separable Gaussian bloom / halation
///  - Half-res volumetric screen-space god-rays
///  - Directional FXAA, velocity-coupled ground-rush & angular motion blur (foreground alpha=0.0 preserved sharp)
///  - Action-cam barrel distortion, corner CA, sun flare + 3 lens ghosts, snow droplets, breath fog, ACES filmic tone map
public final class PostOpticsPipeline: PostOpticsSubsystem {
    private let device: MTLDevice
    private let quadMesh: ProceduralMesh

    private let brightPipeline: MTLRenderPipelineState
    private let blurVPipeline: MTLRenderPipelineState
    private let godRayPipeline: MTLRenderPipelineState
    private var compositePipeline: MTLRenderPipelineState
    private var compositePixelFormat: MTLPixelFormat = .bgra8Unorm

    private var width: Int
    private var height: Int
    private var requestedSamples: Int = 4

    public private(set) var msaaSampleCount: Int = 4
    public private(set) var sceneMSAATexture: MTLTexture?
    public private(set) var sceneHDRTexture: MTLTexture
    public private(set) var sceneDepthTexture: MTLTexture

    private var bloomATexture: MTLTexture
    private var bloomBTexture: MTLTexture
    private var godRayTexture: MTLTexture

    public init(device: MTLDevice, width: Int = 1280, height: Int = 720, msaaSamples: Int = 4) throws {
        self.device = device
        self.width = max(1, width)
        self.height = max(1, height)

        guard let quad = MeshPrimitives.makeFullscreenQuad(device: device) else {
            throw ShaderLibraryError.compileFailed("PostOptics.metal", "Failed to build fullscreen quad")
        }
        self.quadMesh = quad

        let lib = try ShaderLibrary.loadLibrary(device: device, named: "PostOptics")
        guard let vFn = lib.makeFunction(name: "postOpticsVertex"),
              let brightFn = lib.makeFunction(name: "postBrightHBlurFragment"),
              let blurVFn = lib.makeFunction(name: "postBlurVFragment"),
              let godRayFn = lib.makeFunction(name: "postGodRaysFragment"),
              let compFn = lib.makeFunction(name: "postCompositeFragment") else {
            throw ShaderLibraryError.compileFailed("PostOptics.metal", "Missing post-optics shader function(s)")
        }

        let vd = MeshPrimitives.standardVertexDescriptor()
        self.brightPipeline = try Self.makePipeline(device: device, vertexFn: vFn, fragmentFn: brightFn, vertexDescriptor: vd, pixelFormat: .rgba16Float, label: "PostBrightHBlur")
        self.blurVPipeline = try Self.makePipeline(device: device, vertexFn: vFn, fragmentFn: blurVFn, vertexDescriptor: vd, pixelFormat: .rgba16Float, label: "PostBlurV")
        self.godRayPipeline = try Self.makePipeline(device: device, vertexFn: vFn, fragmentFn: godRayFn, vertexDescriptor: vd, pixelFormat: .rgba16Float, label: "PostGodRays")
        self.compositePipeline = try Self.makePipeline(device: device, vertexFn: vFn, fragmentFn: compFn, vertexDescriptor: vd, pixelFormat: .bgra8Unorm, label: "PostComposite")

        let clampedSamples = Self.supportedSampleCount(device: device, requested: msaaSamples)
        self.requestedSamples = msaaSamples
        self.msaaSampleCount = clampedSamples

        let targets = Self.createTargets(device: device, width: self.width, height: self.height, sampleCount: clampedSamples)
        self.sceneMSAATexture = targets.msaa
        self.sceneHDRTexture = targets.hdr
        self.sceneDepthTexture = targets.depth
        self.bloomATexture = targets.bloomA
        self.bloomBTexture = targets.bloomB
        self.godRayTexture = targets.godRay
    }

    public func setMSAA(samples: Int) {
        requestedSamples = samples
        let target = Self.supportedSampleCount(device: device, requested: samples)
        guard target != msaaSampleCount else { return }
        msaaSampleCount = target
        rebuildTargets()
    }

    public func resize(width: Int, height: Int) {
        let w = max(1, width)
        let h = max(1, height)
        guard w != self.width || h != self.height else { return }
        self.width = w
        self.height = h
        rebuildTargets()
    }

    public func render(
        commandBuffer: MTLCommandBuffer,
        drawableTexture: MTLTexture,
        params: PostOpticsParams,
        context: RenderPassContext
    ) {
        let dw = max(1, drawableTexture.width)
        let dh = max(1, drawableTexture.height)
        if dw != width || dh != height {
            resize(width: dw, height: dh)
        }

        if drawableTexture.pixelFormat != compositePixelFormat {
            if let lib = try? ShaderLibrary.loadLibrary(device: device, named: "PostOptics"),
               let vFn = lib.makeFunction(name: "postOpticsVertex"),
               let compFn = lib.makeFunction(name: "postCompositeFragment"),
               let rebuilt = try? Self.makePipeline(
                device: device,
                vertexFn: vFn,
                fragmentFn: compFn,
                vertexDescriptor: MeshPrimitives.standardVertexDescriptor(),
                pixelFormat: drawableTexture.pixelFormat,
                label: "PostComposite"
               ) {
                compositePipeline = rebuilt
                compositePixelFormat = drawableTexture.pixelFormat
            }
        }

        let halfW = max(1, width >> 1)
        let halfH = max(1, height >> 1)

        let useBloom = !params.skipBloom && params.bloomStrength > 0.005
        let runGodRays = !params.skipGodRays && params.godRayVis > 0.005 && params.godRayStrength > 0.005

        var uniforms = PostOpticsUniforms(
            texelSize: SIMD2<Float>(1.0 / Float(width), 1.0 / Float(height)),
            sunUV: params.sunUV,
            horizonUV: params.horizonUV,
            angVel: params.disableMotionBlur ? .zero : params.angVel,
            sunColor: SIMD4<Float>(params.sunColor, params.sunVis),
            exposure: params.exposure,
            bloomStrength: useBloom ? params.bloomStrength : 0.0,
            barrel: params.barrel,
            speedFactor: params.disableMotionBlur ? 0.0 : params.speedFactor,
            rollVel: params.disableMotionBlur ? 0.0 : params.rollVel,
            godRayVis: params.godRayVis,
            godRayStrength: params.godRayStrength,
            hasGodRays: runGodRays ? 1.0 : 0.0,
            faceShot: params.faceShot,
            lensWet: params.lensWet,
            breath: params.breath,
            stormStreaks: params.stormStreaks,
            time: params.time,
            aspect: params.aspect,
            threshold: 1.18,
            pad0: 0.0
        )

        if useBloom {
            // Pass 1: Bright-extract + Horizontal Gaussian Blur -> bloomA (half-res)
            encodeQuadPass(
                commandBuffer: commandBuffer,
                targetTexture: bloomATexture,
                pipeline: brightPipeline,
                label: "PostBrightHBlur",
                uniforms: &uniforms,
                textures: [(sceneHDRTexture, TextureIndex.sceneHDR.rawValue)]
            )

            // Pass 2: Vertical Gaussian Blur -> bloomB (half-res)
            var blurUniforms = uniforms
            blurUniforms.texelSize = SIMD2<Float>(1.0 / Float(halfW), 1.0 / Float(halfH))
            encodeQuadPass(
                commandBuffer: commandBuffer,
                targetTexture: bloomBTexture,
                pipeline: blurVPipeline,
                label: "PostBlurV",
                uniforms: &blurUniforms,
                textures: [(bloomATexture, TextureIndex.bloom.rawValue)]
            )
        }

        if runGodRays {
            // Pass 3: Half-res Screen-Space Volumetric God-Rays -> godRayTexture
            encodeQuadPass(
                commandBuffer: commandBuffer,
                targetTexture: godRayTexture,
                pipeline: godRayPipeline,
                label: "PostGodRays",
                uniforms: &uniforms,
                textures: [(sceneHDRTexture, TextureIndex.sceneHDR.rawValue)]
            )
        }

        // Pass 4: Final Composite (FXAA, motion blur, CA, bloom, god-rays, flare, ACES, gamma, dither) -> drawable
        encodeQuadPass(
            commandBuffer: commandBuffer,
            targetTexture: drawableTexture,
            pipeline: compositePipeline,
            label: "PostComposite",
            uniforms: &uniforms,
            textures: [
                (sceneHDRTexture, TextureIndex.sceneHDR.rawValue),
                (bloomBTexture, TextureIndex.bloom.rawValue),
                (godRayTexture, TextureIndex.godRays.rawValue),
            ]
        )

        _ = context
    }

    // MARK: - Private Helpers

    private func encodeQuadPass(
        commandBuffer: MTLCommandBuffer,
        targetTexture: MTLTexture,
        pipeline: MTLRenderPipelineState,
        label: String,
        uniforms: inout PostOpticsUniforms,
        textures: [(MTLTexture, Int)]
    ) {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = targetTexture
        rp.colorAttachments[0].loadAction = .dontCare
        rp.colorAttachments[0].storeAction = .store

        guard let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rp) else { return }
        enc.label = label
        enc.setRenderPipelineState(pipeline)
        enc.setViewport(MTLViewport(
            originX: 0, originY: 0,
            width: Double(targetTexture.width),
            height: Double(targetTexture.height),
            znear: 0, zfar: 1
        ))
        enc.setVertexBuffer(quadMesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
        enc.setFragmentBytes(&uniforms, length: MemoryLayout<PostOpticsUniforms>.stride, index: BufferIndex.subsystemUniforms.rawValue)
        for (tex, slot) in textures {
            enc.setFragmentTexture(tex, index: slot)
        }
        enc.drawIndexedPrimitives(
            type: .triangle,
            indexCount: quadMesh.indexCount,
            indexType: .uint32,
            indexBuffer: quadMesh.indexBuffer,
            indexBufferOffset: 0
        )
        enc.endEncoding()
    }

    private func rebuildTargets() {
        let targets = Self.createTargets(device: device, width: width, height: height, sampleCount: msaaSampleCount)
        sceneMSAATexture = targets.msaa
        sceneHDRTexture = targets.hdr
        sceneDepthTexture = targets.depth
        bloomATexture = targets.bloomA
        bloomBTexture = targets.bloomB
        godRayTexture = targets.godRay
    }

    private static func supportedSampleCount(device: MTLDevice, requested: Int) -> Int {
        let clamped = max(1, requested)
        for candidate in [clamped, 4, 2, 1] where candidate <= clamped {
            if device.supportsTextureSampleCount(candidate) {
                return candidate
            }
        }
        return 1
    }

    private static func makePipeline(
        device: MTLDevice,
        vertexFn: MTLFunction,
        fragmentFn: MTLFunction,
        vertexDescriptor: MTLVertexDescriptor,
        pixelFormat: MTLPixelFormat,
        label: String
    ) throws -> MTLRenderPipelineState {
        let desc = MTLRenderPipelineDescriptor()
        desc.label = label
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        desc.vertexDescriptor = vertexDescriptor
        desc.colorAttachments[0].pixelFormat = pixelFormat
        desc.depthAttachmentPixelFormat = .invalid
        return try device.makeRenderPipelineState(descriptor: desc)
    }

    private static func createTargets(
        device: MTLDevice,
        width: Int,
        height: Int,
        sampleCount: Int
    ) -> (
        msaa: MTLTexture?,
        hdr: MTLTexture,
        depth: MTLTexture,
        bloomA: MTLTexture,
        bloomB: MTLTexture,
        godRay: MTLTexture
    ) {
        let hdrDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        hdrDesc.usage = [.renderTarget, .shaderRead]
        hdrDesc.storageMode = .private
        let hdr = device.makeTexture(descriptor: hdrDesc)!
        hdr.label = "PostOptics_SceneHDR"

        var msaa: MTLTexture? = nil
        if sampleCount > 1 {
            let msaaDesc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float,
                width: width,
                height: height,
                mipmapped: false
            )
            msaaDesc.textureType = .type2DMultisample
            msaaDesc.sampleCount = sampleCount
            msaaDesc.usage = [.renderTarget]
            msaaDesc.storageMode = .private
            msaa = device.makeTexture(descriptor: msaaDesc)
            msaa?.label = "PostOptics_SceneMSAA"
        }

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        if sampleCount > 1 {
            depthDesc.textureType = .type2DMultisample
            depthDesc.sampleCount = sampleCount
        }
        depthDesc.usage = [.renderTarget]
        depthDesc.storageMode = .private
        let depth = device.makeTexture(descriptor: depthDesc)!
        depth.label = "PostOptics_SceneDepth"

        let bw = max(1, width >> 1)
        let bh = max(1, height >> 1)
        let halfDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: bw,
            height: bh,
            mipmapped: false
        )
        halfDesc.usage = [.renderTarget, .shaderRead]
        halfDesc.storageMode = .private

        let bloomA = device.makeTexture(descriptor: halfDesc)!
        bloomA.label = "PostOptics_BloomA"
        let bloomB = device.makeTexture(descriptor: halfDesc)!
        bloomB.label = "PostOptics_BloomB"
        let godRay = device.makeTexture(descriptor: halfDesc)!
        godRay.label = "PostOptics_GodRays"

        return (msaa, hdr, depth, bloomA, bloomB, godRay)
    }
}
