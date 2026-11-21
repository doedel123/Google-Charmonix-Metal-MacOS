import Foundation
import CoreGraphics
import ImageIO
import Metal
import MetalKit
import UniformTypeIdentifiers
import simd

public struct SubsystemContainer {
    public var terrainData: any TerrainQueryProtocol
    public var trackBuffer: any SnowTracksSubsystem
    public var terrain: any TerrainSubsystem
    public var sky: any SkySubsystem
    public var props: any AlpinePropsSubsystem
    public var particles: any ParticlesSubsystem
    public var rig: any RigSubsystem
    public var postOptics: any PostOpticsSubsystem
    public var audio: any AudioSubsystem
    public var simulation: any SimulationSubsystem

    public init(
        terrainData: any TerrainQueryProtocol,
        trackBuffer: any SnowTracksSubsystem,
        terrain: any TerrainSubsystem,
        sky: any SkySubsystem,
        props: any AlpinePropsSubsystem,
        particles: any ParticlesSubsystem,
        rig: any RigSubsystem,
        postOptics: any PostOpticsSubsystem,
        audio: any AudioSubsystem,
        simulation: any SimulationSubsystem
    ) {
        self.terrainData = terrainData
        self.trackBuffer = trackBuffer
        self.terrain = terrain
        self.sky = sky
        self.props = props
        self.particles = particles
        self.rig = rig
        self.postOptics = postOptics
        self.audio = audio
        self.simulation = simulation
    }
}

public final class MetalRenderer: NSObject, MTKViewDelegate {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public private(set) var subsystems: SubsystemContainer?

    // Triple-buffered FrameUniforms
    private let maxInFlight = 3
    private let frameSemaphore: DispatchSemaphore
    private var frameUniformBuffers: [MTLBuffer] = []
    private var frameIndex: Int = 0

    // Shadow map (1024x1024, depth32Float)
    public let shadowMapTexture: MTLTexture?

    // Simulation & Camera state
    public var settings: SimulationSettings
    public var input = InputState()
    public var camera = CameraState()
    public private(set) var lightIdx: Int = 0
    public private(set) var zoneIdx: Int = 0
    public private(set) var camModeIdx: Int = 0
    public private(set) var simTime: Float = 0
    public private(set) var started: Bool = false
    public private(set) var pausedSim: Bool = false
    public private(set) var hudMode: Int = 0 // 0=Full, 1=RAW REC, 2=Hidden
    public private(set) var scoutActive: Bool = false
    public private(set) var scoutPhase: Float = 0
    public private(set) var replayActive: Bool = false
    public private(set) var replayCursor: Float = 0
    public var inspectBodyView: String? = nil // "body_close" / "body_back"

    // Optics smoothing
    private var physAccum: Float = 0
    private var lastFrameTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
    private var lensWet: Float = 0
    private var smoothExposure: Float = 0.72
    private var smoothSunVis: Float = 0
    private var smoothAngVel: SIMD2<Float> = .zero
    private var smoothRollVel: Float = 0
    private var prevCamQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var hasPrevCam: Bool = false
    private var horizonUV = SIMD2<Float>(0.5, 0.56)
    private var sunUV = SIMD2<Float>(0.5, 0.5)

    // FPS tracking
    private var fpsFrames: Int = 0
    private var fpsAccum: Float = 0
    private var lastFps: Float = 60

    // Toast
    private var customToast: String = ""
    private var customToastUntil: Float = -99

    // Debug Frame Capture
    private var captureLock = NSLock()
    private var pendingCapturePath: String?
    private var captureAfterFrame: Int = 6
    private var quitAfterCapture: Bool = false
    private var renderedFrames: Int = 0
    private var captureStagingTexture: MTLTexture?

    // Telemetry callback (invoked on main queue)
    public var onTelemetryUpdate: ((HUDTelemetry) -> Void)?

    // Phase 2 integrated subsystem factory
    public static var buildSubsystems: ((MTLDevice, MTLCommandQueue, Int, Int) throws -> SubsystemContainer)? = { device, commandQueue, width, height in
        let terrainData = try ChamonixTerrainData(device: device, commandQueue: commandQueue)
        let trackBuffer = try SnowTrackBuffer(device: device)
        let terrain = try TerrainRenderer(device: device, terrainData: terrainData)
        let sky = try SkyAtmosphere(device: device)
        let props = try AlpinePropsSystem(device: device, terrainData: terrainData)
        let particles = try AlpineParticleSystem(
            device: device,
            puffTexture: terrainData.textures.powderPuff,
            peaks: terrainData.meta.peaks
        )
        let rig = try SkiRigRenderer(device: device, commandQueue: commandQueue, terrainData: terrainData)
        let postOptics = try PostOpticsPipeline(device: device, width: width, height: height, msaaSamples: 4)
        let audio = AlpineAudioEngine()
        let simulation = GameSimulation(terrainData: terrainData, colliders: props.colliders, kickers: props.kickers)
        return SubsystemContainer(
            terrainData: terrainData,
            trackBuffer: trackBuffer,
            terrain: terrain,
            sky: sky,
            props: props,
            particles: particles,
            rig: rig,
            postOptics: postOptics,
            audio: audio,
            simulation: simulation
        )
    }

    public init?(mtkView: MTKView, subsystems: SubsystemContainer? = nil, settings: SimulationSettings = SimulationSettings()) {
        guard let dev = mtkView.device ?? MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue() else {
            return nil
        }
        self.device = dev
        self.commandQueue = queue
        self.subsystems = subsystems
        self.settings = settings
        self.frameSemaphore = DispatchSemaphore(value: maxInFlight)

        mtkView.device = dev
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .invalid // Main HDR pass uses PostOptics depth texture
        mtkView.framebufferOnly = false

        for i in 0..<maxInFlight {
            guard let buf = dev.makeBuffer(length: MemoryLayout<FrameUniforms>.stride, options: .storageModeShared) else {
                return nil
            }
            buf.label = "FrameUniforms_\(i)"
            frameUniformBuffers.append(buf)
        }

        let shDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: 1024,
            height: 1024,
            mipmapped: false
        )
        shDesc.usage = [.renderTarget, .shaderRead]
        shDesc.storageMode = .private
        self.shadowMapTexture = dev.makeTexture(descriptor: shDesc)
        self.shadowMapTexture?.label = "DirectionalShadowMap"

        super.init()

        // Check environment variable fallback for automated frame capture
        if let envPath = ProcessInfo.processInfo.environment["CHAMONIX_CAPTURE_FRAME"], !envPath.isEmpty {
            let quit = ProcessInfo.processInfo.environment["CHAMONIX_CAPTURE_QUIT"] == "1"
            requestCapture(to: envPath, afterFrames: 6, quitAfter: quit)
        }
    }

    public func attachSubsystems(_ container: SubsystemContainer, initialZoneIdx: Int = 0, initialLightIdx: Int = 0) {
        self.subsystems = container
        self.lightIdx = ((initialLightIdx % LightingPresets.all.count) + LightingPresets.all.count) % LightingPresets.all.count
        self.smoothExposure = LightingPresets.all[self.lightIdx].exposure
        spawnAtZone(initialZoneIdx)
    }

    // MARK: - Controls & Actions

    public func requestCapture(to path: String, afterFrames: Int = 6, quitAfter: Bool = false) {
        captureLock.lock()
        pendingCapturePath = path
        captureAfterFrame = renderedFrames + max(1, afterFrames)
        quitAfterCapture = quitAfter
        captureLock.unlock()
    }

    public func showToast(_ msg: String, duration: Float = 2.6) {
        customToast = msg
        customToastUntil = simTime + duration
    }

    public func setStarted(_ value: Bool) {
        started = value
        if value { subsystems?.audio.unlock() }
    }

    public func togglePause() {
        pausedSim.toggle()
    }

    public func setPaused(_ value: Bool) {
        pausedSim = value
    }

    public func setHudMode(_ mode: Int) {
        hudMode = ((mode % 3) + 3) % 3
    }

    public func cycleHudMode() {
        setHudMode(hudMode + 1)
    }

    public func setLightPreset(_ idx: Int) {
        let count = LightingPresets.all.count
        lightIdx = ((idx % count) + count) % count
        let preset = LightingPresets.all[lightIdx]
        smoothExposure = preset.exposure
    }

    public func cycleLightPreset() {
        setLightPreset(lightIdx + 1)
    }

    public func setCameraMode(_ idx: Int) {
        let modes = CameraMode.allCases
        camModeIdx = ((idx % modes.count) + modes.count) % modes.count
        camera.mode = modes[camModeIdx]
    }

    public func cycleCameraMode() {
        setCameraMode(camModeIdx + 1)
    }

    public func exitSpecialModes() {
        scoutActive = false
        replayActive = false
    }

    public func toggleScout(_ force: Bool? = nil) {
        let next = force ?? !scoutActive
        if next {
            replayActive = false
            scoutActive = true
            scoutPhase = 0
        } else {
            scoutActive = false
        }
    }

    public func toggleReplay(_ force: Bool? = nil) {
        guard let sim = subsystems?.simulation else { return }
        let next = force ?? !replayActive
        if next && !sim.replayFrames.isEmpty {
            scoutActive = false
            replayActive = true
            replayCursor = 0
        } else {
            replayActive = false
        }
    }

    public func spawnAtZone(_ idx: Int) {
        guard let sub = subsystems else { return }
        exitSpecialModes()
        let zones = sub.terrainData.meta.dropZones
        guard !zones.isEmpty else { return }
        zoneIdx = ((idx % zones.count) + zones.count) % zones.count
        let z = zones[zoneIdx]
        sub.simulation.spawn(zone: z)
        sub.trackBuffer.reset(x: z.x, z: z.z)

        if sub.simulation.state.surface != .powder {
            let fx = sinf(z.yaw), fz = -cosf(z.yaw)
            let rx = cosf(z.yaw), rz = sinf(z.yaw)
            let sep: Float = 0.192
            sub.trackBuffer.stampTwinSkis(
                lA: SIMD2<Float>(z.x - rx * sep - fx * 0.68, z.z - rz * sep - fz * 0.68),
                lB: SIMD2<Float>(z.x - rx * sep + fx * 0.74, z.z - rz * sep + fz * 0.74),
                rA: SIMD2<Float>(z.x + rx * sep - fx * 0.68, z.z + rz * sep - fz * 0.68),
                rB: SIMD2<Float>(z.x + rx * sep + fx * 0.74, z.z + rz * sep + fz * 0.74),
                radius: 0.068, depth: 0.08, berm: 0.018, iceGlint: 0.03, skew: 0.0
            )
        }
        lensWet = 0
        hasPrevCam = false
    }

    public func applySettings(_ newSettings: SimulationSettings) {
        settings = newSettings
        subsystems?.audio.setVolume(newSettings.volume)
        subsystems?.audio.setMuted(newSettings.muted)
        subsystems?.simulation.smoothCarve = newSettings.smoothCarve
        subsystems?.simulation.avalanchesEnabled = newSettings.avalanchesEnabled
        switch newSettings.quality {
        case .auto, .high: subsystems?.postOptics.setMSAA(samples: 4)
        case .med:         subsystems?.postOptics.setMSAA(samples: 2)
        case .low:         subsystems?.postOptics.setMSAA(samples: 1)
        }
    }

    // MARK: - MTKViewDelegate

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        let w = max(1, Int(size.width))
        let h = max(1, Int(size.height))
        camera.aspect = Float(w) / Float(h)
        subsystems?.postOptics.resize(width: w, height: h)
    }

    public func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let sub = subsystems else { return }

        _ = frameSemaphore.wait(timeout: .distantFuture)

        let now = CFAbsoluteTimeGetCurrent()
        let dt = clamp(Float(now - lastFrameTime), 0.001, 0.05)
        lastFrameTime = now

        fpsFrames += 1
        fpsAccum += dt
        if fpsAccum >= 0.4 {
            lastFps = Float(fpsFrames) / fpsAccum
            fpsFrames = 0
            fpsAccum = 0
        }

        let w = max(1, drawable.texture.width)
        let h = max(1, drawable.texture.height)
        camera.aspect = Float(w) / Float(h)

        let preset = LightingPresets.all[lightIdx]
        let sunDir = preset.sunDirection
        let liveStep = started && !pausedSim && !scoutActive && !replayActive && inspectBodyView == nil

        if liveStep {
            physAccum = min(0.08, physAccum + dt)
            while physAccum >= ChamonixConstants.fixedPhysicsDt {
                simTime += ChamonixConstants.fixedPhysicsDt
                sub.simulation.step(dt: ChamonixConstants.fixedPhysicsDt, time: simTime, input: input)
                input.lookDeltaYaw = 0
                input.lookDeltaPitch = 0
                consumePhysicsEvents(sub: sub)
                physAccum -= ChamonixConstants.fixedPhysicsDt
            }

            let windDecay: Float = preset.isStorm ? 0.0005 : 0.0
            sub.trackBuffer.recenterIfNeeded(
                skierX: sub.simulation.state.x,
                skierZ: sub.simulation.state.z,
                decayAmt: windDecay
            )
            sub.particles.emitAvalancheWave(avalanche: sub.simulation.avalanche, terrainData: sub.terrainData, dt: dt)
            emitSpindrift(sub: sub, preset: preset)
        } else if !pausedSim {
            simTime += dt
        }

        let s = sub.simulation.state
        if s.faceShotIntensity > 0.40 && liveStep {
            lensWet = min(0.85, lensWet + dt * s.faceShotIntensity * 0.25)
        } else if liveStep {
            lensWet = max(0, lensWet - dt * 0.28)
        }

        var activeState = s
        var focusXZ = SIMD2<Float>(s.x, s.z)
        var isThirdPerson = false
        var replayProg: Float = 0

        if scoutActive {
            if !pausedSim {
                scoutPhase += dt / ChamonixConstants.scoutDuration
                if scoutPhase >= 1.0 { toggleScout(false) }
            }
            if scoutActive {
                focusXZ = applyScoutCamera(phase: scoutPhase, sub: sub)
                isThirdPerson = true
            }
        } else if replayActive && !sub.simulation.replayFrames.isEmpty {
            let res = applyReplayFrame(dt: pausedSim ? 0 : dt, sub: sub)
            activeState = res.state
            replayProg = res.prog
            focusXZ = SIMD2<Float>(activeState.x, activeState.z)
            isThirdPerson = true
        } else if let inspect = inspectBodyView {
            isThirdPerson = true
            let fx = sinf(s.yaw), fz = -cosf(s.yaw)
            let rx = cosf(s.yaw), rz = sinf(s.yaw)
            let isBack = inspect == "body_back"
            let cx = s.x + (isBack ? -fx * 2.15 + rx * 1.15 : fx * 2.05 + rx * 1.32)
            let cz = s.z + (isBack ? -fz * 2.15 + rz * 1.15 : fz * 2.05 + rz * 1.32)
            let cy = max(sub.terrainData.sampleHeight(x: cx, z: cz) + 0.65, s.y + (isBack ? 1.55 : 1.32))
            camera.position = SIMD3<Float>(cx, cy, cz)
            let target = SIMD3<Float>(s.x + fx * 0.08, s.y + 0.96, s.z + fz * 0.08)
            let lookMat = matrixLookAtRightHand(eye: camera.position, target: target)
            camera.quaternion = simd_quatf(simd_inverse(lookMat))
            camera.fovDeg = 48
        }

        if !scoutActive && !replayActive && inspectBodyView == nil {
            sub.simulation.applyCamera(camera: &camera, mode: camera.mode, time: simTime, dt: dt)
        }
        camera.isThirdPerson = isThirdPerson
        let isFP = !isThirdPerson && !scoutActive && !replayActive
        let isChesty = (camera.mode == .chesty && isFP)
        camera.isChesty = isChesty

        let isStorm = preset.isStorm
        let mtnSunSkier = isStorm ? Float(0.15) : sampleSunLOS(terrain: sub.terrainData, ox: activeState.x, oy: activeState.y + 1.1, oz: activeState.z, sunDir: sunDir)
        let camSunLOS = isStorm ? Float(0.0) : sampleSunLOS(terrain: sub.terrainData, ox: camera.position.x, oy: camera.position.y + 0.2, oz: camera.position.z, sunDir: sunDir)

        let renderTime: Float = pausedSim ? (simTime == 0 ? 4.2 : simTime) : simTime

        // Update subsystems
        sub.rig.update(
            state: activeState,
            dt: pausedSim ? 0.016 : dt,
            isThirdPerson: isThirdPerson,
            time: renderTime,
            sunDir: sunDir,
            isStorm: isStorm,
            lookYawOff: sub.simulation.lookYawOff,
            mtnSunVis: mtnSunSkier,
            isAlpenglow: preset.isAlpenglow,
            isChesty: isChesty,
            preset: preset
        )
        sub.sky.update(cameraPos: camera.position, preset: preset, sunDir: sunDir, time: renderTime)
        sub.terrain.update(
            cameraPos: camera.position,
            focusXZ: focusXZ,
            preset: preset,
            sunDir: sunDir,
            time: renderTime,
            trackBuffer: sub.trackBuffer,
            avalanche: sub.simulation.avalanche
        )
        sub.props.update(cameraPos: camera.position, preset: preset, sunDir: sunDir, time: renderTime)
        sub.particles.update(
            dt: pausedSim ? 0 : dt,
            cameraPos: camera.position,
            viewportHeightPx: Float(h),
            fovDeg: camera.fovDeg,
            preset: preset,
            sunDir: sunDir,
            time: renderTime
        )
        sub.audio.update(state: s, preset: preset, avalanche: sub.simulation.avalanche, dt: dt)

        // Matrices & Uniforms
        let viewMat = camera.viewMatrix
        let projMat = camera.projectionMatrix
        let viewProjMat = projMat * viewMat

        let lightTarget = SIMD3<Float>(activeState.x, activeState.y + 0.90, activeState.z)
        let lightEye = lightTarget + sunDir * 14.0
        let lightView = matrixLookAtRightHand(eye: lightEye, target: lightTarget)
        let lightProj = matrixOrthoRightHand(left: -3.6, right: 3.6, bottom: -3.6, top: 3.6, nearZ: 1.0, farZ: 62.0)
        let lightViewProj = lightProj * lightView

        var fu = FrameUniforms()
        fu.viewMatrix = viewMat
        fu.projectionMatrix = projMat
        fu.viewProjectionMatrix = viewProjMat
        fu.lightViewProjectionMatrix = lightViewProj
        fu.cameraPosition = SIMD4<Float>(camera.position, renderTime)
        fu.sunDirection = SIMD4<Float>(sunDir, asinf(clamp(sunDir.y, -1.0, 1.0)))
        fu.sunColor = SIMD4<Float>(preset.sunColor, preset.sunIntensity)
        fu.skyZenith = SIMD4<Float>(preset.skyZenith, preset.cloudCoverage)
        fu.skyHorizon = SIMD4<Float>(preset.skyHorizon, isStorm ? 1.0 : 0.0)
        fu.shadowSkyColor = SIMD4<Float>(preset.shadowSkyColor, preset.shadowFillStrength)
        fu.groundBounceColor = SIMD4<Float>(preset.groundBounceColor, preset.twilightFactor)
        fu.sssColor = SIMD4<Float>(preset.sssColor, preset.sparkleIntensity)
        fu.fogColor = SIMD4<Float>(preset.fogColor, preset.fogDensity)
        fu.cloudParams = SIMD4<Float>(
            preset.valleyCloudAlt,
            preset.valleyCloudDensity,
            preset.cloudShadowStrength,
            Float(sub.terrainData.coreSize)
        )
        fu.horizonBlend = preset.horizonBlend
        fu.screenParams = SIMD4<Float>(Float(w), Float(h), degToRad(camera.fovDeg), Float(sub.terrainData.outerSize))
        let avy = sub.simulation.avalanche
        fu.avalancheParams = SIMD4<Float>(avy.active ? avy.strength : 0.0, avy.originX, avy.originZ, avy.frontDist)
        fu.avalancheDirWidth = SIMD4<Float>(avy.dirX, avy.dirZ, avy.width, mtnSunSkier)
        fu.trackParams = SIMD4<Float>(
            sub.trackBuffer.originX,
            sub.trackBuffer.originZ,
            sub.trackBuffer.worldSpan,
            Float(sub.trackBuffer.resolution)
        )
        fu.macroTrackParams = SIMD4<Float>(
            sub.trackBuffer.macroOriginX,
            sub.trackBuffer.macroOriginZ,
            sub.trackBuffer.macroSpan,
            Float(sub.trackBuffer.macroResolution)
        )
        fu.extraParams = SIMD4<Float>(
            preset.snowfallRate,
            sub.terrainData.meta.coreHalfSpan,
            sub.terrainData.meta.outerHalfSpan,
            sub.terrainData.meta.maxElev
        )

        let bufIdx = frameIndex % maxInFlight
        frameIndex += 1
        let frameBuf = frameUniformBuffers[bufIdx]
        memcpy(frameBuf.contents(), &fu, MemoryLayout<FrameUniforms>.stride)

        // PostOptics & optical telemetry
        let postParams = computePostOpticsParams(
            activeState: activeState,
            preset: preset,
            sunDir: sunDir,
            viewProj: viewProjMat,
            camSunLOS: camSunLOS,
            mtnSunSkier: mtnSunSkier,
            isFP: isFP,
            dt: dt,
            renderTime: renderTime
        )

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            frameSemaphore.signal()
            return
        }
        cmdBuf.label = "ChamonixFrame"

        let ctx = RenderPassContext(
            device: device,
            commandBuffer: cmdBuf,
            frameUniforms: fu,
            frameUniformBuffer: frameBuf,
            camera: camera,
            preset: preset,
            sunDir: sunDir,
            mtnSunVis: mtnSunSkier,
            time: renderTime,
            dt: dt,
            isThirdPerson: isThirdPerson,
            isChesty: isChesty,
            viewportSize: SIMD2<Float>(Float(w), Float(h)),
            shadowMapTexture: shadowMapTexture,
            msaaSampleCount: sub.postOptics.msaaSampleCount
        )

        // 1. Directional Shadow Map Pass
        if let shadowTex = shadowMapTexture {
            let shadowRP = MTLRenderPassDescriptor()
            shadowRP.depthAttachment.texture = shadowTex
            shadowRP.depthAttachment.loadAction = .clear
            shadowRP.depthAttachment.storeAction = .store
            shadowRP.depthAttachment.clearDepth = 1.0
            if let enc = cmdBuf.makeRenderCommandEncoder(descriptor: shadowRP) {
                enc.label = "ShadowMapPass"
                sub.rig.renderShadowPass(encoder: enc, context: ctx)
                sub.props.renderShadowPass(encoder: enc, context: ctx)
                enc.endEncoding()
            }
        }

        // 2. Snow Tracks RT Pass
        sub.trackBuffer.encodeRenderPasses(commandBuffer: cmdBuf)

        // 3. HDR Scene Pass (rgba16Float, alpha = 1.0 sky clear)
        let hdrRP = MTLRenderPassDescriptor()
        if sub.postOptics.msaaSampleCount > 1, let msaaTex = sub.postOptics.sceneMSAATexture {
            hdrRP.colorAttachments[0].texture = msaaTex
            hdrRP.colorAttachments[0].resolveTexture = sub.postOptics.sceneHDRTexture
            hdrRP.colorAttachments[0].storeAction = .multisampleResolve
        } else {
            hdrRP.colorAttachments[0].texture = sub.postOptics.sceneHDRTexture
            hdrRP.colorAttachments[0].storeAction = .store
        }
        hdrRP.colorAttachments[0].loadAction = .clear
        hdrRP.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1.0)
        hdrRP.depthAttachment.texture = sub.postOptics.sceneDepthTexture
        hdrRP.depthAttachment.loadAction = .clear
        hdrRP.depthAttachment.storeAction = .dontCare
        hdrRP.depthAttachment.clearDepth = 1.0

        if let enc = cmdBuf.makeRenderCommandEncoder(descriptor: hdrRP) {
            enc.label = "HDRScenePass"
            sub.sky.renderSkyPass(encoder: enc, context: ctx)
            sub.terrain.renderHDRPass(encoder: enc, context: ctx, trackBuffer: sub.trackBuffer)
            sub.props.renderGroundShadowsPass(encoder: enc, context: ctx)
            sub.rig.renderGroundShadowsPass(encoder: enc, context: ctx)
            sub.props.renderHDRPass(encoder: enc, context: ctx)
            sub.rig.renderHDRPass(encoder: enc, context: ctx)
            sub.sky.renderValleyCloudsPass(encoder: enc, context: ctx)
            sub.particles.renderHDRPass(encoder: enc, context: ctx)
            enc.endEncoding()
        }

        // 4. Post-Optics HDR -> LDR Pass
        sub.postOptics.render(
            commandBuffer: cmdBuf,
            drawableTexture: drawable.texture,
            params: postParams,
            context: ctx
        )

        // 5. Debug Frame Capture Hook
        renderedFrames += 1
        var captureToPath: String?
        var shouldQuit = false
        captureLock.lock()
        if let path = pendingCapturePath, renderedFrames >= captureAfterFrame {
            captureToPath = path
            shouldQuit = quitAfterCapture
            pendingCapturePath = nil
        }
        captureLock.unlock()

        if let outPath = captureToPath {
            let staging = ensureCaptureStagingTexture(width: w, height: h, pixelFormat: drawable.texture.pixelFormat)
            if let staging, let blit = cmdBuf.makeBlitCommandEncoder() {
                blit.label = "CaptureBlit"
                blit.copy(
                    from: drawable.texture,
                    sourceSlice: 0,
                    sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: MTLSize(width: w, height: h, depth: 1),
                    to: staging,
                    destinationSlice: 0,
                    destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                )
                blit.endEncoding()
            }
            cmdBuf.present(drawable)
            cmdBuf.addCompletedHandler { [weak self] _ in
                self?.frameSemaphore.signal()
                if let staging {
                    Self.saveTexturePNG(staging, to: outPath)
                }
                if shouldQuit {
                    DispatchQueue.main.async {
                        exit(0)
                    }
                }
            }
        } else {
            cmdBuf.present(drawable)
            cmdBuf.addCompletedHandler { [weak self] _ in
                self?.frameSemaphore.signal()
            }
        }

        cmdBuf.commit()

        // Emit HUD telemetry
        if let onTelemetryUpdate {
            let tel = buildTelemetry(activeState: activeState, replayProg: replayProg, sub: sub)
            DispatchQueue.main.async {
                onTelemetryUpdate(tel)
            }
        }
    }

    // MARK: - Helpers

    private func sampleSunLOS(terrain: any TerrainQueryProtocol, ox: Float, oy: Float, oz: Float, sunDir: SIMD3<Float>) -> Float {
        if sunDir.y <= 0.02 { return 0.0 }
        var minClear: Float = 1e9
        let dists: [Float] = [45, 110, 240, 480, 860, 1450, 2250, 3250, 4450, 5800]
        for d in dists {
            let px = ox + sunDir.x * d
            let pz = oz + sunDir.z * d
            if abs(px) > 6300 || abs(pz) > 6300 { break }
            let rayY = oy + sunDir.y * d
            let terrY = terrain.sampleHeight(x: px, z: pz)
            let angleDiff = (rayY - terrY) / d
            if angleDiff < minClear { minClear = angleDiff }
            if minClear < -0.028 { return 0.0 }
        }
        return smoothstep(-0.022, 0.036, minClear)
    }

    private func consumePhysicsEvents(sub: SubsystemContainer) {
        let events = sub.simulation.events
        for ev in events {
            switch ev {
            case let .track(lA, lB, rA, rB, radius, depth, berm, iceGlint, skew):
                sub.trackBuffer.stampTwinSkis(lA: lA, lB: lB, rA: rA, rB: rB, radius: radius, depth: depth, berm: berm, iceGlint: iceGlint, skew: skew)
            case let .spray(x, y, z, vx, vy, vz, rate, sluff, slopeX, slopeZ):
                sub.particles.emitCarveRoosterTail(x: x, y: y, z: z, vx: vx, vy: vy, vz: vz, rate: rate, sluff: sluff, slopeX: slopeX, slopeZ: slopeZ)
                if sluff && Float.random(in: 0..<1) < 0.38 {
                    let sn = max(0.001, hypotf(slopeX, slopeZ))
                    sub.particles.emitPuff(x: x, y: y + 0.18, z: z, vx: (slopeX / sn) * 7.2, vy: -1.0, vz: (slopeZ / sn) * 7.2, size: 1.8, expand: 2.4, peakAlpha: 0.42, life: 1.8, kind: 0)
                }
            case let .polePlant(side, x, y, z, strength, powder):
                sub.audio.handleEvent(ev)
                sub.trackBuffer.stampPoleDimple(x: x, z: z, radius: powder ? 0.15 : 0.11, depth: powder ? 0.56 : 0.34)
                sub.particles.emitBurst(x: x, y: y + 0.06, z: z, count: powder ? 5 : 3, speed: 1.15 + strength * 0.75, sizeMin: 0.18, sizeMax: 0.36, expand: 0.55 + strength * 0.25, peakAlpha: powder ? 0.56 : 0.34, life: 0.65, kind: 0)
                _ = side
            case .treeWhoosh, .bootCreak:
                sub.audio.handleEvent(ev)
            case let .sluffStream(x, y, z, dirX, dirZ, _, strength):
                let spd = 6.2 + strength * 5.4
                let sx = -dirZ, sz = dirX
                for k in 0..<2 {
                    let lat = (Float.random(in: 0..<1) - 0.5) * 0.65
                    sub.particles.emitPuff(
                        x: x + sx * lat, y: y + 0.05 + Float(k) * 0.04, z: z + sz * lat,
                        vx: dirX * spd + (Float.random(in: 0..<1) - 0.5) * 0.9,
                        vy: -0.65,
                        vz: dirZ * spd + (Float.random(in: 0..<1) - 0.5) * 0.9,
                        size: 0.42 + strength * 0.42, expand: 1.35 + Float.random(in: 0..<1) * 0.55,
                        peakAlpha: 0.36 + strength * 0.22, life: 0.72, kind: 0
                    )
                }
            case let .pop(x, y, z, _):
                sub.audio.handleEvent(ev)
                sub.particles.emitBurst(x: x, y: y + 0.1, z: z, count: 12, speed: 3.2, sizeMin: 0.65, sizeMax: 1.35, expand: 1.05, peakAlpha: 0.54, life: 1.8, kind: 0)
            case let .land(x, y, z, impact, powder, _):
                sub.audio.handleEvent(ev)
                let rad = 0.48 + min(0.85, impact * 0.055)
                sub.trackBuffer.stampCapsule(ax: x - 0.24, az: z, bx: x + 0.24, bz: z, radius: rad, depth: 0.95, berm: 0.92, iceGlint: 0.6, skew: 0)
                sub.particles.emitBurst(x: x, y: y + 0.2, z: z, count: min(22, 8 + Int(roundf(impact * 1.2))), speed: 3.8 + impact * 0.35, sizeMin: 0.85, sizeMax: 2.0, expand: 1.5, peakAlpha: 0.50, life: 1.9, kind: 0)
                if powder && impact > 6 {
                    lensWet = min(0.85, lensWet + 0.16)
                }
            case let .crash(x, y, z, _, _):
                sub.audio.handleEvent(ev)
                sub.trackBuffer.stampCapsule(ax: x - 0.5, az: z - 0.5, bx: x + 0.5, bz: z + 0.5, radius: 1.35, depth: 1.0, berm: 1.0, iceGlint: 0.5, skew: 0)
                sub.particles.emitBurst(x: x, y: y + 0.25, z: z, count: 26, speed: 6.2, sizeMin: 1.0, sizeMax: 2.6, expand: 1.9, peakAlpha: 0.56, life: 2.1, kind: 0)
                lensWet = 0.60
            case let .crashTrail(x, y, z, _):
                sub.trackBuffer.stampCapsule(ax: x - 0.2, az: z, bx: x + 0.2, bz: z, radius: 0.80, depth: 0.9, berm: 0.85, iceGlint: 0.3, skew: 0)
                sub.particles.emitPuff(x: x, y: y + 0.2, z: z, vx: (Float.random(in: 0..<1) - 0.5) * 4, vy: 2.0, vz: (Float.random(in: 0..<1) - 0.5) * 4, size: 1.4, expand: 1.6, peakAlpha: 0.60, life: 2.0, kind: 0)
            case .avalancheStart:
                sub.audio.handleEvent(ev)
                let a = sub.simulation.avalanche
                let sx = -a.dirZ, sz = a.dirX
                for k in -8...8 {
                    let across = (Float(k) / 8.0) * a.width * 0.85
                    let wx = a.originX + sx * across
                    let wz = a.originZ + sz * across
                    let wy = sub.terrainData.sampleHeight(x: wx, z: wz)
                    sub.particles.emitPuff(
                        x: wx, y: wy + 1.2, z: wz,
                        vx: a.dirX * 10 + (Float.random(in: 0..<1) - 0.5) * 3,
                        vy: 2.8 + Float.random(in: 0..<1) * 2.8,
                        vz: a.dirZ * 10 + (Float.random(in: 0..<1) - 0.5) * 3,
                        size: 3.4 + Float.random(in: 0..<1) * 2.8, expand: 2.8, peakAlpha: 0.78, life: 2.6, kind: 1
                    )
                }
            }
        }
        sub.simulation.clearEvents()
    }

    private func emitSpindrift(sub: SubsystemContainer, preset: LightingPreset) {
        let s = sub.simulation.state
        let rate: Float = preset.isStorm ? 0.65 : (s.y > 2900 ? 0.24 : 0.08)
        if Float.random(in: 0..<1) < rate {
            let ang = Float.random(in: 0..<1) * .pi * 2
            let dist = 16 + Float.random(in: 0..<1) * 48
            let rx = s.x + cosf(ang) * dist
            let rz = s.z + sinf(ang) * dist
            let t = sub.terrainData.sampleTerrain(x: rx, z: rz)
            if t.curv > 0.54 && t.rawY > 2400 {
                sub.particles.emitPuff(
                    x: rx, y: t.y + 0.35, z: rz,
                    vx: 5.5 + Float.random(in: 0..<1) * 4.5,
                    vy: 0.4 + Float.random(in: 0..<1) * 1.1,
                    vz: 2.2 + Float.random(in: 0..<1) * 2.5,
                    size: 2.4 + Float.random(in: 0..<1) * 3.2,
                    expand: 2.7, peakAlpha: 0.28, life: 1.4, kind: 2
                )
            }
        }
    }

    private func applyScoutCamera(phase: Float, sub: SubsystemContainer) -> SIMD2<Float> {
        let zones = sub.terrainData.meta.dropZones
        guard !zones.isEmpty else { return .zero }
        let z = zones[zoneIdx]
        var dx = sinf(z.yaw)
        var dz = -cosf(z.yaw)
        if let te = z.traceEnd, te.count == 2 {
            let tx = te[0] - z.x
            let tz = te[1] - z.z
            let len = hypotf(tx, tz)
            if len > 40 { dx = tx / len; dz = tz / len }
        }
        let rx = -dz, rz = dx
        let u = sinf(phase * .pi)
        let along = 24 + u * 275
        let sway = sinf(phase * .pi * 2.0) * 58 + 26 * u
        let cx = z.x + dx * along + rx * sway
        let cz = z.z + dz * along + rz * sway
        let gY = sub.terrainData.sampleHeight(x: cx, z: cz)
        let altAbove = 28 + sinf(phase * .pi) * 44
        camera.position = SIMD3<Float>(cx, gY + altAbove, cz)
        let lookAhead = phase < 0.62 ? (along + 260) : (along - 120)
        let lx = z.x + dx * lookAhead
        let lz = z.z + dz * lookAhead
        let ly = sub.terrainData.sampleHeight(x: lx, z: lz)
        let lookMat = matrixLookAtRightHand(eye: camera.position, target: SIMD3<Float>(lx, ly + 8, lz))
        camera.quaternion = simd_quatf(simd_inverse(lookMat))
        sub.simulation.enforceCameraClearance(camera: &camera, minClearance: 8.0, lockRoll: false)
        camera.fovDeg = 66
        return SIMD2<Float>(cx + dx * 45, cz + dz * 45)
    }

    private func applyReplayFrame(dt: Float, sub: SubsystemContainer) -> (state: SkierState, prog: Float) {
        let frames = sub.simulation.replayFrames
        if dt > 0 {
            replayCursor += dt * 30.0
            if replayCursor >= Float(frames.count) { replayCursor = 0 }
        }
        let idx = min(frames.count - 1, Int(floorf(replayCursor)))
        let f = frames[idx]
        let st = f.toSkierState()

        if dt > 0 && f.sprayRate > 0.25 && f.grounded {
            let rx = cosf(f.yaw), rz = sinf(f.yaw)
            let side: Float = f.edgeAngle >= 0 ? -1 : 1
            sub.particles.emitPuff(
                x: f.x + rx * side * 0.35, y: f.y + 0.15, z: f.z + rz * side * 0.35,
                vx: rx * side * 3.8, vy: 2.2, vz: rz * side * 3.8,
                size: 1.35 + f.sprayRate * 0.85, expand: 1.65, peakAlpha: 0.62, life: 2.1, kind: 0
            )
        }

        let prog = frames.count > 1 ? Float(idx) / Float(frames.count - 1) : 0
        let fwdX = sinf(f.yaw), fwdZ = -cosf(f.yaw)
        let rX = cosf(f.yaw), rZ = sinf(f.yaw)
        let sideSign: Float = f.edgeAngle >= 0 ? 1 : -1
        let cx = f.x + fwdX * 12.5 + rX * sideSign * 14.0
        let cz = f.z + fwdZ * 12.5 + rZ * sideSign * 14.0
        let gY = sub.terrainData.sampleHeight(x: cx, z: cz)
        let cy = max(gY + 3.8, f.y + 5.6)
        camera.position = SIMD3<Float>(cx, cy, cz)
        let lookMat = matrixLookAtRightHand(eye: camera.position, target: SIMD3<Float>(f.x - fwdX * 4.5, f.y + 1.0, f.z - fwdZ * 4.5))
        camera.quaternion = simd_quatf(simd_inverse(lookMat))
        sub.simulation.enforceCameraClearance(camera: &camera, minClearance: 4.2, lockRoll: true)
        camera.fovDeg = 50
        return (st, prog)
    }

    private func computePostOpticsParams(
        activeState: SkierState,
        preset: LightingPreset,
        sunDir: SIMD3<Float>,
        viewProj: simd_float4x4,
        camSunLOS: Float,
        mtnSunSkier: Float,
        isFP: Bool,
        dt: Float,
        renderTime: Float
    ) -> PostOpticsParams {
        let isStorm = preset.isStorm
        let camFwd = camera.quaternion.act(SIMD3<Float>(0, 0, -1))
        let camRgt = camera.quaternion.act(SIMD3<Float>(1, 0, 0))
        let camUp = camera.quaternion.act(SIMD3<Float>(0, 1, 0))
        let sunDot = simd_dot(camFwd, sunDir)

        var targetSunVis: Float = 0.0
        var godRayVis: Float = 0.0
        if sunDot > 0.08 && !isStorm {
            sunUV = projectWorldToScreenUV(camera.position + sunDir * 5000.0, viewProjection: viewProj)
            godRayVis = smoothstep(0.08, 0.50, sunDot)
            if sunDot > 0.18 && camSunLOS > 0.01 {
                targetSunVis = camSunLOS * smoothstep(0.18, 0.55, sunDot)
            }
        }
        smoothSunVis = pausedSim ? targetSunVis : (smoothSunVis + (targetSunVis - smoothSunVis) * min(1, dt * 10.0))

        let sunLook = max(0, sunDot) * camSunLOS
        let shadeLift = (1.0 - max(0, simd_dot(activeState.normal, sunDir)) * mtnSunSkier) * 0.16
        let flareSquint = smoothSunVis * 0.12
        let targetExp = preset.exposure * (1.0 - sunLook * 0.18 - flareSquint + shadeLift)
        smoothExposure = pausedSim ? targetExp : (smoothExposure + (targetExp - smoothExposure) * min(1, dt * 1.45))

        if hasPrevCam && !pausedSim && !scoutActive && !replayActive && dt > 0.0005 {
            let prevFwd = prevCamQuat.act(SIMD3<Float>(0, 0, -1))
            let prevUp = prevCamQuat.act(SIMD3<Float>(0, 1, 0))
            let dX = simd_dot(prevFwd, camRgt)
            let dY = simd_dot(prevFwd, camUp)
            let dR = simd_dot(prevUp, camRgt)
            if hypotf(dX, dY) < 0.35 {
                let shutter = 0.0085 / max(0.004, dt)
                let fovScale = 1.0 / tanf(degToRad(camera.fovDeg * 0.5))
                let tx = clamp(dX * fovScale * shutter * 0.55, -0.042, 0.042)
                let ty = clamp(dY * fovScale * shutter * 0.55, -0.042, 0.042)
                smoothAngVel.x += (tx - smoothAngVel.x) * min(1, dt * 18)
                smoothAngVel.y += (ty - smoothAngVel.y) * min(1, dt * 18)
                smoothRollVel += (clamp(dR * shutter * 0.35, -0.025, 0.025) - smoothRollVel) * min(1, dt * 18)
            } else {
                smoothAngVel = .zero
                smoothRollVel = 0
            }
        } else if pausedSim {
            smoothAngVel = .zero
            smoothRollVel = 0
        }
        prevCamQuat = camera.quaternion
        hasPrevCam = true

        let spd = activeState.speed
        if spd > 1.5 {
            let velDir = SIMD3<Float>(activeState.vx / spd, activeState.vy / spd, activeState.vz / spd)
            let vDot = simd_dot(velDir, camFwd)
            if abs(vDot) > 0.15 {
                let targetPt = camera.position + velDir * (vDot >= 0 ? 5000 : -5000)
                let puv = projectWorldToScreenUV(targetPt, viewProjection: viewProj)
                horizonUV = SIMD2<Float>(clamp(puv.x, -0.35, 1.35), clamp(puv.y, -0.25, 1.25))
            } else {
                horizonUV = SIMD2<Float>(0.5, 0.56)
            }
        } else {
            horizonUV = SIMD2<Float>(0.5, 0.56)
        }

        let kmh = activeState.speed * 3.6
        let speedBlur: Float = (scoutActive || replayActive || inspectBodyView != nil) ? 0.0 : clamp((kmh - 22) / 68, 0, 1.15)
        let breathAmt: Float = isFP ? clamp((activeState.breathing - 0.15) * (1.0 - kmh / 20.0), 0, 1) : 0.0

        var p = PostOpticsParams()
        p.exposure = smoothExposure
        p.bloomStrength = isStorm ? 0.10 : (preset.isAlpenglow ? 0.34 : 0.27)
        p.skipBloom = settings.quality == .low
        p.skipGodRays = settings.quality == .low
        p.disableMotionBlur = !settings.motionBlur
        p.barrel = ((camera.mode == .gopro || camera.mode == .chesty) && isFP) ? 0.055 : 0.0
        p.speedFactor = speedBlur
        p.angVel = smoothAngVel
        p.rollVel = smoothRollVel
        p.horizonUV = horizonUV
        p.sunUV = sunUV
        p.sunVis = isStorm ? 0.0 : smoothSunVis
        p.godRayVis = godRayVis
        if isStorm {
            p.godRayStrength = 0.0
        } else if preset.isAlpenglow {
            p.godRayStrength = 0.92
        } else if preset.id == "afternoon" {
            p.godRayStrength = 0.80
        } else if preset.id == "morning" {
            p.godRayStrength = 0.65
        } else {
            p.godRayStrength = 0.42
        }
        p.sunColor = preset.sunColor
        p.faceShot = isFP ? activeState.faceShotIntensity : 0.0
        p.lensWet = isFP ? lensWet : 0.0
        p.breath = breathAmt
        p.stormStreaks = isStorm ? 0.88 : 0.0
        p.time = renderTime
        p.aspect = camera.aspect
        return p
    }

    private func buildTelemetry(activeState: SkierState, replayProg: Float, sub: SubsystemContainer) -> HUDTelemetry {
        let avy = sub.simulation.avalanche
        var avyGap: Float = 0
        if avy.active {
            let relX = sub.simulation.state.x - avy.originX
            let relZ = sub.simulation.state.z - avy.originZ
            let along = relX * avy.dirX + relZ * avy.dirZ
            avyGap = max(0, along - avy.frontDist)
        }
        var t = HUDTelemetry()
        t.speedMps = activeState.speed
        t.altitude = activeState.y
        t.slopeDeg = activeState.slopeDeg
        t.edgeDeg = activeState.edgeDeg
        t.fallAngleDeg = activeState.fallAngleDeg
        t.verticalDropped = sub.simulation.state.verticalDropped
        t.maxSpeedKmh = sub.simulation.state.maxSpeedKmh
        t.surfaceId = activeState.surface.id
        t.surfaceLabel = activeState.grounded ? activeState.surface.label : "AIRBORNE"
        t.stanceMode = activeState.grounded ? activeState.stanceMode : "AIR"
        t.isSwitch = activeState.isSwitch
        t.grounded = activeState.grounded
        t.crashed = sub.simulation.state.crashed
        t.crashReason = sub.simulation.state.crashReason
        t.popCharge = sub.simulation.state.popCharge
        t.airTime = sub.simulation.state.airTime
        t.peakAirHeight = sub.simulation.state.peakAirHeight
        t.airYaw = sub.simulation.state.airYaw
        t.airPitch = sub.simulation.state.airPitch
        t.grabbing = sub.simulation.state.grabbing
        t.lastTrickText = sub.simulation.state.lastTrickText
        t.lastLandingQuality = sub.simulation.state.lastLandingQuality
        t.lastTrickTime = sub.simulation.state.lastTrickTime
        t.fps = lastFps
        t.simTime = simTime
        t.started = started
        t.pausedSim = pausedSim
        t.hudMode = hudMode
        t.camMode = camera.mode
        t.lightIdx = lightIdx
        t.zoneIdx = zoneIdx
        t.scoutActive = scoutActive
        t.replayActive = replayActive
        t.replayProgress = replayProg
        t.replayFrameCount = sub.simulation.replayFrames.count
        t.avyActive = avy.active
        t.avySpeedMps = avy.speed
        t.avyGapMeters = avyGap
        t.customToast = customToast
        t.customToastActive = !customToast.isEmpty && simTime < customToastUntil
        t.cameraPosition = camera.position
        t.cameraQuaternion = camera.quaternion
        t.cameraFovDeg = camera.fovDeg
        t.cameraAspect = camera.aspect
        return t
    }

    private func ensureCaptureStagingTexture(width: Int, height: Int, pixelFormat: MTLPixelFormat) -> MTLTexture? {
        if let existing = captureStagingTexture,
           existing.width == width,
           existing.height == height,
           existing.pixelFormat == pixelFormat {
            return existing
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        let tex = device.makeTexture(descriptor: desc)
        tex?.label = "CaptureStaging"
        captureStagingTexture = tex
        return tex
    }

    private static func saveTexturePNG(_ texture: MTLTexture, to path: String) {
        let w = texture.width
        let h = texture.height
        let bytesPerRow = w * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * h)
        bytes.withUnsafeMutableBytes { ptr in
            if let base = ptr.baseAddress {
                texture.getBytes(base, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            }
        }
        // Convert BGRA8 -> RGBA8 if needed
        if texture.pixelFormat == .bgra8Unorm || texture.pixelFormat == .bgra8Unorm_srgb {
            var i = 0
            while i + 3 < bytes.count {
                let b = bytes[i]
                bytes[i] = bytes[i + 2]
                bytes[i + 2] = b
                bytes[i + 3] = 255
                i += 4
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage = CGImage(
                width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: bytesPerRow, space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
              ) else {
            return
        }
        let outURL = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            return
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        CGImageDestinationFinalize(dest)
    }
}
