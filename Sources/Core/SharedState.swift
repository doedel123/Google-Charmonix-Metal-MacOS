import Foundation
import Metal
import simd

// MARK: - Terrain Constants & Surface Classification

public enum ChamonixConstants {
    public static let coreHalf: Float = 6400.0
    public static let outerHalf: Float = 24000.0
    public static let maxElev: Float = 5000.0
    public static let fixedPhysicsDt: Float = 1.0 / 120.0
    public static let scoutDuration: Float = 11.5
}

public enum SurfaceType: String, CaseIterable, Sendable, Codable {
    case powder
    case packed
    case crust
    case ice
    case granite
    case corn

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .powder:  return "COLD POWDER"
        case .packed:  return "WIND PACK"
        case .crust:   return "WIND CRUST"
        case .ice:     return "BLUE ICE"
        case .granite: return "EXPOSED GRANITE"
        case .corn:    return "SPRING CORN"
        }
    }

    public var grip: Float {
        switch self {
        case .powder:  return 0.86
        case .packed:  return 0.96
        case .crust:   return 0.76
        case .ice:     return 0.42
        case .granite: return 0.28
        case .corn:    return 0.92
        }
    }

    public var baseDrag: Float {
        switch self {
        case .powder:  return 0.0155
        case .packed:  return 0.0085
        case .crust:   return 0.0068
        case .ice:     return 0.0038
        case .granite: return 0.024
        case .corn:    return 0.0105
        }
    }

    public var sinkDepth: Float {
        switch self {
        case .powder:  return 0.25
        case .packed:  return 0.09
        case .crust:   return 0.05
        case .ice:     return 0.015
        case .granite: return 0.0
        case .corn:    return 0.08
        }
    }

    public var chatter: Float {
        switch self {
        case .powder:  return 0.08
        case .packed:  return 0.24
        case .crust:   return 0.56
        case .ice:     return 0.95
        case .granite: return 1.25
        case .corn:    return 0.18
        }
    }

    public var sprayMult: Float {
        switch self {
        case .powder:  return 1.35
        case .packed:  return 0.72
        case .crust:   return 0.42
        case .ice:     return 0.16
        case .granite: return 0.06
        case .corn:    return 0.60
        }
    }

    public static func from(id: String) -> SurfaceType {
        SurfaceType(rawValue: id) ?? .powder
    }
}

// MARK: - Metadata & Course Types

public struct AiguilleSpire: Codable, Sendable {
    public let x: Float
    public let z: Float
    public let elev: Float

    public init(x: Float, z: Float, elev: Float) {
        self.x = x
        self.z = z
        self.elev = elev
    }
}

public struct PeakInfo: Codable, Sendable, Identifiable {
    public var id: String { name }
    public let name: String
    public let altLabel: String
    public let x: Float
    public let z: Float
    public let elev: Float

    public init(name: String, altLabel: String, x: Float, z: Float, elev: Float) {
        self.name = name
        self.altLabel = altLabel
        self.x = x
        self.z = z
        self.elev = elev
    }
}

public typealias PeakMarker = PeakInfo

public struct DropZone: Codable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let subtitle: String
    public let x: Float
    public let z: Float
    public let elev: Float
    public let slopeDeg: Float
    public let yaw: Float
    public let compass: String
    public let verticalDrop: Float
    public let traceEnd: [Float]?

    public init(
        id: String, name: String, subtitle: String,
        x: Float, z: Float, elev: Float, slopeDeg: Float,
        yaw: Float, compass: String, verticalDrop: Float, traceEnd: [Float]?
    ) {
        self.id = id
        self.name = name
        self.subtitle = subtitle
        self.x = x
        self.z = z
        self.elev = elev
        self.slopeDeg = slopeDeg
        self.yaw = yaw
        self.compass = compass
        self.verticalDrop = verticalDrop
        self.traceEnd = traceEnd
    }
}

public struct ChamonixMeta: Codable, Sendable {
    public let originLat: Double
    public let originLon: Double
    public let coreHalfSpan: Float
    public let outerHalfSpan: Float
    public let maxElev: Float
    public let aiguilleSpire: AiguilleSpire
    public let peaks: [PeakInfo]
    public let dropZones: [DropZone]

    public init(
        originLat: Double, originLon: Double,
        coreHalfSpan: Float, outerHalfSpan: Float, maxElev: Float,
        aiguilleSpire: AiguilleSpire, peaks: [PeakInfo], dropZones: [DropZone]
    ) {
        self.originLat = originLat
        self.originLon = originLon
        self.coreHalfSpan = coreHalfSpan
        self.outerHalfSpan = outerHalfSpan
        self.maxElev = maxElev
        self.aiguilleSpire = aiguilleSpire
        self.peaks = peaks
        self.dropZones = dropZones
    }
}

public struct GateState: Sendable {
    public var index: Int
    public var position: SIMD3<Float>
    public var passed: Bool
    public init(index: Int = 0, position: SIMD3<Float> = .zero, passed: Bool = false) {
        self.index = index
        self.position = position
        self.passed = passed
    }
}

public struct CourseDefinition: Sendable {
    public var dropZone: DropZone
    public var gates: [GateState]
    public init(dropZone: DropZone, gates: [GateState] = []) {
        self.dropZone = dropZone
        self.gates = gates
    }
}

public struct RunState: Sendable {
    public var started: Bool = false
    public var pausedSim: Bool = false
    public var elapsedTime: Float = 0
    public var verticalDropped: Float = 0
    public var maxSpeedKmh: Float = 0
    public var gatesPassed: Int = 0
    public var totalGates: Int = 0
    public init() {}
}

// MARK: - Terrain Samples & GPU Textures

public struct TerrainRawSample: Sendable {
    public let elev: Float
    public let ao: Float
    public let powder: Float
    public let curv: Float
    public let inCore: Bool

    public init(elev: Float, ao: Float, powder: Float, curv: Float, inCore: Bool) {
        self.elev = elev
        self.ao = ao
        self.powder = powder
        self.curv = curv
        self.inCore = inCore
    }
}

public struct TerrainSample: Sendable {
    public let y: Float
    public let rawY: Float
    public let normal: SIMD3<Float>
    public let gx: Float
    public let gz: Float
    public let slopeRad: Float
    public let slopeDeg: Float
    public let ao: Float
    public let powder: Float
    public let curv: Float
    public let surface: SurfaceType

    public var nx: Float { normal.x }
    public var ny: Float { normal.y }
    public var nz: Float { normal.z }

    public init(
        y: Float, rawY: Float, normal: SIMD3<Float>,
        gx: Float, gz: Float, slopeRad: Float, slopeDeg: Float,
        ao: Float, powder: Float, curv: Float, surface: SurfaceType
    ) {
        self.y = y
        self.rawY = rawY
        self.normal = normal
        self.gx = gx
        self.gz = gz
        self.slopeRad = slopeRad
        self.slopeDeg = slopeDeg
        self.ao = ao
        self.powder = powder
        self.curv = curv
        self.surface = surface
    }
}

public struct TerrainTextures {
    public let coreDem: MTLTexture
    public let coreAux1: MTLTexture
    public let coreAux2: MTLTexture
    public let outerDem: MTLTexture
    public let outerAux1: MTLTexture
    public let outerAux2: MTLTexture
    public let graniteAlbedo: MTLTexture
    public let graniteNormal: MTLTexture
    public let snowNormal: MTLTexture
    public let snowCavity: MTLTexture
    public let iceNormal: MTLTexture
    public let powderPuff: MTLTexture
    public let spruceBough: MTLTexture
    public let skiTopsheet: MTLTexture

    public init(
        coreDem: MTLTexture, coreAux1: MTLTexture, coreAux2: MTLTexture,
        outerDem: MTLTexture, outerAux1: MTLTexture, outerAux2: MTLTexture,
        graniteAlbedo: MTLTexture, graniteNormal: MTLTexture,
        snowNormal: MTLTexture, snowCavity: MTLTexture, iceNormal: MTLTexture,
        powderPuff: MTLTexture, spruceBough: MTLTexture, skiTopsheet: MTLTexture
    ) {
        self.coreDem = coreDem
        self.coreAux1 = coreAux1
        self.coreAux2 = coreAux2
        self.outerDem = outerDem
        self.outerAux1 = outerAux1
        self.outerAux2 = outerAux2
        self.graniteAlbedo = graniteAlbedo
        self.graniteNormal = graniteNormal
        self.snowNormal = snowNormal
        self.snowCavity = snowCavity
        self.iceNormal = iceNormal
        self.powderPuff = powderPuff
        self.spruceBough = spruceBough
        self.skiTopsheet = skiTopsheet
    }
}

public protocol TerrainQueryProtocol: AnyObject {
    var meta: ChamonixMeta { get }
    var coreSize: Int { get }
    var outerSize: Int { get }
    var textures: TerrainTextures { get }
    func sampleRawDEM(x: Float, z: Float) -> TerrainRawSample
    func sampleHeight(x: Float, z: Float) -> Float
    func sampleTerrain(x: Float, z: Float) -> TerrainSample
}

// MARK: - World Colliders & Kickers

public enum PropColliderKind: String, Sendable {
    case tree
    case rock
    case serac
    case station
    case stake
}

public struct PropCollider: Sendable {
    public let x: Float
    public let y: Float
    public let z: Float
    public let r: Float
    public let h: Float
    public let kind: PropColliderKind

    public init(x: Float, y: Float, z: Float, r: Float, h: Float, kind: PropColliderKind) {
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.h = h
        self.kind = kind
    }
}

public struct WindLipKicker: Sendable {
    public let x: Float
    public let y: Float
    public let z: Float
    public let fwdX: Float
    public let fwdZ: Float
    public let radius: Float
    public let rampSlope: Float
    public let boost: Float
    public let width: Float

    public init(
        x: Float, y: Float, z: Float,
        fwdX: Float, fwdZ: Float,
        radius: Float, rampSlope: Float = 0.45,
        boost: Float, width: Float
    ) {
        self.x = x
        self.y = y
        self.z = z
        self.fwdX = fwdX
        self.fwdZ = fwdZ
        self.radius = radius
        self.rampSlope = rampSlope
        self.boost = boost
        self.width = width
    }
}

public typealias KickerRamp = WindLipKicker

// MARK: - Avalanche, Skier, Replay, Events

public struct AvalancheState: Sendable {
    public var active: Bool = false
    public var strength: Float = 0
    public var originX: Float = 0
    public var originZ: Float = 0
    public var dirX: Float = 0
    public var dirZ: Float = 1
    public var frontDist: Float = 0
    public var width: Float = 28
    public var speed: Float = 20
    public var age: Float = 0
    public var maxDist: Float = 920
    public var proximity: Float = 0
    public var engulfed: Bool = false
    public var cooldown: Float = 0

    public init() {}
}

public struct SkierState: Sendable {
    public var x: Float = 0
    public var y: Float = 0
    public var z: Float = 0
    public var vx: Float = 0
    public var vy: Float = 0
    public var vz: Float = 0
    public var speed: Float = 0
    public var yaw: Float = 0
    public var edgeAngle: Float = 0
    public var crouch: Float = 0
    public var popCharge: Float = 0
    public var foreAft: Float = 0
    public var slipAngle: Float = 0
    public var turnLead: Float = 0
    public var shiftyAngle: Float = 0
    public var isSwitch: Bool = false
    public var grounded: Bool = true
    public var crashed: Bool = false
    public var crashTimer: Float = 0
    public var crashReason: String = ""
    public var grabbing: Bool = false
    public var airTime: Float = 0
    public var peakAirHeight: Float = 0
    public var launchY: Float = 0
    public var airPitch: Float = 0
    public var airYaw: Float = 0
    public var airRoll: Float = 0
    public var angPitch: Float = 0
    public var angYaw: Float = 0
    public var angRoll: Float = 0
    public var grabDuration: Float = 0
    public var surface: SurfaceType = .powder
    public var slopeDeg: Float = 0
    public var terrainY: Float = 0
    public var normal: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
    public var quat: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    public var bodyQuat: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    public var rollContact: Float = 0
    public var pitchContact: Float = 0

    // Telemetry / Stats
    public var startElev: Float = 0
    public var verticalDropped: Float = 0
    public var maxSpeedKmh: Float = 0
    public var lastTrickText: String = ""
    public var lastTrickTime: Float = -99
    public var lastLandingQuality: String = ""
    public var faceShotIntensity: Float = 0
    public var camShake: Float = 0
    public var landImpact: Float = 0
    public var sluffTrigger: Float = 0
    public var sprayRate: Float = 0
    public var breathing: Float = 0

    // Tactile skiing feel & body dynamics
    public var polePlantL: Float = 0
    public var polePlantR: Float = 0
    public var skatePhase: Float = 0
    public var skatePush: Float = 0
    public var insideUnweight: Float = 0
    public var scissorSpread: Float = 0
    public var bowWave: Float = 0

    // Fall-line / edging / poling telemetry
    public var edgeDeg: Float = 0
    public var fallAngleDeg: Float = 0
    public var uphillEdge: Float = 0
    public var sideslip: Float = 0
    public var washOut: Float = 0
    public var carveLoadG: Float = 1.0
    public var stanceMode: String = "FLAT"
    public var poleStrokePulse: Float = 0
    public var stepBounce: Float = 0

    public init() {}
}

public struct ReplayFrame: Sendable {
    public var x: Float
    public var y: Float
    public var z: Float
    public var vx: Float
    public var vy: Float
    public var vz: Float
    public var terrainY: Float
    public var normal: SIMD3<Float>
    public var quat: simd_quatf
    public var yaw: Float
    public var edgeAngle: Float
    public var crouch: Float
    public var foreAft: Float
    public var slipAngle: Float
    public var turnLead: Float
    public var shiftyAngle: Float
    public var landImpact: Float
    public var grabbing: Bool
    public var speed: Float
    public var grounded: Bool
    public var crashed: Bool
    public var isSwitch: Bool
    public var slopeDeg: Float
    public var surfId: String
    public var sprayRate: Float
    public var polePlantL: Float
    public var polePlantR: Float
    public var skatePhase: Float
    public var skatePush: Float
    public var insideUnweight: Float
    public var scissorSpread: Float
    public var bowWave: Float
    public var edgeDeg: Float
    public var fallAngleDeg: Float
    public var stanceMode: String

    public init(from s: SkierState) {
        self.x = s.x
        self.y = s.y
        self.z = s.z
        self.vx = s.vx
        self.vy = s.vy
        self.vz = s.vz
        self.terrainY = s.terrainY
        self.normal = s.normal
        self.quat = s.quat
        self.yaw = s.yaw
        self.edgeAngle = s.edgeAngle
        self.crouch = s.crouch
        self.foreAft = s.foreAft
        self.slipAngle = s.slipAngle
        self.turnLead = s.turnLead
        self.shiftyAngle = s.shiftyAngle
        self.landImpact = s.landImpact
        self.grabbing = s.grabbing
        self.speed = s.speed
        self.grounded = s.grounded
        self.crashed = s.crashed
        self.isSwitch = s.isSwitch
        self.slopeDeg = s.slopeDeg
        self.surfId = s.surface.id
        self.sprayRate = s.sprayRate
        self.polePlantL = s.polePlantL
        self.polePlantR = s.polePlantR
        self.skatePhase = s.skatePhase
        self.skatePush = s.skatePush
        self.insideUnweight = s.insideUnweight
        self.scissorSpread = s.scissorSpread
        self.bowWave = s.bowWave
        self.edgeDeg = s.edgeDeg
        self.fallAngleDeg = s.fallAngleDeg
        self.stanceMode = s.stanceMode
    }

    public func toSkierState() -> SkierState {
        var s = SkierState()
        s.x = x; s.y = y; s.z = z
        s.vx = vx; s.vy = vy; s.vz = vz
        s.terrainY = terrainY
        s.normal = normal
        s.quat = quat
        s.bodyQuat = quat
        s.yaw = yaw
        s.edgeAngle = edgeAngle
        s.crouch = crouch
        s.foreAft = foreAft
        s.slipAngle = slipAngle
        s.turnLead = turnLead
        s.shiftyAngle = shiftyAngle
        s.landImpact = landImpact
        s.grabbing = grabbing
        s.speed = speed
        s.grounded = grounded
        s.crashed = crashed
        s.isSwitch = isSwitch
        s.slopeDeg = slopeDeg
        s.surface = SurfaceType.from(id: surfId)
        s.sprayRate = sprayRate
        s.polePlantL = polePlantL
        s.polePlantR = polePlantR
        s.skatePhase = skatePhase
        s.skatePush = skatePush
        s.insideUnweight = insideUnweight
        s.scissorSpread = scissorSpread
        s.bowWave = bowWave
        s.edgeDeg = edgeDeg
        s.fallAngleDeg = fallAngleDeg
        s.stanceMode = stanceMode
        return s
    }
}

public enum PhysicsEvent: Sendable {
    case track(
        lA: SIMD2<Float>, lB: SIMD2<Float>,
        rA: SIMD2<Float>, rB: SIMD2<Float>,
        radius: Float, depth: Float, berm: Float, iceGlint: Float, skew: Float
    )
    case spray(
        x: Float, y: Float, z: Float,
        vx: Float, vy: Float, vz: Float,
        rate: Float, sluff: Bool, slopeX: Float, slopeZ: Float
    )
    case polePlant(side: Int, x: Float, y: Float, z: Float, strength: Float, powder: Bool)
    case treeWhoosh(pan: Float, closeness: Float, speed: Float)
    case bootCreak(strength: Float)
    case sluffStream(x: Float, y: Float, z: Float, dirX: Float, dirZ: Float, slopeDeg: Float, strength: Float)
    case pop(x: Float, y: Float, z: Float, charge: Float)
    case land(x: Float, y: Float, z: Float, impact: Float, powder: Bool, sketchy: Bool)
    case crash(x: Float, y: Float, z: Float, impact: Float, reason: String)
    case crashTrail(x: Float, y: Float, z: Float, speed: Float)
    case avalancheStart(x: Float, y: Float, z: Float, width: Float)
}

// MARK: - Camera, Input, Settings, HUD, PostOptics

public enum CameraMode: String, CaseIterable, Sendable {
    case gopro
    case chesty
    case clean

    public var label: String {
        switch self {
        case .gopro:  return "HELMET POV"
        case .chesty: return "CHESTY CAM"
        case .clean:  return "VISOR POV"
        }
    }
}

public struct CameraState: Sendable {
    public var position: SIMD3<Float> = .zero
    public var quaternion: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    public var fovDeg: Float = 86.0
    public var nearZ: Float = 0.09
    public var farZ: Float = 52000.0
    public var aspect: Float = 16.0 / 9.0
    public var mode: CameraMode = .gopro
    public var lookYawOff: Float = 0
    public var lookPitchOff: Float = 0
    public var isThirdPerson: Bool = false
    public var isChesty: Bool = false

    public var viewMatrix: simd_float4x4 {
        let rot = simd_float4x4(quaternion)
        let trans = matrixTranslation(position)
        return simd_inverse(trans * rot)
    }

    public var projectionMatrix: simd_float4x4 {
        matrixPerspectiveRightHand(
            fovyRadians: degToRad(fovDeg),
            aspect: aspect,
            nearZ: nearZ,
            farZ: farZ
        )
    }

    public init() {}
}

public struct InputState: Sendable {
    public var left: Bool = false
    public var right: Bool = false
    public var tuck: Bool = false
    public var brake: Bool = false
    public var poleLeft: Bool = false
    public var poleRight: Bool = false
    public var poleBoth: Bool = false
    public var cork: Bool = false
    public var jumpHeld: Bool = false
    public var grabHeld: Bool = false
    public var freeLookHeld: Bool = false
    public var mouseSteer: Float = 0
    public var gamepadSteer: Float = 0
    public var gamepadFore: Float = 0
    public var gamepadBrake: Float = 0
    public var gamepadCork: Float = 0
    public var smoothCarve: Bool = false
    public var headSteerEnabled: Bool = false
    public var lookDeltaYaw: Float = 0
    public var lookDeltaPitch: Float = 0
    public var activeKeys: Set<String> = []

    public init() {}
}

public enum QualityMode: String, CaseIterable, Codable, Sendable {
    case auto
    case high
    case med
    case low
}

public struct SimulationSettings: Codable, Sendable {
    public var quality: QualityMode = .auto
    public var volume: Float = 0.85
    public var muted: Bool = false
    public var lookSens: Float = 1.0
    public var invertY: Bool = false
    public var motionBlur: Bool = true
    public var smoothCarve: Bool = false
    public var avalanchesEnabled: Bool = true

    public init() {}
}

public struct HUDTelemetry: Sendable {
    public var speedMps: Float = 0
    public var altitude: Float = 0
    public var slopeDeg: Float = 0
    public var edgeDeg: Float = 0
    public var fallAngleDeg: Float = 0
    public var verticalDropped: Float = 0
    public var maxSpeedKmh: Float = 0
    public var surfaceId: String = SurfaceType.powder.id
    public var surfaceLabel: String = SurfaceType.powder.label
    public var stanceMode: String = "FLAT"
    public var isSwitch: Bool = false
    public var grounded: Bool = true
    public var crashed: Bool = false
    public var crashReason: String = ""
    public var popCharge: Float = 0
    public var airTime: Float = 0
    public var peakAirHeight: Float = 0
    public var airYaw: Float = 0
    public var airPitch: Float = 0
    public var grabbing: Bool = false
    public var lastTrickText: String = ""
    public var lastLandingQuality: String = ""
    public var lastTrickTime: Float = -99
    public var fps: Float = 60
    public var simTime: Float = 0

    // Modes & Banners
    public var started: Bool = false
    public var pausedSim: Bool = false
    public var hudMode: Int = 0 // 0=Full, 1=RAW REC, 2=Hidden
    public var camMode: CameraMode = .gopro
    public var lightIdx: Int = 0
    public var zoneIdx: Int = 0
    public var scoutActive: Bool = false
    public var replayActive: Bool = false
    public var replayProgress: Float = 0
    public var replayFrameCount: Int = 0
    public var avyActive: Bool = false
    public var avySpeedMps: Float = 0
    public var avyGapMeters: Float = 0
    public var customToast: String = ""
    public var customToastActive: Bool = false

    // Camera state for compass peak pins
    public var cameraPosition: SIMD3<Float> = .zero
    public var cameraQuaternion: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    public var cameraFovDeg: Float = 86
    public var cameraAspect: Float = 16.0 / 9.0

    public init() {}
}

public struct PostOpticsParams: Sendable {
    public var exposure: Float = 0.72
    public var bloomStrength: Float = 0.27
    public var skipBloom: Bool = false
    public var godRayVis: Float = 0.0
    public var godRayStrength: Float = 0.80
    public var skipGodRays: Bool = false
    public var sunUV: SIMD2<Float> = SIMD2<Float>(0.5, 0.5)
    public var sunVis: Float = 0.0
    public var sunColor: SIMD3<Float> = SIMD3<Float>(1.0, 0.91, 0.78)
    public var horizonUV: SIMD2<Float> = SIMD2<Float>(0.5, 0.56)
    public var barrel: Float = 0.055
    public var speedFactor: Float = 0.0
    public var angVel: SIMD2<Float> = .zero
    public var rollVel: Float = 0.0
    public var disableMotionBlur: Bool = false
    public var faceShot: Float = 0.0
    public var lensWet: Float = 0.0
    public var breath: Float = 0.0
    public var stormStreaks: Float = 0.0
    public var time: Float = 0.0
    public var aspect: Float = 16.0 / 9.0

    public init() {}
}

// MARK: - Lighting Lighting/Render Context

public struct RenderPassContext {
    public let device: MTLDevice
    public let commandBuffer: MTLCommandBuffer
    public var frameUniforms: FrameUniforms
    public var frameUniformBuffer: MTLBuffer
    public let camera: CameraState
    public let preset: LightingPreset
    public let sunDir: SIMD3<Float>
    public let mtnSunVis: Float
    public let time: Float
    public let dt: Float
    public let isThirdPerson: Bool
    public let isChesty: Bool
    public let viewportSize: SIMD2<Float>
    public let shadowMapTexture: MTLTexture?
    public let msaaSampleCount: Int

    public init(
        device: MTLDevice,
        commandBuffer: MTLCommandBuffer,
        frameUniforms: FrameUniforms,
        frameUniformBuffer: MTLBuffer,
        camera: CameraState,
        preset: LightingPreset,
        sunDir: SIMD3<Float>,
        mtnSunVis: Float,
        time: Float,
        dt: Float,
        isThirdPerson: Bool,
        isChesty: Bool,
        viewportSize: SIMD2<Float>,
        shadowMapTexture: MTLTexture?,
        msaaSampleCount: Int = 4
    ) {
        self.device = device
        self.commandBuffer = commandBuffer
        self.frameUniforms = frameUniforms
        self.frameUniformBuffer = frameUniformBuffer
        self.camera = camera
        self.preset = preset
        self.sunDir = sunDir
        self.mtnSunVis = mtnSunVis
        self.time = time
        self.dt = dt
        self.isThirdPerson = isThirdPerson
        self.isChesty = isChesty
        self.viewportSize = viewportSize
        self.shadowMapTexture = shadowMapTexture
        self.msaaSampleCount = msaaSampleCount
    }
}

// MARK: - Subsystem Protocols

public protocol SnowTracksSubsystem: AnyObject {
    var worldSpan: Float { get }
    var resolution: Int { get }
    var originX: Float { get }
    var originZ: Float { get }
    var macroSpan: Float { get }
    var macroResolution: Int { get }
    var macroOriginX: Float { get }
    var macroOriginZ: Float { get }
    var trackTexture: MTLTexture { get }
    var macroTrackTexture: MTLTexture { get }

    func reset(x: Float, z: Float)
    func recenterIfNeeded(skierX: Float, skierZ: Float, decayAmt: Float)
    func stampTwinSkis(
        lA: SIMD2<Float>, lB: SIMD2<Float>,
        rA: SIMD2<Float>, rB: SIMD2<Float>,
        radius: Float, depth: Float, berm: Float, iceGlint: Float, skew: Float
    )
    func stampCapsule(
        ax: Float, az: Float, bx: Float, bz: Float,
        radius: Float, depth: Float, berm: Float, iceGlint: Float, skew: Float
    )
    func stampPoleDimple(x: Float, z: Float, radius: Float, depth: Float)
    func encodeRenderPasses(commandBuffer: MTLCommandBuffer)
}

public protocol TerrainSubsystem: AnyObject {
    func update(
        cameraPos: SIMD3<Float>,
        focusXZ: SIMD2<Float>,
        preset: LightingPreset,
        sunDir: SIMD3<Float>,
        time: Float,
        trackBuffer: any SnowTracksSubsystem,
        avalanche: AvalancheState
    )
    func renderHDRPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext, trackBuffer: any SnowTracksSubsystem)
}

public protocol SkySubsystem: AnyObject {
    func update(cameraPos: SIMD3<Float>, preset: LightingPreset, sunDir: SIMD3<Float>, time: Float)
    func renderSkyPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
    func renderValleyCloudsPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
}

public protocol RigSubsystem: AnyObject {
    func update(
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
    )
    func renderShadowPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
    func renderGroundShadowsPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
    func renderHDRPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
}

public protocol AlpinePropsSubsystem: AnyObject {
    var colliders: [PropCollider] { get }
    var kickers: [WindLipKicker] { get }
    func update(cameraPos: SIMD3<Float>, preset: LightingPreset, sunDir: SIMD3<Float>, time: Float)
    func renderShadowPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
    func renderGroundShadowsPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
    func renderHDRPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
}

public protocol ParticlesSubsystem: AnyObject {
    func emitPuff(
        x: Float, y: Float, z: Float,
        vx: Float, vy: Float, vz: Float,
        size: Float, expand: Float, peakAlpha: Float, life: Float, kind: Float
    )
    func emitBurst(
        x: Float, y: Float, z: Float,
        count: Int, speed: Float, sizeMin: Float, sizeMax: Float,
        expand: Float, peakAlpha: Float, life: Float, kind: Float
    )
    func emitCarveRoosterTail(
        x: Float, y: Float, z: Float,
        vx: Float, vy: Float, vz: Float,
        rate: Float, sluff: Bool, slopeX: Float, slopeZ: Float
    )
    func emitAvalancheWave(avalanche: AvalancheState, terrainData: any TerrainQueryProtocol, dt: Float)
    func clearAvalanchePuffs()
    func update(
        dt: Float,
        cameraPos: SIMD3<Float>,
        viewportHeightPx: Float,
        fovDeg: Float,
        preset: LightingPreset,
        sunDir: SIMD3<Float>,
        time: Float
    )
    func renderHDRPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext)
}

public protocol PostOpticsSubsystem: AnyObject {
    var sceneMSAATexture: MTLTexture? { get }
    var sceneHDRTexture: MTLTexture { get }
    var sceneDepthTexture: MTLTexture { get }
    var msaaSampleCount: Int { get }
    func setMSAA(samples: Int)
    func resize(width: Int, height: Int)
    func render(
        commandBuffer: MTLCommandBuffer,
        drawableTexture: MTLTexture,
        params: PostOpticsParams,
        context: RenderPassContext
    )
}

public protocol AudioSubsystem: AnyObject {
    func unlock()
    func setVolume(_ volume: Float)
    func setMuted(_ muted: Bool)
    func update(state: SkierState, preset: LightingPreset, avalanche: AvalancheState, dt: Float)
    func handleEvent(_ event: PhysicsEvent)
}

public protocol SimulationSubsystem: AnyObject {
    var terrainData: any TerrainQueryProtocol { get }
    var state: SkierState { get }
    var avalanche: AvalancheState { get }
    var events: [PhysicsEvent] { get set }
    var replayFrames: [ReplayFrame] { get }
    var avalanchesEnabled: Bool { get set }
    var lookYawOff: Float { get set }
    var lookPitchOff: Float { get set }
    var headSteerEnabled: Bool { get set }
    var smoothCarve: Bool { get set }

    func spawn(zone: DropZone)
    func step(dt: Float, time: Float, input: InputState)
    func applyCamera(camera: inout CameraState, mode: CameraMode, time: Float, dt: Float)
    func enforceCameraClearance(camera: inout CameraState, minClearance: Float, lockRoll: Bool)
    func addHeadLookDelta(dYaw: Float, dPitch: Float)
    func resetHeadLook()
    func toggleSwitch()
    func triggerPoleStroke(side: Int)
    func triggerAvalanche()
    func standUp()
    func clearEvents()
}
