import Foundation
import Metal
import simd

public final class AlpinePropsSystem: AlpinePropsSubsystem {
    public let colliders: [PropCollider]
    public let kickers: [WindLipKicker]

    private let device: MTLDevice
    private let terrainData: any TerrainQueryProtocol
    private let library: MTLLibrary

    // Pipelines keyed by MSAA sample count
    private var propPipelines: [Int: MTLRenderPipelineState] = [:]
    private var sprucePipelines: [Int: MTLRenderPipelineState] = [:]
    private var stationPipelines: [Int: MTLRenderPipelineState] = [:]
    private var cablePipelines: [Int: MTLRenderPipelineState] = [:]
    private var groundShadowPipelines: [Int: MTLRenderPipelineState] = [:]
    private var shadowDepthPipeline: MTLRenderPipelineState?

    private let depthWriteLessState: MTLDepthStencilState
    private let depthReadOnlyLessEqualState: MTLDepthStencilState

    // Instanced prop meshes & buffers
    private struct InstancedPropBatch {
        let mesh: PropMesh
        let instanceBuffer: MTLBuffer
        let instanceCount: Int
        let mode: Float
    }
    private var spruceBatches: [InstancedPropBatch] = []
    private var propBatches: [InstancedPropBatch] = [] // rocks, seracs, kickers, midi pitons, stake mounds

    // Ground shadows (single draw call)
    private let shadowQuadMesh: ProceduralMesh
    private let shadowInstanceBuffer: MTLBuffer?
    private let shadowInstanceCount: Int

    // Static station + stake parts
    private struct StationPart {
        let mesh: ProceduralMesh
        let modelMatrix: simd_float4x4
        let normalMatrix: simd_float4x4
        let baseColor: SIMD3<Float>
        let roughness: Float
        let metalness: Float
        let emissiveColor: SIMD3<Float>
        var emissiveIntensity: Float
        let role: StationRole
        let castShadowMap: Bool
    }
    private enum StationRole {
        case solid
        case galleryGlass
        case beacon
        case stake
    }
    private var stationParts: [StationPart] = []

    // Animated cable car parts & cables
    private struct CableCarPartSpec {
        let mesh: ProceduralMesh
        let localOffset: SIMD3<Float>
        let baseColor: SIMD3<Float>
        let roughness: Float
        let metalness: Float
        let emissiveColor: SIMD3<Float>
        let isWindow: Bool
    }
    private var cableCarPartSpecs: [CableCarPartSpec] = []
    private var cableMeshes: [MTLBuffer] = []
    private let cableVertexCount = 33
    private var cableStationBasePos: SIMD3<Float> = .zero
    private var cablePlanOffset: SIMD3<Float> = .zero
    private var cableAzimuth: Float = 0

    // Animated state updated each frame
    private var cabinUpMatrix: simd_float4x4 = matrix_identity_float4x4
    private var cabinDownMatrix: simd_float4x4 = matrix_identity_float4x4
    private var galleryEmissive: Float = 0.55
    private var cabinWinEmissive: Float = 0.30
    private var beaconEmissive: Float = 0.85

    public init(device: MTLDevice, terrainData: any TerrainQueryProtocol) throws {
        self.device = device
        self.terrainData = terrainData
        self.library = try ShaderLibrary.loadLibrary(device: device, named: "AlpineProps")

        // Depth states
        let dsWrite = MTLDepthStencilDescriptor()
        dsWrite.depthCompareFunction = .less
        dsWrite.isDepthWriteEnabled = true
        self.depthWriteLessState = device.makeDepthStencilState(descriptor: dsWrite)!

        let dsRead = MTLDepthStencilDescriptor()
        dsRead.depthCompareFunction = .lessEqual
        dsRead.isDepthWriteEnabled = false
        self.depthReadOnlyLessEqualState = device.makeDepthStencilState(descriptor: dsRead)!

        // Shadow quad geometry
        let (sqVerts, sqIdx) = MeshPrimitives.makePlaneXZ(width: 1.0, depth: 1.0, segX: 4, segZ: 10)
        self.shadowQuadMesh = MeshPrimitives.upload(device: device, vertices: sqVerts, indices: sqIdx, label: "GroundShadowQuad")!

        var builtColliders: [PropCollider] = []
        var builtKickers: [WindLipKicker] = []
        var builtShadows: [ShadowInstanceData] = []
        var builtPropBatches: [InstancedPropBatch] = []
        var builtSpruceBatches: [InstancedPropBatch] = []
        var builtStationParts: [StationPart] = []
        var builtCableCarSpecs: [CableCarPartSpec] = []
        var builtCableMeshes: [MTLBuffer] = []

        // 1. Aiguille du Midi Station + Pitons + Animated Cable Cars
        let spire = terrainData.meta.aiguilleSpire
        let baseY = terrainData.sampleHeight(x: spire.x, z: spire.z)
        let stationOrigin = SIMD3<Float>(spire.x, baseY, spire.z)
        self.cableStationBasePos = stationOrigin

        if let pitonMain = AlpinePropsGeometry.buildStratifiedPitonMesh(device: device, radius: 33, height: 80, seed: 17, label: "MidiPitonMain"),
           let pitonNorth = AlpinePropsGeometry.buildStratifiedPitonMesh(device: device, radius: 26, height: 66, seed: 41, label: "MidiPitonNorth") {
            let mMain = matrixTranslation(stationOrigin + SIMD3<Float>(0, 14, 0)) * matrixScale(SIMD3<Float>(1.08, 1.0, 0.88))
            let mNorth = matrixTranslation(stationOrigin + SIMD3<Float>(38, 6, -26)) * matrixScale(SIMD3<Float>(0.95, 1.0, 0.85))
            if let bMain = Self.makeSinglePropBatch(device: device, mesh: pitonMain, matrix: mMain, mode: 0) {
                builtPropBatches.append(bMain)
            }
            if let bNorth = Self.makeSinglePropBatch(device: device, mesh: pitonNorth, matrix: mNorth, mode: 0) {
                builtPropBatches.append(bNorth)
            }
        }
        let cableInfo = Self.buildMidiStationArchitecture(
            device: device,
            terrainData: terrainData,
            stationOrigin: stationOrigin,
            cableVertexCount: cableVertexCount,
            stationParts: &builtStationParts,
            cableCarPartSpecs: &builtCableCarSpecs,
            cableMeshes: &builtCableMeshes
        )
        self.cablePlanOffset = cableInfo.planOffset
        self.cableAzimuth = cableInfo.azimuth

        // 2. Drop-in wooden safety stakes + snow collars
        Self.buildDropInStakes(
            device: device,
            terrainData: terrainData,
            stationParts: &builtStationParts,
            propBatches: &builtPropBatches,
            shadows: &builtShadows
        )

        // 3. Volumetric 3D Norway Spruces (4 LOD pools, Mulberry32(4808) deterministic placement)
        var rand = Mulberry32(seed: 4808)

        struct TreeItem {
            let x: Float, y: Float, z: Float, sc: Float, hScale: Float, yaw: Float, tint: Float
        }
        var listClose: [TreeItem] = []
        var listTall: [TreeItem] = []
        var listBroad: [TreeItem] = []
        var listFar: [TreeItem] = []

        let planZone = terrainData.meta.dropZones.first(where: { $0.id == "plan_aiguille" })
        let planX: Float = planZone?.x ?? -1613
        let planZ: Float = planZone?.z ?? -2487
        let planYaw: Float = planZone?.yaw ?? -2.1158

        @discardableResult
        func pushTree(_ x: Float, _ z: Float, forceHero: Bool = false) -> Bool {
            let s = terrainData.sampleTerrain(x: x, z: z)
            if s.rawY < 1040 || s.rawY > 2265 || s.slopeDeg > 38.5 { return false }
            if hypotf(x - planX, z - planZ) < 20 { return false }
            let altScale = clamp((2340.0 - s.rawY) / 480.0, 0.58, 1.38)
            let sc = (0.78 + rand.next() * 0.70) * altScale
            let hScale = sc * (0.90 + rand.next() * 0.28)
            let yaw = rand.next() * .pi * 2.0
            let tint = rand.next()
            let slopeSink: Float = 0.62 + max(0.0, s.slopeDeg - 16.0) * 0.032
            let item = TreeItem(x: x, y: s.y - slopeSink * sc, z: z, sc: sc, hScale: hScale, yaw: yaw, tint: tint)

            let dPlan = hypotf(x - planX, z - planZ)
            if (forceHero || dPlan < 230) && listClose.count < 130 {
                listClose.append(item)
                builtColliders.append(PropCollider(x: x, y: s.y, z: z, r: 0.75 * sc, h: 10.5 * hScale, kind: .tree))
            } else if dPlan < 680 && (listTall.count + listBroad.count) < 880 {
                if rand.next() < 0.58 {
                    listTall.append(item)
                } else {
                    listBroad.append(item)
                }
                builtColliders.append(PropCollider(x: x, y: s.y, z: z, r: 0.75 * sc, h: 10.5 * hScale, kind: .tree))
            } else if listFar.count < 1350 {
                listFar.append(item)
                if listFar.count < 700 {
                    builtColliders.append(PropCollider(x: x, y: s.y, z: z, r: 0.75 * sc, h: 10.5 * hScale, kind: .tree))
                }
            } else {
                return false
            }

            builtShadows.append(ShadowInstanceData(
                posRadius: SIMD4<Float>(x, s.y, z, 3.15 * sc),
                normalHeight: SIMD4<Float>(s.nx, s.ny, s.nz, 12.6 * hScale),
                extra: SIMD4<Float>(0.85, 0.0, rand.next(), 0.0)
            ))
            return true
        }

        let pFwdX = sinf(planYaw), pFwdZ = -cosf(planYaw)
        let pRX = cosf(planYaw), pRZ = sinf(planYaw)
        let heroSeeds: [(Float, Float)] = [
            (28, -14), (34, 16), (42, -22), (48, 12), (56, -15), (64, 24),
            (72, -28), (76, 17), (84, -12), (92, 26), (104, -19), (116, 14),
        ]
        for (al, ac) in heroSeeds {
            pushTree(planX + pFwdX * al + pRX * ac, planZ + pFwdZ * al + pRZ * ac, forceHero: true)
        }

        var iTree = 0
        while iTree < 5600 && (listClose.count + listTall.count + listBroad.count + listFar.count) < 2350 {
            defer { iTree += 1 }
            let x: Float
            let z: Float
            if iTree < 1150 {
                let along = 32 + rand.next() * 840
                let across = (rand.next() - 0.5) * 640
                if along < 95 && abs(across) < 14 { continue }
                x = planX + pFwdX * along + pRX * across
                z = planZ + pFwdZ * along + pRZ * across
            } else if iTree < 3000 {
                x = planX - 1650 + rand.next() * 2600
                z = planZ - 2200 + rand.next() * 2600
            } else {
                x = -5600 + rand.next() * 11200
                z = -5800 + rand.next() * 5200
            }
            pushTree(x, z, forceHero: false)
        }

        func mountSprucePool(mesh: PropMesh?, list: [TreeItem]) {
            guard let mesh, !list.isEmpty else { return }
            var instances: [PropInstanceData] = []
            instances.reserveCapacity(list.count)
            for t in list {
                let rx = (rand.next() - 0.5) * 0.03
                let rz = (rand.next() - 0.5) * 0.03
                let q = quatFromEulerXYZ(x: rx, y: t.yaw, z: rz)
                let mat = matrixTRS(
                    translation: SIMD3<Float>(t.x, t.y, t.z),
                    rotation: q,
                    scale: SIMD3<Float>(t.sc, t.hScale, t.sc)
                )
                instances.append(PropInstanceData(
                    instanceMatrix: mat,
                    propParams: SIMD4<Float>(0, 0, 0, t.tint)
                ))
            }
            let bufLen = instances.count * MemoryLayout<PropInstanceData>.stride
            if let buf = instances.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: bufLen, options: .storageModeShared)
            }) {
                builtSpruceBatches.append(InstancedPropBatch(mesh: mesh, instanceBuffer: buf, instanceCount: instances.count, mode: 0))
            }
        }

        mountSprucePool(mesh: AlpinePropsGeometry.buildVolumetricSpruceMesh(device: device, seed: 4808, style: .close, label: "SpruceClose"), list: listClose)
        mountSprucePool(mesh: AlpinePropsGeometry.buildVolumetricSpruceMesh(device: device, seed: 5112, style: .tall, label: "SpruceTall"), list: listTall)
        mountSprucePool(mesh: AlpinePropsGeometry.buildVolumetricSpruceMesh(device: device, seed: 7321, style: .broad, label: "SpruceBroad"), list: listBroad)
        mountSprucePool(mesh: AlpinePropsGeometry.buildVolumetricSpruceMesh(device: device, seed: 9104, style: .far, label: "SpruceFar"), list: listFar)

        // 4. Protogine Granite Boulders
        struct RockItem {
            let x: Float, y: Float, z: Float, sc: Float, sy: Float, rx: Float, ry: Float, rz: Float
        }
        var rockList: [RockItem] = []
        for dz in terrainData.meta.dropZones {
            let fwdX = sinf(dz.yaw), fwdZ = -cosf(dz.yaw)
            let rightX = cosf(dz.yaw), rightZ = sinf(dz.yaw)
            for _ in 0..<125 {
                let along = 68 + rand.next() * 1850
                let across = (rand.next() - 0.5) * 760
                if along < 120 && abs(across) < 34 { continue }
                let x = dz.x + fwdX * along + rightX * across
                let z = dz.z + fwdZ * along + rightZ * across
                if abs(x) > 5900 || abs(z) > 5900 { continue }
                let s = terrainData.sampleTerrain(x: x, z: z)
                if s.slopeDeg < 18 { continue }
                let sc = 1.6 + powf(rand.next(), 2.0) * 6.2
                let sy = sc * (0.80 + rand.next() * 0.38)
                let rockSink = sc * (0.28 + max(0.0, s.slopeDeg - 18.0) * 0.010)
                rockList.append(RockItem(
                    x: x, y: s.y - rockSink, z: z, sc: sc, sy: sy,
                    rx: (rand.next() - 0.5) * 0.24,
                    ry: rand.next() * .pi * 2.0,
                    rz: (rand.next() - 0.5) * 0.24
                ))
                if rockList.count < 800 {
                    builtColliders.append(PropCollider(x: x, y: s.y, z: z, r: sc * 0.78, h: sy * 0.95, kind: .rock))
                }
                builtShadows.append(ShadowInstanceData(
                    posRadius: SIMD4<Float>(x, s.y, z, sc * 0.96),
                    normalHeight: SIMD4<Float>(s.nx, s.ny, s.nz, sy * 1.15),
                    extra: SIMD4<Float>(0.58, 1.0, rand.next(), 0.0)
                ))
            }
        }
        if let rockMesh = AlpinePropsGeometry.buildRockMesh(device: device, seed: 77), !rockList.isEmpty {
            var rockInstances: [PropInstanceData] = []
            rockInstances.reserveCapacity(rockList.count)
            for r in rockList {
                let q = quatFromEulerXYZ(x: r.rx, y: r.ry, z: r.rz)
                let mat = matrixTRS(
                    translation: SIMD3<Float>(r.x, r.y, r.z),
                    rotation: q,
                    scale: SIMD3<Float>(r.sc * (0.9 + rand.next() * 0.45), r.sy, r.sc * (0.9 + rand.next() * 0.45))
                )
                rockInstances.append(PropInstanceData(instanceMatrix: mat, propParams: .zero))
            }
            if let buf = Self.makeInstanceBuffer(device: device, instances: rockInstances) {
                builtPropBatches.append(InstancedPropBatch(mesh: rockMesh, instanceBuffer: buf, instanceCount: rockInstances.count, mode: 0))
            }
        }

        // 5. Glacial Icefall / Serac Clusters
        struct SeracItem {
            let x: Float, y: Float, z: Float, sc: Float, ry: Float, nx: Float, ny: Float, nz: Float
        }
        var seracList: [SeracItem] = []
        let valleeDz = terrainData.meta.dropZones.first(where: { $0.id == "vallee_blanche" })
        let vbX: Float = valleeDz?.x ?? -587.5
        let vbZ: Float = valleeDz?.z ?? 172.3
        let vbYaw: Float = valleeDz?.yaw ?? 2.0672

        @discardableResult
        func pushSerac(_ x: Float, _ z: Float, scOverride: Float?) -> Bool {
            if abs(x) > 5800 || abs(z) > 5800 { return false }
            let s = terrainData.sampleTerrain(x: x, z: z)
            if scOverride == nil && (s.rawY < 2520 || s.rawY > 3680 || s.slopeDeg < 22 || s.slopeDeg > 42 || s.curv < 0.49) {
                return false
            }
            for prev in seracList {
                if hypotf(x - prev.x, z - prev.z) < 95 { return false }
            }
            let gLen = hypotf(s.gx, s.gz)
            let contourYaw: Float = gLen > 0.05 ? atan2f(s.gx, -s.gz) + (rand.next() - 0.5) * 0.45 : rand.next() * .pi * 2.0
            let sc = scOverride ?? (3.4 + rand.next() * 5.2)
            let seracSink = sc * (0.28 + max(0.0, s.slopeDeg - 18.0) * 0.018)
            seracList.append(SeracItem(x: x, y: s.y - seracSink, z: z, sc: sc, ry: contourYaw, nx: s.nx, ny: s.ny, nz: s.nz))
            builtColliders.append(PropCollider(x: x, y: s.y, z: z, r: sc * 1.45, h: sc * 1.25, kind: .serac))
            builtShadows.append(ShadowInstanceData(
                posRadius: SIMD4<Float>(x, s.y, z, sc * 1.75),
                normalHeight: SIMD4<Float>(s.nx, s.ny, s.nz, sc * 1.35),
                extra: SIMD4<Float>(0.70, 2.0, rand.next(), 0.0)
            ))
            return true
        }

        let vbFx = sinf(vbYaw), vbFz = -cosf(vbYaw)
        let vbRx = cosf(vbYaw), vbRz = sinf(vbYaw)
        let heroSeracSeeds: [(Float, Float, Float)] = [
            (165, -72, 5.2),
            (290, 96, 6.0),
            (435, -110, 6.8),
        ]
        for (al, ac, scVal) in heroSeracSeeds {
            pushSerac(vbX + vbFx * al + vbRx * ac, vbZ + vbFz * al + vbRz * ac, scOverride: scVal)
        }
        var iSerac = 0
        while iSerac < 650 && seracList.count < 48 {
            defer { iSerac += 1 }
            let x = -2500 + rand.next() * 4800
            let z = -1000 + rand.next() * 4200
            var tooClose = false
            for dz in terrainData.meta.dropZones {
                if hypotf(x - dz.x, z - dz.z) < 110 {
                    tooClose = true
                    break
                }
            }
            if tooClose { continue }
            pushSerac(x, z, scOverride: nil)
        }

        if let seracMesh = AlpinePropsGeometry.buildSeracClusterMesh(device: device, seed: 311), !seracList.isEmpty {
            let up = SIMD3<Float>(0, 1, 0)
            var seracInstances: [PropInstanceData] = []
            seracInstances.reserveCapacity(seracList.count)
            for b in seracList {
                let slopeN = simd_normalize(lerp(SIMD3<Float>(b.nx, b.ny, b.nz), up, 0.54))
                let tiltQ = quatFromUnitVectors(from: up, to: slopeN)
                let yawQ = simd_quatf(angle: b.ry, axis: up)
                let q = tiltQ * yawQ
                let mat = matrixTRS(
                    translation: SIMD3<Float>(b.x, b.y, b.z),
                    rotation: q,
                    scale: SIMD3<Float>(b.sc * (0.95 + rand.next() * 0.25), b.sc * (0.85 + rand.next() * 0.30), b.sc * (0.90 + rand.next() * 0.20))
                )
                seracInstances.append(PropInstanceData(instanceMatrix: mat, propParams: .zero))
            }
            if let buf = Self.makeInstanceBuffer(device: device, instances: seracInstances) {
                builtPropBatches.append(InstancedPropBatch(mesh: seracMesh, instanceBuffer: buf, instanceCount: seracInstances.count, mode: 1))
            }
        }

        // 6. Natural Freeride Wind-Lip Kickers
        var kickerTransforms: [(x: Float, y: Float, z: Float, nx: Float, ny: Float, nz: Float, kx: Float, kz: Float, sc: Float)] = []
        for dz in terrainData.meta.dropZones {
            let fwdX = sinf(dz.yaw), fwdZ = -cosf(dz.yaw)
            let rightX = cosf(dz.yaw), rightZ = sinf(dz.yaw)
            let placements: [(along: Float, across: Float, sc: Float)] = [
                (135, 14, 1.05),
                (280, -24, 1.25),
                (460, 28, 1.35),
                (680, -36, 1.40),
                (940, 18, 1.30),
            ]
            for pl in placements {
                let x = dz.x + fwdX * pl.along + rightX * pl.across
                let z = dz.z + fwdZ * pl.along + rightZ * pl.across
                if abs(x) > 5800 || abs(z) > 5800 { continue }
                let s = terrainData.sampleTerrain(x: x, z: z)
                if s.slopeDeg > 41 || s.slopeDeg < 11 { continue }
                let gLen = hypotf(s.gx, s.gz)
                let kx = gLen > 0.08 ? (-s.gx / gLen) : fwdX
                let kz = gLen > 0.08 ? (-s.gz / gLen) : fwdZ
                kickerTransforms.append((x, s.y - 0.12, z, s.nx, s.ny, s.nz, kx, kz, pl.sc))
                builtKickers.append(WindLipKicker(
                    x: x, y: s.y, z: z,
                    fwdX: kx, fwdZ: kz,
                    radius: 4.8 * pl.sc,
                    rampSlope: 0.21 + 0.035 * pl.sc,
                    boost: 2.8 + 0.8 * pl.sc,
                    width: 4.0 * pl.sc
                ))
            }
        }
        if let kickerMesh = AlpinePropsGeometry.buildWindLipKickerMesh(device: device), !kickerTransforms.isEmpty {
            var kickerInstances: [PropInstanceData] = []
            kickerInstances.reserveCapacity(kickerTransforms.count)
            for k in kickerTransforms {
                let tmpN = simd_normalize(SIMD3<Float>(k.nx, k.ny, k.nz))
                var tmpFwd = SIMD3<Float>(k.kx, 0, k.kz)
                tmpFwd = simd_normalize(tmpFwd - tmpN * simd_dot(tmpFwd, tmpN))
                let tmpRight = simd_normalize(simd_cross(tmpFwd, tmpN))
                let trueUp = simd_normalize(simd_cross(tmpRight, tmpFwd))
                let q = quatFromBasis(right: tmpRight, up: trueUp, forwardZ: -tmpFwd)
                let mat = matrixTRS(
                    translation: SIMD3<Float>(k.x, k.y, k.z),
                    rotation: q,
                    scale: SIMD3<Float>(k.sc, k.sc * 0.9, k.sc)
                )
                kickerInstances.append(PropInstanceData(instanceMatrix: mat, propParams: .zero))
            }
            if let buf = Self.makeInstanceBuffer(device: device, instances: kickerInstances) {
                builtPropBatches.append(InstancedPropBatch(mesh: kickerMesh, instanceBuffer: buf, instanceCount: kickerInstances.count, mode: 3))
            }
        }

        // 7. Ground shadow instance buffer
        self.shadowInstanceCount = builtShadows.count
        if !builtShadows.isEmpty {
            let shLen = builtShadows.count * MemoryLayout<ShadowInstanceData>.stride
            self.shadowInstanceBuffer = builtShadows.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: shLen, options: .storageModeShared)
            }
        } else {
            self.shadowInstanceBuffer = nil
        }

        self.spruceBatches = builtSpruceBatches
        self.propBatches = builtPropBatches
        self.stationParts = builtStationParts
        self.cableCarPartSpecs = builtCableCarSpecs
        self.cableMeshes = builtCableMeshes
        self.colliders = builtColliders
        self.kickers = builtKickers

        updateCableCars(time: 0, twilight: 0)
    }

    // MARK: - Station & Stakes Construction

    private static func buildMidiStationArchitecture(
        device: MTLDevice,
        terrainData: any TerrainQueryProtocol,
        stationOrigin: SIMD3<Float>,
        cableVertexCount: Int,
        stationParts: inout [StationPart],
        cableCarPartSpecs: inout [CableCarPartSpec],
        cableMeshes: inout [MTLBuffer]
    ) -> (planOffset: SIMD3<Float>, azimuth: Float) {
        func addPart(
            _ geom: ([StandardVertex], [UInt32]),
            localPos: SIMD3<Float>,
            rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
            color: SIMD3<Float>,
            roughness: Float,
            metalness: Float,
            emissiveColor: SIMD3<Float> = .zero,
            emissiveIntensity: Float = 0,
            role: StationRole = .solid,
            label: String
        ) {
            guard let mesh = MeshPrimitives.upload(device: device, vertices: geom.0, indices: geom.1, label: label) else { return }
            let m = matrixTRS(translation: stationOrigin + localPos, rotation: rotation)
            let nm = normalMatrix(from: m)
            stationParts.append(StationPart(
                mesh: mesh,
                modelMatrix: m,
                normalMatrix: nm,
                baseColor: color,
                roughness: roughness,
                metalness: metalness,
                emissiveColor: emissiveColor,
                emissiveIntensity: emissiveIntensity,
                role: role,
                castShadowMap: false
            ))
        }

        let concreteCol = SIMD3<Float>(0.784, 0.812, 0.847) // 0xc8cfd8
        let darkSteelCol = SIMD3<Float>(0.149, 0.176, 0.212) // 0x262d36
        let glassCol = SIMD3<Float>(0.231, 0.431, 0.580) // 0x3b6e94
        let glassEmissive = SIMD3<Float>(1.0, 0.659, 0.259) // 0xffa842
        let beaconCol = SIMD3<Float>(1.0, 0.231, 0.133) // 0xff3b22
        let beaconEmissive = SIMD3<Float>(1.0, 0.173, 0.078) // 0xff2c14
        let snowCapCol = SIMD3<Float>(0.945, 0.961, 0.980) // 0xf1f5fa
        let antRedCol = SIMD3<Float>(0.851, 0.220, 0.165) // 0xd9382a
        let antWhiteCol = SIMD3<Float>(0.933, 0.953, 0.973) // 0xeef3f8

        addPart(MeshPrimitives.makeCylinder(radiusTop: 16.5, radiusBottom: 19.5, height: 12, radialSegs: 16), localPos: SIMD3<Float>(0, 44, 0), color: concreteCol, roughness: 0.78, metalness: 0.06, label: "MidiPlinth")
        addPart(MeshPrimitives.makeCylinder(radiusTop: 12.2, radiusBottom: 14.5, height: 24, radialSegs: 16), localPos: SIMD3<Float>(0, 60, 0), color: concreteCol, roughness: 0.78, metalness: 0.06, label: "MidiTower")
        addPart(MeshPrimitives.makeCylinder(radiusTop: 14.6, radiusBottom: 13.6, height: 2.2, radialSegs: 20), localPos: SIMD3<Float>(0, 69.0, 0), color: concreteCol, roughness: 0.78, metalness: 0.06, label: "MidiGalleryRing")
        addPart(MeshPrimitives.makeCylinder(radiusTop: 13.8, radiusBottom: 13.8, height: 5.2, radialSegs: 20), localPos: SIMD3<Float>(0, 72.5, 0), color: glassCol, roughness: 0.12, metalness: 0.45, emissiveColor: glassEmissive, emissiveIntensity: 0.55, role: .galleryGlass, label: "MidiGallery")
        addPart(MeshPrimitives.makeCylinder(radiusTop: 13.6, radiusBottom: 14.8, height: 3.2, radialSegs: 20), localPos: SIMD3<Float>(0, 76.5, 0), color: snowCapCol, roughness: 0.78, metalness: 0.0, label: "MidiRoofCap")
        addPart(MeshPrimitives.makeCylinder(radiusTop: 1.5, radiusBottom: 3.8, height: 22, radialSegs: 10), localPos: SIMD3<Float>(0, 87, 0), color: concreteCol, roughness: 0.78, metalness: 0.06, label: "MidiSpireBase")

        for i in 0..<5 {
            let fi = Float(i)
            let isRed = (i % 2 == 0)
            addPart(
                MeshPrimitives.makeCylinder(radiusTop: max(0.18, 1.05 - fi * 0.2), radiusBottom: max(0.24, 1.25 - fi * 0.2), height: 6.2, radialSegs: 8),
                localPos: SIMD3<Float>(0, 100 + fi * 6.2, 0),
                color: isRed ? antRedCol : antWhiteCol,
                roughness: isRed ? 0.42 : 0.50,
                metalness: isRed ? 0.32 : 0.25,
                label: "MidiAntennaSeg_\(i)"
            )
        }
        addPart(MeshPrimitives.makeSphere(radius: 1.15, widthSegs: 10, heightSegs: 8), localPos: SIMD3<Float>(0, 100 + 5 * 6.2 + 0.6, 0), color: beaconCol, roughness: 0.25, metalness: 0.1, emissiveColor: beaconEmissive, emissiveIntensity: 1.4, role: .beacon, label: "MidiBeacon")

        // Piton Nord station
        addPart(MeshPrimitives.makeBox(width: 24, height: 15, depth: 18), localPos: SIMD3<Float>(37, 42, -25), color: concreteCol, roughness: 0.78, metalness: 0.06, label: "NorthStation")
        addPart(MeshPrimitives.makeBox(width: 24.6, height: 4.2, depth: 14), localPos: SIMD3<Float>(37, 44.5, -25), color: glassCol, roughness: 0.12, metalness: 0.45, emissiveColor: glassEmissive, emissiveIntensity: 0.55, role: .galleryGlass, label: "NorthWin")
        addPart(MeshPrimitives.makeBox(width: 25.2, height: 2.4, depth: 19), localPos: SIMD3<Float>(37, 50.6, -25), color: snowCapCol, roughness: 0.78, metalness: 0.0, label: "NorthRoof")

        // Footbridge
        let bridgeYaw = atan2f(25, 37)
        let bridgeQ = simd_quatf(angle: bridgeYaw, axis: SIMD3<Float>(0, 1, 0))
        let bridgeCenter = SIMD3<Float>(18.5, 49, -12.5)
        addPart(MeshPrimitives.makeBox(width: 46, height: 1.8, depth: 4.2), localPos: bridgeCenter, rotation: bridgeQ, color: darkSteelCol, roughness: 0.48, metalness: 0.72, label: "BridgeDeck")
        let railOffsetL = bridgeQ.act(SIMD3<Float>(0, 1.9, 1.9))
        let railOffsetR = bridgeQ.act(SIMD3<Float>(0, 1.9, -1.9))
        addPart(MeshPrimitives.makeBox(width: 46, height: 2.4, depth: 0.35), localPos: bridgeCenter + railOffsetL, rotation: bridgeQ, color: darkSteelCol, roughness: 0.48, metalness: 0.72, label: "BridgeRailL")
        addPart(MeshPrimitives.makeBox(width: 46, height: 2.4, depth: 0.35), localPos: bridgeCenter + railOffsetR, rotation: bridgeQ, color: darkSteelCol, roughness: 0.48, metalness: 0.72, label: "BridgeRailR")

        // Dual steel Telepherique cables descending north toward Plan de l'Aiguille
        let planX = -1503.0 - stationOrigin.x
        let planZ = -2545.0 - stationOrigin.z
        let planY = terrainData.sampleHeight(x: -1503, z: -2545) - stationOrigin.y
        let planOffset = SIMD3<Float>(planX, planY, planZ)
        let azimuth = atan2f(planX - 37.0, planZ + 25.0)

        for lat: Float in [-4.5, 4.5] {
            var pts: [SIMD4<Float>] = []
            pts.reserveCapacity(cableVertexCount)
            for i in 0..<cableVertexCount {
                let t = Float(i) / Float(cableVertexCount - 1)
                let p = Self.evalCableLocalPos(t: t, lateralOff: lat, planOffset: planOffset)
                pts.append(SIMD4<Float>(p, 1.0))
            }
            let len = pts.count * MemoryLayout<SIMD4<Float>>.stride
            if let buf = pts.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: len, options: .storageModeShared) }) {
                cableMeshes.append(buf)
            }
        }

        // Cable car geometry specs (2 identical cabins)
        let bodyCol = SIMD3<Float>(0.831, 0.173, 0.141) // 0xd42c24
        let roofCol = SIMD3<Float>(0.941, 0.957, 0.973) // 0xf0f4f8
        let steelCol = SIMD3<Float>(0.227, 0.259, 0.302) // 0x3a424d
        let winCol = SIMD3<Float>(0.118, 0.173, 0.227) // 0x1e2c3a
        let winEmissive = SIMD3<Float>(1.0, 0.651, 0.227) // 0xffa63a

        let partsToBuild: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, Float, Float, SIMD3<Float>, Bool, String)] = [
            (SIMD3<Float>(1.4, 1.1, 6.2),   SIMD3<Float>(0, 0, 0),       steelCol, 0.45, 0.72, .zero,       false, "Carriage"),
            (SIMD3<Float>(0.65, 9.5, 1.1),  SIMD3<Float>(0, -4.8, 0),    steelCol, 0.45, 0.72, .zero,       false, "Hanger"),
            (SIMD3<Float>(5.6, 3.0, 8.8),   SIMD3<Float>(0, -13.0, 0),   bodyCol,  0.36, 0.22, .zero,       false, "LowerBody"),
            (SIMD3<Float>(5.8, 2.5, 8.4),   SIMD3<Float>(0, -10.5, 0),   winCol,   0.14, 0.65, winEmissive, true,  "WindowBand"),
            (SIMD3<Float>(5.75, 0.75, 8.95),SIMD3<Float>(0, -11.95, 0),  roofCol,  0.42, 0.18, .zero,       false, "WhiteStripe"),
            (SIMD3<Float>(5.4, 0.9, 8.6),   SIMD3<Float>(0, -9.0, 0),    roofCol,  0.42, 0.18, .zero,       false, "Roof"),
        ]
        for spec in partsToBuild {
            let (v, idx) = MeshPrimitives.makeBox(width: spec.0.x, height: spec.0.y, depth: spec.0.z)
            if let m = MeshPrimitives.upload(device: device, vertices: v, indices: idx, label: "CableCar_\(spec.7)") {
                cableCarPartSpecs.append(CableCarPartSpec(
                    mesh: m,
                    localOffset: spec.1,
                    baseColor: spec.2,
                    roughness: spec.3,
                    metalness: spec.4,
                    emissiveColor: spec.5,
                    isWindow: spec.6
                ))
            }
        }
        return (planOffset, azimuth)
    }

    private static func buildDropInStakes(
        device: MTLDevice,
        terrainData: any TerrainQueryProtocol,
        stationParts: inout [StationPart],
        propBatches: inout [InstancedPropBatch],
        shadows: inout [ShadowInstanceData]
    ) {
        let stakeGeom = MeshPrimitives.makeCylinder(radiusTop: 0.044, radiusBottom: 0.054, height: 2.65, radialSegs: 8)
        let bandGeom = MeshPrimitives.makeCylinder(radiusTop: 0.049, radiusBottom: 0.050, height: 0.28, radialSegs: 8)
        let stakeMesh = MeshPrimitives.upload(device: device, vertices: stakeGeom.0, indices: stakeGeom.1, label: "DropStake")
        let bandMesh = MeshPrimitives.upload(device: device, vertices: bandGeom.0, indices: bandGeom.1, label: "DropStakeBand")
        let moundMesh = AlpinePropsGeometry.buildStakeSnowMoundMesh(device: device)

        let stakeCol = SIMD3<Float>(0.294, 0.224, 0.169) // 0x4b392b
        let bandCol = SIMD3<Float>(0.878, 0.322, 0.133) // 0xe05222
        let up = SIMD3<Float>(0, 1, 0)
        var moundInstances: [PropInstanceData] = []

        for dz in terrainData.meta.dropZones {
            let rightX = cosf(dz.yaw)
            let rightZ = sinf(dz.yaw)
            for side: Float in [-4.2, 4.2] {
                let px = dz.x + rightX * side
                let pz = dz.z + rightZ * side
                let ts = terrainData.sampleTerrain(x: px, z: pz)
                let py = ts.y
                let tiltZ: Float = side > 0 ? -0.045 : 0.045
                let qPole = simd_quatf(angle: tiltZ, axis: SIMD3<Float>(0, 0, 1))
                let mPole = matrixTRS(translation: SIMD3<Float>(px, py + 0.78, pz), rotation: qPole)

                if let stakeMesh {
                    stationParts.append(StationPart(
                        mesh: stakeMesh,
                        modelMatrix: mPole,
                        normalMatrix: normalMatrix(from: mPole),
                        baseColor: stakeCol,
                        roughness: 0.86,
                        metalness: 0.0,
                        emissiveColor: .zero,
                        emissiveIntensity: 0,
                        role: .stake,
                        castShadowMap: true
                    ))
                }
                if let bandMesh {
                    let mBand = mPole * matrixTranslation(SIMD3<Float>(0, 1.02, 0))
                    stationParts.append(StationPart(
                        mesh: bandMesh,
                        modelMatrix: mBand,
                        normalMatrix: normalMatrix(from: mBand),
                        baseColor: bandCol,
                        roughness: 0.52,
                        metalness: 0.0,
                        emissiveColor: .zero,
                        emissiveIntensity: 0,
                        role: .stake,
                        castShadowMap: true
                    ))
                }

                let slopeN = simd_normalize(ts.normal)
                let mQ = quatFromUnitVectors(from: up, to: slopeN)
                let mMound = matrixTRS(
                    translation: SIMD3<Float>(px - slopeN.x * 0.008, py - slopeN.y * 0.008, pz - slopeN.z * 0.008),
                    rotation: mQ
                )
                moundInstances.append(PropInstanceData(instanceMatrix: mMound, propParams: .zero))

                shadows.append(ShadowInstanceData(
                    posRadius: SIMD4<Float>(px, py, pz, 0.34),
                    normalHeight: SIMD4<Float>(ts.nx, ts.ny, ts.nz, 2.1),
                    extra: SIMD4<Float>(0.72, 3.0, 0.5, 0.0)
                ))
            }
        }

        if let moundMesh, let buf = Self.makeInstanceBuffer(device: device, instances: moundInstances) {
            propBatches.append(InstancedPropBatch(mesh: moundMesh, instanceBuffer: buf, instanceCount: moundInstances.count, mode: 2))
        }
    }

    private static func evalCableLocalPos(t: Float, lateralOff: Float, planOffset: SIMD3<Float>) -> SIMD3<Float> {
        let x0 = 37.0 + lateralOff
        let z0: Float = -25.0
        let y0: Float = 44.0
        let x1 = planOffset.x + lateralOff
        let z1 = planOffset.z
        let y1 = planOffset.y + 26.0
        let sag = sinf(t * .pi) * -98.0
        return SIMD3<Float>(
            x0 + (x1 - x0) * t,
            y0 + (y1 - y0) * t + sag,
            z0 + (z1 - z0) * t
        )
    }

    private func updateCableCars(time: Float, twilight: Float) {
        let u = 0.5 - 0.5 * cosf(time * 0.028)
        let tA = 0.09 + u * 0.82
        let tB = 0.91 - u * 0.82

        let posUp = cableStationBasePos + Self.evalCableLocalPos(t: tA, lateralOff: -4.5, planOffset: cablePlanOffset)
        let posDown = cableStationBasePos + Self.evalCableLocalPos(t: tB, lateralOff: 4.5, planOffset: cablePlanOffset)

        let qUp = quatFromEulerXYZ(x: 0, y: cableAzimuth, z: sinf(time * 0.85) * 0.035)
        let qDown = quatFromEulerXYZ(x: 0, y: cableAzimuth, z: cosf(time * 0.78) * 0.035)

        cabinUpMatrix = matrixTRS(translation: posUp, rotation: qUp)
        cabinDownMatrix = matrixTRS(translation: posDown, rotation: qDown)

        galleryEmissive = 0.35 + twilight * 2.55
        cabinWinEmissive = 0.30 + twilight * 2.25
        let pulse = 0.65 + 0.35 * sinf(time * 3.1)
        beaconEmissive = (0.85 + twilight * 3.8) * pulse
    }

    // MARK: - Buffer Helpers

    private static func makeSinglePropBatch(device: MTLDevice, mesh: PropMesh, matrix: simd_float4x4, mode: Float) -> InstancedPropBatch? {
        let inst = [PropInstanceData(instanceMatrix: matrix, propParams: .zero)]
        guard let buf = makeInstanceBuffer(device: device, instances: inst) else { return nil }
        return InstancedPropBatch(mesh: mesh, instanceBuffer: buf, instanceCount: 1, mode: mode)
    }

    private static func makeInstanceBuffer(device: MTLDevice, instances: [PropInstanceData]) -> MTLBuffer? {
        guard !instances.isEmpty else { return nil }
        let len = instances.count * MemoryLayout<PropInstanceData>.stride
        return instances.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: len, options: .storageModeShared)
        }
    }

    // MARK: - Pipeline Builders

    private func ensurePropPipeline(msaa: Int) -> MTLRenderPipelineState? {
        if let p = propPipelines[msaa] { return p }
        guard let vf = library.makeFunction(name: "propVertex"),
              let ff = library.makeFunction(name: "propFragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "AlpinePropPipeline_\(msaa)"
        desc.vertexFunction = vf
        desc.fragmentFunction = ff
        desc.vertexDescriptor = PropMesh.vertexDescriptor()
        desc.rasterSampleCount = msaa
        desc.colorAttachments[0].pixelFormat = .rgba16Float
        desc.depthAttachmentPixelFormat = .depth32Float
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        if let pso { propPipelines[msaa] = pso }
        return pso
    }

    private func ensureSprucePipeline(msaa: Int) -> MTLRenderPipelineState? {
        if let p = sprucePipelines[msaa] { return p }
        guard let vf = library.makeFunction(name: "spruceVertex"),
              let ff = library.makeFunction(name: "spruceFragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "SprucePipeline_\(msaa)"
        desc.vertexFunction = vf
        desc.fragmentFunction = ff
        desc.vertexDescriptor = PropMesh.vertexDescriptor()
        desc.rasterSampleCount = msaa
        desc.colorAttachments[0].pixelFormat = .rgba16Float
        desc.depthAttachmentPixelFormat = .depth32Float
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        if let pso { sprucePipelines[msaa] = pso }
        return pso
    }

    private func ensureStationPipeline(msaa: Int) -> MTLRenderPipelineState? {
        if let p = stationPipelines[msaa] { return p }
        guard let vf = library.makeFunction(name: "stationVertex"),
              let ff = library.makeFunction(name: "stationFragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "StationPipeline_\(msaa)"
        desc.vertexFunction = vf
        desc.fragmentFunction = ff
        desc.vertexDescriptor = MeshPrimitives.standardVertexDescriptor()
        desc.rasterSampleCount = msaa
        desc.colorAttachments[0].pixelFormat = .rgba16Float
        desc.depthAttachmentPixelFormat = .depth32Float
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        if let pso { stationPipelines[msaa] = pso }
        return pso
    }

    private func ensureCablePipeline(msaa: Int) -> MTLRenderPipelineState? {
        if let p = cablePipelines[msaa] { return p }
        guard let vf = library.makeFunction(name: "cableVertex"),
              let ff = library.makeFunction(name: "cableFragment") else { return nil }
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float4
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = BufferIndex.vertices.rawValue
        vd.layouts[BufferIndex.vertices.rawValue].stride = MemoryLayout<SIMD4<Float>>.stride
        vd.layouts[BufferIndex.vertices.rawValue].stepFunction = .perVertex
        vd.layouts[BufferIndex.vertices.rawValue].stepRate = 1

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "CablePipeline_\(msaa)"
        desc.vertexFunction = vf
        desc.fragmentFunction = ff
        desc.vertexDescriptor = vd
        desc.rasterSampleCount = msaa
        desc.depthAttachmentPixelFormat = .depth32Float
        let att = desc.colorAttachments[0]!
        att.pixelFormat = .rgba16Float
        att.isBlendingEnabled = true
        att.rgbBlendOperation = .add
        att.sourceRGBBlendFactor = .sourceAlpha
        att.destinationRGBBlendFactor = .oneMinusSourceAlpha
        att.alphaBlendOperation = .add
        att.sourceAlphaBlendFactor = .zero
        att.destinationAlphaBlendFactor = .one
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        if let pso { cablePipelines[msaa] = pso }
        return pso
    }

    private func ensureGroundShadowPipeline(msaa: Int) -> MTLRenderPipelineState? {
        if let p = groundShadowPipelines[msaa] { return p }
        guard let vf = library.makeFunction(name: "groundShadowVertex"),
              let ff = library.makeFunction(name: "groundShadowFragment") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "PropGroundShadowPipeline_\(msaa)"
        desc.vertexFunction = vf
        desc.fragmentFunction = ff
        desc.vertexDescriptor = MeshPrimitives.standardVertexDescriptor()
        desc.rasterSampleCount = msaa
        desc.depthAttachmentPixelFormat = .depth32Float
        let att = desc.colorAttachments[0]!
        att.pixelFormat = .rgba16Float
        att.isBlendingEnabled = true
        att.rgbBlendOperation = .add
        att.sourceRGBBlendFactor = .destinationColor
        att.destinationRGBBlendFactor = .zero
        att.alphaBlendOperation = .add
        att.sourceAlphaBlendFactor = .zero
        att.destinationAlphaBlendFactor = .one
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        if let pso { groundShadowPipelines[msaa] = pso }
        return pso
    }

    private func ensureShadowDepthPipeline() -> MTLRenderPipelineState? {
        if let p = shadowDepthPipeline { return p }
        guard let vf = library.makeFunction(name: "stationShadowVertex") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "PropShadowDepthPipeline"
        desc.vertexFunction = vf
        desc.fragmentFunction = nil
        desc.vertexDescriptor = MeshPrimitives.standardVertexDescriptor()
        desc.colorAttachments[0].pixelFormat = .invalid
        desc.depthAttachmentPixelFormat = .depth32Float
        let pso = try? device.makeRenderPipelineState(descriptor: desc)
        shadowDepthPipeline = pso
        return pso
    }

    // MARK: - AlpinePropsSubsystem

    public func update(cameraPos: SIMD3<Float>, preset: LightingPreset, sunDir: SIMD3<Float>, time: Float) {
        _ = cameraPos
        _ = sunDir
        updateCableCars(time: time, twilight: preset.twilightFactor)
    }

    public func renderShadowPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        guard let pso = ensureShadowDepthPipeline() else { return }
        encoder.setRenderPipelineState(pso)
        encoder.setDepthStencilState(depthWriteLessState)
        encoder.setCullMode(.back)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)

        for part in stationParts where part.castShadowMap {
            var mu = ModelUniforms()
            mu.modelMatrix = part.modelMatrix
            encoder.setVertexBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
            encoder.setVertexBuffer(part.mesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: part.mesh.indexCount,
                indexType: .uint32,
                indexBuffer: part.mesh.indexBuffer,
                indexBufferOffset: 0
            )
        }
    }

    public func renderGroundShadowsPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        guard shadowInstanceCount > 0,
              let shBuf = shadowInstanceBuffer,
              let pso = ensureGroundShadowPipeline(msaa: context.msaaSampleCount) else { return }

        encoder.setRenderPipelineState(pso)
        encoder.setDepthStencilState(depthReadOnlyLessEqualState)
        encoder.setCullMode(.none)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setDepthBias(-2.8, slopeScale: -2.8, clamp: 0.0)

        let tex = terrainData.textures
        encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
        encoder.setVertexBuffer(shBuf, offset: 0, index: BufferIndex.instances.rawValue)
        encoder.setVertexBuffer(shadowQuadMesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
        encoder.setVertexTexture(tex.coreDem, index: TextureIndex.coreDem.rawValue)
        encoder.setVertexTexture(tex.coreAux1, index: TextureIndex.coreAux1.rawValue)
        encoder.setVertexTexture(tex.coreAux2, index: TextureIndex.coreAux2.rawValue)
        encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)

        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: shadowQuadMesh.indexCount,
            indexType: .uint32,
            indexBuffer: shadowQuadMesh.indexBuffer,
            indexBufferOffset: 0,
            instanceCount: shadowInstanceCount
        )

        encoder.setDepthBias(0, slopeScale: 0, clamp: 0)
    }

    public func renderHDRPass(encoder: MTLRenderCommandEncoder, context: RenderPassContext) {
        let msaa = context.msaaSampleCount
        let tex = terrainData.textures

        encoder.setDepthStencilState(depthWriteLessState)
        encoder.setFrontFacing(.counterClockwise)

        // 1. Rocks, Seracs, Kickers, Midi Pitons, Stake Snow Mounds
        if let propPso = ensurePropPipeline(msaa: msaa) {
            encoder.setRenderPipelineState(propPso)
            encoder.setCullMode(.back)
            encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setFragmentTexture(tex.coreDem, index: TextureIndex.coreDem.rawValue)
            encoder.setFragmentTexture(tex.coreAux1, index: TextureIndex.coreAux1.rawValue)
            encoder.setFragmentTexture(tex.coreAux2, index: TextureIndex.coreAux2.rawValue)
            encoder.setFragmentTexture(tex.graniteAlbedo, index: TextureIndex.graniteAlbedo.rawValue)
            encoder.setFragmentTexture(tex.graniteNormal, index: TextureIndex.graniteNormal.rawValue)
            encoder.setFragmentTexture(tex.snowNormal, index: TextureIndex.snowNormal.rawValue)
            encoder.setFragmentTexture(tex.snowCavity, index: TextureIndex.snowCavity.rawValue)
            encoder.setFragmentTexture(tex.iceNormal, index: TextureIndex.iceNormal.rawValue)

            for batch in propBatches where batch.instanceCount > 0 {
                var subUniforms = SIMD4<Float>(batch.mode, 0, 0, 0)
                encoder.setFragmentBytes(&subUniforms, length: MemoryLayout<SIMD4<Float>>.stride, index: BufferIndex.subsystemUniforms.rawValue)
                encoder.setVertexBuffer(batch.mesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
                encoder.setVertexBuffer(batch.instanceBuffer, offset: 0, index: BufferIndex.instances.rawValue)
                if let ib = batch.mesh.indexBuffer {
                    encoder.drawIndexedPrimitives(
                        type: .triangle,
                        indexCount: batch.mesh.indexCount,
                        indexType: .uint32,
                        indexBuffer: ib,
                        indexBufferOffset: 0,
                        instanceCount: batch.instanceCount
                    )
                } else {
                    encoder.drawPrimitives(
                        type: .triangle,
                        vertexStart: 0,
                        vertexCount: batch.mesh.vertexCount,
                        instanceCount: batch.instanceCount
                    )
                }
            }
        }

        // 2. Volumetric 3D Norway Spruces (DoubleSide)
        if let sprucePso = ensureSprucePipeline(msaa: msaa) {
            encoder.setRenderPipelineState(sprucePso)
            encoder.setCullMode(.none)
            encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setVertexTexture(tex.coreAux1, index: TextureIndex.coreAux1.rawValue)
            encoder.setVertexTexture(tex.coreAux2, index: TextureIndex.coreAux2.rawValue)
            encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setFragmentTexture(tex.snowNormal, index: TextureIndex.snowNormal.rawValue)
            encoder.setFragmentTexture(tex.snowCavity, index: TextureIndex.snowCavity.rawValue)

            for batch in spruceBatches where batch.instanceCount > 0 {
                encoder.setVertexBuffer(batch.mesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
                encoder.setVertexBuffer(batch.instanceBuffer, offset: 0, index: BufferIndex.instances.rawValue)
                encoder.drawPrimitives(
                    type: .triangle,
                    vertexStart: 0,
                    vertexCount: batch.mesh.vertexCount,
                    instanceCount: batch.instanceCount
                )
            }
        }

        // 3. Aiguille du Midi Station, Animated Cable Cars & Drop-in Stakes
        if let stationPso = ensureStationPipeline(msaa: msaa) {
            encoder.setRenderPipelineState(stationPso)
            encoder.setCullMode(.back)
            encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setFragmentBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            encoder.setFragmentTexture(tex.coreAux1, index: TextureIndex.coreAux1.rawValue)
            encoder.setFragmentTexture(tex.coreAux2, index: TextureIndex.coreAux2.rawValue)

            for part in stationParts {
                var mu = ModelUniforms()
                mu.modelMatrix = part.modelMatrix
                mu.normalMatrix = part.normalMatrix
                mu.baseColor = SIMD4<Float>(part.baseColor, 1.0)
                let emissive: Float
                switch part.role {
                case .galleryGlass: emissive = galleryEmissive
                case .beacon: emissive = beaconEmissive
                case .solid, .stake: emissive = part.emissiveIntensity
                }
                mu.pbrParams = SIMD4<Float>(part.roughness, part.metalness, emissive, 0.55)
                mu.fgBounceColor = SIMD4<Float>(part.emissiveColor, 0.0)

                encoder.setVertexBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
                encoder.setFragmentBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
                encoder.setVertexBuffer(part.mesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
                encoder.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: part.mesh.indexCount,
                    indexType: .uint32,
                    indexBuffer: part.mesh.indexBuffer,
                    indexBufferOffset: 0
                )
            }

            // Two gliding cable cars
            for cabinMat in [cabinUpMatrix, cabinDownMatrix] {
                for spec in cableCarPartSpecs {
                    let m = cabinMat * matrixTranslation(spec.localOffset)
                    var mu = ModelUniforms()
                    mu.modelMatrix = m
                    mu.normalMatrix = normalMatrix(from: m)
                    mu.baseColor = SIMD4<Float>(spec.baseColor, 1.0)
                    mu.pbrParams = SIMD4<Float>(spec.roughness, spec.metalness, spec.isWindow ? cabinWinEmissive : 0.0, 0.55)
                    mu.fgBounceColor = SIMD4<Float>(spec.emissiveColor, 0.0)

                    encoder.setVertexBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
                    encoder.setFragmentBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
                    encoder.setVertexBuffer(spec.mesh.vertexBuffer, offset: 0, index: BufferIndex.vertices.rawValue)
                    encoder.drawIndexedPrimitives(
                        type: .triangle,
                        indexCount: spec.mesh.indexCount,
                        indexType: .uint32,
                        indexBuffer: spec.mesh.indexBuffer,
                        indexBufferOffset: 0
                    )
                }
            }
        }

        // 4. Twin Telepherique steel cables
        if let cablePso = ensureCablePipeline(msaa: msaa), !cableMeshes.isEmpty {
            encoder.setRenderPipelineState(cablePso)
            encoder.setDepthStencilState(depthReadOnlyLessEqualState)
            encoder.setCullMode(.none)
            encoder.setVertexBuffer(context.frameUniformBuffer, offset: 0, index: BufferIndex.frameUniforms.rawValue)
            var mu = ModelUniforms()
            mu.modelMatrix = matrixTranslation(cableStationBasePos)
            mu.baseColor = SIMD4<Float>(0.102, 0.125, 0.161, 1.0) // 0x1a2029
            encoder.setVertexBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)
            encoder.setFragmentBytes(&mu, length: MemoryLayout<ModelUniforms>.stride, index: BufferIndex.modelUniforms.rawValue)

            for cBuf in cableMeshes {
                encoder.setVertexBuffer(cBuf, offset: 0, index: BufferIndex.vertices.rawValue)
                encoder.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: cableVertexCount)
            }
        }
    }
}
