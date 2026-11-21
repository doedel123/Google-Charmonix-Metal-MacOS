import XCTest
import Metal
import simd

/// Synthetic inclined-plane terrain for isolated SkiPhysics invariant tests.
private final class SyntheticPlaneTerrain: TerrainQueryProtocol {
    let meta: ChamonixMeta
    let coreSize: Int = 1024
    let outerSize: Int = 512
    let textures: TerrainTextures

    private let slopeTan: Float
    private let baseElev: Float
    private let surfaceType: SurfaceType

    init(device: MTLDevice, slopeDeg: Float, baseElev: Float = 3200.0, surface: SurfaceType = .powder) {
        self.slopeTan = tanf(degToRad(slopeDeg))
        self.baseElev = baseElev
        self.surfaceType = surface

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead]
        let dummy = device.makeTexture(descriptor: desc)!
        self.textures = TerrainTextures(
            coreDem: dummy, coreAux1: dummy, coreAux2: dummy,
            outerDem: dummy, outerAux1: dummy, outerAux2: dummy,
            graniteAlbedo: dummy, graniteNormal: dummy,
            snowNormal: dummy, snowCavity: dummy, iceNormal: dummy,
            powderPuff: dummy, spruceBough: dummy, skiTopsheet: dummy
        )

        let dz = DropZone(
            id: "test_slope",
            name: "Test Slope",
            subtitle: "Synthetic plane",
            x: 0, z: 0, elev: baseElev, slopeDeg: slopeDeg,
            yaw: 0, compass: "N", verticalDrop: 500, traceEnd: nil
        )
        self.meta = ChamonixMeta(
            originLat: 45.8787,
            originLon: 6.905,
            coreHalfSpan: 6400,
            outerHalfSpan: 24000,
            maxElev: 5000,
            aiguilleSpire: AiguilleSpire(x: 0, z: 0, elev: 3800),
            peaks: [],
            dropZones: [dz]
        )
    }

    func sampleRawDEM(x: Float, z: Float) -> TerrainRawSample {
        // Slopes downhill toward -Z (North, yaw=0 -> fwd=(0,0,-1))
        let elev = baseElev + z * slopeTan
        return TerrainRawSample(elev: elev, ao: 0.9, powder: 0.8, curv: 0.5, inCore: true)
    }

    func sampleHeight(x: Float, z: Float) -> Float {
        baseElev + z * slopeTan
    }

    func sampleTerrain(x: Float, z: Float) -> TerrainSample {
        let y = sampleHeight(x: x, z: z)
        let gx: Float = 0.0
        let gz: Float = slopeTan
        let inv = 1.0 / sqrtf(gx * gx + 1.0 + gz * gz)
        let normal = SIMD3<Float>(-gx * inv, inv, -gz * inv)
        let slopeRad = acosf(clamp(normal.y, 0.0, 1.0))
        let slopeDeg = radToDeg(slopeRad)
        return TerrainSample(
            y: y, rawY: y, normal: normal,
            gx: gx, gz: gz,
            slopeRad: slopeRad, slopeDeg: slopeDeg,
            ao: 0.9, powder: 0.8, curv: 0.5,
            surface: surfaceType
        )
    }
}

final class PhysicsParityTests: XCTestCase {

    private func makeDevice() throws -> MTLDevice {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device unavailable")
        }
        return dev
    }

    func testDownhillAccelerationOnSlope() throws {
        let dev = try makeDevice()
        let terrain = SyntheticPlaneTerrain(device: dev, slopeDeg: 25.0, surface: .packed)
        let sim = GameSimulation(terrainData: terrain)
        sim.avalanchesEnabled = false
        sim.spawnAtZone(index: 0)

        XCTAssertEqual(sim.state.y, 3200.0, accuracy: 1e-3)
        XCTAssertTrue(sim.state.grounded)
        XCTAssertEqual(sim.state.slopeDeg, 25.0, accuracy: 0.1)

        // Step 2 seconds straight down fall line (yaw = 0 -> -Z)
        _ = sim.stepSeconds(2.0)
        XCTAssertTrue(sim.state.grounded)
        XCTAssertGreaterThan(sim.state.speed, 5.0, "Skier should accelerate down 25 deg slope")
        XCTAssertLessThan(sim.state.z, -1.0, "Yaw=0 faces -Z downhill")
        XCTAssertGreaterThan(sim.state.verticalDropped, 1.0)
        XCTAssertEqual(sim.state.y, terrain.sampleHeight(x: sim.state.x, z: sim.state.z), accuracy: 1e-3)
        XCTAssertFalse(sim.state.speed.isNaN)
        XCTAssertFalse(sim.state.x.isNaN)
        XCTAssertFalse(sim.state.y.isNaN)
        XCTAssertFalse(sim.state.z.isNaN)
    }

    func testCarvingTurnsHeadingAndProducesTracks() throws {
        let dev = try makeDevice()
        let terrain = SyntheticPlaneTerrain(device: dev, slopeDeg: 24.0, surface: .powder)
        let sim = GameSimulation(terrainData: terrain)
        sim.avalanchesEnabled = false
        sim.spawnAtZone(index: 0)

        // Build speed first
        _ = sim.stepSeconds(1.5)
        let yawBefore = sim.state.yaw
        let speedBefore = sim.state.speed
        XCTAssertGreaterThan(speedBefore, 4.0)

        // Carve right ('KeyD') for 1.2s and count track events
        var trackEvents = 0
        let dt = ChamonixConstants.fixedPhysicsDt
        var input = InputState()
        input.right = true
        var t: Float = 1.5
        for _ in 0..<Int(1.2 / dt) {
            t += dt
            sim.step(dt: dt, time: t, input: input)
            for ev in sim.events {
                if case .track = ev { trackEvents += 1 }
            }
            sim.clearEvents()
        }

        XCTAssertGreaterThan(sim.state.yaw - yawBefore, 0.25, "Holding right steer should carve yaw clockwise")
        XCTAssertGreaterThan(sim.state.edgeAngle, 0.3, "Carving right should engage positive edgeAngle")
        XCTAssertGreaterThan(trackEvents, 5, "Carving on powder should emit twin-ski track events")
        XCTAssertFalse(sim.state.crashed)
    }

    func testHockeyStopBrakingDecelerates() throws {
        let dev = try makeDevice()
        let terrain = SyntheticPlaneTerrain(device: dev, slopeDeg: 18.0, surface: .packed)
        let sim = GameSimulation(terrainData: terrain)
        sim.avalanchesEnabled = false
        sim.spawnAtZone(index: 0)

        _ = sim.stepSeconds(2.5)
        let fastSpeed = sim.state.speed
        XCTAssertGreaterThan(fastSpeed, 6.0)

        _ = sim.stepSeconds(2.0, keys: ["ShiftLeft"])
        XCTAssertLessThan(sim.state.speed, fastSpeed * 0.45, "Shift brake should significantly decelerate skier")
    }

    func testOllieJumpAndLanding() throws {
        let dev = try makeDevice()
        let terrain = SyntheticPlaneTerrain(device: dev, slopeDeg: 20.0, surface: .packed)
        let sim = GameSimulation(terrainData: terrain)
        sim.avalanchesEnabled = false
        sim.spawnAtZone(index: 0)

        _ = sim.stepSeconds(2.0)
        // Charge jump (Space held 0.4s) then release
        _ = sim.stepSeconds(0.4, keys: ["Space"], startTime: 2.0)
        XCTAssertGreaterThan(sim.state.popCharge, 0.5)

        var sawAir = false
        var sawLand = false
        let dt = ChamonixConstants.fixedPhysicsDt
        let empty = InputState()
        var t: Float = 2.4
        for _ in 0..<240 {
            t += dt
            sim.step(dt: dt, time: t, input: empty)
            if !sim.state.grounded { sawAir = true }
            for ev in sim.events {
                if case .land = ev { sawLand = true }
            }
            sim.clearEvents()
            if sawAir && sim.state.grounded { break }
        }
        XCTAssertTrue(sawAir, "Releasing Space after charge should launch skier airborne")
        XCTAssertTrue(sawLand || sim.state.grounded, "Skier should cleanly land back on slope")
        XCTAssertFalse(sim.state.crashed)
    }

    func testCoreMassifBoundaryClamping() throws {
        let dev = try makeDevice()
        let terrain = SyntheticPlaneTerrain(device: dev, slopeDeg: 10.0, surface: .packed)
        let physics = SkiPhysics(terrainData: terrain)
        physics.avalanchesEnabled = false
        let edgeZone = DropZone(
            id: "edge", name: "Edge", subtitle: "",
            x: 5849.0, z: -5849.0, elev: 2000, slopeDeg: 10,
            yaw: 0, compass: "N", verticalDrop: 0, traceEnd: nil
        )
        physics.spawn(zone: edgeZone)
        let dt = ChamonixConstants.fixedPhysicsDt
        var input = InputState()
        input.tuck = true
        for i in 0..<300 {
            physics.step(dt: dt, time: Float(i) * dt, input: input)
        }
        XCTAssertLessThanOrEqual(abs(physics.state.x), 5850.01)
        XCTAssertLessThanOrEqual(abs(physics.state.z), 5850.01)
    }

    func testHeadLookBoundsAndSwitchMode() throws {
        let dev = try makeDevice()
        let terrain = SyntheticPlaneTerrain(device: dev, slopeDeg: 20.0, surface: .powder)
        let sim = GameSimulation(terrainData: terrain)
        sim.spawnAtZone(index: 0)

        sim.addHeadLookDelta(dYaw: 10.0, dPitch: 10.0)
        XCTAssertLessThanOrEqual(sim.lookYawOff, degToRad(82.0) + 1e-4)
        XCTAssertLessThanOrEqual(sim.lookPitchOff, degToRad(46.0) + 1e-4)

        sim.toggleSwitch()
        XCTAssertTrue(sim.state.isSwitch)
    }
}
