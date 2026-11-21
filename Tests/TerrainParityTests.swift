import XCTest
import Metal
import simd

final class TerrainParityTests: XCTestCase {

    private struct TerrainExpectation {
        let label: String
        let x: Float
        let z: Float
        let rawY: Float
        let y: Float
        let normal: SIMD3<Float>
        let slopeDeg: Float
        let ao: Float
        let powder: Float
        let curv: Float
        let surface: SurfaceType
        let inCore: Bool
    }

    func testChamonixDEMAndSamplingParityWithJS() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal device unavailable")
        }

        let terrain = try ChamonixTerrainData(device: device, commandQueue: queue)
        XCTAssertEqual(terrain.coreSize, 1024)
        XCTAssertEqual(terrain.outerSize, 512)
        XCTAssertEqual(terrain.meta.dropZones.count, 6)
        XCTAssertEqual(terrain.meta.peaks.count, 10)

        let expected: [TerrainExpectation] = [
            TerrainExpectation(
                label: "dz:vallee_blanche", x: -587.5, z: 172.3,
                rawY: 3492.5158, y: 3491.6634,
                normal: SIMD3<Float>(0.33251936, 0.93298886, 0.1377050),
                slopeDeg: 21.094378, ao: 0.9406947, powder: 0.48223116, curv: 0.5308515,
                surface: .powder, inCore: true
            ),
            TerrainExpectation(
                label: "dz:tacul_north", x: -1522.5, z: 2222.6,
                rawY: 4105.3457, y: 4105.5969,
                normal: SIMD3<Float>(-0.21394392, 0.86890605, -0.44635219),
                slopeDeg: 29.668237, ao: 0.93092477, powder: 0.31684165, curv: 0.54488506,
                surface: .packed, inCore: true
            ),
            TerrainExpectation(
                label: "dz:midi_north", x: -1450.4, z: -17.7,
                rawY: 3710.6496, y: 3710.4186,
                normal: SIMD3<Float>(-0.40434888, 0.90537126, -0.12963360),
                slopeDeg: 25.126669, ao: 0.97198766, powder: 0.22499955, curv: 0.92598809,
                surface: .crust, inCore: true
            ),
            TerrainExpectation(
                label: "dz:plan_aiguille", x: -1613.3, z: -2487.2,
                rawY: 2254.5196, y: 2254.2587,
                normal: SIMD3<Float>(-0.39137511, 0.87504856, 0.28480791),
                slopeDeg: 28.949277, ao: 0.87182996, powder: 0.16997415, curv: 0.58909794,
                surface: .packed, inCore: true
            ),
            TerrainExpectation(
                label: "dz:cosmiques_couloir", x: -1502.5, z: 135.6,
                rawY: 3677.2162, y: 3676.9086,
                normal: SIMD3<Float>(-0.47032736, 0.86415658, 0.17895694),
                slopeDeg: 30.213464, ao: 0.90622320, powder: 0.11884730, curv: 0.75446227,
                surface: .crust, inCore: true
            ),
            TerrainExpectation(
                label: "dz:glacier_rond", x: -2571.9, z: 3027.2,
                rawY: 4122.8029, y: 4122.4410,
                normal: SIMD3<Float>(-0.08936374, 0.82790059, -0.55370998),
                slopeDeg: 34.116322, ao: 0.88024293, powder: 0.29447755, curv: 0.71920473,
                surface: .crust, inCore: true
            ),
            TerrainExpectation(
                label: "peak:mont_blanc", x: -3084.0, z: 5123.0,
                rawY: 4777.4854, y: 4777.1420,
                normal: SIMD3<Float>(0.35413821, 0.93494546, 0.02152022),
                slopeDeg: 20.780665, ao: 0.97206154, powder: 0.38918510, curv: 0.75316257,
                surface: .crust, inCore: true
            ),
            TerrainExpectation(
                label: "peak:aiguille_du_midi", x: -1431.2, z: 25.0,
                rawY: 3717.9886, y: 3717.8422,
                normal: SIMD3<Float>(0.12340285, 0.99159118, -0.03897021),
                slopeDeg: 7.435498, ao: 0.99886547, powder: 0.30126319, curv: 0.91478282,
                surface: .crust, inCore: true
            ),
            TerrainExpectation(
                label: "peak:chamonix", x: -2759.0, z: -5001.0,
                rawY: 1041.0293, y: 1040.5943,
                normal: SIMD3<Float>(0.02861765, 0.99890065, 0.03712846),
                slopeDeg: 2.686863, ao: 0.92156863, powder: 0.25959751, curv: 0.49019608,
                surface: .packed, inCore: true
            ),
            TerrainExpectation(
                label: "origin", x: 0.0, z: 0.0,
                rawY: 3280.0031, y: 3280.2108,
                normal: SIMD3<Float>(0.35915095, 0.93249552, 0.03824528),
                slopeDeg: 21.172776, ao: 0.90882353, powder: 0.62058826, curv: 0.47745098,
                surface: .powder, inCore: true
            ),
            TerrainExpectation(
                label: "outer:grandes_jorasses", x: 6447.9, z: 1133.6,
                rawY: 4063.5703, y: 4063.5703,
                normal: SIMD3<Float>(0.57867015, 0.78715967, -0.21335536),
                slopeDeg: 38.079135, ao: 0.7, powder: 0.5, curv: 0.5,
                surface: .powder, inCore: false
            ),
            TerrainExpectation(
                label: "outer:far_nw", x: -12000.0, z: -12000.0,
                rawY: 1918.7457, y: 1918.7457,
                normal: SIMD3<Float>(0.57426549, 0.81454831, -0.08203778),
                slopeDeg: 35.457275, ao: 0.7, powder: 0.5, curv: 0.5,
                surface: .packed, inCore: false
            ),
        ]

        for exp in expected {
            let raw = terrain.sampleRawDEM(x: exp.x, z: exp.z)
            XCTAssertEqual(raw.inCore, exp.inCore, "\(exp.label) inCore mismatch")
            XCTAssertEqual(raw.elev, exp.rawY, accuracy: 0.01, "\(exp.label) rawY mismatch")

            let sample = terrain.sampleTerrain(x: exp.x, z: exp.z)
            XCTAssertEqual(sample.rawY, exp.rawY, accuracy: 0.01, "\(exp.label) sampleTerrain rawY mismatch")
            XCTAssertEqual(sample.y, exp.y, accuracy: 0.02, "\(exp.label) y mismatch")
            XCTAssertEqual(sample.normal.x, exp.normal.x, accuracy: 0.002, "\(exp.label) nx mismatch")
            XCTAssertEqual(sample.normal.y, exp.normal.y, accuracy: 0.002, "\(exp.label) ny mismatch")
            XCTAssertEqual(sample.normal.z, exp.normal.z, accuracy: 0.002, "\(exp.label) nz mismatch")
            XCTAssertEqual(sample.slopeDeg, exp.slopeDeg, accuracy: 0.05, "\(exp.label) slopeDeg mismatch")
            XCTAssertEqual(sample.ao, exp.ao, accuracy: 0.002, "\(exp.label) ao mismatch")
            XCTAssertEqual(sample.powder, exp.powder, accuracy: 0.002, "\(exp.label) powder mismatch")
            XCTAssertEqual(sample.curv, exp.curv, accuracy: 0.002, "\(exp.label) curv mismatch")
            XCTAssertEqual(sample.surface, exp.surface, "\(exp.label) surface mismatch")
        }
    }

    func testRealChamonixSimulationDescentOnValleeBlanche() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal device unavailable")
        }
        let terrain = try ChamonixTerrainData(device: device, commandQueue: queue)
        let sim = GameSimulation(terrainData: terrain)
        sim.avalanchesEnabled = false
        let dz = try XCTUnwrap(sim.spawnAtZone(index: 0))
        XCTAssertEqual(dz.id, "vallee_blanche")

        let startY = sim.state.y
        _ = sim.stepSeconds(4.0, keys: ["KeyW"])
        XCTAssertTrue(sim.state.grounded)
        XCTAssertFalse(sim.state.crashed)
        XCTAssertGreaterThan(sim.state.speed, 10.0)
        XCTAssertLessThan(sim.state.y, startY - 5.0)
        XCTAssertGreaterThan(sim.replayFrames.count, 50)
    }
}
