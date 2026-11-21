import XCTest
import Metal
import simd

final class CoreParityTests: XCTestCase {

    // MARK: - Math & Projection Parity

    func testPerspectiveMatrixMetalNDCZ() {
        let fovy = degToRad(86.0)
        let aspect: Float = 16.0 / 9.0
        let nearZ: Float = 0.09
        let farZ: Float = 52000.0
        let proj = matrixPerspectiveRightHand(fovyRadians: fovy, aspect: aspect, nearZ: nearZ, farZ: farZ)

        // Point on near plane at center (0, 0, -nearZ) -> NDC z == 0
        let pNear = proj * SIMD4<Float>(0, 0, -nearZ, 1)
        let ndcNear = pNear.z / pNear.w
        XCTAssertEqual(ndcNear, 0.0, accuracy: 1e-5, "Near plane should map to Metal NDC Z = 0")

        // Point on far plane at center (0, 0, -farZ) -> NDC z == 1
        let pFar = proj * SIMD4<Float>(0, 0, -farZ, 1)
        let ndcFar = pFar.z / pFar.w
        XCTAssertEqual(ndcFar, 1.0, accuracy: 1e-5, "Far plane should map to Metal NDC Z = 1")

        // Point ahead of camera at (0, 0, -10) -> 0 < NDC z < 1, w == 10
        let pMid = proj * SIMD4<Float>(0, 0, -10.0, 1)
        XCTAssertEqual(pMid.w, 10.0, accuracy: 1e-5)
        let ndcMid = pMid.z / pMid.w
        XCTAssertGreaterThan(ndcMid, 0.0)
        XCTAssertLessThan(ndcMid, 1.0)
    }

    func testLookAtRightHand() {
        let eye = SIMD3<Float>(0, 10, 20)
        let target = SIMD3<Float>(0, 10, 0)
        let view = matrixLookAtRightHand(eye: eye, target: target)
        let pView = view * SIMD4<Float>(target, 1)
        // Looking down -Z: target at distance 20 is at (0, 0, -20) in view space
        XCTAssertEqual(pView.x, 0.0, accuracy: 1e-5)
        XCTAssertEqual(pView.y, 0.0, accuracy: 1e-5)
        XCTAssertEqual(pMidSafe(pView.z), -20.0, accuracy: 1e-5)
    }

    private func pMidSafe(_ v: Float) -> Float { v }

    func testEulerYXZRoundTrip() {
        let pitch: Float = 0.31
        let yaw: Float = -1.24
        let roll: Float = 0.18
        let q = quatFromEulerYXZ(pitch: pitch, yaw: yaw, roll: roll)
        let (p2, y2, r2) = eulerYXZFromQuat(q)
        XCTAssertEqual(p2, pitch, accuracy: 1e-5)
        XCTAssertEqual(y2, yaw, accuracy: 1e-5)
        XCTAssertEqual(r2, roll, accuracy: 1e-5)
    }

    func testProjectWorldToScreenUV() {
        let view = matrixLookAtRightHand(eye: SIMD3<Float>(0, 0, 10), target: SIMD3<Float>(0, 0, 0))
        let proj = matrixPerspectiveRightHand(fovyRadians: degToRad(60.0), aspect: 1.0, nearZ: 0.1, farZ: 100.0)
        let vp = proj * view
        let centerUV = projectWorldToScreenUV(SIMD3<Float>(0, 0, 0), viewProjection: vp)
        XCTAssertEqual(centerUV.x, 0.5, accuracy: 1e-5)
        XCTAssertEqual(centerUV.y, 0.5, accuracy: 1e-5)

        // Top of view (+Y in world) maps to uv.y < 0.5 in Metal top-left UV convention
        let topUV = projectWorldToScreenUV(SIMD3<Float>(0, 3, 0), viewProjection: vp)
        XCTAssertEqual(topUV.x, 0.5, accuracy: 1e-5)
        XCTAssertLessThan(topUV.y, 0.5)
    }

    func testMulberry32ParityWithJS() {
        var rng = Mulberry32(seed: 4808)
        let expected: [Float] = [
            0.1764770329464227,
            0.8827490569092333,
            0.43914003088139,
            0.5961605950724334,
            0.20871524419635534,
        ]
        for (idx, exp) in expected.enumerated() {
            let val = rng.next()
            XCTAssertEqual(val, exp, accuracy: 1e-7, "Mulberry32 mismatch at step \(idx)")
        }
    }

    // MARK: - Lighting Presets Parity

    func testLightingPresetsMatchJS() {
        XCTAssertEqual(LightingPresets.all.count, 5)
        XCTAssertEqual(LightingPresets.all.map(\.id), ["afternoon", "morning", "noon", "alpenglow", "storm"])

        for preset in LightingPresets.all {
            let dir = preset.sunDirection
            XCTAssertEqual(simd_length(dir), 1.0, accuracy: 1e-5, "Sun direction must be unit length for \(preset.id)")
            XCTAssertGreaterThan(dir.y, 0.0, "Sun elevation must be above horizon for \(preset.id)")
            let hbSum = preset.horizonBlend.x + preset.horizonBlend.y + preset.horizonBlend.z + preset.horizonBlend.w
            XCTAssertEqual(hbSum, 1.0, accuracy: 1e-5, "horizonBlend weights must sum to 1 for \(preset.id)")
        }

        // Afternoon preset (218 deg az, 21 deg el)
        let afternoon = LightingPresets.preset(id: "afternoon")!
        XCTAssertEqual(afternoon.azDeg, 218.0, accuracy: 1e-4)
        XCTAssertEqual(afternoon.elDeg, 21.0, accuracy: 1e-4)
        XCTAssertEqual(afternoon.sunIntensity, 2.15, accuracy: 1e-4)
        XCTAssertEqual(afternoon.exposure, 0.72, accuracy: 1e-4)
    }

    // MARK: - Asset & Texture Loader Parity

    func testAllPublicAssetsAccessibleAndWebPDecodes() throws {
        let requiredAssets = [
            "chamonix_meta.json",
            "dem_packed.json",
            "core_dem_res.webp",
            "core_dem_ao.webp",
            "core_aux1.webp",
            "core_aux2.webp",
            "outer_dem_res.webp",
            "outer_dem_ao.webp",
            "outer_aux1.webp",
            "outer_aux2.webp",
            "granite_albedo.webp",
            "granite_normal.webp",
            "snow_normal.webp",
            "snow_cavity.webp",
            "ice_normal.webp",
            "powder_puff.webp",
            "spruce_bough.webp",
            "ski_topsheet.webp",
        ]
        for name in requiredAssets {
            let url = try TextureLoader.assetURL(for: name)
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "Missing asset: \(name)")
        }

        // Verify opaque DEM WebP decodes to 1024x1024 with alpha=255 and exact first pixels
        let coreResURL = try TextureLoader.assetURL(for: "core_dem_res.webp")
        let raw = try TextureLoader.loadRawRGBA8(url: coreResURL)
        XCTAssertEqual(raw.width, 1024)
        XCTAssertEqual(raw.height, 1024)
        XCTAssertEqual(raw.data.count, 1024 * 1024 * 4)
        XCTAssertEqual(Array(raw.data.prefix(8)), [128, 128, 128, 255, 142, 142, 142, 255])

        // Verify transparent WebP (powder_puff.webp) preserves zero alpha at borders
        let puffURL = try TextureLoader.assetURL(for: "powder_puff.webp")
        let puffRaw = try TextureLoader.loadRawRGBA8(url: puffURL)
        XCTAssertEqual(puffRaw.width, 128)
        XCTAssertEqual(puffRaw.height, 128)
        XCTAssertEqual(puffRaw.data[3], 0, "Corner pixel of powder_puff.webp must have alpha=0")
        XCTAssertGreaterThan(puffRaw.data[(64 * 128 + 64) * 4 + 3], 180, "Center pixel of powder_puff.webp must be opaque")

        // Verify MTLTexture upload if Metal device available
        if let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() {
            let graniteURL = try TextureLoader.assetURL(for: "granite_albedo.webp")
            let tex = try TextureLoader.loadMTLTexture(device: device, commandQueue: queue, url: graniteURL, srgb: true, mipmaps: true)
            XCTAssertGreaterThan(tex.width, 0)
            XCTAssertGreaterThan(tex.height, 0)
            XCTAssertGreaterThan(tex.mipmapLevelCount, 1)
        }
    }

    // MARK: - Mesh Primitives Validity

    func testMeshPrimitivesWindingAndBounds() {
        let (verts, indices) = MeshPrimitives.makeBox(width: 2, height: 3, depth: 4)
        XCTAssertFalse(verts.isEmpty)
        XCTAssertEqual(indices.count % 3, 0)
        for idx in indices {
            XCTAssertLessThan(Int(idx), verts.count)
        }
        // Check outward CCW winding on every triangle
        var t = 0
        while t + 2 < indices.count {
            let v0 = verts[Int(indices[t])]
            let v1 = verts[Int(indices[t + 1])]
            let v2 = verts[Int(indices[t + 2])]
            let faceNormal = simd_cross(v1.position - v0.position, v2.position - v0.position)
            XCTAssertGreaterThan(simd_dot(faceNormal, v0.normal), 0.0, "Box triangle \(t/3) winding must match outward normal")
            t += 3
        }
    }
}
