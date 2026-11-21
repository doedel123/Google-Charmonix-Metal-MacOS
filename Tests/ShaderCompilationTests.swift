import XCTest
import Metal

final class ShaderCompilationTests: XCTestCase {

    func testAllMetalShadersCompileAtRuntime() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device unavailable")
        }

        // Discover all .metal files in Shaders/ (either via bundle resources or workspace fallback)
        let fm = FileManager.default
        var shaderNames: [String] = []

        if let resURL = Bundle.main.resourceURL?.appendingPathComponent("Shaders"),
           let items = try? fm.contentsOfDirectory(atPath: resURL.path) {
            shaderNames = items.filter { $0.hasSuffix(".metal") }.map { ($0 as NSString).deletingPathExtension }
        }
        if shaderNames.isEmpty {
            let devShadersDir = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent() // Tests
                .deletingLastPathComponent() // ChamonixSkiSim
                .appendingPathComponent("Shaders")
            if let items = try? fm.contentsOfDirectory(atPath: devShadersDir.path) {
                shaderNames = items.filter { $0.hasSuffix(".metal") }.map { ($0 as NSString).deletingPathExtension }
            }
        }

        XCTAssertFalse(shaderNames.isEmpty, "Expected at least one .metal shader in Shaders/")

        for name in shaderNames.sorted() {
            let lib = try ShaderLibrary.loadLibrary(device: device, named: name)
            XCTAssertFalse(lib.functionNames.isEmpty, "Shader library \(name).metal compiled with zero functions")
        }
    }
}
