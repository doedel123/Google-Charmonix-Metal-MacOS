import Foundation
import Metal

public enum ShaderLibraryError: Error, LocalizedError {
    case fileNotFound(String)
    case headerNotFound
    case compileFailed(String, String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let name): return "Metal shader source not found: \(name)"
        case .headerNotFound: return "ShaderTypes.h not found in bundle or workspace"
        case .compileFailed(let name, let reason): return "Metal compile failed for \(name): \(reason)"
        }
    }
}

/// Loads and compiles `.metal` shader files from `Shaders/` at runtime via `MTLDevice.makeLibrary(source:options:)`,
/// or from `device.makeDefaultLibrary()` when precompiled `.metallib` is present.
public final class ShaderLibrary {
    private static let lock = NSLock()
    private static var cache: [String: MTLLibrary] = [:]

    /// Loads a `MTLLibrary` for a specific `.metal` file (e.g. `"Terrain"` or `"Terrain.metal"`).
    public static func loadLibrary(device: MTLDevice, named shaderName: String) throws -> MTLLibrary {
        let baseName = shaderName.hasSuffix(".metal") ? String(shaderName.dropLast(6)) : shaderName
        let cacheKey = "\(device.registryID):\(baseName)"

        lock.lock()
        if let existing = cache[cacheKey] {
            lock.unlock()
            return existing
        }
        lock.unlock()

        let sourceURL = try resolveShaderURL(baseName: baseName)
        let rawSource = try String(contentsOf: sourceURL, encoding: .utf8)
        let resolvedSource = try resolveIncludes(in: rawSource)

        let opts = MTLCompileOptions()
        opts.fastMathEnabled = true
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: resolvedSource, options: opts)
        } catch {
            throw ShaderLibraryError.compileFailed("\(baseName).metal", error.localizedDescription)
        }

        lock.lock()
        cache[cacheKey] = library
        lock.unlock()
        return library
    }

    public static func makeFunction(device: MTLDevice, libraryNamed: String, functionName: String) throws -> MTLFunction {
        let lib = try loadLibrary(device: device, named: libraryNamed)
        if let fn = lib.makeFunction(name: functionName) {
            return fn
        }
        if let defLib = device.makeDefaultLibrary(), let fn = defLib.makeFunction(name: functionName) {
            return fn
        }
        throw ShaderLibraryError.compileFailed("\(libraryNamed).metal", "Missing function '\(functionName)'")
    }

    private static func resolveIncludes(in source: String) throws -> String {
        let needsHeader = source.contains("ShaderTypes.h")
        guard needsHeader else { return source }
        let headerSource = try loadShaderTypesHeader()
        var out = ""
        for line in source.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("#include") && trimmed.contains("ShaderTypes.h") {
                out += headerSource + "\n"
            } else {
                out += line + "\n"
            }
        }
        return out
    }

    private static func resolveShaderURL(baseName: String) throws -> URL {
        let fileName = "\(baseName).metal"
        if let bundled = Bundle.main.url(forResource: baseName, withExtension: "metal", subdirectory: "Shaders") {
            return bundled
        }
        if let resURL = Bundle.main.resourceURL {
            let direct = resURL.appendingPathComponent("Shaders").appendingPathComponent(fileName)
            if FileManager.default.fileExists(atPath: direct.path) {
                return direct
            }
        }
        let bundle = Bundle(for: ShaderLibrary.self)
        if let bundled = bundle.url(forResource: baseName, withExtension: "metal", subdirectory: "Shaders") {
            return bundled
        }
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let devCandidates = [
            cwd.appendingPathComponent("Shaders").appendingPathComponent(fileName),
            cwd.appendingPathComponent("ChamonixSkiSim/Shaders").appendingPathComponent(fileName),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent() // Core
                .deletingLastPathComponent() // Sources
                .deletingLastPathComponent() // ChamonixSkiSim
                .appendingPathComponent("Shaders")
                .appendingPathComponent(fileName),
        ]
        for url in devCandidates where fm.fileExists(atPath: url.path) {
            return url
        }
        throw ShaderLibraryError.fileNotFound(fileName)
    }

    private static func loadShaderTypesHeader() throws -> String {
        if let url = Bundle.main.url(forResource: "ShaderTypes", withExtension: "h") {
            return try String(contentsOf: url, encoding: .utf8)
        }
        if let url = Bundle.main.url(forResource: "ShaderTypes", withExtension: "h", subdirectory: "Shaders") {
            return try String(contentsOf: url, encoding: .utf8)
        }
        let bundle = Bundle(for: ShaderLibrary.self)
        if let url = bundle.url(forResource: "ShaderTypes", withExtension: "h") {
            return try String(contentsOf: url, encoding: .utf8)
        }
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let devCandidates = [
            cwd.appendingPathComponent("Sources/Core/ShaderTypes.h"),
            cwd.appendingPathComponent("ChamonixSkiSim/Sources/Core/ShaderTypes.h"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .appendingPathComponent("ShaderTypes.h"),
        ]
        for url in devCandidates where fm.fileExists(atPath: url.path) {
            return try String(contentsOf: url, encoding: .utf8)
        }
        throw ShaderLibraryError.headerNotFound
    }
}
