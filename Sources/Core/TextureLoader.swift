import Foundation
import CoreGraphics
import ImageIO
import Metal

public enum TextureLoaderError: Error, LocalizedError {
    case fileNotFound(String)
    case imageDecodeFailed(String)
    case textureCreationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let name): return "Asset file not found: \(name)"
        case .imageDecodeFailed(let name): return "Failed to decode image via ImageIO: \(name)"
        case .textureCreationFailed(let name): return "Failed to allocate MTLTexture: \(name)"
        }
    }
}

public struct RawRGBAImage: Sendable {
    public let data: [UInt8]
    public let width: Int
    public let height: Int
}

public enum TextureLoader {
    private final class BundleToken {}

    /// Resolves an asset file inside `Bundle.main` (`assets/`), test bundle, or falls back to workspace `public/assets/`.
    public static func assetURL(for filename: String, searchPaths: [URL] = []) throws -> URL {
        let ns = filename as NSString
        let base = ns.deletingPathExtension
        let ext = ns.pathExtension

        for bundle in [Bundle.main, Bundle(for: BundleToken.self)] {
            if let bundled = bundle.url(forResource: base, withExtension: ext.isEmpty ? nil : ext, subdirectory: "assets") {
                return bundled
            }
            if let resURL = bundle.resourceURL {
                let direct = resURL.appendingPathComponent("assets").appendingPathComponent(filename)
                if FileManager.default.fileExists(atPath: direct.path) {
                    return direct
                }
            }
        }
        for root in searchPaths {
            let candidate = root.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        // Development / XCTest fallback relative to CWD or source tree
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let devCandidates = [
            cwd.appendingPathComponent("Resources/assets").appendingPathComponent(filename),
            cwd.appendingPathComponent("ChamonixSkiSim/Resources/assets").appendingPathComponent(filename),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent() // Core
                .deletingLastPathComponent() // Sources
                .deletingLastPathComponent() // ChamonixSkiSim
                .appendingPathComponent("Resources/assets")
                .appendingPathComponent(filename),
            cwd.appendingPathComponent("public/assets").appendingPathComponent(filename),
            cwd.appendingPathComponent("../public/assets").appendingPathComponent(filename),
            cwd.appendingPathComponent("ChamonixSkiSim/public/assets").appendingPathComponent(filename),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent() // Core
                .deletingLastPathComponent() // Sources
                .deletingLastPathComponent() // ChamonixSkiSim
                .deletingLastPathComponent() // repo root
                .appendingPathComponent("public/assets")
                .appendingPathComponent(filename),
        ]
        for url in devCandidates where fm.fileExists(atPath: url.path) {
            return url
        }
        throw TextureLoaderError.fileNotFound(filename)
    }

    /// Decodes a `.webp` (or PNG/JPEG) into raw RGBA8 bytes without premultiplied alpha or color-space re-encoding.
    /// Row 0 is top of image (North = -halfSpan for DEMs, matching `flipY = false`).
    public static func loadRawRGBA8(url: URL) throws -> RawRGBAImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw TextureLoaderError.imageDecodeFailed(url.lastPathComponent)
        }
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerRow = width * 4

        // Check if image has alpha; for opaque RGB/DEM images use noneSkipLast so RGB bytes are never premultiplied/altered.
        let alphaInfo = cgImage.alphaInfo
        let hasAlpha = (alphaInfo == .first || alphaInfo == .last || alphaInfo == .premultipliedFirst || alphaInfo == .premultipliedLast)
        let targetAlpha: CGImageAlphaInfo = hasAlpha ? .premultipliedLast : .noneSkipLast
        var rgba = [UInt8](repeating: hasAlpha ? 0 : 255, count: width * height * 4)
        let colorSpace = cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | targetAlpha.rawValue
        let drawRect = CGRect(x: 0, y: 0, width: width, height: height)

        let ok = rgba.withUnsafeMutableBytes { ptr -> Bool in
            guard let base = ptr.baseAddress,
                  let ctx = CGContext(
                    data: base,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: colorSpace,
                    bitmapInfo: bitmapInfo
                  ) else {
                return false
            }
            ctx.interpolationQuality = .none
            if hasAlpha {
                ctx.clear(drawRect)
            }
            ctx.setBlendMode(.copy)
            ctx.draw(cgImage, in: drawRect)
            return true
        }
        if !ok {
            // Fallback to DeviceRGB if source colorspace wasn't writable in CGContext
            let fallbackSpace = CGColorSpaceCreateDeviceRGB()
            let okFallback = rgba.withUnsafeMutableBytes { ptr -> Bool in
                guard let base = ptr.baseAddress,
                      let ctx = CGContext(
                        data: base,
                        width: width,
                        height: height,
                        bitsPerComponent: 8,
                        bytesPerRow: bytesPerRow,
                        space: fallbackSpace,
                        bitmapInfo: bitmapInfo
                      ) else {
                    return false
                }
                ctx.interpolationQuality = .none
                if hasAlpha {
                    ctx.clear(drawRect)
                }
                ctx.setBlendMode(.copy)
                ctx.draw(cgImage, in: drawRect)
                return true
            }
            if !okFallback {
                throw TextureLoaderError.imageDecodeFailed(url.lastPathComponent)
            }
        }
        return RawRGBAImage(data: rgba, width: width, height: height)
    }

    /// Loads an image file into a Metal `MTLTexture` (optionally sRGB and/or mipmapped).
    public static func loadMTLTexture(
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        url: URL,
        srgb: Bool = false,
        mipmaps: Bool = true
    ) throws -> MTLTexture {
        let raw = try loadRawRGBA8(url: url)
        let pixelFormat: MTLPixelFormat = srgb ? .rgba8Unorm_srgb : .rgba8Unorm
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: raw.width,
            height: raw.height,
            mipmapped: mipmaps
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw TextureLoaderError.textureCreationFailed(url.lastPathComponent)
        }
        tex.label = url.lastPathComponent
        raw.data.withUnsafeBytes { ptr in
            if let base = ptr.baseAddress {
                tex.replace(
                    region: MTLRegionMake2D(0, 0, raw.width, raw.height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: raw.width * 4
                )
            }
        }
        if mipmaps, tex.mipmapLevelCount > 1,
           let cmdBuf = commandQueue.makeCommandBuffer(),
           let blit = cmdBuf.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: tex)
            blit.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        return tex
    }

    /// Creates a non-mipmapped `MTLTexture` directly from raw RGBA8 bytes (used for decoded 16-bit packed DEMs and procedural textures).
    public static func makeDataTexture(
        device: MTLDevice,
        data: [UInt8],
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat = .rgba8Unorm,
        label: String? = nil
    ) throws -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw TextureLoaderError.textureCreationFailed(label ?? "DataTexture")
        }
        tex.label = label
        data.withUnsafeBytes { ptr in
            if let base = ptr.baseAddress {
                tex.replace(
                    region: MTLRegionMake2D(0, 0, width, height),
                    mipmapLevel: 0,
                    withBytes: base,
                    bytesPerRow: width * 4
                )
            }
        }
        return tex
    }
}
