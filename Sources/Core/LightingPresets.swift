import Foundation
import simd

/// Calibrated physical alpine lighting presets for high-altitude Chamonix (1,000–4,800 m).
/// Exact 1:1 port of `src/lighting_presets.js`.
public struct LightingPreset: Identifiable, Sendable {
    public let id: String
    public let name: String
    public let subtitle: String
    public let azDeg: Float
    public let elDeg: Float
    public let sunColor: SIMD3<Float>
    public let sunIntensity: Float
    public let skyZenith: SIMD3<Float>
    public let skyHorizon: SIMD3<Float>
    public let shadowSkyColor: SIMD3<Float>
    public let shadowFillStrength: Float
    public let groundBounceColor: SIMD3<Float>
    public let sssColor: SIMD3<Float>
    public let fogColor: SIMD3<Float>
    public let fogDensity: Float
    public let valleyCloudAlt: Float
    public let valleyCloudDensity: Float
    public let cloudShadowStrength: Float
    public let cloudCoverage: Float
    public let snowfallRate: Float
    public let exposure: Float
    public let sparkleIntensity: Float
    public let horizonBlend: SIMD4<Float>

    public var isStorm: Bool { id == "storm" }
    public var isAlpenglow: Bool { id == "alpenglow" }
    public var twilightFactor: Float {
        if id == "alpenglow" { return 1.0 }
        if id == "morning" { return 0.18 }
        return 0.0
    }

    public var sunDirection: SIMD3<Float> {
        let az = degToRad(azDeg)
        let el = degToRad(elDeg)
        let cosEl = cosf(el)
        return simd_normalize(SIMD3<Float>(
            sinf(az) * cosEl,
            sinf(el),
            -cosf(az) * cosEl
        ))
    }
}

public enum LightingPresets {
    public static let all: [LightingPreset] = [
        LightingPreset(
            id: "afternoon",
            name: "15:30 ALPINE SUN",
            subtitle: "Crisp low-raking SW sun · deep cobalt shadows",
            azDeg: 218.0,
            elDeg: 21.0,
            sunColor: SIMD3<Float>(1.0, 0.91, 0.78),
            sunIntensity: 2.15,
            skyZenith: SIMD3<Float>(0.018, 0.068, 0.28),
            skyHorizon: SIMD3<Float>(0.38, 0.54, 0.78),
            shadowSkyColor: SIMD3<Float>(0.32, 0.48, 0.78),
            shadowFillStrength: 0.64,
            groundBounceColor: SIMD3<Float>(0.46, 0.56, 0.74),
            sssColor: SIMD3<Float>(0.14, 0.48, 0.90),
            fogColor: SIMD3<Float>(0.42, 0.57, 0.78),
            fogDensity: 0.000046,
            valleyCloudAlt: 1480.0,
            valleyCloudDensity: 0.68,
            cloudShadowStrength: 0.44,
            cloudCoverage: 0.24,
            snowfallRate: 0.10,
            exposure: 0.72,
            sparkleIntensity: 3.4,
            horizonBlend: SIMD4<Float>(0.0, 0.0, 1.0, 0.0)
        ),
        LightingPreset(
            id: "morning",
            name: "09:15 CRISP MORNING",
            subtitle: "Sharp eastern raking light · long blue valley shadows",
            azDeg: 112.0,
            elDeg: 19.0,
            sunColor: SIMD3<Float>(1.0, 0.89, 0.74),
            sunIntensity: 2.05,
            skyZenith: SIMD3<Float>(0.015, 0.062, 0.27),
            skyHorizon: SIMD3<Float>(0.42, 0.56, 0.78),
            shadowSkyColor: SIMD3<Float>(0.30, 0.46, 0.78),
            shadowFillStrength: 0.60,
            groundBounceColor: SIMD3<Float>(0.44, 0.54, 0.72),
            sssColor: SIMD3<Float>(0.12, 0.46, 0.92),
            fogColor: SIMD3<Float>(0.45, 0.58, 0.76),
            fogDensity: 0.000052,
            valleyCloudAlt: 1580.0,
            valleyCloudDensity: 0.82,
            cloudShadowStrength: 0.36,
            cloudCoverage: 0.20,
            snowfallRate: 0.08,
            exposure: 0.76,
            sparkleIntensity: 3.8,
            horizonBlend: SIMD4<Float>(1.0, 0.0, 0.0, 0.0)
        ),
        LightingPreset(
            id: "noon",
            name: "12:30 HIGH NOON",
            subtitle: "High-altitude sapphire zenith · vivid diamond dust",
            azDeg: 162.0,
            elDeg: 38.0,
            sunColor: SIMD3<Float>(1.0, 0.96, 0.90),
            sunIntensity: 2.18,
            skyZenith: SIMD3<Float>(0.012, 0.048, 0.23),
            skyHorizon: SIMD3<Float>(0.34, 0.52, 0.80),
            shadowSkyColor: SIMD3<Float>(0.29, 0.48, 0.80),
            shadowFillStrength: 0.60,
            groundBounceColor: SIMD3<Float>(0.48, 0.58, 0.76),
            sssColor: SIMD3<Float>(0.12, 0.52, 0.94),
            fogColor: SIMD3<Float>(0.39, 0.56, 0.80),
            fogDensity: 0.000036,
            valleyCloudAlt: 1360.0,
            valleyCloudDensity: 0.38,
            cloudShadowStrength: 0.30,
            cloudCoverage: 0.14,
            snowfallRate: 0.05,
            exposure: 0.66,
            sparkleIntensity: 4.2,
            horizonBlend: SIMD4<Float>(0.0, 1.0, 0.0, 0.0)
        ),
        LightingPreset(
            id: "alpenglow",
            name: "17:35 ALPENGLOW",
            subtitle: "Low amber-rose west sun · deep indigo twilight",
            azDeg: 258.0,
            elDeg: 10.5,
            sunColor: SIMD3<Float>(1.0, 0.48, 0.26),
            sunIntensity: 2.45,
            skyZenith: SIMD3<Float>(0.02, 0.04, 0.16),
            skyHorizon: SIMD3<Float>(0.76, 0.42, 0.34),
            shadowSkyColor: SIMD3<Float>(0.11, 0.19, 0.48),
            shadowFillStrength: 0.40,
            groundBounceColor: SIMD3<Float>(0.20, 0.24, 0.40),
            sssColor: SIMD3<Float>(0.18, 0.38, 0.84),
            fogColor: SIMD3<Float>(0.40, 0.35, 0.50),
            fogDensity: 0.000058,
            valleyCloudAlt: 1620.0,
            valleyCloudDensity: 0.78,
            cloudShadowStrength: 0.32,
            cloudCoverage: 0.32,
            snowfallRate: 0.06,
            exposure: 0.86,
            sparkleIntensity: 3.0,
            horizonBlend: SIMD4<Float>(0.0, 0.0, 0.0, 1.0)
        ),
        LightingPreset(
            id: "storm",
            name: "ALPINE SQUALL",
            subtitle: "Flat overcast light · driving snow & low visibility",
            azDeg: 218.0,
            elDeg: 24.0,
            sunColor: SIMD3<Float>(0.70, 0.78, 0.88),
            sunIntensity: 0.55,
            skyZenith: SIMD3<Float>(0.18, 0.23, 0.31),
            skyHorizon: SIMD3<Float>(0.40, 0.46, 0.54),
            shadowSkyColor: SIMD3<Float>(0.28, 0.35, 0.46),
            shadowFillStrength: 0.76,
            groundBounceColor: SIMD3<Float>(0.32, 0.38, 0.46),
            sssColor: SIMD3<Float>(0.24, 0.46, 0.72),
            fogColor: SIMD3<Float>(0.46, 0.52, 0.60),
            fogDensity: 0.00056,
            valleyCloudAlt: 2250.0,
            valleyCloudDensity: 0.94,
            cloudShadowStrength: 0.14,
            cloudCoverage: 0.92,
            snowfallRate: 1.0,
            exposure: 0.90,
            sparkleIntensity: 0.15,
            horizonBlend: SIMD4<Float>(0.0, 0.0, 1.0, 0.0)
        ),
    ]

    public static func preset(id: String) -> LightingPreset? {
        all.first { $0.id == id }
    }

    public static func index(forId id: String) -> Int? {
        all.firstIndex { $0.id == id }
    }
}
