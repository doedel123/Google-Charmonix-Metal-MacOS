import Foundation
import simd

// MARK: - Scalar & Vector Helpers

@inlinable
public func clamp(_ x: Float, _ minVal: Float, _ maxVal: Float) -> Float {
    Swift.min(maxVal, Swift.max(minVal, x))
}

@inlinable
public func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
    a + (b - a) * t
}

@inlinable
public func lerp(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
    a + (b - a) * t
}

@inlinable
public func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
    let denom = edge1 - edge0
    guard abs(denom) > 1e-8 else { return x >= edge1 ? 1.0 : 0.0 }
    let t = clamp((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
}

@inlinable
public func degToRad(_ degrees: Float) -> Float {
    degrees * (.pi / 180.0)
}

@inlinable
public func radToDeg(_ radians: Float) -> Float {
    radians * (180.0 / .pi)
}

@inlinable
public func wrapAngle(_ angle: Float) -> Float {
    var r = angle
    while r > .pi { r -= .pi * 2.0 }
    while r < -.pi { r += .pi * 2.0 }
    return r
}

// MARK: - Right-Handed Y-Up Matrices (Metal NDC Z in [0, 1])

/// Right-handed Y-up perspective matrix matching Three.js vertical FOV convention with Metal [0, 1] NDC depth.
@inlinable
public func matrixPerspectiveRightHand(fovyRadians fovy: Float, aspect: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
    let ys = 1.0 / tanf(fovy * 0.5)
    let xs = ys / max(aspect, 1e-4)
    let zs = farZ / (nearZ - farZ)
    return simd_float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, -1),
        SIMD4<Float>(0, 0, zs * nearZ, 0)
    ))
}

/// Right-handed orthographic projection matrix with Metal [0, 1] NDC depth.
@inlinable
public func matrixOrthoRightHand(left: Float, right: Float, bottom: Float, top: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
    let rml = right - left
    let tmb = top - bottom
    let fmn = farZ - nearZ
    return simd_float4x4(columns: (
        SIMD4<Float>(2.0 / rml, 0, 0, 0),
        SIMD4<Float>(0, 2.0 / tmb, 0, 0),
        SIMD4<Float>(0, 0, -1.0 / fmn, 0),
        SIMD4<Float>(-(right + left) / rml, -(top + bottom) / tmb, -nearZ / fmn, 1.0)
    ))
}

/// Right-handed look-at matrix (camera looks down -Z in view space).
@inlinable
public func matrixLookAtRightHand(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> simd_float4x4 {
    let z = simd_normalize(eye - target)
    var x = simd_cross(up, z)
    if simd_length_squared(x) < 1e-8 {
        x = simd_cross(SIMD3<Float>(0, 0, 1), z)
    }
    x = simd_normalize(x)
    let y = simd_cross(z, x)
    return simd_float4x4(columns: (
        SIMD4<Float>(x.x, y.x, z.x, 0),
        SIMD4<Float>(x.y, y.y, z.y, 0),
        SIMD4<Float>(x.z, y.z, z.z, 0),
        SIMD4<Float>(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)
    ))
}

@inlinable
public func matrixTranslation(_ t: SIMD3<Float>) -> simd_float4x4 {
    simd_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(t.x, t.y, t.z, 1)
    ))
}

@inlinable
public func matrixScale(_ s: SIMD3<Float>) -> simd_float4x4 {
    simd_float4x4(columns: (
        SIMD4<Float>(s.x, 0, 0, 0),
        SIMD4<Float>(0, s.y, 0, 0),
        SIMD4<Float>(0, 0, s.z, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}

@inlinable
public func matrixScale(_ s: Float) -> simd_float4x4 {
    matrixScale(SIMD3<Float>(repeating: s))
}

@inlinable
public func matrixTRS(translation: SIMD3<Float>, rotation: simd_quatf, scale: SIMD3<Float> = SIMD3<Float>(repeating: 1)) -> simd_float4x4 {
    let r = simd_float4x4(rotation)
    return simd_float4x4(columns: (
        r.columns.0 * scale.x,
        r.columns.1 * scale.y,
        r.columns.2 * scale.z,
        SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    ))
}

/// Computes an inverse-transpose 4x4 matrix suitable for transforming normals under non-uniform scale.
@inlinable
public func normalMatrix(from m: simd_float4x4) -> simd_float4x4 {
    let c0 = SIMD3<Float>(m.columns.0.x, m.columns.0.y, m.columns.0.z)
    let c1 = SIMD3<Float>(m.columns.1.x, m.columns.1.y, m.columns.1.z)
    let c2 = SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z)
    let m3 = simd_float3x3(columns: (c0, c1, c2))
    let det = simd_determinant(m3)
    if abs(det) < 1e-8 {
        return matrix_identity_float4x4
    }
    let invT = simd_transpose(simd_inverse(m3))
    return simd_float4x4(columns: (
        SIMD4<Float>(invT.columns.0, 0),
        SIMD4<Float>(invT.columns.1, 0),
        SIMD4<Float>(invT.columns.2, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}

// MARK: - Quaternions & Euler YXZ (matching Three.js Euler(x, y, z, 'YXZ'))

/// Constructs a quaternion from intrinsic YXZ Euler angles (matching Three.js `Euler(x, y, z, 'YXZ')`).
@inlinable
public func quatFromEulerYXZ(pitch x: Float, yaw y: Float, roll z: Float) -> simd_quatf {
    let qY = simd_quatf(angle: y, axis: SIMD3<Float>(0, 1, 0))
    let qX = simd_quatf(angle: x, axis: SIMD3<Float>(1, 0, 0))
    let qZ = simd_quatf(angle: z, axis: SIMD3<Float>(0, 0, 1))
    return simd_normalize(qY * qX * qZ)
}

/// Constructs a quaternion from intrinsic XYZ Euler angles (matching Three.js default `'XYZ'`).
@inlinable
public func quatFromEulerXYZ(x: Float, y: Float, z: Float) -> simd_quatf {
    let qX = simd_quatf(angle: x, axis: SIMD3<Float>(1, 0, 0))
    let qY = simd_quatf(angle: y, axis: SIMD3<Float>(0, 1, 0))
    let qZ = simd_quatf(angle: z, axis: SIMD3<Float>(0, 0, 1))
    return simd_normalize(qX * qY * qZ)
}

/// Extracts YXZ Euler angles (pitch, yaw, roll) from a unit quaternion, matching Three.js `Euler.setFromQuaternion(q, 'YXZ')`.
@inlinable
public func eulerYXZFromQuat(_ q: simd_quatf) -> (pitch: Float, yaw: Float, roll: Float) {
    let m = simd_float3x3(q)
    let m13 = clamp(-m.columns.2.y, -1.0, 1.0)
    let pitch = asinf(m13)
    let yaw: Float
    let roll: Float
    if abs(m13) < 0.9999999 {
        yaw = atan2f(m.columns.2.x, m.columns.2.z)
        roll = atan2f(m.columns.0.y, m.columns.1.y)
    } else {
        yaw = atan2f(-m.columns.0.z, m.columns.0.x)
        roll = 0.0
    }
    return (pitch, yaw, roll)
}

@inlinable
public func quatFromBasis(right: SIMD3<Float>, up: SIMD3<Float>, forwardZ: SIMD3<Float>) -> simd_quatf {
    let m = simd_float3x3(columns: (right, up, forwardZ))
    return simd_normalize(simd_quatf(m))
}

@inlinable
public func quatFromUnitVectors(from vFrom: SIMD3<Float>, to vTo: SIMD3<Float>) -> simd_quatf {
    var r = simd_dot(vFrom, vTo) + 1.0
    if r < 1e-6 {
        r = 0.0
        if abs(vFrom.x) > abs(vFrom.z) {
            return simd_normalize(simd_quatf(ix: -vFrom.y, iy: vFrom.x, iz: 0, r: 0))
        } else {
            return simd_normalize(simd_quatf(ix: 0, iy: -vFrom.z, iz: vFrom.y, r: 0))
        }
    }
    let c = simd_cross(vFrom, vTo)
    return simd_normalize(simd_quatf(ix: c.x, iy: c.y, iz: c.z, r: r))
}

/// Projects a world-space point into Metal screen UV space where (0, 0) is top-left and (1, 1) is bottom-right.
@inlinable
public func projectWorldToScreenUV(_ worldPos: SIMD3<Float>, viewProjection: simd_float4x4) -> SIMD2<Float> {
    let clip = viewProjection * SIMD4<Float>(worldPos, 1.0)
    let w = abs(clip.w) > 1e-6 ? clip.w : 1e-6
    let ndcX = clip.x / w
    let ndcY = clip.y / w
    return SIMD2<Float>(ndcX * 0.5 + 0.5, 0.5 - ndcY * 0.5)
}

// MARK: - Deterministic PRNG (Mulberry32 matching alpine_props.js)

public struct Mulberry32 {
    private var state: UInt32

    public init(seed: UInt32) {
        self.state = seed
    }

    public mutating func next() -> Float {
        state &+= 0x6D2B79F5
        var t = (state ^ (state >> 15)) &* (1 | state)
        t = (t &+ ((t ^ (t >> 7)) &* (61 | t))) ^ t
        let res = t ^ (t >> 14)
        return Float(res) / 4294967296.0
    }
}
