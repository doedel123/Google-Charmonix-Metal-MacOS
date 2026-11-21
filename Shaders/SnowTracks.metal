#include "ShaderTypes.h"

struct SnowTrackVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex SnowTrackVertexOut snowTracksVertex(
    uint vertexID [[vertex_id]],
    const device StandardVertex *vertices [[buffer(BufferIndexVertices)]]
) {
    SnowTrackVertexOut out;
    StandardVertex v = vertices[vertexID];
    out.position = float4(v.position.xy, 0.0, 1.0);
    out.uv = v.uv;
    return out;
}

constexpr sampler trackRTSampler(coord::normalized, address::clamp_to_edge, filter::linear);

fragment float4 snowTracksCopyFragment(
    SnowTrackVertexOut in [[stage_in]],
    texture2d<float> prevTex [[texture(0)]],
    constant SnowTrackUniforms &u [[buffer(BufferIndexSubsystemUniforms)]]
) {
    float2 srcUv = in.uv + u.shiftUV;
    if (srcUv.x < 0.002 || srcUv.x > 0.998 || srcUv.y < 0.002 || srcUv.y > 0.998) {
        return float4(0.0);
    }
    float4 c = prevTex.sample(trackRTSampler, srcUv);
    c.rgb = max(float3(0.0), c.rgb - float3(u.decay));
    return c;
}

inline void evalCapsule(
    float2 uv,
    float2 a,
    float2 b,
    float r,
    float depth,
    float berm,
    float iceGlint,
    float skewSide,
    thread float &tr,
    thread float &bm,
    thread float &gl
) {
    float2 ab = b - a;
    float ab2 = dot(ab, ab);
    float t = ab2 > 1e-10 ? clamp(dot(uv - a, ab) / ab2, 0.0, 1.0) : 0.0;
    float2 diff = uv - (a + ab * t);
    float dist = length(diff);
    float2 dir = ab2 > 1e-10 ? normalize(ab) : float2(0.0, 1.0);
    float2 perp = float2(-dir.y, dir.x);
    float side = dot(diff, perp);

    float aa = max(r * 0.48, 0.00145);
    tr = smoothstep(r + aa, max(0.0, r - aa * 0.75), dist) * depth;

    float bermCenter = r + aa * 1.45;
    float bermW = max(r * 0.72, 0.0018);
    float bermRing = exp(-pow((dist - bermCenter) / bermW, 2.0));
    float sideBoost = clamp(1.0 + sign(side) * skewSide * 0.72, 0.40, 1.72);
    bm = bermRing * berm * sideBoost * (1.0 - smoothstep(r + aa * 0.5, max(0.0, r - aa * 0.5), dist));
    gl = smoothstep(r + aa * 0.5, max(0.0, r * 0.15), dist) * iceGlint;
}

fragment float4 snowTracksStampFragment(
    SnowTrackVertexOut in [[stage_in]],
    texture2d<float> prevTex [[texture(0)]],
    constant SnowTrackUniforms &u [[buffer(BufferIndexSubsystemUniforms)]]
) {
    float4 prev = prevTex.sample(trackRTSampler, in.uv);
    float r = max(u.radiusUV, 0.0005);

    float trL = 0.0, bmL = 0.0, glL = 0.0;
    evalCapsule(in.uv, u.lA, u.lB, r, u.depth, u.berm, u.iceGlint, u.skewSide, trL, bmL, glL);

    float tr = trL, bm = bmL, gl = glL;
    if (u.singleMode < 0.5) {
        float trR = 0.0, bmR = 0.0, glR = 0.0;
        evalCapsule(in.uv, u.rA, u.rB, r, u.depth, u.berm, u.iceGlint, u.skewSide, trR, bmR, glR);
        tr = max(trL, trR);
        bm = clamp(max(bmL, bmR) + min(bmL, bmR) * 0.35, 0.0, 1.0);
        gl = max(glL, glR);
    }

    float outR = max(prev.r, tr);
    float outG = clamp(max(prev.g, bm) * (1.0 - outR * 0.78), 0.0, 1.0);
    float outB = max(prev.b, gl);
    return float4(outR, outG, outB, max(prev.a, max(outR, outG)));
}
