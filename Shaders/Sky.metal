#include "ShaderTypes.h"

struct SkyVertexOut {
    float4 position [[position]];
    float3 worldDir;
};

struct ValleyCloudVertexOut {
    float4 position [[position]];
    float3 worldPos;
};

inline float skyHash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

inline float skyVnoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = skyHash21(i);
    float b = skyHash21(i + float2(1.0, 0.0));
    float c = skyHash21(i + float2(0.0, 1.0));
    float d = skyHash21(i + float2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

inline float skyFbm(float2 p) {
    float v = 0.0;
    float amp = 0.52;
    float2x2 rot = float2x2(float2(0.80, -0.60), float2(0.60, 0.80));
    for (int i = 0; i < 5; i++) {
        v += amp * skyVnoise(p);
        p = rot * p * 2.04 + float2(13.1, 7.7);
        amp *= 0.5;
    }
    return v;
}

vertex SkyVertexOut skyVertex(
    uint vertexID [[vertex_id]],
    const device StandardVertex *vertices [[buffer(BufferIndexVertices)]],
    constant FrameUniforms &frame [[buffer(BufferIndexFrameUniforms)]]
) {
    SkyVertexOut out;
    float3 pos = vertices[vertexID].position;
    out.worldDir = normalize(pos);
    float3 wp = pos + frame.cameraPosition.xyz;
    out.position = frame.viewProjectionMatrix * float4(wp, 1.0);
    return out;
}

fragment float4 skyFragment(
    SkyVertexOut in [[stage_in]],
    constant FrameUniforms &frame [[buffer(BufferIndexFrameUniforms)]]
) {
    float3 uSunDir = frame.sunDirection.xyz;
    float3 uSunColor = frame.sunColor.rgb;
    float3 uZenith = frame.skyZenith.rgb;
    float3 uHorizon = frame.skyHorizon.rgb;
    float uTime = frame.cameraPosition.w;
    float uCloudCov = frame.skyZenith.w;
    float uStorm = frame.skyHorizon.w;

    float3 d = normalize(in.worldDir);
    float el = clamp(d.y, -0.15, 1.0);
    float tGrad = pow(max(0.0, el), 0.52);

    float3 col = mix(uHorizon, uZenith, tGrad);

    float3 sunN = normalize(uSunDir);
    float sunCos = clamp(dot(d, sunN), -1.0, 1.0);
    float sunAng = acos(sunCos);
    float mieWide = pow(max(0.0, sunCos), 5.5) * (1.0 - max(0.0, el) * 0.62);
    float mieMid = exp(-sunAng * 10.5);
    float mieCore = exp(-sunAng * 38.0);
    float sunDisk = (1.0 - smoothstep(0.011, 0.022, sunAng)) * 0.85 + exp(-sunAng * sunAng * 6800.0);

    float clearFactor = 1.0 - uStorm * 0.88;
    col += uSunColor * (0.26 * mieWide + 0.42 * mieMid + 1.15 * mieCore) * clearFactor;
    col += mix(uSunColor, float3(1.0, 0.99, 0.96), 0.55) * 4.4 * sunDisk * clearFactor;

    if (d.y < 0.0) {
        col = mix(uHorizon, uHorizon * 0.68, clamp(-d.y * 4.5, 0.0, 1.0));
    }

    if (d.y > 0.012) {
        float2 windDir = float2(0.0085, 0.0032);
        float horFade = smoothstep(0.012, 0.14, d.y);
        float2 sunStep = normalize(sunN.xz + float2(1e-4));

        // Layer 1: High thin alpine cirrus streaks
        float2 uvHi = d.xz / (d.y + 0.18);
        float hiF = skyFbm(uvHi * float2(1.4, 0.55) + windDir * uTime * 1.15);
        float hiStreak = skyFbm(uvHi * float2(3.1, 1.2) - windDir * uTime * 0.4 + float2(hiF * 0.75, 0.0));
        float cirrusMask = smoothstep(mix(0.66, 0.34, uCloudCov), mix(0.92, 0.68, uCloudCov), hiF * 0.6 + hiStreak * 0.4)
                         * horFade * smoothstep(0.02, 0.25, d.y);
        float3 cirrusCol = mix(uHorizon * 1.25, float3(0.98, 0.97, 0.99) * (0.55 + 0.45 * uSunColor), 0.72)
                         + uSunColor * 0.32 * mieWide * clearFactor;
        col = mix(col, cirrusCol, cirrusMask * mix(0.42, 0.68, uStorm));

        // Layer 2: Volumetric sunlit orographic cumulus/lenticular billows with 3-tap sun-direction optical depth
        float2 uv = d.xz / (d.y + 0.11);
        float2 wind = windDir * uTime;
        float w = skyFbm(uv * float2(0.72, 1.32) + wind);
        float c2 = skyFbm(uv * float2(1.95, 0.92) - wind * 0.55 + float2(w * 0.88, -w * 0.48));
        float baseDens = w * 0.62 + c2 * 0.44;
        float thresh = mix(0.64, 0.26, uCloudCov);
        float cloudMask = smoothstep(thresh, thresh + 0.25, baseDens) * horFade;

        if (cloudMask > 0.005) {
            float s1 = skyFbm((uv + sunStep * 0.045) * float2(0.72, 1.32) + wind);
            float s2 = skyFbm((uv + sunStep * 0.105) * float2(0.72, 1.32) + wind);
            float s3 = skyFbm((uv + sunStep * 0.190) * float2(0.72, 1.32) + wind);
            float depthOcc = clamp((s1 * 0.48 + s2 * 0.34 + s3 * 0.18) - thresh * 0.72, 0.0, 0.95);
            float lit = clamp(exp(-depthOcc * 2.8) * 0.92 + (w - s1) * 1.85 + 0.18, 0.0, 1.15);
            float powderPhase = (1.0 - exp(-cloudMask * 2.4)) * pow(max(0.0, sunCos), 3.0) * clearFactor;
            float3 cloudDark = mix(uZenith * 1.40 + float3(0.22, 0.27, 0.37), float3(0.30, 0.35, 0.43), uStorm);
            float3 cloudLit = mix(float3(0.95, 0.97, 1.0) * (0.58 + 0.42 * uSunColor), uSunColor * 1.38, 0.36 + 0.38 * mieWide);
            float3 cloudCol = mix(cloudDark, cloudLit, clamp(lit * (1.0 - uStorm * 0.55) + 0.12, 0.0, 1.0));
            float silverRim = (1.0 - cloudMask * 0.62) * (0.48 * pow(max(0.0, sunCos), 8.0) + 0.35 * powderPhase) * clearFactor;
            cloudCol += uSunColor * silverRim;
            col = mix(col, cloudCol, cloudMask * mix(0.78, 0.95, uStorm));
        }
    }

    return float4(col, 1.0);
}

// MARK: - Valley Sea-of-Clouds (Mer de Nuages)

inline float vcHash21(float2 p) {
    p = fract(p * float2(127.1, 311.7));
    p += dot(p, p + 34.5);
    return fract(p.x * p.y);
}

inline float vcVnoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(vcHash21(i), vcHash21(i + float2(1.0, 0.0)), u.x),
        mix(vcHash21(i + float2(0.0, 1.0)), vcHash21(i + float2(1.0, 1.0)), u.x),
        u.y
    );
}

inline float vcFbm(float2 p) {
    return vcVnoise(p) * 0.55 + vcVnoise(p * 2.13 + 7.4) * 0.30 + vcVnoise(p * 4.31 - 3.9) * 0.15;
}

vertex ValleyCloudVertexOut valleyCloudVertex(
    uint vertexID [[vertex_id]],
    const device StandardVertex *vertices [[buffer(BufferIndexVertices)]],
    constant FrameUniforms &frame [[buffer(BufferIndexFrameUniforms)]]
) {
    ValleyCloudVertexOut out;
    float3 pos = vertices[vertexID].position;
    pos.y += frame.cloudParams.x; // valleyCloudAlt
    out.worldPos = pos;
    out.position = frame.viewProjectionMatrix * float4(pos, 1.0);
    return out;
}

fragment float4 valleyCloudFragment(
    ValleyCloudVertexOut in [[stage_in]],
    constant FrameUniforms &frame [[buffer(BufferIndexFrameUniforms)]]
) {
    float uTime = frame.cameraPosition.w;
    float3 uCamPos = frame.cameraPosition.xyz;
    float3 uSunDir = frame.sunDirection.xyz;
    float3 uSunColor = frame.sunColor.rgb;
    float3 uShadowColor = frame.shadowSkyColor.rgb;
    float uDensity = frame.cloudParams.y;
    float uTwilight = frame.groundBounceColor.w;

    float2 p = in.worldPos.xz * 0.00042 + float2(uTime * 0.004, uTime * 0.0018);
    float n = vcFbm(p);
    float n2 = vcFbm(p + normalize(uSunDir.xz + float2(1e-4)) * 0.14);
    float dist = length(in.worldPos.xz - uCamPos.xz);

    float aboveFade = smoothstep(80.0, 380.0, uCamPos.y - in.worldPos.y);
    float distFade = smoothstep(550.0, 2600.0, dist) * (1.0 - smoothstep(18000.0, 25000.0, dist));
    float billow = smoothstep(mix(0.32, 0.44, uTwilight), 0.80, n);
    float lit = clamp((n - n2) * 3.2 + 0.62, 0.0, 1.0);
    float3 col = mix(uShadowColor * 1.15, float3(0.95, 0.97, 1.0) * (0.5 + 0.5 * uSunColor), lit);

    if (uTwilight > 0.02) {
        float2 dChamV = in.worldPos.xz - float2(-2759.0, -5001.0);
        float2 dBossV = in.worldPos.xz - float2(-3900.0, -3600.0);
        float2 dHouchV = in.worldPos.xz - float2(-5200.0, -2500.0);
        float2 dPrazV = in.worldPos.xz - float2(-1750.0, -5850.0);
        float dCham = length(dChamV);
        float dBoss = length(dBossV);
        float dHouch = length(dHouchV);
        float dPraz = length(dPrazV);
        float townHalo = exp(-dCham * dCham / 1.9e6) * 1.15
                       + exp(-dBoss * dBoss / 1.1e6) * 0.72
                       + exp(-dHouch * dHouch / 1.2e6) * 0.68
                       + exp(-dPraz * dPraz / 0.9e6) * 0.55;
        col += float3(1.0, 0.34, 0.03) * min(townHalo, 1.35) * uTwilight * (0.28 + 0.42 * (1.0 - billow));
    }

    float alpha = billow * uDensity * mix(1.0, 0.44, uTwilight) * distFade * aboveFade * 0.68;
    if (alpha < 0.01) {
        discard_fragment();
    }
    return float4(col, alpha);
}
