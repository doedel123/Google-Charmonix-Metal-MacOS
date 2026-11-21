#include "ShaderTypes.h"

constant float CORE_HALF  = 6400.0;
constant float OUTER_HALF = 24000.0;
constant float MAX_ELEV   = 5000.0;

struct TerrainVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float  isOuter;
    float  dist;
};

constexpr sampler demSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
constexpr sampler auxSampler(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler trackSampler(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler detailSampler(coord::normalized, address::repeat, filter::linear, mip_filter::linear, max_anisotropy(8));

inline float glslMod(float x, float y) {
    return x - y * floor(x / y);
}

inline float2 glslMod(float2 x, float y) {
    return x - y * floor(x / y);
}

inline float decodeRG(float4 t) {
    return ((t.r * 255.0 * 256.0 + t.g * 255.0) / 65535.0) * MAX_ELEV;
}

inline float3 samplePackedDEM(texture2d<float> demTex, float texSize, float2 worldXZ, float halfSpan) {
    float2 uvf = clamp((worldXZ + halfSpan) / (2.0 * halfSpan), 0.0, 1.0) * (texSize - 1.0);
    float2 i0 = floor(uvf);
    float2 iMax = float2(texSize - 1.0);
    float2 i1 = min(i0 + float2(1.0), iMax);
    float2 f = fract(uvf);
    float invS = 1.0 / texSize;

    float4 t00 = demTex.sample(demSampler, (i0 + 0.5) * invS, level(0.0));
    float4 t10 = demTex.sample(demSampler, (float2(i1.x, i0.y) + 0.5) * invS, level(0.0));
    float4 t01 = demTex.sample(demSampler, (float2(i0.x, i1.y) + 0.5) * invS, level(0.0));
    float4 t11 = demTex.sample(demSampler, (i1 + 0.5) * invS, level(0.0));

    float h00 = decodeRG(t00);
    float h10 = decodeRG(t10);
    float h01 = decodeRG(t01);
    float h11 = decodeRG(t11);
    float h = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
    float ao = mix(mix(t00.b, t10.b, f.x), mix(t01.b, t11.b, f.x), f.y);
    return float3(h, ao, 1.0);
}

inline float microRelief(float2 xz, float slopeMag, float powder) {
    float x = xz.x;
    float z = xz.y;
    float w1 = sin(x * 0.047 + z * 0.029) * cos(x * 0.023 - z * 0.051);
    float w2 = sin(x * 0.115 - z * 0.088 + 1.3) * cos(x * 0.072 + z * 0.131 - 0.7);
    float w3 = sin(x * 0.240 + z * 0.195 - 2.1) * sin(x * 0.180 - z * 0.270 + 0.5);
    float roller = sin((x * 0.74 + z * 0.67) * 0.082 + w1 * 1.4) * 0.68;
    float midSlope = clamp((slopeMag - 0.08) / 0.45, 0.0, 1.0);
    float steepRock = clamp((slopeMag - 0.78) / 0.45, 0.0, 1.0);
    float snowAmp = (0.55 + 0.45 * powder) * (0.45 + 0.55 * midSlope) * (1.0 - steepRock * 0.45);
    float rockRib = steepRock * (abs(w2) * 1.65 + w3 * 0.55);
    return (w1 * 1.35 + roller * 0.85 + w2 * 0.42 + w3 * 0.14) * snowAmp + rockRib;
}

inline float evalCoreHeight(
    float2 wxz,
    texture2d<float> uCoreDem,
    texture2d<float> uCoreAux1,
    texture2d<float> uOuterDem,
    texture2d<float> uTrackTex,
    float uCoreSize,
    float uOuterSize,
    float uLevel,
    float2 uTrackOrigin,
    float uTrackSpan
) {
    float3 s = samplePackedDEM(uCoreDem, uCoreSize, wxz, CORE_HALF);
    float2 uv = clamp((wxz + CORE_HALF) / (2.0 * CORE_HALF), 0.0, 1.0);
    float powder = uCoreAux1.sample(auxSampler, uv, level(0.0)).r;

    float e = 8.0;
    float hR = samplePackedDEM(uCoreDem, uCoreSize, wxz + float2(e, 0.0), CORE_HALF).x;
    float hL = samplePackedDEM(uCoreDem, uCoreSize, wxz - float2(e, 0.0), CORE_HALF).x;
    float hS = samplePackedDEM(uCoreDem, uCoreSize, wxz + float2(0.0, e), CORE_HALF).x;
    float hN = samplePackedDEM(uCoreDem, uCoreSize, wxz - float2(0.0, e), CORE_HALF).x;
    float slopeMag = length(float2(hR - hL, hS - hN) / (2.0 * e));
    float h = s.x + microRelief(wxz, slopeMag, powder);

    // Blend smoothly into outer DEM right at the core margin (±5950m..±6400m)
    float edgeD = max(abs(wxz.x), abs(wxz.y));
    if (edgeD > CORE_HALF - 450.0) {
        float blendOut = smoothstep(CORE_HALF - 450.0, CORE_HALF - 20.0, edgeD);
        float hOut = samplePackedDEM(uOuterDem, uOuterSize, wxz, OUTER_HALF).x;
        h = mix(h, hOut, blendOut);
    }

    // Ultra-gentle blurred physical sag near camera
    if (uLevel < 2.5) {
        float2 tUv = (wxz - uTrackOrigin) / uTrackSpan + 0.5;
        if (tUv.x > 0.01 && tUv.x < 0.99 && tUv.y > 0.01 && tUv.y < 0.99) {
            float o = 0.35 / uTrackSpan;
            float trR = (uTrackTex.sample(trackSampler, tUv, level(0.0)).r
                      + uTrackTex.sample(trackSampler, tUv + float2(o, 0.0), level(0.0)).r
                      + uTrackTex.sample(trackSampler, tUv - float2(o, 0.0), level(0.0)).r
                      + uTrackTex.sample(trackSampler, tUv + float2(0.0, o), level(0.0)).r
                      + uTrackTex.sample(trackSampler, tUv - float2(0.0, o), level(0.0)).r) * 0.2;
            float edgeFade = smoothstep(0.01, 0.08, min(min(tUv.x, 1.0 - tUv.x), min(tUv.y, 1.0 - tUv.y)));
            float rockSuppress = 1.0 - smoothstep(0.82, 1.15, slopeMag);
            h -= trR * 0.018 * edgeFade * rockSuppress;
        }
    }

    // Analytical ridged granite sharpening on steep far spires (>36°)
    if (slopeMag > 0.72) {
        float sharpM = clamp((slopeMag - 0.72) / 0.55, 0.0, 1.0);
        float ridgeS = 1.0 - abs(sin(wxz.x * 0.019 + wxz.y * 0.014) * cos(wxz.x * 0.012 - wxz.y * 0.023));
        h += (ridgeS - 0.5) * 4.5 * sharpM;
    }
    return h;
}

vertex TerrainVertexOut terrainVertex(
    uint vertexID [[vertex_id]],
    const device StandardVertex *vertices [[buffer(BufferIndexVertices)]],
    constant FrameUniforms &frame [[buffer(BufferIndexFrameUniforms)]],
    constant TerrainLevelUniforms &level [[buffer(BufferIndexSubsystemUniforms)]],
    texture2d<float> uCoreDem [[texture(TextureIndexCoreDem)]],
    texture2d<float> uCoreAux1 [[texture(TextureIndexCoreAux1)]],
    texture2d<float> uOuterDem [[texture(TextureIndexOuterDem)]],
    texture2d<float> uTrackTex [[texture(TextureIndexTrackTex)]]
) {
    TerrainVertexOut out;
    float3 pos = vertices[vertexID].position;
    float uCoreSize = frame.cloudParams.w;
    float uOuterSize = frame.screenParams.w;
    float3 uCamPos = frame.cameraPosition.xyz;

    if (level.isOuterMesh > 0.5) {
        float2 wxz = pos.xz;
        float h = samplePackedDEM(uOuterDem, uOuterSize, wxz, OUTER_HALF).x;
        float edgeD = max(abs(wxz.x), abs(wxz.y));
        if (edgeD < CORE_HALF + 80.0) {
            h -= (1.0 - smoothstep(CORE_HALF - 380.0, CORE_HALF + 80.0, edgeD)) * 22.0;
        }
        float3 wp = float3(wxz.x, h, wxz.y);
        out.worldPos = wp;
        out.isOuter = 1.0;
        out.dist = length(wp - uCamPos);
        out.position = frame.viewProjectionMatrix * float4(wp, 1.0);
        return out;
    }

    float2 local = pos.xz;
    float2 wxz = level.snapCenter + local * level.gridStep;
    float uGridHalf = level.gridN * 0.5;

    float maxAbs = max(abs(local.x), abs(local.y));
    float morph = smoothstep(uGridHalf * 0.72, uGridHalf * 0.999, maxAbs);
    float2 fracOdd = glslMod(abs(local), 2.0);
    float2 offsetStep = fracOdd * level.gridStep;

    float2 uTrackOrigin = frame.trackParams.xy;
    float uTrackSpan = frame.trackParams.z;

    float h = evalCoreHeight(
        wxz, uCoreDem, uCoreAux1, uOuterDem, uTrackTex,
        uCoreSize, uOuterSize, level.level, uTrackOrigin, uTrackSpan
    );
    if (morph > 0.001 && (fracOdd.x > 0.5 || fracOdd.y > 0.5)) {
        float hCoarse = 0.5 * (
            evalCoreHeight(wxz - offsetStep, uCoreDem, uCoreAux1, uOuterDem, uTrackTex, uCoreSize, uOuterSize, level.level, uTrackOrigin, uTrackSpan)
          + evalCoreHeight(wxz + offsetStep, uCoreDem, uCoreAux1, uOuterDem, uTrackTex, uCoreSize, uOuterSize, level.level, uTrackOrigin, uTrackSpan)
        );
        h = mix(h, hCoarse, morph);
    }

    float3 wp = float3(wxz.x, h, wxz.y);
    out.worldPos = wp;
    out.isOuter = (max(abs(wxz.x), abs(wxz.y)) > CORE_HALF - 40.0) ? 1.0 : 0.0;
    out.dist = length(wp - uCamPos);
    out.position = frame.viewProjectionMatrix * float4(wp, 1.0);
    return out;
}

// MARK: - Fragment Helpers

inline float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

inline float vnoise2(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i), hash21(i + float2(1.0, 0.0)), u.x),
        mix(hash21(i + float2(0.0, 1.0)), hash21(i + float2(1.0, 1.0)), u.x),
        u.y
    );
}

inline float fbm3(float2 p) {
    float2x2 rot = float2x2(float2(0.80, -0.60), float2(0.60, 0.80));
    float v = 0.52 * vnoise2(p);
    p = rot * p * 2.07 + float2(13.7, -9.2);
    v += 0.31 * vnoise2(p);
    p = rot * p * 2.11 + float2(-5.4, 17.3);
    v += 0.17 * vnoise2(p);
    return v;
}

inline float cloudShadowAt(float2 xz, float t) {
    float2 p = (xz - float2(t * 22.0, t * 8.5)) * 0.00052;
    float n = vnoise2(p) * 0.62 + vnoise2(p * 2.15 + float2(7.3, -4.1)) * 0.38;
    return smoothstep(0.50, 0.78, n);
}

inline void sampleTriplanarGranite(
    float3 wp,
    float3 wNorm,
    texture2d<float> uGraniteAlbedo,
    texture2d<float> uGraniteNormal,
    thread float3 &alb,
    thread float3 &triN
) {
    float3 blend = pow(abs(wNorm), float3(4.0));
    blend /= max(blend.x + blend.y + blend.z, 1e-4);

    float sA = 0.022;
    float sB = 0.0055;
    float2 uvX = float2(wp.z * sA + wp.y * 0.009, 1.0 - wp.y * sA);
    float2 uvY = float2(wp.x * sA, 1.0 - wp.z * sA);
    float2 uvZ = float2(wp.x * sA - wp.y * 0.009, 1.0 - wp.y * sA);

    float3 cX = uGraniteAlbedo.sample(detailSampler, uvX).rgb;
    float3 cY = uGraniteAlbedo.sample(detailSampler, uvY).rgb;
    float3 cZ = uGraniteAlbedo.sample(detailSampler, uvZ).rgb;
    float3 cMacro = uGraniteAlbedo.sample(detailSampler, float2(wp.x * sB + wp.y * 0.002, 1.0 - (wp.z * sB + wp.y * 0.002))).rgb;
    alb = (cX * blend.x + cY * blend.y + cZ * blend.z) * (0.68 + 0.62 * cMacro);

    float3 nX = uGraniteNormal.sample(detailSampler, uvX).xyz * 2.0 - 1.0;
    float3 nY = uGraniteNormal.sample(detailSampler, uvY).xyz * 2.0 - 1.0;
    float3 nZ = uGraniteNormal.sample(detailSampler, uvZ).xyz * 2.0 - 1.0;
    float3 tX = float3(0.0, nX.y, nX.x);
    float3 tY = float3(nY.x, 0.0, nY.y);
    float3 tZ = float3(nZ.x, nZ.y, 0.0);
    triN = normalize(wNorm + (tX * blend.x + tY * blend.y + tZ * blend.z) * 0.62);
}

inline float3 snowSparkle(float3 wp, float3 wNorm, float3 viewDir, float3 sunDir, float dist) {
    if (dist > 78.0) return float3(0.0);
    float farFade = 1.0 - smoothstep(14.0, 78.0, dist);
    float nearFade = 1.0 - smoothstep(3.5, 24.0, dist);
    float3 hVec = normalize(sunDir + viewDir);
    float sunAlign = 0.36 + 0.64 * pow(max(0.0, dot(wNorm, hVec)), 2.2);

    float2 wMod = glslMod(wp.xz, 32.0);
    float2 warp = float2(vnoise2(wMod * 0.65 + float2(3.1, 7.4)), vnoise2(wMod * 0.65 - float2(5.2, 2.7))) * 0.22;

    float2x2 rot1 = float2x2(float2(0.848, -0.530), float2(0.530, 0.848));
    float2x2 rot2 = float2x2(float2(0.616, -0.788), float2(0.788, 0.616));

    float2 p1 = rot1 * (wMod + warp) * 26.0;
    float2 c1 = floor(p1);
    float2 h1 = float2(hash21(c1 + float2(1.7, 9.2)), hash21(c1 + float2(8.3, 2.8)));
    float2 f1 = fract(p1) - (0.15 + 0.70 * h1);
    float rad1 = mix(0.20, 0.34, smoothstep(6.0, 46.0, dist));
    float dot1 = smoothstep(rad1, 0.02, length(f1));
    float3 rnd1 = normalize(float3(hash21(c1 + 3.1) - 0.5, 0.85 + hash21(c1 + 7.7) * 0.4, hash21(c1 + 5.9) - 0.5) * 1.25 + wNorm * 0.85);
    float gate1 = step(0.64, hash21(c1 + float2(17.1, 13.7)));
    float g1 = pow(max(0.0, dot(rnd1, hVec)), 88.0) * gate1 * dot1;

    float phase1 = hash21(c1 + float2(41.2, 19.8));
    float3 tint1 = mix(
        mix(float3(1.18, 0.94, 0.72), float3(1.0, 1.04, 1.12), smoothstep(0.0, 0.55, phase1)),
        float3(0.76, 0.96, 1.24),
        smoothstep(0.55, 1.0, phase1)
    );

    float2 p2 = rot2 * (glslMod(wp.xz + float2(11.3, 5.7), 16.0) + warp * 0.6) * 66.0;
    float2 c2 = floor(p2);
    float2 h2 = float2(hash21(c2 + float2(4.1, 6.7)), hash21(c2 + float2(2.9, 8.1)));
    float2 f2 = fract(p2) - (0.14 + 0.72 * h2);
    float dot2 = smoothstep(0.24, 0.02, length(f2));
    float3 rnd2 = normalize(float3(hash21(c2 + 11.3) - 0.5, 0.78 + hash21(c2 + 15.1) * 0.5, hash21(c2 + 23.7) - 0.5) * 1.40 + wNorm * 0.72);
    float gate2 = step(0.60, hash21(c2 + float2(31.7, 7.3)));
    float g2 = pow(max(0.0, dot(rnd2, hVec)), 105.0) * gate2 * dot2 * nearFade;
    float phase2 = hash21(c2 + float2(12.6, 54.1));
    float3 tint2 = mix(float3(1.15, 0.98, 0.78), float3(0.80, 0.98, 1.22), phase2);

    return (tint1 * g1 * 3.1 + tint2 * g2 * 4.4) * farFade * sunAlign;
}

fragment float4 terrainFragment(
    TerrainVertexOut in [[stage_in]],
    constant FrameUniforms &frame [[buffer(BufferIndexFrameUniforms)]],
    constant TerrainLevelUniforms &level [[buffer(BufferIndexSubsystemUniforms)]],
    texture2d<float> uCoreDem [[texture(TextureIndexCoreDem)]],
    texture2d<float> uCoreAux1 [[texture(TextureIndexCoreAux1)]],
    texture2d<float> uCoreAux2 [[texture(TextureIndexCoreAux2)]],
    texture2d<float> uOuterDem [[texture(TextureIndexOuterDem)]],
    texture2d<float> uOuterAux1 [[texture(TextureIndexOuterAux1)]],
    texture2d<float> uOuterAux2 [[texture(TextureIndexOuterAux2)]],
    texture2d<float> uGraniteAlbedo [[texture(TextureIndexGraniteAlbedo)]],
    texture2d<float> uGraniteNormal [[texture(TextureIndexGraniteNormal)]],
    texture2d<float> uSnowNormal [[texture(TextureIndexSnowNormal)]],
    texture2d<float> uSnowCavity [[texture(TextureIndexSnowCavity)]],
    texture2d<float> uIceNormal [[texture(TextureIndexIceNormal)]],
    texture2d<float> uTrackTex [[texture(TextureIndexTrackTex)]],
    texture2d<float> uMacroTrackTex [[texture(TextureIndexMacroTrackTex)]]
) {
    float2 xz = in.worldPos.xz;
    if (level.isOuterMesh < 0.5 && (abs(xz.x) > CORE_HALF - 38.0 || abs(xz.y) > CORE_HALF - 38.0)) {
        discard_fragment();
    }
    if (level.childHalf > 0.0) {
        float2 dc = abs(xz - level.childCenter);
        if (dc.x < level.childHalf && dc.y < level.childHalf) {
            discard_fragment();
        }
    }
    bool inCore = (level.isOuterMesh < 0.5) && (abs(xz.x) < CORE_HALF - 30.0) && (abs(xz.y) < CORE_HALF - 30.0);

    float3 uCamPos = frame.cameraPosition.xyz;
    float uTime = frame.cameraPosition.w;
    float3 uSunDir = frame.sunDirection.xyz;
    float uSunElev = frame.sunDirection.w;
    float3 uSunColor = frame.sunColor.rgb;
    float uSunIntensity = frame.sunColor.w;
    float uStorm = frame.skyHorizon.w;
    float3 uShadowSkyColor = frame.shadowSkyColor.rgb;
    float uShadowFillStrength = frame.shadowSkyColor.w;
    float3 uGroundBounceColor = frame.groundBounceColor.rgb;
    float uTwilight = frame.groundBounceColor.w;
    float3 uSSSColor = frame.sssColor.rgb;
    float uSparkleIntensity = frame.sssColor.w;
    float3 uFogColor = frame.fogColor.rgb;
    float uFogDensity = frame.fogColor.w;
    float uValleyCloudAlt = frame.cloudParams.x;
    float uValleyCloudDensity = frame.cloudParams.y;
    float uCloudShadowStrength = frame.cloudParams.z;
    float uCoreSize = frame.cloudParams.w;
    float4 uHorizonBlend = frame.horizonBlend;
    float uOuterSize = frame.screenParams.w;

    float uAvyActive = frame.avalancheParams.x;
    float2 uAvyOrigin = frame.avalancheParams.yz;
    float uAvyFront = frame.avalancheParams.w;
    float2 uAvyDir = frame.avalancheDirWidth.xy;
    float uAvyWidth = frame.avalancheDirWidth.z;

    float2 uTrackOrigin = frame.trackParams.xy;
    float uTrackSpan = frame.trackParams.z;
    float uTrackRes = frame.trackParams.w;
    float2 uMacroTrackOrigin = frame.macroTrackParams.xy;
    float uMacroTrackSpan = frame.macroTrackParams.z;
    float uMacroTrackRes = frame.macroTrackParams.w;

    float hCenter = 0.0, ao = 0.0, powder = 0.0, curv = 0.0, hShadowAng = 0.0;
    float3 baseN;

    if (inCore) {
        float2 uv = clamp((xz + CORE_HALF) / (2.0 * CORE_HALF), 0.0, 1.0);
        float3 aux1 = uCoreAux1.sample(auxSampler, uv).rgb;
        float3 aux2 = uCoreAux2.sample(auxSampler, uv).rgb;
        powder = aux1.r;
        curv = aux2.b;
        float4 horAngles = float4(aux1.g, aux1.b, aux2.r, aux2.g) * (3.14159265 * 0.5);
        hShadowAng = dot(horAngles, uHorizonBlend);

        float epsRaw = mix(6.0, 16.0, clamp(in.dist / 2500.0, 0.0, 1.0));
        float3 c0 = samplePackedDEM(uCoreDem, uCoreSize, xz, CORE_HALF);
        float hR = samplePackedDEM(uCoreDem, uCoreSize, xz + float2(epsRaw, 0.0), CORE_HALF).x;
        float hL = samplePackedDEM(uCoreDem, uCoreSize, xz - float2(epsRaw, 0.0), CORE_HALF).x;
        float hS = samplePackedDEM(uCoreDem, uCoreSize, xz + float2(0.0, epsRaw), CORE_HALF).x;
        float hN = samplePackedDEM(uCoreDem, uCoreSize, xz - float2(0.0, epsRaw), CORE_HALF).x;
        float2 rawGrad = float2(hR - hL, hS - hN) / (2.0 * epsRaw);
        float rawSlope = length(rawGrad);

        float microW = 1.0 - smoothstep(260.0, 1800.0, in.dist);
        float2 mGrad = float2(0.0);
        if (microW > 0.01) {
            float me = 0.75;
            float mR = microRelief(xz + float2(me, 0.0), rawSlope, powder);
            float mL = microRelief(xz - float2(me, 0.0), rawSlope, powder);
            float mS = microRelief(xz + float2(0.0, me), rawSlope, powder);
            float mN = microRelief(xz - float2(0.0, me), rawSlope, powder);
            mGrad = float2(mR - mL, mS - mN) / (2.0 * me) * microW;
        }
        float2 grad = rawGrad + mGrad;
        baseN = normalize(float3(-grad.x, 1.0, -grad.y));
        hCenter = c0.x;
        ao = c0.y;
    } else {
        float2 uv = clamp((xz + OUTER_HALF) / (2.0 * OUTER_HALF), 0.0, 1.0);
        float3 aux1 = uOuterAux1.sample(auxSampler, uv).rgb;
        float3 aux2 = uOuterAux2.sample(auxSampler, uv).rgb;
        powder = aux1.r;
        curv = aux2.b;
        float4 horAngles = float4(aux1.g, aux1.b, aux2.r, aux2.g) * (3.14159265 * 0.5);
        hShadowAng = dot(horAngles, uHorizonBlend);

        float eps = 72.0;
        float3 c0 = samplePackedDEM(uOuterDem, uOuterSize, xz, OUTER_HALF);
        float hR = samplePackedDEM(uOuterDem, uOuterSize, xz + float2(eps, 0.0), OUTER_HALF).x;
        float hL = samplePackedDEM(uOuterDem, uOuterSize, xz - float2(eps, 0.0), OUTER_HALF).x;
        float hS = samplePackedDEM(uOuterDem, uOuterSize, xz + float2(0.0, eps), OUTER_HALF).x;
        float hN = samplePackedDEM(uOuterDem, uOuterSize, xz - float2(0.0, eps), OUTER_HALF).x;
        float2 grad = float2(hR - hL, hS - hN) / (2.0 * eps);
        baseN = normalize(float3(-grad.x, 1.0, -grad.y));
        hCenter = c0.x;
        ao = c0.y;
    }

    float3 sunDir = normalize(uSunDir);
    float3 viewDir = normalize(uCamPos - in.worldPos);
    float slopeRad = acos(clamp(baseN.y, 0.0, 1.0));
    float slopeDeg = slopeRad * (180.0 / 3.14159265);

    // Dual-cascade persistent snow tracks
    float2 tUv = (xz - uTrackOrigin) / uTrackSpan + 0.5;
    float trackDepth = 0.0;
    float trackBerm = 0.0;
    float trackIce = 0.0;
    float trackSunOcc = 1.0;
    float2 trackGrad = float2(0.0);
    float nearW = 0.0;
    if (inCore && tUv.x > 0.01 && tUv.x < 0.99 && tUv.y > 0.01 && tUv.y < 0.99) {
        nearW = smoothstep(0.01, 0.11, min(min(tUv.x, 1.0 - tUv.x), min(tUv.y, 1.0 - tUv.y)));
        float4 tr = uTrackTex.sample(trackSampler, tUv);
        trackDepth = tr.r * nearW;
        trackBerm = tr.g * nearW;
        trackIce = tr.b * nearW;
        if (tr.a > 0.004) {
            float du = 1.15 / max(512.0, uTrackRes);
            float4 tx1 = uTrackTex.sample(trackSampler, tUv + float2(du, 0.0));
            float4 tx0 = uTrackTex.sample(trackSampler, tUv - float2(du, 0.0));
            float4 tz1 = uTrackTex.sample(trackSampler, tUv + float2(0.0, du));
            float4 tz0 = uTrackTex.sample(trackSampler, tUv - float2(0.0, du));
            float hX1 = tx1.g * 0.16 - tx1.r * 0.44;
            float hX0 = tx0.g * 0.16 - tx0.r * 0.44;
            float hZ1 = tz1.g * 0.16 - tz1.r * 0.44;
            float hZ0 = tz0.g * 0.16 - tz0.r * 0.44;
            float stepM = 2.0 * du * uTrackSpan;
            trackGrad = float2(hX1 - hX0, hZ1 - hZ0) / stepM * 2.2 * nearW;

            float2 sStep = normalize(sunDir.xz + float2(1e-4)) * (0.28 / uTrackSpan);
            float4 tSun = uTrackTex.sample(trackSampler, clamp(tUv + sStep, 0.005, 0.995));
            float rimH = (tSun.g * 0.22 - tSun.r * 0.28) - (tr.g * 0.22 - tr.r * 0.34);
            trackSunOcc = 1.0 - smoothstep(0.02, 0.24, rimH) * 0.65 * nearW;
        }
    }
    if (inCore && nearW < 0.99) {
        float2 mUv = (xz - uMacroTrackOrigin) / uMacroTrackSpan + 0.5;
        if (mUv.x > 0.01 && mUv.x < 0.99 && mUv.y > 0.01 && mUv.y < 0.99) {
            float mEdge = smoothstep(0.01, 0.09, min(min(mUv.x, 1.0 - mUv.x), min(mUv.y, 1.0 - mUv.y)));
            float mw = (1.0 - nearW) * mEdge;
            float4 mr = uMacroTrackTex.sample(trackSampler, mUv);
            if (mr.a > 0.004) {
                trackDepth += mr.r * mw * 0.92;
                trackBerm += mr.g * mw * 0.85;
                trackIce += mr.b * mw * 0.78;
                float mdu = 1.15 / max(512.0, uMacroTrackRes);
                float4 mx1 = uMacroTrackTex.sample(trackSampler, mUv + float2(mdu, 0.0));
                float4 mx0 = uMacroTrackTex.sample(trackSampler, mUv - float2(mdu, 0.0));
                float4 mz1 = uMacroTrackTex.sample(trackSampler, mUv + float2(0.0, mdu));
                float4 mz0 = uMacroTrackTex.sample(trackSampler, mUv - float2(0.0, mdu));
                float stepMM = 2.0 * mdu * uMacroTrackSpan;
                float2 mGrad = float2(
                    (mx1.g * 0.14 - mx1.r * 0.42) - (mx0.g * 0.14 - mx0.r * 0.42),
                    (mz1.g * 0.14 - mz1.r * 0.42) - (mz0.g * 0.14 - mz0.r * 0.42)
                ) / stepMM * 2.8;
                trackGrad += mGrad * mw;
            }
        }
    }

    // Dynamic Slab Avalanche Crown Fracture Line & Scraped Bed Surface
    float avyCrown = 0.0;
    float avyCrownRim = 0.0;
    float avyBed = 0.0;
    float avyDebris = 0.0;
    if (uAvyActive > 0.01 && inCore) {
        float2 relA = xz - uAvyOrigin;
        float along = dot(relA, uAvyDir);
        float2 sideVec = float2(-uAvyDir.y, uAvyDir.x);
        float across = dot(relA, sideVec);
        float zig = (vnoise2(xz * 0.095) - 0.5) * 7.5 + sin(across * 0.24) * 3.4 + (vnoise2(xz * 0.26) - 0.5) * 2.2;
        float wMask = 1.0 - smoothstep(uAvyWidth * 0.62, uAvyWidth * 1.08, abs(across));
        if (wMask > 0.01) {
            float dCrown = along - zig;
            avyCrown = smoothstep(-0.3, 0.6, dCrown) * (1.0 - smoothstep(1.2, 4.2, dCrown)) * wMask * uAvyActive;
            avyCrownRim = smoothstep(-2.4, -0.3, dCrown) * (1.0 - smoothstep(-0.1, 0.8, dCrown)) * wMask * uAvyActive;
            avyBed = smoothstep(0.6, 4.0, dCrown) * (1.0 - smoothstep(uAvyFront - 8.0, uAvyFront + 2.0, along)) * wMask * uAvyActive;
            float chunk = smoothstep(0.54, 0.78, vnoise2(float2(across * 0.42, (along - uAvyFront * 0.4) * 0.35)));
            avyDebris = chunk * avyBed;
        }
    }

    // Dual-frequency wind-sculpted sastrugi snow normals
    float detailFade = 1.0 - smoothstep(140.0, 1800.0, in.dist);
    float closeFade = 1.0 - smoothstep(18.0, 220.0, in.dist);
    float2 windUV1 = float2(xz.x * 0.042 + xz.y * 0.015, 1.0 - (-xz.x * 0.015 + xz.y * 0.042));
    float2 windUV2 = float2(xz.x * 0.22 + xz.y * 0.09, 1.0 - (-xz.x * 0.09 + xz.y * 0.22));
    float2 windUV3 = float2(xz.x * 0.95 - xz.y * 0.32, 1.0 - (xz.x * 0.32 + xz.y * 0.95));
    float3 sN1 = uSnowNormal.sample(detailSampler, windUV1).xyz * 2.0 - 1.0;
    float3 sN2 = uSnowNormal.sample(detailSampler, windUV2).xyz * 2.0 - 1.0;
    float3 sN3 = uSnowNormal.sample(detailSampler, windUV3).xyz * 2.0 - 1.0;
    float3 sCav1 = uSnowCavity.sample(detailSampler, windUV1).rgb;
    float3 sCav2 = uSnowCavity.sample(detailSampler, windUV2).rgb;
    float microCavityAO = mix(1.0, clamp(sCav1.r * 0.58 + sCav2.r * 0.42, 0.36, 1.08), detailFade * 0.85);
    float microGrain = (sCav2.b - 0.5) * 0.065 * closeFade;
    float3 iN = uIceNormal.sample(detailSampler, float2(xz.x * 0.068, 1.0 - xz.y * 0.068)).xyz * 2.0 - 1.0;

    float driftMacro = vnoise2(xz * 0.0085 + float2(1.7, -3.2));
    float driftMid = vnoise2(float2(xz.x * 0.032 + xz.y * 0.014, -xz.x * 0.014 + xz.y * 0.032));
    float driftCavity = smoothstep(0.22, 0.78, driftMacro * 0.55 + driftMid * 0.45);
    float dR = vnoise2((xz + float2(1.8, 0.0)) * 0.042);
    float dL = vnoise2((xz - float2(1.8, 0.0)) * 0.042);
    float dS = vnoise2((xz + float2(0.0, 1.8)) * 0.042);
    float dN = vnoise2((xz - float2(0.0, 1.8)) * 0.042);
    float2 driftGrad = float2(dR - dL, dS - dN) * 0.92 * detailFade;

    float3 granAlb;
    float3 granN;
    sampleTriplanarGranite(in.worldPos, baseN, uGraniteAlbedo, uGraniteNormal, granAlb, granN);

    // Multi-octave Protogine granite cleavage strata & finite-difference normal perturbation (zero single-octave contour rings!)
    float2 strataCoord = float2(
        xz.x * 0.011 + xz.y * 0.0068 + in.worldPos.y * 0.0085,
        xz.y * 0.011 - xz.x * 0.0068 - in.worldPos.y * 0.0055
    );
    float strataWarp = fbm3(xz * 0.0065 + float2(in.worldPos.y * 0.003, -in.worldPos.y * 0.002));
    float2 stW = strataCoord + float2(strataWarp * 0.95, -strataWarp * 0.72);
    float strataBand = fbm3(float2(stW.x * 1.4, stW.y * 3.8));
    float strataR = fbm3(float2((stW.x + 0.08) * 1.4, stW.y * 3.8));
    float strataU = fbm3(float2(stW.x * 1.4, (stW.y + 0.08) * 3.8));
    float fineCrack = fbm3(xz * 0.036 + float2(in.worldPos.y * 0.014, -in.worldPos.y * 0.011));
    float ridgeCrack = clamp(strataBand * 0.58 + fineCrack * 0.42, 0.0, 1.0);
    float farSharpen = mix(0.45, 0.85, smoothstep(220.0, 2600.0, in.dist));
    float dSX = (strataR - strataBand) * 3.2 * farSharpen;
    float dSY = (strataU - strataBand) * 3.2 * farSharpen;
    float3 ridgeFissureN = normalize(granN + float3(dSX, dSY * 0.45, dSY * 0.85));

    float highAlt = smoothstep(2350.0, 3250.0, hCenter);
    float strataJitter = (granAlb.r - 0.46) * 12.0 + (curv - 0.5) * 18.0 + (ridgeCrack - 0.5) * 14.0;
    float rockMask = smoothstep(32.0 - highAlt * 4.2, 43.5 - highAlt * 3.5, slopeDeg + strataJitter);
    float ledgeDust = smoothstep(0.60, 0.85, ridgeFissureN.y) * smoothstep(1850.0, 2700.0, hCenter);
    float couloirSnow = (1.0 - smoothstep(0.34, 0.62, curv)) * (1.0 - smoothstep(31.0, 46.0, slopeDeg)) * 0.52;
    float couloirVein = pow(strataBand, 2.8) * smoothstep(2450.0, 3300.0, hCenter) * (1.0 - smoothstep(52.0, 72.0, slopeDeg)) * 0.42;
    rockMask *= (1.0 - clamp(ledgeDust * 0.62 + couloirSnow + couloirVein, 0.0, 0.88));

    float lowValley = (1.0 - smoothstep(1060.0, 1620.0, hCenter)) * (1.0 - rockMask);
    float forestFloor = (1.0 - smoothstep(1450.0, 2060.0, hCenter)) * smoothstep(26.0, 40.0, slopeDeg + (vnoise2(xz * 0.008) - 0.5) * 12.0);

    // Glacial crevasse icefalls with 3D parallax chasm depth
    float iceNoise = vnoise2(xz * 0.0095 + float2(3.1, -5.7));
    float crevPatch = smoothstep(0.62, 0.80, vnoise2(xz * 0.0038 + float2(4.2, -7.1)));
    float crevBreak = smoothstep(0.42, 0.72, vnoise2(xz * 0.019 - float2(1.5, 6.2)));
    float crevPhase = xz.x * 0.022 + xz.y * 0.017 + iceNoise * 5.2;
    float crevSin = sin(crevPhase);
    float crevWave = abs(crevSin);
    float crevGate = crevPatch * crevBreak
                   * smoothstep(16.0, 29.0, slopeDeg) * (1.0 - smoothstep(34.0, 42.0, slopeDeg))
                   * smoothstep(2450.0, 2950.0, hCenter) * (1.0 - smoothstep(3450.0, 3750.0, hCenter))
                   * (1.0 - rockMask);
    float crevasseMask = (1.0 - smoothstep(0.012, 0.095, crevWave)) * crevGate;
    float2 crevPar = -viewDir.xz / max(0.38, viewDir.y) * 1.15;
    float innerNoise = vnoise2((xz + crevPar) * 0.0095 + float2(3.1, -5.7));
    float innerWave = abs(sin((xz.x + crevPar.x) * 0.022 + (xz.y + crevPar.y) * 0.017 + innerNoise * 5.2));
    float crevAbyss = (1.0 - smoothstep(0.006, 0.044, innerWave)) * crevasseMask;
    float crevLip = smoothstep(0.055, 0.092, crevWave) * (1.0 - smoothstep(0.092, 0.145, crevWave)) * crevGate;
    float2 crevDir = normalize(float2(0.022, 0.017));
    float2 crevGrad = crevDir * sign(crevSin) * (crevLip * 1.45 - crevasseMask * 1.95);

    float iceMask = smoothstep(27.0, 39.0, slopeDeg) * smoothstep(0.51, 0.70, curv + (iceNoise - 0.5) * 0.28)
                  * smoothstep(2150.0, 2650.0, hCenter) * (1.0 - rockMask);
    float snowMask = 1.0 - rockMask;

    float2 crownGrad = uAvyDir * (-avyCrown * 2.2 + avyDebris * 0.65);
    float3 snowN = normalize(
        baseN
        + float3(-driftGrad.x, 0.0, -driftGrad.y) * (0.85 + 0.45 * powder)
        + float3(sN1.x, 0.0, sN1.y) * (0.42 * detailFade)
        + float3(sN2.x, 0.0, sN2.y) * (0.38 * closeFade * (1.0 - trackDepth * 0.55))
        + float3(sN3.x, 0.0, sN3.y) * (0.24 * (1.0 - smoothstep(8.0, 55.0, in.dist)) * (1.0 - trackDepth * 0.72))
        + float3(iN.x, 0.0, iN.y) * (0.42 * (iceMask + crevasseMask * 0.7) * detailFade)
        + float3(-trackGrad.x, 0.0, -trackGrad.y) * snowMask * 1.42
        + float3(crownGrad.x + crevGrad.x, 0.0, crownGrad.y + crevGrad.y) * snowMask
    );

    float3 N = normalize(mix(snowN, ridgeFissureN, rockMask));

    // Surface Albedos
    float3 snowAlb = mix(float3(0.78, 0.83, 0.89), float3(0.87, 0.90, 0.94), powder);
    snowAlb *= mix(float3(0.76, 0.84, 0.94), float3(1.02, 1.02, 1.02), driftCavity * 0.55 + microCavityAO * 0.45);
    snowAlb += float3(microGrain);
    float3 iceAlb = float3(0.28, 0.58, 0.82);
    snowAlb = mix(snowAlb, iceAlb, iceMask * 0.78);
    snowAlb = mix(snowAlb, float3(0.95, 0.97, 1.00), crevLip * 0.55);
    float3 crevWallCol = mix(float3(0.06, 0.32, 0.54), float3(0.015, 0.055, 0.16), crevAbyss * 0.92);
    snowAlb = mix(snowAlb, crevWallCol, crevasseMask * 0.94);
    snowAlb = mix(snowAlb, float3(0.46, 0.63, 0.82), avyBed * 0.56);
    snowAlb = mix(snowAlb, float3(0.92, 0.95, 0.99), avyDebris * 0.62 + avyCrownRim * 0.45);
    snowAlb = mix(snowAlb, float3(0.07, 0.20, 0.44), avyCrown * 0.94);
    snowAlb = mix(snowAlb, float3(0.64, 0.76, 0.90), trackDepth * 0.34);
    snowAlb = mix(snowAlb, float3(0.96, 0.98, 1.00), trackBerm * 0.44);

    float3 valleyAlb = mix(float3(0.14, 0.19, 0.14), float3(0.62, 0.68, 0.74), smoothstep(0.35, 0.72, vnoise2(xz * 0.0045)));
    float townSpot = smoothstep(1120.0, 1020.0, hCenter) * smoothstep(14.0, 3.0, slopeDeg);
    float roofs = step(0.62, vnoise2(xz * 0.032)) * townSpot;
    valleyAlb = mix(valleyAlb, float3(0.22, 0.20, 0.19), roofs * 0.72);
    snowAlb = mix(snowAlb, valleyAlb, clamp(lowValley * 0.58 + forestFloor * 0.36, 0.0, 0.82));

    float farRockCrisp = fbm3(xz * 0.042 + float2(in.worldPos.y * 0.022, -in.worldPos.y * 0.016));
    float creviceAO = smoothstep(0.22, 0.78, ridgeCrack) * (0.75 + 0.40 * farRockCrisp);
    float3 rockCol = granAlb * float3(0.42, 0.42, 0.46) * mix(float3(0.36, 0.40, 0.48), float3(1.26, 1.14, 1.02), clamp(creviceAO, 0.0, 1.15));
    float frostRim = smoothstep(0.56, 0.84, ridgeFissureN.y) * smoothstep(0.45, 0.82, farRockCrisp) * rockMask * 0.34;
    rockCol = mix(rockCol, float3(0.86, 0.90, 0.96), frostRim);
    float3 albedo = mix(snowAlb, rockCol, rockMask);

    // Lighting Hierarchy
    float horDiff = uSunElev - hShadowAng;
    float mtnShadow = smoothstep(-0.020, 0.044, horDiff);
    mtnShadow = mix(mtnShadow, 0.32, uStorm * 0.85);

    float csh = cloudShadowAt(xz, uTime) * uCloudShadowStrength * (1.0 - uStorm * 0.8);
    float directVis = mtnShadow * (1.0 - csh);

    float trackAO = (1.0 - trackDepth * 0.26) * (1.0 - crevasseMask * 0.78) * (1.0 - avyCrown * 0.65);

    float NdotL = max(0.0, dot(N, sunDir));
    float wrapNdotL = max(0.0, (dot(N, sunDir) + 0.06) / 1.06);
    float diffFactor = mix(wrapNdotL * 0.14 + NdotL * 0.86, NdotL, rockMask);
    float3 directLight = uSunColor * uSunIntensity * diffFactor * directVis * trackAO * trackSunOcc;

    float skyFactor = clamp(0.42 + 0.58 * N.y, 0.16, 1.0);
    float shadowMicro = clamp(0.74 + 0.34 * dot(N, normalize(float3(-sunDir.x, 0.72, -sunDir.z))), 0.52, 1.24);
    float cavityAO = mix(0.42, 1.0, pow(ao, 1.05)) * trackAO * mix(0.76, 1.08, driftCavity * 0.45 + microCavityAO * 0.55);
    float3 skyFill = mix(uGroundBounceColor * 0.64, uShadowSkyColor, skyFactor) * uShadowFillStrength * cavityAO * shadowMicro;

    float backLight = pow(clamp(dot(viewDir, -sunDir + N * 0.42), 0.0, 1.0), 3.0);
    float rimFres = pow(1.0 - max(0.0, dot(N, viewDir)), 3.2);
    float sssMask = snowMask * (0.28 + 0.72 * powder) * (0.35 + 0.65 * curv + trackDepth * 1.45 + crevasseMask * 1.8);
    float3 sss = uSSSColor * (backLight * 0.42 + rimFres * 0.16 + trackDepth * 0.36 + crevasseMask * 0.45)
               * sssMask * (0.22 + 0.78 * directVis) * (1.0 - uStorm * 0.7);

    // Dual-lobe photoreal snow gleam
    float3 H = normalize(sunDir + viewDir);
    float NdotH = max(0.0, dot(N, H));
    float NdotV = max(0.0, dot(N, viewDir));
    float fresnel = pow(1.0 - NdotV, 3.6);
    float crustNoise = vnoise2(xz * 0.013 + float2(2.4, -6.1)) * 0.62 + vnoise2(xz * 0.048 - float2(5.1, 3.7)) * 0.38;
    float crustPatch = smoothstep(0.34, 0.78, crustNoise + (curv - 0.5) * 0.45) * (1.0 - powder * 0.42);
    float glazeWeight = clamp(0.24 + crustPatch * 0.58 + iceMask * 0.85 + trackIce * 0.72 + trackDepth * 0.45, 0.0, 1.5);
    float fwdCone = pow(max(0.0, dot(-viewDir, normalize(float3(sunDir.x, -sunDir.y * 0.25, sunDir.z)))), 3.0);
    float broadGlaze = pow(NdotH, mix(18.0, 10.0, crustPatch * 0.5)) * (0.26 + 0.56 * crustPatch + 0.35 * fwdCone) * (0.35 + 0.65 * fresnel);
    float tightPow = mix(46.0, 132.0, clamp(iceMask + trackIce * 0.75 + trackDepth * 0.35, 0.0, 1.0));
    float tightGleam = pow(NdotH, tightPow) * (0.28 + 0.72 * crustPatch + 1.15 * iceMask + 1.05 * trackIce + 0.65 * trackBerm);
    float sheenSpec = fresnel * (0.16 + 0.36 * fwdCone) * (0.28 + 0.72 * NdotL);
    float rockSpec = pow(NdotH, 32.0) * (0.16 + 0.18 * ridgeCrack) * rockMask;
    float3 specTint = mix(uSunColor, float3(0.96, 0.98, 1.04), 0.32);
    float3 specular = specTint * ((broadGlaze * 0.72 + tightGleam * 0.85 + sheenSpec) * snowMask * (0.65 + 0.35 * glazeWeight) + rockSpec)
                    * directVis * uSunIntensity * 0.52 * (1.0 - uStorm * 0.82);

    float3 spk = snowSparkle(in.worldPos, N, viewDir, sunDir, in.dist) * snowMask * (1.0 - lowValley * 0.8);
    float3 sparkleCol = spk * (uSunColor * 0.85 + float3(0.28, 0.34, 0.44)) * uSparkleIntensity * directVis * (0.32 + 0.68 * NdotL) * (0.78 + 0.45 * crustPatch) * (1.0 - uStorm * 0.90);

    float3 col = albedo * (directLight + skyFill) + sss + specular + sparkleCol;

    // Aerial perspective & Valley sea-of-clouds height mist
    float distFog = 1.0 - exp(-in.dist * uFogDensity * (1.0 + 0.25 * max(0.0, (2500.0 - hCenter) / 1500.0)));
    distFog = clamp(distFog, 0.0, 0.94);

    float sunScatter = pow(max(0.0, dot(-viewDir, sunDir)), 4.0) * (1.0 - uStorm * 0.9);
    float3 aerialCol = mix(uFogColor, mix(uFogColor, uSunColor * 1.08, 0.52), sunScatter);
    col = mix(col, aerialCol, distFog);

    float valleyFogAmt = 0.0;
    if (hCenter < uValleyCloudAlt + 260.0) {
        float hBand = smoothstep(uValleyCloudAlt + 260.0, uValleyCloudAlt - 180.0, hCenter);
        float mistNoise = 0.65 + 0.35 * vnoise2(xz * 0.0008 + float2(uTime * 0.006, 0.0));
        float mistScale = mix(1.0, 0.56, uTwilight);
        valleyFogAmt = clamp(hBand * uValleyCloudDensity * mistNoise * mistScale * smoothstep(140.0, 900.0, in.dist), 0.0, 0.90);
        float3 cloudMistCol = mix(uShadowSkyColor * 1.15, float3(0.94, 0.96, 0.99) * (0.55 + 0.45 * uSunColor), 0.62);
        col = mix(col, cloudMistCol, valleyFogAmt);
    }

    // Glowing Chamonix valley town lights in Alpenglow / Twilight
    if (uTwilight > 0.02 && hCenter < 1340.0 && slopeDeg < 21.0) {
        float elevMask = smoothstep(1340.0, 1110.0, hCenter);
        float flatMask = smoothstep(20.5, 4.5, slopeDeg);
        float2 dCham = xz - float2(-2759.0, -5001.0);
        float2 dPraz = xz - float2(-1800.0, -5850.0);
        float2 dBoss = xz - float2(-3880.0, -3650.0);
        float2 dHouch = xz - float2(-5180.0, -2520.0);
        float wCham = exp(-dot(dCham, dCham) / 9.0e5);
        float wPraz = exp(-dot(dPraz, dPraz) / 6.5e5) * 0.78;
        float wBoss = exp(-dot(dBoss, dBoss) / 7.5e5) * 0.85;
        float wHouch = exp(-dot(dHouch, dHouch) / 9.5e5) * 0.82;
        float valleySpine = exp(-pow(abs((xz.x + 2759.0) * 0.70 + (xz.y + 5001.0) * 0.71) / 340.0, 2.0));
        float parcelMask = smoothstep(0.26, 0.74, vnoise2(xz * 0.0028 + float2(4.1, -2.7)) * 0.65 + vnoise2(xz * 0.0072) * 0.35);
        float coreBoost = clamp(wCham * 1.35 + wPraz * 0.92 + wBoss * 0.95 + wHouch * 0.92, 0.0, 1.5);
        float popDensity = clamp((0.08 + coreBoost) * (0.35 + 0.85 * parcelMask) + valleySpine * 0.28, 0.0, 1.45);
        float townBase = elevMask * flatMask * smoothstep(240.0, 820.0, in.dist) * (1.0 - uStorm * 0.82);

        if (townBase > 0.005) {
            float distExpand = clamp(1.0 + in.dist * 0.00011, 1.0, 1.75);

            float2 c1 = xz / 44.0;
            float2 id1 = floor(c1);
            float2 gv1 = fract(c1) - 0.5;
            float2 mid1 = glslMod(id1, 512.0);
            float r1 = hash21(mid1 * float2(1.13, 1.79) + float2(3.1, 7.4));
            float2 jit1 = (float2(hash21(mid1 + 11.7), hash21(mid1 + 29.3)) - 0.5) * 0.68;
            float d1 = length(gv1 - jit1);
            float on1 = step(clamp(0.86 - popDensity * 0.40, 0.38, 0.92), r1);
            float rad1 = 0.088 * distExpand;
            float twk1 = 0.80 + 0.20 * sin(uTime * 2.1 + r1 * 62.8);
            float dot1 = exp(-(d1 * d1) / max(0.001, rad1 * rad1)) * on1 * twk1;
            float3 col1 = (r1 > 0.965) ? float3(1.00, 0.82, 0.42)
                        : ((r1 > 0.80) ? float3(1.00, 0.46, 0.045) : float3(1.00, 0.28, 0.018));

            float2 c2 = xz / 92.0;
            float2 id2 = floor(c2);
            float2 gv2 = fract(c2) - 0.5;
            float2 mid2 = glslMod(id2, 512.0);
            float r2 = hash21(mid2 * float2(2.31, 0.91) + float2(19.2, 5.8));
            float2 jit2 = (float2(hash21(mid2 + 41.2), hash21(mid2 + 73.9)) - 0.5) * 0.54;
            float d2 = length(gv2 - jit2);
            float on2 = step(clamp(0.84 - popDensity * 0.38, 0.42, 0.90), r2);
            float rad2 = 0.112 * distExpand;
            float twk2 = 0.88 + 0.12 * sin(uTime * 1.45 + r2 * 41.0);
            float dot2 = exp(-(d2 * d2) / max(0.001, rad2 * rad2)) * on2 * twk2;
            float3 col2 = mix(float3(1.00, 0.30, 0.02), float3(1.00, 0.54, 0.06), fract(r2 * 13.7));

            float roadWave = sin(xz.x * 0.0018 + xz.y * 0.0015 + vnoise2(xz * 0.0016) * 2.4);
            float roadLine = smoothstep(0.095, 0.01, abs(roadWave)) * smoothstep(0.32, 0.78, valleySpine);
            float lampPulse = 0.35 + 0.65 * pow(max(0.0, sin((xz.x - xz.y) * 0.065)), 6.0);

            float clusterGlow = smoothstep(0.24, 0.90, vnoise2(xz * 0.0042)) * coreBoost;
            float3 ambientPoll = float3(1.00, 0.34, 0.03) * (clusterGlow * 0.30 + roadLine * 0.24) * (0.48 + 0.52 * wCham);

            float3 lights = col1 * dot1 * 2.15
                          + col2 * dot2 * 2.65
                          + float3(1.00, 0.36, 0.025) * roadLine * lampPulse * 1.15
                          + ambientPoll;
            float duskDeep = clamp(townBase * uTwilight * 0.46, 0.0, 0.50);
            col = mix(col, col * float3(0.50, 0.55, 0.72), duskDeep);
            float airTrans = exp(-in.dist * uFogDensity * 0.32);
            float mistTrans = mix(1.0, 0.48, valleyFogAmt);
            col += lights * townBase * uTwilight * airTrans * mistTrans;
        }
    }

    // Alpha = 0.55 marks solid world terrain for PostOptics worldMask & god-ray occlusion
    return float4(col, 0.55);
}
