#include "ShaderTypes.h"

using namespace metal;

// MARK: - Samplers & Helpers

constexpr sampler samNearestClamp(coord::normalized, address::clamp_to_edge, filter::nearest);
constexpr sampler samLinearClamp(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler samRepeatAniso(coord::normalized, address::repeat, filter::linear, mip_filter::linear, max_anisotropy(8));

struct PropVertexIn {
    float4 position [[attribute(0)]];
    float4 normal   [[attribute(1)]];
    float4 prop     [[attribute(2)]];
};

struct ShadowQuadVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal   [[attribute(VertexAttributeNormal)]];
    float2 uv       [[attribute(VertexAttributeUV)]];
    float4 color    [[attribute(VertexAttributeColor)]];
};

struct StationVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal   [[attribute(VertexAttributeNormal)]];
    float2 uv       [[attribute(VertexAttributeUV)]];
    float4 color    [[attribute(VertexAttributeColor)]];
};

struct CableVertexIn {
    float4 position [[attribute(0)]];
};

struct PropUniforms {
    float mode; // 0 = granite, 1 = serac, 2 = stake snow mound, 3 = wind-lip kicker
    float pad0;
    float pad1;
    float pad2;
};

static inline float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

static inline float vnoise2(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0f - 2.0f * f);
    return mix(
        mix(hash21(i), hash21(i + float2(1.0f, 0.0f)), u.x),
        mix(hash21(i + float2(0.0f, 1.0f)), hash21(i + float2(1.0f, 1.0f)), u.x),
        u.y
    );
}

static inline float hash31(float3 p) {
    p = fract(p * float3(0.1031f, 0.1030f, 0.0973f));
    p += dot(p, p.yxz + 33.33f);
    return fract((p.x + p.y) * p.z);
}

static inline float vnoise3(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    float n000 = hash31(i);
    float n100 = hash31(i + float3(1.0f, 0.0f, 0.0f));
    float n010 = hash31(i + float3(0.0f, 1.0f, 0.0f));
    float n110 = hash31(i + float3(1.0f, 1.0f, 0.0f));
    float n001 = hash31(i + float3(0.0f, 0.0f, 1.0f));
    float n101 = hash31(i + float3(1.0f, 0.0f, 1.0f));
    float n011 = hash31(i + float3(0.0f, 1.0f, 1.0f));
    float n111 = hash31(i + float3(1.0f, 1.0f, 1.0f));
    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
        f.z
    );
}

static inline float cloudShadowAt(float2 xz, float t) {
    float2 p = (xz - float2(t * 22.0f, t * 8.5f)) * 0.00052f;
    float n = vnoise2(p) * 0.62f + vnoise2(p * 2.15f + float2(7.3f, -4.1f)) * 0.38f;
    return smoothstep(0.50f, 0.78f, n);
}

static inline float3 blendTriNormal(float3 baseN, float3 nX, float3 nY, float3 nZ, float3 w, float strength) {
    float3 wnX = float3(0.0f, nX.y, nX.x);
    float3 wnY = float3(nY.x, 0.0f, nY.y);
    float3 wnZ = float3(nZ.x, nZ.y, 0.0f);
    return normalize(baseN + (wnX * w.x + wnY * w.y + wnZ * w.z) * strength);
}

static inline float decodeRG(float4 t, float maxElev) {
    return ((t.r * 255.0f * 256.0f + t.g * 255.0f) / 65535.0f) * maxElev;
}

static inline float sampleDemH(float2 wxz, texture2d<float> coreDem, float coreSize, float coreHalf, float maxElev) {
    float2 uvf = clamp((wxz + coreHalf) / (2.0f * coreHalf), 0.0f, 1.0f) * (coreSize - 1.0f);
    float2 i0 = floor(uvf);
    float2 i1 = min(i0 + float2(1.0f), float2(coreSize - 1.0f));
    float2 f = fract(uvf);
    float invS = 1.0f / coreSize;
    float h00 = decodeRG(coreDem.sample(samNearestClamp, (i0 + 0.5f) * invS), maxElev);
    float h10 = decodeRG(coreDem.sample(samNearestClamp, (float2(i1.x, i0.y) + 0.5f) * invS), maxElev);
    float h01 = decodeRG(coreDem.sample(samNearestClamp, (float2(i0.x, i1.y) + 0.5f) * invS), maxElev);
    float h11 = decodeRG(coreDem.sample(samNearestClamp, (i1 + 0.5f) * invS), maxElev);
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

static inline float microRelief(float2 xz, float slopeMag, float powder) {
    float x = xz.x, z = xz.y;
    float w1 = sin(x * 0.047f + z * 0.029f) * cos(x * 0.023f - z * 0.051f);
    float w2 = sin(x * 0.115f - z * 0.088f + 1.3f) * cos(x * 0.072f + z * 0.131f - 0.7f);
    float w3 = sin(x * 0.240f + z * 0.195f - 2.1f) * sin(x * 0.180f - z * 0.270f + 0.5f);
    float roller = sin((x * 0.74f + z * 0.67f) * 0.082f + w1 * 1.4f) * 0.68f;
    float midSlope = clamp((slopeMag - 0.08f) / 0.45f, 0.0f, 1.0f);
    float steepRock = clamp((slopeMag - 0.78f) / 0.45f, 0.0f, 1.0f);
    float snowAmp = (0.55f + 0.45f * powder) * (0.45f + 0.55f * midSlope) * (1.0f - steepRock * 0.45f);
    float rockRib = steepRock * (abs(w2) * 1.65f + w3 * 0.55f);
    return (w1 * 1.35f + roller * 0.85f + w2 * 0.42f + w3 * 0.14f) * snowAmp + rockRib;
}

// MARK: - 1. Rocks, Seracs, Kickers, Midi Pitons, Stake Snow Mounds

struct PropVarying {
    float4 position [[position]];
    float3 worldPos;
    float3 worldNormal;
    float3 prop;
};

vertex PropVarying propVertex(
    PropVertexIn in [[stage_in]],
    uint instanceId [[instance_id]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant PropInstanceData* instances [[buffer(BufferIndexInstances)]]
) {
    PropInstanceData inst = instances[instanceId];
    float4x4 M = inst.instanceMatrix;
    float4 wp = M * float4(in.position.xyz, 1.0f);

    float3 c0 = M[0].xyz;
    float3 c1 = M[1].xyz;
    float3 c2 = M[2].xyz;
    float3 invSq = float3(
        1.0f / max(1e-5f, dot(c0, c0)),
        1.0f / max(1e-5f, dot(c1, c1)),
        1.0f / max(1e-5f, dot(c2, c2))
    );
    float3x3 nMat = float3x3(c0 * invSq.x, c1 * invSq.y, c2 * invSq.z);

    PropVarying out;
    out.worldPos = wp.xyz;
    out.worldNormal = normalize(nMat * in.normal.xyz);
    out.prop = in.prop.xyz;
    out.position = frame.viewProjectionMatrix * wp;
    return out;
}

fragment float4 propFragment(
    PropVarying in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant PropUniforms& pu [[buffer(BufferIndexSubsystemUniforms)]],
    texture2d<float> coreDem [[texture(TextureIndexCoreDem)]],
    texture2d<float> coreAux1 [[texture(TextureIndexCoreAux1)]],
    texture2d<float> coreAux2 [[texture(TextureIndexCoreAux2)]],
    texture2d<float> graniteAlbedo [[texture(TextureIndexGraniteAlbedo)]],
    texture2d<float> graniteNormal [[texture(TextureIndexGraniteNormal)]],
    texture2d<float> snowNormal [[texture(TextureIndexSnowNormal)]],
    texture2d<float> snowCavity [[texture(TextureIndexSnowCavity)]],
    texture2d<float> iceNormal [[texture(TextureIndexIceNormal)]]
) {
    float3 camPos = frame.cameraPosition.xyz;
    float uTime = frame.cameraPosition.w;
    float3 L = normalize(frame.sunDirection.xyz);
    float uSunElev = frame.sunDirection.w;
    float3 uSunColor = frame.sunColor.rgb;
    float uSunIntensity = frame.sunColor.w;
    float uStorm = frame.skyHorizon.w;
    float3 uShadowSky = frame.shadowSkyColor.rgb;
    float uShadowFill = frame.shadowSkyColor.w;
    float3 uGroundBounce = frame.groundBounceColor.rgb;
    float uTwilight = frame.groundBounceColor.w;
    float3 uSSSColor = frame.sssColor.rgb;
    float3 uFogColor = frame.fogColor.rgb;
    float uFogDensity = frame.fogColor.w;
    float uValleyCloudAlt = frame.cloudParams.x;
    float uValleyCloudDensity = frame.cloudParams.y;
    float uCloudShadowStrength = frame.cloudParams.z;
    float CORE_HALF = frame.extraParams.y;

    float3 baseN = normalize(in.worldNormal);
    float3 V = normalize(camPos - in.worldPos);
    float dist = length(in.worldPos - camPos);
    float closeFade = 1.0f - smoothstep(18.0f, 180.0f, dist);
    float expose = clamp(in.prop.x, 0.0f, 1.0f);
    float cavVertAO = clamp(in.prop.y, 0.20f, 1.05f);

    float2 xz = in.worldPos.xz;
    float2 pUv = clamp((xz + CORE_HALF) / (2.0f * CORE_HALF), 0.0f, 1.0f);
    float3 aux1 = coreAux1.sample(samLinearClamp, pUv).rgb;
    float3 aux2 = coreAux2.sample(samLinearClamp, pUv).rgb;
    float demAO = coreDem.sample(samNearestClamp, pUv).b;
    float hAng = dot(float4(aux1.g, aux1.b, aux2.r, aux2.g) * 1.5707963f, frame.horizonBlend);
    float mtnShadow = smoothstep(-0.020f, 0.044f, uSunElev - hAng);
    mtnShadow = mix(mtnShadow, 0.32f, uStorm * 0.85f);
    float csh = cloudShadowAt(xz, uTime) * uCloudShadowStrength * (1.0f - uStorm * 0.8f);
    float directVis = mtnShadow * (1.0f - csh);
    float envAO = mix(0.42f, 1.0f, pow(max(0.0f, demAO), 1.05f));

    float3 wp = fmod(fmod(in.worldPos, 64.0f) + 64.0f, 64.0f);
    float3 w = pow(abs(baseN), float3(4.0f));
    w /= max(w.x + w.y + w.z, 1e-4f);

    float3 N = baseN;
    float3 alb = float3(0.84f, 0.88f, 0.92f);
    float3 sss = float3(0.0f);
    float3 spec = float3(0.0f);
    float3 H = normalize(L + V);
    float propSnowMask = 0.0f;

    if (pu.mode < 0.5f) {
        // 0. Protogine Granite Outcrop / Aiguille du Midi Piton
        float s = 0.24f;
        float2 uvX = float2(wp.z * s + wp.y * 0.08f, wp.y * s);
        float2 uvY = wp.xz * s;
        float2 uvZ = float2(wp.x * s - wp.y * 0.08f, wp.y * s);

        float3 cX = graniteAlbedo.sample(samRepeatAniso, uvX).rgb;
        float3 cY = graniteAlbedo.sample(samRepeatAniso, uvY).rgb;
        float3 cZ = graniteAlbedo.sample(samRepeatAniso, uvZ).rgb;
        float3 cMacro = graniteAlbedo.sample(samRepeatAniso, wp.xz * 0.065f + float2(wp.y * 0.04f, -wp.y * 0.03f)).rgb;
        float3 granTex = (cX * w.x + cY * w.y + cZ * w.z) * (0.58f + 0.72f * cMacro);

        float3 nX = graniteNormal.sample(samRepeatAniso, uvX).xyz * 2.0f - 1.0f;
        float3 nY = graniteNormal.sample(samRepeatAniso, uvY).xyz * 2.0f - 1.0f;
        float3 nZ = graniteNormal.sample(samRepeatAniso, uvZ).xyz * 2.0f - 1.0f;
        float3 rockN = blendTriNormal(baseN, nX, nY, nZ, w, 0.92f * (0.45f + 0.55f * closeFade));

        float2 sUv = wp.xz * 0.26f;
        float3 sN = snowNormal.sample(samRepeatAniso, sUv).xyz * 2.0f - 1.0f;
        float sCav = snowCavity.sample(samRepeatAniso, sUv).r;
        float3 snowN = normalize(baseN + float3(sN.x, 0.0f, sN.y) * (0.28f * closeFade + 0.08f));

        float topSnow = smoothstep(0.66f, 0.88f, baseN.y + (rockN.y - baseN.y) * 0.35f + (cMacro.r - 0.5f) * 0.16f);
        float snowMask = clamp(max(topSnow, 1.0f - expose), 0.0f, 1.0f);
        float creviceAO = smoothstep(0.14f, 0.68f, granTex.r);
        float3 granAlb = granTex * float3(0.38f, 0.37f, 0.41f) * mix(float3(0.28f, 0.32f, 0.42f), float3(1.26f, 1.12f, 0.98f), creviceAO);
        float3 snowAlb = mix(float3(0.78f, 0.83f, 0.89f), float3(0.89f, 0.92f, 0.96f), sCav * 0.45f + 0.45f);
        alb = mix(granAlb, snowAlb, snowMask);
        N = normalize(mix(rockN, snowN, snowMask));

        float NdotH = max(0.0f, dot(N, H));
        float micaSpec = pow(NdotH, 34.0f) * (0.20f + 0.28f * (1.0f - creviceAO)) * (1.0f - snowMask);
        float snowSpec = pow(NdotH, 24.0f) * 0.22f * snowMask;
        spec = uSunColor * (micaSpec + snowSpec) * uSunIntensity * directVis * 0.48f * (1.0f - uStorm * 0.82f);
    } else if (pu.mode < 1.5f) {
        // 1. Glacial Serac Icefall
        float s1 = 0.22f;
        float s2 = 0.068f;
        float2 uvX = float2(wp.z * s1, wp.y * s1 * 1.15f);
        float2 uvY = wp.xz * s1;
        float2 uvZ = float2(wp.x * s1, wp.y * s1 * 1.15f);
        float3 nX = iceNormal.sample(samRepeatAniso, uvX).xyz * 2.0f - 1.0f;
        float3 nY = iceNormal.sample(samRepeatAniso, uvY).xyz * 2.0f - 1.0f;
        float3 nZ = iceNormal.sample(samRepeatAniso, uvZ).xyz * 2.0f - 1.0f;
        float3 nMacro = iceNormal.sample(samRepeatAniso, wp.xz * s2 + float2(wp.y * s2, -wp.y * s2)).xyz * 2.0f - 1.0f;

        float gX = graniteAlbedo.sample(samRepeatAniso, float2(wp.z * 0.16f, wp.y * 0.24f)).r;
        float gZ = graniteAlbedo.sample(samRepeatAniso, float2(wp.x * 0.16f, wp.y * 0.24f)).r;
        float gY = graniteAlbedo.sample(samRepeatAniso, wp.xz * 0.16f).r;
        float iceGrain = gX * w.x + gY * w.y + gZ * w.z;

        float3 iceN = normalize(blendTriNormal(baseN, nX, nY, nZ, w, 0.94f * (0.42f + 0.58f * closeFade))
                              + float3(nMacro.x, nMacro.y * 0.35f, nMacro.z) * 0.26f);

        float2 sUv = wp.xz * 0.18f + float2(wp.y * 0.07f, -wp.y * 0.05f);
        float3 sN = snowNormal.sample(samRepeatAniso, sUv).xyz * 2.0f - 1.0f;
        float sCav = snowCavity.sample(samRepeatAniso, sUv).r;
        float3 snowN = normalize(baseN + float3(sN.x, 0.0f, sN.y) * (0.30f * closeFade + 0.09f));

        float topShelf = smoothstep(0.68f, 0.88f, baseN.y + (nMacro.y * 0.12f));
        float frostRim = smoothstep(0.58f, 0.88f, iceGrain) * 0.28f;
        float snowMask = clamp(max(topShelf, 1.0f - expose), 0.0f, 1.0f);
        propSnowMask = snowMask;

        float firnNoise = vnoise2(float2(wp.y * 1.15f + in.prop.z * 7.0f, (wp.x + wp.z) * 0.18f));
        float firnBand = smoothstep(0.24f, 0.74f, firnNoise + (iceGrain - 0.5f) * 0.44f);
        float crackDark = smoothstep(0.18f, 0.58f, iceGrain);

        float3 cleftBlue  = float3(0.20f, 0.37f, 0.54f);
        float3 firnMid    = float3(0.54f, 0.67f, 0.78f);
        float3 frostedIce = float3(0.79f, 0.85f, 0.90f);
        float deepMask = clamp((1.0f - cavVertAO) * 1.55f + (1.0f - crackDark) * 0.58f, 0.0f, 1.0f);
        float3 iceAlb = mix(firnMid, frostedIce, clamp(firnBand * 0.58f + frostRim, 0.0f, 1.0f));
        iceAlb = mix(iceAlb, cleftBlue, deepMask * 0.78f);
        float3 snowAlb = mix(float3(0.78f, 0.83f, 0.89f), float3(0.89f, 0.92f, 0.96f), sCav * 0.45f + 0.45f);
        alb = mix(iceAlb, snowAlb, snowMask);
        N = normalize(mix(iceN, snowN, snowMask));

        float NdotV = max(0.0f, dot(N, V));
        float fres = pow(1.0f - NdotV, 3.6f);
        float back = pow(clamp(dot(V, -L + N * 0.44f), 0.0f, 1.0f), 3.2f);
        float sunLum = dot(uSunColor, float3(0.299f, 0.587f, 0.114f));
        float3 sssTint = mix(uSSSColor, uSunColor * float3(0.52f, 0.78f, 1.0f), 0.32f);
        sss = sssTint * (back * 0.30f + fres * 0.05f) * (1.0f - snowMask * 0.65f) * directVis * sunLum * (1.0f - uStorm * 0.85f);

        float NdotH = max(0.0f, dot(N, H));
        float iceSpec = (pow(NdotH, 54.0f) * 0.36f + pow(NdotH, 16.0f) * 0.14f) * (1.0f - snowMask * 0.65f);
        float snowSpec = pow(NdotH, 24.0f) * 0.22f * snowMask;
        spec = mix(uSunColor, float3(0.95f, 0.98f, 1.0f), 0.25f) * (iceSpec + snowSpec) * uSunIntensity * directVis * 0.44f * (1.0f - uStorm * 0.82f);
    } else {
        // 2 & 3. Stake Snow Mound / Wind-Lip Kicker
        float2 uv1 = wp.xz * 0.14f;
        float2 uv2 = wp.xz * 0.42f;
        float3 sN1 = snowNormal.sample(samRepeatAniso, uv1).xyz * 2.0f - 1.0f;
        float3 sN2 = snowNormal.sample(samRepeatAniso, uv2).xyz * 2.0f - 1.0f;
        float c1 = snowCavity.sample(samRepeatAniso, uv1).r;
        float c2 = snowCavity.sample(samRepeatAniso, uv2).r;
        float cav = c1 * 0.60f + c2 * 0.40f;
        N = normalize(baseN + float3(sN1.x, 0.0f, sN1.y) * 0.26f + float3(sN2.x, 0.0f, sN2.y) * (0.22f * closeFade));
        alb = mix(float3(0.78f, 0.83f, 0.89f), float3(0.89f, 0.92f, 0.96f), cav * 0.48f + clamp(N.y * 0.5f + 0.5f, 0.0f, 1.0f) * 0.32f + 0.20f);
        float back = pow(clamp(dot(V, -L + N * 0.38f), 0.0f, 1.0f), 3.0f);
        sss = uSSSColor * back * 0.28f * directVis * (1.0f - uStorm * 0.78f);
        float NdotH = max(0.0f, dot(N, H));
        float fres = pow(1.0f - max(0.0f, dot(N, V)), 3.2f);
        spec = uSunColor * (pow(NdotH, 28.0f) * 0.26f + fres * 0.12f) * uSunIntensity * directVis * 0.44f * (1.0f - uStorm * 0.82f);
    }

    float NdotL = max(0.0f, dot(N, L));
    float wrapN = max(0.0f, (dot(N, L) + 0.06f) / 1.06f);
    float diff = (pu.mode < 0.5f) ? NdotL : mix(NdotL, wrapN, 0.14f);
    float localDirectAO = mix(0.32f, 1.0f, smoothstep(0.24f, 0.82f, cavVertAO));
    float3 direct = uSunColor * uSunIntensity * diff * directVis * localDirectAO;

    float skyFactor = clamp(0.42f + 0.58f * N.y, 0.16f, 1.0f);
    float shadowMicro = clamp(0.76f + 0.30f * dot(N, normalize(float3(-L.x, 0.72f, -L.z))), 0.56f, 1.22f);
    float3 fill = mix(uGroundBounce * 0.64f, uShadowSky, skyFactor) * uShadowFill * envAO * cavVertAO * shadowMicro;

    if (pu.mode > 0.5f && pu.mode < 1.5f) {
        float iceOnly = 1.0f - propSnowMask;
        float cleftShade = (1.0f - clamp(cavVertAO, 0.22f, 1.0f)) * (1.0f - clamp(NdotL * directVis, 0.0f, 1.0f));
        fill *= mix(float3(1.0f), float3(0.76f, 0.94f, 1.16f), cleftShade * 0.72f * iceOnly * (1.0f - uStorm * 0.75f));
        float iceLift = iceOnly * (1.0f - uStorm * 0.75f) * (0.04f + 0.10f * uTwilight) * (1.0f - clamp(NdotL * directVis, 0.0f, 1.0f));
        fill += mix(uShadowSky * 0.65f, uSSSColor * 0.55f + float3(0.05f, 0.16f, 0.28f), 0.55f) * iceLift * (0.45f + 0.55f * clamp(cavVertAO, 0.25f, 1.0f));
    }

    float3 col = alb * (direct + fill) + sss + spec;

    float distFog = clamp(1.0f - exp(-dist * uFogDensity * (1.0f + 0.25f * max(0.0f, (2500.0f - in.worldPos.y) / 1500.0f))), 0.0f, 0.94f);
    float sunScatter = pow(max(0.0f, dot(-V, L)), 4.0f) * (1.0f - uStorm * 0.9f);
    float3 aerialCol = mix(uFogColor, mix(uFogColor, uSunColor * 1.08f, 0.52f), sunScatter);
    col = mix(col, aerialCol, distFog);

    if (in.worldPos.y < uValleyCloudAlt + 260.0f) {
        float hBand = 1.0f - smoothstep(uValleyCloudAlt - 180.0f, uValleyCloudAlt + 260.0f, in.worldPos.y);
        float mistNoise = 0.65f + 0.35f * vnoise2(xz * 0.0008f + float2(uTime * 0.006f, 0.0f));
        float mistScale = mix(1.0f, 0.56f, uTwilight);
        float valleyFog = clamp(hBand * uValleyCloudDensity * mistNoise * mistScale * smoothstep(140.0f, 900.0f, dist), 0.0f, 0.90f);
        float3 cloudMistCol = mix(uShadowSky * 1.15f, float3(0.94f, 0.96f, 0.99f) * (0.55f + 0.45f * uSunColor), 0.62f);
        col = mix(col, cloudMistCol, valleyFog);
    }

    return float4(col, 0.55f);
}

// MARK: - 2. Volumetric 3D Snow-Laden Norway Spruces

struct SpruceVarying {
    float4 position [[position]];
    float3 worldPos;
    float3 worldNormal;
    float3 mat;
    float tint;
    float mtnSun;
};

vertex SpruceVarying spruceVertex(
    PropVertexIn in [[stage_in]],
    uint instanceId [[instance_id]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant PropInstanceData* instances [[buffer(BufferIndexInstances)]],
    texture2d<float> coreAux1 [[texture(TextureIndexCoreAux1)]],
    texture2d<float> coreAux2 [[texture(TextureIndexCoreAux2)]]
) {
    PropInstanceData inst = instances[instanceId];
    float4x4 M = inst.instanceMatrix;
    float aTint = inst.propParams.w;
    float3 aMat = in.prop.xyz;

    float4 wp = M * float4(in.position.xyz, 1.0f);
    float3x3 nMat = float3x3(M[0].xyz, M[1].xyz, M[2].xyz);
    float3 worldNormal = normalize(nMat * in.normal.xyz);

    float uTime = frame.cameraPosition.w;
    float uSunElev = frame.sunDirection.w;
    float uStorm = frame.skyHorizon.w;
    float CORE_HALF = frame.extraParams.y;

    float2 tXZ = (M * float4(0.0f, 0.0f, 0.0f, 1.0f)).xz;
    float phase = fmod(fmod(tXZ.x * 0.13f + tXZ.y * 0.17f, 62.83f) + 62.83f, 62.83f) + aTint * 6.283f;
    float gust = sin(uTime * 1.15f + phase + sin(uTime * 0.42f + fmod(tXZ.x, 128.0f) * 0.05f))
               + 0.42f * sin(uTime * 2.35f + phase * 1.6f);
    float bendMask = pow(clamp(aMat.z, 0.0f, 1.0f), 1.65f);
    float swayAmp = mix(0.11f, 0.46f, uStorm);
    float2 windDir = float2(0.84f, 0.54f);
    float2 crossDir = float2(-0.54f, 0.84f);
    float lat = sin(uTime * 1.65f + phase * 1.3f) * swayAmp * 0.34f * bendMask;
    float tipMask = smoothstep(0.45f, 0.98f, aMat.y) * smoothstep(0.08f, 0.48f, aMat.z);
    float flutter = sin(uTime * 4.6f + in.position.x * 2.1f + in.position.z * 2.4f + phase) * tipMask * mix(0.028f, 0.11f, uStorm);

    wp.xz += windDir * (gust * swayAmp * bendMask + flutter) + crossDir * lat;
    wp.y  -= abs(gust * swayAmp * bendMask) * 0.08f + flutter * 0.30f;

    float2 pUv = clamp((tXZ + CORE_HALF) / (2.0f * CORE_HALF), 0.0f, 1.0f);
    float3 aux1 = coreAux1.sample(samLinearClamp, pUv).rgb;
    float3 aux2 = coreAux2.sample(samLinearClamp, pUv).rgb;
    float hAng = dot(float4(aux1.g, aux1.b, aux2.r, aux2.g) * 1.5707963f, frame.horizonBlend);
    float mtnSun = smoothstep(-0.025f, 0.045f, uSunElev - hAng);

    SpruceVarying out;
    out.worldPos = wp.xyz;
    out.worldNormal = worldNormal;
    out.mat = aMat;
    out.tint = aTint;
    out.mtnSun = mix(mtnSun, 0.30f, uStorm * 0.85f);
    out.position = frame.viewProjectionMatrix * wp;
    return out;
}

fragment float4 spruceFragment(
    SpruceVarying in [[stage_in]],
    bool isFrontFace [[front_facing]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    texture2d<float> snowNormal [[texture(TextureIndexSnowNormal)]],
    texture2d<float> snowCavity [[texture(TextureIndexSnowCavity)]]
) {
    float3 camPos = frame.cameraPosition.xyz;
    float3 L = normalize(frame.sunDirection.xyz);
    float3 uSunColor = frame.sunColor.rgb;
    float uSunIntensity = frame.sunColor.w;
    float uStorm = frame.skyHorizon.w;
    float3 uShadowSky = frame.shadowSkyColor.rgb;
    float uShadowFill = frame.shadowSkyColor.w;
    float3 uSSSColor = frame.sssColor.rgb;
    float3 uFogColor = frame.fogColor.rgb;
    float uFogDensity = frame.fogColor.w;

    float3 N = normalize(isFrontFace ? in.worldNormal : -in.worldNormal);
    float3 V = normalize(camPos - in.worldPos);
    float dist = length(in.worldPos - camPos);
    float matType = in.mat.x;
    float coreAO = clamp(in.mat.y, 0.16f, 1.0f);
    float hNorm = clamp(in.mat.z, 0.0f, 1.0f);
    float closeFade = 1.0f - smoothstep(25.0f, 180.0f, dist);
    float stormVis = mix(1.0f, 0.32f, uStorm) * in.mtnSun;

    float3 wMod = fmod(fmod(in.worldPos, 32.0f) + 32.0f, 32.0f);
    float3 col;
    if (matType < 0.5f) {
        float barkF = vnoise3(float3(wMod.x * 5.0f, wMod.y * 1.4f, wMod.z * 5.0f));
        float3 barkAlb = mix(float3(0.065f, 0.044f, 0.032f), float3(0.14f, 0.098f, 0.072f), barkF);
        float NdotL = max(0.0f, dot(N, L));
        col = barkAlb * (uSunColor * uSunIntensity * NdotL * stormVis * 0.65f + uShadowSky * uShadowFill * 0.42f) * (0.45f + 0.55f * coreAO);
    } else if (matType < 1.5f) {
        float tuft1 = vnoise3(wMod * 3.8f);
        float tuft2 = vnoise3(wMod * 10.5f);
        float ultraFade = 1.0f - smoothstep(10.0f, 56.0f, dist);
        float tuft3 = 0.5f;
        if (ultraFade > 0.02f) {
            tuft3 = vnoise3(wMod * 24.5f);
        }
        float needleGrain = tuft1 * 0.46f + tuft2 * 0.34f + mix(0.5f, tuft3, ultraFade) * 0.20f;

        float3 bN = normalize(N + float3((tuft1 - 0.5f) * 0.58f + (tuft3 - 0.5f) * 0.24f * ultraFade,
                                         (tuft2 - 0.5f) * 0.44f,
                                         (needleGrain - 0.5f) * 0.58f) * (0.45f + 0.55f * closeFade));

        float3 deepPine = mix(float3(0.011f, 0.036f, 0.018f), float3(0.022f, 0.054f, 0.024f), in.tint);
        float3 tipPine  = mix(float3(0.026f, 0.070f, 0.034f), float3(0.044f, 0.090f, 0.042f), in.tint);
        float3 needleAlb = mix(deepPine, tipPine, needleGrain * (0.4f + 0.6f * coreAO));

        float frost = smoothstep(0.58f, 0.90f, bN.y + (needleGrain - 0.52f) * 0.38f) * smoothstep(0.45f, 0.95f, coreAO);
        needleAlb = mix(needleAlb, float3(0.80f, 0.85f, 0.92f), frost * 0.18f);

        float crownAO = mix(0.32f, 1.05f, coreAO) * mix(0.68f, 1.05f, hNorm);
        float wrapSun = clamp((dot(bN, L) + 0.22f) / 1.22f, 0.0f, 1.0f);
        float3 direct = uSunColor * uSunIntensity * wrapSun * stormVis * 0.86f;
        float3 fill = uShadowSky * uShadowFill * (0.36f + 0.64f * clamp(bN.y * 0.5f + 0.5f, 0.0f, 1.0f));

        float3 H = normalize(L + V);
        float waxySpec = pow(max(0.0f, dot(bN, H)), 28.0f) * (0.04f + 0.14f * frost) * coreAO * stormVis;

        float back = pow(clamp(dot(V, -L + bN * 0.25f), 0.0f, 1.0f), 2.5f) * (1.0f - uStorm * 0.85f) * in.mtnSun;
        float3 rim = float3(0.13f, 0.30f, 0.13f) * uSunColor * back * coreAO * 1.10f;
        col = needleAlb * (direct + fill) * crownAO + rim + uSunColor * waxySpec * uSunIntensity;
    } else {
        float2 sUv = wMod.xz * 0.34f + float2(wMod.y * 0.18f, -wMod.y * 0.14f);
        float3 sN = snowNormal.sample(samRepeatAniso, sUv).xyz * 2.0f - 1.0f;
        float3 sCav = snowCavity.sample(samRepeatAniso, sUv).rgb;
        float3 snowN = normalize(N + float3(sN.x, 0.0f, sN.y) * (0.28f * closeFade + 0.08f));

        float3 snowAlb = mix(float3(0.76f, 0.82f, 0.90f), float3(0.90f, 0.93f, 0.97f), sCav.r * 0.48f + hNorm * 0.22f + 0.30f);
        float NdotL = max(0.0f, dot(snowN, L));
        float wrapN = max(0.0f, (dot(snowN, L) + 0.10f) / 1.10f);
        float diff = mix(NdotL, wrapN, 0.18f);

        float aoShade = mix(0.62f, 1.04f, coreAO);
        float3 direct = uSunColor * uSunIntensity * diff * stormVis * 0.96f;
        float3 fill = uShadowSky * uShadowFill * (0.48f + 0.52f * clamp(snowN.y * 0.5f + 0.5f, 0.0f, 1.0f)) * aoShade;

        float back = pow(clamp(dot(V, -L + snowN * 0.40f), 0.0f, 1.0f), 2.8f) * (1.0f - uStorm * 0.78f);
        float rimFres = pow(1.0f - max(0.0f, dot(snowN, V)), 3.2f);
        float3 sss = uSSSColor * (back * 0.30f + rimFres * 0.10f) * (0.12f + 0.88f * stormVis);
        col = snowAlb * (direct + fill) + sss;
    }

    float fog = clamp(1.0f - exp(-dist * uFogDensity * (1.0f + 0.25f * uStorm)), 0.0f, 0.94f);
    float sunScatter = pow(max(0.0f, dot(-V, L)), 4.0f) * (1.0f - uStorm * 0.9f);
    float3 aerialCol = mix(uFogColor, mix(uFogColor, uSunColor * 1.08f, 0.52f), sunScatter);
    col = mix(col, aerialCol, fog);
    return float4(col, 0.55f);
}

// MARK: - 3. Station Architecture, Cable Cars & Wooden Drop-In Stakes

struct StationVarying {
    float4 position [[position]];
    float3 worldPos;
    float3 worldNormal;
};

vertex StationVarying stationVertex(
    StationVertexIn in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ModelUniforms& model [[buffer(BufferIndexModelUniforms)]]
) {
    float4 wp = model.modelMatrix * float4(in.position, 1.0f);
    float3x3 nMat = float3x3(model.normalMatrix[0].xyz, model.normalMatrix[1].xyz, model.normalMatrix[2].xyz);
    StationVarying out;
    out.worldPos = wp.xyz;
    out.worldNormal = normalize(nMat * in.normal);
    out.position = frame.viewProjectionMatrix * wp;
    return out;
}

fragment float4 stationFragment(
    StationVarying in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ModelUniforms& model [[buffer(BufferIndexModelUniforms)]],
    texture2d<float> coreAux1 [[texture(TextureIndexCoreAux1)]],
    texture2d<float> coreAux2 [[texture(TextureIndexCoreAux2)]]
) {
    float3 camPos = frame.cameraPosition.xyz;
    float uTime = frame.cameraPosition.w;
    float3 L = normalize(frame.sunDirection.xyz);
    float uSunElev = frame.sunDirection.w;
    float3 uSunColor = frame.sunColor.rgb;
    float uSunIntensity = frame.sunColor.w;
    float uStorm = frame.skyHorizon.w;
    float3 uShadowSky = frame.shadowSkyColor.rgb;
    float uShadowFill = frame.shadowSkyColor.w;
    float3 uGroundBounce = frame.groundBounceColor.rgb;
    float uTwilight = frame.groundBounceColor.w;
    float3 uFogColor = frame.fogColor.rgb;
    float uFogDensity = frame.fogColor.w;
    float uValleyCloudAlt = frame.cloudParams.x;
    float uValleyCloudDensity = frame.cloudParams.y;
    float uCloudShadowStrength = frame.cloudParams.z;
    float CORE_HALF = frame.extraParams.y;

    float3 N = normalize(in.worldNormal);
    float3 V = normalize(camPos - in.worldPos);
    float3 H = normalize(L + V);
    float dist = length(in.worldPos - camPos);

    float2 xz = in.worldPos.xz;
    float2 pUv = clamp((xz + CORE_HALF) / (2.0f * CORE_HALF), 0.0f, 1.0f);
    float3 aux1 = coreAux1.sample(samLinearClamp, pUv).rgb;
    float3 aux2 = coreAux2.sample(samLinearClamp, pUv).rgb;
    float hAng = dot(float4(aux1.g, aux1.b, aux2.r, aux2.g) * 1.5707963f, frame.horizonBlend);
    float mtnShadow = smoothstep(-0.020f, 0.044f, uSunElev - hAng);
    mtnShadow = mix(mtnShadow, 0.32f, uStorm * 0.85f);
    float csh = cloudShadowAt(xz, uTime) * uCloudShadowStrength * (1.0f - uStorm * 0.8f);
    float directVis = mtnShadow * (1.0f - csh);

    float3 alb = model.baseColor.rgb;
    float roughness = clamp(model.pbrParams.x, 0.05f, 1.0f);
    float metalness = clamp(model.pbrParams.y, 0.0f, 1.0f);
    float emissiveIntensity = model.pbrParams.z;
    float3 emissiveCol = model.fgBounceColor.rgb;

    float NdotL = max(0.0f, dot(N, L));
    float3 direct = uSunColor * uSunIntensity * NdotL * directVis;
    float skyFactor = clamp(0.42f + 0.58f * N.y, 0.16f, 1.0f);
    float3 fill = mix(uGroundBounce * 0.64f, uShadowSky, skyFactor) * uShadowFill;

    float specPow = mix(128.0f, 8.0f, roughness * roughness);
    float3 F0 = mix(float3(0.04f), alb, metalness);
    float specScalar = pow(max(0.0f, dot(N, H)), specPow) * (1.0f - roughness * 0.72f);
    float3 spec = F0 * uSunColor * uSunIntensity * specScalar * directVis * (1.0f - uStorm * 0.8f);

    float3 col = alb * (1.0f - metalness * 0.45f) * (direct + fill) + spec + emissiveCol * emissiveIntensity;

    float distFog = clamp(1.0f - exp(-dist * uFogDensity * (1.0f + 0.25f * max(0.0f, (2500.0f - in.worldPos.y) / 1500.0f))), 0.0f, 0.94f);
    float sunScatter = pow(max(0.0f, dot(-V, L)), 4.0f) * (1.0f - uStorm * 0.9f);
    float3 aerialCol = mix(uFogColor, mix(uFogColor, uSunColor * 1.08f, 0.52f), sunScatter);
    col = mix(col, aerialCol, distFog);

    if (in.worldPos.y < uValleyCloudAlt + 260.0f) {
        float hBand = 1.0f - smoothstep(uValleyCloudAlt - 180.0f, uValleyCloudAlt + 260.0f, in.worldPos.y);
        float mistNoise = 0.65f + 0.35f * vnoise2(xz * 0.0008f + float2(uTime * 0.006f, 0.0f));
        float mistScale = mix(1.0f, 0.56f, uTwilight);
        float valleyFog = clamp(hBand * uValleyCloudDensity * mistNoise * mistScale * smoothstep(140.0f, 900.0f, dist), 0.0f, 0.90f);
        float3 cloudMistCol = mix(uShadowSky * 1.15f, float3(0.94f, 0.96f, 0.99f) * (0.55f + 0.45f * uSunColor), 0.62f);
        col = mix(col, cloudMistCol, valleyFog);
    }

    return float4(col, 0.55f);
}

// MARK: - 4. Telepherique Steel Cables

struct CableVarying {
    float4 position [[position]];
};

vertex CableVarying cableVertex(
    CableVertexIn in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ModelUniforms& model [[buffer(BufferIndexModelUniforms)]]
) {
    float4 wp = model.modelMatrix * float4(in.position.xyz, 1.0f);
    CableVarying out;
    out.position = frame.viewProjectionMatrix * wp;
    return out;
}

fragment float4 cableFragment(
    constant ModelUniforms& model [[buffer(BufferIndexModelUniforms)]]
) {
    return float4(model.baseColor.rgb, 0.72f);
}

// MARK: - 5. Single-Draw-Call GPU Ground Shadows + Tree-Well AO

struct GroundShadowVarying {
    float4 position [[position]];
    float2 meters;
    float4 params;
    float3 dirMult;
    float litFactor;
    float seed;
    float dist;
};

vertex GroundShadowVarying groundShadowVertex(
    ShadowQuadVertexIn in [[stage_in]],
    uint instanceId [[instance_id]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ShadowInstanceData* shadows [[buffer(BufferIndexInstances)]],
    texture2d<float> coreDem [[texture(TextureIndexCoreDem)]],
    texture2d<float> coreAux1 [[texture(TextureIndexCoreAux1)]],
    texture2d<float> coreAux2 [[texture(TextureIndexCoreAux2)]]
) {
    ShadowInstanceData sh = shadows[instanceId];
    float3 aOrigin = sh.posRadius.xyz;
    float radius = sh.posRadius.w;
    float3 N = normalize(sh.normalHeight.xyz);
    float h = sh.normalHeight.w;
    float baseOp = sh.extra.x;
    float kind = sh.extra.y;
    float aSeed = sh.extra.z;

    float3 camPos = frame.cameraPosition.xyz;
    float3 L = normalize(frame.sunDirection.xyz);
    float uSunElev = frame.sunDirection.w;
    float3 uSunColor = frame.sunColor.rgb;
    float uSunIntensity = frame.sunColor.w;
    float uStorm = frame.skyHorizon.w;
    float3 uShadowSky = frame.shadowSkyColor.rgb;
    float uShadowFill = frame.shadowSkyColor.w;
    float3 uGroundBounce = frame.groundBounceColor.rgb;
    float uCoreSize = frame.cloudParams.w;
    float CORE_HALF = frame.extraParams.y;
    float MAX_ELEV = frame.extraParams.w;

    float2 oUv = clamp((aOrigin.xz + CORE_HALF) / (2.0f * CORE_HALF), 0.0f, 1.0f);
    float3 oAux1 = coreAux1.sample(samLinearClamp, oUv).rgb;
    float3 oAux2 = coreAux2.sample(samLinearClamp, oUv).rgb;
    float hAng = dot(float4(oAux1.g, oAux1.b, oAux2.r, oAux2.g) * 1.5707963f, frame.horizonBlend);
    float mtnShadow = smoothstep(-0.020f, 0.044f, uSunElev - hAng);
    float ndl = max(0.0f, dot(N, L));
    float litFactor = mtnShadow * smoothstep(0.02f, 0.20f, ndl) * (1.0f - uStorm * 0.92f);

    float3 directL = uSunColor * uSunIntensity * ndl * mtnShadow * (1.0f - uStorm * 0.92f);
    float skyFactor = clamp(0.42f + 0.58f * N.y, 0.16f, 1.0f);
    float3 skyL = mix(uGroundBounce * 0.64f, uShadowSky, skyFactor) * uShadowFill;
    float3 occL = skyL * 0.62f;
    float3 rawRatio = occL / max(directL + skyL, float3(0.06f));

    float2 castXZ = -L.xz;
    float cLen = length(castXZ);
    float2 alongXZ = cLen > 0.001f ? (castXZ / cLen) : float2(1.0f, 0.0f);
    float2 acrossXZ = float2(-alongXZ.y, alongXZ.x);

    float denom = max(0.26f, dot(N, L));
    float rawCast = (h * max(0.35f, N.y) / denom) * cLen;
    float castLen = clamp(rawCast, radius * 0.6f, min(h * 1.75f, 22.0f)) * litFactor;

    float backPad = radius * 1.38f;
    float halfSpanX = radius * 1.48f + castLen * 0.10f;
    float zMin = -backPad;
    float zMax = max(backPad, castLen + radius * 0.85f);

    float lx = in.position.x * 2.0f * halfSpanX;
    float lz = mix(zMin, zMax, in.position.z + 0.5f);
    float2 wxz = aOrigin.xz + acrossXZ * lx + alongXZ * lz;

    float e = 8.0f;
    float hC = sampleDemH(wxz, coreDem, uCoreSize, CORE_HALF, MAX_ELEV);
    float sMag = length(float2(
        sampleDemH(wxz + float2(e, 0.0f), coreDem, uCoreSize, CORE_HALF, MAX_ELEV) - sampleDemH(wxz - float2(e, 0.0f), coreDem, uCoreSize, CORE_HALF, MAX_ELEV),
        sampleDemH(wxz + float2(0.0f, e), coreDem, uCoreSize, CORE_HALF, MAX_ELEV) - sampleDemH(wxz - float2(0.0f, e), coreDem, uCoreSize, CORE_HALF, MAX_ELEV)
    ) / (2.0f * e));
    float2 pUv = clamp((wxz + CORE_HALF) / (2.0f * CORE_HALF), 0.0f, 1.0f);
    float powd = coreAux1.sample(samLinearClamp, pUv).r;
    float gy = hC + microRelief(wxz, sMag, powd);
    if (sMag > 0.72f) {
        float sharpM = clamp((sMag - 0.72f) / 0.55f, 0.0f, 1.0f);
        float ridgeS = 1.0f - abs(sin(wxz.x * 0.019f + wxz.y * 0.014f) * cos(wxz.x * 0.012f - wxz.y * 0.023f));
        gy += (ridgeS - 0.5f) * 4.5f * sharpM;
    }

    float dist = length(float3(wxz.x, gy, wxz.y) - camPos);
    float distLift = clamp((dist - 18.0f) * 0.0026f, 0.0f, 2.0f);
    float baseLift = kind > 2.5f ? 0.035f : 0.18f;
    float3 wp = float3(wxz.x, gy + baseLift + distLift, wxz.y);

    GroundShadowVarying out;
    out.position = frame.viewProjectionMatrix * float4(wp, 1.0f);
    out.meters = float2(lx, lz);
    out.params = float4(radius, max(0.01f, castLen), kind, baseOp);
    out.dirMult = clamp(rawRatio, float3(0.07f, 0.13f, 0.25f), float3(1.0f));
    out.litFactor = litFactor;
    out.seed = aSeed;
    out.dist = dist;
    return out;
}

fragment float4 groundShadowFragment(
    GroundShadowVarying in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]]
) {
    float distFade = 1.0f - smoothstep(950.0f, 2600.0f, in.dist);
    if (distFade < 0.01f) discard_fragment();

    float crownR = in.params.x;
    float castLen = in.params.y;
    float kind = in.params.z;
    float baseOp = in.params.w;
    float lx = in.meters.x;
    float lz = in.meters.y;
    float uFogDensity = frame.fogColor.w;

    float rNorm = length(float2(lx, lz)) / max(0.22f, crownR * (kind > 2.5f ? 0.58f : 1.24f));
    float aoCore = exp(-rNorm * rNorm * (kind > 2.5f ? 3.2f : 1.85f)) * (1.0f - smoothstep(0.10f, 1.42f, rNorm));
    float aoStrength = aoCore * (kind < 0.5f ? 0.84f : (kind > 2.5f ? 0.62f : 0.56f)) * baseOp;

    float dirMask = 0.0f;
    if (in.litFactor > 0.02f && castLen > 0.45f) {
        float hFrac = clamp(lz / castLen, 0.0f, 1.0f);
        float startFade = smoothstep(kind > 2.5f ? -0.08f : -crownR * 0.60f, kind > 2.5f ? 0.05f : crownR * 0.08f, lz);
        float endPen = kind > 2.5f ? (0.28f + castLen * 0.055f) : (0.45f + castLen * 0.075f);
        float endFade = 1.0f - smoothstep(castLen - endPen * 0.45f, castLen + endPen, lz);

        if (kind > 2.5f) {
            float halfW = mix(0.048f, 0.088f, hFrac);
            float pen = 0.052f + hFrac * 0.085f;
            float across = 1.0f - smoothstep(halfW, halfW + pen, abs(lx));
            dirMask = across * startFade * endFade * in.litFactor * baseOp * 0.86f;
        } else if (kind < 0.5f) {
            float coneEnv = crownR * mix(1.16f, 0.18f, pow(hFrac, 0.86f));
            float tiers = 1.0f + (0.20f * sin(hFrac * 19.0f + in.seed * 6.283f) + 0.11f * cos(hFrac * 35.0f - in.seed * 9.0f))
                        * smoothstep(0.0f, 0.12f, hFrac) * (1.0f - hFrac * 0.28f);
            float halfW = coneEnv * tiers;
            float pen = crownR * (0.14f + hFrac * 0.26f) + castLen * 0.024f * hFrac;
            float across = 1.0f - smoothstep(halfW - pen * 0.45f, halfW + pen, abs(lx));
            float canopyDensity = mix(1.0f, 0.72f, pow(hFrac, 1.35f));
            dirMask = across * startFade * endFade * canopyDensity * in.litFactor * clamp(baseOp * 1.18f, 0.0f, 1.0f);
        } else {
            float domeEnv = crownR * mix(1.04f, 0.36f, pow(hFrac, 0.9f));
            float pen = crownR * (0.24f + hFrac * 0.22f);
            float across = 1.0f - smoothstep(domeEnv - pen * 0.5f, domeEnv + pen, abs(lx));
            dirMask = across * startFade * endFade * in.litFactor * baseOp * 0.92f;
        }
    }

    float fogFade = exp(-in.dist * uFogDensity * 1.05f) * distFade;
    float totalMask = clamp(max(dirMask, aoStrength), 0.0f, 1.0f) * fogFade;
    if (totalMask < 0.004f) discard_fragment();

    float angW = (lx * lx + lz * lz) > 1e-6f ? atan2(lz, lx) : 0.0f;
    float wellPert = 1.0f + 0.14f * sin(angW * 5.0f + in.seed * 19.3f) + 0.09f * cos(angW * 9.0f - in.seed * 11.7f);
    float rWell = rNorm * wellPert;
    float3 aoMult = float3(0.22f, 0.29f, 0.41f);
    if (kind < 0.5f) {
        float deepWell = exp(-rWell * rWell * 3.8f);
        float litterCore = (1.0f - smoothstep(0.08f, 0.32f, rWell)) * (0.72f + 0.28f * sin(angW * 13.0f + in.seed * 31.0f));
        aoMult = mix(float3(0.22f, 0.30f, 0.42f), float3(0.12f, 0.21f, 0.35f), deepWell * 0.85f);
        aoMult = mix(aoMult, float3(0.09f, 0.11f, 0.12f), litterCore * 0.65f);
    }
    float3 multDir = mix(float3(1.0f), in.dirMult, clamp(dirMask * fogFade, 0.0f, 1.0f));
    float3 multAo  = mix(float3(1.0f), aoMult, clamp(aoStrength * fogFade * 0.95f, 0.0f, 1.0f));
    float3 finalMult = min(multDir, multAo);
    return float4(clamp(finalMult, float3(0.06f, 0.11f, 0.14f), float3(1.0f)), 1.0f);
}

// MARK: - 6. Directional Shadow Map Depth Pass (Stakes)

vertex float4 stationShadowVertex(
    StationVertexIn in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ModelUniforms& model [[buffer(BufferIndexModelUniforms)]]
) {
    float4 wp = model.modelMatrix * float4(in.position, 1.0f);
    return frame.lightViewProjectionMatrix * wp;
}
