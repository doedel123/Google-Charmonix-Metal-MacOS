#include "ShaderTypes.h"

using namespace metal;

constexpr sampler samLinearClamp(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);

struct ParticleUniforms {
    float4 windAndFocal; // xyz = flake wind, w = focalScale
};

struct BannerInstanceData {
    float4x4 modelMatrix;
};

struct BannerVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal   [[attribute(VertexAttributeNormal)]];
    float2 uv       [[attribute(VertexAttributeUV)]];
    float4 color    [[attribute(VertexAttributeColor)]];
};

static inline float2 quadCorner(uint vid) {
    // Triangle strip: 0=(-0.5,+0.5), 1=(-0.5,-0.5), 2=(+0.5,+0.5), 3=(+0.5,-0.5)
    float x = (vid & 2) ? 0.5f : -0.5f;
    float y = (vid & 1) ? -0.5f : 0.5f;
    return float2(x, y);
}

static inline float2 quadPointCoord(uint vid) {
    // Matches WebGL gl_PointCoord: (0,0) top-left -> (1,1) bottom-right
    float x = (vid & 2) ? 1.0f : 0.0f;
    float y = (vid & 1) ? 1.0f : 0.0f;
    return float2(x, y);
}

static inline float hash21(float2 p) {
    p = fract(p * float2(123.34f, 456.21f));
    p += dot(p, p + 45.32f);
    return fract(p.x * p.y);
}

static inline float vnoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    return mix(
        mix(hash21(i), hash21(i + float2(1.0f, 0.0f)), f.x),
        mix(hash21(i + float2(0.0f, 1.0f)), hash21(i + float2(1.0f, 1.0f)), f.x),
        f.y
    );
}

static inline float fbm(float2 p) {
    float v = 0.0f, a = 0.52f;
    for (int i = 0; i < 4; i++) {
        v += a * vnoise(p);
        p = p * 2.08f + float2(9.2f, -5.7f);
        a *= 0.5f;
    }
    return v;
}

// MARK: - 1. Volumetric Coldsmoke / Powder Spray / Avalanche Clouds / Spindrift Puffs

struct PuffVarying {
    float4 position [[position]];
    float2 pointCoord;
    float alpha;
    float rot;
    float kind;
    float3 worldPos;
};

vertex PuffVarying puffVertex(
    uint vid [[vertex_id]],
    uint instanceId [[instance_id]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ParticleUniforms& pu [[buffer(BufferIndexSubsystemUniforms)]],
    constant ParticleVertexData* puffs [[buffer(BufferIndexInstances)]]
) {
    ParticleVertexData p = puffs[instanceId];
    float3 pos = p.posSize.xyz;
    float aSize = p.posSize.w;
    float aAlpha = p.colorAlpha.x;
    float aRot = p.colorAlpha.y;
    float aKind = p.colorAlpha.z;

    PuffVarying out;
    out.pointCoord = quadPointCoord(vid);
    out.rot = aRot;
    out.kind = aKind;
    out.worldPos = pos;

    if (aAlpha <= 0.004f || aSize <= 0.001f) {
        out.alpha = 0.0f;
        out.position = float4(2.0f, 2.0f, 2.0f, 1.0f);
        return out;
    }

    float3 camPos = frame.cameraPosition.xyz;
    float4 mv = frame.viewMatrix * float4(pos, 1.0f);
    float camDist = length(pos - camPos);

    float nearStart = aKind > 0.5f ? 2.2f : 0.95f;
    float nearEnd   = aKind > 0.5f ? 5.0f : 2.35f;
    float nearFade  = smoothstep(nearStart, nearEnd, camDist) * smoothstep(0.65f, 1.95f, -mv.z);
    out.alpha = aAlpha * nearFade;

    if (out.alpha <= 0.004f || -mv.z <= 0.05f) {
        out.alpha = 0.0f;
        out.position = float4(2.0f, 2.0f, 2.0f, 1.0f);
        return out;
    }

    float dist = max(0.35f, -mv.z);
    float maxPx = aKind > 0.5f ? 280.0f : 160.0f;
    float pxSize = clamp((aSize * pu.windAndFocal.w) / dist, 1.0f, maxPx);

    float4 clip = frame.projectionMatrix * mv;
    float2 corner = quadCorner(vid);
    float2 viewport = max(frame.screenParams.xy, float2(1.0f));
    clip.xy += corner * (pxSize * 2.0f / viewport) * clip.w;
    out.position = clip;
    return out;
}

fragment float4 puffFragment(
    PuffVarying in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    texture2d<float> puffTex [[texture(TextureIndexAlbedo)]]
) {
    if (in.alpha <= 0.004f) discard_fragment();

    float2 uv = in.pointCoord - 0.5f;
    float c = cos(in.rot), s = sin(in.rot);
    uv = float2(c * uv.x - s * uv.y, s * uv.x + c * uv.y) + 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) discard_fragment();

    float4 tex = puffTex.sample(samLinearClamp, uv);
    float a = tex.a * in.alpha;
    if (a < 0.008f) discard_fragment();

    // Un-premultiply CGContext RGBA8 so tex.g / tex.b match WebGL unpremultiplied texture channels 1:1
    float3 straightRGB = clamp(tex.rgb / max(tex.a, 0.001f), 0.0f, 1.0f);

    float3 camPos = frame.cameraPosition.xyz;
    float3 uSunDir = normalize(frame.sunDirection.xyz);
    float3 uSunColor = frame.sunColor.rgb;
    float3 uShadowColor = frame.shadowSkyColor.rgb;
    float uStorm = frame.skyHorizon.w;

    float3 viewDir = normalize(in.worldPos - camPos);
    float fwdScatter = pow(max(0.0f, dot(viewDir, uSunDir)), 3.0f) * (1.0f - uStorm * 0.85f);
    float topMask = clamp(1.0f - in.pointCoord.y, 0.0f, 1.0f);
    float lit = clamp(straightRGB.g * 0.52f + topMask * 0.42f + 0.08f * (1.0f - straightRGB.b), 0.0f, 1.0f);

    float3 belly = mix(uShadowColor * 1.02f, uShadowColor * 0.68f, step(0.5f, in.kind));
    float3 crest = float3(0.96f, 0.98f, 1.0f) * mix(float3(0.84f, 0.88f, 0.94f), uSunColor * 1.32f, 1.0f - uStorm * 0.72f);
    float3 col = mix(belly, crest, lit);

    float rimEdge = (1.0f - straightRGB.b) * smoothstep(0.02f, 0.45f, tex.a);
    col += uSunColor * rimEdge * (0.36f + 0.85f * fwdScatter) * (1.0f - uStorm * 0.8f);
    return float4(col, clamp(a, 0.0f, 0.94f));
}

// MARK: - 2. Ambient Alpine Blowing Snow / Diamond Dust / Storm Flakes

struct FlakeVarying {
    float4 position [[position]];
    float2 pointCoord;
    float alpha;
    float spark;
    float diamond;
    float phase;
};

vertex FlakeVarying flakeVertex(
    uint vid [[vertex_id]],
    uint instanceId [[instance_id]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ParticleUniforms& pu [[buffer(BufferIndexSubsystemUniforms)]],
    constant float4* flakeSeeds [[buffer(BufferIndexInstances)]] // xyz = baseOffset, w = seed
) {
    float4 fd = flakeSeeds[instanceId];
    float3 basePos = fd.xyz;
    float aSeed = fd.w;

    float3 camPos = frame.cameraPosition.xyz;
    float uTime = frame.cameraPosition.w;
    float uRate = frame.extraParams.x;
    float uStorm = frame.skyHorizon.w;
    float3 uWind = pu.windAndFocal.xyz;
    float uFocalScale = pu.windAndFocal.w;
    float3 uSunDir = normalize(frame.sunDirection.xyz);

    float3 box = float3(44.0f, 26.0f, 44.0f);
    float isDiamond = step(0.86f, aSeed) * (1.0f - uStorm * 0.92f);
    float3 wind = mix(uWind, uWind * 0.22f + float3(0.6f, -0.35f, 0.3f), isDiamond);
    float3 p = basePos + wind * (uTime * (0.55f + aSeed * 0.95f));
    p.x += sin(uTime * 1.9f + aSeed * 30.0f) * mix(1.25f, 0.45f, isDiamond);
    p.z += cos(uTime * 1.5f + aSeed * 25.0f) * mix(1.25f, 0.45f, isDiamond);

    float3 shifted = p - camPos + box * 0.5f;
    float3 rel = shifted - box * floor(shifted / box) - box * 0.5f;
    float3 wp = camPos + mix(rel, rel * 0.44f, isDiamond);

    float4 mv = frame.viewMatrix * float4(wp, 1.0f);
    float dist = max(0.3f, -mv.z);

    float activeSnow = step(aSeed, uRate);
    float isActive = max(activeSnow, isDiamond);

    FlakeVarying out;
    out.pointCoord = quadPointCoord(vid);

    if (isActive < 0.5f || -mv.z <= 0.05f) {
        out.alpha = 0.0f;
        out.spark = 0.0f;
        out.diamond = 0.0f;
        out.phase = 0.0f;
        out.position = float4(2.0f, 2.0f, 2.0f, 1.0f);
        return out;
    }

    float3 viewDir = normalize(wp - camPos);
    float fwdSun = pow(max(0.0f, dot(viewDir, uSunDir)), 2.2f);
    float twinkle = pow(0.5f + 0.5f * sin(uTime * (4.2f + fract(aSeed * 37.0f) * 5.0f) + aSeed * 190.0f), 4.0f);
    float snowA = activeSnow * smoothstep(2.2f, 4.5f, dist) * (1.0f - smoothstep(14.0f, 22.0f, dist)) * (0.30f + 0.52f * uRate);
    float diamA = isDiamond * smoothstep(1.8f, 3.6f, dist) * (1.0f - smoothstep(9.0f, 17.0f, dist)) * (0.10f + 0.85f * twinkle) * (0.32f + 0.68f * fwdSun) * 0.68f;

    out.alpha = max(snowA, diamA);
    out.spark = step(0.84f, aSeed);
    out.diamond = isDiamond * (0.35f + 0.65f * twinkle) * (0.55f + 0.95f * fwdSun);
    out.phase = fract(aSeed * 71.3f);

    float szSnow = mix(0.05f, 0.11f, fract(aSeed * 13.7f)) * (0.75f + 0.85f * uRate);
    float szDiam = mix(0.015f, 0.032f, fract(aSeed * 29.1f)) * (0.75f + 0.65f * twinkle);
    float sz = mix(szSnow, szDiam, isDiamond);
    float pxSize = clamp((sz * uFocalScale) / dist, 1.2f, isDiamond > 0.5f ? 4.4f : 8.5f) * isActive;

    float4 clip = frame.projectionMatrix * mv;
    float2 corner = quadCorner(vid);
    float2 viewport = max(frame.screenParams.xy, float2(1.0f));
    clip.xy += corner * (pxSize * 2.0f / viewport) * clip.w;
    out.position = clip;
    return out;
}

fragment float4 flakeFragment(
    FlakeVarying in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]]
) {
    if (in.alpha <= 0.005f) discard_fragment();

    float2 d = in.pointCoord - 0.5f;
    float r2 = dot(d, d);
    if (r2 > 0.25f) discard_fragment();

    float3 uSunColor = frame.sunColor.rgb;
    float soft = smoothstep(0.25f, 0.02f, r2);
    float star = in.diamond > 0.1f ? exp(-abs(d.x) * 18.0f - abs(d.y) * 4.5f) + exp(-abs(d.y) * 18.0f - abs(d.x) * 4.5f) : 0.0f;
    float3 prism = mix(float3(1.22f, 0.96f, 0.72f), float3(0.76f, 0.98f, 1.26f), in.phase);
    float3 baseCol = mix(float3(0.95f, 0.98f, 1.02f), uSunColor * 1.85f, in.spark * 0.65f);
    float3 diamCol = prism * (uSunColor * 1.15f + float3(0.55f)) * (1.5f + star * 1.4f);
    float3 col = mix(baseCol, diamCol, clamp(in.diamond, 0.0f, 1.0f));
    return float4(col, clamp((soft + star * 0.35f * in.diamond) * in.alpha, 0.0f, 0.95f));
}

// MARK: - 3. Summit Ridge Snow Banners

struct BannerVarying {
    float4 position [[position]];
    float2 uv;
    float3 worldPos;
};

vertex BannerVarying bannerVertex(
    BannerVertexIn in [[stage_in]],
    uint instanceId [[instance_id]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant BannerInstanceData* banners [[buffer(BufferIndexInstances)]]
) {
    float4 wp = banners[instanceId].modelMatrix * float4(in.position, 1.0f);
    BannerVarying out;
    out.uv = in.uv;
    out.worldPos = wp.xyz;
    out.position = frame.viewProjectionMatrix * wp;
    return out;
}

fragment float4 bannerFragment(
    BannerVarying in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]]
) {
    float u = in.uv.x;
    float v = in.uv.y;
    float uTime = frame.cameraPosition.w;
    float3 camPos = frame.cameraPosition.xyz;
    float3 uSunDir = normalize(frame.sunDirection.xyz);
    float3 uSunColor = frame.sunColor.rgb;
    float3 uShadowColor = frame.shadowSkyColor.rgb;
    float3 uFogColor = frame.fogColor.rgb;
    float uFogDensity = frame.fogColor.w;
    float uStorm = frame.skyHorizon.w;

    float arch = 0.42f + u * 0.14f - u * u * 0.22f;
    float spread = mix(0.14f, 0.46f, pow(max(0.0f, u), 0.65f));
    float dy = abs(v - arch) / spread;
    float envY = exp(-dy * dy * 2.6f) * smoothstep(1.0f, 0.70f, dy);
    float envX = smoothstep(0.0f, 0.09f, u) * pow(max(0.0f, 1.0f - u), 1.55f);

    float2 flow = float2(u * 3.8f - uTime * 0.14f, (v - arch) * 5.2f + sin(u * 4.5f - uTime * 0.3f) * 0.25f);
    float wisp = fbm(flow);
    float wisp2 = fbm(flow * 2.2f + float2(-uTime * 0.12f, 3.1f));
    float dens = smoothstep(0.28f, 0.78f, wisp * 0.65f + wisp2 * 0.35f) * envX * envY;
    float dist = length(in.worldPos - camPos);
    float prox = smoothstep(260.0f, 950.0f, dist);
    float a = dens * (0.42f + 0.16f * uStorm) * prox;
    if (a < 0.008f) discard_fragment();

    float3 viewDir = normalize(in.worldPos - camPos);
    float backLit = pow(max(0.0f, dot(viewDir, uSunDir)), 2.4f) * (1.0f - uStorm * 0.85f);
    float3 col = mix(uShadowColor * 1.35f, float3(0.95f, 0.97f, 1.0f) * (0.85f + 0.45f * uSunColor), 0.55f + 0.35f * v);
    col += uSunColor * (0.45f + 1.35f * backLit) * (1.0f - uStorm * 0.82f);
    float fog = clamp(1.0f - exp(-dist * uFogDensity * 0.55f), 0.0f, 0.78f);
    col = mix(col, uFogColor, fog * 0.5f);
    return float4(col, clamp(a * (1.0f - fog * 0.4f), 0.0f, 0.62f));
}
