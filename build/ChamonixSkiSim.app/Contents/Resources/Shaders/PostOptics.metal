#include "ShaderTypes.h"

struct PostVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal   [[attribute(VertexAttributeNormal)]];
    float2 uv       [[attribute(VertexAttributeUV)]];
    float4 color    [[attribute(VertexAttributeColor)]];
};

struct PostVertexOut {
    float4 position [[position]];
    float2 uv;
};

constexpr sampler kPostLinearClamp(coord::normalized, address::clamp_to_edge, filter::linear);

vertex PostVertexOut postOpticsVertex(PostVertexIn in [[stage_in]]) {
    PostVertexOut out;
    out.position = float4(in.position.xy, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

static inline float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

static inline float ign(float2 p) {
    return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
}

static inline float postLuma(float3 c) {
    return dot(c, float3(0.299, 0.587, 0.114));
}

static inline float3 acesFilm(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

static inline float3 extractBright(texture2d<float> sceneTex, float2 uv, float thresh) {
    float3 c = sceneTex.sample(kPostLinearClamp, clamp(uv, 0.0, 1.0)).rgb;
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    float soft = max(0.0, lum - thresh) / (lum + 0.001);
    return c * clamp(soft, 0.0, 2.5);
}

fragment float4 postBrightHBlurFragment(
    PostVertexOut in [[stage_in]],
    constant PostOpticsUniforms& u [[buffer(BufferIndexSubsystemUniforms)]],
    texture2d<float> sceneTex [[texture(TextureIndexSceneHDR)]]
) {
    float2 uv = in.uv;
    float3 sum = extractBright(sceneTex, uv, u.threshold) * 0.227027;
    float2 stepX = float2(u.texelSize.x * 2.2, 0.0);
    sum += (extractBright(sceneTex, uv + stepX * 1.0, u.threshold) + extractBright(sceneTex, uv - stepX * 1.0, u.threshold)) * 0.1945946;
    sum += (extractBright(sceneTex, uv + stepX * 2.0, u.threshold) + extractBright(sceneTex, uv - stepX * 2.0, u.threshold)) * 0.1216216;
    sum += (extractBright(sceneTex, uv + stepX * 3.0, u.threshold) + extractBright(sceneTex, uv - stepX * 3.0, u.threshold)) * 0.054054;
    sum += (extractBright(sceneTex, uv + stepX * 4.0, u.threshold) + extractBright(sceneTex, uv - stepX * 4.0, u.threshold)) * 0.016216;
    return float4(sum, 1.0);
}

fragment float4 postBlurVFragment(
    PostVertexOut in [[stage_in]],
    constant PostOpticsUniforms& u [[buffer(BufferIndexSubsystemUniforms)]],
    texture2d<float> bloomSrcTex [[texture(TextureIndexBloom)]]
) {
    float2 uv = in.uv;
    float3 sum = bloomSrcTex.sample(kPostLinearClamp, uv).rgb * 0.227027;
    float2 stepY = float2(0.0, u.texelSize.y * 1.8);
    sum += (bloomSrcTex.sample(kPostLinearClamp, clamp(uv + stepY * 1.0, 0.0, 1.0)).rgb + bloomSrcTex.sample(kPostLinearClamp, clamp(uv - stepY * 1.0, 0.0, 1.0)).rgb) * 0.1945946;
    sum += (bloomSrcTex.sample(kPostLinearClamp, clamp(uv + stepY * 2.0, 0.0, 1.0)).rgb + bloomSrcTex.sample(kPostLinearClamp, clamp(uv - stepY * 2.0, 0.0, 1.0)).rgb) * 0.1216216;
    sum += (bloomSrcTex.sample(kPostLinearClamp, clamp(uv + stepY * 3.0, 0.0, 1.0)).rgb + bloomSrcTex.sample(kPostLinearClamp, clamp(uv - stepY * 3.0, 0.0, 1.0)).rgb) * 0.054054;
    sum += (bloomSrcTex.sample(kPostLinearClamp, clamp(uv + stepY * 4.0, 0.0, 1.0)).rgb + bloomSrcTex.sample(kPostLinearClamp, clamp(uv - stepY * 4.0, 0.0, 1.0)).rgb) * 0.016216;
    return float4(sum, 1.0);
}

fragment float4 postGodRaysFragment(
    PostVertexOut in [[stage_in]],
    constant PostOpticsUniforms& u [[buffer(BufferIndexSubsystemUniforms)]],
    texture2d<float> sceneTex [[texture(TextureIndexSceneHDR)]]
) {
    float2 vUv = in.uv;
    float2 asp = float2(u.aspect, 1.0);
    float2 delta = u.sunUV - vUv;
    float distToSun = length(delta * asp);
    if (distToSun > 1.22 || u.godRayVis <= 0.005 || u.godRayStrength <= 0.005) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float offScreen = smoothstep(-0.55, 0.06, u.sunUV.x) * smoothstep(1.55, 0.94, u.sunUV.x)
                    * smoothstep(-0.45, 0.06, u.sunUV.y) * smoothstep(1.45, 0.94, u.sunUV.y);
    if (offScreen <= 0.001) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    const int NUM_SAMPLES = 22;
    float density = 0.94;
    float2 stepUv = delta * (density / float(NUM_SAMPLES));
    float jitter = ign(in.position.xy);
    float2 tc = vUv + stepUv * jitter;

    float illumDecay = 1.0;
    float decay = 0.942;
    float3 accum = float3(0.0);
    float3 warmSun = mix(u.sunColor.rgb, float3(1.0, 0.74, 0.36), 0.32);

    for (int i = 0; i < NUM_SAMPLES; i++) {
        float inBounds = step(0.001, tc.x) * step(tc.x, 0.999) * step(0.001, tc.y) * step(tc.y, 0.999);
        float4 tap = sceneTex.sample(kPostLinearClamp, clamp(tc, 0.001, 0.999));
        float skyMask = smoothstep(0.72, 0.96, tap.a) * inBounds;
        float tapR = length((tc - u.sunUV) * asp);
        float coronaW = exp(-tapR * 4.2) * 0.95 + exp(-tapR * 12.5) * 2.85 + exp(-tapR * 28.0) * 2.10;
        float tapLum = clamp(dot(tap.rgb, float3(0.25, 0.60, 0.15)), 0.0, 3.6);
        float3 emitCol = mix(warmSun, min(tap.rgb, float3(2.6)), 0.22);
        accum += emitCol * (skyMask * coronaW * (0.26 + 0.50 * min(tapLum, 2.2))) * illumDecay;
        illumDecay *= decay;
        tc += stepUv;
    }

    float radialEnv = exp(-distToSun * 1.12) * smoothstep(1.24, 0.04, distToSun);
    float3 shafts = (accum / float(NUM_SAMPLES)) * u.godRayStrength * u.godRayVis * offScreen * radialEnv;
    return float4(shafts, 1.0);
}

static inline float2 barrelUV(float2 uv, float k) {
    float2 c = uv - 0.5;
    float r2 = dot(c, c);
    float scale = 1.0 / (1.0 + k * 0.53);
    return 0.5 + c * (1.0 + k * r2) * scale;
}

static inline float3 sampleFXAA(texture2d<float> sceneTex, float2 uv, float2 texel) {
    float3 cM  = sceneTex.sample(kPostLinearClamp, uv).rgb;
    float3 cNW = sceneTex.sample(kPostLinearClamp, uv + float2(-1.0, -1.0) * texel).rgb;
    float3 cNE = sceneTex.sample(kPostLinearClamp, uv + float2( 1.0, -1.0) * texel).rgb;
    float3 cSW = sceneTex.sample(kPostLinearClamp, uv + float2(-1.0,  1.0) * texel).rgb;
    float3 cSE = sceneTex.sample(kPostLinearClamp, uv + float2( 1.0,  1.0) * texel).rgb;
    float lM = postLuma(cM), lNW = postLuma(cNW), lNE = postLuma(cNE), lSW = postLuma(cSW), lSE = postLuma(cSE);
    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    if (lMax - lMin < max(0.05, lMax * 0.12)) return cM;
    float2 dir = float2(-((lNW + lNE) - (lSW + lSE)), (lNW + lSW) - (lNE + lSE));
    float dirReduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcp, float2(-4.0), float2(4.0)) * texel;
    float3 rgbA = 0.5 * (
        sceneTex.sample(kPostLinearClamp, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
        sceneTex.sample(kPostLinearClamp, uv + dir * (2.0 / 3.0 - 0.5)).rgb
    );
    float3 rgbB = rgbA * 0.5 + 0.25 * (
        sceneTex.sample(kPostLinearClamp, uv + dir * -0.5).rgb +
        sceneTex.sample(kPostLinearClamp, uv + dir *  0.5).rgb
    );
    float lB = postLuma(rgbB);
    return (lB < lMin || lB > lMax) ? rgbA : rgbB;
}

static inline float2 lensDroplets(float2 uv, float wet, float aspect) {
    if (wet <= 0.01) return float2(0.0);
    float2 p = float2(uv.x * aspect, uv.y) * 9.5;
    float2 id = floor(p);
    float2 gv = fract(p) - 0.5;
    float rnd = hash21(id + float2(4.7, 9.2));
    if (rnd < 0.72) return float2(0.0);
    float2 off = float2(hash21(id + 1.1) - 0.5, hash21(id + 2.3) - 0.5) * 0.45;
    float rad = (0.10 + 0.16 * fract(rnd * 13.7)) * wet;
    float2 d = gv - off;
    float dist = length(d);
    float edgeMask = smoothstep(0.18, 0.52, length(uv - 0.5));
    float drop = smoothstep(rad, rad * 0.25, dist) * edgeMask;
    return -d * drop * 0.18;
}

static inline float drivingSnowLayer(float2 uv, float t, float scale, float speed, float density, float ang, float aspect) {
    float2 p = float2(uv.x * aspect, uv.y);
    p.y += sin(p.x * 3.8 + t * 2.1) * 0.032;
    float ca = cos(ang);
    float sa = sin(ang);
    float2 q = float2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);
    q.x *= scale * 0.62;
    q.y *= scale * 1.45;
    q.x -= t * speed;
    float2 id = floor(q);
    float2 gv = fract(q) - 0.5;
    float h = hash21(id);
    if (h > density) return 0.0;
    float yOff = (fract(h * 19.3) - 0.5) * 0.62;
    float xOff = (fract(h * 27.7) - 0.5) * 0.25;
    float xLen = 0.05 + 0.14 * fract(h * 7.1);
    float dY = abs(gv.y - yOff);
    float dX = max(0.0, abs(gv.x - xOff) - xLen);
    float rad = 0.045 + 0.055 * fract(h * 53.1);
    float wisp = exp(-pow(length(float2(dX * 1.8, dY * 1.6)) / rad, 2.0));
    return wisp * (0.30 + 0.65 * fract(h * 37.9));
}

static inline float3 sampleGodRays(
    float2 suv,
    float centerAlpha,
    texture2d<float> godRaysTex,
    float hasGodRays,
    float2 texel
) {
    if (hasGodRays < 0.5) return float3(0.0);
    float2 t = texel * 1.15;
    float3 shafts = godRaysTex.sample(kPostLinearClamp, suv).rgb * 0.48
                  + godRaysTex.sample(kPostLinearClamp, clamp(suv + float2( t.x, 0.0), 0.001, 0.999)).rgb * 0.13
                  + godRaysTex.sample(kPostLinearClamp, clamp(suv + float2(-t.x, 0.0), 0.001, 0.999)).rgb * 0.13
                  + godRaysTex.sample(kPostLinearClamp, clamp(suv + float2(0.0,  t.y), 0.001, 0.999)).rgb * 0.13
                  + godRaysTex.sample(kPostLinearClamp, clamp(suv + float2(0.0, -t.y), 0.001, 0.999)).rgb * 0.13;
    float isSkyPx = smoothstep(0.72, 0.96, centerAlpha);
    float isFgPx  = 1.0 - smoothstep(0.10, 0.42, centerAlpha);
    float pixelGain = mix(1.62, 0.26, isSkyPx) * mix(1.0, 0.45, isFgPx);
    shafts *= pixelGain;
    return shafts / (1.0 + shafts * 0.82);
}

static inline float3 evalSunFlare(
    float2 uv,
    float2 sunUv,
    float vis,
    float3 sunCol,
    float aspect,
    texture2d<float> sceneTex
) {
    if (vis <= 0.005) return float3(0.0);
    float gpuVis = 1.0;
    if (sunUv.x > 0.02 && sunUv.x < 0.98 && sunUv.y > 0.02 && sunUv.y < 0.98) {
        float2 o = float2(0.011 / aspect, 0.011);
        float v0 = smoothstep(0.72, 0.96, sceneTex.sample(kPostLinearClamp, sunUv).a);
        float v1 = smoothstep(0.72, 0.96, sceneTex.sample(kPostLinearClamp, clamp(sunUv + float2( o.x, 0.0), 0.001, 0.999)).a);
        float v2 = smoothstep(0.72, 0.96, sceneTex.sample(kPostLinearClamp, clamp(sunUv + float2(-o.x, 0.0), 0.001, 0.999)).a);
        float v3 = smoothstep(0.72, 0.96, sceneTex.sample(kPostLinearClamp, clamp(sunUv + float2(0.0,  o.y), 0.001, 0.999)).a);
        float v4 = smoothstep(0.72, 0.96, sceneTex.sample(kPostLinearClamp, clamp(sunUv + float2(0.0, -o.y), 0.001, 0.999)).a);
        gpuVis = clamp((v0 * 2.0 + v1 + v2 + v3 + v4) / 6.0, 0.0, 1.0);
        if (gpuVis <= 0.01) return float3(0.0);
    }
    float2 asp = float2(aspect, 1.0);
    float2 d = (uv - sunUv) * asp;
    float r = length(d);

    float ang = r > 1e-4 ? atan2(d.y, d.x) : 0.0;
    float rays6 = pow(abs(cos(ang * 3.0 + 0.26)), 18.0) * 0.55 + pow(abs(cos(ang * 6.0)), 26.0) * 0.22;
    float star = exp(-r * 7.2) * (0.35 + 1.05 * rays6) * smoothstep(0.65, 0.0, r);
    float streak = exp(-abs(d.y) * 58.0) * exp(-abs(d.x) * 3.2) * 0.22;
    float veil = exp(-r * 2.15) * 0.14;

    float3 flare = sunCol * (star * 0.42 + streak * 0.48 + veil);

    float2 axis = float2(0.5) - sunUv;
    float g1r = length((uv - (float2(0.5) + axis * 0.48)) * asp);
    float g1 = exp(-pow((g1r - 0.042) / 0.024, 2.0)) * 0.024 + exp(-g1r * g1r * 360.0) * 0.018;
    flare += float3(1.05, 0.78, 0.52) * g1;

    float g2r = length((uv - (float2(0.5) + axis * 0.96)) * asp);
    float g2 = smoothstep(0.075, 0.018, g2r) * smoothstep(0.004, 0.036, g2r) * 0.020;
    flare += float3(0.54, 0.86, 1.15) * g2;

    float g3r = length((uv - (float2(0.5) + axis * 1.48)) * asp);
    float g3 = exp(-pow((g3r - 0.095) / 0.036, 2.0)) * 0.014;
    flare += float3(0.95, 0.58, 0.78) * g3;

    float edgeSoft = smoothstep(-0.35, 0.08, sunUv.x) * smoothstep(1.35, 0.92, sunUv.x)
                   * smoothstep(-0.30, 0.08, sunUv.y) * smoothstep(1.30, 0.92, sunUv.y);
    return flare * vis * gpuVis * edgeSoft;
}

fragment float4 postCompositeFragment(
    PostVertexOut in [[stage_in]],
    constant PostOpticsUniforms& u [[buffer(BufferIndexSubsystemUniforms)]],
    texture2d<float> sceneTex [[texture(TextureIndexSceneHDR)]],
    texture2d<float> bloomTex [[texture(TextureIndexBloom)]],
    texture2d<float> godRaysTex [[texture(TextureIndexGodRays)]]
) {
    float2 vUv = in.uv;
    float2 uv = barrelUV(vUv, u.barrel);
    float2 dropOff = lensDroplets(uv, u.lensWet, u.aspect);
    float2 suv = clamp(uv + dropOff, 0.001, 0.999);

    float2 fromCenter = suv - 0.5;
    float r2 = dot(fromCenter, fromCenter);
    float ca = (0.0012 + u.speedFactor * 0.0020) * r2 * (u.barrel > 0.01 ? 1.0 : 0.35);

    float4 centerRaw = sceneTex.sample(kPostLinearClamp, suv);
    float worldMask = smoothstep(0.10, 0.44, centerRaw.a);

    // Metal UV has (0,0) top-left and (1,1) bottom-right, so ground lies at suv.y > u.horizonUV.y.
    float2 vpDelta = suv - u.horizonUV;
    float groundW = mix(0.08, 1.0, smoothstep(u.horizonUV.y - 0.14, u.horizonUV.y + 0.56, suv.y));
    float periph = smoothstep(0.05, 0.50, length(vpDelta));
    float2 radialVec = vpDelta * (u.speedFactor * 0.032 * groundW * (0.28 + 0.72 * periph));
    float2 rollVec = float2(suv.y - 0.5, -(suv.x - 0.5)) * u.rollVel;
    float2 angVelUV = float2(u.angVel.x, -u.angVel.y);
    float2 blurVec = (radialVec + angVelUV + rollVec) * worldMask;
    float blurMag = length(blurVec);
    if (blurMag > 0.048) blurVec *= (0.048 / blurMag);

    float3 baseAA = sampleFXAA(sceneTex, suv, u.texelSize);
    float3 col;
    if (blurMag > 0.0014 && worldMask > 0.08) {
        float3 acc = baseAA;
        float wSum = 1.0;
        float j = (hash21(in.position.xy) - 0.5) * 0.20;
        for (int i = -4; i <= 4; i++) {
            if (i == 0) continue;
            float fi = (float(i) + j) / 4.0;
            float2 buv = clamp(suv + blurVec * fi, 0.001, 0.999);
            float4 tap = sceneTex.sample(kPostLinearClamp, buv);
            float tw = smoothstep(0.14, 0.46, tap.a) * (1.0 - abs(fi) * 0.35);
            acc += tap.rgb * tw;
            wSum += tw;
        }
        float3 blurred = acc / max(wSum, 0.001);
        blurred.r = mix(blurred.r, sceneTex.sample(kPostLinearClamp, clamp(suv + fromCenter * ca, 0.0, 1.0)).r, 0.25);
        blurred.b = mix(blurred.b, sceneTex.sample(kPostLinearClamp, clamp(suv - fromCenter * ca, 0.0, 1.0)).b, 0.25);
        col = mix(baseAA, blurred, worldMask);
    } else {
        col.r = mix(baseAA.r, sceneTex.sample(kPostLinearClamp, clamp(suv + fromCenter * ca, 0.0, 1.0)).r, 0.50);
        col.g = baseAA.g;
        col.b = mix(baseAA.b, sceneTex.sample(kPostLinearClamp, clamp(suv - fromCenter * ca, 0.0, 1.0)).b, 0.50);
    }

    if (u.bloomStrength > 0.0001) {
        col += bloomTex.sample(kPostLinearClamp, suv).rgb * u.bloomStrength;
    }
    col += sampleGodRays(suv, centerRaw.a, godRaysTex, u.hasGodRays, u.texelSize);
    col += evalSunFlare(suv, u.sunUV, u.sunColor.w, u.sunColor.rgb, u.aspect, sceneTex);

    if (u.stormStreaks > 0.02) {
        float s1 = drivingSnowLayer(vUv, u.time + 11.0, 11.0, 11.0, 0.24, -0.34, u.aspect);
        float s2 = drivingSnowLayer(vUv, u.time + 47.0, 21.0, 17.0, 0.30, -0.26, u.aspect);
        float s3 = drivingSnowLayer(vUv, u.time + 83.0, 38.0, 24.0, 0.36, -0.41, u.aspect);
        float snow = clamp((s1 * 0.48 + s2 * 0.38 + s3 * 0.28) * u.stormStreaks, 0.0, 0.85);
        col = mix(col, float3(0.92, 0.96, 1.02), snow * 0.78);
    }

    if (u.faceShot > 0.005) {
        float2 dv = vUv - 0.5;
        float r = length(dv);
        float rimMask = smoothstep(0.28, 0.74, r);
        if (rimMask > 0.001) {
            float ang = atan2(dv.y, dv.x);
            float wisp = 0.58 + 0.42 * sin(ang * 5.0 + u.time * 3.8 + r * 6.5) * cos(ang * 3.0 - u.time * 2.4);
            float fgKeep = mix(0.35, 1.0, smoothstep(0.10, 0.42, centerRaw.a));
            float wash = u.faceShot * rimMask * (0.55 + 0.45 * wisp) * 0.36 * fgKeep;
            col = mix(col, float3(0.90, 0.94, 1.00), clamp(wash, 0.0, 0.28));
        }
    }

    // Lower goggle rim breath fog (in Metal UV, bottom rim is vUv.y -> 1.0)
    if (u.breath > 0.02) {
        float pulse = 0.5 + 0.5 * sin(u.time * 2.2);
        float lowMask = smoothstep(0.74, 1.0, vUv.y) * smoothstep(0.58, 0.14, abs(vUv.x - 0.5));
        col = mix(col, float3(0.84, 0.90, 0.96), lowMask * u.breath * pulse * 0.14);
    }

    if (u.lensWet > 0.02) {
        float rim = smoothstep(0.48, 0.78, length(vUv - 0.5)) * u.lensWet * 0.04;
        col = mix(col, float3(0.86, 0.92, 0.99), rim);
    }

    float vig = 1.0 - smoothstep(0.36, 0.88, length(vUv - 0.5)) * (u.barrel > 0.01 ? 0.22 : 0.13);
    col = acesFilm(col * u.exposure * vig);
    col = pow(col, float3(1.0 / 2.2));
    float dither = (hash21(in.position.xy) - 0.5) / 255.0;
    col = clamp(col + dither, 0.0, 1.0);
    return float4(col, 1.0);
}
