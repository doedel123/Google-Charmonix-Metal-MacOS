#include "ShaderTypes.h"

// MARK: - Rig-Specific Material & Auxiliary Uniforms

struct RigMaterialParams {
    float2 uvScale;
    float bumpScale;
    float hasRoughnessMap;
    float opacity;
    float isDropShadow;
    float pad0;
    float pad1;
};

struct BowWaveUniforms {
    float4x4 modelMatrix;
    float4x4 normalMatrix;
    float4 params0;     // x = flowDist, y = intensity, z = nestAmt, w = edgeBias
    float4 params1;     // x = skiRoll, y = sideSign, z = storm, w = sunVis
    float4 snowLitCol;
    float4 snowShadCol;
    float4 snowSSSCol;
};

struct ShadowRibbonUniforms {
    float4 tintOpacity; // rgb = uTint, w = uOpacity
    float emergeT;
    float pad0;
    float pad1;
    float pad2;
};

struct ShadowCatcherUniforms {
    float4 tintStrength; // rgb = uCatchTint, w = uCatchStrength
};

// MARK: - Vertex I/O Structs

struct PBRVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal   [[attribute(VertexAttributeNormal)]];
    float2 uv       [[attribute(VertexAttributeUV)]];
    float4 color    [[attribute(VertexAttributeColor)]];
};

struct PBRVertexOut {
    float4 clipPos     [[position]];
    float3 worldPos;
    float3 worldNormal;
    float3 viewPos;
    float3 viewNormal;
    float2 uv;
    float4 color;
    float3 localPos;
};

struct ShadowVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
};

struct ShadowVertexOut {
    float4 clipPos [[position]];
};

struct BowWaveVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal   [[attribute(VertexAttributeNormal)]];
    float2 uv       [[attribute(VertexAttributeUV)]];
    float4 bowData  [[attribute(VertexAttributeCustom0)]]; // xyz = aBow, w = aBaseY
};

struct BowWaveVertexOut {
    float4 clipPos     [[position]];
    float2 uv;
    float3 bow;
    float3 worldPos;
    float3 worldNormal;
};

struct ShadowRibbonVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
    float2 uv       [[attribute(VertexAttributeUV)]];
    float4 custom0  [[attribute(VertexAttributeCustom0)]]; // x = aLift
};

struct ShadowRibbonVertexOut {
    float4 clipPos [[position]];
    float2 uv;
    float lift;
};

struct ShadowCatcherVertexIn {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal   [[attribute(VertexAttributeNormal)]];
    float2 catchUv  [[attribute(VertexAttributeUV)]];
};

struct ShadowCatcherVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    float2 catchUv;
    float sunLit;
};

// MARK: - Noise & Helper Functions (1:1 port of ski_rig.js GLSL)

static inline float skiHash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

static inline float skiVnoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(skiHash21(i), skiHash21(i + float2(1.0, 0.0)), u.x),
        mix(skiHash21(i + float2(0.0, 1.0)), skiHash21(i + float2(1.0, 1.0)), u.x),
        u.y
    );
}

static inline float sampleShadowPCF(
    depth2d<float> shadowMap,
    float3 worldPos,
    float3 worldNormal,
    float3 sunDir,
    float4x4 lightViewProj
) {
    float4 lp = lightViewProj * float4(worldPos, 1.0);
    if (abs(lp.w) < 1e-5) return 1.0;
    float3 ndc = lp.xyz / lp.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    constexpr sampler shadowSamp(
        coord::normalized,
        address::clamp_to_edge,
        filter::linear,
        compare_func::less_equal
    );

    float ndl = max(0.0, dot(normalize(worldNormal), normalize(sunDir)));
    float bias = mix(0.0045, 0.0012, ndl);
    float refDepth = ndc.z - bias;

    float2 texel = float2(1.0 / 1024.0);
    float sum = 0.0;
    for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
            float2 suv = clamp(uv + float2(float(ox), float(oy)) * texel * 1.35, float2(0.001), float2(0.999));
            sum += shadowMap.sample_compare(shadowSamp, suv, refDepth);
        }
    }
    return sum / 9.0;
}

static inline float3 perturbNormalBump(
    float3 N,
    float3 viewPos,
    float2 uv,
    texture2d<float> bumpTex,
    sampler texSamp,
    float bumpScale
) {
    float2 dSTdx = dfdx(uv);
    float2 dSTdy = dfdy(uv);
    float h0 = bumpTex.sample(texSamp, uv).r;
    float scale = bumpScale * 0.15;
    float dBx = (bumpTex.sample(texSamp, uv + dSTdx).r - h0) * scale;
    float dBy = (bumpTex.sample(texSamp, uv + dSTdy).r - h0) * scale;

    float3 dPosX = dfdx(viewPos);
    float3 dPosY = dfdy(viewPos);
    float3 r1 = cross(dPosY, N);
    float3 r2 = cross(N, dPosX);
    float det = dot(dPosX, r1);
    if (abs(det) < 1e-10) {
        return N;
    }
    float3 surfGrad = (det >= 0.0 ? 1.0 : -1.0) * (dBx * r1 + dBy * r2);
    return normalize(abs(det) * N - surfGrad);
}

// MARK: - 1. Directional Shadow Map Pass

vertex ShadowVertexOut rig_shadow_vertex(
    ShadowVertexIn in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ModelUniforms& model [[buffer(BufferIndexModelUniforms)]]
) {
    ShadowVertexOut out;
    float4 wp = model.modelMatrix * float4(in.position, 1.0);
    out.clipPos = frame.lightViewProjectionMatrix * wp;
    return out;
}

fragment void rig_shadow_fragment() {
    // Depth-only pass
}

// MARK: - 2. Standard PBR + Foreground Bounce + Ski Powder Dissolve

vertex PBRVertexOut rig_pbr_vertex(
    PBRVertexIn in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ModelUniforms& model [[buffer(BufferIndexModelUniforms)]]
) {
    PBRVertexOut out;
    float4 wp = model.modelMatrix * float4(in.position, 1.0);
    float4 vp = frame.viewMatrix * wp;
    float3 wn = normalize((model.normalMatrix * float4(in.normal, 0.0)).xyz);
    float3 vn = normalize((frame.viewMatrix * float4(wn, 0.0)).xyz);

    out.clipPos = frame.projectionMatrix * vp;
    out.worldPos = wp.xyz;
    out.worldNormal = wn;
    out.viewPos = vp.xyz;
    out.viewNormal = vn;
    out.uv = in.uv;
    out.color = in.color;
    out.localPos = in.position;
    return out;
}

fragment float4 rig_pbr_fragment(
    PBRVertexOut in [[stage_in]],
    bool isFrontFace [[front_facing]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ModelUniforms& model [[buffer(BufferIndexModelUniforms)]],
    constant RigMaterialParams& mat [[buffer(BufferIndexMaterialUniforms)]],
    texture2d<float> albedoTex [[texture(TextureIndexAlbedo)]],
    texture2d<float> bumpTex [[texture(TextureIndexNormal)]],
    texture2d<float> roughTex [[texture(TextureIndexRoughness)]],
    depth2d<float> shadowMap [[texture(TextureIndexShadowMap)]]
) {
    constexpr sampler repeatSamp(coord::normalized, address::repeat, filter::linear, mip_filter::linear, max_anisotropy(8));
    constexpr sampler clampSamp(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);

    float isSki = model.fgBounceColor.w;
    float2 scaledUv = in.uv * mat.uvScale;

    // Drop shadow blob mode (groundShadow)
    if (mat.isDropShadow > 0.5) {
        float4 texSample = albedoTex.sample(clampSamp, in.uv);
        float alpha = texSample.a * mat.opacity;
        if (alpha < 0.005) discard_fragment();
        return float4(texSample.rgb, alpha);
    }

    // Dust clump on front ski shovel (translucent matte snow)
    float hasAlbedo = model.snowShadowColor.w;
    float3 texAlb = float3(1.0);
    if (hasAlbedo > 0.5) {
        if (isSki > 0.5) {
            texAlb = albedoTex.sample(clampSamp, float2(in.uv.x, 1.0 - in.uv.y)).rgb;
        } else {
            texAlb = albedoTex.sample(repeatSamp, scaledUv).rgb;
        }
    }
    float3 albedo = model.baseColor.rgb * in.color.rgb * texAlb;

    float roughness = model.pbrParams.x;
    if (mat.hasRoughnessMap > 0.5) {
        roughness *= roughTex.sample(repeatSamp, scaledUv).g;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    float metalness = clamp(model.pbrParams.y, 0.0, 1.0);
    float emissive = model.pbrParams.z;

    float3 N = normalize(in.worldNormal);
    if (!isFrontFace) N = -N;

    float3 viewN = normalize(in.viewNormal);
    if (!isFrontFace) viewN = -viewN;

    // Apply bump perturbation in view space & transform back to world space for consistent BRDF + rim
    if (model.snowSSSColor.w > 0.5 && mat.bumpScale > 0.0001) {
        float3 bumpedViewN = perturbNormalBump(viewN, in.viewPos, scaledUv, bumpTex, repeatSamp, mat.bumpScale);
        viewN = bumpedViewN;
        float3x3 invView3 = float3x3(
            float3(frame.viewMatrix[0][0], frame.viewMatrix[1][0], frame.viewMatrix[2][0]),
            float3(frame.viewMatrix[0][1], frame.viewMatrix[1][1], frame.viewMatrix[2][1]),
            float3(frame.viewMatrix[0][2], frame.viewMatrix[1][2], frame.viewMatrix[2][2])
        );
        N = normalize(invView3 * bumpedViewN);
    }

    float3 V = normalize(frame.cameraPosition.xyz - in.worldPos);
    float3 L = normalize(frame.sunDirection.xyz);
    float3 H = normalize(V + L);

    float NdotV = max(dot(N, V), 1e-4);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Cook-Torrance GGX PBR
    float3 F0 = mix(float3(0.04), albedo, metalness);
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * denom * denom + 1e-6);

    float k = ((roughness + 1.0) * (roughness + 1.0)) * 0.125;
    float gV = NdotV / (NdotV * (1.0 - k) + k);
    float gL = NdotL / (NdotL * (1.0 - k) + k);
    float G = gV * gL;

    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    float3 specBRDF = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);
    float3 kD = (1.0 - F) * (1.0 - metalness);
    float3 diffBRDF = kD * albedo / 3.14159265;

    // Self-shadowing via directional shadow map + mountain sun visibility
    float isStorm = frame.skyHorizon.w;
    float mtnSunVis = frame.avalancheDirWidth.w;
    float shadowFactor = (mat.opacity < 0.99) ? 1.0 : sampleShadowPCF(shadowMap, in.worldPos, N, L, frame.lightViewProjectionMatrix);
    float directScale = (isStorm > 0.5 ? 0.25 : mtnSunVis) * shadowFactor;
    float3 sunRadiance = frame.sunColor.rgb * frame.sunColor.w * directScale;
    float3 directLight = (diffBRDF + specBRDF) * sunRadiance * NdotL;

    // Alpine sky + snowfield hemisphere ambient + specular environment reflection (vital for toric goggles & metal buckles!)
    float skyHemi = clamp(0.42 + 0.58 * N.y, 0.16, 1.0);
    float3 hemiFill = mix(frame.groundBounceColor.rgb * 0.68, frame.shadowSkyColor.rgb, skyHemi) * frame.shadowSkyColor.w;
    float3 R = reflect(-V, N);
    float horizonBand = smoothstep(-0.08, 0.14, R.y);
    float3 upperSky = mix(frame.skyHorizon.rgb, frame.skyZenith.rgb, pow(clamp(R.y, 0.0, 1.0), 0.65));
    float3 lowerSnow = frame.groundBounceColor.rgb * 1.45;
    float3 envReflCol = mix(lowerSnow, upperSky, horizonBand);
    // Crisp mountain-ridge contrast glint for mirror goggles & polished metal
    float mtnRidge = exp(-pow((R.y - 0.03) / 0.055, 2.0)) * (1.0 - roughness);
    envReflCol = mix(envReflCol, frame.shadowSkyColor.rgb * 0.55, mtnRidge * 0.35 * metalness);
    // Add crisp solar glint reflection on smooth metallic/lacquered surfaces (goggles, steel edges, ski topsheet)
    float sunRefl = pow(max(0.0, dot(R, L)), mix(8.0, 160.0, 1.0 - roughness));
    envReflCol += frame.sunColor.rgb * frame.sunColor.w * directScale * sunRefl * (0.28 + 0.72 * (1.0 - roughness));

    float3 F_env = F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(1.0 - NdotV, 5.0);
    float3 ambientDiff = kD * albedo * hemiFill * 0.85;
    float3 ambientSpec = envReflCol * F_env * mix(1.0, 0.22, roughness * roughness);

    float3 color = directLight + ambientDiff + ambientSpec + albedo * emissive;

    // Exact foreground snow-bounce, rim, velvet sheen & dynamic frost (skip for translucent dustMesh)
    if (mat.opacity > 0.99) {
        float3 fgUpV = normalize((frame.viewMatrix * float4(0.0, 1.0, 0.0, 0.0)).xyz);
        float3 fgV = normalize(-in.viewPos);
        float3 fgN = normalize(viewN);
        float fgDown = clamp(dot(fgN, -fgUpV) * 0.55 + 0.45, 0.0, 1.0);
        float fgUpHem = clamp(dot(fgN, fgUpV) * 0.5 + 0.5, 0.0, 1.0);
        float fgNdotV = max(0.0, dot(fgN, fgV));

        if (isSki > 0.5) {
            float fgRim = pow(1.0 - fgNdotV, 2.75);
            float3 fgLift = model.fgBounceColor.rgb * (fgDown * 0.16 + 0.036) * model.fgParams.z
                          + model.fgRimColor.rgb * (fgRim * 0.22 * model.fgParams.z);
            color += fgLift * (0.28 + 0.72 * albedo + float3(0.18));

            float powSink = model.skiPowderParams.x;
            float edgeRoll = model.skiPowderParams.y;
            float flowDist = model.skiPowderParams.z;
            float powSunVis = model.skiPowderParams.w;
            float sideSign = model.fgRimColor.w;
            float powStorm = model.snowLitColor.w;

            float emergeAlpha = 1.0;
            if (powSink > 0.02) {
                float acrossSigned = in.uv.x * 2.0 - 1.0;
                float across = abs(acrossSigned);
                float t = in.uv.y;

                float2 pMet = float2((in.uv.x - 0.5) * 0.15 + sideSign * 0.43, in.uv.y * 1.84 + flowDist * 0.82);
                float n1 = skiVnoise(pMet * 7.2 + float2(2.3, -1.7));
                float n2 = skiVnoise(pMet * 19.5 + float2(5.1, 3.4));
                float n3 = skiVnoise(pMet * 46.0 - float2(1.9, 2.8));

                float sinkNorm = clamp(powSink, 0.0, 1.0);
                float emergeT = mix(0.28, 0.80, sinkNorm);
                float isWall = smoothstep(0.92, 0.995, across) * (1.0 - smoothstep(0.002, 0.012, in.localPos.y));
                float edgeArch = pow(across, 1.55) * 0.095 + isWall * 0.085;
                float edgeTilt = clamp(acrossSigned * edgeRoll * 0.145, -0.09, 0.14);
                float ripple = (n1 - 0.5) * 0.068 + (n2 - 0.5) * 0.028;

                float subDepth = (emergeT + edgeArch + edgeTilt + ripple) - t;
                float frostWeight = smoothstep(0.0, 0.35, sinkNorm);
                float softSpan = mix(0.18, 0.27, sinkNorm);
                float vis = 1.0 - smoothstep(0.0, softSpan, subDepth);
                emergeAlpha = mix(1.0, vis, frostWeight);
                if (emergeAlpha < 0.012) discard_fragment();

                float3 snowLitCol = model.snowLitColor.rgb;
                float3 snowShadCol = model.snowShadowColor.rgb;
                float3 snowSSSCol = model.snowSSSColor.rgb;

                float3 powVeil = mix(snowShadCol, snowLitCol, clamp(0.45 + 0.45 * n1, 0.0, 1.0));
                float frostDust = smoothstep(-0.14, 0.02, subDepth) * (1.0 - smoothstep(0.04, 0.18, subDepth))
                                * (0.14 + 0.22 * n2 + 0.16 * pow(across, 1.8)) * frostWeight;
                float shallowScatter = smoothstep(-0.02, softSpan * 0.72, subDepth) * 0.58 * frostWeight;
                float3 coolScatter = mix(powVeil, snowShadCol * 0.85 + snowSSSCol * 0.40, 0.32 * (1.0 - powStorm * 0.7));
                float veilMix = clamp(max(shallowScatter, frostDust), 0.0, 0.72);
                float spark = pow(max(0.0, n3 * n2), 5.6) * (1.0 - powStorm * 0.85) * powSunVis * frostDust * 2.2;
                color = mix(color, coolScatter, veilMix) + snowLitCol * spark * 0.35;
                float slitAO = smoothstep(-0.025, 0.015, subDepth) * (1.0 - smoothstep(0.025, 0.085, subDepth));
                color *= (1.0 - slitAO * 0.08 * frostWeight);
            }
            return float4(color, emergeAlpha);
        } else {
            float fgRim = pow(1.0 - fgNdotV, 2.45);
            float fgVelvet = pow(1.0 - fgNdotV, 1.85) * model.fgParams.y;
            float3 fgBounce = model.fgBounceColor.rgb * (fgDown * 0.24 + 0.068)
                            + model.fgRimColor.rgb * (fgUpHem * 0.095 + fgRim * 0.24 + fgVelvet * 0.36);
            color += fgBounce * model.fgParams.z * (0.24 + 0.76 * albedo + float3(0.14));
            if (model.fgParams.x > 0.001) {
                float fgUp = smoothstep(0.32, 0.85, dot(fgN, fgUpV));
                float fgFrostMix = clamp(fgUp * model.fgParams.w * model.fgParams.x, 0.0, 0.32);
                color = mix(color, model.fgRimColor.rgb * 0.82, fgFrostMix);
            }
            return float4(color, model.pbrParams.w);
        }
    } else {
        return float4(color, mat.opacity);
    }
}

// MARK: - 3. Dynamic Powder Bow-Wave Snow Wash

vertex BowWaveVertexOut bow_wave_vertex(
    BowWaveVertexIn in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant BowWaveUniforms& u [[buffer(BufferIndexModelUniforms)]]
) {
    BowWaveVertexOut out;
    float3 aBow = in.bowData.xyz;
    float aBaseY = in.bowData.w;
    float uFlowDist = u.params0.x;
    float uIntensity = u.params0.y;
    float uNestAmt = u.params0.z;
    float uEdgeBias = u.params0.w;
    float uSkiRoll = u.params1.x;
    float uSideSign = u.params1.y;

    float surge = clamp(uIntensity, 0.0, 1.25);
    float bowAmt = max(surge, uNestAmt * 0.44);
    float3 p = in.position;

    float wave1 = sin(in.uv.y * 14.0 + uFlowDist * 1.85 + aBow.y * 5.0 + uSideSign * 1.7);
    float wave2 = cos(in.uv.y * 23.0 + uFlowDist * 2.40 - in.uv.x * 9.0);
    float swell = (0.08 + 0.92 * surge) * aBow.z;
    float sidePush = in.uv.x * 2.0 - 1.0;

    p.x *= mix(0.56, 1.15, bowAmt);
    p.x += (-uEdgeBias * 0.14) * aBow.y * aBow.z * surge;
    float waveH = p.y - aBaseY;
    float yScale = mix(mix(0.12, 0.22, uNestAmt), 1.05, clamp(surge, 0.0, 1.0));
    p.y = aBaseY + waveH * yScale + (wave1 * 0.020 + wave2 * 0.011) * aBow.y * swell * (1.0 + aBow.x * 0.45);

    float rClamp = clamp(uSkiRoll, -0.96, 0.96);
    float wingWeight = smoothstep(0.16, 0.62, aBow.y);
    float biteGain = (p.x * (-rClamp)) > 0.0 ? 1.08 : 0.92;
    p.y += (-p.x * tan(rClamp)) * biteGain * wingWeight;
    p.z += wave1 * 0.022 * aBow.y * swell;

    float4 wp = u.modelMatrix * float4(p, 1.0);
    out.worldPos = wp.xyz;
    float3 localN = normalize(in.normal + float3(sidePush * 0.35, 0.55, 0.18));
    out.worldNormal = normalize((u.normalMatrix * float4(localN, 0.0)).xyz);
    out.uv = in.uv;
    out.bow = aBow;
    out.clipPos = frame.viewProjectionMatrix * wp;
    return out;
}

fragment float4 bow_wave_fragment(
    BowWaveVertexOut in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant BowWaveUniforms& u [[buffer(BufferIndexModelUniforms)]]
) {
    float uFlowDist = u.params0.x;
    float uIntensity = u.params0.y;
    float uNestAmt = u.params0.z;
    float uSideSign = u.params1.y;
    float uStorm = u.params1.z;
    float uSunVis = u.params1.w;

    float spray = clamp(uIntensity, 0.0, 1.0);
    float bowAmt = max(spray, uNestAmt * 0.52);
    if (bowAmt < 0.02) discard_fragment();
    if (in.bow.x > 0.5 && uIntensity < 0.12) discard_fragment();

    float across = abs(in.uv.x * 2.0 - 1.0);
    float edgeFade = 1.0 - smoothstep(0.62, 0.99, across);
    float longFade = smoothstep(0.04, 0.22, in.uv.y) * (1.0 - smoothstep(0.84, 0.99, in.uv.y));
    float wingFocus = smoothstep(0.26, 0.58, across);
    float layerGain = in.bow.x > 0.5 ? spray : clamp(bowAmt, 0.0, 1.0);
    float envAlpha = edgeFade * longFade * wingFocus * layerGain;
    if (envAlpha < 0.02) discard_fragment();

    float2 flowUv = float2(in.uv.x * 3.6 + uSideSign * 1.9, in.uv.y * 9.8 + uFlowDist * 0.85);
    float n1 = skiVnoise(flowUv);
    float n2 = skiVnoise(flowUv * 2.15 + float2(3.7 + n1 * 0.75, uFlowDist * 0.35));
    float cloud = smoothstep(0.22, 0.82, n1 * 0.62 + n2 * 0.44);

    float layerAlpha = mix(mix(0.24, 0.56, clamp(spray * 1.8, 0.0, 1.0)), 0.38, in.bow.x);
    float alpha = cloud * envAlpha * layerAlpha;
    if (alpha < 0.015) discard_fragment();

    float3 N = normalize(in.worldNormal);
    float3 L = normalize(frame.sunDirection.xyz);
    float3 V = normalize(frame.cameraPosition.xyz - in.worldPos);
    float directVis = uSunVis * (1.0 - uStorm * 0.85);
    float sunSculpt = clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0);
    float crestMix = mix(0.45 + 0.40 * n1, clamp(0.35 + 0.45 * n1 + 0.28 * sunSculpt, 0.0, 1.0), spray);
    float3 baseCol = mix(u.snowShadCol.rgb, u.snowLitCol.rgb, crestMix) * (0.94 + 0.10 * (n2 - 0.5));
    float innerRim = (1.0 - smoothstep(0.30, 0.62, across)) * (1.0 - uStorm * 0.75);
    baseCol = mix(baseCol, u.snowShadCol.rgb * 0.78 + u.snowSSSCol.rgb * 0.35, innerRim * 0.38);
    float backScatter = pow(max(0.0, dot(-V, L)), 2.8) * directVis * spray;
    float rim = smoothstep(0.35, 0.75, across) * (1.0 - cloud * 0.5) * spray;
    baseCol += frame.sunColor.rgb * rim * (0.10 + 0.34 * backScatter) * directVis;

    return float4(baseCol, clamp(alpha, 0.0, 0.72));
}

// MARK: - 4. Multiplicative Ski Contact Shadow Ribbon

vertex ShadowRibbonVertexOut shadow_ribbon_vertex(
    ShadowRibbonVertexIn in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]]
) {
    ShadowRibbonVertexOut out;
    out.clipPos = frame.viewProjectionMatrix * float4(in.position, 1.0);
    out.uv = in.uv;
    out.lift = in.custom0.x;
    return out;
}

fragment float4 shadow_ribbon_fragment(
    ShadowRibbonVertexOut in [[stage_in]],
    constant ShadowRibbonUniforms& u [[buffer(BufferIndexModelUniforms)]]
) {
    float absU = abs(in.uv.x);
    float t = in.uv.y;
    float noseFade = 1.0 - smoothstep(0.92, 1.0, t);
    float tailFade = smoothstep(0.0, 0.08, t);
    float emergeFade = u.emergeT > 0.0 ? smoothstep(u.emergeT - 0.06, u.emergeT + 0.12, t) : 1.0;

    float sharpness = mix(3.4, 1.55, clamp(in.lift * 7.5, 0.0, 1.0));
    float across = exp(-pow(absU * 1.18, 2.0) * sharpness) * (1.0 - smoothstep(0.58, 1.0, absU));
    float tipSoft = mix(1.0, 0.68, clamp(in.lift * 6.5, 0.0, 1.0));
    float m = across * noseFade * tailFade * emergeFade * tipSoft * u.tintOpacity.w;
    if (m < 0.004) discard_fragment();

    float3 col = mix(float3(1.0), u.tintOpacity.rgb, clamp(m, 0.0, 1.0));
    return float4(col, 1.0);
}

// MARK: - 5. DEM-Snapped Multiplicative PCF Shadow Catcher

vertex ShadowCatcherVertexOut shadow_catcher_vertex(
    ShadowCatcherVertexIn in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]]
) {
    ShadowCatcherVertexOut out;
    out.clipPos = frame.viewProjectionMatrix * float4(in.position, 1.0);
    out.worldPos = in.position;
    out.catchUv = in.catchUv;
    out.sunLit = smoothstep(-0.05, 0.10, dot(normalize(in.normal), normalize(frame.sunDirection.xyz)));
    return out;
}

fragment float4 shadow_catcher_fragment(
    ShadowCatcherVertexOut in [[stage_in]],
    constant FrameUniforms& frame [[buffer(BufferIndexFrameUniforms)]],
    constant ShadowCatcherUniforms& u [[buffer(BufferIndexModelUniforms)]],
    depth2d<float> shadowMap [[texture(TextureIndexShadowMap)]]
) {
    float4 lp = frame.lightViewProjectionMatrix * float4(in.worldPos, 1.0);
    if (abs(lp.w) < 1e-5) discard_fragment();
    float3 ndc = lp.xyz / lp.w;
    float2 suv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (suv.x <= 0.002 || suv.x >= 0.998 || suv.y <= 0.002 || suv.y >= 0.998 || ndc.z <= 0.0 || ndc.z >= 1.0) {
        discard_fragment();
    }

    constexpr sampler shadowSamp(
        coord::normalized,
        address::clamp_to_edge,
        filter::linear,
        compare_func::less_equal
    );

    float2 texel = float2(1.0 / 1024.0);
    float refDepth = ndc.z - 0.0022;
    float shadowLit = 0.0;
    for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
            float2 tapUv = clamp(suv + float2(float(ox), float(oy)) * texel * 1.4, float2(0.001), float2(0.999));
            shadowLit += shadowMap.sample_compare(shadowSamp, tapUv, refDepth);
        }
    }
    shadowLit /= 9.0;

    float fadeX = smoothstep(0.0, 0.14, in.catchUv.x) * (1.0 - smoothstep(0.86, 1.0, in.catchUv.x));
    float fadeY = smoothstep(0.0, 0.06, in.catchUv.y) * (1.0 - smoothstep(0.88, 1.0, in.catchUv.y));
    float borderFade = fadeX * fadeY * in.sunLit;
    float shOcc = clamp((1.0 - shadowLit) * u.tintStrength.w * borderFade, 0.0, 1.0);
    if (shOcc < 0.004) discard_fragment();

    float3 col = mix(float3(1.0), u.tintStrength.rgb, shOcc);
    return float4(col, 1.0);
}
