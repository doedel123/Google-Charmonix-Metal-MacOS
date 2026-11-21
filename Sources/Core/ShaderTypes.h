#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#include <metal_stdlib>
using namespace metal;
typedef float2 simd_float2;
typedef float3 simd_float3;
typedef float4 simd_float4;
typedef float4x4 simd_float4x4;
#else
#import <Foundation/Foundation.h>
#import <simd/simd.h>
#endif

// Coordinate Convention:
//   - Right-Handed, Y-up (+X East, +Y Up, -Z North) matching Three.js 1:1
//   - CCW front-face winding (MTLWindingCounterClockwise), back-face culled unless double-sided
//   - Metal NDC depth [0, 1]
//   - HDR Scene RGBA16Float Alpha Mask:
//       0.00 = Foreground skier/skis/poles/gloves (excluded from motion blur)
//       0.55 = World geometry (terrain, props, trees, rocks, seracs, particles)
//       1.00 = Sky dome (emits god-rays & solar lens flare)

typedef NS_ENUM(NSInteger, VertexAttribute) {
    VertexAttributePosition = 0,
    VertexAttributeNormal   = 1,
    VertexAttributeUV       = 2,
    VertexAttributeColor    = 3,
    VertexAttributeCustom0  = 4,
};

typedef NS_ENUM(NSInteger, BufferIndex) {
    BufferIndexVertices          = 0,
    BufferIndexInstances         = 1,
    BufferIndexFrameUniforms     = 2,
    BufferIndexModelUniforms     = 3,
    BufferIndexMaterialUniforms  = 4,
    BufferIndexSubsystemUniforms = 5,
};

typedef NS_ENUM(NSInteger, TextureIndex) {
    TextureIndexCoreDem        = 0,
    TextureIndexCoreAux1       = 1,
    TextureIndexCoreAux2       = 2,
    TextureIndexOuterDem       = 3,
    TextureIndexOuterAux1      = 4,
    TextureIndexOuterAux2      = 5,
    TextureIndexGraniteAlbedo  = 6,
    TextureIndexGraniteNormal  = 7,
    TextureIndexSnowNormal     = 8,
    TextureIndexSnowCavity     = 9,
    TextureIndexIceNormal      = 10,
    TextureIndexTrackTex       = 11,
    TextureIndexMacroTrackTex  = 12,
    TextureIndexShadowMap      = 13,
    TextureIndexAlbedo         = 14,
    TextureIndexNormal         = 15,
    TextureIndexRoughness      = 16,
    TextureIndexSceneHDR       = 17,
    TextureIndexBloom          = 18,
    TextureIndexGodRays        = 19,
};

typedef struct {
    simd_float3 position;
    simd_float3 normal;
    simd_float2 uv;
    simd_float4 color;
} StandardVertex;

typedef struct {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewProjectionMatrix;
    simd_float4x4 lightViewProjectionMatrix;
    simd_float4 cameraPosition;     // xyz = camPos, w = time
    simd_float4 sunDirection;       // xyz = normalized sunDir, w = sunElev (rad)
    simd_float4 sunColor;           // rgb = sunColor, w = sunIntensity
    simd_float4 skyZenith;          // rgb = skyZenith, w = cloudCoverage
    simd_float4 skyHorizon;         // rgb = skyHorizon, w = storm (0 or 1)
    simd_float4 shadowSkyColor;     // rgb = shadowSkyColor, w = shadowFillStrength
    simd_float4 groundBounceColor;  // rgb = groundBounceColor, w = twilight
    simd_float4 sssColor;           // rgb = sssColor, w = sparkleIntensity
    simd_float4 fogColor;           // rgb = fogColor, w = fogDensity
    simd_float4 cloudParams;        // x = valleyCloudAlt, y = valleyCloudDensity, z = cloudShadowStrength, w = coreSize
    simd_float4 horizonBlend;       // xyzw = 4-azimuth horizon blend weights
    simd_float4 screenParams;       // x = viewportWidth, y = viewportHeight, z = fovYRad, w = outerSize
    simd_float4 avalancheParams;    // x = activeStrength, y = originX, z = originZ, w = frontDist
    simd_float4 avalancheDirWidth;  // x = dirX, y = dirZ, z = width, w = mtnSunVis
    simd_float4 trackParams;        // x = originX, y = originZ, z = worldSpan, w = res
    simd_float4 macroTrackParams;   // x = macroOriginX, y = macroOriginZ, z = macroSpan, w = macroRes
    simd_float4 extraParams;        // x = snowfallRate, y = coreHalf (6400), z = outerHalf (24000), w = maxElev (5000)
} FrameUniforms;

typedef struct {
    simd_float4x4 modelMatrix;
    simd_float4x4 normalMatrix;     // inverse-transpose of upper-left 3x3
    simd_float4 baseColor;
    simd_float4 pbrParams;          // x = roughness, y = metalness, z = emissive, w = alphaMask (0.0 fg, 0.55 world)
    simd_float4 fgParams;           // x = frostScale, y = sheenAmt, z = bounceAmt, w = fgFrostGlobal
    simd_float4 fgBounceColor;      // rgb = bounce, w = isSki (1.0 or 0.0)
    simd_float4 fgRimColor;         // rgb = rim, w = sideSign (-1 L, +1 R)
    simd_float4 skiPowderParams;    // x = powSink, y = edgeRoll, z = flowDist, w = powSunVis
    simd_float4 snowLitColor;       // rgb = snowLit, w = powStorm
    simd_float4 snowShadowColor;    // rgb = snowShadow, w = hasAlbedoTex
    simd_float4 snowSSSColor;       // rgb = snowSSS, w = hasNormalTex
} ModelUniforms;

typedef struct {
    simd_float4x4 instanceMatrix;
    simd_float4 propParams;         // x = expose, y = cavityAO, z = strataSeed, w = variant
} PropInstanceData;

typedef struct {
    simd_float4 posRadius;          // xyz = world pos, w = crownRadius
    simd_float4 normalHeight;       // xyz = terrain normal, w = height
    simd_float4 extra;              // x = opacity, y = kind (0=tree,1=rock,2=serac,3=stake), z = seed, w = 0
} ShadowInstanceData;

typedef struct {
    simd_float2 snapCenter;
    simd_float2 childCenter;
    float gridStep;
    float gridN;
    float childHalf;
    float isOuterMesh;
    float level;
    float pad0;
    float pad1;
    float pad2;
} TerrainLevelUniforms;

typedef struct {
    simd_float2 shiftUV;
    float decay;
    float singleMode;
    simd_float2 lA;
    simd_float2 lB;
    simd_float2 rA;
    simd_float2 rB;
    float radiusUV;
    float depth;
    float berm;
    float iceGlint;
    float skewSide;
    float pad0;
} SnowTrackUniforms;

typedef struct {
    simd_float4 posSize;            // xyz = world pos, w = size
    simd_float4 colorAlpha;         // x = alpha, y = rot, z = kind (0=powder,1=avy,2=spindrift,3=flake), w = unused
} ParticleVertexData;

typedef struct {
    simd_float2 texelSize;
    simd_float2 sunUV;
    simd_float2 horizonUV;
    simd_float2 angVel;
    simd_float4 sunColor;           // rgb = sunColor, w = sunVis
    float exposure;
    float bloomStrength;
    float barrel;
    float speedFactor;
    float rollVel;
    float godRayVis;
    float godRayStrength;
    float hasGodRays;
    float faceShot;
    float lensWet;
    float breath;
    float stormStreaks;
    float time;
    float aspect;
    float threshold;
    float pad0;
} PostOpticsUniforms;

#endif /* ShaderTypes_h */
