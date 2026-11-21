# Chamonix Ski Sim (macOS / Metal)

Native Swift + Metal macOS port of the Chamonix / Mont Blanc freeride ski simulator (originally built in Three.js / WebGL2).

## Prebuilt macOS App

A compiled Apple Silicon (`arm64`, macOS 14+) Release app bundle is included at:

```text
build/ChamonixSkiSim.app
```

Launch directly from Finder or Terminal:

```bash
open build/ChamonixSkiSim.app
```

> If macOS Gatekeeper warns about an ad-hoc signed app downloaded from GitHub, right-click `build/ChamonixSkiSim.app` → **Open** (or run `xattr -dr com.apple.quarantine build/ChamonixSkiSim.app`).

## Building & Testing in Xcode

Open the Xcode project:

```bash
open ChamonixSkiSim.xcodeproj
```

Or build and run the test suite from the command line:

```bash
xcodebuild -project ChamonixSkiSim.xcodeproj -scheme ChamonixSkiSim -configuration Release build
xcodebuild -project ChamonixSkiSim.xcodeproj -scheme ChamonixSkiSim -destination 'platform=macOS' test
```

If you add or remove files under `Sources/`, `Shaders/`, or `Tests/`, regenerate the Xcode project with `xcodegen`:

```bash
python3 scripts/generate_xcodeproj.py
```

## Architecture

- `Sources/Core/` — `MetalRenderer`, `ShaderTypes.h`, SIMD math, `LightingPresets`, `TextureLoader` (native ImageIO WebP decoding), `ShaderLibrary` (runtime Metal shader compilation via `MTLDevice.makeLibrary`), shared protocols & state.
- `Sources/Terrain/` + `Shaders/Terrain.metal`, `Sky.metal`, `SnowTracks.metal` — 16-bit MED-predicted EU-DEM decoding, 11-level clipmap + outer Alps mesh, procedural Rayleigh/Mie sky & valley clouds, dual-cascade snow track render targets.
- `Sources/Simulation/` — 120 Hz 4-point ski carving/skidding physics, poling, ollie/kicker air tricks, switch stance, slab avalanche dynamics, drop zones, action-cam kinematics, and 30 Hz replay ring buffer.
- `Sources/Rig/` + `Shaders/StandardPBR.metal` — Procedural twin-tip skis (`ski_topsheet.webp`), bindings, 3-buckle boots, articulated freeride skier, goat-leather gloves, carbon poles, and shadow proxies.
- `Sources/World/` + `Shaders/AlpineProps.metal`, `Particles.metal` — Instanced spruce trees, granite boulders, Aiguille du Midi station, wind-lip kickers, valley town lights, and volumetric snow/avalanche particles.
- `Sources/PostFX/` + `Shaders/PostOptics.metal` — 4× MSAA `rgba16Float` HDR pipeline, bloom, god-rays, radial motion blur with foreground skier mask, barrel distortion, and ACES tonemapping.
- `Sources/Audio/` — Real-time procedural `AVAudioEngine` sound synthesis (wind, edge carving, powder rush, ice chatter, pole plants, jumps/landings, avalanche).
- `Sources/App/` — Native SwiftUI + AppKit `MTKView` host, HUD overlays, drop-in start screen, settings, and CLI flags (`--autostart`, `--zone`, `--light`, `--cam`, `--demo`, `--capture-frame`).
- `Resources/assets/` — Bundled Chamonix DEM, auxiliary maps, and material WebP textures.
- `Tests/` — XCTest suite covering core math/textures, 1:1 JS DEM elevation parity, ski physics invariants, and Metal shader compilation.
