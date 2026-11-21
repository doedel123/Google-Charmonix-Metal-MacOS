import Foundation
import simd

/// High-level simulation coordinator combining `SkiPhysics`, `CameraController`, and `CourseManager`.
/// Conforms to `SimulationSubsystem` so it can be plugged directly into `SubsystemContainer.simulation`.
public final class GameSimulation: SimulationSubsystem {
    public let physics: SkiPhysics
    public let courses = CourseManager()

    public var terrainData: any TerrainQueryProtocol { physics.terrainData }
    public var state: SkierState { physics.state }
    public var avalanche: AvalancheState { physics.avalanche }
    public var events: [PhysicsEvent] {
        get { physics.events }
        set { physics.events = newValue }
    }
    public var replayFrames: [ReplayFrame] { physics.replayFrames }

    public var avalanchesEnabled: Bool {
        get { physics.avalanchesEnabled }
        set { physics.avalanchesEnabled = newValue }
    }
    public var lookYawOff: Float {
        get { physics.lookYawOff }
        set { physics.lookYawOff = newValue }
    }
    public var lookPitchOff: Float {
        get { physics.lookPitchOff }
        set { physics.lookPitchOff = newValue }
    }
    public var headSteerEnabled: Bool {
        get { physics.headSteerEnabled }
        set { physics.headSteerEnabled = newValue }
    }
    public var smoothCarve: Bool {
        get { physics.smoothCarve }
        set { physics.smoothCarve = newValue }
    }

    public var runState: RunState { courses.runState }
    public var activeCourse: CourseDefinition? { courses.activeCourse }
    public var activeZoneIndex: Int { courses.activeZoneIndex }

    public init(
        terrainData: any TerrainQueryProtocol,
        colliders: [PropCollider] = [],
        kickers: [WindLipKicker] = []
    ) {
        self.physics = SkiPhysics(terrainData: terrainData, colliders: colliders, kickers: kickers)
    }

    public func updateWorldObstacles(colliders: [PropCollider], kickers: [WindLipKicker]) {
        physics.colliders = colliders
        physics.kickers = kickers
    }

    public func spawn(zone: DropZone) {
        if let idx = terrainData.meta.dropZones.firstIndex(where: { $0.id == zone.id }) {
            _ = courses.selectDropZone(index: idx, terrainData: terrainData)
        } else {
            courses.setCourse(CourseDefinition(dropZone: zone, gates: []))
        }
        physics.spawn(zone: zone)
    }

    @discardableResult
    public func spawnAtZone(index: Int) -> DropZone? {
        guard let dz = courses.selectDropZone(index: index, terrainData: terrainData) else { return nil }
        physics.spawn(zone: dz)
        return dz
    }

    public func step(dt: Float, time: Float, input: InputState) {
        physics.step(dt: dt, time: time, input: input)
        courses.updateRunState(state: physics.state, dt: dt, started: true, paused: false)
    }

    public func stepSeconds(
        _ seconds: Float,
        keys: Set<String> = [],
        startTime: Float = 0,
        trackBuffer: (any SnowTracksSubsystem)? = nil,
        particles: (any ParticlesSubsystem)? = nil,
        preset: LightingPreset = LightingPresets.all[0]
    ) -> Float {
        let dt = ChamonixConstants.fixedPhysicsDt
        let steps = Int(roundf(seconds / dt))
        var t = startTime
        var input = InputState()
        input.activeKeys = keys
        for _ in 0..<steps {
            t += dt
            physics.step(dt: dt, time: t, input: input)
            courses.updateRunState(state: physics.state, dt: dt, started: true, paused: false)
            if let tb = trackBuffer, let pt = particles {
                CourseManager.flushPhysicsEvents(physics: physics, trackBuffer: tb, particles: pt)
            } else {
                physics.clearEvents()
            }
        }
        if let pt = particles {
            pt.update(
                dt: seconds,
                cameraPos: SIMD3<Float>(physics.state.x, physics.state.y + 1.6, physics.state.z),
                viewportHeightPx: 1080,
                fovDeg: 86,
                preset: preset,
                sunDir: preset.sunDirection,
                time: t
            )
        }
        return t
    }

    public func runDeterministicDemo(
        mode: String,
        trackBuffer: any SnowTracksSubsystem,
        particles: any ParticlesSubsystem,
        preset: LightingPreset,
        initialSimTime: Float = 0,
        cameraPos: SIMD3<Float> = .zero,
        viewportHeightPx: Float = 1080,
        fovDeg: Float = 86
    ) -> DemoScriptResult {
        CourseManager.runDeterministicScript(
            mode: mode,
            physics: physics,
            terrainData: terrainData,
            trackBuffer: trackBuffer,
            particles: particles,
            colliders: physics.colliders,
            preset: preset,
            initialSimTime: initialSimTime,
            cameraPos: cameraPos,
            viewportHeightPx: viewportHeightPx,
            fovDeg: fovDeg
        )
    }

    public func applyCamera(camera: inout CameraState, mode: CameraMode, time: Float, dt: Float) {
        physics.applyCamera(camera: &camera, mode: mode, time: time, dt: dt)
    }

    public func enforceCameraClearance(camera: inout CameraState, minClearance: Float, lockRoll: Bool) {
        physics.enforceCameraClearance(camera: &camera, minClearance: minClearance, lockRoll: lockRoll)
    }

    public func addHeadLookDelta(dYaw: Float, dPitch: Float) {
        physics.addHeadLookDelta(dYaw: dYaw, dPitch: dPitch)
    }

    public func resetHeadLook() {
        physics.resetHeadLook()
    }

    public func toggleSwitch() {
        physics.toggleSwitch()
    }

    public func triggerPoleStroke(side: Int) {
        physics.triggerPoleStroke(side: side)
    }

    public func triggerAvalanche() {
        physics.triggerAvalanche()
    }

    public func standUp() {
        physics.standUp()
    }

    public func clearEvents() {
        physics.clearEvents()
    }
}
