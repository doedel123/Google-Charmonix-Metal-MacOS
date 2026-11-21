import AVFoundation
import Foundation
import os
import simd

/// Zero-asset procedural spatial audio soundscape for off-piste freeride skiing:
///  - High-altitude wind rush + jacket flutter scaled by speed & altitude
///  - Stereo L/R panned edge-pressure carving bed (Cold Powder / Wind Pack / Blue Ice / Granite)
///  - Instant airborne silence drop when skis leave snow
///  - Metallic-carbon binding flex click + snow release snap on pop
///  - Stereo-panned pole plants, Doppler tree whooshes, boot shell creaks, distant glacier groans
///  - High-altitude cardio breathing & chest-thumping slab avalanche roar
///  - Lock-free / non-blocking audio render thread; gracefully degrades to silent no-op if no audio device exists.
public final class AlpineAudioEngine: AudioSubsystem, @unchecked Sendable {

    private enum FilterKind {
        case lowpass
        case bandpass
        case highpass
    }

    private enum OscWaveform {
        case sine
        case triangle
        case sawtooth
    }

    private struct BiquadFilter {
        var kind: FilterKind
        var q: Float
        var b0: Float = 1
        var b1: Float = 0
        var b2: Float = 0
        var a1: Float = 0
        var a2: Float = 0
        var z1: Float = 0
        var z2: Float = 0
        var lastFreq: Float = -1

        init(kind: FilterKind, freq: Float, q: Float, sampleRate: Float) {
            self.kind = kind
            self.q = q
            updateCoefficients(freq: freq, sampleRate: sampleRate)
        }

        mutating func updateCoefficients(freq: Float, sampleRate: Float) {
            guard abs(freq - lastFreq) > 0.25 else { return }
            lastFreq = freq
            let fs = max(8000.0, sampleRate)
            let f = min(max(20.0, freq), fs * 0.45)
            let w0 = 2.0 * Float.pi * f / fs
            let cosW0 = cosf(w0)
            let sinW0 = sinf(w0)
            let alpha = sinW0 / (2.0 * max(0.05, q))

            let a0: Float
            switch kind {
            case .lowpass:
                b0 = (1.0 - cosW0) * 0.5
                b1 = 1.0 - cosW0
                b2 = (1.0 - cosW0) * 0.5
                a0 = 1.0 + alpha
                a1 = -2.0 * cosW0
                a2 = 1.0 - alpha
            case .highpass:
                b0 = (1.0 + cosW0) * 0.5
                b1 = -(1.0 + cosW0)
                b2 = (1.0 + cosW0) * 0.5
                a0 = 1.0 + alpha
                a1 = -2.0 * cosW0
                a2 = 1.0 - alpha
            case .bandpass:
                b0 = alpha
                b1 = 0.0
                b2 = -alpha
                a0 = 1.0 + alpha
                a1 = -2.0 * cosW0
                a2 = 1.0 - alpha
            }

            let invA0 = 1.0 / max(1e-6, a0)
            b0 *= invA0
            b1 *= invA0
            b2 *= invA0
            a1 *= invA0
            a2 *= invA0
        }

        @inline(__always)
        mutating func process(_ x: Float) -> Float {
            var y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            if abs(y) < 1e-18 { y = 0 }
            return y
        }
    }

    private struct LoopChannel {
        var filter: BiquadFilter
        var phase: Int
        var currentFreq: Float
        var targetFreq: Float
        var freqTau: Float = 0.06
        var currentGain: Float = 0.0001
        var targetGain: Float = 0.0001
        var gainTau: Float = 0.06
        var currentPan: Float = 0.0
        var targetPan: Float = 0.0
        var panTau: Float = 0.05

        init(kind: FilterKind, freq: Float, q: Float, phaseOffset: Int, sampleRate: Float) {
            self.filter = BiquadFilter(kind: kind, freq: freq, q: q, sampleRate: sampleRate)
            self.phase = phaseOffset
            self.currentFreq = freq
            self.targetFreq = freq
        }
    }

    private struct OneShotVoice {
        var active: Bool = false
        var isNoise: Bool = true
        var waveform: OscWaveform = .sine
        var oscPhase: Float = 0
        var noiseIndex: Int = 0
        var age: Float = 0
        var duration: Float = 0.2
        var attackTime: Float = 0.002
        var freqStart: Float = 440
        var freqEnd: Float = 220
        var freqExp: Bool = true
        var gainStart: Float = 0.2
        var gainEnd: Float = 0.001
        var gainExp: Bool = true
        var panStart: Float = 0
        var panEnd: Float = 0
        var useFilter: Bool = false
        var filter: BiquadFilter = BiquadFilter(kind: .lowpass, freq: 1000, q: 1.0, sampleRate: 48000)
        var filterFreqStart: Float = 1000
        var filterFreqEnd: Float = 1000
        var filterFreqExp: Bool = false
    }

    private struct LoopTargets {
        var masterGain: Float = 0.0
        var masterTau: Float = 0.15

        var windFreq: Float = 260
        var windGain: Float = 0.0001
        var windHiGain: Float = 0.0001

        var powderFreq: Float = 620
        var powderGain: Float = 0.0001
        var powderPan: Float = 0
        var powderTau: Float = 0.05

        var carveFreq: Float = 1850
        var carveGain: Float = 0.0001
        var carvePan: Float = 0
        var carveTau: Float = 0.05

        var iceFreq: Float = 3600
        var iceGain: Float = 0.0001
        var icePan: Float = 0
        var iceTau: Float = 0.05

        var sluffGain: Float = 0.0001

        var breathFreq: Float = 720
        var breathGain: Float = 0.0001

        var avySubGain: Float = 0.0001
        var avyMidFreq: Float = 380
        var avyMidGain: Float = 0.0001

        var jacketFreq: Float = 2400
        var jacketGain: Float = 0.0001
    }

    private struct SharedAudioMailbox {
        var targets = LoopTargets()
        var pendingVoices: [OneShotVoice] = []
    }

    private let mailbox = OSAllocatedUnfairLock(initialState: SharedAudioMailbox())

    private var engine: AVAudioEngine?
    private var sourceNode: AVAudioSourceNode?
    private var sampleRate: Float = 48000.0
    private var noiseBuffer: [Float] = []

    // Main-thread control state
    private var enabled: Bool = false
    private var muted: Bool = false
    private var volume: Float = 0.85
    private var nextCreakTime: Float = 12.0
    private var breathPhase: Float = 0.0

    // Audio-thread-only DSP state
    private var loops: [LoopChannel] = []
    private var voices: [OneShotVoice] = Array(repeating: OneShotVoice(), count: 32)
    private var masterGain: Float = 0.0
    private var masterTarget: Float = 0.0
    private var masterTau: Float = 0.15
    private var localTargets = LoopTargets()

    public init() {
        setupAudioEngine()
    }

    deinit {
        engine?.stop()
    }

    private func setupAudioEngine() {
        let eng = AVAudioEngine()
        let outFormat = eng.outputNode.outputFormat(forBus: 0)
        guard outFormat.sampleRate > 0, outFormat.channelCount > 0 else {
            return
        }

        let sr = Float(outFormat.sampleRate)
        self.sampleRate = sr
        guard let stereoFormat = AVAudioFormat(standardFormatWithSampleRate: outFormat.sampleRate, channels: 2) else {
            return
        }

        // Pre-generate 3 seconds of smooth pink/white noise buffer matching audio.js
        let len = max(1024, Int(sr * 3.0))
        var buf = [Float](repeating: 0, count: len)
        var b0: Float = 0, b1: Float = 0, b2: Float = 0
        for i in 0..<len {
            let white = Float.random(in: -1.0...1.0)
            b0 = 0.997 * b0 + white * 0.029
            b1 = 0.985 * b1 + white * 0.032
            b2 = 0.950 * b2 + white * 0.048
            buf[i] = min(1.0, max(-1.0, (b0 + b1 + b2 + white * 0.16) * 1.4))
        }
        self.noiseBuffer = buf

        // 10 continuous loop channels (indices 0..9)
        let stride = max(1, len / 11)
        self.loops = [
            LoopChannel(kind: .bandpass, freq: 260, q: 0.9, phaseOffset: 0 * stride, sampleRate: sr),    // 0 wind
            LoopChannel(kind: .highpass, freq: 1400, q: 0.7, phaseOffset: 1 * stride, sampleRate: sr),   // 1 windHi
            LoopChannel(kind: .lowpass, freq: 620, q: 0.7, phaseOffset: 2 * stride, sampleRate: sr),     // 2 powder
            LoopChannel(kind: .bandpass, freq: 1850, q: 1.4, phaseOffset: 3 * stride, sampleRate: sr),   // 3 carve
            LoopChannel(kind: .bandpass, freq: 3600, q: 3.8, phaseOffset: 4 * stride, sampleRate: sr),   // 4 ice
            LoopChannel(kind: .bandpass, freq: 920, q: 0.8, phaseOffset: 5 * stride, sampleRate: sr),    // 5 sluff
            LoopChannel(kind: .bandpass, freq: 720, q: 1.8, phaseOffset: 6 * stride, sampleRate: sr),    // 6 breath
            LoopChannel(kind: .lowpass, freq: 115, q: 1.1, phaseOffset: 7 * stride, sampleRate: sr),     // 7 avySub
            LoopChannel(kind: .bandpass, freq: 380, q: 0.75, phaseOffset: 8 * stride, sampleRate: sr),   // 8 avyMid
            LoopChannel(kind: .bandpass, freq: 2400, q: 1.1, phaseOffset: 9 * stride, sampleRate: sr),   // 9 jacketSpray
        ]

        let node = AVAudioSourceNode(format: stereoFormat) { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self else { return noErr }
            let abl = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let count = Int(frameCount)
            guard abl.count >= 2,
                  let leftPtr = abl[0].mData?.assumingMemoryBound(to: Float.self),
                  let rightPtr = abl[1].mData?.assumingMemoryBound(to: Float.self) else {
                for buf in abl {
                    if let p = buf.mData {
                        memset(p, 0, Int(buf.mDataByteSize))
                    }
                }
                return noErr
            }
            self.renderAudioBlock(left: leftPtr, right: rightPtr, frameCount: count)
            return noErr
        }

        eng.attach(node)
        eng.connect(node, to: eng.mainMixerNode, format: stereoFormat)
        eng.connect(eng.mainMixerNode, to: eng.outputNode, format: stereoFormat)

        do {
            try eng.start()
            self.engine = eng
            self.sourceNode = node
        } catch {
            // Headless / CI / no audio device: degrade gracefully to silent no-op
            self.engine = nil
            self.sourceNode = nil
        }
    }

    public func unlock() {
        enabled = true
        if let eng = engine, !eng.isRunning {
            try? eng.start()
        }
        updateMasterTarget(tau: 0.15)
    }

    public func setVolume(_ volume: Float) {
        self.volume = min(1.0, max(0.0, volume))
        if self.volume == 0 {
            muted = true
        } else if muted && self.volume > 0 {
            muted = false
        }
        updateMasterTarget(tau: 0.06)
    }

    public func setMuted(_ muted: Bool) {
        self.muted = muted
        updateMasterTarget(tau: 0.10)
    }

    @discardableResult
    public func toggleMute() -> Bool {
        enabled = true
        muted.toggle()
        if !muted && volume <= 0.01 { volume = 0.75 }
        updateMasterTarget(tau: 0.10)
        return !muted
    }

    private func updateMasterTarget(tau: Float) {
        let target: Float = (!enabled || muted) ? 0.0 : (0.68 * volume)
        mailbox.withLock { box in
            box.targets.masterGain = target
            box.targets.masterTau = tau
        }
    }

    public func update(state: SkierState, preset: LightingPreset, avalanche: AvalancheState, dt: Float) {
        guard enabled, !muted, engine != nil else { return }

        let spd = state.speed
        let kmh = spd * 3.6
        let storm: Float = preset.isStorm ? 1.0 : 0.0
        let altWind = max(0.08, min(0.45, (state.y - 1600.0) / 3600.0)) + storm * 0.35

        // 1. High-altitude wind & airspeed rushing past helmet
        let airRatio = min(1.0, kmh / 105.0)
        let airGain = (0.04 + altWind * 0.12 + airRatio * airRatio * 0.42) * (state.grounded ? 1.0 : 1.32)
        let windFreq: Float = 180.0 + airRatio * 720.0 + (state.grounded ? 0.0 : 160.0)
        let windHiGain: Float = airRatio * airRatio * (state.grounded ? 0.06 : 0.15) + storm * 0.05

        // 2. Instant silence when airborne
        let slip = abs(sinf(state.slipAngle))
        let sideSlip = state.sideslip
        let wash = state.washOut
        let edge = min(1.0, abs(state.edgeAngle))
        let panSide = max(-0.78, min(0.78, -state.edgeAngle * 0.78))
        let surf = state.surface
        let moveFactor: Float = (state.grounded && !state.crashed) ? min(1.0, spd / 15.0) : 0.0
        let carveEnergy = moveFactor * (0.24 + slip * 1.15 + sideSlip * 0.72 + edge * 0.42)

        let powGain: Float
        if surf == .powder { powGain = carveEnergy * 0.44 }
        else if surf == .packed || surf == .corn { powGain = carveEnergy * 0.14 }
        else { powGain = 0.0001 }

        let crvGain: Float
        if surf == .packed || surf == .corn || surf == .crust { crvGain = carveEnergy * 0.34 }
        else if surf == .powder { crvGain = carveEnergy * 0.12 }
        else { crvGain = carveEnergy * 0.20 }

        let iceGain: Float
        if surf == .ice || surf == .granite { iceGain = carveEnergy * 0.40 + wash * 0.24 }
        else if surf == .crust { iceGain = carveEnergy * 0.15 + wash * 0.18 }
        else { iceGain = wash * 0.14 }

        let tc: Float = state.grounded ? 0.05 : 0.025

        // 3. Heavy high-altitude cardio breathing when slowing/pausing
        breathPhase += dt * (2.2 + state.breathing * 1.4)
        let slowFade = max(0.0, 1.0 - kmh / 18.0)
        let brActive = state.breathing > 0.15 ? (state.breathing - 0.15) * slowFade : 0.0
        let cycle = sinf(breathPhase)
        let breathEnv = powf(abs(cycle), 1.6)
        let breathFreq: Float = cycle > 0 ? 820.0 : 520.0

        // 4. Distant glacier creak above 2,750 m when quiet
        nextCreakTime -= dt
        if nextCreakTime <= 0 {
            if state.y > 2750 && kmh < 22 && !avalanche.active {
                playGlacierCreak()
            }
            nextCreakTime = 11.0 + Float.random(in: 0..<14.0)
        }

        // 5. Slab avalanche roar
        let avyProx: Float = avalanche.active ? avalanche.proximity : 0.0

        // 6. Hardshell jacket spray patter
        let sprayHit = max(state.faceShotIntensity, state.bowWave * 0.55)

        mailbox.withLock { box in
            box.targets.masterGain = 0.68 * volume
            box.targets.masterTau = 0.08
            box.targets.windFreq = windFreq
            box.targets.windGain = max(0.0001, airGain)
            box.targets.windHiGain = max(0.0001, windHiGain)

            box.targets.powderFreq = 380.0 + moveFactor * 520.0 + slip * 300.0
            box.targets.powderGain = max(0.0001, powGain)
            box.targets.powderPan = panSide
            box.targets.powderTau = tc

            box.targets.carveFreq = 1400.0 + kmh * 16.0 + slip * 700.0
            box.targets.carveGain = max(0.0001, crvGain)
            box.targets.carvePan = panSide
            box.targets.carveTau = tc

            box.targets.iceFreq = surf == .granite ? 2400.0 : (3400.0 + kmh * 14.0)
            box.targets.iceGain = max(0.0001, iceGain)
            box.targets.icePan = panSide
            box.targets.iceTau = tc

            box.targets.sluffGain = max(0.0001, state.sluffTrigger * 0.14)
            box.targets.breathFreq = breathFreq
            box.targets.breathGain = max(0.0001, brActive * breathEnv * 0.16)

            box.targets.avySubGain = max(0.0001, avyProx * 0.62)
            box.targets.avyMidFreq = 260.0 + avyProx * 380.0
            box.targets.avyMidGain = max(0.0001, avyProx * 0.36)

            box.targets.jacketFreq = 1950.0 + sprayHit * 1350.0
            box.targets.jacketGain = max(0.0001, sprayHit * 0.19)
        }
    }

    public func handleEvent(_ event: PhysicsEvent) {
        guard enabled, !muted, engine != nil else { return }
        switch event {
        case let .polePlant(side, _, _, _, strength, powder):
            playPolePlant(side: Float(side), strength: strength, isPowder: powder)
        case let .treeWhoosh(pan, closeness, speed):
            playTreeWhoosh(panSide: pan, closeness: closeness, speedMs: speed)
        case let .bootCreak(strength):
            playBootCreak(strength: strength)
        case let .pop(_, _, _, charge):
            playPop(charge: charge)
        case let .land(_, _, _, impact, powder, _):
            playImpact(strength: min(1.0, impact / 14.0), isPowder: powder, isCrash: false)
        case .crash:
            playImpact(strength: 1.0, isPowder: true, isCrash: true)
        case .avalancheStart:
            playAvalancheCrack()
        case .track, .spray, .sluffStream, .crashTrail:
            break
        }
    }

    // MARK: - Procedural One-Shot Triggers

    public func playImpact(strength: Float = 0.6, isPowder: Bool = true, isCrash: Bool = false) {
        guard enabled, !muted, engine != nil else { return }
        var osc = OneShotVoice()
        osc.active = true
        osc.isNoise = false
        osc.waveform = .sine
        osc.freqStart = isCrash ? 95.0 : (isPowder ? 76.0 : 118.0)
        osc.freqEnd = 28.0
        osc.freqExp = true
        osc.gainStart = min(0.65, 0.18 + strength * 0.35)
        osc.gainEnd = 0.001
        osc.gainExp = true
        osc.duration = isCrash ? 0.42 : 0.25

        var noise = OneShotVoice()
        noise.active = true
        noise.isNoise = true
        noise.noiseIndex = Int.random(in: 0..<max(1, noiseBuffer.count))
        noise.useFilter = true
        noise.filter = BiquadFilter(kind: isPowder ? .lowpass : .bandpass, freq: isPowder ? 950 : 2100, q: 1.0, sampleRate: sampleRate)
        noise.filterFreqStart = isPowder ? 950 : 2100
        noise.filterFreqEnd = noise.filterFreqStart
        noise.gainStart = min(0.55, 0.14 + strength * 0.32)
        noise.gainEnd = 0.001
        noise.gainExp = true
        noise.duration = isCrash ? 0.85 : 0.42

        enqueueVoices([osc, noise])
    }

    public func playPop(charge: Float = 0.5) {
        guard enabled, !muted, engine != nil else { return }
        var noise = OneShotVoice()
        noise.active = true
        noise.isNoise = true
        noise.noiseIndex = Int.random(in: 0..<max(1, noiseBuffer.count))
        noise.useFilter = true
        noise.filter = BiquadFilter(kind: .bandpass, freq: 650, q: 1.0, sampleRate: sampleRate)
        noise.filterFreqStart = 650
        noise.filterFreqEnd = 1900
        noise.filterFreqExp = true
        noise.gainStart = 0.14 + charge * 0.16
        noise.gainEnd = 0.001
        noise.gainExp = true
        noise.duration = 0.18

        var click = OneShotVoice()
        click.active = true
        click.isNoise = false
        click.waveform = .triangle
        click.freqStart = 1950 + charge * 550
        click.freqEnd = 420
        click.freqExp = true
        click.gainStart = 0.11 + charge * 0.10
        click.gainEnd = 0.0008
        click.gainExp = true
        click.duration = 0.042

        enqueueVoices([noise, click])
    }

    public func playAvalancheCrack() {
        guard enabled, !muted, engine != nil else { return }
        var sub = OneShotVoice()
        sub.active = true
        sub.isNoise = false
        sub.waveform = .sine
        sub.freqStart = 64
        sub.freqEnd = 22
        sub.freqExp = true
        sub.gainStart = 0.75
        sub.gainEnd = 0.001
        sub.gainExp = true
        sub.duration = 1.15

        var noise = OneShotVoice()
        noise.active = true
        noise.isNoise = true
        noise.noiseIndex = Int(sampleRate * 0.2) % max(1, noiseBuffer.count)
        noise.useFilter = true
        noise.filter = BiquadFilter(kind: .bandpass, freq: 280, q: 0.6, sampleRate: sampleRate)
        noise.filterFreqStart = 280
        noise.filterFreqEnd = 280
        noise.gainStart = 0.55
        noise.gainEnd = 0.001
        noise.gainExp = true
        noise.duration = 1.4

        enqueueVoices([sub, noise])
    }

    public func playPolePlant(side: Float = 1.0, strength: Float = 0.7, isPowder: Bool = true) {
        guard enabled, !muted, engine != nil else { return }
        let pan = max(-0.75, min(0.75, side * 0.56))

        var noise = OneShotVoice()
        noise.active = true
        noise.isNoise = true
        noise.noiseIndex = Int.random(in: 0..<max(1, noiseBuffer.count))
        noise.useFilter = true
        noise.filter = BiquadFilter(kind: .bandpass, freq: isPowder ? 720 : 1650, q: isPowder ? 1.0 : 2.2, sampleRate: sampleRate)
        noise.filterFreqStart = isPowder ? 720 : 1650
        noise.filterFreqEnd = isPowder ? 260 : 620
        noise.filterFreqExp = true
        noise.gainStart = min(0.32, 0.09 + strength * 0.14)
        noise.gainEnd = 0.001
        noise.gainExp = true
        noise.panStart = pan
        noise.panEnd = pan
        noise.duration = 0.095

        var tap = OneShotVoice()
        tap.active = true
        tap.isNoise = false
        tap.waveform = .triangle
        tap.freqStart = isPowder ? 640 : 1480
        tap.freqEnd = 140
        tap.freqExp = true
        tap.gainStart = 0.04 + strength * (isPowder ? 0.045 : 0.09)
        tap.gainEnd = 0.0008
        tap.gainExp = true
        tap.panStart = pan
        tap.panEnd = pan
        tap.duration = 0.034

        enqueueVoices([noise, tap])
    }

    public func playTreeWhoosh(panSide: Float = 0.7, closeness: Float = 0.7, speedMs: Float = 18.0) {
        guard enabled, !muted, engine != nil else { return }
        let spFactor = min(1.5, max(0.35, speedMs / 20.0))
        let dur = max(0.18, min(0.46, 0.44 - spFactor * 0.16))
        let peakF = 760 + spFactor * 640
        let peakGain = min(0.42, (0.09 + closeness * 0.26) * spFactor)

        var noise = OneShotVoice()
        noise.active = true
        noise.isNoise = true
        noise.noiseIndex = Int.random(in: 0..<max(1, noiseBuffer.count))
        noise.useFilter = true
        noise.filter = BiquadFilter(kind: .bandpass, freq: peakF, q: 1.65, sampleRate: sampleRate)
        noise.filterFreqStart = peakF
        noise.filterFreqEnd = 195
        noise.filterFreqExp = true
        noise.attackTime = dur * 0.28
        noise.gainStart = peakGain
        noise.gainEnd = 0.0008
        noise.gainExp = true
        noise.panStart = max(-0.92, min(0.92, panSide * 0.85))
        noise.panEnd = max(-0.92, min(0.92, panSide * 0.35))
        noise.duration = dur

        enqueueVoices([noise])
    }

    public func playBootCreak(strength: Float = 0.5) {
        guard enabled, !muted, engine != nil else { return }
        var osc = OneShotVoice()
        osc.active = true
        osc.isNoise = false
        osc.waveform = .sawtooth
        osc.freqStart = 155 + Float.random(in: 0..<40)
        osc.freqEnd = 118 + Float.random(in: 0..<25)
        osc.freqExp = false
        osc.useFilter = true
        osc.filter = BiquadFilter(kind: .bandpass, freq: 460, q: 6.5, sampleRate: sampleRate)
        osc.filterFreqStart = 460
        osc.filterFreqEnd = 460
        osc.attackTime = 0.04
        osc.gainStart = 0.032 * min(1.0, strength)
        osc.gainEnd = 0.0006
        osc.gainExp = true
        osc.duration = 0.16

        enqueueVoices([osc])
    }

    private func playGlacierCreak() {
        var osc = OneShotVoice()
        osc.active = true
        osc.isNoise = false
        osc.waveform = .sawtooth
        let baseF = 48 + Float.random(in: 0..<44)
        osc.freqStart = baseF
        osc.freqEnd = baseF * 0.52
        osc.freqExp = true
        osc.useFilter = true
        let fFreq = 210 + Float.random(in: 0..<240)
        osc.filter = BiquadFilter(kind: .bandpass, freq: fFreq, q: 4.2, sampleRate: sampleRate)
        osc.filterFreqStart = fFreq
        osc.filterFreqEnd = fFreq
        osc.attackTime = 0.14
        osc.gainStart = 0.085
        osc.gainEnd = 0.0008
        osc.gainExp = true
        let pan = Float.random(in: -0.7...0.7)
        osc.panStart = pan
        osc.panEnd = pan
        osc.duration = 1.75

        enqueueVoices([osc])
    }

    private func enqueueVoices(_ newVoices: [OneShotVoice]) {
        mailbox.withLock { box in
            if box.pendingVoices.count < 32 {
                box.pendingVoices.append(contentsOf: newVoices)
            }
        }
    }

    // MARK: - Real-Time Audio Render Thread (Lock-Free / Non-Blocking)

    private func renderAudioBlock(left: UnsafeMutablePointer<Float>, right: UnsafeMutablePointer<Float>, frameCount: Int) {
        // Non-blocking tryLock: if main thread is mid-update, keep previous targets with zero waiting.
        mailbox.withLockIfAvailable { box in
            localTargets = box.targets
            masterTarget = box.targets.masterGain
            masterTau = box.targets.masterTau
            if !box.pendingVoices.isEmpty {
                for nv in box.pendingVoices {
                    if let freeIdx = voices.firstIndex(where: { !$0.active }) {
                        voices[freeIdx] = nv
                    }
                }
                box.pendingVoices.removeAll(keepingCapacity: true)
            }
        }

        // Apply latest targets to loop channels
        loops[0].targetFreq = localTargets.windFreq
        loops[0].targetGain = localTargets.windGain
        loops[1].targetGain = localTargets.windHiGain

        loops[2].targetFreq = localTargets.powderFreq
        loops[2].targetGain = localTargets.powderGain
        loops[2].targetPan = localTargets.powderPan
        loops[2].gainTau = localTargets.powderTau

        loops[3].targetFreq = localTargets.carveFreq
        loops[3].targetGain = localTargets.carveGain
        loops[3].targetPan = localTargets.carvePan
        loops[3].gainTau = localTargets.carveTau

        loops[4].targetFreq = localTargets.iceFreq
        loops[4].targetGain = localTargets.iceGain
        loops[4].targetPan = localTargets.icePan
        loops[4].gainTau = localTargets.iceTau

        loops[5].targetGain = localTargets.sluffGain
        loops[5].gainTau = 0.25

        loops[6].targetFreq = localTargets.breathFreq
        loops[6].targetGain = localTargets.breathGain
        loops[6].gainTau = 0.08

        loops[7].targetGain = localTargets.avySubGain
        loops[7].gainTau = 0.16

        loops[8].targetFreq = localTargets.avyMidFreq
        loops[8].targetGain = localTargets.avyMidGain
        loops[8].gainTau = 0.16

        loops[9].targetFreq = localTargets.jacketFreq
        loops[9].targetGain = localTargets.jacketGain
        loops[9].gainTau = 0.06

        let noiseCount = noiseBuffer.count
        guard noiseCount > 0 else {
            for i in 0..<frameCount { left[i] = 0; right[i] = 0 }
            return
        }
        let dtSample = 1.0 / sampleRate
        let masterAlpha = 1.0 - expf(-dtSample / max(0.005, masterTau))

        for i in 0..<frameCount {
            masterGain += (masterTarget - masterGain) * masterAlpha
            if masterGain < 1e-5 && masterTarget < 1e-5 {
                left[i] = 0
                right[i] = 0
                continue
            }

            var mixL: Float = 0
            var mixR: Float = 0

            // Render 10 continuous filtered-noise beds
            let updateCoeffs = (i & 15) == 0
            for c in 0..<loops.count {
                let raw = noiseBuffer[loops[c].phase]
                loops[c].phase += 1
                if loops[c].phase >= noiseCount { loops[c].phase = 0 }

                let fAlpha = 1.0 - expf(-dtSample / max(0.005, loops[c].freqTau))
                let gAlpha = 1.0 - expf(-dtSample / max(0.005, loops[c].gainTau))
                let pAlpha = 1.0 - expf(-dtSample / max(0.005, loops[c].panTau))

                loops[c].currentFreq += (loops[c].targetFreq - loops[c].currentFreq) * fAlpha
                loops[c].currentGain += (loops[c].targetGain - loops[c].currentGain) * gAlpha
                loops[c].currentPan += (loops[c].targetPan - loops[c].currentPan) * pAlpha

                if updateCoeffs {
                    loops[c].filter.updateCoefficients(freq: loops[c].currentFreq, sampleRate: sampleRate)
                }

                let filtered = loops[c].filter.process(raw) * loops[c].currentGain
                let angle = (loops[c].currentPan + 1.0) * (Float.pi * 0.25)
                mixL += filtered * cosf(angle)
                mixR += filtered * sinf(angle)
            }

            // Render active one-shot procedural voices
            for v in 0..<voices.count where voices[v].active {
                let age = voices[v].age
                let dur = max(0.005, voices[v].duration)
                if age >= dur {
                    voices[v].active = false
                    continue
                }
                let u = min(1.0, age / dur)

                // Gain envelope
                let att = voices[v].attackTime
                let env: Float
                if att > 0.003 && age < att {
                    env = (age / att) * voices[v].gainStart
                } else {
                    let remT = max(0.0, (age - att) / max(0.001, dur - att))
                    if voices[v].gainExp {
                        env = voices[v].gainStart * powf(max(1e-4, voices[v].gainEnd / max(1e-4, voices[v].gainStart)), remT)
                    } else {
                        env = voices[v].gainStart + (voices[v].gainEnd - voices[v].gainStart) * remT
                    }
                }

                var sample: Float
                if voices[v].isNoise {
                    sample = noiseBuffer[voices[v].noiseIndex]
                    voices[v].noiseIndex += 1
                    if voices[v].noiseIndex >= noiseCount { voices[v].noiseIndex = 0 }
                } else {
                    let freq: Float
                    if voices[v].freqExp {
                        freq = voices[v].freqStart * powf(max(1e-4, voices[v].freqEnd / max(1e-4, voices[v].freqStart)), u)
                    } else {
                        freq = voices[v].freqStart + (voices[v].freqEnd - voices[v].freqStart) * u
                    }
                    voices[v].oscPhase += freq * dtSample
                    if voices[v].oscPhase >= 1.0 { voices[v].oscPhase -= floorf(voices[v].oscPhase) }
                    let p = voices[v].oscPhase
                    switch voices[v].waveform {
                    case .sine:
                        sample = sinf(p * 2.0 * Float.pi)
                    case .triangle:
                        sample = 1.0 - 4.0 * abs(p - 0.5)
                    case .sawtooth:
                        sample = 2.0 * p - 1.0
                    }
                }

                if voices[v].useFilter {
                    if updateCoeffs && abs(voices[v].filterFreqEnd - voices[v].filterFreqStart) > 1.0 {
                        let fFreq: Float
                        if voices[v].filterFreqExp {
                            fFreq = voices[v].filterFreqStart * powf(max(1e-4, voices[v].filterFreqEnd / max(1e-4, voices[v].filterFreqStart)), u)
                        } else {
                            fFreq = voices[v].filterFreqStart + (voices[v].filterFreqEnd - voices[v].filterFreqStart) * u
                        }
                        voices[v].filter.updateCoefficients(freq: fFreq, sampleRate: sampleRate)
                    }
                    sample = voices[v].filter.process(sample)
                }

                let pan = voices[v].panStart + (voices[v].panEnd - voices[v].panStart) * u
                let angle = (pan + 1.0) * (Float.pi * 0.25)
                let outSample = sample * env
                mixL += outSample * cosf(angle)
                mixR += outSample * sinf(angle)

                voices[v].age += dtSample
            }

            left[i] = tanhf(mixL * masterGain)
            right[i] = tanhf(mixR * masterGain)
        }
    }
}
