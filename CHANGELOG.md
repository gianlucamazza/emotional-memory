# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **affective-fly host adapter.** `FlyAffectHost` in `emotional_memory.integrations.fly`
  constructs this package's store and `EmotionalMemory`, owns wall-clock /
  HostFrame `mood_dt`, and each tick asks
  [affective-fly](https://github.com/gianlucamazza/affective-fly) for valence /
  arousal / approach-avoid. Fly Policy / LaunchGate thresholds and Phase 6
  hypothesis taus (300 / 60 / 180) are unchanged. Install from GitHub
  (`make install-fly`); not a locked extra (fly depends on this package and is
  not on PyPI). See `docs/tutorials/fly.md` and `examples/fly_affect_source.py`.

