# Changelog
---

## Staged
---

### Added (new features/APIs/variables/...)

- [PR 124](https://github.com/athelas-astro/athelas/pull/124) Use minimum internal energy in BEL
- [PR 123](https://github.com/athelas-astro/athelas/pull/123) Move basis, eos, opac objects into MeshState
- [PR 122](https://github.com/athelas-astro/athelas/pull/122) Threaded StageData into BEL
- [PR 121](https://github.com/athelas-astro/athelas/pull/121) New Saha solver; optimizations

### Changed (changing behavior/API/variables/...)

- [PR 126](https://github.com/athelas-astro/athelas/pull/126) Remove inner `ionization_enabled` branching in radhydro implicit sources
- [PR 125](https://github.com/athelas-astro/athelas/pull/125) Remove inner `ionization_enabled` branching in fill_derived kernels

### Fixed (not changing behavior/API/variables/...)

- [PR 129](https://github.com/athelas-astro/athelas/pull/129) Optimizations in `composition_fill_derived`

### Infrastructure (organization/...)

- [PR 124](https://github.com/athelas-astro/athelas/pull/124) Introduce `EOSLambda` lambda wrapper

### Removed (removing behavior/API/variables/...)

### Incompatibilities (i.e. breaking changes)
