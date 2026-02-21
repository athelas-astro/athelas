# Changelog
---

## Staged
---

### Added (new features/APIs/variables/...)

- [PR 146](https://github.com/athelas-astro/athelas/pull/140) Tabular opacity, opacity floor
- [PR 140](https://github.com/athelas-astro/athelas/pull/140) `athelas.py` forms cell averages on `get` call.
- [PR 140](https://github.com/athelas-astro/athelas/pull/140) Made slope limiter aware of non-uniform mesh.
- [PR 133](https://github.com/athelas-astro/athelas/pull/133) rho, T powerlaw opacity with floors.
- [PR 124](https://github.com/athelas-astro/athelas/pull/124) Use minimum internal energy in BEL
- [PR 123](https://github.com/athelas-astro/athelas/pull/123) Move basis, eos, opac objects into MeshState
- [PR 122](https://github.com/athelas-astro/athelas/pull/122) Threaded StageData into BEL
- [PR 121](https://github.com/athelas-astro/athelas/pull/121) New Saha solver; optimizations

### Changed (changing behavior/API/variables/...)

- [PR 140](https://github.com/athelas-astro/athelas/pull/140) Move to nodal DG formulation.
- [PR 138](https://github.com/athelas-astro/athelas/pull/138) Add `params` into hdf5 output
- [PR 137](https://github.com/athelas-astro/athelas/pull/137) Update `athelas.py`, add `snap.py`
- [PR 136](https://github.com/athelas-astro/athelas/pull/136) New HDF5 structure
- [PR 131](https://github.com/athelas-astro/athelas/pull/131) Added interfaces into nodal grid.
- [PR 126](https://github.com/athelas-astro/athelas/pull/126) Remove inner `ionization_enabled` branching in radhydro implicit sources
- [PR 125](https://github.com/athelas-astro/athelas/pull/125) Remove inner `ionization_enabled` branching in fill_derived kernels

### Fixed (not changing behavior/API/variables/...)

- [PR 130](https://github.com/athelas-astro/athelas/pull/130) Fix variable ordering in `VarialeMap::list` and so in output.
- [PR 134](https://github.com/athelas-astro/athelas/pull/134) Improve radiation wavespeeds
- [PR 132](https://github.com/athelas-astro/athelas/pull/132) Fixed inocorrect setting of sqrt_gm(nNodes + 1)
- [PR 130](https://github.com/athelas-astro/athelas/pull/130) Incorrect temperature inversion template params in radhydro pkg
- [PR 129](https://github.com/athelas-astro/athelas/pull/129) Optimizations in `composition_fill_derived`

### Infrastructure (organization/...)

- [PR 124](https://github.com/athelas-astro/athelas/pull/124) Introduce `EOSLambda` lambda wrapper

### Removed (removing behavior/API/variables/...)

### Incompatibilities (i.e. breaking changes)
- [PR 140](https://github.com/athelas-astro/athelas/pull/140) Move to nodal DG formulation; changed output format.
- [PR 136](https://github.com/athelas-astro/athelas/pull/136) New HDF5 structure breaks python scripts
- [PR 132](https://github.com/athelas-astro/athelas/pull/132) Added interfaces into nodal grid: changed hdf5 io.
