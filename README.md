# athelas (Astrophysical Transients with Hydrodynamics and Emission using a Lagrangian Arbitrary-order Scheme )

<p align="center">1D Lagrangian radiation-hydrodynamics solver written in C++ </p>

[![Build](https://github.com/athelas-astro/athelas/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/athelas-astro/athelas/actions/workflows/cmake-multi-platform.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://athelas-astro.github.io/)
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>


`Athelas` solves the 1D Lagrangian equation of non-relativistic radiation hydrodynamics using a nodal discontinuous Galerkin scheme. 
Key features:
* Two moment radiation transport with fully implicit integration and an IMEX treatment.
* Lagrangian hydrodynamics
* LTE Saha ionization
* A nickel decay network and heating
* Newtonian self gravity
* High order, conservative discontinuous Galerkin methods
* Spherical and flat Cartesian geometries
* Fully coupled IMEX time integration
* Artificially driven explosions
* A sophisticated package system to thread new physics into the timestepper

Work in progress [docs](https://athelas-astro.github.io) are available.

# Installation
`athelas` uses submodules to include dependencies. 
The best way to get the source is the following 
```sh
git clone --recursive https://github.com/athelas-astro/athelas.git
```

# Building

CMake remains supported. From the root directory of `athelas`, run:

```sh
mkdir build && cd build
cmake ..
cmake --build . # or make -j
```

An experimental xmake build is also available:

```sh
xmake f
xmake build athelas
xmake run athelas -i inputs/sod.lua
```

Alongside xmake's `debug`, `release`, `releasedbg`, and `profile` modes,
Athelas provides `relwithdebinfo` (CMake-style `RelWithDebInfo`) and `perf`
(sampling profiles without `-pg`). The `asan`, `tsan`, `ubsan`, and `lsan`
modes use xmake's corresponding sanitizer policies.
`allsan` combines the compatible address, leak, and undefined-behavior sanitizers
with debug settings; ThreadSanitizer must be run separately.

Dependencies built from submodules are cached under `build/.packages/` and
keyed on the submodule revision, so a bump rebuilds them. A dependency is cached
as a unit, so that rebuild is a full one. 

# Running
To run `athelas` simply execute `./athelas -i ../inputs/sod.lua`, for instance.

# Tests

Regression tests live in `test/regression`. To run all test, run 
`python run_regression_tests.py`. Pass `-e /path/to/athelas/executable` to 
avoid rebuilding each test. To run a specific test, run 
`python run_regression_tests.py --test test_sod -e /path/to/athelas/executable` etc.

To build and run the unit tests with xmake:

```sh
xmake f --unit_tests=y
xmake test -g unit
```

The group filter runs the unit suite without invoking regression tests, even
when both test options are enabled.

Regression tests are registered separately:

```sh
xmake f --regression_tests=y
xmake test -g regression
xmake test regression_tests/sod
```

When available, xmake runs the regression suite in the locked `uv` environment
under `scripts/python/athelas_tools`. Otherwise, the active Python environment
must provide the regression-test dependencies.


# Kokkos
We use [Kokkos](https://github.com/kokkos) for shared memory parallelism. 

# Sol
We use [Sol](https://sol2.readthedocs.io/en/latest/index.html) for parsing input configuration
using the Lua language. This means that Lua is required for running `Athelas`.

# Code Style

We use `clang format` and `ruff` for code cleanliness. 
Rules are listed in `.clang-format`.
The current version of `clang-format` used is 20.1.0.
Simply call `tools/bash/format.sh` to format the `.hpp` and `.cpp` files.

Python code linting and formatting is done with `ruff`. 
Rules are listed in `ruff.toml`. 
To check all python in the current directory, you may `ruff ..`
To format a given file according to `ruff.toml`, run `ruff format file.py`. 

Checks for formatting are performed on each PR.

There is also a Git pre-commit hook available in `scripts/hooks` that will 
perform this formatting on a commit. You can enable this simply by 

```bash
./scripts/hooks/install-hooks.sh
```
which will automatically symlink the hook into `.git/hooks`.

# Dependencies
* Eigen (submodule)
* Kokkos (submodule)
* Sol (submodule)
* Lua
* HDF5

# Contributors
| Name | Handle | Affiliation |
| :--- | :--- | :--- |
| Brandon L. Barker | @AstroBarker | LANL |
