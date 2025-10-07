# athelas (Astrophysical Transients with Hydrodynamics and Emission using a Lagrangian Arbitrary-order Scheme )

<p align="center">1D Lagrangian radiation-hydrodynamics solver written in C++ </p>

[![Build](https://github.com/athelas-astro/athelas/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/athelas-astro/athelas/actions/workflows/cmake-multi-platform.yml)
<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

`Athelas` solves the 1D Lagrangian equation of non-relativistic radiation hydrodynamics using a discontinuous Galerkin scheme. 
LTE Saha ionization is included.
It includes planar geometry and spherical symmetry. Self gravity is included.


# Installation
`athelas` uses submodules to include dependencies. 
The best way to get the source is the following 
```sh
git clone --recursive git@github.com:AstroBarker/athelas.git
```

# Building
`athelas` is built using `cmake`. From the root directory of `athelas`, run the following:

```sh
mkdir build && cd build
cmake ..
cmake --build . # or make -j
```

# Running
To run `athelas` simply execute `./athelas -i ../inputs/sod.toml`, for instance.

# Tests
Regression tests live in `test/regression`. To run all test, run 
`python run_regression_tests.py`. Pass `-e /path/to/athelas/executable` to 
avoid rebuilding each test. To run a specific test, run 
`python run_regression_tests.py --test test_sod -e /path/to/athelas/executable` etc.


# Kokkos
We use [Kokkos](https://github.com/kokkos) for shared memory parallelism. 

# TOML++
We use [toml++](https://github.com/marzer/tomlplusplus) for parsing input files in the `.toml` format.

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
* TOML++ (submodule)
* HDF5
