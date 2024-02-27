# athelas (A radiaTion-Hydrodynamics codE for modeLing sphericAl Supernovae )

<p align="center">sypherically symmetric Lagrangian radiation-hydrodynamics solver written in C++ </p>

[![Build](https://github.com/AstroBarker/athelas/actions/workflows/cmake.yml/badge.svg)](https://github.com/AstroBarker/athelas/actions/workflows/cmake.yml)
<p align="center">
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-GPL-blue.svg"></a>
</p>

Currently, `athelas` solves the 1D Lagrangian equation of non-relativistic hydrodynamics using a discontinuous Galerkin scheme. It includes planar geometry and spherical symmetry.
It will be extended to special relativistic hydrodynamics.

Future work will include two-moment radiation transport, a Saha ionization solver, and appropriate stellar equation of state.

# Kokkos
We use [Kokkos](https://github.com/kokkos) for shared memory parallelism. 
Currently, most significant data structures use `Kokkos::Views` and loops are parallelised with `Kokkos`.
More work to port other parts of the code.

# SimpleIni
We use [SimpleIni](https://github.com/brofield/simpleini) for parsing input files in the for parsing input files in the `.ini` format.

# Installation
`athelas` uses submoduless to include dependencies. 
The best way to get the source is the following 
```sh
git clone --recursive git@github.com:AstroBarker/athelas.git
```

# Building
`athelas` is installed using cmake. From the root directory of `athelas`, run the following:

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

As a temporary fix for Ubuntu CI, we need to pass a `MACHINE` flag.
On Mac we support `-DMACHINE=MACOS`,
Ubuntu supports `-DMACHINE=UBUNTU` (primaryily because the CI fails to find `lapacke.h` unless we hold its hand and this is how we do that, for now.)
The default is `UBUNTU`. Passing anything else should let cmake's `find_package` do its thing. 

This will create a directory `bin` in the root directory that contains the executable.

## NOTE: 
The build system may not be perfect yet. Your mileage may vary.

## TODO
- function naming overhaul
- class accessor refactor
- kokkos parallel slope limiter
- repalce simpleini reader with different format (TOML, YAML)
- Rename values in hdf5 output...

### Radiation TODO:
- ... microphysics...
- implicit update


# Future Work

- Grey M1 radiation [WIP]
- Relativistic hydro
- Gravity
- Multigroup radiation
- Non-relativistic MHD

# Dependencies
* LAPACKE
* HDF5
* Kokkos (avialable as a submodule)

Hopefully `lapacke` won't be necessary forever, but at present it is needed for initializing the quadrature.
I find that, on Arch Linux systems, `lapack`, `lapacke`, and `openblas` is sufficient for all `lapacke` needs.

## Clang-format

We use clang format for code cleanliness. 
The current version of `clang-format` used is `clang-format-13`.
Simply call `tools/bash/format.sh` to format the `.hpp` and `.cpp` files.

# TODO:
 - [x] Initialize with input file at runtime
 - [x] Bound enforcing limiter
 - [ ] New TCI
 - [ ] format on make
 - [ ] Rad: Riemann solvers beyond LLF?
 - [ ] Adjust numerical fluxes for 1D MHD
 - [ ] Implement 1.5D NRMHD


# BUGS: 
- [ ] nNodes > order, crash when SlopeLimiter applied.
