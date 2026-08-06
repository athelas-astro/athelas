target("athelas")
set_kind("binary")
add_deps("athelas_configuration")

-- Stamps the git hash, compiler, and build time into the binary; reported at
-- startup and written to the HDF5 output metadata.
add_rules("toolkit.provenance")
set_values("toolkit.provenance.template", path.join(os.projectdir(), "src", "build_info.cpp.in"))

-- This legacy implementation is also excluded from ATHELAS_SOURCES in CMake.
-- We will eventually remove that file..
add_files(path.join(os.projectdir(), "src", "**.cpp|basis/polynomial_basis.cpp"))

add_includedirs(athelas_includedirs)

add_packages("athelas_kokkos_kernels", "athelas_spiner", "athelas_eigen", "athelas_sol2", "athelas_hdf5")
if get_config("backend") == "openmp" then
  add_packages("openmp")
end
add_cxxflags("-Wall")
add_syslinks("stdc++exp")

on_load(function(target)
  target:set("rundir", os.workingdir())
end)

-- openmp options.
set_runenv("OMP_PROC_BIND", "false")
set_runenv("OMP_PLACES", "threads")

if athelas_debug_mode then
  add_defines("ATHELAS_DEBUG")
end
