target("unit_tests")
set_kind("binary")
set_default(false)
set_group("unit")
set_rundir(os.projectdir())
add_deps("athelas_configuration")

add_files(path.join(os.projectdir(), "test", "unit", "**.cpp"))

-- Unit tests compile only the production implementations they exercise.
for _, source in ipairs({
  "basis/nodal_basis.cpp",
  "bc/boundary_conditions.cpp",
  "composition/compdata.cpp",
  "eos/eos_ideal.cpp",
  "eos/eos_marshak.cpp",
  "eos/eos_paczynski.cpp",
  "eos/eos_polytropic.cpp",
  "geometry/mesh.cpp",
  "interface/params.cpp",
  "interface/state.cpp",
  "io/restart.cpp",
  "io/tables.cpp",
  "math/linear_algebra.cpp",
  "math/quadrature.cpp",
  "pgen/lua_validator.cpp",
  "pgen/problem_in.cpp",
  "timestepper/tableau.cpp",
}) do
  add_files(path.join(os.projectdir(), "src", source))
end

add_includedirs(athelas_includedirs)
add_includedirs(path.join(os.projectdir(), "test", "unit"))

add_packages("athelas_kokkos_kernels", "athelas_spiner", "athelas_eigen", "athelas_sol2", "athelas_hdf5")
if get_config("backend") == "openmp" then
  add_packages("openmp")
end
add_packages("catch2", { components = { "lib" } })
add_syslinks("stdc++exp")

set_runenv("OMP_PROC_BIND", "false")
set_runenv("OMP_PLACES", "threads")

add_tests("all", {
  runenvs = {
    OMP_NUM_THREADS = "1",
    OMP_PROC_BIND = "false",
    OMP_PLACES = "threads",
  },
})

if athelas_debug_mode then
  add_defines("ATHELAS_DEBUG")
end
