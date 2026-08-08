set_project("athelas")
set_version("0.10.0")

set_config("builddir", "build")
set_languages("cxx23")
set_allowedmodes("debug", "release", "releasedbg", "profile", "relwithdebinfo", "perf", "asan", "tsan", "ubsan", "lsan", "allsan")

-- Matches CMake RelWithDebInfo.
set_defaultmode("relwithdebinfo")

-- relwithdebinfo, perf, the sanitizer modes, and the build-provenance rule come
-- from xmake/toolkit, which is project-agnostic and can be shared
-- See xmake/toolkit/README.md.
includes("xmake/toolkit")
add_moduledirs("xmake/modules")

add_rules("mode.debug", "mode.release", "mode.releasedbg", "mode.profile")
add_rules("toolkit.modes", "toolkit.sanitizers")

-- Set up which rules get debug builds.
athelas_sanitizer = is_mode("asan", "tsan", "ubsan", "lsan", "allsan")
athelas_debug_mode = is_mode("debug", "asan") or athelas_sanitizer

-- Include directories
athelas_includedirs = { path.join(os.projectdir(), "src") }
for _, dir in ipairs(os.dirs(path.join(os.projectdir(), "src", "*"))) do
  table.insert(athelas_includedirs, dir)
end

-- Set the backend. Currently only openmp is supported.
option("backend")
set_default("openmp")
set_values("openmp", "cuda", "hip")
set_description("Kokkos execution backend")
set_showmenu(true)

-- GPU builds not yet supported.
after_check(function(option)
  local backend = option:value()
  if backend ~= "openmp" then
    raise("Kokkos backend '%s' is not supported yet; configure with --backend=openmp", backend)
  end
end)
option_end()

-- Controls layout of parallel loops.
-- Use the default.
option("par_loop_layout")
set_default("MANUAL1D_LOOP")
set_values("MANUAL1D_LOOP", "SIMDFOR_LOOP", "MDRANGE_LOOP", "TPTTR_LOOP", "TPTVR_LOOP", "TPTTRTVR_LOOP")
set_description("Default loop layout for the parallel_for wrapper")
set_showmenu(true)
option_end()

option("par_loop_flat_layout")
set_default("MANUAL1D_LOOP")
set_values("MANUAL1D_LOOP", "SIMDFOR_LOOP")
set_description("Default loop layout for the one-dimensional parallel_for wrapper")
set_showmenu(true)
option_end()

option("par_loop_inner_layout")
set_default("SIMDFOR_INNER_LOOP")
set_values("SIMDFOR_INNER_LOOP", "TVR_INNER_LOOP")
set_description("Default loop layout for the inner parallel_for wrapper")
set_showmenu(true)
option_end()

-- Enable unit tests
option("unit_tests")
set_default(false)
set_description("Build the Catch2 unit test suite")
set_showmenu(true)
option_end()

-- Allow xmake to run regression tests.
option("regression_tests")
set_default(false)
set_description("Register the Python regression test suite")
set_showmenu(true)
option_end()

-- Package setups.
-- These package names are called athelas_kokkos, for example, to avoid
-- collisions with xmake package manager packages. This allows us to
-- provide Kokkos and friends alongside the code without requiring an
-- internet connection for a fresh build.
includes("xmake/packages/kokkos.lua")
includes("xmake/packages/kokkos-kernels.lua")
includes("xmake/packages/eigen.lua")
includes("xmake/packages/lua.lua")
includes("xmake/packages/sol2.lua")
includes("xmake/packages/hdf5.lua")
includes("xmake/packages/spiner.lua")

-- Tools for configuring dependencies.
includes("xmake/dependencies.lua")
includes("xmake/configuration.lua")

includes("xmake/targets/athelas.lua")

-- Tests
if has_config("unit_tests") then
  includes("xmake/targets/unit-tests.lua")
end

if has_config("regression_tests") then
  includes("xmake/targets/regression-tests.lua")
end
