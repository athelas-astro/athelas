package("athelas_kokkos")
set_homepage("https://kokkos.org/kokkos-core-wiki/")
set_description("The Kokkos submodule used by Athelas")
set_license("Apache-2.0")

set_sourcedir(path.join(os.scriptdir(), "..", "..", "external", "Kokkos"))

add_deps("cmake")
add_links("kokkoscontainers", "kokkossimd", "kokkoscore")

add_configs("backend", {
  description = "Kokkos execution backend",
  default = "openmp",
  type = "string",
  values = { "openmp", "cuda", "hip" },
})
add_configs("cmake_build_type", {
  description = "CMake build type",
  default = "Release",
  type = "string",
  values = { "Debug", "Release", "RelWithDebInfo" },
})
add_configs("debug_checks", {
  description = "Enable Kokkos debug checks",
  default = false,
  type = "boolean",
})
add_configs("source_revision", {
  description = "Submodule source revision used to invalidate the package cache",
  default = "unknown",
  type = "string",
})

on_load(function(package)
  local backend = package:config("backend")
  if backend ~= "openmp" then
    raise("Kokkos backend '%s' is not supported yet", backend)
  end
  package:add("deps", "openmp")
end)

on_install("linux", function(package)
  local source_dir = package:sourcedir()
  if not os.isfile(path.join(source_dir, "CMakeLists.txt")) then
    raise("Kokkos submodule is missing; run `git submodule update --init external/Kokkos`")
  end

  -- Build straight from the submodule. package.tools.cmake configures from the
  -- current directory into the given build directory, so this is out-of-source
  -- and writes nothing into external/Kokkos. Copying the tree first would only
  -- duplicate several hundred MB per variant.
  os.cd(source_dir)

  local enabled = package:config("debug_checks") and "ON" or "OFF"
  local backend = package:config("backend")
  local configs = {
    "-DCMAKE_BUILD_TYPE=" .. package:config("cmake_build_type"),
    -- Not forced elsewhere: xmake's cmake tool only overrides the compiler
    -- for Windows/MinGW/cross/Apple builds, so a native Linux build like this
    -- one otherwise falls through to CMake's own detection, which honors
    -- $CC/$CXX. That let a stray env var silently build this against a
    -- different compiler than the rest of the project.
    "-DCMAKE_C_COMPILER=" .. package:build_getenv("cc"),
    "-DCMAKE_CXX_COMPILER=" .. package:build_getenv("cxx"),
    "-DCMAKE_CXX_STANDARD=23",
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
    "-DCMAKE_CXX_EXTENSIONS=OFF",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DKokkos_ENABLE_OPENMP=" .. (backend == "openmp" and "ON" or "OFF"),
    "-DKokkos_ENABLE_SERIAL=OFF",
    "-DKokkos_ENABLE_CUDA=" .. (backend == "cuda" and "ON" or "OFF"),
    "-DKokkos_ENABLE_HIP=" .. (backend == "hip" and "ON" or "OFF"),
    "-DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON",
    "-DKokkos_ENABLE_IMPL_VIEW_LEGACY=OFF",
    "-DKokkos_ENABLE_TESTS=OFF",
    "-DKokkos_ENABLE_EXAMPLES=OFF",
    "-DKokkos_ENABLE_BENCHMARKS=OFF",
    "-DKokkos_ENABLE_DEBUG=" .. enabled,
    "-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=" .. enabled,
  }

  -- Kokkos does its own find_package(OpenMP REQUIRED) in cmake/kokkos_tpls.cmake,
  -- a separate CMake configure with none of the flags below inherited from
  -- Athelas's own CMakeLists.txt. Stock FindOpenMP does not reliably locate
  -- Clang's OpenMP runtime on every platform, and the whole project links
  -- against libstdc++ (see stdc++exp in targets/athelas.lua), so Kokkos must
  -- be built against it too rather than Clang's libc++ default. Same fix as
  -- the Clang branch in CMakeLists.txt, applied here since that one only
  -- covers Athelas's own configure, not Kokkos's independent one.
  if package:has_tool("cxx", "clang") then
    table.insert(configs, "-DCMAKE_CXX_FLAGS=-stdlib=libstdc++ -fopenmp=libomp")
  end

  import("package.tools.cmake").install(package, configs, {
    builddir = path.join(package:builddir(), "cmake-build"),
  })
end)

on_test(function(package)
  assert(package:check_cxxsnippets({
    test = [[
          void test(int argc, char **argv) {
            Kokkos::initialize(argc, argv);
            Kokkos::finalize();
          }
        ]],
  }, {
    configs = { languages = "c++23" },
    includes = "Kokkos_Core.hpp",
  }))
end)
package_end()
