package("athelas_kokkos_kernels")
set_homepage("https://github.com/kokkos/kokkos-kernels")
set_description("The Kokkos Kernels submodule used by Athelas")
set_license("Apache-2.0")

set_sourcedir(path.join(os.scriptdir(), "..", "..", "external", "kokkos-kernels"))

add_deps("cmake")
add_links("kokkoskernels")

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
add_configs("kokkos_debug_checks", {
  description = "Enable debug checks in the Kokkos dependency",
  default = false,
  type = "boolean",
})
add_configs("kokkos_source_revision", {
  description = "Kokkos source revision used to invalidate the dependency cache",
  default = "unknown",
  type = "string",
})
add_configs("source_revision", {
  description = "Submodule source revision used to invalidate the package cache",
  default = "unknown",
  type = "string",
})

on_load(function(package)
  package:add("deps", "athelas_kokkos", {
    configs = {
      backend = package:config("backend"),
      cmake_build_type = package:config("cmake_build_type"),
      debug_checks = package:config("kokkos_debug_checks"),
      source_revision = package:config("kokkos_source_revision"),
    },
  })
end)

on_install("linux", function(package)
  local source_dir = package:sourcedir()
  if not os.isfile(path.join(source_dir, "CMakeLists.txt")) then
    raise("Kokkos Kernels submodule is missing; run `git submodule update --init external/kokkos-kernels`")
  end

  -- Build straight from the submodule; see the note in kokkos.lua.
  os.cd(source_dir)

  local configs = {
    "-DCMAKE_BUILD_TYPE=" .. package:config("cmake_build_type"),
    -- See the note in kokkos.lua: not forced elsewhere on native Linux.
    "-DCMAKE_C_COMPILER=" .. package:build_getenv("cc"),
    "-DCMAKE_CXX_COMPILER=" .. package:build_getenv("cxx"),
    "-DCMAKE_CXX_STANDARD=23",
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
    "-DCMAKE_CXX_EXTENSIONS=OFF",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DKokkosKernels_ENABLE_TESTS=OFF",
    "-DKokkosKernels_ENABLE_EXAMPLES=OFF",
    "-DKokkosKernels_ENABLE_PERFTESTS=OFF",
    "-DKokkosKernels_ENABLE_BENCHMARKS=OFF",
    "-DKokkosKernels_INST_DOUBLE=ON",
    "-DKokkosKernels_INST_FLOAT=OFF",
    "-DKokkosKernels_INST_COMPLEX_DOUBLE=OFF",
    "-DKokkosKernels_INST_COMPLEX_FLOAT=OFF",
    "-DKokkosKernels_INST_LAYOUTLEFT=OFF",
    "-DKokkosKernels_INST_LAYOUTRIGHT=ON",
    "-DKokkosKernels_INST_ORDINAL_INT=ON",
    "-DKokkosKernels_INST_ORDINAL_INT64_T=OFF",
    "-DKokkosKernels_INST_OFFSET_INT=ON",
    "-DKokkosKernels_INST_OFFSET_SIZE_T=OFF",
    "-DKokkosKernels_ENABLE_ALL_COMPONENTS=OFF",
    "-DKokkosKernels_ENABLE_COMPONENT_BATCHED=ON",
    "-DKokkosKernels_ENABLE_COMPONENT_BLAS=ON",
    "-DKokkosKernels_ENABLE_TPL_BLAS=OFF",
    "-DKokkosKernels_ENABLE_TPL_LAPACK=OFF",
    "-DKokkosKernels_ENABLE_TPL_CUBLAS=OFF",
    "-DKokkosKernels_ENABLE_TPL_CUSOLVER=OFF",
    "-DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF",
  }

  import("package.tools.cmake").install(package, configs, {
    builddir = path.join(package:builddir(), "cmake-build"),
  })
end)

on_test(function(package)
  assert(package:check_cxxsnippets({
    test = [[
          void test() {
            Kokkos::initialize();
            {
              Kokkos::View<double *> x("x", 1);
              (void)KokkosBlas::dot(x, x);
            }
            Kokkos::finalize();
          }
        ]],
  }, {
    configs = { languages = "c++23" },
    includes = { "Kokkos_Core.hpp", "KokkosBlas1_dot.hpp" },
  }))
end)
package_end()
