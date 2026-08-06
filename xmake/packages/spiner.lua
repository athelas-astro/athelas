package("athelas_spiner")
set_kind("library", { headeronly = true })
set_homepage("https://github.com/lanl/spiner")
set_description("The Spiner submodule used by Athelas")
set_license("BSD-3-Clause")

set_sourcedir(path.join(os.scriptdir(), "..", "..", "external", "spiner"))

add_deps("cmake")

add_configs("cmake_build_type", {
  description = "CMake build type",
  default = "Release",
  type = "string",
  values = { "Debug", "Release", "RelWithDebInfo" },
})
add_configs("source_revision", {
  description = "Submodule source revision used to invalidate the package cache",
  default = "unknown",
  type = "string",
})

on_install("linux", function(package)
  local source_dir = package:sourcedir()
  if not os.isfile(path.join(source_dir, "CMakeLists.txt")) then
    raise("Spiner submodule is missing; run `git submodule update --init external/spiner`")
  end

  -- Build straight from the submodule; see the note in kokkos.lua.
  os.cd(source_dir)

  local configs = {
    "-DCMAKE_BUILD_TYPE=" .. package:config("cmake_build_type"),
    -- See the note in kokkos.lua: not forced elsewhere on native Linux.
    "-DCMAKE_C_COMPILER=" .. package:build_getenv("cc"),
    "-DCMAKE_CXX_COMPILER=" .. package:build_getenv("cxx"),
    "-DSPINER_USE_HDF=OFF",
    "-DSPINER_BUILD_TESTS=OFF",
  }

  import("package.tools.cmake").install(package, configs, {
    builddir = path.join(package:builddir(), "cmake-build"),
  })
end)

-- Define a simple test ran on package installation.
-- Compiles and runs; a second, compile-only on_test here would be silently
-- replaced by this one.
on_test(function(package)
  assert(package:check_cxxsnippets({
    test = [[
        int main() {
          const Spiner::RegularGrid1D<double> grid(0.0, 1.0, 5);
          assert(grid.x(2) == 0.5);
          return 0;
        }
      ]],
  }, {
    tryrun = true,
    configs = { languages = "c++23" },
    includes = { "cassert", "spiner/regular_grid_1d.hpp" },
  }))
end)
package_end()
