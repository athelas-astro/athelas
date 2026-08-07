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

  -- Kokkos's independent CMake configure needs the same clang+libstdc++ fix
  -- as CMakeLists.txt (its own find_package(OpenMP) doesn't inherit ours).
  -- Clang's libomp isn't reliably on the default linker path -- breaking
  -- even CMake's own compiler-works check -- and guessing the path (e.g. via
  -- `-print-file-name=`) isn't reliable either (confirmed by CI). Reuse the
  -- "libomp" dependency's resolved install path instead.
  if package:has_tool("cxx", "clang") then
    local openmp_dep = package:dep("openmp")
    local libomp_dep = openmp_dep and openmp_dep:dep("libomp")
    local fetchinfo = libomp_dep and libomp_dep:fetch()

    local cxxflags_extra = ""
    if fetchinfo and fetchinfo.linkdirs then
      for _, linkdir in ipairs(fetchinfo.linkdirs) do
        table.insert(configs, "-DCMAKE_EXE_LINKER_FLAGS=-L" .. linkdir)
        table.insert(configs, "-DCMAKE_SHARED_LINKER_FLAGS=-L" .. linkdir)
        table.insert(configs, "-DCMAKE_LIBRARY_PATH=" .. linkdir)
      end
      for _, includedir in ipairs(fetchinfo.sysincludedirs or {}) do
        table.insert(configs, "-DCMAKE_INCLUDE_PATH=" .. includedir)
        cxxflags_extra = cxxflags_extra .. " -isystem " .. includedir
      end
    else
      -- Fallback if the dependency has no linkdirs: ask the compiler itself.
      local cxx = package:build_getenv("cxx")
      local libomp = try({
        function()
          return os.iorunv(cxx, { "-stdlib=libstdc++", "-fopenmp", "-print-file-name=libomp.so" })
        end,
      })
      if libomp then
        libomp = libomp:trim()
        if os.isfile(libomp) then
          local libomp_dir = path.directory(libomp)
          table.insert(configs, "-DCMAKE_EXE_LINKER_FLAGS=-L" .. libomp_dir)
          table.insert(configs, "-DCMAKE_SHARED_LINKER_FLAGS=-L" .. libomp_dir)
          table.insert(configs, "-DCMAKE_LIBRARY_PATH=" .. libomp_dir)
        else
          wprint(
            "athelas_kokkos: could not resolve libomp's install location " .. "via the xmake dependency or `%s -print-file-name=libomp.so`; " .. "clang OpenMP link may fail.",
            cxx
          )
        end
      end
    end

    table.insert(configs, "-DCMAKE_CXX_FLAGS=-stdlib=libstdc++ -fopenmp=libomp" .. cxxflags_extra)
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
