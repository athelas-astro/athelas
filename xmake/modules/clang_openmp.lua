-- Shared by athelas_kokkos and athelas_kokkos_kernels: each runs its own
-- independent CMake configure that ends up calling find_package(OpenMP)
-- (Kokkos directly; Kokkos-Kernels transitively via find_dependency(OpenMP)
-- in Kokkos's exported CMake config), so each needs the same clang fix as
-- CMakeLists.txt's own Clang branch. Clang's libomp isn't reliably on the
-- default linker path -- breaking even CMake's own compiler-works check --
-- and guessing the path (e.g. via `-print-file-name=`) isn't reliable either
-- (confirmed by CI). Reuse the "libomp" dependency's resolved install path
-- instead; `package` must declare a direct "openmp" dep for this to work.
function apply(package, configs)
  if not package:has_tool("cxx", "clang") then
    return
  end

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
          "%s: could not resolve libomp's install location via the xmake " .. "dependency or `%s -print-file-name=libomp.so`; clang OpenMP " .. "link may fail.",
          package:name(),
          cxx
        )
      end
    end
  end

  table.insert(configs, "-DCMAKE_CXX_FLAGS=-stdlib=libstdc++ -fopenmp=libomp" .. cxxflags_extra)
end
