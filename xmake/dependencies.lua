-- Locate a submodule's git directory without depending on the name it was
-- registered under. The `.gitmodules` names do not match their paths
-- consistently (Kokkos is "External/Kokkos", the rest are lowercase), and
-- hard-coding a name silently degrades to the CMakeLists fallback below if it
-- ever changes. The description sandbox has no `io`, so the git directory is
-- found by globbing rather than by reading the submodule's `.git` file.
local function submodule_git_dir(relative_dir)
  local wanted = relative_dir:lower():gsub("\\", "/")
  local basename = wanted:gsub(".*/", "")
  local prefix = path.join(os.projectdir(), ".git", "modules"):gsub("\\", "/") .. "/"
  local fallback = nil

  -- Only the module roots, which sit one or two levels down ("x" or
  -- "external/x"). A recursive glob would also pick up the HEAD files under
  -- each module's logs/ and refs/ directories.
  local candidates = os.files(path.join(os.projectdir(), ".git", "modules", "*", "HEAD"))
  table.join2(candidates, os.files(path.join(os.projectdir(), ".git", "modules", "*", "*", "HEAD")))

  for _, headfile in ipairs(candidates) do
    local git_dir = path.directory(headfile)
    local name = git_dir:gsub("\\", "/"):sub(#prefix + 1):lower()
    if name == wanted then
      -- The registered name matches the path, modulo casing. Always preferred.
      return git_dir
    elseif name == basename then
      -- A module registered under a bare name. `.gitmodules` carries both
      -- "spiner" and "external/spiner", so this is only used when no exact
      -- match exists, and never decides between the two.
      fallback = git_dir
    end
  end

  return fallback
end

-- A revision identifier for a submodule, used only to invalidate the prebuilt
-- package when the dependency's source changes. A detached submodule's HEAD
-- file holds the raw commit SHA, so hashing it tracks the revision. The index
-- is deliberately not hashed: it changes whenever git refreshes its stat cache,
-- which would force a full dependency rebuild for no reason.
--
-- Known gap: a submodule left on a branch has HEAD = "ref: refs/heads/...",
-- which does not change across commits, so a bump there goes undetected.
-- `git submodule update` detaches by default, so this is the unusual case.
local function source_revision(relative_dir)
  local source_dir = path.join(os.projectdir(), relative_dir)
  if not os.isdir(source_dir) then
    return "missing"
  end

  local git_dir = submodule_git_dir(relative_dir)
  if git_dir then
    local headfile = path.join(git_dir, "HEAD")
    if os.isfile(headfile) then
      return hash.sha256(headfile)
    end
  end

  -- Keep source archives usable even when the submodule metadata is absent.
  local cmakelists = path.join(source_dir, "CMakeLists.txt")
  return os.isfile(cmakelists) and hash.sha256(cmakelists) or "unknown"
end

local backend = get_config("backend") or "openmp"

local cmake_build_type = "Release"
if is_mode("debug") or is_mode("allsan") or is_mode("tsan") then
  cmake_build_type = "Debug"
elseif is_mode("releasedbg", "relwithdebinfo") then
  cmake_build_type = "RelWithDebInfo"
end

-- The compiler is not part of these keys: xmake already folds the selected
-- toolchain into the package build hash. A bare system compiler upgrade with no
-- toolchain change is not detected; rerun `xmake f -c` after one.
local kokkos_revision = source_revision("external/Kokkos")
local kernels_revision = source_revision("external/kokkos-kernels")
local kokkos_debug_checks = athelas_debug_mode

add_requires("athelas_kokkos", {
  configs = {
    backend = backend,
    cmake_build_type = cmake_build_type,
    debug_checks = kokkos_debug_checks,
    source_revision = kokkos_revision,
  },
})

add_requires("athelas_kokkos_kernels", {
  configs = {
    backend = backend,
    cmake_build_type = cmake_build_type,
    kokkos_debug_checks = kokkos_debug_checks,
    kokkos_source_revision = kokkos_revision,
    source_revision = kernels_revision,
  },
})

add_requires("athelas_eigen", {
  configs = { source_revision = source_revision("external/eigen") },
})

add_requires("athelas_lua", { system = true })

add_requires("athelas_sol2", {
  configs = { source_revision = source_revision("external/sol2") },
})

add_requires("athelas_hdf5", { system = true })

add_requires("athelas_spiner", {
  configs = {
    cmake_build_type = cmake_build_type,
    source_revision = source_revision("external/spiner"),
  },
})

if backend == "openmp" then
  add_requires("openmp")
end

if has_config("unit_tests") then
  add_requires("catch2 3.9.0", { debug = is_mode("debug") })
end
