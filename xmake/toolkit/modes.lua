-- Extra build modes that xmake does not ship.
--
--   relwithdebinfo  -O2 -g -DNDEBUG          CMake's RelWithDebInfo
--   perf            -O3 -g -DNDEBUG          sampling profilers (no -pg)
--                   -fno-omit-frame-pointer
--
-- xmake's own profile mode instruments with -pg, which suits gprof but distorts
-- sampling profilers such as perf and VTune. The perf mode keeps full
-- optimization and just preserves frame pointers and symbols.
--
-- Link-time optimization is enabled for every optimized mode, including xmake's
-- built-in release and releasedbg, so that a mode switch does not silently
-- change whether LTO applies.
--
-- Usage:
--
--   set_allowedmodes("debug", "release", "relwithdebinfo", "perf")
--   add_rules("toolkit.modes")
--
-- Applied per target rather than at project scope, so a target that sets its own
-- optimization or symbols is left alone.

rule("toolkit.modes")
after_load(function(target)
  if is_mode("relwithdebinfo") then
    if not target:get("optimize") then
      target:set("optimize", "faster")
    end
    if not target:get("symbols") then
      target:set("symbols", "debug")
    end
    target:add("defines", "NDEBUG")
  elseif is_mode("perf") then
    if not target:get("optimize") then
      target:set("optimize", "fastest")
    end
    if not target:get("symbols") then
      target:set("symbols", "debug")
    end
    target:add("defines", "NDEBUG")
    target:add("cxxflags", "-fno-omit-frame-pointer")
    target:add("cflags", "-fno-omit-frame-pointer")
  end

  if is_mode("release", "releasedbg", "relwithdebinfo", "perf") then
    target:set("policy", "build.optimization.lto", true)
  end
end)
rule_end()
