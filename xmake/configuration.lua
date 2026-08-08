local builddir = get_config("builddir") or "build"
if not path.is_absolute(builddir) then
  builddir = path.join(os.projectdir(), builddir)
end
local generated_dir = path.join(builddir, "generated")

local loop_layout_tags = {
  MANUAL1D_LOOP = "loop_pattern_flatrange_tag",
  SIMDFOR_LOOP = "loop_pattern_simdfor_tag",
  MDRANGE_LOOP = "loop_pattern_mdrange_tag",
  TPTTR_LOOP = "loop_pattern_tpttr_tag",
  TPTVR_LOOP = "loop_pattern_tptvr_tag",
  TPTTRTVR_LOOP = "loop_pattern_tpttrtvr_tag",
}

local flat_loop_layout_tags = {
  MANUAL1D_LOOP = "loop_pattern_flatrange_tag",
  SIMDFOR_LOOP = "loop_pattern_simdfor_tag",
}

local inner_loop_layout_tags = {
  SIMDFOR_INNER_LOOP = "inner_loop_pattern_simdfor_tag",
  TVR_INNER_LOOP = "InnerLoopPatternTVR()",
}

local loop_layout = get_config("par_loop_layout") or "MANUAL1D_LOOP"
local flat_loop_layout = get_config("par_loop_flat_layout") or "MANUAL1D_LOOP"
local inner_loop_layout = get_config("par_loop_inner_layout") or "SIMDFOR_INNER_LOOP"

target("athelas_configuration")
set_kind("headeronly")
set_policy("build.always_update_configfiles", true)
set_configdir(generated_dir)
add_includedirs(generated_dir, { public = true })
on_config(function(target)
  -- Remove the header left in the source tree by earlier versions of the CMake
  -- build. src/ comes first on the include path, so a stale copy shadows the
  -- generated one and silently pins the loop layout.
  os.tryrm(path.join(os.projectdir(), "src", "loop_layout.hpp"))
end)
on_load(function(target)
  -- set_values() only populates the configuration menu; it does not reject a bad
  -- value. Without this check the generated header keeps an unsubstituted tag and
  -- the build fails much later with an unrelated compiler error. This runs in
  -- script scope because `raise` is not available in the description sandbox.
  local function layout_tag(tags, option, value)
    local tag = tags[value]
    if not tag then
      local names = {}
      for name, _ in pairs(tags) do
        table.insert(names, name)
      end
      table.sort(names)
      raise("unknown %s value '%s'; use one of %s", option, value, table.concat(names, ", "))
    end
    return tag
  end

  target:add("configfiles", path.join(os.projectdir(), "src", "(loop_layout.hpp.in)"), {
    filename = "loop_layout.hpp",
    pattern = "@(.-)@",
    variables = {
      PAR_LOOP_LAYOUT_TAG = layout_tag(loop_layout_tags, "par_loop_layout", loop_layout),
      PAR_LOOP_FLAT_LAYOUT_TAG = layout_tag(flat_loop_layout_tags, "par_loop_flat_layout", flat_loop_layout),
      PAR_LOOP_INNER_LAYOUT_TAG = layout_tag(inner_loop_layout_tags, "par_loop_inner_layout", inner_loop_layout),
    },
  })

  target:add("configfiles", path.join(os.projectdir(), "(lua_schema.hpp.in)"), {
    filename = "lua_schema.hpp",
    pattern = "@(.-)@",
    variables = {
      ATHELAS_SCHEMA_LUA_CONTENT = io.readfile(path.join(os.projectdir(), "inputs", "schema.lua")),
    },
  })
end)
