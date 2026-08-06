-- Stamp build provenance into a generated source file.
--
-- Renders a template into the build directory before every build of the target,
-- substituting @NAME@ placeholders with the current git commit, compiler,
-- timestamp, mode, architecture, and platform. The rendered file is added to
-- the target's sources automatically.
--
-- The rendered output is rewritten only when something other than the timestamp
-- changes, so an unchanged rebuild does not relink.
--
-- Usage:
--
--   target("app")
--     add_rules("toolkit.provenance")
--     set_values("toolkit.provenance.template", "src/build_info.cpp.in")
--
-- Optional; defaults to <builddir>/provenance:
--
--     set_values("toolkit.provenance.outputdir", "some/other/dir")
--
-- The output keeps the template's name with a trailing ".in" removed, so
-- build_info.cpp.in renders to build_info.cpp.
--
-- Available placeholders: GIT_HASH, COMPILER, OPTIMIZATION, ARCH, OS,
-- BUILD_TIMESTAMP. An unknown placeholder in the template is an error.

local placeholders = { "GIT_HASH", "COMPILER", "OPTIMIZATION", "ARCH", "OS", "BUILD_TIMESTAMP" }

-- Resolved here rather than in the rule scripts: the description sandbox is
-- where get_config is available, and the locals below are captured as upvalues
-- by the rule closures.
local builddir = get_config("builddir") or "build"
if not path.is_absolute(builddir) then
  builddir = path.join(os.projectdir(), builddir)
end
local default_outputdir = path.join(builddir, "provenance")

-- Values arrive as a list even when a single value is set.
local function value_of(target, name)
  local value = target:values(name)
  if type(value) == "table" then
    value = value[1]
  end
  return value
end

local function template_file(target)
  local template = value_of(target, "toolkit.provenance.template")
  if not template then
    raise('target(%s): add_rules("toolkit.provenance") requires set_values("toolkit.provenance.template", <file>)', target:name())
  end
  return path.absolute(template, os.projectdir())
end

local function output_file(target)
  local outputdir = value_of(target, "toolkit.provenance.outputdir")
  outputdir = outputdir and path.absolute(outputdir, os.projectdir()) or default_outputdir
  local name = path.filename(template_file(target)):gsub("%.in$", "")
  return path.join(outputdir, name)
end

rule("toolkit.provenance")
on_load(function(target)
  target:add("files", output_file(target), { always_added = true })
end)
before_build(function(target)
  local template_path = template_file(target)
  local output_path = output_file(target)
  local state_path = output_path .. ".state"

  if not os.isfile(template_path) then
    raise("target(%s): provenance template not found: %s", target:name(), template_path)
  end

  local compiler, compiler_name = target:tool("cxx")
  local compiler_description = compiler_name
  local compiler_output = try({
    function()
      return os.iorunv(compiler, { "--version" })
    end,
  })
  if compiler_output then
    compiler_description = compiler_output:split("\n")[1]:trim()
  end

  local git_hash = "unknown"
  local git_output = try({
    function()
      return os.iorunv("git", { "log", "-1", "--format=%h" }, { curdir = os.projectdir() })
    end,
  })
  if git_output then
    git_hash = git_output:trim()
  end

  local template = io.readfile(template_path)
  local variables = {
    GIT_HASH = git_hash,
    COMPILER = compiler_description,
    OPTIMIZATION = get_config("mode") or "release",
    ARCH = target:arch(),
    OS = target:plat(),
  }

  -- The timestamp is deliberately excluded from the state. Including it would
  -- rewrite the file, and so force a relink, on every build.
  local state = table.concat({
    variables.GIT_HASH,
    variables.COMPILER,
    variables.OPTIMIZATION,
    variables.ARCH,
    variables.OS,
    template,
  }, "\n")

  local cached_state
  if os.isfile(state_path) then
    cached_state = io.readfile(state_path)
  end
  if cached_state == state and os.isfile(output_path) then
    return
  end

  variables.BUILD_TIMESTAMP = os.date("%Y-%m-%d %H:%M:%S %Z")
  local content = template:gsub("@([A-Z_]+)@", function(name)
    local replacement = variables[name]
    if not replacement then
      raise("target(%s): unknown provenance placeholder @%s@ in %s; available: %s", target:name(), name, template_path, table.concat(placeholders, ", "))
    end
    -- Rendered into a C++ string literal.
    return replacement:gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\n", "\\n")
  end)

  os.mkdir(path.directory(output_path))
  io.writefile(output_path, content)
  io.writefile(state_path, state)
end)
rule_end()
