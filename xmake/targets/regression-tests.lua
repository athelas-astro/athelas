-- Derived from the suite on disk so the two cannot drift apart. Each test is
-- named after its module with the "test_" prefix removed, so test_sod.py is
-- runnable as `xmake test regression_tests/sod`.
local regression_cases = {}
for _, source in ipairs(os.files(path.join(os.projectdir(), "test", "regression", "test_*.py"))) do
  local module = path.basename(source)
  table.insert(regression_cases, { name = module:gsub("^test_", ""), module = module })
end
table.sort(regression_cases, function(lhs, rhs)
  return lhs.name < rhs.name
end)

target("regression_tests")
set_kind("phony")
set_default(false)
set_group("regression")
add_deps("athelas")

for _, case in ipairs(regression_cases) do
  add_tests(case.name, {
    runargs = { "--test", case.module },
  })
end

on_test(function(target, opt)
  import("lib.detect.find_tool")

  local project_dir = os.projectdir()
  local regression_dir = path.join(project_dir, "test", "regression")
  local harness = path.join(regression_dir, "run_regression_tests.py")
  local executable = path.absolute(target:dep("athelas"):targetfile())
  local args = {}

  local runner = find_tool("uv")
  if runner then
    args = {
      "run",
      "--project",
      path.join(project_dir, "scripts", "python", "athelas_tools"),
      "--frozen",
      "python",
      harness,
    }
  else
    runner = find_tool("python3") or find_tool("python")
    if not runner then
      raise("Regression tests require uv, python3, or python")
    end
    table.insert(args, harness)
  end

  table.join2(args, opt.runargs)
  table.insert(args, "--executable")
  table.insert(args, executable)

  local status = os.execv(runner.program, args, {
    try = true,
    curdir = regression_dir,
    envs = {
      MPLBACKEND = "Agg",
      OMP_NUM_THREADS = "1",
      OMP_PROC_BIND = "false",
      OMP_PLACES = "threads",
    },
  })
  return status == 0
end)
