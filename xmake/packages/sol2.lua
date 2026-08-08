package("athelas_sol2")
set_kind("library", { headeronly = true })
set_homepage("https://github.com/ThePhD/sol2")
set_description("The sol2 submodule used by Athelas")
set_license("MIT")

set_sourcedir(path.join(os.scriptdir(), "..", "..", "external", "sol2"))

add_deps("athelas_lua", { system = true })

add_configs("source_revision", {
  description = "Submodule source revision used to invalidate the package cache",
  default = "unknown",
  type = "string",
})

on_install(function(package)
  local source_dir = package:sourcedir()
  if not os.isfile(path.join(source_dir, "include", "sol", "sol.hpp")) then
    raise("sol2 submodule is missing; run `git submodule update --init external/sol2`")
  end

  os.cp(path.join(source_dir, "include", "sol"), package:installdir("include"))
end)

on_test(function(package)
  assert(package:check_cxxsnippets({
    test = [[
          void test() {
            sol::state lua;
            const int answer = lua.script("return 6 * 7");
            assert(answer == 42);
          }
        ]],
  }, {
    configs = { languages = "c++23" },
    includes = { "cassert", "sol/sol.hpp" },
  }))
end)
package_end()
