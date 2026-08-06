package("athelas_lua")
set_homepage("https://www.lua.org/")
set_description("A compatible system Lua library used by Athelas")
set_license("MIT")

on_fetch("linux", function(package, opt)
  if opt.system and package.find_package then
    for _, version in ipairs({ "5.4", "5.3", "5.2", "5.1", "5.0" }) do
      local compact = version:gsub("%.", "")
      for _, name in ipairs({ "lua" .. version, "lua-" .. version, "lua" .. compact }) do
        local result = package:find_package("pkgconfig::" .. name)
        if result then
          return result
        end
      end
    end
  end
end)

on_test(function(package)
  assert(package:has_cfuncs("lua_gettop", {
    configs = { languages = "c23" },
    includes = "lua.h",
  }))
end)
package_end()
