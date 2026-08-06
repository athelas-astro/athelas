package("athelas_hdf5")
set_homepage("https://www.hdfgroup.org/solutions/hdf5/")
set_description("The system HDF5 C++ and high-level libraries used by Athelas")
set_license("BSD-3-Clause")

on_fetch("linux", function(package, opt)
  if opt.system and package.find_package then
    for _, name in ipairs({ "hdf5_hl_cpp", "hdf5_serial_hl_cpp" }) do
      local result = package:find_package("pkgconfig::" .. name)
      if result then
        return result
      end
    end

    -- Debian and Ubuntu ship only a base hdf5-serial.pc file even though the
    -- C++, C high-level, and C++ high-level libraries are installed beside it.
    for _, name in ipairs({ "hdf5-serial", "hdf5" }) do
      local result = package:find_package("pkgconfig::" .. name)
      if result then
        result = table.clone(result)
        result.links = {
          "hdf5_hl_cpp",
          "hdf5_cpp",
          "hdf5_hl",
          "hdf5",
        }
        return result
      end
    end
  end
end)

on_test(function(package)
  assert(package:check_cxxsnippets({
    test = [[
          void test() {
            H5::H5Library::open();
          }
        ]],
  }, {
    configs = { languages = "c++23" },
    includes = "H5Cpp.h",
  }))
end)
package_end()
