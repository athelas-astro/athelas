package("athelas_eigen")
set_kind("library", { headeronly = true })
set_homepage("https://eigen.tuxfamily.org/")
set_description("The Eigen submodule used by Athelas")
set_license("MPL-2.0")

set_sourcedir(path.join(os.scriptdir(), "..", "..", "external", "eigen"))

add_configs("source_revision", {
  description = "Submodule source revision used to invalidate the package cache",
  default = "unknown",
  type = "string",
})

on_install(function(package)
  local source_dir = package:sourcedir()
  if not os.isfile(path.join(source_dir, "Eigen", "Dense")) then
    raise("Eigen submodule is missing; run `git submodule update --init external/eigen`")
  end

  os.cp(path.join(source_dir, "Eigen"), package:installdir("include"))
  os.cp(path.join(source_dir, "unsupported"), package:installdir("include"))
end)

on_test(function(package)
  assert(package:check_cxxsnippets({
    test = [[
          void test() {
            Eigen::Matrix2d matrix;
            matrix << 1.0, 2.0, 3.0, 4.0;
            (void)matrix.determinant();
          }
        ]],
  }, {
    configs = { languages = "c++23" },
    includes = "Eigen/Dense",
  }))
end)
package_end()
