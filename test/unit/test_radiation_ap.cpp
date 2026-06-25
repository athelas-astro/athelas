#include <catch2/catch_test_macros.hpp>

#include "radiation/rad_utilities.hpp"
#include "test_utils.hpp"

TEST_CASE("Radiation AP dissipation factor has expected limits",
          "[radiation]") {
  using athelas::radiation::ap_dissipation_factor;

  REQUIRE(soft_equal(ap_dissipation_factor(0.0), 1.0, 1.0e-15));
  REQUIRE(soft_equal(ap_dissipation_factor(1.0), 0.5, 1.0e-15));
  REQUIRE(soft_equal(ap_dissipation_factor(2.0, 0.25), 2.0 / 3.0, 1.0e-15));
  REQUIRE(ap_dissipation_factor(1.0e12) < 1.0e-11);
}

TEST_CASE("Radiation AP factor damps only LLF jump dissipation",
          "[radiation]") {
  using athelas::radiation::llf_flux;
  using athelas::radiation::LLFRiemannState;

  const LLFRiemannState left{.u = 3.0, .f = 11.0, .alpha = 5.0};
  const LLFRiemannState right{.u = 1.0, .f = 7.0, .alpha = 5.0};

  REQUIRE(soft_equal(llf_flux(left, right, 1.0), 14.0, 1.0e-15));
  REQUIRE(soft_equal(llf_flux(left, right, 0.25), 10.25, 1.0e-15));
  REQUIRE(soft_equal(llf_flux(left, right, 0.0), 9.0, 1.0e-15));
}
