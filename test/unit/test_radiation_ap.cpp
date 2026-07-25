#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "constants.hpp"
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

TEST_CASE("Radiation AP LLF face flux remains conservative", "[radiation]") {
  using athelas::radiation::llf_flux;
  using athelas::radiation::LLFRiemannState;

  const LLFRiemannState left{.u = 2.5, .f = -4.0, .alpha = 7.0};
  const LLFRiemannState right{.u = 0.5, .f = 6.0, .alpha = 7.0};

  for (const double beta : {1.0, 0.2}) {
    const double face_flux = llf_flux(left, right, beta);
    // A shared face contributes with opposite signs to the two adjacent P0
    // cell averages, independently of the AP damping of LLF dissipation.
    const double left_increment = -face_flux;
    const double right_increment = face_flux;
    REQUIRE(soft_equal(left_increment + right_increment, 0.0, 1.0e-15));
  }
}

TEST_CASE("Perpendicular radiation pressure derivatives match finite "
          "differences",
          "[radiation]") {
  using athelas::constants::c_cgs;
  using athelas::radiation::p_rad_perp;
  using athelas::radiation::p_rad_perp_with_derivatives;

  constexpr double E = 4.0;
  constexpr double h_E = 1.0e-6 * E;
  constexpr double h_F = 1.0e-6 * c_cgs * E;

  for (const double reduced_flux : {-0.95, -0.4, 0.0, 0.25, 0.85}) {
    const double F = reduced_flux * c_cgs * E;
    const auto pressure = p_rad_perp_with_derivatives(E, F);
    const double finite_difference_E =
        (p_rad_perp(E + h_E, F) - p_rad_perp(E - h_E, F)) / (2.0 * h_E);
    const double finite_difference_F =
        (p_rad_perp(E, F + h_F) - p_rad_perp(E, F - h_F)) / (2.0 * h_F);

    REQUIRE(pressure.pressure ==
            Catch::Approx(p_rad_perp(E, F)).epsilon(1.0e-14));
    REQUIRE(pressure.d_pressure_dE ==
            Catch::Approx(finite_difference_E).epsilon(1.0e-9));
    REQUIRE(pressure.d_pressure_dF ==
            Catch::Approx(finite_difference_F).epsilon(1.0e-9).margin(1.0e-20));
  }
}
