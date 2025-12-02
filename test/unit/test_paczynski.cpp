#include <print>

#include <catch2/catch_test_macros.hpp>

#include "eos.hpp"
#include "test_utils.hpp"

#include "eos/eos_variant.hpp"

TEST_CASE("Paczynski EOS", "[paczynski]") {

  using athelas::eos::EOS;
  using athelas::eos::Paczynski;

  // values taken from SNEC
  const double rho_in = 125595.39051366;
  const double temp_in = 1268613166.2349212;
  const double e_in = 91989082019428928.0; // erg / g
  const double ne_in = 3.7531582194195113e28;
  const double N_in = 3.2495671287613204e22;
  const double ybar_in = 9.1959724075803901;
  const double ye_in = 0.499828389872625;
  const double sigma1 = 0.0;
  const double sigma2 = 0.0;
  const double sigma3 = 0.0;
  const double e_ioncorr_in = 287375046768323.38;
  const double pressure_ans = 7.3548947139e21;
  const double ffactor_ans = 1.635712907366;
  const double temp_guess = temp_in + temp_in / 10.0;
  const double cs_ans = std::sqrt(95956495901503152.0);

  // Things for eos
  const double abstol = 1.0e-16;
  const double reltol = 1.0e-16;
  const double max_iters = 100;
  const EOS eos = Paczynski(abstol, reltol, max_iters);
  const double tau = 1.0 / rho_in;
  const double vel = 0.0;
  const double EmT = e_in;
  double lambda[8];
  lambda[0] = N_in;
  lambda[1] = ye_in;
  lambda[2] = ybar_in;
  lambda[3] = sigma1;
  lambda[4] = sigma2;
  lambda[5] = sigma3;
  lambda[6] = e_ioncorr_in;
  lambda[7] = temp_guess;

  // Inputs differ slightly from SNEC so we don't require a strong equality.
  const double tol = 1.0e-2;

  SECTION("Temperature inversion") {
    const double temperature =
        temperature_from_conserved(&eos, tau, vel, EmT, lambda);
    REQUIRE(soft_equal(temperature, temp_in, tol));
  }

  SECTION("Computing pressure") {
    const double pressure =
        pressure_from_conserved(&eos, tau, vel, EmT, lambda);
    REQUIRE(soft_equal(pressure_ans, pressure, tol));
  }

  SECTION("Sound speed") {
    const double cs = sound_speed_from_conserved(&eos, tau, vel, EmT, lambda);
    REQUIRE(soft_equal(cs, cs_ans, tol));
  }

  SECTION("Sie from rho pressure") {
    const double pressure =
        pressure_from_conserved(&eos, tau, vel, EmT, lambda);
    const double sie =
        sie_from_density_pressure(&eos, rho_in, pressure, lambda);
    REQUIRE(soft_equal(sie, e_in));
  }

  SECTION("Sie from rho pressure -> pressure") {
    const double pressure =
        pressure_from_conserved(&eos, tau, vel, EmT, lambda);
    const double sie =
        sie_from_density_pressure(&eos, rho_in, pressure, lambda);
    const double pressure_2 =
        pressure_from_conserved(&eos, tau, vel, sie, lambda);
    REQUIRE(soft_equal(pressure, pressure_2));
  }

  SECTION("temperature -> pressure -> sie -> temperature") {
    const double temperature =
        temperature_from_density_sie(&eos, rho_in, e_in, lambda);
    const double pressure =
        pressure_from_density_temperature(&eos, rho_in, temperature, lambda);
    const double sie =
        sie_from_density_pressure(&eos, rho_in, pressure, lambda);
    const double new_temperature =
        temperature_from_density_sie(&eos, rho_in, sie, lambda);
    REQUIRE(soft_equal(temperature, new_temperature));
  }

  SECTION("pressure -> temperature -> pressure") {
    const double pressure =
        pressure_from_density_temperature(&eos, rho_in, temp_in, lambda);
    const double temperature =
        temperature_from_density_pressure(&eos, rho_in, pressure, lambda);
    const double new_pressure =
        pressure_from_density_temperature(&eos, rho_in, temperature, lambda);
    REQUIRE(soft_equal(pressure, new_pressure));
  }

  SECTION("pressure -> temperature -> sie -> temperature") {
    const double temperature =
        temperature_from_density_pressure(&eos, rho_in, pressure_ans, lambda);
    const double sie =
        sie_from_density_temperature(&eos, rho_in, temperature, lambda);
    const double new_temperature =
        temperature_from_density_sie(&eos, rho_in, sie, lambda);
    REQUIRE(soft_equal(temperature, new_temperature));
  }

  SECTION("temperature two ways") {
    const double temperature =
        temperature_from_density_pressure(&eos, rho_in, pressure_ans, lambda);
    const double new_temperature =
        temperature_from_density_sie(&eos, rho_in, e_in, lambda);
    REQUIRE(soft_equal(temperature, temp_in, tol));
    REQUIRE(soft_equal(temp_in, new_temperature, tol));
    REQUIRE(soft_equal(temperature, new_temperature, tol));
  }
}
