/**
 * @file constants.hpp
 * --------------
 *
 * @brief Physical constants
 *
 * Hopefully consistent with astropy
 * TODO(astrobarker): implement different unit systems
 */

#pragma once

#include <cmath>
#include <numbers>

namespace athelas::constants {

static constexpr double PI = std::numbers::pi;
static constexpr double FOURPI = 4.0 * std::numbers::pi;
static constexpr double FOUR_THIRDS_PI = FOURPI / 3.0;
static constexpr double G_GRAV = 6.674299999999999e-8; // cgs
static constexpr double L_sun = 3.828e33; // cgs
static constexpr double M_sun = 1.98840987e+33; // cgs
static constexpr double sigma_sb = 5.670374419184431e-5; // cgs
static constexpr double a = 7.5657332502800007e-15; // cgs
static constexpr double k_B = 1.380649e-16; // cgs
static constexpr double k_Bev = 8.61733326e-5; // eV / K
static constexpr double m_e = 9.1093837015e-28; // cgs
static constexpr double m_p = 1.67262192369e-24; // cgs
static constexpr double h = 6.62607015e-27; // cgs
static constexpr double hbar = 1.05457182e-27; // cgs
static constexpr double N_A = 6.02214076e+23; // 1 / mol
static constexpr double c_cgs = 2.99792458e+10; // cgs
static constexpr double c = 1.0; // natural
static constexpr double amu_to_g = 1.66053906660e-24;
static constexpr double seconds_to_days = 86400.0;
static constexpr double ev_to_erg = 1.602e-12;

static const double k_saha = std::pow(2. * PI * m_e * k_B / (h * h), 1.5);
} // namespace athelas::constants
