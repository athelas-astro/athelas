/**
 * @file tableau.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Class for holding implicit and explicit RK tableaus.
 *
 * @details TODO: describe tableaus.
 * TODO(astrobarker): add order, effective cfl to tableaus
 */

#include <unordered_map>

#include "timestepper/tableau.hpp"
#include "utils/error.hpp"

namespace athelas {

// converts an input string to its associated MethodID
auto string_to_id(const std::string &method_name) -> MethodID {
  static const std::unordered_map<std::string, MethodID> method_map = {
      {"ex_ssprk11", MethodID::EX_SSPRK11},
      {"ex_ssprk22", MethodID::EX_SSPRK22},
      {"ex_ssprk33", MethodID::EX_SSPRK33},
      {"ex_ssprk54", MethodID::EX_SSPRK54},
      {"ex_ssprk52", MethodID::EX_SSPRK52},
      {"ex_ssprk53", MethodID::EX_SSPRK53},
      {"imex_ssprk11", MethodID::IMEX_SSPRK11},
      {"imex_ssprk22_dirk", MethodID::IMEX_SSPRK22_DIRK},
      {"imex_ark32_esdirk", MethodID::IMEX_ARK32_ESDIRK},
      {"imex_ssprk33_dirk", MethodID::IMEX_SSPRK33_DIRK},
      {"imex_pdars_esdirk", MethodID::IMEX_PDARS_ESDIRK},
  };

  auto it = method_map.find(method_name);
  if (it != method_map.end()) {
    return it->second;
  }
  throw_athelas_error((std::string("Unknown method: ") + method_name));
}

auto create_tableau(MethodID method_id) -> RKIntegrator {
  switch (method_id) {
  case MethodID::EX_SSPRK11: {
    // --- Forward Euler --- //
    constexpr static MethodType type = MethodType::EX;
    constexpr static int stages = 1;
    constexpr static int explicit_order = 1;
    constexpr static int implicit_order = 0; // dummy
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(0, 0) = 0.0;
    b_ex(0) = 1.0;
    c_ex(0) = 1.0;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", 1, 1);
    HostArray1D<double> b_im("b_i_im", 1);
    HostArray1D<double> c_im("c_i_im", 1);

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end EX_SSPRK11
  case MethodID::EX_SSPRK22: {
    // --- Heun's method SSPRK(2,2) --- //
    constexpr static MethodType type = MethodType::EX;
    constexpr static int stages = 2;
    constexpr static int explicit_order = 2;
    constexpr static int implicit_order = 0; // dummy
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(1, 0) = 1.0;
    b_ex(0) = 0.5;
    b_ex(1) = 0.5;
    c_ex(0) = 0.0;
    c_ex(1) = 1.0;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", 1, 1);
    HostArray1D<double> b_im("b_i_im", 1);
    HostArray1D<double> c_im("c_i_im", 1);

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end EX_SSPRK22
  case MethodID::EX_SSPRK33: {
    // --- classic SSPRK(3,3) --- //
    constexpr static MethodType type = MethodType::EX;
    constexpr static int stages = 3;
    constexpr static int explicit_order = 3;
    constexpr static int implicit_order = 0; // dummy
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(1, 0) = 1.0;
    a_ex(2, 0) = 0.25;
    a_ex(2, 1) = 0.25;
    b_ex(0) = 1.0 / 6.0;
    b_ex(1) = 1.0 / 6.0;
    b_ex(2) = 2.0 / 3.0;
    c_ex(0) = 0.0;
    c_ex(1) = 1.0;
    c_ex(2) = 0.5;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", 1, 1);
    HostArray1D<double> b_im("b_i_im", 1);
    HostArray1D<double> c_im("c_i_im", 1);

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end EX_SSPRK33
  case MethodID::EX_SSPRK54: {
    // --- SSPRK(5,4) of Spiteri and Ruuth 2002 --- //
    // TODO(astrobarker) if I want to test other SSPRK(5,4), use _EXT in name
    constexpr static MethodType type = MethodType::EX;
    constexpr static int stages = 5;
    constexpr static int explicit_order = 4;
    constexpr static int implicit_order = 0; // dummy
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(1, 0) = 0.39175222700392;
    a_ex(2, 0) = 0.21766909633821;
    a_ex(2, 1) = 0.36841059262959;
    a_ex(3, 0) = 0.08269208670950;
    a_ex(3, 1) = 0.13995850206999;
    a_ex(3, 2) = 0.25189177424738;
    a_ex(4, 0) = 0.06796628370320;
    a_ex(4, 1) = 0.11503469844438;
    a_ex(4, 2) = 0.20703489864929;
    a_ex(4, 3) = 0.54497475021237;

    b_ex(0) = 0.14681187618661;
    b_ex(1) = 0.24848290924556;
    b_ex(2) = 0.10425883036650;
    b_ex(3) = 0.27443890091960;
    b_ex(4) = 0.22600748319395;

    c_ex(0) = 0.0;
    c_ex(1) = 0.39175222700392;
    c_ex(2) = 0.58607968896779;
    c_ex(3) = 0.47454236302687;
    c_ex(4) = 0.93501063100924;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", 1, 1);
    HostArray1D<double> b_im("b_i_im", 1);
    HostArray1D<double> c_im("c_i_im", 1);

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end EX_SSPRK54
  case MethodID::EX_SSPRK52: {
    // --- SSPRK(5,2) --- //
    // radius of absolute monotonicity 4.0
    constexpr static MethodType type = MethodType::EX;
    constexpr static int stages = 5;
    constexpr static int explicit_order = 2;
    constexpr static int implicit_order = 0; // dummy
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(1, 0) = 0.25;
    a_ex(2, 0) = 0.25;
    a_ex(3, 0) = 0.25;
    a_ex(4, 0) = 0.25;
    a_ex(2, 1) = 0.25;
    a_ex(3, 1) = 0.25;
    a_ex(4, 1) = 0.25;
    a_ex(3, 2) = 0.25;
    a_ex(4, 2) = 0.25;
    a_ex(4, 3) = 0.25;

    b_ex(0) = 1.0 / 5.0;
    b_ex(1) = 1.0 / 5.0;
    b_ex(2) = 1.0 / 5.0;
    b_ex(3) = 1.0 / 5.0;
    b_ex(4) = 1.0 / 5.0;

    c_ex(0) = 0.0;
    c_ex(1) = 0.25;
    c_ex(2) = 0.5;
    c_ex(3) = 0.75;
    c_ex(4) = 1.0;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", 1, 1);
    HostArray1D<double> b_im("b_i_im", 1);
    HostArray1D<double> c_im("c_i_im", 1);

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end EX_SSPRK52
  case MethodID::EX_SSPRK53: {
    // --- SSPRK(5,3) --- //
    // radius of absolute monotonicity 2.65
    constexpr static MethodType type = MethodType::EX;
    constexpr static int stages = 5;
    constexpr static int explicit_order = 3;
    constexpr static int implicit_order = 0; // dummy
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(1, 0) = 0.377;
    a_ex(2, 0) = 0.377;
    a_ex(3, 0) = 0.164;
    a_ex(4, 0) = 0.149;
    a_ex(2, 1) = 0.377;
    a_ex(3, 1) = 0.164;
    a_ex(4, 1) = 0.148;
    a_ex(3, 2) = 0.164;
    a_ex(4, 2) = 0.148;
    a_ex(4, 3) = 0.342;

    b_ex(0) = 0.197;
    b_ex(1) = 0.118;
    b_ex(2) = 0.117;
    b_ex(3) = 0.270;
    b_ex(4) = 0.298;

    c_ex(0) = 0.0;
    c_ex(1) = 0.377;
    c_ex(2) = 0.755;
    c_ex(3) = 0.491;
    c_ex(4) = 0.788;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", 1, 1);
    HostArray1D<double> b_im("b_i_im", 1);
    HostArray1D<double> c_im("c_i_im", 1);

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end EX_SSPRK52
  case MethodID::IMEX_SSPRK11: {
    // --- Forward Euler + Backward Euler --- //
    constexpr static MethodType type = MethodType::IMEX;
    constexpr static int stages = 1;
    constexpr static int explicit_order = 1;
    constexpr static int implicit_order = 1;
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(0, 0) = 0.0;
    b_ex(0) = 1.0;
    c_ex(0) = 1.0;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // implicit tableau
    HostArray2D<double> a_im("a_ij_im", stages, stages);
    HostArray1D<double> b_im("b_i_im", stages);
    HostArray1D<double> c_im("c_i_im", stages);

    a_im(0, 0) = 1.0;
    b_im(0) = 1.0;
    c_im(0) = 1.0;

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end IMEX_SSPRK11
  case MethodID::IMEX_SSPRK22_DIRK: {
    // --- Heun's method SSPRK(2,2) with Pareschi & Russo DIRK(2,2) --- //
    constexpr static MethodType type = MethodType::IMEX;
    constexpr static int stages = 2;
    constexpr static int explicit_order = 2;
    constexpr static int implicit_order = 2;
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(1, 0) = 1.0;
    b_ex(0) = 0.5;
    b_ex(1) = 0.5;
    c_ex(0) = 0.0;
    c_ex(1) = 1.0;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", stages, stages);
    HostArray1D<double> b_im("b_i_im", stages);
    HostArray1D<double> c_im("c_i_im", stages);

    constexpr static double gam = 1.0 - (1.0 / std::numbers::sqrt2);
    a_im(0, 0) = gam;
    a_im(1, 0) = 1.0 - 2.0 * gam;
    a_im(1, 1) = gam;
    b_im(0) = 0.5;
    b_im(1) = 0.5;
    c_im(0) = gam;
    c_im(1) = 1.0 - gam;

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end IMEX_SSPRK22_DIRK
  case MethodID::IMEX_ARK32_ESDIRK: {
    // --- Giraldo et al 2013 ARK32 ESDIRK --- //
    constexpr static MethodType type = MethodType::IMEX;
    constexpr static int stages = 3;
    constexpr static int explicit_order = 2;
    constexpr static int implicit_order = 2;
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    constexpr static double a32 = (3.0 + 2.0 * std::numbers::sqrt2) / 6.0;
    a_ex(1, 0) = 2.0 - std::numbers::sqrt2;
    a_ex(2, 0) = 1.0 - a32;
    a_ex(2, 1) = a32;
    b_ex(0) = +1.0 / (2.0 * std::numbers::sqrt2);
    b_ex(1) = +1.0 / (2.0 * std::numbers::sqrt2);
    b_ex(2) = 1.0 - 1.0 / std::numbers::sqrt2;
    c_ex(0) = 0.0;
    c_ex(1) = 2.0 - std::numbers::sqrt2;
    c_ex(2) = 1.0;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", stages, stages);
    HostArray1D<double> b_im("b_i_im", stages);
    HostArray1D<double> c_im("c_i_im", stages);

    a_im(1, 0) = 1.0 - 1.0 / std::numbers::sqrt2;
    a_im(2, 0) = +1.0 / (2.0 * std::numbers::sqrt2);
    a_im(1, 1) = 1.0 - 1.0 / std::numbers::sqrt2;
    a_im(2, 1) = +1.0 / (2.0 * std::numbers::sqrt2);
    a_im(2, 2) = 1.0 - 1.0 / std::numbers::sqrt2;
    b_im(0) = +1.0 / (2.0 * std::numbers::sqrt2);
    b_im(1) = +1.0 / (2.0 * std::numbers::sqrt2);
    b_im(2) = 1.0 - 1.0 / std::numbers::sqrt2;
    c_im(0) = 0.0;
    c_im(1) = 2.0 - std::numbers::sqrt2;
    c_im(2) = 1.0;

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end IMEX_ARK32_ESDIRK
  case MethodID::IMEX_PDARS_ESDIRK: {
    // --- Ran Chu et al 2019 PD-ARS ESDIRK --- //
    constexpr static MethodType type = MethodType::IMEX;
    constexpr static int stages = 3;
    constexpr static int explicit_order = 2;
    constexpr static int implicit_order = 2;
    HostArray2D<double> a_ex("a_ij_ex", stages, stages);
    HostArray1D<double> b_ex("b_i_ex", stages);
    HostArray1D<double> c_ex("c_i_ex", stages);

    a_ex(1, 0) = 1.0;
    a_ex(2, 0) = 0.5;
    a_ex(2, 1) = 0.5;
    b_ex(0) = 0.5;
    b_ex(1) = 0.5;
    b_ex(2) = 0.0;
    c_ex(0) = 0.0;
    c_ex(1) = 1.0;
    c_ex(2) = 1.0;

    auto explicit_tableau = RKTableau(TableauType::Explicit, explicit_order,
                                      stages, a_ex, b_ex, c_ex);

    // dummy implicit tableau
    HostArray2D<double> a_im("a_ij_im", stages, stages);
    HostArray1D<double> b_im("b_i_im", stages);
    HostArray1D<double> c_im("c_i_im", stages);

    constexpr static double eps = 0.0;
    a_im(1, 1) = 1.0;
    a_im(2, 1) = 0.5 - eps;
    a_im(2, 2) = 0.5 + eps;
    b_im(0) = 0.0;
    b_im(1) = 0.5 - eps;
    b_im(2) = 0.5 + eps;
    c_im(0) = 0.0;
    c_im(1) = 1.0;
    c_im(2) = 1.0;

    auto implicit_tableau = RKTableau(TableauType::Implicit, implicit_order,
                                      stages, a_im, b_im, c_im);

    return RKIntegrator(method_id, type, explicit_order, implicit_order, stages,
                        explicit_tableau, implicit_tableau);
  } // end IMEX_PDARS_ESDIRK
  // TODO(astrobarker) fill out implicit tableaus
  default:
    throw_athelas_error("Unknown tableau!"); // Shouldn't reach
  }
}

ButcherTableau::ButcherTableau(const int nStages_, const int tOrder_,
                               const TableauType type)
    : nStages(nStages_), tOrder(tOrder_),
      a_ij("butcher a_ij", nStages_, nStages_), b_i("butcher b_i", nStages_),
      type_(type) {
  initialize_tableau();
}
// Initialize arrays for timestepper
// TODO(astrobarker): Separate nStages from a tOrder
void ButcherTableau::initialize_tableau() {

  if (tOrder == 1 and nStages > 1) {
    throw_athelas_error("\n \
      ! ButcherTableau :: Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. \n");
  }
  if ((nStages != tOrder && nStages != 5)) {
    throw_athelas_error("\n \
      ! ButcherTableau :: Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. \n");
  }
  if ((tOrder == 4 && nStages != 5)) {
    throw_athelas_error("\n \
      ! ButcherTableau :: Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. \n");
  }
  if (tOrder > 4) {
    throw_athelas_error(
        "\n ! ButcherTableau :: Temporal torder > 4 not supported! \n");
  }

  // Init to zero
  for (int i = 0; i < nStages; i++) {
    for (int j = 0; j < nStages; j++) {
      a_ij(i, j) = 0.0;
      b_i(i) = 0.0;
    }
  }

  if (type_ == TableauType::Explicit) {
    // Forward Euler //
    if (nStages == 1 and tOrder == 1) {
      a_ij(0, 0) = 0.0;
      b_i(0) = 1.0;
    } else if (nStages == 2 && tOrder == 2) {
      a_ij(1, 0) = 1.0;
      b_i(0) = 0.5;
      b_i(1) = 0.5;
    } else if (nStages == 3 && tOrder == 3) {
      a_ij(1, 0) = 1.0;
      a_ij(2, 0) = 0.25;
      a_ij(2, 1) = 0.25;
      b_i(0) = 1.0 / 6.0;
      b_i(1) = 1.0 / 6.0;
      b_i(2) = 2.0 / 3.0;
      // (pex, pim, plin) = (2,2,5)
    } else if (nStages == 5 && tOrder == 2) {
      a_ij(1, 0) = 1.0;
      a_ij(2, 0) = 0.99132899;
      a_ij(3, 0) = 0.99130196;
      a_ij(4, 0) = 0.99191257;
      a_ij(2, 1) = 0.99132899;
      a_ij(3, 1) = 0.96542648;
      a_ij(4, 1) = 0.94453690;
      a_ij(3, 2) = 0.97387092;
      a_ij(4, 2) = 0.90549663;
      a_ij(4, 3) = 0.92979121;
      b_i(0) = 0.63253575;
      b_i(1) = 0.25781844;
      b_i(2) = 0.09173050;
      b_i(3) = 0.00863176;
      b_i(4) = 0.00928355;
    } else if (nStages == 5 && tOrder == 3) {
      a_ij(1, 0) = 1.0;
      a_ij(2, 0) = 0.19736166;
      a_ij(3, 0) = 0.06602780;
      a_ij(4, 0) = 0.04161484;
      a_ij(2, 1) = 0.19736166;
      a_ij(3, 1) = 0.06602780;
      a_ij(4, 1) = 0.02887068;
      a_ij(3, 2) = 0.33455230;
      a_ij(4, 2) = 0.14628314;
      a_ij(4, 3) = 0.43725043;
      b_i(0) = 0.15562497;
      b_i(1) = 0.13677868;
      b_i(2) = 0.29274344;
      b_i(3) = 0.12620947;
      b_i(4) = 0.28864344;
    } else if (nStages == 5 && tOrder == 4) {
      a_ij(1, 0) = 0.51047914;
      a_ij(2, 0) = 0.08515080;
      a_ij(3, 0) = 0.29902100;
      a_ij(4, 0) = 0.01438455;
      a_ij(2, 1) = 0.21940489;
      a_ij(3, 1) = 0.07704762;
      a_ij(4, 1) = 0.03706414;
      a_ij(3, 2) = 0.46190055;
      a_ij(4, 2) = 0.22219957;
      a_ij(4, 3) = 0.63274729;
      b_i(0) = 0.12051432;
      b_i(1) = 0.22614012;
      b_i(2) = 0.27630606;
      b_i(3) = 0.12246455;
      b_i(4) = 0.25457495;
    } else {
      throw_athelas_error(
          " ! ButcherTableau :: Explicit :: Please choose a valid "
          "tableau! \n");
    }
  }

  if (type_ == TableauType::Implicit) {

    // Backwards Euler //
    if (nStages == 1 && tOrder == 1) {
      a_ij(0, 0) = 1.0;
      b_i(0) = 1.0;
      /*
    } else if ( nStages == 2 && tOrder == 2 ) {
      //const static double gam = 1.0 - ( 1.0 / std::sqrt( 2 ) );
      const static double gam = 1.0 + ( std::sqrt( 2 ) / 2.0 );
      a_ij( 0, 0 )          = gam; // 0.71921758;
      a_ij( 1, 0 )          = 1.0 - 2.0 * gam; // 0.11776435;
      a_ij( 1, 1 )          = gam; // 0.16301806;
      b_i( 0 )              = 0.5;
      b_i( 1 )              = 0.5;
      // L-stable
      */
    } else if (nStages == 2 && tOrder == 2) {
      // const static double gam = 1.0 - ( 1.0 / std::sqrt( 2 ) );
      a_ij(0, 0) = 0.25;
      a_ij(1, 0) = 0.5;
      a_ij(1, 1) = 0.25;
      b_i(0) = 0.5;
      b_i(1) = 0.5;
    } else if (nStages == 3 && tOrder == 3) {
      a_ij(2, 0) = 1.0 / 6.0;
      a_ij(1, 1) = 1.0;
      a_ij(2, 1) = -1.0 / 3.0;
      a_ij(2, 2) = 2.0 / 3.0;
      b_i(0) = 1.0 / 6.0;
      b_i(1) = 1.0 / 6.0;
      b_i(2) = 2.0 / 3.0;
    } else if (nStages == 5 && tOrder == 4) {
      a_ij(0, 0) = 1.03217796e-16; // just 0?
      a_ij(1, 0) = 0.510479144;
      a_ij(2, 0) = 5.06048136e-3;
      a_ij(3, 0) = 8.321807e-2;
      a_ij(4, 0) = 7.56636565e-2;
      a_ij(1, 1) = 1.00124199e-14;
      a_ij(2, 1) = 1.00953283e-1;
      a_ij(3, 1) = 1.60838280e-1;
      a_ij(4, 1) = 1.25319139e-1;
      a_ij(2, 2) = 1.98541931e-1;
      a_ij(3, 2) = 3.28641063e-1;
      a_ij(4, 2) = 7.08147871e-2;
      a_ij(3, 3) = -3.84714236e-3;
      a_ij(4, 3) = 6.34597980e-1;
      a_ij(4, 4) = -7.22101223e-17;
      b_i(0) = 0.12051432;
      b_i(1) = 0.22614012;
      b_i(2) = 0.27630606;
      b_i(3) = 0.12246455;
      b_i(4) = 0.25457495;
    } else {
      throw_athelas_error(" ! ButcherTableau :: Implicit :: Please choose a "
                          "valid tableau! \n");
    }

    // TODO(astrobarker): more tableaus
  }
}

ShuOsherTableau::ShuOsherTableau(const int nStages_, const int tOrder_,
                                 const TableauType /*type*/)
    : nStages(nStages_), tOrder(tOrder_),
      a_ij("butcher a_ij", nStages_, nStages_),
      b_ij("butcher b_i", nStages_, nStages_) {
  initialize_tableau();
}

// Initialize arrays for timestepper
// TODO(astrobarker): Separate nStages from a tOrder
void ShuOsherTableau::initialize_tableau() {

  if (tOrder == 1 and nStages > 1) {
    throw_athelas_error("\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n");
  }
  if ((nStages != tOrder && nStages != 5)) {
    throw_athelas_error("\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n");
  }
  if ((tOrder == 4 && nStages != 5)) {
    throw_athelas_error("\n \
      ! Issue in setting SSPRK coefficients.\n \
      Please enter an appropriate SSPRK temporal order and nStages\n \
      combination. We support first through fourth order timesteppers\n \
      using 1-3 stages for first-thrid order and 5 stages for second\n \
      through fourth order.\n === \n");
  }
  if (tOrder > 4) {
    throw_athelas_error("\n ! Temporal torder > 4 not supported! \n");
  }

  // Init to zero
  for (int i = 0; i < nStages; i++) {
    for (int j = 0; j < nStages; j++) {
      a_ij(i, j) = 0.0;
      b_ij(i, j) = 0.0;
    }
  }

  if (type_ == TableauType::Implicit) {
    throw_athelas_error(
        " ! ShuOsherTableau :: No implicit ShuOsher form tableaus "
        "implemented.");
  }

  if (nStages < 5) {

    if (tOrder == 1) {
      a_ij(0, 0) = 1.0;
      b_ij(0, 0) = 1.0;
    } else if (tOrder == 2) {
      a_ij(0, 0) = 1.0;
      a_ij(1, 0) = 0.5;
      a_ij(1, 1) = 0.5;

      b_ij(0, 0) = 1.0;
      b_ij(1, 0) = 0.0;
      b_ij(1, 1) = 0.5;
    } else if (tOrder == 3) {
      a_ij(0, 0) = 1.0;
      a_ij(1, 0) = 0.75;
      a_ij(1, 1) = 0.25;
      a_ij(2, 0) = 1.0 / 3.0;
      a_ij(2, 1) = 0.0;
      a_ij(2, 2) = 2.0 / 3.0;

      b_ij(0, 0) = 1.0;
      b_ij(1, 0) = 0.0;
      b_ij(1, 1) = 0.25;
      b_ij(2, 0) = 0.0;
      b_ij(2, 1) = 0.0;
      b_ij(2, 2) = 2.0 / 3.0;
    }
  } else if (nStages == 5) {
    if (tOrder == 1) {
      throw_athelas_error("\n ! We do support a 1st order, 5 stage SSPRK "
                          "integrator. \n");
    } else if (tOrder == 2) {
      a_ij(0, 0) = 1.0;
      a_ij(4, 0) = 0.2;
      a_ij(1, 1) = 1.0;
      a_ij(2, 2) = 1.0;
      a_ij(3, 3) = 1.0;
      a_ij(4, 4) = 0.8;

      b_ij(0, 0) = 0.25;
      b_ij(1, 1) = 0.25;
      b_ij(2, 2) = 0.25;
      b_ij(3, 3) = 0.25;
      b_ij(4, 4) = 0.20;
    } else if (tOrder == 3) {
      a_ij(0, 0) = 1.0;
      a_ij(1, 0) = 0.0;
      a_ij(2, 0) = 0.56656131914033;
      a_ij(3, 0) = 0.09299483444413;
      a_ij(4, 0) = 0.00736132260920;
      a_ij(1, 1) = 1.0;
      a_ij(3, 1) = 0.00002090369620;
      a_ij(4, 1) = 0.20127980325145;
      a_ij(2, 2) = 0.43343868085967;
      a_ij(4, 2) = 0.00182955389682;
      a_ij(3, 3) = 0.90698426185967;
      a_ij(4, 4) = 0.78952932024253;

      b_ij(0, 0) = 0.37726891511710;
      b_ij(3, 0) = 0.00071997378654;
      b_ij(4, 0) = 0.00277719819460;
      b_ij(1, 1) = 0.37726891511710;
      b_ij(4, 1) = 0.00001567934613;
      b_ij(2, 2) = 0.16352294089771;
      b_ij(3, 3) = 0.34217696850008;
      b_ij(4, 4) = 0.29786487010104;
    } else if (tOrder == 4) {
      // a_ij( 0, 0 ) = 1.0;
      // a_ij( 1, 0 ) = 0.44437049406734;
      // a_ij( 2, 0 ) = 0.62010185138540;
      // a_ij( 3, 0 ) = 0.17807995410773;
      // a_ij( 4, 0 ) = 0.00683325884039;
      // a_ij( 1, 1 ) = 0.55562950593266;
      // a_ij( 2, 2 ) = 0.37989814861460;
      // a_ij( 4, 2 ) = 0.51723167208978;
      // a_ij( 3, 3 ) = 0.82192004589227;
      // a_ij( 4, 3 ) = 0.12759831133288;
      // a_ij( 4, 4 ) = 0.34833675773694;

      // b_ij( 0, 0 ) = 0.39175222700392;
      // b_ij( 1, 1 ) = 0.36841059262959;
      // b_ij( 2, 2 ) = 0.25189177424738;
      // b_ij( 3, 3 ) = 0.54497475021237;
      // b_ij( 4, 3 ) = 0.08460416338212;
      // b_ij( 4, 4 ) = 0.22600748319395;
      a_ij(0, 0) = 1.0;
      a_ij(1, 0) = 0.444370493651235;
      a_ij(2, 0) = 0.620101851488403;
      a_ij(3, 0) = 0.178079954393132;
      a_ij(4, 0) = 0.000000000000000;
      a_ij(1, 1) = 0.555629506348765;
      a_ij(2, 2) = 0.379898148511597;
      a_ij(4, 2) = 0.517231671970585;
      a_ij(3, 3) = 0.821920045606868;
      a_ij(4, 3) = 0.096059710526147;
      a_ij(4, 4) = 0.386708617503269;

      b_ij(0, 0) = 0.391752226571890;
      b_ij(1, 1) = 0.368410593050371;
      b_ij(2, 2) = 0.251891774271694;
      b_ij(3, 3) = 0.544974750228521;
      b_ij(4, 3) = 0.063692468666290;
      b_ij(4, 4) = 0.226007483236906;
    }
  }
}

} // namespace athelas
