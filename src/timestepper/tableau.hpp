/**
 * @file tableau.hpp
 * --------------
 *
 * @brief Class for holding implicit and explicit RK tableaus.
 *
 * @details TODO: describe tableaus.
 */

#pragma once

#include "kokkos_types.hpp"

namespace athelas {

enum class TableauType { Implicit, Explicit };
enum class MethodType { EX, IM, IMEX };
enum class MethodData { DIRK, ESDIRK }; // Where to put this info?

/**
 * @enum MethodID
 * @brief Runge Kutta tableau identifiers
 *
 * Naming convention:
 * - [TYPE]_[METHODNAME]_[EXT]
 *
 * For fully explicit methods, [TYPE] is EX and [METHODNAME] is generally
 * SSPRK[stages][order], e.g., SSPRK(5,2) would be EX_SSPRK52.
 * For now, [EXT] is unused here.
 *
 * For IMEX methods, [TYPE] is IMEX and [METHODNAME] should also
 * include the implicit order and [EXT] notes the form of the implicit
 * tableau \in {DIRK, SDIRK, ESDIRK}, e.g., IMEX_SSPRK222_ESDIRK
 **/
enum class MethodID {
  EX_SSPRK11, // Forward Euler
  EX_SSPRK22, // 2 stage, 2nd order SSP
  EX_SSPRK33,
  EX_SSPRK54,
  EX_SSPRK52,
  EX_SSPRK53,
  IMEX_SSPRK11, // Forward Euler + Backward Euler
  IMEX_SSPRK22_DIRK,
  IMEX_SSPRK33_DIRK,
  IMEX_ARK32_ESDIRK,
  IMEX_PDARS_ESDIRK // Chu 2019 PD-ARS
};

auto string_to_id(const std::string &method_name) -> MethodID;

/**
 * @brief Butcher tableau class.
 * NOTE: The tableau coefficients are stored on host!
 **/
struct RKTableau {
  TableauType type; // explicit or implicit
  int order;
  int num_stages;
  HostArray2D<double> a_ij;
  HostArray1D<double> b_i;
  HostArray1D<double> c_i;

  // Constructor
  RKTableau(TableauType t, int num_stages_, int order_,
            HostArray2D<double> a_ij_, HostArray1D<double> b_i_,
            AthelasArray1D<double> c_i_)
      : type(t), order(order_), num_stages(num_stages_), a_ij(a_ij_), b_i(b_i_),
        c_i(c_i_) {}
};

/**
 * @brief IMEX double tableau class.
 * TODO(astrobarker): make implicit_tableau std::optional?
 **/
struct RKIntegrator {
  MethodID name;
  MethodType method; // EX, IM, IMEX
  int explicit_order;
  int implicit_order;
  int num_stages; // duplicate

  RKTableau explicit_tableau;
  RKTableau implicit_tableau;

  RKIntegrator(MethodID id, MethodType t, int ex_order, int im_order,
               int num_stages_, RKTableau ex, RKTableau im)
      : name(id), method(t), explicit_order(ex_order), implicit_order(im_order),
        num_stages(num_stages_), explicit_tableau(ex), implicit_tableau(im) {}
};

auto create_tableau(MethodID method_id) -> RKIntegrator;

/**
 * @brief Butcher tableau class.
 **/
class ButcherTableau {
 public:
  ButcherTableau(int nStages_, int tOrder_, TableauType type);

  int nStages;
  int tOrder;

  HostArray2D<double> a_ij;
  HostArray1D<double> b_i;

 private:
  TableauType type_;
  // int c_effective_{};

  void initialize_tableau();
};

/**
 * @brief Shu Osher tableau class.
 **/
class ShuOsherTableau {
 public:
  ShuOsherTableau(int nStages, int tOrder, TableauType type);

  int nStages;
  int tOrder;

  HostArray2D<double> a_ij;
  AthelasArray2D<double> b_ij;

 private:
  TableauType type_;
  // int c_effective_{};

  void initialize_tableau();
};

} // namespace athelas
