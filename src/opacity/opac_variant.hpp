#pragma once

#include <variant>

#include "io/tables.hpp"
#include "opacity/opac.hpp"
#include "opacity/opac_floor.hpp"
#include "pgen/problem_in.hpp"

namespace athelas {

/**
 * @brief Wrapper class for opacity models that provides member method interface
 */
class Opacity {
 public:
  // Default constructor
  Opacity() = default;

  // Constructors from concrete types
  explicit Opacity(const TabularOpacity &opac) : variant_(opac) {}
  explicit Opacity(TabularOpacity &&opac) : variant_(std::move(opac)) {}
  explicit Opacity(const Constant &opac) : variant_(opac) {}
  explicit Opacity(Constant &&opac) : variant_(std::move(opac)) {}
  explicit Opacity(const Powerlaw &opac) : variant_(opac) {}
  explicit Opacity(Powerlaw &&opac) : variant_(std::move(opac)) {}

  // Assignment operators
  auto operator=(const TabularOpacity &opac) -> Opacity & {
    variant_ = opac;
    return *this;
  }
  auto operator=(TabularOpacity &&opac) -> Opacity & {
    variant_ = std::move(opac);
    return *this;
  }
  auto operator=(const Constant &opac) -> Opacity & {
    variant_ = opac;
    return *this;
  }
  auto operator=(Constant &&opac) -> Opacity & {
    variant_ = std::move(opac);
    return *this;
  }
  auto operator=(const Powerlaw &opac) -> Opacity & {
    variant_ = opac;
    return *this;
  }
  auto operator=(Powerlaw &&opac) -> Opacity & {
    variant_ = std::move(opac);
    return *this;
  }

  // Member methods for opacity calculations
  KOKKOS_INLINE_FUNCTION auto planck_mean(const double rho, const double T,
                                          const double X,
                                          const double Z, double *lambda) const
      -> double {
    return std::visit(
        [&rho, &T, &X, &Z, &lambda](const auto &opac) {
          return opac.planck_mean(rho, T, X, Z, lambda);
        },
        variant_);
  }

  KOKKOS_INLINE_FUNCTION auto rosseland_mean(const double rho, const double T,
                                             const double X,
                                             const double Z, double *lambda) const
      -> double {
    return std::visit(
        [&rho, &T, &X, &Z, &lambda](const auto &opac) {
          return opac.rosseland_mean(rho, T, X, Z, lambda);
        },
        variant_);
  }

 private:
  using OpacVariant = std::variant<TabularOpacity, Powerlaw, Constant>;
  OpacVariant variant_;
};

// put init function here..

inline auto initialize_opacity_floor(const ProblemIn *pin) -> OpacityFloor {
  const auto floor_type =
      pin->param()->get<std::string>("opacity.floors.type", "constant");
  
  if (floor_type == "core_envelope") {
    return OpacityFloor(CoreEnvelopeFloor(
        pin->param()->get<double>("opacity.floors.core_rosseland"),
        pin->param()->get<double>("opacity.floors.core_planck"),
        pin->param()->get<double>("opacity.floors.env_rosseland"),
        pin->param()->get<double>("opacity.floors.env_planck")));
  } 
    // constant.
    const double kP_floor =
        pin->param()->get<double>("opacity.floors.planck", 0.0);
    const double kR_floor =
        pin->param()->get<double>("opacity.floors.rosseland", 0.0);
    return OpacityFloor(ConstantFloor(kR_floor, kP_floor));
}

inline auto initialize_opacity(const ProblemIn *pin) -> Opacity {
  const auto type = pin->param()->get<std::string>("opacity.type", "constant");
  const auto floor = initialize_opacity_floor(pin);
  
  if (type == "tabular") {
    auto fn = pin->param()->get<std::string>("opacity.filename");
    return Opacity(TabularOpacity(fn, std::move(floor)));
  } else if (type == "constant") {
    return Opacity(Constant(pin->param()->get<double>("opacity.kP", 1.0),
                            pin->param()->get<double>("opacity.kR", 1.0)));
  } else { // powerlaw rho
    return Opacity(Powerlaw(
        pin->param()->get<double>("opacity.kP", 1.0),
        pin->param()->get<double>("opacity.kR", 1.0),
        pin->param()->get<double>("opacity.rho_exp", 1.0),
        pin->param()->get<double>("opacity.t_exp", 1.0),
        std::move(floor),
        pin->param()->get<double>("opacity.kP_offset", 0.0),
        pin->param()->get<double>("opacity.kR_offset", 0.0)));
  }
}

} // namespace athelas
