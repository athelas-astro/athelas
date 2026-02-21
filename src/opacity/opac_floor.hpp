/**
 * @file opac_floor.hpp
 * --------------
 *
 * @brief Opacity floor model implementations using CRTP
 *
 * @details Provides a CRTP base class for opacity floor models and concrete
 *          implementations. The OpacityFloor wrapper class provides type-erased
 *          access to floor models via std::variant.
 */

#pragma once

#include <variant>

#include "Kokkos_Macros.hpp"

namespace athelas {

/**
 * @brief CRTP base class for opacity floor models
 *
 * @tparam FLOOR The concrete floor model type
 */
template <class FLOOR>
class OpacityFloorModel {
 public:
  KOKKOS_INLINE_FUNCTION auto rosseland(double z) const -> double {
    return static_cast<const FLOOR *>(this)->rosseland_impl(z);
  }

  KOKKOS_INLINE_FUNCTION auto planck(double z) const -> double {
    return static_cast<const FLOOR *>(this)->planck_impl(z);
  }
};

/**
 * @brief Constant opacity floor model
 */
class ConstantFloor : public OpacityFloorModel<ConstantFloor> {
 public:
  KOKKOS_INLINE_FUNCTION explicit ConstantFloor(double rosseland = 0.0,
                                                double planck = 0.0)
      : rosseland_(rosseland), planck_(planck) {}

  KOKKOS_INLINE_FUNCTION auto rosseland_impl(double /* metallicity */) const
      -> double {
    return rosseland_;
  }

  KOKKOS_INLINE_FUNCTION auto planck_impl(double /* metallicity */) const
      -> double {
    return planck_;
  }

 private:
  double rosseland_;
  double planck_;
};

/**
 * @brief Core-Envelope opacity floor model
 */
class CoreEnvelopeFloor : public OpacityFloorModel<CoreEnvelopeFloor> {
 public:
  KOKKOS_INLINE_FUNCTION CoreEnvelopeFloor(double core_rosseland,
                                            double core_planck,
                                            double env_rosseland,
                                            double env_planck)
      : core_rosseland_(core_rosseland),
        core_planck_(core_planck),
        env_rosseland_(env_rosseland),
        env_planck_(env_planck) {}

  KOKKOS_INLINE_FUNCTION auto rosseland_impl(double metallicity) const
      -> double {
    return compute_floor(core_rosseland_, env_rosseland_, metallicity);
  }

  KOKKOS_INLINE_FUNCTION auto planck_impl(double metallicity) const -> double {
    return compute_floor(core_planck_, env_planck_, metallicity);
  }

 private:
  double core_rosseland_;
  double core_planck_;
  double env_rosseland_;
  double env_planck_;

  KOKKOS_INLINE_FUNCTION static auto compute_floor(double f_core, double f_env,
                                                     double z) -> double {
    constexpr double Z_sun = 0.02;
    return (f_core * Z_sun - f_env + z * (f_env - f_core)) / (Z_sun - 1.0);
  }
};

/**
 * @brief Type-erased wrapper for opacity floor models
 *
 * @details Provides a unified interface to different floor model types via
 *          std::variant. This allows opacity models to store a floor model
 *          without being templated on the floor type.
 */
class OpacityFloor {
 public:
  // Default constructor
  OpacityFloor() = default;

  // Constructors from concrete floor types
  OpacityFloor(const ConstantFloor &floor) : variant_(floor) {}
  OpacityFloor(ConstantFloor &&floor) : variant_(std::move(floor)) {}
  OpacityFloor(const CoreEnvelopeFloor &floor) : variant_(floor) {}
  OpacityFloor(CoreEnvelopeFloor &&floor) : variant_(std::move(floor)) {}

  // Assignment operators
  auto operator=(const ConstantFloor &floor) -> OpacityFloor & {
    variant_ = floor;
    return *this;
  }
  auto operator=(ConstantFloor &&floor) -> OpacityFloor & {
    variant_ = std::move(floor);
    return *this;
  }
  auto operator=(const CoreEnvelopeFloor &floor) -> OpacityFloor & {
    variant_ = floor;
    return *this;
  }
  auto operator=(CoreEnvelopeFloor &&floor) -> OpacityFloor & {
    variant_ = std::move(floor);
    return *this;
  }

  // Member methods that dispatch via std::visit
  KOKKOS_INLINE_FUNCTION auto rosseland(double z) const -> double {
    return std::visit(
        [z](const auto &floor) { return floor.rosseland(z); }, variant_);
  }

  KOKKOS_INLINE_FUNCTION auto planck(double z) const -> double {
    return std::visit(
        [z](const auto &floor) { return floor.planck(z); }, variant_);
  }

 private:
  std::variant<ConstantFloor, CoreEnvelopeFloor> variant_;
};

} // namespace athelas


