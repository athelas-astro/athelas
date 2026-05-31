#pragma once

#include <concepts>

#include "basic_types.hpp"
#include "interface/state.hpp"

namespace athelas {

// Concepts for package validation
template <typename T>
concept ExplicitPackage =
    requires(T &pkg, const StageData &stage_data, StageData &stage_data_derived,
             AthelasArray3D<double> lhs, AthelasArray3D<double> dU,
             const TimeStepInfo &dt_info) {
      { pkg.update_explicit(stage_data, dt_info) } -> std::same_as<void>;
      { pkg.apply_delta(lhs, dt_info) } -> std::same_as<void>;
      { pkg.zero_delta() } -> std::same_as<void>;
      { pkg.min_timestep(stage_data, dt_info) } -> std::convertible_to<double>;
      { pkg.name() } -> std::convertible_to<std::string_view>;
      { pkg.is_active() } -> std::convertible_to<bool>;
      { pkg.fill_derived(stage_data_derived, dt_info) } -> std::same_as<void>;
    };

template <typename T>
concept ImplicitPackage =
    requires(T &pkg, const StageData &stage_data, StageData &stage_data_derived,
             AthelasArray3D<double> lhs, AthelasArray3D<double> dU,
             const TimeStepInfo &dt_info) {
      { pkg.update_implicit(stage_data, dU, dt_info) } -> std::same_as<void>;
      { pkg.apply_delta(lhs, dt_info) } -> std::same_as<void>;
      { pkg.zero_delta() } -> std::same_as<void>;
      { pkg.min_timestep(stage_data, dt_info) } -> std::convertible_to<double>;
      { pkg.name() } -> std::convertible_to<std::string_view>;
      { pkg.is_active() } -> std::convertible_to<bool>;
      { pkg.fill_derived(stage_data_derived, dt_info) } -> std::same_as<void>;
    };

template <typename T>
concept IMEXPackage =
    requires(T &pkg, const StageData &stage_data, StageData &stage_data_derived,
             AthelasArray3D<double> lhs, AthelasArray3D<double> dU,
             const TimeStepInfo &dt_info) {
      { pkg.update_explicit(stage_data, dt_info) } -> std::same_as<void>;
      { pkg.update_implicit(stage_data, dU, dt_info) } -> std::same_as<void>;
      { pkg.apply_delta(lhs, dt_info) } -> std::same_as<void>;
      { pkg.zero_delta() } -> std::same_as<void>;
      { pkg.min_timestep(stage_data, dt_info) } -> std::convertible_to<double>;
      { pkg.name() } -> std::convertible_to<std::string_view>;
      { pkg.is_active() } -> std::convertible_to<bool>;
      { pkg.fill_derived(stage_data_derived, dt_info) } -> std::same_as<void>;
    };

template <typename T>
concept PhysicsPackage =
    ExplicitPackage<T> || ImplicitPackage<T> || IMEXPackage<T>;

// Type traits to detect package capabilities
template <typename T>
constexpr bool has_explicit_update_v =
    requires(T &pkg, const StageData &stage_data, const TimeStepInfo &dt_info) {
      pkg.update_explicit(stage_data, dt_info);
    };

template <typename T>
constexpr bool has_implicit_update_v =
    requires(T &pkg, const StageData &stage_data, AthelasArray3D<double> lhs,
             const TimeStepInfo &dt_info) {
      pkg.update_implicit(stage_data, lhs, dt_info);
    };

} // namespace athelas
