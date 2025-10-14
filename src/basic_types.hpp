#pragma once

#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <Kokkos_Core.hpp>

namespace athelas {

namespace vars {

namespace cons {
constexpr int SpecificVolume = 0;
constexpr int Velocity = 1;
constexpr int Energy = 2;
constexpr int RadEnergy = 3;
constexpr int RadFlux = 4;
} // namespace cons

namespace prim {
constexpr int Rho = 0;
constexpr int Momentum = 1;
constexpr int Sie = 2;
} // namespace prim

namespace aux {
constexpr int Pressure = 0;
constexpr int Tgas = 1;
constexpr int Cs = 2;
} // namespace aux

namespace modes {
constexpr int CellAverage = 0;
constexpr int Slope = 1;
constexpr int Curvature = 2;
// Won't define more..
} // namespace modes

} // namespace vars

struct IndexRange {

  IndexRange() = default;
  explicit IndexRange(const int n) : e(n - 1) {}
  IndexRange(const int start, const int end) : s(start), e(end) {}
  explicit IndexRange(std::pair<int, int> domain)
      : s(domain.first), e(domain.second) {}

  int s = 0; /// Starting Index (inclusive)
  int e = 0; /// Ending Index (inclusive)
  [[nodiscard]] auto size() const -> int { return e - s + 1; }
  explicit operator std::pair<int, int>() const { return {s, e}; }
};

/**
 * @struct TimeStepInfo
 * @brief holds information related to a timestep
 */
struct TimeStepInfo {
  double t;
  double dt;
  double dt_a; // dt * tableau coefficient
  int stage;
};

enum class poly_basis { legendre, taylor };

enum class GravityModel { Constant, Spherical };

template <typename T>
using Dictionary = std::unordered_map<std::string, T>;

template <typename T>
using Triple_t = std::tuple<T, T, T>;
} // namespace athelas
