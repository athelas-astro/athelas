#pragma once

#include <algorithm>
#include <cmath>

#include "concepts/arithmetic.hpp"
#include "solvers/root_finder_opts.hpp"
#include "utils/utilities.hpp"

// TODO(astrobarker): make a solvers namespace? If it grows beyond rf, yes.
namespace athelas::root_finders {

template <typename T, typename F, typename... Args>
auto residual(F g, T x0, const int k, const int iC, Args... args) -> double {
  return g(x0, k, iC, args...) - x0(iC, k);
}

template <Subtractable T>
KOKKOS_INLINE_FUNCTION auto residual(const T f, const T x) -> T {
  return f - x;
}

KOKKOS_INLINE_FUNCTION
auto alpha_aa(const double r_n, const double r_nm1) -> double {
  return std::clamp(utilities::ratio(r_n, (r_n - r_nm1)), 0.0, 1.0);
}

// physical scales for normalization
// TODO(BLB): This can be simplified into an array or such.
struct PhysicalScales {
  double velocity_scale;
  double energy_scale;
  double rad_energy_scale;
  double rad_flux_scale;
};

// robust convergence metric for modal dg radiation hydrodynamics
template <typename t>
class RadHydroConvergence {
 private:
  PhysicalScales scales_;
  double abs_tol_;
  double rel_tol_;
  int num_modes_;

  std::vector<double> mode_weights_;

 public:
  explicit RadHydroConvergence(const PhysicalScales &scales,
                               double abs_tol = 1e-10, double rel_tol = 1e-8,
                               int num_modes = 1)
      : scales_(scales), abs_tol_(abs_tol), rel_tol_(rel_tol),
        num_modes_(num_modes) {

    mode_weights_.resize(num_modes);
    for (int k = 0; k < num_modes; ++k) {
      // exponential decay for higher modes
      mode_weights_[k] = std::exp(-0.5 * k);
    }
  }

  // separate error metrics for different variable types
  // TODO(astrobarker) combine the following
  auto fluid_velocity_error(const t u_n, const t u_nm1, const int q) -> double {
    double max_error = 0.0;
    for (int k = 0; k < num_modes_; ++k) {
      const double abs_err = std::abs(u_n(k, q) - u_nm1(k, q));
      const double scale = std::max(
          {scales_.velocity_scale, std::abs(u_n(k, q)), std::abs(u_nm1(k, q))});
      const double normalized_err = abs_err / scale;
      const double weighted_err = normalized_err * mode_weights_[k];
      max_error = std::max(max_error, weighted_err);
    }
    return max_error;
  }

  auto fluid_energy_error(const t u_n, const t u_nm1, const int q) -> double {
    double max_error = 0.0;
    for (int k = 0; k < num_modes_; ++k) {
      const double abs_err = std::abs(u_n(k, q) - u_nm1(k, q));
      const double scale = std::max(
          {scales_.energy_scale, std::abs(u_n(k, q)), std::abs(u_nm1(k, q))});
      const double normalized_err = abs_err / scale;
      const double weighted_err = normalized_err * mode_weights_[k];
      max_error = std::max(max_error, weighted_err);
    }
    return max_error;
  }

  auto radiation_energy_error(const t u_n, const t u_nm1, const int q)
      -> double {
    double max_error = 0.0;
    for (int k = 0; k < num_modes_; ++k) {
      const double abs_err = std::abs(u_n(k, q) - u_nm1(k, q));
      const double scale =
          std::max({scales_.rad_energy_scale, std::abs(u_n(k, q)),
                    std::abs(u_nm1(k, q))});
      const double normalized_err = abs_err / scale;
      const double weighted_err = normalized_err * mode_weights_[k];
      max_error = std::max(max_error, weighted_err);
    }
    return max_error;
  }

  auto radiation_flux_error(const t u_n, const t u_nm1, const int q) -> double {
    double max_error = 0.0;
    for (int k = 0; k < num_modes_; ++k) {
      const double abs_err = std::abs(u_n(k, q) - u_nm1(k, q));
      const double scale = std::max(
          {scales_.rad_flux_scale, std::abs(u_n(k, q)), std::abs(u_nm1(k, q))});
      const double normalized_err = abs_err / scale;
      const double weighted_err = normalized_err * mode_weights_[k];
      max_error = std::max(max_error, weighted_err);
    }
    return max_error;
  }

  // combined convergence check
  template <typename T>
  auto check_convergence(const T state_n, const T state_nm1) -> bool {

    double max_velocity_error = 0.0;
    double max_energy_error = 0.0;
    double max_rad_energy_error = 0.0;
    double max_rad_flux_error = 0.0;

    max_velocity_error = std::max(max_velocity_error,
                                  fluid_velocity_error(state_n, state_nm1, 1));
    max_energy_error =
        std::max(max_energy_error, fluid_energy_error(state_n, state_nm1, 2));
    max_rad_energy_error = std::max(
        max_rad_energy_error, radiation_energy_error(state_n, state_nm1, 3));
    max_rad_flux_error = std::max(max_rad_flux_error,
                                  radiation_flux_error(state_n, state_nm1, 4));

    bool velocity_converged = max_velocity_error < rel_tol_;
    bool energy_converged = max_energy_error < rel_tol_;
    bool rad_energy_converged = max_rad_energy_error < rel_tol_;
    bool rad_flux_converged = max_rad_flux_error < rel_tol_;

    return velocity_converged && energy_converged && rad_energy_converged &&
           rad_flux_converged;
  }
};

/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 * Assumes target is in f(x) = x form
 **/
template <typename T, typename F, typename... Args>
KOKKOS_INLINE_FUNCTION auto fixed_point_aa(F target, T x0, Args... args) -> T {

  unsigned int n = 0;

  T error = 1.0;
  T xkm1 = 0.0;
  T xk = 0.0;
  xk = target(x0, args...); // one fixed point step
  xkm1 = x0;
  if (std::abs(xk - x0) <= root_finders::ABSTOL) {
    return xk;
  }
  while (n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL) {
    //--- Anderson acceleration step --- //
    const T fk = target(xk, args...);
    const T fkm1 = target(xkm1, args...);
    const T rk = residual(fk, xk);
    const T rkm1 = residual(fkm1, xkm1);
    T alpha = -rk / (rkm1 - rk);

    T xkp1 = (alpha * fkm1) + ((1.0 - alpha) * fk);
    error = std::abs(xk - xkp1);

    xkm1 = xk;
    xk = xkp1;

    ++n;
  }

  return xk;
}

/**
 * Anderson accelerated fixed point solver templated on type, function, args...
 * Assumes target is in f(x) = 0 form
 **/
template <typename T, typename F, typename... Args>
KOKKOS_INLINE_FUNCTION auto fixed_point_aa_root(F target, T x0, Args... args)
    -> T {

  // puts f(x) = 0 into fixed point form
  auto f = [&](const double x, Args... args) { return target(x, args...) + x; };

  unsigned int n = 0;

  T error = 1.0;
  T xkm1 = 0.0;
  T xk = 0.0;
  xk = f(x0, args...); // one fixed point step
  xkm1 = x0;
  if (std::abs(xk - x0) <= root_finders::ABSTOL) {
    return xk;
  }
  while (n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL) {
    //--- Anderson acceleration step --- //
    const T fk = f(xk, args...);
    const T fkm1 = f(xkm1, args...);
    const T rk = residual(fk, xk);
    const T rkm1 = residual(fkm1, xkm1);
    const T alpha = alpha_aa(rk, rkm1);

    const T xkp1 = (alpha * fkm1) + ((1.0 - alpha) * fk);
    error = std::abs(xk - xkp1);

    xkm1 = xk;
    xk = xkp1;

    ++n;
  }

  return xk;
}

/* Fixed point solver templated on type, function, and args for func */
template <typename T, typename F, typename... Args>
auto fixed_point(F target, T x0, Args... args) -> T {

  unsigned int n = 0;
  T error = 1.0;
  while (n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL) {
    T x1 = target(x0, args...);
    error = std::abs(residual(target, x0, args...));
    x0 = x1;
    ++n;
  }

  return x0;
}

/* Fixed point solver templated on type, function, and args for func */
template <typename T, typename F, typename... Args>
auto fixed_point_root(F target, T x0, Args... args) -> T {

  // puts f(x) = 0 into fixed point form
  auto f = [&](const double x, Args... args) { return target(x, args...) + x; };

  unsigned int n = 0;
  T error = 1.0;
  while (n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL) {
    T x1 = f(x0, args...);
    error = std::abs(x1 - x0);
    x0 = x1;
    ++n;
  }

  return x0;
}

template <typename T, typename F>
auto fixed_point(F target, T x0) -> T {

  unsigned int n = 0;
  T error = 1.0;
  while (n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL) {
    T x1 = target(x0);
    error = std::abs(residual(target, x0));
    x0 = x1;
    ++n;
  }

  return x0;
}

// Newton iteration templated on type, function
template <typename T, typename F>
auto newton(F target, F d_target, T x0) -> T {

  unsigned int n = 0;
  T error = 1.0;
  while (n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL) {
    T xn = x0;
    T h = target(xn) / d_target(xn);
    x0 = xn - h;
    error = std::abs(xn - x0);
    ++n;
  }
  return x0;
}

// Newton iteration templated on type, function, args
template <typename T, typename F, typename G, typename... Args>
auto newton(F target, G d_target, T x0, Args... args) -> T {

  unsigned int n = 0;
  T error = 1.0;
  while (n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL) {
    T xn = x0;
    T h = target(xn, args...) / d_target(xn, args...);
    x0 = xn - h;
    error = std::abs(xn - x0);
    ++n;
  }
  return x0;
}

// Anderson Accelerated newton iteration templated on type, function
template <typename T, typename F, typename... Args>
auto newton_aa(F target, F d_target, T x0, Args... args) -> T {

  unsigned int n = 1;

  T h = target(x0, args...) / d_target(x0, args...);
  T error = 1.0;
  T xkm1 = 0.0;
  T xk = 0.0;
  T xkp1 = 0.0;
  xk = x0 - h; // kickstart with one iteration of standard newton raphson
  xkm1 = x0;
  if (std::abs(xk - x0) <= root_finders::ABSTOL) {
    return xk;
  }
  while (n <= root_finders::MAX_ITERS && error >= root_finders::ABSTOL) {
    const T hp1 = target(xk, args...) / d_target(xk, args...);

    // Anderson acceleration step
    const T gamma = alpha_aa(hp1, h);

    const double term = xk - xkm1 - hp1 + h;
    xkp1 = std::fma(gamma, -term, xk - hp1);
    error = std::abs(xk - xkp1);

    xkm1 = xk;
    xk = xkp1;
    h = hp1;

    ++n;
  }
  return xk;
}

// Error metric types - empty structs for compile-time dispatch
/**
 * @brief Absolute error metric.
 *
 * Convergence is determined by: |x_new - x_old| < abs_tol
 */
struct AbsoluteError {};
/**
 * @brief Relative error metric.
 *
 * Convergence is determined by: |x_new - x_old| < rel_tol * |x_new|
 */
struct RelativeError {};
/**
 * @brief Hybrid error metric.
 *
 * Convergence is determined by: |x_new - x_old| < abs_tol + rel_tol * |x_new|
 */
struct HybridError {};

/**
 * @brief Configuration structure for convergence tolerances and iteration
 * limits.
 *
 * This class encapsulates all parameters needed for convergence testing in root
 * finding algorithms. The error metric is specified at compile time through the
 * template parameter, allowing for zero-overhead convergence checking in
 * performance-critical code.
 *
 * @tparam T Floating-point type for tolerance values (e.g., double, float)
 * @tparam ErrorMetric Error metric type (AbsoluteError, RelativeError, or
 * CombinedError)
 *
 * @par Example Usage:
 * @code
 * // Create config with combined error checking
 * ToleranceConfig<double, CombinedError> config{1e-12, 1e-10, 100};
 *
 * // Check convergence
 * if (config.converged(x_new, x_old)) {
 *     // Solution has converged
 * }
 * @endcode
 */
template <typename T, typename ErrorMetric = HybridError>
struct ToleranceConfig {
  T abs_tol = T(1e-12);
  T rel_tol = T(1e-12);
  int max_iterations = 100;

  constexpr auto converged(T current, T previous) const -> bool {
    return converged_impl(current, previous, ErrorMetric{});
  }

 private:
  // Specialized implementations for each error metric
  constexpr bool converged_impl(T current, T previous,
                                AbsoluteError /*error*/) const {
    return std::abs(current - previous) < abs_tol;
  }

  constexpr bool converged_impl(T current, T previous,
                                RelativeError /*error*/) const {
    return std::abs(current - previous) < rel_tol * std::abs(current);
  }

  constexpr bool converged_impl(T current, T previous,
                                HybridError /*error*/) const {
    return std::abs(current - previous) < abs_tol + rel_tol * std::abs(current);
  }
};

/**
 * @brief Root finding class with compile-time algorithm and error metrics.
 *
 * This class provides an interface for different root finding algorithms.
 * The algorithm and error metric are selected at compile time,
 * allowing the optimizer to generate specialized code for each combination.
 *
 * TODO(astrobarker): Could be nice to have this return  a status type wrapping
 * std::expected
 *
 * @tparam T Floating-point type for computations (e.g., double, float)
 * @tparam Algorithm Root finding algorithm class (e.g., NewtonAlgorithm<T>)
 * @tparam ErrorMetric Error metric for convergence testing (default:
 * CombinedError)
 *
 * @par Design Philosophy:
 * - Zero virtual function overhead through templates
 * - Compile-time algorithm selection
 * - Flexible tolerance configuration per solver instance
 * - Type-safe interfaces enforced through C++20 concepts
 *
 * @par Example Usage:
 * @code
 * // Create solvers with different configurations
 * auto eos_solver = RootFinder<double, NewtonAlgorithm<double>,
 * RelativeError>{} .set_tolerance(1e-10, 1e-8) .set_max_iterations(50);
 *
 * auto integration_solver = RootFinder<double, NewtonAlgorithm<double>,
 * CombinedError>{} .set_tolerance(1e-14, 1e-12);
 *
 * // Solve equations
 * double temp = eos_solver.solve(eos_func, eos_dfunc, T_guess, pressure,
 * density);
 * @endcode
 */
template <typename T, typename Algorithm, typename ErrorMetric = HybridError>
class RootFinder {
 private:
  Algorithm algorithm_;
  ToleranceConfig<T, ErrorMetric> config_;

 public:
  RootFinder() = default;

  explicit RootFinder(const ToleranceConfig<T, ErrorMetric> config)
      : config_(config) {}

  RootFinder(const Algorithm &algo,
             const ToleranceConfig<T, ErrorMetric> &config)
      : algorithm_(algo), config_(config) {}

  // Setters for configuration
  auto set_tolerance(const T abs_tol, const T rel_tol) -> RootFinder & {
    config_.abs_tol = abs_tol;
    config_.rel_tol = rel_tol;
    return *this;
  }

  auto set_max_iterations(const int max_iter) -> RootFinder & {
    config_.max_iterations = max_iter;
    return *this;
  }

  // Solve methods - constrained by concepts
  template <typename F, typename G, typename... Args>
  auto solve(F func, G dfunc, T x0, Args &&...args) const -> T {
    return algorithm_(func, dfunc, x0, config_, std::forward<Args>(args)...);
  }

  template <typename F, typename... Args>
  auto solve(F func, T x0, Args &&...args) const -> T {
    return algorithm_(func, x0, config_, std::forward<Args>(args)...);
  }

  auto config() const noexcept -> ToleranceConfig<T, ErrorMetric> & {
    return config_;
  }
  auto algorithm() const noexcept -> Algorithm & { return algorithm_; }
};

/**
 * @brief Newton's method root finding algorithm implementation.
 *
 * This class implements the classic Newton-Raphson method.
 * x_{n+1} = x_n - f(x_n)/f'(x_n)
 *
 * @tparam T Floating-point type for computations
 */
template <typename T>
class NewtonAlgorithm {
 public:
  template <typename F, typename G, typename ErrorMetric, typename... Args>
  auto operator()(F target, G d_target, T x0,
                  const ToleranceConfig<T, ErrorMetric> &config,
                  Args &&...args) const -> T {
    T x = x0;
    for (int i = 0; i < config.max_iterations; ++i) {
      const T h = target(x, std::forward<Args>(args)...) /
                  d_target(x, std::forward<Args>(args)...);

      const T x_new = x - h;

      if (config.converged(x_new, x)) {
        return x_new;
      }
      x = x_new;
    }
    return x;
  }
};

/**
 * @brief Anderson accelerated Newton's method root finding algorithm.
 *
 * This class implements an Anderson accelerated Newton-Raphson method.
 * x_{n+1} = -gamma * (x_{n} - x_{n-1} - h_{n} + h_{n-1}) + x_{n-1} - h_{n}
 * with h_{n} = f(x_{n}) / f'(x_{n})
 * and gamma = h_{n} / (h_{n} - h_{n-1})
 *
 * @tparam T Floating-point type for computations
 */
template <typename T>
class AANewtonAlgorithm {
 public:
  template <typename F, typename G, typename ErrorMetric, typename... Args>
  auto operator()(F target, G d_target, T x0,
                  const ToleranceConfig<T, ErrorMetric> &config,
                  Args &&...args) const -> T {
    T x = x0;

    // Jumpstart the AA algorithm with 1 iteration of the base algorithm
    T h = target(x, std::forward<Args>(args)...) /
          d_target(x, std::forward<Args>(args)...);
    T x_new = x - h;
    // Must check convergence before moving on.
    if (config.converged(x_new, x0)) {
      return x_new;
    }
    T x_prev = x0;
    x = x_new;
    for (int i = 1; i <= config.max_iterations; ++i) {
      const T h_new = target(x, std::forward<Args>(args)...) /
                      d_target(x, std::forward<Args>(args)...);
      const T gamma = alpha_aa(h_new, h);

      x_new = std::fma(1.0 - gamma, x - h_new, gamma * (x_prev - h));

      if (config.converged(x_new, x)) {
        return x_new;
      }
      x_prev = x;
      x = x_new;
      h = h_new;
    }
    return x;
  }
};

/**
 * @brief Fixed-point algorithm for solving equations of the form x = g(x).
 *
 * This class implements the basic fixed-point iteration: x_{n+1} = g(x_n)
 *
 * @tparam T Floating-point type for computations
 *
 * @par Usage Note:
 * To solve f(x) = 0, reformulate as x = x - f(x) = g(x), then use this solver.
 */
template <typename T>
class FixedPointAlgorithm {
 public:
  template <typename F, typename ErrorMetric, typename... Args>
  auto operator()(F target, T x0, const ToleranceConfig<T, ErrorMetric> &config,
                  Args &&...args) const -> T {
    T x = x0;
    for (int i = 0; i < config.max_iterations; ++i) {
      const T x_new = target(x, std::forward<Args>(args)...);

      if (config.converged(x_new, x)) {
        return x_new;
      }
      x = x_new;
    }
    return x;
  }
};

/**
 * @brief Anderson Accelerated Fixed-point algorithm for root finding.
 *
 * This class implements an Anderson accelerated fixed-point iteration:
 * x_{n+1} = (alpha * f_{n-1}) + (1.0 - alpha) * f(x_{n})
 * with alpha = r_{n} / (r_{n} - r_{n-1})
 * and r_{n} = f(x_{n}) - x_{n} is the residual
 *
 * @tparam T Floating-point type for computations
 *
 * @par Usage Note:
 * To solve f(x) = 0, reformulate as x = x - f(x) = g(x), then use this solver.
 *
 * TODO(astrobarker): The loop/storage can be optimized to only evaluate
 * target once per iteration.
 */
template <typename T>
class AAFixedPointAlgorithm {
 public:
  template <typename F, typename ErrorMetric, typename... Args>
  auto operator()(F target, T x0, const ToleranceConfig<T, ErrorMetric> &config,
                  Args &&...args) const -> T {
    T x = x0;
    // Jumpstart the AA algorithm with 1 iteration of the base algorithm
    T x_new = target(x, args...);
    // Must check convergence before moving on.
    if (config.converged(x_new, x0)) {
      return x_new;
    }
    T x_prev = x0;
    for (int i = 1; i < config.max_iterations; ++i) {
      const T x_new = target(x, std::forward<Args>(args)...);
      T fn = target(x_new, std::forward<Args>(args)...);
      T fnm1 = target(x_prev, std::forward<Args>(args)...);
      T rn = residual(fn, x_new);
      T rnm1 = residual(fnm1, x_prev);
      const T alpha = alpha_aa(rn, rnm1);

      T xkp1 = (alpha * fnm1) + ((1.0 - alpha) * fn);

      if (config.converged(x_new, x)) {
        return x_new;
      }
      x_prev = x;
      x = x_new;
    }
    return x;
  }
};

} // namespace athelas::root_finders
