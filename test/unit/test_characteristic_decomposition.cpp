#include <catch2/catch_test_macros.hpp>

#include "eos/eos_variant.hpp"
#include "limiters/characteristic_decomposition.hpp"
#include "test_utils.hpp"

namespace {

template <class M>
auto matrix_product_entry(M A, M B, const int i, const int j, const int n)
    -> double {
  double sum = 0.0;
  for (int k = 0; k < n; ++k) {
    sum += A(i, k) * B(k, j);
  }
  return sum;
}

template <class M>
void require_inverse_pair(M R, M R_inv, const int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double expected = (i == j) ? 1.0 : 0.0;
      REQUIRE(soft_equal(matrix_product_entry(R_inv, R, i, j, n), expected,
                         1.0e-10));
    }
  }
}

} // namespace

TEST_CASE("Hydro characteristic decomposition is invertible", "[limiters]") {
  using athelas::compute_characteristic_decomposition;
  using athelas::eos::EOS;
  using athelas::eos::IdealGas;

  Kokkos::View<double *, Kokkos::HostSpace> U("U", 3);
  Kokkos::View<double **, Kokkos::HostSpace> R("R", 3, 3);
  Kokkos::View<double **, Kokkos::HostSpace> R_inv("R_inv", 3, 3);
  const EOS eos = IdealGas(5.0 / 3.0);

  U(0) = 0.25;
  U(1) = 1.0e7;
  U(2) = 3.0e15;

  athelas::eos::EOSLambda lambda;
  lambda.ptr()[athelas::eos::EOS_LAMBDA_TEMPERATURE] = 1.0e8;
  compute_characteristic_decomposition(U, R, R_inv, eos, lambda.ptr());
  REQUIRE(R(1, 1) > 0.0);
  REQUIRE(R(1, 2) < 0.0);
  require_inverse_pair(R, R_inv, 3);
}

TEST_CASE("Radiation characteristic decomposition is invertible",
          "[limiters]") {
  using athelas::compute_characteristic_decomposition;
  using athelas::constants::c_cgs;
  using athelas::eos::EOS;
  using athelas::eos::IdealGas;

  Kokkos::View<double *, Kokkos::HostSpace> U("U", 2);
  Kokkos::View<double **, Kokkos::HostSpace> R("R", 2, 2);
  Kokkos::View<double **, Kokkos::HostSpace> R_inv("R_inv", 2, 2);
  const EOS eos = IdealGas(5.0 / 3.0);

  U(0) = 1.0e12;
  U(1) = 0.2 * c_cgs * U(0);

  athelas::eos::EOSLambda lambda;
  compute_characteristic_decomposition(U, R, R_inv, eos, lambda.ptr());
  REQUIRE(R(1, 0) < 0.0);
  REQUIRE(R(1, 1) > 0.0);
  require_inverse_pair(R, R_inv, 2);
}
