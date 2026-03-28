#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "linalg/linear_algebra.hpp"
#include "test_utils.hpp"

using athelas::block_thomas_solve;
using athelas::ThomasScratch;

TEST_CASE("Block Thomas Solver", "[Block Thomas]") {

  // -----------------------------------------------------------------
  // TC-1: N=1 — degenerates to a single dense block solve.
  // Verifies the sweep/back-sub loops are never entered.
  // -----------------------------------------------------------------
  SECTION("N=1 degenerates to single block solve") {
    constexpr int N = 1, m = 3;
    BlockStore A("A", N, m, m), B("B", N, m, m), C("C", N, m, m);
    VecStore d("d", N, m);
    BlockStore W("W", (N > 0 ? N - 1 : 0), m, m);
    VecStore Y("Y", (N > 0 ? N - 1 : 0), m);

    Kokkos::View<Scalar **, Layout, MemSpace> Bi_lu("Bi_lu", m, m);
    ThomasScratch scratch{.W = W,
                          .Y = Y,
                          .Bi_lu = Bi_lu};

    // B(0) = 2*I,  b = [2, 4, 6]  =>  x = [1, 2, 3]
    set_block(B, 0, m, {2, 0, 0, 0, 2, 0, 0, 0, 2});
    set_vec(d, 0, m, {2.0, 4.0, 6.0});

    block_thomas_solve(N, m, A, B, C, d, scratch);

    const auto x = get_vec(d, 0, m);
    REQUIRE(x[0] == Catch::Approx(1.0).epsilon(1e-10));
    REQUIRE(x[1] == Catch::Approx(2.0).epsilon(1e-10));
    REQUIRE(x[2] == Catch::Approx(3.0).epsilon(1e-10));
  }

  // -----------------------------------------------------------------
  // TC-2: N=2 — one forward step and one back-sub step.
  // -----------------------------------------------------------------
  SECTION("N=2 minimal tridiagonal") {
    constexpr int N = 2, m = 2;
    BlockStore A("A", N - 1, m, m), B("B", N, m, m), C("C", N - 1, m, m);
    VecStore d("d", N, m);
    BlockStore W("W", N - 1, m, m);
    VecStore Y("Y", N - 1, m);

    Kokkos::View<Scalar **, Layout, MemSpace> Bi_lu("Bi_lu", m, m);
    ThomasScratch scratch{.W = W,
                          .Y = Y,
                          .Bi_lu = Bi_lu};

    // B(i) = 3*I,  C(0) = A(0) = -I,  x_exact = ones
    set_block(B, 0, m, {3, 0, 0, 3});
    set_block(B, 1, m, {3, 0, 0, 3});
    set_block(C, 0, m, {-1, 0, 0, -1});
    set_block(A, 0, m, {-1, 0, 0, -1});
    set_vec(d, 0, m, {2.0, 2.0});
    set_vec(d, 1, m, {2.0, 2.0});

    block_thomas_solve(N, m, A, B, C, d, scratch);

    for (int i = 0; i < N; ++i) {
      const auto x = get_vec(d, i, m);
      for (int j = 0; j < m; ++j)
        REQUIRE(x[static_cast<std::size_t>(j)] ==
                Catch::Approx(1.0).epsilon(1e-10));
    }
  }

  // -----------------------------------------------------------------
  // TC-3: N=5, m=3 — canonical demo case, x_exact = ones.
  // -----------------------------------------------------------------
  SECTION("N=5 m=3 diagonal block system, x_exact=ones") {
    constexpr int N = 5, m = 3;
    BlockStore A("A", N - 1, m, m), B("B", N, m, m), C("C", N - 1, m, m);
    VecStore d("d", N, m);
    BlockStore W("W", N - 1, m, m);
    VecStore Y("Y", N - 1, m);

    build_identity_block_problem(N, m, A, B, C, d);
    Kokkos::View<Scalar **, Layout, MemSpace> Bi_lu("Bi_lu", m, m);
    ThomasScratch scratch{.W = W,
                          .Y = Y,
                          .Bi_lu = Bi_lu};

    block_thomas_solve(N, m, A, B, C, d, scratch);

    const std::vector<double> x_exact(static_cast<std::size_t>(N * m), 1.0);
    REQUIRE(max_error(d, N, m, x_exact) < 1e-10);
  }

  // -----------------------------------------------------------------
  // TC-4: N=20, m=4 — stress test loop counts, catches off-by-one
  // errors in the sweep or back substitution.
  // -----------------------------------------------------------------
  SECTION("N=20 m=4 diagonal block system, x_exact=ones") {
    constexpr int N = 20, m = 4;
    BlockStore A("A", N - 1, m, m), B("B", N, m, m), C("C", N - 1, m, m);
    VecStore d("d", N, m);
    BlockStore W("W", N - 1, m, m);
    VecStore Y("Y", N - 1, m);

    build_identity_block_problem(N, m, A, B, C, d);
    Kokkos::View<Scalar **, Layout, MemSpace> Bi_lu("Bi_lu", m, m);
    ThomasScratch scratch{.W = W,
                          .Y = Y,
                          .Bi_lu = Bi_lu};

    block_thomas_solve(N, m, A, B, C, d, scratch);

    const std::vector<double> x_exact(static_cast<std::size_t>(N * m), 1.0);
    REQUIRE(max_error(d, N, m, x_exact) < 1e-10);
  }

  // -----------------------------------------------------------------
  // TC-5: N=3, m=3, dense lower-triangular off-diagonal blocks.
  // Exercises the full gemm/gemv paths with non-trivial coupling.
  //
  // B(i) = 20*I,  C(i) = A(i) = [[2,0,0],[1,2,0],[0,1,2]]
  // x_exact = [1, 2, 3] at every block.
  // -----------------------------------------------------------------
  SECTION("N=3 m=3 dense lower-triangular off-diagonal blocks") {
    constexpr int N = 3, m = 3;
    const std::vector<double> x_block = {1.0, 2.0, 3.0};

    BlockStore A("A", N - 1, m, m), B("B", N, m, m), C("C", N - 1, m, m);
    VecStore d("d", N, m);
    BlockStore W("W", N - 1, m, m);
    VecStore Y("Y", N - 1, m);

    Kokkos::View<Scalar **, Layout, MemSpace> Bi_lu("Bi_lu", m, m);
    ThomasScratch scratch{.W = W,
                          .Y = Y,
                          .Bi_lu = Bi_lu};

    auto A_h = Kokkos::create_mirror_view(A);
    auto B_h = Kokkos::create_mirror_view(B);
    auto C_h = Kokkos::create_mirror_view(C);
    auto d_h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(A_h, Scalar(0));
    Kokkos::deep_copy(B_h, Scalar(0));
    Kokkos::deep_copy(C_h, Scalar(0));
    Kokkos::deep_copy(d_h, Scalar(0));

    const double off[3][3] = {{2, 0, 0}, {1, 2, 0}, {0, 1, 2}};
    constexpr double bdiag = 20.0;

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < m; ++j)
        B_h(i, j, j) = bdiag;

    for (int i = 0; i < N - 1; ++i)
      for (int r = 0; r < m; ++r)
        for (int c = 0; c < m; ++c) {
          C_h(i, r, c) = off[r][c];
          A_h(i, r, c) = off[r][c];
        }

    for (int i = 0; i < N; ++i)
      for (int r = 0; r < m; ++r) {
        double val = B_h(i, r, r) * x_block[static_cast<std::size_t>(r)];
        if (i < N - 1)
          for (int c = 0; c < m; ++c)
            val += C_h(i, r, c) * x_block[static_cast<std::size_t>(c)];
        if (i > 0)
          for (int c = 0; c < m; ++c)
            val += A_h(i - 1, r, c) * x_block[static_cast<std::size_t>(c)];
        d_h(i, r) = val;
      }

    Kokkos::deep_copy(A, A_h);
    Kokkos::deep_copy(B, B_h);
    Kokkos::deep_copy(C, C_h);
    Kokkos::deep_copy(d, d_h);

    block_thomas_solve(N, m, A, B, C, d, scratch);

    for (int i = 0; i < N; ++i) {
      const auto x = get_vec(d, i, m);
      for (int j = 0; j < m; ++j)
        REQUIRE(
            x[static_cast<std::size_t>(j)] ==
            Catch::Approx(x_block[static_cast<std::size_t>(j)]).epsilon(1e-9));
    }
  }

  // -----------------------------------------------------------------
  // TC-6: N=5, m=2, x_exact(i) = (i+1)*ones(m).
  // A non-uniform solution catches accidental hardcoding to all-ones.
  // -----------------------------------------------------------------
  SECTION("N=5 m=2 varying x_exact per block (x_i = i+1)") {
    constexpr int N = 5, m = 2;
    constexpr double bdiag = 6.0;

    BlockStore A("A", N - 1, m, m), B("B", N, m, m), C("C", N - 1, m, m);
    VecStore d("d", N, m);
    BlockStore W("W", N - 1, m, m);
    VecStore Y("Y", N - 1, m);

    Kokkos::View<Scalar **, Layout, MemSpace> Bi_lu("Bi_lu", m, m);
    ThomasScratch scratch{.W = W,
                          .Y = Y,
                          .Bi_lu = Bi_lu};

    auto A_h = Kokkos::create_mirror_view(A);
    auto B_h = Kokkos::create_mirror_view(B);
    auto C_h = Kokkos::create_mirror_view(C);
    auto d_h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(A_h, Scalar(0));
    Kokkos::deep_copy(B_h, Scalar(0));
    Kokkos::deep_copy(C_h, Scalar(0));
    Kokkos::deep_copy(d_h, Scalar(0));

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < m; ++j)
        B_h(i, j, j) = bdiag;
    for (int i = 0; i < N - 1; ++i)
      for (int j = 0; j < m; ++j) {
        A_h(i, j, j) = -1.0;
        C_h(i, j, j) = -1.0;
      }

    for (int i = 0; i < N; ++i) {
      const double xi = i + 1.0;
      const double xim1 = (i > 0) ? static_cast<double>(i) : 0.0;
      const double xip1 = (i < N - 1) ? static_cast<double>(i + 2) : 0.0;
      for (int j = 0; j < m; ++j)
        d_h(i, j) = -xim1 + bdiag * xi - xip1;
    }

    Kokkos::deep_copy(A, A_h);
    Kokkos::deep_copy(B, B_h);
    Kokkos::deep_copy(C, C_h);
    Kokkos::deep_copy(d, d_h);

    block_thomas_solve(N, m, A, B, C, d, scratch);

    for (int i = 0; i < N; ++i) {
      const auto x = get_vec(d, i, m);
      const double expected = i + 1.0;
      for (int j = 0; j < m; ++j)
        REQUIRE(x[static_cast<std::size_t>(j)] ==
                Catch::Approx(expected).epsilon(1e-10));
    }
  }
}
