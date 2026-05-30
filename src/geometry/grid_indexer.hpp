#pragma once
#include "Kokkos_Macros.hpp"
#include <cassert>
#include <tuple>

/**
 * @class RadialGridIndexer
 * @brief Handles indexing conversion between 2D radial mesh coordinates and
 * flattened indices
 */
class RadialGridIndexer {
 private:
  int nx_;
  int nnodes_;
  int total_;

 public:
  /**
   * @brief Construct indexer for radial mesh
   * @param nx Number of radial cells (must be > 0)
   * @param nnodes Number of quadrature points per cell (must be > 0)
   */
  RadialGridIndexer(int nx, int nnodes)
      : nx_(nx), nnodes_(nnodes), total_(nx * nnodes) {
    assert(nx > 0 && "Number of cells must be positive");
    assert(nnodes > 0 && "Number of nodes per cell must be positive");
  }

  /**
   * @brief Convert 2D mesh coordinates to flattened index
   * @param ix Cell index [0, nx)
   * @param in Node index within cell [0, nnodes)
   * @return Flattened index [0, nx * nnodes)
   */
  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto to_flat(int ix, int in) const
      -> int {
    assert(ix >= 0 && ix <= nx_ && "Cell index out of bounds");
    assert(in >= 0 && in < nnodes_ && "Node index out of bounds");
    return ix * nnodes_ + in;
  }

  /**
   * @brief Convert flattened index to 2D mesh coordinates
   * @param i_flat Flattened index [0, nx * nnodes)
   * @return std::tuple<int, int> containing (ix, in)
   */
  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto to_2d(int i_flat) const
      -> std::tuple<int, int> {
    assert(i_flat >= 0 && i_flat <= total_ && "Flattened index out of bounds");
    return std::make_tuple(i_flat / nnodes_, i_flat % nnodes_);
  }

  // Accessor methods
  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto nx() const noexcept -> int {
    return nx_;
  }
  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto nnodes() const noexcept
      -> int {
    return nnodes_;
  }
  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto total_size() const noexcept
      -> int {
    return total_;
  }

  /**
   * @brief Get the starting flattened index for a given cell
   * @param ix Cell index [0, nx)
   * @return Starting flattened index for cell ix
   */
  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto cell_start(int ix) const
      -> int {
    assert(ix >= 0 && ix <= nx_ && "Cell index out of bounds");
    return ix * nnodes_;
  }

  /**
   * @brief Get the ending flattened index (exclusive) for a given cell
   * @param ix Cell index [0, nx)
   * @return Ending flattened index for cell ix (exclusive)
   */
  [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION auto cell_end(int ix) const -> int {
    assert(ix >= 0 && ix <= nx_ && "Cell index out of bounds");
    return (ix + 1) * nnodes_;
  }
};
