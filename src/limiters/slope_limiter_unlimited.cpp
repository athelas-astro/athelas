#include "limiters/slope_limiter.hpp"

namespace athelas {

/*
 * No op
 */
void Unlimited::apply_slope_limiter(AthelasArray3D<double> /*U*/,
                                    const GridStructure * /*grid*/,
                                    const basis::ModalBasis & /*basis*/,
                                    const eos::EOS & /*eos*/) {}

auto Unlimited::get_limited(const int /*i*/) const -> int { return 0.0; }
auto Unlimited::limited() const -> AthelasArray1D<int> { return limited_cell_; }
} // namespace athelas
