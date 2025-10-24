#pragma once

#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "state/state.hpp"

namespace athelas::atom {

void fill_derived_comps(State *state, const GridStructure *grid,
                        const basis::ModalBasis *basis);
void fill_derived_ionization(State *state, const GridStructure *grid,
                             const basis::ModalBasis *basis);

void paczynski_terms(const State *state, int ix, int node, double *lambda);

// Compute total element number density (all ionization states)
auto element_number_density(double mass_frac, double atomic_mass, double rho)
    -> double;

// Compute electron number density (derived quantity)
auto electron_density(const AthelasArray3D<double> mass_fractions,
                      const AthelasArray4D<double> ion_fractions,
                      const AthelasArray1D<int> charges, int i, int q,
                      double rho) -> double;

} // namespace athelas::atom
