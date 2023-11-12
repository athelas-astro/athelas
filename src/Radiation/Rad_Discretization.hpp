#ifndef _RAD_DISCRETIZATION_HPP_
#define _RAD_DISCRETIZATION_HPP_

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"

void ComputeIncrement_Rad_Divergence(
    const Kokkos::View<Real ***> U, GridStructure *Grid, ModalBasis *Basis,
    Kokkos::View<Real ***> dU, Kokkos::View<Real ***> Flux_q,
    Kokkos::View<Real **> dFlux_num, Kokkos::View<Real **> uCF_F_L,
    Kokkos::View<Real **> uCF_F_R, Kokkos::View<Real *> Flux_U,
    Kokkos::View<Real *> Flux_P );

void Compute_Increment_Explicit_Rad(
    const Kokkos::View<Real ***> U, GridStructure *Grid, ModalBasis *Basis,
    Kokkos::View<Real ***> dU, Kokkos::View<Real ***> Flux_q,
    Kokkos::View<Real **> dFlux_num, Kokkos::View<Real **> uCF_F_L,
    Kokkos::View<Real **> uCF_F_R, Kokkos::View<Real *> Flux_U,
    Kokkos::View<Real *> Flux_P, const std::string BC );

#endif
