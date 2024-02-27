#ifndef _POLYNOMIALBASIS_HPP_
#define _POLYNOMIALBASIS_HPP_

/**
 * File     :  PolynomialBasis.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Functions for polynomial basis
 * Contains : Class for Taylor basis.
 * Also  Lagrange, Legendre polynomials, arbitrary degree.
 **/

#include <algorithm> // std::copy
#include <vector>

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Grid.hpp"

typedef Real BasisFuncType( int, Real, Real );

class ModalBasis {
 public:
  ModalBasis( PolyBasis::PolyBasis basis, Kokkos::View<Real ***> uCF,
              GridStructure *Grid, int pOrder, int nN, int nElements,
              int nGuard );
  static Real Taylor( int order, Real eta, Real eta_c );
  static Real dTaylor( int order, Real eta, Real eta_c );
  Real Ortho( const int order, const int iX, const int i_eta, const Real eta,
              Real eta_c, const Kokkos::View<Real ***> uCF, GridStructure *Grid,
              const bool derivative_option );
  Real InnerProduct( const int m, const int n, const int iX, const Real eta_c,
                     const Kokkos::View<Real ***> uCF, GridStructure *Grid );
  Real InnerProduct( const int n, const int iX, const Real eta_c,
                     const Kokkos::View<Real ***> uCF, GridStructure *Grid );
  void InitializeTaylorBasis( const Kokkos::View<Real ***> U,
                              GridStructure *Grid );
  void InitializeBasis( const PolyBasis::PolyBasis basis,
                        const Kokkos::View<Real ***> uCF, GridStructure *Grid );
  void CheckOrthogonality( const Kokkos::View<Real ***> uCF,
                           GridStructure *Grid );
  Real BasisEval( Kokkos::View<Real ***> U, const int iX, const int iCF,
                  const int i_eta, const bool DerivativeOption ) const;
  void ComputeMassMatrix( const Kokkos::View<Real ***> uCF,
                          GridStructure *Grid );

  Real Get_Phi( int iX, int i_eta, int k ) const;
  Real Get_dPhi( int iX, int i_eta, int k ) const;
  Real Get_MassMatrix( int iX, int k ) const;

  int Get_Order( ) const;

  static Real Legendre( int order, Real x );
  static Real dLegendre( int order, Real x );
  static Real Legendre( int order, Real x, Real x_c );
  static Real dLegendre( int order, Real x, Real x_c );

 private:
  int nX;
  int order;
  int nNodes;
  int mSize;

  Kokkos::View<Real **> MassMatrix;

  Kokkos::View<Real ***> Phi;
  Kokkos::View<Real ***> dPhi;

  Real ( *func )( int n, Real x, Real x_c );
  Real ( *dfunc )( int n, Real x, Real x_c );
};

#endif // _POLYNOMIALBASIS_HPP_
