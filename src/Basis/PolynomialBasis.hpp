#ifndef POLYNOMIALBASIS_H
#define POLYNOMIALBASIS_H

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

typedef Real BasisFuncType( UInt, Real, Real );

class ModalBasis
{
 public:
  ModalBasis( PolyBasis::PolyBasis basis, Kokkos::View<Real ***> uCF, 
              GridStructure *Grid, UInt pOrder,
              UInt nN, UInt nElements, UInt nGuard );
  static Real Taylor( UInt order, Real eta, Real eta_c );
  static Real dTaylor( UInt order, Real eta, Real eta_c );
  Real Ortho( const UInt order, const UInt iX, const UInt i_eta,
              const Real eta, Real eta_c,
              const Kokkos::View<Real ***> uCF, GridStructure *Grid,
              const bool derivative_option );
  Real InnerProduct( const UInt m, const UInt n, const UInt iX,
                     const Real eta_c, const Kokkos::View<Real ***> uCF,
                     GridStructure *Grid );
  Real InnerProduct( const UInt n, const UInt iX, const Real eta_c,
                     const Kokkos::View<Real ***> uCF, GridStructure *Grid );
  void InitializeTaylorBasis( const Kokkos::View<Real ***> U,
                              GridStructure *Grid );
  void InitializeBasis( const PolyBasis::PolyBasis basis, 
                        const Kokkos::View<Real ***> uCF,
                        GridStructure *Grid );
  void CheckOrthogonality( const Kokkos::View<Real ***> uCF,
                           GridStructure *Grid );
  Real BasisEval( Kokkos::View<Real ***> U, const UInt iX, const UInt iCF,
                  const UInt i_eta, const bool DerivativeOption ) const;
  void ComputeMassMatrix( const Kokkos::View<Real ***> uCF,
                          GridStructure *Grid );

  Real Get_Phi( UInt iX, UInt i_eta, UInt k ) const;
  Real Get_dPhi( UInt iX, UInt i_eta, UInt k ) const;
  Real Get_MassMatrix( UInt iX, UInt k ) const;

  int Get_Order( ) const;

  static Real Legendre( UInt order, Real x );
  static Real dLegendre( UInt order, Real x );
  static Real Legendre( UInt order, Real x, Real x_c );
  static Real dLegendre( UInt order, Real x, Real x_c );

 private:
  UInt nX;
  UInt order;
  UInt nNodes;
  UInt mSize;

  Kokkos::View<Real **> MassMatrix;

  Kokkos::View<Real ***> Phi;
  Kokkos::View<Real ***> dPhi;

  Real (*func)(UInt n, Real x, Real x_c );
  Real (*dfunc)(UInt n, Real x, Real x_c );
};

#endif
