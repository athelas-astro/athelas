#ifndef _GRID_HPP_
#define _GRID_HPP_

/**
 * File     :  Grid.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the grid data.
 *  For a loop over real zones, loop from ilo to ihi (inclusive).
 *  ilo = nGhost
 *  ihi = nElements - nGhost + 1
 *
 **/

#include <algorithm> // std::copy
#include <iostream>
#include <vector>

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Error.hpp"
#include "Geometry.hpp"
#include "ProblemIn.hpp"
#include "QuadratureLibrary.hpp"

class GridStructure {
 public:
  GridStructure( ProblemIn *pin );
  Real NodeCoordinate( int iC, int iN ) const;
  Real Get_Centers( int iC ) const;
  Real Get_Widths( int iC ) const;
  Real Get_Nodes( int nN ) const;
  Real Get_Weights( int nN ) const;
  Real Get_Mass( int iX ) const;
  Real Get_CenterOfMass( int iX ) const;
  Real Get_xL( ) const;
  Real Get_xR( ) const;
  Real Get_SqrtGm( Real X ) const;
  Real Get_LeftInterface( int iX ) const;

  bool DoGeometry( ) const;

  int Get_Guard( ) const;
  int Get_ilo( ) const;
  int Get_ihi( ) const;
  int Get_nNodes( ) const;
  int Get_nElements( ) const;

  void CreateGrid( );
  void UpdateGrid( Kokkos::View<Real *> SData );
  void ComputeMass( Kokkos::View<Real ***> uPF );
  void ComputeCenterOfMass( Kokkos::View<Real ***> uPF );
  void ComputeCenterOfMass_Radius( Kokkos::View<Real ***> uPF );

  Real &operator( )( int i, int j );
  Real operator( )( int i, int j ) const;

 private:
  int nElements;
  int nNodes;
  int nGhost;
  int mSize;

  Real xL;
  Real xR;

  geometry::Geometry Geometry;

  Kokkos::View<Real *> Nodes;
  Kokkos::View<Real *> Weights;

  Kokkos::View<Real *> Centers;
  Kokkos::View<Real *> Widths;
  Kokkos::View<Real *> X_L; // left interface coordinate

  Kokkos::View<Real *> Mass;
  Kokkos::View<Real *> CenterOfMass;

  Kokkos::View<Real **> Grid;
};

#endif // _GRID_HPP_
