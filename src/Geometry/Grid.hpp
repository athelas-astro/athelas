#ifndef GRID_H
#define GRID_H

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


class GridStructure
{
 public:
  GridStructure( ProblemIn *pin );
  Real NodeCoordinate( UInt iC, UInt iN ) const;
  Real Get_Centers( UInt iC ) const;
  Real Get_Widths( UInt iC ) const;
  Real Get_Nodes( UInt nN ) const;
  Real Get_Weights( UInt nN ) const;
  Real Get_Mass( UInt iX ) const;
  Real Get_CenterOfMass( UInt iX ) const;
  Real Get_xL( ) const;
  Real Get_xR( ) const;
  Real Get_SqrtGm( Real X ) const;
  Real Get_LeftInterface( UInt iX ) const;

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

  Real &operator( )( UInt i, UInt j );
  Real operator( )( UInt i, UInt j ) const;

 private:
  UInt nElements;
  UInt nNodes;
  UInt nGhost;
  UInt mSize;

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

#endif
