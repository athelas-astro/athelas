/**
 * File     :  Grid.h
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the grid data.
 *  For a loop over real zones, loop from ilo to ihi (inclusive).
 *  ilo = nGhost
 *  ihi = nElements - nGhost + 1
 *
 * TODO: Can we initialize a Grid object from another?
 **/

#include "Grid.hpp"
#include "Constants.hpp"
#include <math.h> /* atan */

GridStructure::GridStructure( ProblemIn *pin ) 
    : nElements( pin->nElements ), nNodes( pin->nNodes ), 
      nGhost( pin->nGhost ), mSize( nElements + 2 * nGhost ), 
      xL( pin->xL ), xR( pin->xR ),
      Geometry( pin->Geometry ), Nodes( "Nodes", pin->nNodes ), 
      Weights( "Weights", pin->nNodes ), Centers( "Cetners", mSize ), 
      Widths( "Widths", mSize ), X_L( "Left Interface", mSize ), 
      Mass( "Cell Mass", mSize ), CenterOfMass( "Center of Mass", mSize ), 
      Grid( "Grid", mSize, nNodes )
{
  // TODO: Allow LG_Quadrature to take in vectors.
  Real *tmp_nodes   = new Real[nNodes];
  Real *tmp_weights = new Real[nNodes];

  for ( UInt iN = 0; iN < nNodes; iN++ )
  {
    tmp_nodes[iN]   = 0.0;
    tmp_weights[iN] = 0.0;
  }

  quadrature::LG_Quadrature( nNodes, tmp_nodes, tmp_weights );

  for ( UInt iN = 0; iN < nNodes; iN++ )
  {
    Nodes( iN )   = tmp_nodes[iN];
    Weights( iN ) = tmp_weights[iN];
  }

  CreateGrid( );

  delete[] tmp_nodes;
  delete[] tmp_weights;
}

// linear shape function on the reference element
KOKKOS_INLINE_FUNCTION
const Real ShapeFunction( const int interface, const Real eta )
{
  Real mult = 1.0;

  if ( interface == 0 ) mult = -1.0;
  if ( interface == 1 ) mult = +1.0;
  if ( interface != 0 && interface != 1 )
    throw Error( " ! Invalid shape func params" );

  return 0.5 + mult * eta;
}

// Give physical grid coordinate from a node.
Real GridStructure::NodeCoordinate( UInt iC, UInt iN ) const
{
  return X_L( iC ) * ShapeFunction( 0, Nodes( iN ) ) +
         X_L( iC + 1 ) * ShapeFunction( 1, Nodes( iN ) );
}

// Return cell center
Real GridStructure::Get_Centers( UInt iC ) const { return Centers( iC ); }

// Return cell width
Real GridStructure::Get_Widths( UInt iC ) const { return Widths( iC ); }

// Return cell mass
Real GridStructure::Get_Mass( UInt iX ) const { return Mass( iX ); }

// Return cell reference Center of Mass
Real GridStructure::Get_CenterOfMass( UInt iX ) const
{
  return CenterOfMass( iX );
}

// Return given quadrature node
Real GridStructure::Get_Nodes( UInt nN ) const { return Nodes( nN ); }

// Return given quadrature weight
Real GridStructure::Get_Weights( UInt nN ) const { return Weights( nN ); }

// Acessor for xL
Real GridStructure::Get_xL( ) const { return xL; }

// Acessor for xR
Real GridStructure::Get_xR( ) const { return xR; }

// Acessor for SqrtGm
Real GridStructure::Get_SqrtGm( Real X ) const
{
  if ( Geometry == geometry::Spherical )
  {
    return X * X;
  }
  else
  {
    return 1.0;
  }
}

// Accessor for X_L
Real GridStructure::Get_LeftInterface( UInt iX ) const { return X_L( iX ); }

// Return nNodes
int GridStructure::Get_nNodes( ) const { return nNodes; }

// Return nElements
int GridStructure::Get_nElements( ) const { return nElements; }

// Return number of guard zones
int GridStructure::Get_Guard( ) const { return nGhost; }

// Return first physical zone
int GridStructure::Get_ilo( ) const { return nGhost; }

// Return last physical zone
int GridStructure::Get_ihi( ) const { return nElements + nGhost - 1; }

// Return true if in spherical symmetry
bool GridStructure::DoGeometry( ) const { return Geometry == geometry::Spherical ? true : false; }

// Equidistant mesh
// TODO: We will need to replace Centers here, right?
void GridStructure::CreateGrid( )
{

  const UInt ilo = nGhost;                 // first real zone
  const UInt ihi = nElements + nGhost - 1; // last real zone

  for ( UInt i = 0; i < nElements + 2 * nGhost; i++ )
  {
    Widths( i ) = ( xR - xL ) / nElements;
  }

  X_L( nGhost ) = xL;
  for ( UInt iX = 2; iX < nElements + 2 * nGhost; iX++ )
  {
    X_L( iX ) = X_L( iX - 1 ) + Widths( iX - 1 );
  }

  Centers( ilo ) = xL + 0.5 * Widths( ilo );
  for ( UInt i = ilo + 1; i <= ihi; i++ )
  {
    Centers( i ) = Centers( i - 1 ) + Widths( i - 1 );
  }

  for ( int i = ilo - 1; i >= 0; i-- )
  {
    Centers( i ) = Centers( i + 1 ) - Widths( i + 1 );
  }
  for ( UInt i = ihi + 1; i < nElements + nGhost + 1; i++ )
  {
    Centers( i ) = Centers( i - 1 ) + Widths( i - 1 );
  }

  for ( UInt iC = ilo; iC <= ihi; iC++ )
  {
    for ( UInt iN = 0; iN < nNodes; iN++ )
    {
      Grid( iC, iN ) = NodeCoordinate( iC, iN );
    }
  }
}

/**
 * Compute cell masses
 **/
void GridStructure::ComputeMass( Kokkos::View<Real ***> uPF )
{
  const UInt nNodes = Get_nNodes( );
  const UInt ilo    = Get_ilo( );
  const UInt ihi    = Get_ihi( );

  Real mass;
  Real X;

  for ( UInt iX = ilo; iX <= ihi; iX++ )
  {
    mass = 0.0;
    for ( UInt iN = 0; iN < nNodes; iN++ )
    {
      X = NodeCoordinate( iX, iN );
      mass += Weights( iN ) * Get_SqrtGm( X ) * uPF( 0, iX, iN );
    }
    mass *= Widths( iX );
    Mass( iX ) = mass;
  }

  // Guard cells
  for ( UInt iX = 0; iX < ilo; iX++ )
  {
    Mass( ilo - 1 - iX ) = Mass( ilo + iX );
    Mass( ihi + 1 + iX ) = Mass( ihi - iX );
  }
}

/**
 * Compute cell centers of masses reference coordinates
 **/
void GridStructure::ComputeCenterOfMass( Kokkos::View<Real ***> uPF )
{
  const UInt nNodes = Get_nNodes( );
  const UInt ilo    = Get_ilo( );
  const UInt ihi    = Get_ihi( );

  Real com;
  Real X;

  for ( UInt iX = ilo; iX <= ihi; iX++ )
  {
    com = 0.0;
    for ( UInt iN = 0; iN < nNodes; iN++ )
    {
      X = NodeCoordinate( iX, iN );
      com += Nodes( iN ) * Weights( iN ) * Get_SqrtGm( X ) * uPF( 0, iX, iN );
    }
    com *= Widths( iX );
    CenterOfMass( iX ) = com / Mass( iX );
  }

  // Guard cells
  for ( UInt iX = 0; iX < ilo; iX++ )
  {
    CenterOfMass( ilo - 1 - iX ) = CenterOfMass( ilo + iX );
    CenterOfMass( ihi + 1 + iX ) = CenterOfMass( ihi - iX );
  }
}

/**
 * Update grid coordinates using interface velocities.
 **/
void GridStructure::UpdateGrid( Kokkos::View<Real *> SData )
{

  const UInt ilo = Get_ilo( );
  const UInt ihi = Get_ihi( );

  Kokkos::parallel_for(
      "Grid Update 1", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
      KOKKOS_LAMBDA( UInt iX ) {
        X_L( iX )     = SData( iX );
        Widths( iX )  = SData( iX + 1 ) - SData( iX );
        //std::printf("%d %f %f\n", iX, SData(iX), SData(iX+1));
        Centers( iX ) = 0.5 * ( SData( iX + 1 ) + SData( iX ) );
      } );

  Kokkos::parallel_for(
      "Grid Update 2", Kokkos::RangePolicy<>( ilo, ihi + 1 ),
      KOKKOS_LAMBDA( UInt iX ) {
        for ( UInt iN = 0; iN < nNodes; iN++ )
        {
          Grid( iX, iN ) = NodeCoordinate( iX, iN );
        }
      } );
}

// Access by (element, node)
Real &GridStructure::operator( )( UInt i, UInt j ) { return Grid( i, j ); }

Real GridStructure::operator( )( UInt i, UInt j ) const { return Grid( i, j ); }
