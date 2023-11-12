/**
 * File     :  IOLibrary.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : HDF5 IO routines
 **/

#include <iostream>
#include <string>
#include <vector>

#include "Con2Prim.hpp"
#include "Grid.hpp"
#include "H5Cpp.h"
#include "PolynomialBasis.hpp"
#include "SlopeLimiter.hpp"
#include "IOLibrary.hpp"

/**
 * Write to standard output some initialization info
 * for the current simulation.
 **/
void PrintSimulationParameters( GridStructure Grid, ProblemIn *pin, const Real CFL )
{
  const UInt nX     = Grid.Get_nElements( );
  const UInt nNodes = Grid.Get_nNodes( );

  std::printf( " ~ --- Order Parameters --- \n" );
  std::printf( " ~ Basis          : %d ( 0 : Legendre, 1: Taylor )\n", pin->Basis );
  std::printf( " ~ Spatial Order  : %d\n", pin->pOrder );
  std::printf( " ~ Temporal Order : %d\n", pin->tOrder );
  std::printf( " ~ RK Stages      : %d\n", pin->nStages );
  std::printf( "\n" );

  std::printf( " ~ --- Grid Parameters --- \n" );
  std::printf( " ~ Mesh Elements  : %d\n", nX );
  std::printf( " ~ Number Nodes   : %d\n", nNodes );
  std::printf( " ~ Lower Boudnary : %f\n", Grid.Get_xL( ) );
  std::printf( " ~ Upper Boudnary : %f\n", Grid.Get_xR( ) );
  std::printf( "\n" );

  std::printf( " ~ --- Limiter Parameters --- \n" );
  if ( pin->pOrder == 1 )
  {
    printf( " ~ Spatial Order 1: Slope limiter not applied.\n" );
  }
  else
  {
    std::printf( " ~ Alpha          : %f\n", pin->alpha );
  }
  if ( pin->TCI_Option )
  {
    std::printf( " ~ TCI Value      : %f\n", pin->TCI_Threshold );
  }
  else
  {
    std::printf( " ~ TCI Not Used.\n" );
  }
  if ( pin->Characteristic )
  {
    std::printf( " ~ Limiting       : Characteristic \n" );
  }
  else
  {
    std::printf( " ~ Limiting       : Componentwise\n" );
  }
  std::printf( "\n" );

  std::printf( " ~ --- Other --- \n" );
  std::cout << " ~ ProblemName    : " << pin->ProblemName << std::endl;
  std::printf( " ~ CFL            : %f\n", CFL );
  std::printf( "\n" );
}

// TODO: add Time
void WriteState( Kokkos::View<Real ***> uCF, Kokkos::View<Real ***> uPF,
                 GridStructure Grid, SlopeLimiter *SL, 
                 const std::string ProblemName, Real time,
                 UInt order, int i_write )
{

  std::string fn = "athelas_";
  auto i_str    = std::to_string( i_write );
  int n_pad;
  if ( i_write < 10 ) {
    n_pad = 4;
  } else if ( i_write >= 10 && i_write < 100 ) { 
    n_pad = 3;
  } else if ( i_write >= 100 && i_write < 1000 ) { 
    n_pad = 2;
  } else if ( i_write >= 1000 && i_write < 10000 ) { 
    n_pad = 1;
  } else {
    n_pad = 0;
  }
  std::string suffix = std::string( n_pad, '0' ).append( i_str );
  fn.append( ProblemName );
  fn.append( "_" );
  if ( i_write != -1 )
  {
    fn.append( suffix );
  }
  else
  {
    fn.append( "final" );
  }
  fn.append( ".h5" );

  // conversion to make HDF5 happy
  const char *fn2 = fn.c_str( );

  const UInt nX  = Grid.Get_nElements( );
  const UInt ilo = Grid.Get_ilo( );
  const UInt ihi = Grid.Get_ihi( );

  const H5std_string FILE_NAME( fn );
  const H5std_string DATASET_NAME( "Grid" );
  const int size = ( nX * order ); // dataset dimensions

  std::vector<DataType> tau( size ); // conserved
  std::vector<DataType> vel( size );
  std::vector<DataType> eint( size );
  std::vector<DataType> Ptau( size ); // primitives
  std::vector<DataType> Pvel( size );
  std::vector<DataType> Peint( size );
  std::vector<DataType> grid( nX );
  std::vector<DataType> dr( nX );
  std::vector<DataType> limiter( nX );

  Real trash = 0.0;
  for ( UInt k = 0; k < order; k++ )
    for ( UInt iX = ilo; iX <= ihi; iX++ )
    {
      grid[( iX - ilo )].x          = Grid.Get_Centers( iX );
      dr[( iX - ilo )].x            = Grid.Get_Widths( iX );
      //std::printf("dr %f\n", Grid.Get_Widths(iX));
      limiter[( iX - ilo )].x       = SL->Get_Limited( iX );
      tau[( iX - ilo ) + k * nX].x  = uCF( 0, iX, k );
      vel[( iX - ilo ) + k * nX].x  = uCF( 1, iX, k );
      eint[( iX - ilo ) + k * nX].x = uCF( 2, iX, k );
      Con2Prim( uCF( 0, iX, 0 ), uCF( 1, iX, 0 ), uCF( 2, iX, 0 ), 
          Ptau[( iX - ilo ) + k * nX].x, Pvel[( iX - ilo ) + k * nX].x, 
          Peint[( iX - ilo ) + k * nX].x, trash );
    }

  // preparation of a dataset and a file.
  hsize_t dim[1];
  dim[0]         = tau.size( ); // using vector::size()
  const int rank = sizeof( dim ) / sizeof( hsize_t );
  H5::DataSpace space( rank, dim );

  hsize_t dim_grid[1];
  dim_grid[0]         = grid.size( ); // using vector::size()
  const int rank_grid = sizeof( dim_grid ) / sizeof( hsize_t );
  H5::DataSpace space_grid( rank_grid, dim_grid );

  hsize_t len       = 1;
  hsize_t dim_md[1] = { len };
  const int rank_md = 1;
  H5::DataSpace md_space( rank_md, dim_md );

  H5::H5File file( fn2, H5F_ACC_TRUNC );
  // Groups
  H5::Group group_md   = file.createGroup( "/Metadata" );
  H5::Group group_grid = file.createGroup( "/Spatial Grid" );
  H5::Group group_CF   = file.createGroup( "/Conserved Fields" );
  H5::Group group_PF   = file.createGroup( "/Primitive Fields" );
  H5::Group group_DF   = file.createGroup( "/Diagnostic Fields" );

  // DataSets
  H5::DataSet dataset_nx( file.createDataSet(
      "/Metadata/nX", H5::PredType::NATIVE_INT, md_space ) );
  H5::DataSet dataset_order( file.createDataSet(
      "/Metadata/Order", H5::PredType::NATIVE_INT, md_space ) );
  H5::DataSet dataset_time( file.createDataSet(
      "/Metadata/Time", H5::PredType::NATIVE_DOUBLE, md_space ) );
  H5::DataSet dataset_grid( file.createDataSet(
      "/Spatial Grid/Grid", H5::PredType::NATIVE_DOUBLE, space_grid ) );
  H5::DataSet dataset_width( file.createDataSet(
      "/Spatial Grid/Widths", H5::PredType::NATIVE_DOUBLE, space_grid ) );
  H5::DataSet dataset_Ctau(
      file.createDataSet( "/Conserved Fields/Specific Volume",
                          H5::PredType::NATIVE_DOUBLE, space ) );
  H5::DataSet dataset_Cvel( file.createDataSet(
      "/Conserved Fields/Velocity", H5::PredType::NATIVE_DOUBLE, space ) );
  H5::DataSet dataset_Ceint(
      file.createDataSet( "/Conserved Fields/Specific Internal Energy",
                          H5::PredType::NATIVE_DOUBLE, space ) );

  // --- Primitives --- 
  H5::DataSet dataset_Ptau(
      file.createDataSet( "/Primitive Fields/Specific Volume",
                          H5::PredType::NATIVE_DOUBLE, space ) );
  H5::DataSet dataset_Pvel( file.createDataSet(
      "/Primitive Fields/Velocity", H5::PredType::NATIVE_DOUBLE, space ) );
  H5::DataSet dataset_Peint(
      file.createDataSet( "/Primitive Fields/Specific Internal Energy",
                          H5::PredType::NATIVE_DOUBLE, space ) );

  H5::DataSet dataset_limiter( file.createDataSet(
      "/Diagnostic Fields/Limiter", H5::PredType::NATIVE_DOUBLE, space_grid ) );

  // --- Write data ---
  dataset_nx.write( &nX, H5::PredType::NATIVE_INT );
  dataset_order.write( &order, H5::PredType::NATIVE_INT );
  dataset_time.write( &time, H5::PredType::NATIVE_DOUBLE );

  dataset_grid.write( grid.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_width.write( dr.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_limiter.write( limiter.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_Ctau.write( tau.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_Cvel.write( vel.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_Ceint.write( eint.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_Ptau.write( Ptau.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_Pvel.write( Pvel.data( ), H5::PredType::NATIVE_DOUBLE );
  dataset_Peint.write( Peint.data( ), H5::PredType::NATIVE_DOUBLE );
}

/**
 * Write Modal Basis coefficients and mass matrix
 **/
void WriteBasis( ModalBasis *Basis, UInt ilo, UInt ihi, UInt nNodes, UInt order,
                 std::string ProblemName )
{
  std::string fn = "athelas_basis_";
  fn.append( ProblemName );
  fn.append( ".h5" );

  const char *fn2 = fn.c_str( );

  const H5std_string FILE_NAME( fn );
  const H5std_string DATASET_NAME( "Basis" );

  Real *data = new Real[ihi * ( nNodes + 2 ) * order];
  for ( UInt iX = ilo; iX <= ihi; iX++ )
    for ( UInt iN = 0; iN < nNodes + 2; iN++ )
      for ( UInt k = 0; k < order; k++ )
      {
        data[( ( iX - ilo ) * ( nNodes + 2 ) + iN ) * order + k] =
            Basis->Get_Phi( iX, iN, k );
      }

  // Create HDF5 file and dataset
  H5::H5File file( fn2, H5F_ACC_TRUNC );
  hsize_t dimsf[3] = { ihi, nNodes + 2, order };
  H5::DataSpace dataspace( 3, dimsf );
  H5::DataSet BasisDataset(
      file.createDataSet( "Basis", H5::PredType::NATIVE_DOUBLE, dataspace ) );
  // Write to File
  BasisDataset.write( data, H5::PredType::NATIVE_DOUBLE );

  delete[] data;
}
