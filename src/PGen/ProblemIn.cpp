/**
 * File     :  ProblemIn.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the problem initialization
 *
 **/

#include "ProblemIn.hpp"
#include "Error.hpp"
#include "SimpleIni.h"

ProblemIn::ProblemIn( std::string fn ) {
  // Load ini
  CSimpleIniA ini;
  SI_Error rc = ini.LoadFile( fn.c_str( ) );

  // Check if everything is happy
  if ( rc < 0 ) {
    throw Error( "Problem reading input deck\n" );
  }
  assert( rc == SI_OK );

  // Grab as char*, cast to respective types
  const char *nN;
  const char *nX;
  const char *nG;
  const char *pO;
  const char *tO;
  const char *nS;
  const char *rest;
  const char *rad;
  const char *geom;
  const char *pn;
  const char *tf;
  const char *x1;
  const char *x2;
  const char *bc;
  const char *cfl;
  const char *al;
  const char *slt;
  const char *tci_opt;
  const char *tci_val;
  const char *characteristic;
  const char *basis;

  pn   = ini.GetValue( "Problem", "problem" );
  rest = ini.GetValue( "Problem", "restart" );
  rad  = ini.GetValue( "Problem", "do_rad" );
  geom = ini.GetValue( "Problem", "geometry" );
  tf   = ini.GetValue( "Problem", "t_end" );
  x1   = ini.GetValue( "Problem", "xL" );
  x2   = ini.GetValue( "Problem", "xR" );
  bc   = ini.GetValue( "Problem", "BC" );
  cfl  = ini.GetValue( "Problem", "CFL" );

  basis = ini.GetValue( "Fluid", "Basis" );
  nN    = ini.GetValue( "Fluid", "nNodes" );
  nX    = ini.GetValue( "Fluid", "nX" );
  nG    = ini.GetValue( "Fluid", "nG" );
  pO    = ini.GetValue( "Fluid", "pOrder" );

  tO = ini.GetValue( "Time", "tOrder" );
  nS = ini.GetValue( "Time", "nStages" );

  al             = ini.GetValue( "Limiters", "alpha" );
  slt            = ini.GetValue( "Limiters", "threshold" );
  tci_opt        = ini.GetValue( "Limiters", "tci_opt" );
  tci_val        = ini.GetValue( "Limiters", "tci_val" );
  characteristic = ini.GetValue( "Limiters", "characteristic" );

  ProblemName = pn;
  BC          = bc;
  Restart     = ( strcmp( rest, "true" ) == 0 ) ? true : false;
  do_rad      = ( strcmp( rad, "true" ) == 0 ) ? true : false;
  Geometry    = ( strcmp( geom, "spherical" ) == 0 ) ? geometry::Spherical
                                                     : geometry::Planar;
  Basis       = ( strcmp( basis, "Legendre" ) == 0 ) ? PolyBasis::Legendre
                                                     : PolyBasis::Taylor;
  xL          = std::atof( x1 );
  xR          = std::atof( x2 );
  t_end       = std::atof( tf );
  CFL         = std::atof( cfl );

  nNodes    = std::atoi( nN );
  nElements = std::atoi( nX );
  nGhost    = std::atoi( nG );
  pOrder    = std::atoi( pO );
  tOrder    = std::atoi( tO );
  nStages   = std::atoi( nS );

  alpha          = std::atof( al );
  SL_Threshold   = std::atof( slt );
  TCI_Option     = ( strcmp( tci_opt, "true" ) == 0 ) ? true : false;
  TCI_Threshold  = std::atof( tci_val );
  Characteristic = ( strcmp( characteristic, "true" ) == 0 ) ? true : false;
}
