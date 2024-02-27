#ifndef STATE_HPP_
#define STATE_HPP_

/**
 * File     :  state.hpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the state data.
 *
 * TODO: pull in eos
 *
 **/

#include "Kokkos_Core.hpp"

#include "Abstractions.hpp"
#include "Error.hpp"

class State {
 public:
  State( const int nCF_, const int nCR_, const int nPF_, const int nAF_,
         const int nX_, const int nG_, const int nNodes_, const int pOrder_ );

  int Get_nCF( ) const;
  int Get_nCR( ) const;
  int Get_nPF( ) const;
  int Get_nAF( ) const;
  int Get_pOrder( ) const;

  View3D Get_uCF( ) const;
  View3D Get_uPF( ) const;
  View3D Get_uAF( ) const;
  View3D Get_uCR( ) const;

 private:
  int nCF;
  int nCR;
  int nPF;
  int nAF;
  int pOrder;

  View3D uCF; // Conserved fluid
  View3D uPF; // primitive fluid
  View3D uAF; // auxiliary fluid
  View3D uCR; // conserved radiation
};

#endif // STATE_HPP_
