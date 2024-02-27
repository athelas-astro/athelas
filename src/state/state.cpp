/**
 * File     :  state.cpp
 * --------------
 *
 * Author   : Brandon L. Barker
 * Purpose  : Class for holding the state data.
 *
 * TODO: separate pOrder_fluid and pOrder_rad
 **/

#include "state.hpp"
#include "Constants.hpp"

State::State( const int nCF_, const int nCR_, const int nPF_, const int nAF_,
              const int nX_, const int nG_, const int nNodes_,
              const int pOrder_ )
    : nCF( nCF_ ), nCR( nCR_ ), nPF( nPF_ ), nAF( nAF_ ), pOrder( pOrder_ ),
      uCF( "uCF", nCF_, nX_ + 2 * nG_, pOrder_ ),
      uPF( "uPF", nPF_, nX_ + 2 * nG_, nNodes_ ),
      uAF( "uAF", nAF_, nX_ + 2 * nG_, nNodes_ ),
      uCR( "uCR", nCR_, nX_ + 2 * nG_, pOrder_ ) {}

// num var accessors
int State::Get_nCF( ) const { return this->nCF; }
int State::Get_nPF( ) const { return this->nPF; }
int State::Get_nAF( ) const { return this->nAF; }
int State::Get_nCR( ) const { return this->nCR; }
int State::Get_pOrder( ) const { return this->pOrder; }

// view accessors
View3D State::Get_uCF( ) const { return this->uCF; }
View3D State::Get_uPF( ) const { return this->uPF; }
View3D State::Get_uAF( ) const { return this->uAF; }
View3D State::Get_uCR( ) const { return this->uCR; }
