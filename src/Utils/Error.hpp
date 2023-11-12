/**
 * File    :  Error.hpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Print error messages ...
 **/

#ifndef ERROR_H
#define ERROR_H

#include <stdexcept>
#include <assert.h>     /* assert */

class Error : public std::runtime_error
{

 public:
  Error( const std::string &message ) : std::runtime_error( message ) {}
};

#endif
