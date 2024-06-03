/**
 * File    :  error.hpp
 * --------------
 *
 * Author  : Brandon L. Barker
 * Purpose : Print error messages ...
 **/

#ifndef ERROR_HPP_
#define ERROR_HPP_

#include <assert.h> /* assert */
#include <exception>
#include <stdexcept>

class Error : public std::runtime_error {

 public:
  Error( const std::string &message ) : std::runtime_error( message ) {}
};

#endif // ERROR_HPP_
