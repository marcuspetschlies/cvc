#pragma once

#include <vector>
#include <exception>
#include <stdexcept>

/**
 * @brief Comparison functor for momentum triplets, suitable for std::sort
 *
 * we sort momenta by psq first.
 * For equal psq, we sort by maximum component: (2,2,1) < (0,0,3)
 * and then by the number of non-zeros (not sure if this last condition would ever occur)
 */
typedef struct momentum_compare_t {
  bool operator()( const std::vector<int> & a, const std::vector<int> & b ){
    if( a.size() != 3 || b.size() != 3 ){
      throw( std::invalid_argument("in momentum_sort, 'a' and 'b' must of size 3!") );
    }
    int asq = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
    int anonzero = static_cast<int>( a[0] != 0 ) + static_cast<int>( a[1] != 0 ) + static_cast<int>( a[2] != 0 );
    int amax = 0;
    for( const auto & p : a ){
      amax = abs(p) > amax ? abs(p) : amax;
    }

    int bsq = b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
    int bnonzero = static_cast<int>( b[0] != 0 ) + static_cast<int>( b[1] != 0 ) + static_cast<int>( b[2] != 0 );
    int bmax = 0;
    for( const auto & p : b ){
      bmax = abs(p) > bmax ? abs(p) : bmax;
    }

    if( asq == bsq ){
      if( amax == bmax ){
        return anonzero < bnonzero;
      } else {
        return amax < bmax;
      }
    } else {
      return asq < bsq;
    }
  }
} momentum_compare_t;

