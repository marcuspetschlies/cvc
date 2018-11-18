#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "SequenceOfUnique.hpp"
#include "Stopwatch.hpp"
#include "Core.hpp"
#include "enums.hpp"

#include <unordered_set>
#include <iostream>
#include <limits.h>

using namespace cvc;

int main(int argc, char ** argv){
  Core core(argc, argv);
  if( !(core.is_initialised()) ){
    std::cout << "Core initialisation failed!\n";
    return(CVC_EXIT_CORE_INIT_FAILURE);
  } 

  std::cout << "Testing SequenceOfUnique" << std::endl;

  std::unordered_set<unsigned long long> uniques;
  SequenceOfUnique sequence(981928, 0);
 
  Stopwatch sw(g_cart_grid);

  unsigned long long curr;
  unsigned long long no_collisions = 0;
  // generate sequence of length 2^28 to see if this works
  for(unsigned int i = 0; i < UINT_MAX/64; ++i){
    curr = sequence.next();
    if( (uniques.emplace(curr)).second == false ){
      std::cout << "Found collision " << curr << std::endl;
      no_collisions++;
    }
    if( i % 100000000 == 0 && i != 0 ){
      std::cout << "Generated 100m numbers, curr=" << curr << std::endl;
      sw.elapsed_print_and_reset("100m numbers");
    }

  }
  std::cout << "Number of collisions: " << no_collisions << std::endl;
  return 0;
}

