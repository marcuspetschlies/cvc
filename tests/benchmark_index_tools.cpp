
#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "index_tools.hpp"
#include "Stopwatch.hpp"
#include "Core.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <vector>

using namespace cvc;

int main(int argc, char** argv)
{
  Core core(argc,argv);
  if( !(core.is_initialised()) ){
    std::cout << "Core initialisation failed!\n";
    return(CVC_EXIT_CORE_INIT_FAILURE);
  } 

#ifdef HAVE_MPI
  Stopwatch sw(g_cart_grid);
#else
  Stopwatch sw(0);
#endif

  const unsigned long long global_volume = g_nproc*VOLUME;

  std::vector<unsigned long long> local_seeds(VOLUME);
  
  sw.reset();
  const unsigned long seed = 98123ULL;
#pragma omp parallel for
  for(unsigned long long i = 0; i < global_volume; ++i){
    if( is_local(i) ){
      local_seeds[i % VOLUME] = seed^(123871823ULL*i);
    }
  }
  std::cout << local_seeds[9823] << std::endl;
  sw.elapsed_print("Loop over global volume requesting is_local(i)");

  return 0;
}
