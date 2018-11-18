
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

  unsigned long long gi = 227823;

  std::cout << "Global index " << gi
    << " on rank " << global_site_get_rank(gi) << std::endl;

  return 0;
}
