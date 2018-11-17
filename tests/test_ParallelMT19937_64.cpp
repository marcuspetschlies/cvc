
#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "ParallelMT19937_64.hpp"
#include "enums.hpp"
#include "index_tools.hpp"
#include "Stopwatch.hpp"
#include "Core.hpp"
#include "debug_printf.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <vector>

using namespace cvc;

int main(int argc, char** argv)
{
  Core core(argc,argv);

#ifdef HAVE_MPI
  Stopwatch sw(g_cart_grid);
#else
  Stopwatch sw(0);
#endif

  debug_printf(0, 0, "\n\n############ TESTING ParallelMT19937_64 ###############\n\n");

  sw.reset();
  ParallelMT19937_64 rangen(982932ULL);
  sw.elapsed_print("ParallelMT19937_64 initialisation");
  
  std::vector<double> testvec(24*VOLUME);
  sw.reset();
  // generate 24*VOLUME random numbers 50 times
  for(int irun = 0; irun < 50; irun++){
    rangen.gen_real(testvec.data(), 24);
    std::cout << testvec[989] << std::endl;
  }
  sw.elapsed_print("ParallelMT19937_64 test generation");

  return 0;
}
