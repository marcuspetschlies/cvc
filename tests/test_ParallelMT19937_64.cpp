
#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "ParallelMT19937_64.hpp"
#include "enums.hpp"
#include "index_tools.hpp"
#include "Stopwatch.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <vector>

using namespace cvc;

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  // mockup that we're running in parallel
  T_global = 128;
  LX_global = 32;
  LY_global = 32;
  LZ_global = 32;
  
  // local lattice will be 32*16^3 which should be a nice timing test
  g_nproc_t = 4;
  g_nproc_x = 2;
  g_nproc_y = 2;
  g_nproc_z = 2;

  T = T_global / g_nproc_t;
  LX = LX_global / g_nproc_x;
  LY = LY_global / g_nproc_y;
  LZ = LZ_global / g_nproc_z;
  
  VOLUME = LX*LY*LZ*T;

  g_proc_coords[DIM_T] = 3;
  g_proc_coords[DIM_X] = 0;
  g_proc_coords[DIM_Y] = 2;
  g_proc_coords[DIM_Z] = 1;

  g_nproc = g_nproc_x*g_nproc_y*g_nproc_z*g_nproc_t; 

  std::cout << local_to_global_site_index(0) << std::endl;

#ifdef HAVE_MPI
  Stopwatch sw(MPI_COMM_WORLD);
#else
  Stopwatch sw(0);
#endif

  sw.reset();
  ParallelMT19937_64 rangen(982932ULL);
  sw.elapsed_print_and_reset("ParallelMT19937_64 initialisation");

  std::vector<double> testvec(8*VOLUME);
  // generate 24*VOLUME random numbers 50 times
  for(int irun = 0; irun < 50; irun++){
    rangen.gen_real(testvec.data(), 24);
    std::cout << testvec[989] << std::endl;
  }
  sw.elapsed_print("ParallelMT19937_64 test generation");

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
