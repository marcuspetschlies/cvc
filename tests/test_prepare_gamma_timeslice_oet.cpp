#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "Core.hpp"
#include "ParallelMT19937_64.hpp"
#include "prepare_source.h"
#include "debug_printf.hpp"

#include <vector>
#include <iostream>
#include <unistd.h>

using namespace cvc;

int main(int argc, char ** argv){
  Core core(argc, argv);

  ParallelMT19937_64 rng(989127212ULL);

  std::vector<double> ranspinor(_GSI(VOLUME));
  std::vector<double> g5_source(_GSI(VOLUME));

  const size_t vol3 = LX * LY * LZ;
  const unsigned int t_src = 14;
  const unsigned int t_src_local = t_src % T;
  const unsigned int gamma_id = 5;
  const bool have_source = (t_src / T) == g_proc_coords[0];
  const int momentum[3] = { -2, 1, 4 };

  sleep(1);

  debug_printf(0, 0, "\n### Placing a source on time slice %u with gamma structure %u\n", t_src, gamma_id);
  debug_printf(0, 0, "### Source momentum will be px: %d py: %d  pz: %d\n\n", momentum[0], momentum[1], momentum[2]);
#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

  // two test indices outside of the source time slice
  // two test indices inside the source time slice, one with + from gamma_5, the other with -
  const size_t test_idcs[4] = { 10, 20, t_src_local*_GSI(vol3)+24*256+10, t_src_local*_GSI(vol3)+24*256+20 };  

  rng.gen_test(ranspinor.data(), 24);

  // first without momentum

  debug_printf(0, 0, "\nTesting without momentum and with the local lattice index in the field first\n");
#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif
  prepare_gamma_timeslice_oet(g5_source.data(), 
                              ranspinor.data(),
                              gamma_id,
                              t_src,
                              NULL);

  for( auto const & idx : test_idcs ){
    // expect the local site index with a sign from the gamma5
    double expected = ((int)have_source) * ((idx/12 % 2 == 0) ? 1 : -1) * (double)(idx/24);
    std::cout << "MPI task " << g_proc_id << " Source at index " << 
      idx << ":       " << g5_source[idx] << " Should be: " << expected << std::endl;
    std::cout << "MPI task " << g_proc_id << " Random field at index " << idx 
      << ": " << ranspinor[idx] << std::endl << std::endl; 
  }
  std::cout << std::endl;
  fflush(stdout);

#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif
  // now we test with z2 noise and some momentum
  debug_printf(0, 0, "\nTesting with momentum and z2 noise\n");
#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif
  rng.gen_z2(ranspinor.data(), 24);
  prepare_gamma_timeslice_oet(g5_source.data(),
                              ranspinor.data(),
                              gamma_id,
                              t_src,
                              momentum);
  for( auto const & idx : test_idcs ){
    // expect the local site index with a sign from the gamma5
    double expected = ((int)have_source) * ((idx/12 % 2 == 0) ? 1 : -1) *  (double)(idx/24);
    std::cout << "MPI task " << g_proc_id << " Source at index " << idx 
      << ":       " << g5_source[idx] << std::endl;
    std::cout << "MPI task " << g_proc_id << " Random field at index " << idx 
      << ": " << ranspinor[idx] << std::endl << std::endl; 
  }
  std::cout << std::endl << std::endl;


  return(0);
}
