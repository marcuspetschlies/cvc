
#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "cvc_utils.h"
#include "Q_phi.h"
#include "Stopwatch.hpp"
#include "Core.hpp"
#include "ParallelMT19937_64.hpp"
#include "enums.hpp"
#include "init_g_gauge_field.hpp"
#include "debug_printf.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <vector>
#include <cstring>

using namespace cvc;

int main(int argc, char** argv)
{
  Core core(argc,argv);
  if( !(core.is_initialised()) ){
    std::cout << "Core initialisation failed!\n";
    return(CVC_EXIT_CORE_INIT_FAILURE);
  }

  int exitstatus = 0; 

#ifdef HAVE_MPI
  Stopwatch sw(g_cart_grid);
#else
  Stopwatch sw(0);
#endif
 
  // initialize and load the gauge field 
  CHECK_EXITSTATUS_NONZERO(
      exitstatus,
      init_g_gauge_field(),
      "[cpff_invert_contract] Error initialising gauge field!",
      true, 
      CVC_EXIT_GAUGE_INIT_FAILURE);

  const unsigned long long global_volume = g_nproc*VOLUME;

  // 8 in
  std::vector<std::vector<double>> spinor1(8);
  // 64 out
  std::vector<std::vector<double>> spinor2(64);

  for( auto & elem : spinor1 ) elem.resize(24*(VOLUME+RAND));
  for( auto & elem : spinor2 ) elem.resize(24*(VOLUME+RAND));

  ParallelMT19937_64 rng(897872134ULL);

  // fill all spinor1 spinors with random numbers
  for( auto & elem : spinor1 ) rng.gen_z2(elem.data(),24);

  const unsigned int iterations = 10;
  unsigned int appcounter = 0;
  sw.reset();
  for(unsigned long long i = 0; i < iterations; ++i){
    debug_printf(0,0, "Iteration %d\n", i);
    // simulate computation of all double shifts (generally not all are required)
    // in order to increase gauge field re-use, we have mu1 and fwdbwd1 in the
    // outer loop, such that 16 cov_shifts are computed
    // with a single gauge field instead of switching all the time
    for(unsigned int mu1 : {0,1,2,3} ){
      for(unsigned int fwdbwd1 : {0,1} ){
        for(unsigned int mu2 : {0,1,2,3} ){
          for(unsigned int fwdbwd2 : {0,1} ){

            const unsigned int idx2 = 2*2*4*mu1 +
                                      2*4*fwdbwd1 +
                                      2*mu2 +
                                      + fwdbwd2; 
            const unsigned int s_idx = 2*mu2 + fwdbwd2;

            spinor_field_eq_cov_displ_spinor_field( spinor2[idx2].data(),
                                                    spinor1[s_idx].data(),
                                                    mu1,
                                                    fwdbwd1,
                                                    g_gauge_field);

          }
        }
      }
    }
  }
  char msg[400];
  snprintf(msg, 400, "%d x %d applications of spinor_field_eq_cov_displ_spinor_field", iterations, 8*8);
  duration bench_dur = sw.elapsed_print(msg);

  debug_printf(
    0,0,
    "At 312 flops per application, this is %lf GFlop/s\n", 
    ((double)312.0)*g_nproc*VOLUME*iterations*8*8*1.0e-9/bench_dur.mean 
  );

  return 0;
}
