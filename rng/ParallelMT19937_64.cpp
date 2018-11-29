#include "index_tools.hpp"
#include "global.h"
#include "ParallelMT19937_64.hpp"
#include "debug_printf.hpp"
#include "SequenceOfUnique.hpp"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include "loop_tools.h"

#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cmath>

namespace cvc {
  // 311ULL is just an offset, we could use any other
  // It's the seed which decides where in the sequence we start
  ParallelMT19937_64::ParallelMT19937_64(const unsigned long long seed) :
    seed_gen(seed, 311ULL)
  { 
    const unsigned long long local_volume = VOLUME;
    const unsigned long long global_volume = VOLUME*g_nproc;
    local_seeds.resize( VOLUME );
    local_rngs.resize( VOLUME );

    for(unsigned long long gi = 0; gi < global_volume; ++gi){
      // use the sequence generator to produce our seeds for the global lattice
      unsigned long long ran64 = seed_gen.next();
      if( is_local(gi) ){
        local_seeds[ gi % VOLUME ] = ran64;
      }
    }
    #pragma omp parallel for
    for(unsigned int li = 0; li < VOLUME; ++li){
      local_rngs[li].init(local_seeds[li]);
    }
  }

  void
  ParallelMT19937_64::gen_z2(double * const buffer, const unsigned int n_per_site)
  {
    const double one_ov_sqrt_2 = 1.0/sqrt(2.0);
    PARALLEL_AND_FOR(li, 0, VOLUME)
    {
      for(unsigned int i_per_site = 0; i_per_site < n_per_site; ++i_per_site){
        buffer[n_per_site*li + i_per_site] = local_rngs[li].gen_real() >= 0.5 ?
          one_ov_sqrt_2 : -one_ov_sqrt_2;
      }
    }
  }

  void
  ParallelMT19937_64::gen_test(double * const buffer, const unsigned int n_per_site)
  {
    PARALLEL_AND_FOR(li, 0, VOLUME)
    {
      for(unsigned int i_per_site = 0; i_per_site < n_per_site; ++i_per_site){
        buffer[n_per_site*li + i_per_site] = (double)li;
      }
    }
  }

  void
  ParallelMT19937_64::gen_real(double * const buffer, const unsigned int n_per_site) 
  {
#pragma omp parallel for
    for(unsigned int li = 0; li < VOLUME; ++li){
      for(unsigned int i_per_site = 0; i_per_site < n_per_site; ++i_per_site){
        buffer[n_per_site*li + i_per_site] = local_rngs[li].gen_real();
      }
    }
  }

  double
  ParallelMT19937_64::gen_real_at_site(size_t local_site_idx){
    return(local_rngs[local_site_idx].gen_real());
  }

} // namespace(cvc)
