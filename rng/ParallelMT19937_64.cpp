#include "ParallelMT19937_64.hpp"
#include "index_tools.hpp"
#include "global.h"

#include <iostream>

namespace cvc {
  ParallelMT19937_64::ParallelMT19937_64(const unsigned long long seed) :
    seed_gen(seed)
  {
    unsigned long long global_volume = VOLUME*g_nproc;
    unsigned long long ran64;
    local_seeds.resize( VOLUME );
    local_rngs.reserve( VOLUME );

    for(unsigned long long gi = 0; gi < global_volume; ++gi){
      ran64 = seed_gen.gen_int64();
      if( is_local(gi) ){
        local_seeds[ gi % VOLUME ] = ran64;
      }
    }
    for(unsigned long long li = 0; li < VOLUME; ++li){
      local_rngs.emplace_back( MT19937_64(local_seeds[li]) );
    }
  }

  void
  ParallelMT19937_64::gen_real(double * buffer, const unsigned int n_per_site)
  {
#pragma omp parallel for
    for(unsigned int li = 0; li < VOLUME; ++li){
      for(int i_per_site = 0; i_per_site < n_per_site; ++i_per_site){
        buffer[n_per_site*li + i_per_site] = local_rngs[li].gen_real();
      }
    }
  } 

}
