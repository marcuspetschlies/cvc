#include "index_tools.hpp"
#include "global.h"
#include "ParallelMT19937_64.hpp"
#include "debug_printf.hpp"
#include "SequenceOfUnique.hpp"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <unistd.h>
#include <stdio.h>

namespace cvc {
  // 311ULL is just an offset, we could use any other
  ParallelMT19937_64::ParallelMT19937_64(const unsigned long long seed) :
    seed_gen(seed, 311ULL)
  { 
    const unsigned long long local_volume = VOLUME;
    const unsigned long long global_volume = VOLUME*g_nproc;
    local_seeds.resize( VOLUME );
    local_rngs.resize( VOLUME );

    for(unsigned long long gi = 0; gi < global_volume; ++gi){
      // if we have more time, we use the sequence generator to produce a more reliable
      // sequence, the sequence generators need to run over the whole
      // global volume 
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
  ParallelMT19937_64::gen_real(double * buffer, const unsigned int n_per_site)
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
