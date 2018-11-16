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
  ParallelMT19937_64::ParallelMT19937_64(const unsigned long long seed,
                                         const bool quick_init) :
    seed_gen(seed, 311ULL)
  { 
    const unsigned long long local_volume = VOLUME;
    const unsigned long long global_volume = VOLUME*g_nproc;
    local_seeds.resize( VOLUME );
    local_rngs.reserve( VOLUME );

    if( quick_init ){
      // with the simple seed computation we can run multithreaded
      // in my tests I didn't see much gain unfortunately
      #pragma omp parallel for
      for(unsigned long long gi = 0; gi < global_volume; ++gi){
        // we XOR with a large, prime multiple of the global index
        // to get a unique seed
        // 16777213ULL = (2^24)-3 so this should be fine also
        // for very large lattices
        unsigned long long ran64 = seed^(gi*16777213ULL);
        if( is_local(gi) ){
          local_seeds[ gi % VOLUME ] = ran64;
        }
      }
    } else {
      for(unsigned long long gi = 0; gi < global_volume; ++gi){
        // if we have more time, we use the sequence generator to produce a more reliable
        // sequence 
        unsigned long long ran64 = seed_gen.next();
        if( is_local(gi) ){
          local_seeds[ gi % VOLUME ] = ran64;
        }
      }
    }

    for(auto const & elem : local_seeds){
      local_rngs.emplace_back( MT19937_64(elem) );
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

} // namespace(cvc)
