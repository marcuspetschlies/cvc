#ifndef PARALLELMT19937_64_HPP
#define PARALLELMT19937_64_HPP

#include "MT19937_64.hpp"
#include "SequenceOfUnique.hpp"
#include <vector>

namespace cvc {

  /**
   * @brief A parallelized version of MT19937_64
   * The parallel MT19937_64 is initialised in two steps. First, an RNG
   * is seeded with the same seed on all MPI tasks and this produces
   * a sequence of 64-bit unsigned integers as long as the global
   * lattice is large.
   *
   * This sequence is then used to initialise one RNG per lattice
   * site. Because the used seed is linked to the global lattice 
   * index (see index_tools.hpp), this is fully independent of the
   * parallelisation.
   *
   * Also, when random numbers are generated, this can be done
   * with multiple threads.
   */
class ParallelMT19937_64 {
  public:
    ParallelMT19937_64() = delete;
    ParallelMT19937_64(const unsigned long long seed, const bool quick_init = false);

    /**
     * @brief Generate real numbers in the interval [0-1]
     *
     * @param buffer Pointer to target memory. Assumption that this 
     * is allocated and has n_per_site*VOLUME elements.
     * @param n_per_site Number of elements per site.
     */
    void gen_real(double * buffer, const unsigned int n_per_site);


    /**
     * @brief Make sure every lattice point is assigned a unique seed.
     * Resolve duplicate seeds by first ensuring that the local_seeds
     * are all unique and then via iterative pairwise comparisons amongst
     * all MPI tasks until complete uniqueness has been reached.
     *
     */
    void resolve_duplicate_seeds(void);

  private:
    SequenceOfUnique seed_gen;
    std::vector<MT19937_64> local_rngs;
    std::vector<unsigned long long> local_seeds;

}; // class(ParallelMT19937_64)

} // namespace(cvc)

#endif
