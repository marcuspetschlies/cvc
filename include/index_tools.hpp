#ifndef INDEX_TOOLS_HPP
#define INDEX_TOOLS_HPP

#include "global.h"
#include "enums.hpp"

#include <array>

namespace cvc {

  /**
   * @brief Convert a local to a global site index
   *
   * @param local_site_index
   *
   * @return global site index
   */
static inline
unsigned long long local_to_global_site_index(const int local_site_index)
{
  return( (unsigned long long)local_site_index +
          VOLUME*(g_proc_coords[DIM_Z] +
                  g_nproc_z*g_proc_coords[DIM_Y] +
                  g_nproc_y*g_nproc_z*g_proc_coords[DIM_X] +
                  g_nproc_x*g_nproc_y*g_nproc_z*g_proc_coords[DIM_T]) );
}

static inline void global_site_get_proc_coords(unsigned long long global_site_index,
                                               std::array<int,4> & proc_coords)
{
  unsigned long long idx = global_site_index / VOLUME;
  proc_coords[DIM_Z] = static_cast<int>( idx % g_nproc_x );
  idx /= g_nproc_z;
  proc_coords[DIM_Y] = static_cast<int>( idx % g_nproc_y );
  idx /= g_nproc_y;
  proc_coords[DIM_X] = static_cast<int>( idx % g_nproc_x );
  idx /= g_nproc_x;
  proc_coords[DIM_T] = static_cast<int>( idx % g_nproc_t );
}

/**
 * @brief check if global site index is within the local lattice
 *
 * @param global_site_index
 *
 * @return
 */
static inline bool is_local(unsigned long long global_site_index)
{
  std::array<int,4> proc_coords;
  global_site_get_proc_coords(global_site_index, proc_coords);

  return( (proc_coords[DIM_Z]  == g_proc_coords[DIM_Z]) &&
          (proc_coords[DIM_Y]  == g_proc_coords[DIM_Y]) &&
          (proc_coords[DIM_X]  == g_proc_coords[DIM_X]) &&
          (proc_coords[DIM_T]  == g_proc_coords[DIM_T]) 
        );
}

static inline int global_site_get_rank(unsigned long long global_site_index){
  std::array<int,4> proc_coords;
  global_site_get_proc_coords(global_site_index, proc_coords);
  int rank;
#ifdef HAVE_MPI
  MPI_Cart_rank(g_cart_grid, proc_coords.data(), &rank);
  return(rank);
#else
  return(0);
#endif
}

} // namespace(cvc)

#endif // INDEX_TOOLS_HPP
