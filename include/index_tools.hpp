#ifndef INDEX_TOOLS_HPP
#define INDEX_TOOLS_HPP

#include "global.h"
#include "enums.hpp"

#include <iostream>
#include <unistd.h>

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

/**
 * @brief check if global site index is within the local lattice
 *
 * @param global_site_index
 *
 * @return
 */
static inline bool is_local(unsigned long long global_site_index)
{
  unsigned long long proc_z, proc_y, proc_x, proc_t;
  unsigned long long idx = global_site_index / VOLUME;

  proc_z = idx % g_nproc_z;
  idx /= g_nproc_z;
  proc_y = idx % g_nproc_y;
  idx /= g_nproc_y;
  proc_x = idx % g_nproc_x;
  idx /= g_nproc_x;
  proc_t = idx % g_nproc_t;

  return( (proc_z  == (unsigned long long)g_proc_coords[DIM_Z]) &&
          (proc_y  == (unsigned long long)g_proc_coords[DIM_Y]) &&
          (proc_x  == (unsigned long long)g_proc_coords[DIM_X]) &&
          (proc_t  == (unsigned long long)g_proc_coords[DIM_T]) 
        );
}

} // namespace

#endif // INDEX_TOOLS_HPP
