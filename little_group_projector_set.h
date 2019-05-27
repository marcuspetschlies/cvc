#ifndef _LITTLE_GROUP_PROJECTOR_SET_H
#define _LITTLE_GROUP_PROJECTOR_SET_H

namespace cvc {

/********************************************************/
/********************************************************/

int little_group_projector_set (
  /***********************************************************/
  little_group_projector_type * const p,
  little_group_type * const lg,
  const char * irrep , const int row_target, const int interpolator_num,
  const int * interpolator_J2_list,
  const int ** interpolator_momentum_list,
  const int * interpolator_bispinor_list,
  const int * interpolator_parity_list,
  const int * interpolator_cartesian_list,
  const int ref_row_target,
  const int * ref_row_spin,
  const char * name,
  int const refframerot
  /***********************************************************/
);

}  /* end of namespace cvc */

#endif
