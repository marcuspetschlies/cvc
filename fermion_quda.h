#ifndef _FERMION_QUDA_H
#define _FERMION_QUDA_H

#ifdef _GFLOW_QUDA
#warning "including quda header file quda.h directly "
#include "quda.h"

namespace cvc {

void init_invert_param ( QudaInvertParam * const inv_param );

void spinor_field_cvc_to_quda ( double * const r, double * const s );

}
#endif  /* _GFLOW_QUDA */
#endif
