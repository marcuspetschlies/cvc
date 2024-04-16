#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#ifdef _GFLOW_QUDA
#warning "including quda header file quda.h directly "
#include "quda.h"

namespace cvc {

int get_gauge_padding ( int x[4] );

void gauge_field_cvc_to_qdp ( double ** g_qdp, double * g_cvc );

void gauge_field_qdp_to_cvc ( double * g_cvc, double ** g_qdp );

void init_gauge_param (QudaGaugeParam * const gauge_param );

void spinor_field_cvc_to_quda ( double * const r, double * const s );

}
#endif  /* _GFLOW_QUDA */
#endif
