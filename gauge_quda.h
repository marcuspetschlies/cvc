#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

namespace cvc {

int get_gauge_padding ( int x[4] );

void gauge_field_cvc_to_qdp ( double ** g_qdp, double * g_cvc );

void gauge_field_qdp_to_cvc ( double * g_cvc, double ** g_qdp );

}

#endif
