#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include "ranlxd.h"
#include "global.h"
#include "cvc_utils.h"

namespace cvc {
#if ( defined DUMMY_SOLVER ) || ( ! defined HAVE_TMLQCD_LIBWRAPPER )
/************************************************************************************
 * dummy solver
 ************************************************************************************/
int dummy_solver ( double * const propagator, double * const source, int const op_id ) {
 return( rangauss(propagator, _GSI(VOLUME) ) );
 memcpy ( propagator, source, _GSI(VOLUME)*sizeof(double) );
 return ( 0 ) ;
}  /* end of dummy_solver */
#endif
}  /* end of namespace cvc */
