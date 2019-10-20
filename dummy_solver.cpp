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
#if ( defined DUMMY_SOLVER ) || !(defined HAVE_TMLQCD_LIBWRAPPER )
/************************************************************************************
 * dummy solver
 ************************************************************************************/
/* int dummy_solver ( double * const propagator, double * const source, const int op_id, int write_prop ) */
int dummy_solver ( double * const propagator, double * const source, const int op_id )
{
    return( rangauss(propagator, _GSI(VOLUME) ) );
}  /* end of dummy_solver */
#endif
}  /* end of namespace cvc */
