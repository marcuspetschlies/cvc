/***************************************************
 * su3
 ***************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "table_init_d.h"
#include "table_init_z.h"

namespace cvc {

double _Complex ***  lambda_gm = NULL;

/***************************************************/
/***************************************************/

void init_lambda_gm (void) {

  double const norm8 = 1. / sqrt ( 3. );

  lambda_gm = init_3level_ztable ( 9, 3, 3 );

  /* lambda_0 = unit matrix */
  lambda_gm[0][0][0] = 1.;
  lambda_gm[0][1][1] = 1.;
  lambda_gm[0][2][2] = 1.;

  /* lambda_1 */
  lambda_gm[1][0][1] = 1.;
  lambda_gm[1][1][0] = 1.;

  /* lambda_2 */
  lambda_gm[2][0][1] = -1. * I;
  lambda_gm[2][1][0] =       I;

  /* lambda_3 */
  lambda_gm[3][0][0] =  1.;
  lambda_gm[3][1][1] = -1.;

  /* lambda_4 */
  lambda_gm[4][0][2] = 1.;
  lambda_gm[4][2][0] = 1.;

  /* lambda_5 */
  lambda_gm[5][0][2] = -1. * I;
  lambda_gm[5][2][0] =       I;

  /* lambda_6 */
  lambda_gm[6][1][2] = 1.;
  lambda_gm[6][2][1] = 1.;

  /* lambda_7 */
  lambda_gm[7][1][2] = -1. * I;
  lambda_gm[7][2][1] =       I;

  /* lambda_8 */
  lambda_gm[8][0][0] =       norm8;
  lambda_gm[8][1][1] =       norm8;
  lambda_gm[8][2][2] = -2. * norm8;
}  /* end of init_lambda_gm */

/***************************************************/
/***************************************************/

void fini_lambda_gm (void) {
  fini_3level_ztable ( &lambda_gm );

}  /* fini_lambda_gm */

}
