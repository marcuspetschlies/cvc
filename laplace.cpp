#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include "cvc_complex.h"
#include "global.h"
#include "cvc_linalg.h"
#include "cvc_geometry.h"
#include "laplace_linalg.h"

namespace cvc {

void cv_eq_laplace_cv(double *v_out, double *g, double*v_in, int ts) {
  unsigned int ix, iix;
  /* unsigned int ix_pl_mu1, ix_pl_mu2, ix_pl_mu3; */
  unsigned int ix_mi_mu1, ix_mi_mu2, ix_mi_mu3;
  unsigned int iy_pl_mu1, iy_pl_mu2, iy_pl_mu3, iy_mi_mu1, iy_mi_mu2, iy_mi_mu3;
  double v1[6], v2[6], v3[6];
  unsigned int VOL3 = LX*LY*LZ;

  for(iix = 0; iix < VOL3; iix++) {
    /* global index */
    ix = ts * VOL3 + iix;
/*
    ix_pl_mu1 = g_iup[ix][1];
    ix_pl_mu2 = g_iup[ix][2];
    ix_pl_mu3 = g_iup[ix][3];
*/
    ix_mi_mu1 = g_idn[ix][1];
    ix_mi_mu2 = g_idn[ix][2];
    ix_mi_mu3 = g_idn[ix][3];

    iy_pl_mu1 = g_iup[iix][1];
    iy_pl_mu2 = g_iup[iix][2];
    iy_pl_mu3 = g_iup[iix][3];
    iy_mi_mu1 = g_idn[iix][1];
    iy_mi_mu2 = g_idn[iix][2];
    iy_mi_mu3 = g_idn[iix][3];

    /* x-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       1), v_in+_GVI(iy_pl_mu1) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu1,1), v_in+_GVI(iy_mi_mu1) );
    _cv_eq_cv_pl_cv(v3, v1, v2);

    /* y-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       2), v_in+_GVI(iy_pl_mu2) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu2,2), v_in+_GVI(iy_mi_mu2) );
    _cv_pl_eq_cv(v3, v1);
    _cv_pl_eq_cv(v3, v2);

    /* z-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       3), v_in+_GVI(iy_pl_mu3) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu3,3), v_in+_GVI(iy_mi_mu3) );
    _cv_pl_eq_cv(v3, v1);
    _cv_pl_eq_cv(v3, v2);

    /* diagonal part */
    _cv_eq_cv_ti_re(v1, v_in+_GVI(iix), -6. );

    /* add together */
    _cv_eq_cv_pl_cv(v_out+_GVI(iix), v3, v1);

  }  /* end of loop on spatial site */

  return;

}  /* end of cv_eq_laplace_cv */

}
