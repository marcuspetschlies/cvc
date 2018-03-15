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
#include "mpi_init.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "laplace_linalg.h"

namespace cvc {

/**************************************************************************************************************
 * application of laplace operator on 4-dim VOLUME color vector field
 **************************************************************************************************************/
void cv_eq_laplace_cv_4d ( double * const v_out, double * const g, double * const v_in ) {

  unsigned int const VOL3 = LX*LY*LZ;

#ifdef HAVE_MPI
  xchanger_type xcolvec;
  mpi_init_xchanger ( &xcolvec, _GVI(1) );

  double * const v_aux = (double * ) malloc ( _GVI(VOLUME+RAND) * sizeof(double) );
  if ( v_aux == NULL ) {
    fprintf( stderr, "[cvc_eq_laplace_cv_4d] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(3);
  }
  memcpy ( v_aux, v_in, _GVI(VOLUME)*sizeof(double) );
  // if ( g_cart_id == 0 ) fprintf ( stdout, "# [cv_eq_laplace_cv_4d] calling mpi xchanger\n");
  mpi_xchanger ( v_aux, &xcolvec );
  xchange_gauge_field ( g );
#else
  double * const v_aux = v_in;
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < VOLUME; ix++) {

    double v1[6], v2[6], v3[6];

/*
    unsigned int const ix_pl_mu1 = g_iup[ix][1];
    unsigned int const ix_pl_mu2 = g_iup[ix][2];
    unsigned int const ix_pl_mu3 = g_iup[ix][3];
*/
    unsigned int const ix_mi_mu1 = g_idn[ix][1];
    unsigned int const ix_mi_mu2 = g_idn[ix][2];
    unsigned int const ix_mi_mu3 = g_idn[ix][3];

    unsigned int const iy_pl_mu1 = g_iup[ix][1];
    unsigned int const iy_pl_mu2 = g_iup[ix][2];
    unsigned int const iy_pl_mu3 = g_iup[ix][3];
    unsigned int const iy_mi_mu1 = g_idn[ix][1];
    unsigned int const iy_mi_mu2 = g_idn[ix][2];
    unsigned int const iy_mi_mu3 = g_idn[ix][3];

    /* x-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       1), v_aux+_GVI(iy_pl_mu1) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu1,1), v_aux+_GVI(iy_mi_mu1) );
    _cv_eq_cv_pl_cv(v3, v1, v2);

    /* y-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       2), v_aux+_GVI(iy_pl_mu2) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu2,2), v_aux+_GVI(iy_mi_mu2) );
    _cv_pl_eq_cv(v3, v1);
    _cv_pl_eq_cv(v3, v2);

    /* z-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       3), v_aux+_GVI(iy_pl_mu3) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu3,3), v_aux+_GVI(iy_mi_mu3) );
    _cv_pl_eq_cv(v3, v1);
    _cv_pl_eq_cv(v3, v2);

    /* diagonal part */
    _cv_eq_cv_ti_re(v1, v_aux+_GVI(ix), -6. );

    /* add together */
    _cv_eq_cv_pl_cv(v_out+_GVI(ix), v3, v1);

  }  /* end of loop on spatial site */


#ifdef HAVE_MPI
  mpi_fini_xchanger ( &xcolvec );
  free ( v_aux );
#endif

  return;

}  /* end of cv_eq_laplace_cv_4d */

/**************************************************************************************************************/
/**************************************************************************************************************/

/**************************************************************************************************************
 * application of laplace operator on 3-dim VOL3 color vector field
 **************************************************************************************************************/
void cv_eq_laplace_cv_3d ( double * const v_out, double * const g, double * const v_in, int const ts) {

  unsigned int const VOL3 = LX*LY*LZ;

#if (defined HAVE_MPI ) && ( (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) )

  double * const v_aux = (double * ) malloc ( _GVI(VOL3+RAND3) * sizeof(double) );
  if ( v_aux == NULL ) {
    fprintf( stderr, "[cvc_eq_laplace_cv] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(3);
  }
  memcpy ( v_aux, v_in, _GVI(VOL3)*sizeof(double) );
  xchange_nvector_3d ( v_aux, _GVI(1), -1 );

  xchange_gauge_field ( g );
#else
  double * const v_aux = v_in;
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int iix = 0; iix < VOL3; iix++) {

    double v1[6], v2[6], v3[6];

    // global index
    unsigned int const ix = ts * VOL3 + iix;

    //unsigned int const ix_pl_mu1 = g_iup[ix][1];
    //unsigned int const ix_pl_mu2 = g_iup[ix][2];
    //unsigned int const ix_pl_mu3 = g_iup[ix][3];

    unsigned int const ix_mi_mu1 = g_idn[ix][1];
    unsigned int const ix_mi_mu2 = g_idn[ix][2];
    unsigned int const ix_mi_mu3 = g_idn[ix][3];

    unsigned int const iy_pl_mu1 = g_iup_3d[iix][0];
    unsigned int const iy_pl_mu2 = g_iup_3d[iix][1];
    unsigned int const iy_pl_mu3 = g_iup_3d[iix][2];
    unsigned int const iy_mi_mu1 = g_idn_3d[iix][0];
    unsigned int const iy_mi_mu2 = g_idn_3d[iix][1];
    unsigned int const iy_mi_mu3 = g_idn_3d[iix][2];

    /* x-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       1), v_aux+_GVI(iy_pl_mu1) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu1,1), v_aux+_GVI(iy_mi_mu1) );
    _cv_eq_cv_pl_cv(v3, v1, v2);

    /* y-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       2), v_aux+_GVI(iy_pl_mu2) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu2,2), v_aux+_GVI(iy_mi_mu2) );
    _cv_pl_eq_cv(v3, v1);
    _cv_pl_eq_cv(v3, v2);

    /* z-direction */
    _cv_eq_cm_ti_cv    (v1, g+_GGI(ix,       3), v_aux+_GVI(iy_pl_mu3) );
    _cv_eq_cm_dag_ti_cv(v2, g+_GGI(ix_mi_mu3,3), v_aux+_GVI(iy_mi_mu3) );
    _cv_pl_eq_cv(v3, v1);
    _cv_pl_eq_cv(v3, v2);

    /* diagonal part */
    _cv_eq_cv_ti_re(v1, v_aux+_GVI(iix), -6. );

    /* add together */
    _cv_eq_cv_pl_cv(v_out+_GVI(iix), v3, v1);

  }  /* end of loop on spatial site */


#ifdef HAVE_MPI
  free ( v_aux );
#endif


  return;

}  /* end of cv_eq_laplace_cv_3d */

/**************************************************************************************************************/
/**************************************************************************************************************/

}  /* end of namespace cvc */
