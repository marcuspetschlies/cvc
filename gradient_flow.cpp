#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_MPI
#include <mpi.h>  
#endif
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "mpi_init.h"
#include "cvc_geometry.h"
#include "table_init_d.h"
#include "cvc_utils.h"
#include "smearing_techniques.h"
#include "gradient_flow.h"

namespace cvc {

/******************************************************************
 * calculate \bar{Z}_X
 * = a Z(g) + b z
 ******************************************************************/
void ZX ( double * const z, double * const g, double const a, double const b ) {
#ifdef HAVE_MPI
    xchange_gauge_field ( g );
#endif
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  double tmp[18], tmp2[18];
  double * _z = NULL;
  double * _g = NULL;
  unsigned int iix;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for ( unsigned int x = 0; x < VOLUME; ++x) {
    for ( int mu = 0; mu < 4; ++mu)
    {
        iix      = _GGI( x, mu);
        _z       = z + iix;
        _g       = g + iix;

        /* calculate new staples */
        generic_staples( tmp, x, mu, g );

        /* multiply by new weight a */
        _cm_ti_eq_re( tmp, a );

        /* complete plaquette by multiplication with U_mu,x^+ */
        _cm_eq_cm_ti_cm_dag ( tmp2, tmp, _g );

        /* antihermitean part */
        _cm_eq_antiherm_trless_cm ( tmp, tmp2 );

        /* z_old <- z_old * b */
        _cm_ti_eq_re ( _z , b );

        /* z_new <- z_old + tmp */
        _cm_pl_eq_cm ( _z , tmp );
    }
  }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  return;
}  /* end of ZX */

/******************************************************************
 * calculate exp ( \bar{Z}_X dt ) g
 ******************************************************************/
void apply_ZX ( double * const g, double * const z, double const dt ) {
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  double tmp[18], tmp2[18];
  double * _z = NULL;
  double * _g = NULL;
  unsigned int iix;

#ifdef HAVE_OPENMP
#pragma omp barrier
#pragma omp for
#endif
    for ( unsigned int x = 0; x < VOLUME; ++x) {
      for(int mu = 0 ; mu < 4; ++mu)
      {
        iix = _GGI( x, mu);
        _z  = z + iix;
        _g  = g + iix;

        /* multiply z by epsilon
         * tmp <- z * dt */
        _cm_eq_cm_ti_re ( tmp, _z, dt );

        /* exponentiate
         * tmp2 <- exp( tmp ) */
        exposu3 ( tmp2, tmp );

        /* tmp <- tmp2 x g = exp( dt z ) g */
        _cm_eq_cm_ti_cm( tmp, tmp2, _g );

        /* g <- tmp = tmp2 g = exp ( dt z ) g */
        _cm_eq_cm ( _g, tmp );
      }
    }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of apply_ZX */

/******************************************************************
 * apply Laplace to spinor field
 *
 * in-place s == r_in is okay, we copy r_in to independent r ahead
 * of application
 *
 * NOTE: input field g MUST HAVE BOUNDARY allocated
 ******************************************************************/
void apply_laplace ( double * const s, double * const r_in, double * const g ) {

  unsigned int const N = VOLUME;
  static double * r = NULL;

  if ( s == NULL ) {
    if ( r != NULL ) {
      free ( r );
      r = NULL;
    }
    fprintf ( stdout, "# [apply_laplace] clean up done %s %d\n", __FILE__, __LINE__ );
    return;
  }

  if ( r == NULL ) {
    r = (double*) malloc ( _GSI( (VOLUME+RAND) ) * sizeof(double) );
    if ( r == NULL ) {
      fprintf ( stderr, "[apply_laplace] Error from malloc %s %d\n", __FILE__, __LINE__ );
      return;
    }
  }
  memcpy ( r, r_in, _GSI(VOLUME)*sizeof(double) );

#ifdef HAVE_MPI
  xchange_field( r );

  xchange_gauge_field( g );
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int ix = 0; ix < N; ix++ ) {

    double * const s_ = s + _GSI(ix);
    double * const r_ = r + _GSI(ix);
    double spinor1[24];

    /* s = - 2 D r , D = 4 */
    _fv_eq_fv_ti_re ( s_, r_, -8. );

    for ( int mu = 0; mu < 4; mu++ ) {
        /* ================================================
           =============== direction mu fwd ===============
           ================================================ */
        double * const U_fwd = g + _GGI(ix,mu);

        /* ix_fwd */
        unsigned int const ix_fwd = g_iup[ix][mu];

        double * const r_fwd_ = r + _GSI(ix_fwd);

        /* spinor1 = U_fwd r_fwd_ */
        _fv_eq_cm_ti_fv( spinor1, U_fwd, r_fwd_ );

        /* s_ <- s_ + spinor1 = s_ + U_fwd r_fwd */
        _fv_pl_eq_fv ( s_, spinor1 );

        /* ================================================
           =============== direction mu bwd ===============
           ================================================ */
        double * const U_bwd = g + _GGI( g_idn[ix][mu], mu);

        /* ix_bwd */
        unsigned int const ix_bwd = g_idn[ix][mu];

        double * const r_bwd_ = r + _GSI(ix_bwd);

        /* spinor1 = U_bwd^+ r_bwd_ */
        _fv_eq_cm_dag_ti_fv( spinor1, U_bwd, r_bwd_ );

        /* s_ <- s + spinor1 */
        _fv_pl_eq_fv ( s_, spinor1 );
      
    }  /* end of loop on mu */

  }  /* end of loop on ix over VOLUME */

  return;
}  /* end of apply_laplace */


/******************************************************************
 * using 4-dimensional isotropic stout smearing as kernel
 ******************************************************************/
void flow_fwd_gauge_spinor_field ( double * const g, double * const chi, unsigned int const niter, double const dt, int const flow_gauge, int const flow_spinor ) {

  size_t const sizeof_gauge_field  = 72 * VOLUME * sizeof ( double );
  size_t const sizeof_spinor_field = _GSI(VOLUME) * sizeof ( double );

  static double * w = NULL, * z = NULL;
  static double ** phi = NULL;

  /******************************************************************
   * trigger clean-up with no flow
   ******************************************************************/
  if ( ! ( flow_gauge  || flow_spinor ) ) {
    apply_laplace ( NULL, NULL, NULL );

    fini_1level_dtable ( &w );
    fini_1level_dtable ( &z );

    fini_2level_dtable ( &phi );

    w = NULL; z = NULL; phi = NULL;
        
    fprintf ( stdout, "# [flow_fwd_gauge_spinor_field] clean up done %s %d\n", __FILE__, __LINE__ );
    return;
  }

  /******************************************************************
   * allocate gauge fields
   ******************************************************************/
  if ( ( flow_gauge || flow_spinor ) && w == NULL ) { 
    w = init_1level_dtable ( 72 * VOLUMEPLUSRAND );
    if ( w == NULL ) {
      fprintf ( stderr, "[flow_fwd_gauge_spinor_field] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
      return;
    }
  }

  if ( flow_gauge && ( z == NULL ) ) { 
    z = init_1level_dtable ( 72 * VOLUME );
    if ( z == NULL ) {
      fprintf ( stderr, "[flow_fwd_gauge_spinor_field] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
      return;
    }
  }

  /******************************************************************
   * allocate phi fields
   ******************************************************************/
  if ( flow_spinor && ( phi == NULL ) ) { 
    phi = init_2level_dtable ( 4,  _GSI(VOLUME+RAND) );
    if ( phi == NULL ) {
      fprintf ( stderr, "[flow_fwd_gauge_spinor_field] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      return;
    }
  }

  /******************************************************************
   * STEP 0
   *
   * W0 = g
   *
   * phi3 = chi
   ******************************************************************/
  memcpy ( w, g, sizeof_gauge_field );
#ifdef HAVE_MPI
  xchange_gauge_field ( w );
#endif

  if ( flow_spinor ) {
    memcpy ( phi[3], chi, sizeof_spinor_field );
  }

  /* loop on time steps */
  for ( unsigned int iter = 0; iter < niter; iter++ )
  {

    /******************************************************************
     * STEP 1
     *
     * W1 = exp( 1/4 Z0 ) W0
     *
     ******************************************************************/

    if ( flow_spinor ) {

      /* update initial field phi0 from phi3,
       * phi3 set either from previous iteration or by initialization STEP 0 above */
      memcpy ( phi[0], phi[3], sizeof_spinor_field );

      /* phi1 = phi0 */
      memcpy ( phi[1], phi[0], sizeof_spinor_field );

      /* phi2 = phi0 */
      memcpy ( phi[2], phi[0], sizeof_spinor_field );

      /* phi0 <- Delta0 phi0 */
      apply_laplace ( phi[0], phi[0], w );


      /* phi1 <- phi1 + dt/4 phi0 = phi1 + dt/4 Delta0 phi0 */
      spinor_field_pl_eq_spinor_field_ti_re ( phi[1], phi[0],  dt/4., VOLUME );

    }



    if ( flow_gauge ) {
      /* calculate Z0 = Z( W0 ) */
      ZX ( z, w, 1./4., 0. );

      /* W1 = exp(dt Z0) W0 */
      apply_ZX ( w, z, dt );
    }


    /******************************************************************
     * STEP 2
     *
     * W2 = exp( 8/9 Z1 - 17/36 Z0 ) W1
     *
     ******************************************************************/

    if ( flow_spinor ) {

      /* phi3 = phi1 */
      memcpy ( phi[3], phi[1], sizeof_spinor_field );

      /* phi1 <- Delta1 phi1 */
      apply_laplace ( phi[1], phi[1], w );

      /* TEST */
      /* memcpy ( phi[3], phi[1], sizeof_spinor_field ); */


      /* phi2 <- phi2 + dt 8/9 phi1 = phi0 + dt 8/9 Delta1 phi1 */
      spinor_field_pl_eq_spinor_field_ti_re ( phi[2], phi[1],  dt*8./9., VOLUME );

      /* phi2 <- phi2 -dt 2/9 phi0 = phi0 + dt 8/9 Delta1 phi1 - 2/9 Delta0 phi0 */
      spinor_field_pl_eq_spinor_field_ti_re ( phi[2], phi[0],  -dt*2./9., VOLUME );

      /* TEST */
      /* memcpy ( phi[3], phi[2], sizeof_spinor_field ); */

    }


    if ( flow_gauge ) {
      /* calculate Z1 = Z( W1 )
       * and z <- 8/9 Z1 -17/9 z = 8/9 Z1 - 17/36 Z0 */
      ZX ( z, w, 8./9., -17./9. );

      /* W2 = exp( dt z ) W1 */
      apply_ZX ( w, z, dt );
    }



    /******************************************************************
     * STEP 3
     *
     * W3 = exp( 3/4 Z2 -8/9 Z1 + 17/36 Z0 ) W2
     *
     ******************************************************************/

    if ( flow_spinor ) {

      /* phi2 <- Delta2 phi2 */
      apply_laplace ( phi[2], phi[2], w );

      /* phi3 <- phi3 + dt 3/4 phi2 = phi1 + dt 3/4 Delta2 phi2 */
      spinor_field_pl_eq_spinor_field_ti_re ( phi[3], phi[2], dt*3./4., VOLUME );
    }

    if ( flow_gauge ) {
   
      /* calculate Z2 = Z( W2 )
       * and z <- 3/4 Z2 - z = 3/4 Z2 - 8/9 Z1 + 17/36 Z1 */
      ZX ( z, w, 3./4., -1. );

      apply_ZX ( w, z, dt );
    }
#if 0
#endif

  }  /* end of loop on iterations */

  if ( flow_gauge ) {
    memcpy ( g, w, sizeof_gauge_field );
  }
  if ( flow_spinor ) {
    memcpy ( chi, phi[3], sizeof_spinor_field );
  }

}  /* end of flow_fwd_gauge_spinor_field */

/******************************************************************/
/******************************************************************/

/******************************************************************
 * for adjoint flow recursion
 *
 * This single iteration step implements eq. (E.8)
 * in ref. 1302.5246v2 [hep-lat]
 *
 * gauge field flow follows eqs. (D.2) for W0/1/2 same ref.
 *
 * g itself remains unchanged
 ******************************************************************/
void flow_adjoint_step_gauge_spinor_field ( double * const g, double * const chi, unsigned int const niter, double const dt, int const flow_gauge, int const flow_spinor ) {

  size_t const sizeof_gauge_field  = 72 * VOLUME * sizeof ( double );
  size_t const sizeof_spinor_field = _GSI(VOLUME) * sizeof ( double );

  static double ** w = NULL, * z = NULL, ** lambda = NULL;

  /******************************************************************
   * trigger clean-up with no flow
   ******************************************************************/
  if ( ! ( flow_gauge  || flow_spinor ) ) {
    apply_laplace ( NULL, NULL, NULL );

    fini_2level_dtable ( &w );
    fini_1level_dtable ( &z );

    fini_2level_dtable ( &lambda );
     
     w = NULL; z = NULL; lambda = NULL;

    fprintf ( stdout, "# [flow_adjoint_step_gauge_spinor_field] clean up done %s %d\n", __FILE__, __LINE__ );
    return;
  }

  /******************************************************************
   * allocate gauge fields
   ******************************************************************/
  if ( ( flow_gauge || flow_spinor ) && w == NULL ) { 
    w = init_2level_dtable ( 3, 72 * VOLUMEPLUSRAND );
    if ( w == NULL ) {
      fprintf ( stderr, "[flow_adjoint_step_gauge_spinor_field] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
      return;
    }
  }

  if ( flow_gauge && ( z == NULL ) ) { 
    z = init_1level_dtable ( 72 * VOLUME );
    if ( z == NULL ) {
      fprintf ( stderr, "[flow_adjoint_step_gauge_spinor_field] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
      return;
    }
  }

  /******************************************************************
   * allocate lambda fields
   ******************************************************************/
  if ( flow_spinor && ( lambda == NULL ) ) { 
    lambda = init_2level_dtable ( 4,  _GSI(VOLUME+RAND) );
    if ( lambda == NULL ) {
      fprintf ( stderr, "[flow_adjoint_step_gauge_spinor_field] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      return;
    }
  }

  /******************************************************************
   * STEP 0
   *
   * W0 = g
   *
   ******************************************************************/
  memcpy ( w[0], g, sizeof_gauge_field );

  if ( flow_gauge ) {

    if ( g_verbose > 2 ) fprintf ( stdout,"# [flow_adjoint_step_gauge_spinor_field] flow gauge field for %u iterations %s %d\n", niter, __FILE__, __LINE__ );

    /******************************************************************
     * apply niter full iterations of gradient flow applications
     ******************************************************************/
    for ( unsigned int i = 0; i < niter; i++ ) {

      /******************************************************************
       * STEP 1
       * W1 = exp( 1/4 Z0 ) W0
       ******************************************************************/
      /* calculate Z0 = Z( W0 ) 
       *
       * NOTE: ZX contains exchange of gauge field 
       */
      ZX ( z, w[0], 1./4., 0. );

      /* W1 <- exp(dt z) W1 = exp( dt Z0 ) W0 */
      apply_ZX ( w[0], z, dt );

      /******************************************************************
       * STEP 2
       * W2 = exp( 8/9 Z1 - 17/36 Z0 ) W1
       ******************************************************************/
      /* calculate Z1 = Z( W1 )
       * and z <- 8/9 Z1 -17/9 z = 8/9 Z1 - 17/36 Z0 */
      ZX ( z, w[0], 8./9., -17./9. );

      /* W2 <- exp( dt z ) W2 = exp( dt Z1 ) W1 */
      apply_ZX ( w[0], z, dt );

      /******************************************************************
       * STEP 3
       * W3 = exp( 3/4 Z2 -8/9 Z1 + 17/36 Z0 ) W2
       ******************************************************************/
      /* calculate Z2 = Z( W2 )
       * and z <- 3/4 Z2 - z = 3/4 Z2 - 8/9 Z1 + 17/36 Z1 */
      ZX ( z, w[0], 3./4., -1. );

      apply_ZX ( w[0], z, dt );

      if ( g_verbose > 2 ) {
        double plaq;
        plaquette2 ( &plaq, w[0] );
        fprintf ( stdout, "# [flow_adjoint_step_gauge_spinor_field] niter %3u iter %3u plaquette %25.16e %s %d\n", niter, i, plaq, __FILE__, __LINE__ );
      }


    }  /* end of loop on iteration of full GF application to gauge field only */

    if ( ! flow_spinor ) {
      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [flow_adjoint_step_gauge_spinor_field] flow = %d %d, copy gauge %s %d\n", flow_gauge, flow_spinor, __FILE__, __LINE__ );
      }

      /* g = w[0] ; w_0 is the result of the iterations */
      memcpy ( g, w[0], sizeof_gauge_field );
    }

  } /* end of if flow_gague */

  /******************************************************************/
  /******************************************************************/

  /******************************************************************
   * the rest of GF application to both gauge and spinor field 
   * is done ONLY IF the spinor field is flowed
   ******************************************************************/
  if ( flow_spinor ) {

    if ( g_verbose > 2 ) {
      double plaq;
      plaquette2 ( &plaq, w[0] );
      fprintf ( stdout, "# [flow_adjoint_step_gauge_spinor_field] start spinor flow at plaquette %25.16e %s %d\n", plaq, __FILE__, __LINE__ );
    }

    if ( g_verbose > 2 ) fprintf ( stdout,"# [flow_adjoint_step_gauge_spinor_field] flow spinor field for single step %s %d\n",  __FILE__, __LINE__ );

    /******************************************************************
     * STEP 0
     *
     *  W0 is already set, either from STEP 0 or from niter
     * full GF iterations
     ******************************************************************/

    /******************************************************************
     * STEP 1
     * W1 = exp( 1/4 Z0 ) W0
     ******************************************************************/

    memcpy ( w[1], w[0], sizeof_gauge_field );

    if ( flow_gauge ) {
      /* calculate Z0 = Z( W0 ) 
       *
       * NOTE: ZX contains exchange of gauge field 
       */
      ZX ( z, w[1], 1./4., 0. );

      /* W1 <- exp(dt z) W1 = exp( dt Z0 ) W0 */
      apply_ZX ( w[1], z, dt );
    }

    /******************************************************************
     * STEP 2
     * W2 = exp( 8/9 Z1 - 17/36 Z0 ) W1
     ******************************************************************/
    memcpy ( w[2], w[1], sizeof_gauge_field );

    if ( flow_gauge ) {
      /* calculate Z1 = Z( W1 )
       * and z <- 8/9 Z1 -17/9 z = 8/9 Z1 - 17/36 Z0 */
      ZX ( z, w[2], 8./9., -17./9. );
  
      /* W2 <- exp( dt z ) W2 = exp( dt Z1 ) W1 */
      apply_ZX ( w[2], z, dt );
    }

    /******************************************************************/
    /******************************************************************/

    /* lambda_k = 0 for k = 3,2,1,0 */
    memset ( lambda[0] , 0, 4*sizeof_spinor_field );

    /* lambda_3 = chi input field */
    memcpy ( lambda[3], chi, sizeof_spinor_field );

    /* lambda_1 = lambda_3 */
    memcpy ( lambda[1], lambda[3], sizeof_spinor_field );

    /* lambda_3 <- Delta_2 lambda_3 in-place */
    apply_laplace ( lambda[3], lambda[3], w[2] );

    /* lambda_2 <- lambda_2 + 3/4 dt lambda_3 = 3/4 dt lambda_3 , lambda_2 = 0 initially */
    spinor_field_pl_eq_spinor_field_ti_re ( lambda[2], lambda[3],  3./4. * dt, VOLUME );

    /* lambda_0 = lambda_2 , since we have lambda_2 finished now */
    memcpy ( lambda[0], lambda[2], sizeof_spinor_field );

    /* lambda_3 = Delta_1 lambda_2 ; lambda_3 as auxilliary field */
    apply_laplace ( lambda[3], lambda[2], w[1] );

    /* lambda_1 <- lambda_1 + 8/9 dt lambda_3 = lambda_3 + 8/9 dt Delta_1 lambda_2 ; since lambda_3 holds Delta_1 lambda_2 */
    spinor_field_pl_eq_spinor_field_ti_re ( lambda[1], lambda[3],  8./9. * dt, VOLUME );

    /* lambda_0 <- lambda_0 + lambda_1 = lambda_1 + lambda_2 ; since lambda_0 had been set to lambda_2 before and lambda_1 is finished now */
    spinor_field_pl_eq_spinor_field_ti_re ( lambda[0], lambda[1],  1., VOLUME );

    /* lambda_1 <- lambda_1 - 8/9 lambda_2 ; since lambda_2 has been kept / not overwritten by D_1 l_2 */
    spinor_field_pl_eq_spinor_field_ti_re ( lambda[1], lambda[2],  -8./9., VOLUME );

    /* lambda_1 <- Deltal_0 lambda_1 ; in-place */
    apply_laplace ( lambda[1], lambda[1], w[0] );

    /* lambda_0 <- lambda_0 + 1/4 dt lambda_1 = lambda_1 + lambda_2 + 1/4 dt Delta_0 ( lambda_1 - 8/9 lambda_2 ) */
    spinor_field_pl_eq_spinor_field_ti_re ( lambda[0], lambda[1],  1./4. * dt, VOLUME );

    /* chi = lambda_0 */
    memcpy ( chi, lambda[0], sizeof_spinor_field );

  }  /* end of flow_spinor */

  return;

}  /* end of flow_adjoint_step_gauge_spinor_field */


/******************************************************************
 *
 ******************************************************************/
void flow_adjoint_gauge_spinor_field ( double ** const g, double * const chi, double const dt, unsigned int const mb, unsigned int const nb, int const store ) {

  size_t const sizeof_gauge_field  = 72 * VOLUME * sizeof ( double );

  if ( g_verbose > 2 ) {
    fprintf ( stdout, "# [flow_adjoint_gauge_spinor_field] entering level %d nb %3d mb %3d  %s %d\n", store, nb, mb, __FILE__, __LINE__ );
  }

  if ( store == 0 ) {
    /******************************************************************
     * end of blocking, do not store any more gauge fields, but
     * flow from current level
     ******************************************************************/

    if ( g_verbose > 2 ) {
      fprintf ( stdout, "# [flow_adjoint_gauge_spinor_field] finish level %d nb %3d mb %3d %s %d\n", store, nb, mb, __FILE__, __LINE__ );
    }


    for ( unsigned int i = 0; i < mb; i++ ) {
      unsigned int const niter = mb - 1 - i;
      flow_adjoint_step_gauge_spinor_field ( g[store], chi, niter, dt, 1, 1 );
    }

  } else {
    unsigned int const kb = mb % nb;
    unsigned int const lb = mb / nb;
    unsigned int niter = mb;

    /* for ( unsigned int ib = nb-1; ib >= 0; ib-- )  */
    for ( unsigned int ib = 0; ib < nb; ib++ ) 
    {
    
      /* flow next lower-level gauge field, start from current level gauge field 
       * g[store-1] <- GF ( g[store] )
       */

      /* current block length ( i.e. how many dt steps in current block )
       *
       * first kb ones will one longer, than the rest
       */
      unsigned int const mb_new = ib < kb ? lb + 1 : lb;
 
      /* number of full GF iterations on gauge field, starting from current level left block end g[store],
       * to reach the next lower level left block end
       */
      niter -= mb_new;

      memcpy ( g[store-1], g[store], sizeof_gauge_field );

      if ( g_verbose > 2 ) fprintf ( stdout, "# [flow_adjoint_gauge_spinor_field] store %d block %3d  kb %3d lb %3d mb_new %3d   %s %d\n",
          store, ib, kb, lb, mb_new, __FILE__, __LINE__ );

      double plaq;
      plaquette2 ( &plaq, g[store-1]);
      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [flow_adjoint_gauge_spinor_field] plaquette before %25.16e %s %d\n", plaq, __FILE__, __LINE__ );
      }

      flow_adjoint_step_gauge_spinor_field ( g[store-1], NULL, niter, dt, 1, 0 );

      plaquette2 ( &plaq, g[store-1]);
      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [flow_adjoint_gauge_spinor_field] plaquette afer niter %u %25.16e %s %d\n", niter, plaq, __FILE__, __LINE__ );
      }

      /******************************************************************
       * continue with blocking
       ******************************************************************/
      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [flow_adjoint_gauge_spinor_field] descend level %d ---> %d  %s %d\n", store, store-1, __FILE__, __LINE__ );
      }

      /* function recalls itself with next lower level gauge field */
      flow_adjoint_gauge_spinor_field ( g, chi, dt, mb_new, nb, store-1 );

    }  /* end of loop on blocks */

  }  /* end of else of if store == 0 */

  return;
}  /* end of flow_adjoint_gauge_spinor_field */

}  /* end of namespace cvc */

