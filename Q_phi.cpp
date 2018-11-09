/********************
 * Q_phi.c
 ********************/

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
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "Q_phi.h"

namespace cvc {

/********************
 Q_phi_tbc 
 - computes xi = Q phi, where Q is the tm Dirac operator with twisted boundary conditions.
 - should be compatible with Carsten's implementation of the Dirac operator 
   in hmc's invert programme
 - Q is written as Q = m_0 + i mu gamma_5 + ...
   (not in the hopping parameter representation)
 - off-diagonal part as:
   -1/2 sum over mu [ (1 - gamma_mu) U_mu(x) phi(x+mu) + 
       (1 + gamma_mu) U_mu(x-mu)^+ phi(x-mu) ]
 - diagonal part as: 1 / (2 kappa) phi(x) + i mu gamma_5 \phi(x)
*/

void Q_phi_tbc(double *xi, double *phi) {

  const double _1_2_kappa = 0.5 / g_kappa;

#ifdef HAVE_MPI
  xchange_field(phi);
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel shared(xi,phi)
{
#endif
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int index_s = 0; index_s < VOLUME; index_s++) {

      xi_ = xi + _GSI(index_s);

      _fv_eq_zero(xi_);

      /* Negative t-direction. */
      phi_ = phi + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][0], 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive t-direction. */
      phi_ = phi + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 0);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[0]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Negative x-direction. */
      phi_ = phi + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][1], 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive x-direction. */
      phi_ = phi + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 1);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[1]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative y-direction. */
      phi_ = phi + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][2], 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Positive y-direction. */
      phi_ = phi + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 2);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[2]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative z-direction. */
      phi_ = phi + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][3], 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive z-direction. */
      phi_ = phi + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);
 
      U_ = g_gauge_field + _GGI(index_s, 3);

      _cm_eq_cm_ti_co(SU3_1, U_, &co_phase_up[3]);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Multiplication with -1/2. */

      _fv_ti_eq_re(xi_, -0.5);

      /* Diagonal elements. */

      phi_ = phi + _GSI(index_s);
		    
      _fv_eq_fv_ti_re(spinor1, phi_, _1_2_kappa);
      _fv_pl_eq_fv(xi_, spinor1);
		    
      _fv_eq_gamma_ti_fv(spinor1, 5, phi_);
      _fv_eq_fv_ti_im(spinor2, spinor1, g_mu);
      _fv_pl_eq_fv(xi_, spinor2);
  }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

} /* end of Q_phi_tbc */

/**********************************************************************
 * Hopping 
 * - computes xi = H phi, where H is the Dirac Wilson hopping matrix
 * - off-diagonal part as:
 *   -1/2 sum over mu [ (1 - gamma_mu) U_mu(x) phi(x+mu) + 
 *      (1 + gamma_mu) U_mu(x-mu)^+ phi(x-mu) ]
 * - NOT in the hopping parameter representation
 **********************************************************************/

void Hopping(double *xi, double *phi, double*gauge_field) {

#ifdef HAVE_MPI
  xchange_field( phi );
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel shared(xi,phi,gauge_field)
{
#endif

  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int index_s = 0; index_s < VOLUME; index_s++) {

      xi_ = xi + _GSI(index_s);

      _fv_eq_zero(xi_);

      /* Negative t-direction. */
      phi_ = phi + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(g_idn[index_s][0], 0);

      _fv_eq_cm_dag_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive t-direction. */
      phi_ = phi + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(index_s, 0);

      _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative x-direction. */
      phi_ = phi + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(g_idn[index_s][1], 1);

      _fv_eq_cm_dag_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive x-direction. */
      phi_ = phi + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(index_s, 1);

      _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative y-direction. */
      phi_ = phi + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(g_idn[index_s][2], 2);

      _fv_eq_cm_dag_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Positive y-direction. */
      phi_ = phi + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(index_s, 2);

      _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative z-direction. */
      phi_ = phi + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(g_idn[index_s][3], 3);

      _fv_eq_cm_dag_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive z-direction. */
      phi_ = phi + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);
 
      U_ = gauge_field + _GGI(index_s, 3);

      _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Multiplication with -1/2 */
      _fv_ti_eq_re(xi_, -0.5);

  }  /* end of loop on index_s */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  
}  /* end of Hopping */


/********************
 *Q_phi
 *- computes xi = Q phi, where Q is the tm Dirac operator 
 *- should be compatible with Carsten's implementation of the Dirac operator 
 *  in hmc's invert programme
 *- Q is written as Q = m_0 + i mu gamma_5 + ...
 *  (not in the hopping parameter representation)
 * - off-diagonal part as:
 *   -1/2 sum over mu [ (1 - gamma_mu) U_mu(x) phi(x+mu) + 
 *      (1 + gamma_mu) U_mu(x-mu)^+ phi(x-mu) ]
 * - diagonal part as: 1 / (2 kappa) phi(x) + i mu gamma_5 \phi(x)
*/

void Q_phi(double *xi, double *phi, double*gauge_field, const double mutm) {

  const double _1_2_kappa = 0.5 / g_kappa;
#ifdef HAVE_MPI
  xchange_field( phi );
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel  shared(xi,phi,gauge_field)
{
#endif

  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(unsigned int index_s = 0; index_s < VOLUME; index_s++) {

      xi_ = xi + _GSI(index_s);

      _fv_eq_zero(xi_);

      /* Negative t-direction. */
      phi_ = phi + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(g_idn[index_s][0], 0);

      _fv_eq_cm_dag_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive t-direction. */
      phi_ = phi + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(index_s, 0);

      _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Negative x-direction. */
      phi_ = phi + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(g_idn[index_s][1], 1);

      _fv_eq_cm_dag_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive x-direction. */
      phi_ = phi + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(index_s, 1);

      _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative y-direction. */
      phi_ = phi + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(g_idn[index_s][2], 2);

      _fv_eq_cm_dag_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Positive y-direction. */
      phi_ = phi + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(index_s, 2);

      _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative z-direction. */
      phi_ = phi + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = gauge_field + _GGI(g_idn[index_s][3], 3);

      _fv_eq_cm_dag_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive z-direction. */
      phi_ = phi + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);
 
      U_ = gauge_field + _GGI(index_s, 3);

      _fv_eq_cm_ti_fv(spinor2, U_, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Multiplication with -1/2. */
      _fv_ti_eq_re(xi_, -0.5);

      /* Diagonal elements. */

      phi_ = phi + _GSI(index_s);
		    
      _fv_eq_fv_ti_re(spinor1, phi_, _1_2_kappa);
      _fv_pl_eq_fv(xi_, spinor1);
		    
      _fv_eq_gamma_ti_fv(spinor1, 5, phi_);
      _fv_eq_fv_ti_im(spinor2, spinor1, mutm);
      _fv_pl_eq_fv(xi_, spinor2);

  }  /* end of loop on index_s */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of Q_phi */


/***********************************************************
 * M_eo and M_oe
 *   input: r, EO = 0/ 1 for eo / oe
 *   output: s
 ***********************************************************/
void Hopping_eo(double *s, double *r, double *gauge_field, int EO) {

  const unsigned int N  = VOLUME / 2;
  const unsigned int N2 = (VOLUME+RAND) / 2;

#ifdef HAVE_MPI
  /* if EO = 0 => M_eo => r is odd  => call with 1-EO = 1 for odd  field argument */
  /* if EO = 1 => M_oe => r is even => call with 1-EO = 0 for even field argument */
  xchange_eo_field (r, 1-EO);
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,gauge_field,EO)
{
#endif
  unsigned int ix_fwd, ix_bwd;
  unsigned int ix, ix_lexic;
  double *U_fwd = NULL, *U_bwd = NULL;
  double sp1[24], sp2[24];
  double *s_ = NULL, *r_fwd_ = NULL, *r_bwd_ = NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    
    s_ = s + _GSI(ix);
      
    _fv_eq_zero(s_);

    /* ix is an even / odd point */
    ix_lexic = g_eo2lexic[ix + (unsigned int)(EO * N2)];

    /* =========================================== */
    /* =============== direction 0 =============== */
    U_fwd = gauge_field+_GGI(ix_lexic,0);

    U_bwd = gauge_field+_GGI( g_idn[ix_lexic][0],0);

    /* ix_fwd and ix_bwd are odd / even points */
    ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][0] ];
    ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][0] ];

    r_fwd_ = r + _GSI(ix_fwd);
    r_bwd_ = r + _GSI(ix_bwd);

    /* s += U_fwd ( 1 - g0 ) r_fwd */
    _fv_eq_gamma_ti_fv(sp1, 0, r_fwd_);
    _fv_eq_fv_mi_fv(sp2, r_fwd_, sp1);
    _fv_eq_cm_ti_fv(sp1, U_fwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* s += U_bwd^+ ( 1 + g0 ) r_bwd */
    _fv_eq_gamma_ti_fv(sp1, 0, r_bwd_);
    _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
    _fv_eq_cm_dag_ti_fv(sp1, U_bwd, sp2);
    _fv_pl_eq_fv(s_, sp1);


    /* =========================================== */
    /* =============== direction 1 =============== */
    U_fwd = gauge_field+_GGI(ix_lexic,1);

    U_bwd = gauge_field+_GGI( g_idn[ix_lexic][1],1);

    /* ix_fwd and ix_bwd are odd points */
    ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][1] ];
    ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][1] ];

    r_fwd_ = r + _GSI(ix_fwd);
    r_bwd_ = r + _GSI(ix_bwd);

    /* s += U_fwd ( 1 - g1 ) r_fwd */
    _fv_eq_gamma_ti_fv(sp1, 1, r_fwd_);
    _fv_eq_fv_mi_fv(sp2, r_fwd_, sp1);
    _fv_eq_cm_ti_fv(sp1, U_fwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* s += U_bwd^+ ( 1 + g1 ) r_bwd */
    _fv_eq_gamma_ti_fv(sp1, 1, r_bwd_);
    _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
    _fv_eq_cm_dag_ti_fv(sp1, U_bwd, sp2);
    _fv_pl_eq_fv(s_, sp1);
    
    /* =========================================== */
    /* =============== direction 2 =============== */
    U_fwd = gauge_field+_GGI(ix_lexic,2);

    U_bwd = gauge_field+_GGI( g_idn[ix_lexic][2],2);

    /* ix_fwd and ix_bwd are odd points */
    ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][2] ];
    ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][2] ];

    r_fwd_ = r + _GSI(ix_fwd);
    r_bwd_ = r + _GSI(ix_bwd);

    /* s += U_fwd ( 1 - g2 ) r_fwd */
    _fv_eq_gamma_ti_fv(sp1, 2, r_fwd_);
    _fv_eq_fv_mi_fv(sp2, r_fwd_, sp1);
    _fv_eq_cm_ti_fv(sp1, U_fwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* s += U_bwd^+ ( 1 + g2 ) r_bwd */
    _fv_eq_gamma_ti_fv(sp1, 2, r_bwd_);
    _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
    _fv_eq_cm_dag_ti_fv(sp1, U_bwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* =========================================== */
    /* =============== direction 3 =============== */
    U_fwd = gauge_field+_GGI(ix_lexic,3);

    U_bwd = gauge_field+_GGI( g_idn[ix_lexic][3],3);

    /* ix_fwd and ix_bwd are odd points */
    ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][3] ];
    ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][3] ];

    r_fwd_ = r + _GSI(ix_fwd);
    r_bwd_ = r + _GSI(ix_bwd);

    /* s += U_fwd ( 1 - g3 ) r_fwd */
    _fv_eq_gamma_ti_fv(sp1, 3, r_fwd_);
    _fv_eq_fv_mi_fv(sp2, r_fwd_, sp1);
    _fv_eq_cm_ti_fv(sp1, U_fwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* s += U_bwd^+ ( 1 + g3 ) r_bwd */
    _fv_eq_gamma_ti_fv(sp1, 3, r_bwd_);
    _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
    _fv_eq_cm_dag_ti_fv(sp1, U_bwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    _fv_ti_eq_re(s_, -0.5);

  }  /* end of loop on ix over VOLUME / 2 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of Hopping_eo */


/***********************************************************
 * M_ee and M_oo
 *
 * safe, if r == s (same memory region)
 ***********************************************************/
void M_zz (double*s, double*r, double mass) {

  const double mutilde            = 2. * g_kappa * mass;
  const double one_over_two_kappa = 0.5/g_kappa;
  const unsigned int N = VOLUME/2;


#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r, mass)
{
#endif
  unsigned int ix;
  double *s_= NULL, *r_ = NULL;
  double sp1[24];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    s_ = s + _GSI(ix);
    r_ = r + _GSI(ix);

    /* sp1 = i mass r_ */
    _fv_eq_fv_ti_im(sp1, r_, mutilde);
    /* sp1 *= g5 */
    _fv_ti_eq_g5(sp1);
    /* s_ = r_ */
    _fv_eq_fv(s_, r_);
    /* s_ <- s_ + sp1 = r_ + i mu g5 r_ = (1 + i mu g5) r_ */
    _fv_pl_eq_fv(s_, sp1);
    /* s_ *= 1/2kappa */
    _fv_ti_eq_re(s_, one_over_two_kappa);

  }  /* end of loop in ix over VOLUME/2 */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of M_zz */

/********************************************************
 * M^-1, same for even and odd
 *
 * safe, if s = r
 ********************************************************/
void M_zz_inv (double*s, double*r, double mass) {

  const double mutilde = 2. * g_kappa * mass;
  const double norm    =  2.*g_kappa / (1.+ mutilde*mutilde);
  const unsigned int N = VOLUME/2;
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) shared(s,r,mass)
{
#endif

  unsigned int ix;
  double *s_= NULL, *r_ = NULL;
  double sp1[24], sp2[24];

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    s_ = s + _GSI(ix);
    r_ = r + _GSI(ix);

    /* sp2 = r_ */
    _fv_eq_fv(sp2, r_);
    /* sp1 = g5 r_ */
    _fv_eq_gamma_ti_fv(sp1, 5, sp2);
    /* s_ = i mass sp1 */
    _fv_eq_fv_ti_im(s_, sp1, -mutilde);
    /* s_ += (1 - i mass g5) r_ */
    _fv_pl_eq_fv(s_, sp2);
    /* s_ *= 2kappa / (1 + mutilde^2) */
    _fv_ti_eq_re(s_, norm );

  }  /* end of loop in ix over VOLUME/2 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of M_zz_inv */

/***********************************************************
 * C_oo
 *   input: r (remains unchanged), gauge_field, mass
 *   output: s (changed)
 *   work space: s_aux (changed)
 ***********************************************************/
void C_oo (double*s, double*r, double *gauge_field, double mass, double *s_aux) {

  const unsigned int N = VOLUME / 2;
  const size_t sizeof_field = _GSI(N) * sizeof(double);

  /* s_aux = <- r */
  memcpy(s_aux, r, sizeof_field);
  /* s <- M_eo s_aux */
  Hopping_eo(s, s_aux, gauge_field, 0);
  /* s <- M^-1 s */
  M_zz_inv(s, s, mass);

  /* xchange before next application of M_oe */
  /* NOTE: s exchanged as even field */
  /* s_aux <- s */
  memcpy(s_aux, s, sizeof_field);
  /* s <- M_oe s_aux = M_oe M^-1 M_eo r */
  Hopping_eo(s, s_aux, gauge_field, 1);
  memcpy(s_aux, s, sizeof_field);

  /* s = M_oo r */
  M_zz(s, r, mass);
  /* s <- s - s_aux = M r - M_oe M^-1 M_eo r */
  spinor_field_eq_spinor_field_mi_spinor_field(s, s, s_aux, N);
  /* s <- g5 s */
  g5_phi(s, N);
 
}  /* end of C_oo */

/***********************************************************
 * apply Dirac full Dirac operator on eve-odd decomposed
 *   field
 ***********************************************************/
void Q_phi_eo (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux) {

  unsigned int N = VOLUME / 2;

  /* e_new = M_ee e_old + M_eo o_old */
  Hopping_eo(e_new, o_old, gauge_field, 0);
  /* aux = M_ee e_old */
  M_zz (aux, e_old, mass);

  /* e_new = e_new + aux = M_ee e_old  M_eo_o_old */
  spinor_field_pl_eq_spinor_field( e_new, aux, N);

  /* o_new = M_oo o_old + M_oe e_old */
  Hopping_eo(o_new, e_old, gauge_field, 1);
  /* aux = M_oo o_old*/
  M_zz (aux, o_old, mass);
  /* o_new  = o_new + aux = M_oe e_old + M_oo o_old */
  spinor_field_pl_eq_spinor_field( o_new, aux, N);

}  /* end of Q_phi_eo */

/********************************************************************
 * ( g5 M_ee & 0 )
 * ( g5 M_oe & 1 )
 *
 * safe, if e_new = e_old and / or o_new = o_old
 ********************************************************************/
void Q_eo_SchurDecomp_A (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux) {

  const unsigned int N = VOLUME / 2;
  const size_t  sizeof_field = _GSI(N) * sizeof(double);

  unsigned int ix;
  size_t offset;
  double *o_new_=NULL, *e_new_=NULL, *o_old_=NULL;

  /* aux <- e_old */
  memcpy(aux, e_old, sizeof_field);
  /* o_new = M_oe aux = M_oe e_old */
  Hopping_eo(e_new, aux, gauge_field, 1);
  /* o_new <- g5 e_new + o_old  = g5 M_oe e_old + o_old */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,offset, o_new_, e_new_, o_old_) shared(o_new,o_old,e_new)
#endif
  for(ix=0; ix<N; ix++) {
    offset = _GSI(ix);
    o_new_ = o_new + offset;
    e_new_ = e_new + offset;
    o_old_ = o_old + offset;
    _fv_ti_eq_g5(e_new_);
    _fv_eq_fv_pl_fv( o_new_, o_old_, e_new_);
  }

  /* e_new = M_zz aux = M_zz e_old */
  M_zz (e_new, aux, mass);
  /* e_new = g5 aux */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,e_new_) shared(e_new)
#endif
  for(ix=0; ix<N; ix++) {
    e_new_ = e_new + _GSI(ix);
    _fv_ti_eq_g5(e_new_);
  }

}  /* end of Q_SchurDecomp_A */

/********************************************************************
 * ( M_ee^-1 g5 & 0 )
 * ( -g5 M_oe M_ee^-1 g5 & 1 )
 *
 * e_new, o_new, e_old, o_old do not need halo sites
 * aux needs halo sites
 *
 * safe, if e_new = e_old and/or o_new = o_old
 ********************************************************************/
void Q_eo_SchurDecomp_Ainv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux) {

  const unsigned int N = VOLUME / 2;
  const size_t size_of_spinor_point = 24*sizeof(double);

  unsigned int ix, iix;
  double spinor1[24];

#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,iix) shared(aux)
#endif
  /* aux = g5 e_old */
  for(ix=0; ix<N; ix++) {
    iix = _GSI(ix);
    _fv_eq_gamma_ti_fv(aux + iix, 5, e_old + iix );
  }

  /* aux <- M_ee^-1 aux = M_ee^-1 g5 e_old */
  M_zz_inv (aux, aux, mass);

  /* e_new = M_oe aux  */
  Hopping_eo(e_new, aux, gauge_field, 1);

#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,iix,spinor1) shared(e_new, o_new, o_old)
#endif
  /* o_new = -g5 o_new + o_old */
  for(ix=0; ix<N; ix++) {
    iix = _GSI(ix);
    _fv_eq_gamma_ti_fv(spinor1, 5, e_new+iix);
    _fv_eq_fv_mi_fv( o_new+iix, o_old+iix, spinor1 );
  }

  memcpy(e_new, aux, N*size_of_spinor_point);

}  /* end of Q_SchurDecomp_Ainv */



/********************************************************************
 * ( 1 & M_ee^(-1) M_eo )
 * ( 0 &        C       )
 ********************************************************************/
void Q_eo_SchurDecomp_B (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux) {

  unsigned int N = VOLUME / 2;
  
  /* aux = M_eo o_old */
  Hopping_eo(aux, o_old, gauge_field, 0);
  /* e_new = M_ee^(-1) aux */
  M_zz_inv(e_new, aux, mass);
  /* e_new += e_old */
  spinor_field_pl_eq_spinor_field( e_new, e_old, N);

  /* o_new = C_oo o_old */
  C_oo (o_new, o_old, gauge_field, mass, aux);

}  /* end of Q_SchurDecomp_B */

/********************************************************************
 * ( 1 & -M_ee^(-1) M_eo 2kappa )
 * ( 0 &        2kappa          )
 *
 * safe, if e_new = e_old or o_new = o_old
 ********************************************************************/
void Q_eo_SchurDecomp_Binv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux) {

  const double twokappa = 2. * g_kappa;
  const unsigned int N = VOLUME / 2;
  const size_t sizeof_eo_spinor_field = _GSI(N) * sizeof(double);

  spinor_field_eq_spinor_field_ti_re (aux, o_old, twokappa, N);

  /* o_new = M_eo aux */
  Hopping_eo(o_new, aux, gauge_field, 0);
  /* o_new = M_ee^(-1) o_new */
  M_zz_inv(o_new, o_new, mass);
  /* e_new = e_old - o_new */
  spinor_field_eq_spinor_field_pl_spinor_field_ti_re(e_new, e_old, o_new, -1, N);
  /* o_new <- aux */
  memcpy(o_new, aux, sizeof_eo_spinor_field);

}  /* end of Q_SchurDecomp_Binv */

/********************************************************************
 * X_eo    = -M_ee^-1 M_eo,    mu > 0
 * Xbar_eo = -Mbar_ee^-1 M_eo, mu < 0
 * the input field is always odd, the output field is always even
 * even does not need halo sites
 * odd needs halo sites
 ********************************************************************/
void X_eo (double *even, double *odd, double mu, double *gauge_field) {

  const unsigned int N = VOLUME/2;
  const double mutilde = 2. * g_kappa * mu;
  const double a_re = -2. * g_kappa / ( 1 + mutilde * mutilde);
  const double a_im = -a_re * mutilde;

  /* M_eo */
  Hopping_eo(even, odd, gauge_field, 0);

//  M_zz_inv(even, even, mu);
//  spinor_field_ti_eq_re(even, -1., N);

#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) shared(even, odd)
{
#endif
  double *ptr, sp[24];
  unsigned int ix;
  /* -M_ee^-1 */
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    ptr = even + _GSI(ix);
    _fv_eq_fv(sp, ptr);
    _fv_eq_a_pl_ib_g5_ti_fv(ptr, sp, a_re, a_im);
  }  /* end of loop in ix = 0, N-1 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of X_eo */

/********************************************************************
 * C_with_Xeo
 * - apply C = g5 M_oo + g5 M_oe X_eo
 ********************************************************************/
void C_with_Xeo (double *r, double *s, double *gauge_field, double mu, double *r_aux) {

  const unsigned int N = VOLUME / 2;
  const double a_re = 1./(2. * g_kappa);
  const double a_im = mu;
  unsigned int ix, iix;

  X_eo (r_aux, s, mu, gauge_field);
  Hopping_eo(r, r_aux, gauge_field, 1);

#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,iix) shared(r,s)
#endif
  for(ix=0; ix<N; ix++) {
    iix = _GSI(ix);
    _fv_ti_eq_g5 (r+iix);
    _fv_pl_eq_a_g5_pl_ib_ti_fv(r+iix, s+iix, a_re, a_im);
  }

}  /* C_with_Xeo */

/********************************************************************
 * C_from_Xeo
 * - apply C = g5 M_oo + g5 M_oe X_eo
 *
 * output t
 * input s = X_eo v
 *       t = v
 *       r = auxilliary
 ********************************************************************/
void C_from_Xeo (double *t, double *s, double *r, double *gauge_field, double mu) {

  const double a_re = 1./(2. * g_kappa);
  const double a_im = mu;
  const unsigned int N = VOLUME / 2;

  unsigned int ix, iix;

  /*r = M_oe s = -M_oe M_ee^-1 M_eo v */
  Hopping_eo(r, s, gauge_field, 1);

#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,iix) shared(r,t)
#endif
  /*  */
  for(ix = 0; ix < N; ix++) {
    iix = _GSI(ix);
    _fv_ti_eq_g5 (r+iix);
    _fv_pl_eq_a_g5_pl_ib_ti_fv(r+iix, t+iix, a_re, a_im);
    _fv_eq_fv(t+iix, r+iix);
  }

}  /* C_from_Xeo */


/***********************************************************
 * apply the cvc vertex structure to an even/odd spinor
 * field
 *   input: r, direction mu \in {0,1,2,3}, fbwd forward 0 / backward 1
 *   EO = 0/ 1 for eo / oe, gauge field
 *   output: s
 ***********************************************************/

void apply_cvc_vertex_eo(double *s, double *r, int mu, int fbwd, double *gauge_field, int EO) {

  const unsigned int N  = VOLUME / 2;
  const unsigned int N2 = (VOLUME+RAND) / 2;

#ifdef HAVE_MPI
  xchange_eo_field ( r , 1-EO);
#endif

  if ( fbwd == 0 ) {
    /**************************************************************
     * FORWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,gauge_field,EO)
{
#endif
    unsigned int ix, ix_lexic;
    unsigned int ix_fwd;
    double *U_fwd = NULL;
    double sp1[24], sp2[24];
    double *s_ = NULL, *r_fwd_ = NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
    
      s_ = s + _GSI(ix);
      
      _fv_eq_zero(s_);

      /* ix is an even / odd point */
      ix_lexic = g_eo2lexic[ix + (unsigned int)(EO * N2)];

      /* ============================================ */
      /* =============== direction mu =============== */
      U_fwd = gauge_field+_GGI(ix_lexic,mu);

      /* ix_fwd is odd / even points */
      ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][mu] ];

      r_fwd_ = r + _GSI(ix_fwd);

      /* s = U_fwd ( g_mu - 1 ) r_fwd */
      _fv_eq_gamma_ti_fv(sp1, mu, r_fwd_);
      _fv_eq_fv_mi_fv(sp2, sp1, r_fwd_ );
      _fv_eq_cm_ti_fv(sp1, U_fwd, sp2);
      _fv_eq_fv_ti_re(s_, sp1, 0.5);

  }  /* end of loop on ix over VOLUME / 2 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
 } else if ( fbwd == 1 ) {
    /**************************************************************
     * BACKWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,gauge_field,EO)
{
#endif
    unsigned int ix, ix_lexic;
    unsigned int ix_bwd;
    double *U_bwd = NULL;
    double sp1[24], sp2[24];
    double *s_ = NULL, *r_bwd_ = NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
    
      s_ = s + _GSI(ix);
      
      _fv_eq_zero(s_);

      /* ix is an even / odd point */
      ix_lexic = g_eo2lexic[ix + (unsigned int)(EO * N2)];

      /* ============================================ */
      /* =============== direction mu =============== */
      U_bwd = gauge_field+_GGI( g_idn[ix_lexic][mu], mu);

      /* ix_bwd is odd / even point */
      ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][mu] ];

      r_bwd_ = r + _GSI(ix_bwd);

      /* s += U_bwd^+ ( g_mu + 1 ) r_bwd */
      _fv_eq_gamma_ti_fv(sp1, mu, r_bwd_);
      _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
      _fv_eq_cm_dag_ti_fv(sp1, U_bwd, sp2);
      _fv_eq_fv_ti_re(s_, sp1, 0.5);
  }  /* end of loop on ix over VOLUME / 2 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
 }  /* end of if fbwd = 0 or 1 */

}  /* end of apply_cvc_vertex_eo */

/***********************************************************
 * apply the cvc vertex structure to a full spinor * field
 *   input: r, direction mu \in {0,1,2,3},
 *   fbwd forward 0 / backward 1,
 *   gauge field
 *   output: s
 ***********************************************************/

void apply_cvc_vertex(double *s, double *r, int mu, int fbwd, double *gauge_field) {

  const unsigned int N = VOLUME;

#ifdef HAVE_MPI
  xchange_field( r );
#endif

  if ( fbwd == 0 ) {
    /**************************************************************
     * FORWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,gauge_field)
{
#endif
    unsigned int ix, ix_lexic;
    unsigned int ix_fwd;
    double *U_fwd = NULL;
    double sp1[24], sp2[24];
    double *s_ = NULL, *r_fwd_ = NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
    
      s_ = s + _GSI(ix);
      
      _fv_eq_zero(s_);

      /* ix is same as ix_lexic */
      ix_lexic = ix;

      /* ============================================ */
      /* =============== direction mu =============== */
      U_fwd = gauge_field+_GGI(ix_lexic,mu);

      /* ix_fwd */
      ix_fwd = g_iup[ix_lexic][mu];

      r_fwd_ = r + _GSI(ix_fwd);

      /* s = U_fwd ( g_mu - 1 ) r_fwd */
      _fv_eq_gamma_ti_fv(sp1, mu, r_fwd_);
      _fv_eq_fv_mi_fv(sp2, sp1, r_fwd_ );
      _fv_eq_cm_ti_fv(sp1, U_fwd, sp2);
      _fv_eq_fv_ti_re(s_, sp1, 0.5);

  }  /* end of loop on ix over VOLUME / 2 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
 } else if ( fbwd == 1 ) {
    /**************************************************************
     * BACKWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,gauge_field)
{
#endif
    unsigned int ix, ix_lexic;
    unsigned int ix_bwd;
    double *U_bwd = NULL;
    double sp1[24], sp2[24];
    double *s_ = NULL, *r_bwd_ = NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
    
      s_ = s + _GSI(ix);
      
      _fv_eq_zero(s_);

      /* ix is ix_lexic */
      ix_lexic = ix;

      /* ============================================ */
      /* =============== direction mu =============== */
      U_bwd = gauge_field+_GGI( g_idn[ix_lexic][mu], mu);

      /* ix_bwd */
      ix_bwd = g_idn[ix_lexic][mu];

      r_bwd_ = r + _GSI(ix_bwd);

      /* s += U_bwd^+ ( g_mu + 1 ) r_bwd */
      _fv_eq_gamma_ti_fv(sp1, mu, r_bwd_);
      _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
      _fv_eq_cm_dag_ti_fv(sp1, U_bwd, sp2);
      _fv_eq_fv_ti_re(s_, sp1, 0.5);
  }  /* end of loop on ix over VOLUME / 2 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
 }  /* end of if fbwd = 0 or 1 */

}  /* end of apply_cvc_vertex */


/***********************************************************
 * apply the cvc vertex structure to an even/odd propagator
 * field
 *   input: r, direction mu \in {0,1,2,3}, fbwd forward 0 / backward 1
 *          r must have halo for exchange
 *          EO = 0/ 1 for eo / oe;  EO property of output field 
 *          (cvc vertex toggles EO property of fields )
 *          gauge field
 *   output: s
 ***********************************************************/

void apply_cvc_vertex_propagator_eo ( fermion_propagator_type *s, fermion_propagator_type *r, int mu, int fbwd, double *gauge_field, int EO) {

  const unsigned int N  = VOLUME / 2;
  const unsigned int N2 = (VOLUME+RAND) / 2;

#ifdef HAVE_MPI
  xchange_eo_propagator ( r , 1-EO, mu);
#endif

  if ( fbwd == 0 ) {
    /**************************************************************
     * FORWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,gauge_field,EO)
{
#endif
    unsigned int ix, ix_lexic;
    unsigned int ix_fwd;
    double *U_fwd = NULL;
    fermion_propagator_type fp1, fp2;
    fermion_propagator_type s_ = NULL, r_fwd_ = NULL;

    create_fp(&fp1);
    create_fp(&fp2);

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
    
      s_ = s[ix];
      
      _fp_eq_zero(s_);

      /* ix is an even / odd point */
      ix_lexic = g_eo2lexic[ix + (unsigned int)(EO * N2)];

      /* ============================================ */
      /* =============== direction mu =============== */
      U_fwd = gauge_field+_GGI(ix_lexic,mu);

      /* ix_fwd is odd / even points */
      ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][mu] ];

      r_fwd_ = r[ix_fwd];

      /* s = U_fwd ( g_mu - 1 ) r_fwd */
      _fp_eq_gamma_ti_fp(fp1, mu, r_fwd_);
      _fp_eq_fp_mi_fp(fp2, fp1, r_fwd_ );
      _fp_eq_cm_ti_fp(fp1, U_fwd, fp2);
      _fp_eq_fp_ti_re(s_, fp1, 0.5);

    }  /* end of loop on ix over VOLUME / 2 */

    free_fp(&fp1);
    free_fp(&fp2);
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
 } else if ( fbwd == 1 ) {
    /**************************************************************
     * BACKWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,gauge_field,EO)
{
#endif
    unsigned int ix, ix_lexic;
    unsigned int ix_bwd;
    double *U_bwd = NULL;
    fermion_propagator_type fp1, fp2;
    fermion_propagator_type s_ = NULL, r_bwd_ = NULL;

    create_fp(&fp1);
    create_fp(&fp2);
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
    
      s_ = s[ix];
      
      _fp_eq_zero(s_);

      /* ix is an even / odd point */
      ix_lexic = g_eo2lexic[ix + (unsigned int)(EO * N2)];

      /* ============================================ */
      /* =============== direction mu =============== */
      U_bwd = gauge_field+_GGI( g_idn[ix_lexic][mu], mu);

      /* ix_bwd is odd / even point */
      ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][mu] ];

      r_bwd_ = r[ix_bwd];

      /* s = U_bwd^+ ( g_mu + 1 ) r_bwd */
      _fp_eq_gamma_ti_fp(fp1, mu, r_bwd_);
      _fp_eq_fp_pl_fp(fp2, r_bwd_, fp1);
      _fp_eq_cm_dagger_ti_fp(fp1, U_bwd, fp2);
      _fp_eq_fp_ti_re(s_, fp1, 0.5);
    }  /* end of loop on ix over VOLUME / 2 */

    free_fp(&fp1);
    free_fp(&fp2);
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
 }  /* end of if fbwd = 0 or 1 */

}  /* end of apply_cvc_vertex_propagator_eo */

/***********************************************************
 * apply the cvc vertex structure to an even/odd propagator
 *   from the right
 * field
 *   input: r, direction mu \in {0,1,2,3}, fbwd forward 0 / backward 1
 *          r must have halo for exchange
 *          (cvc vertex toggles EO property of fields )
 *          gauge field
 *   output: s
 *
 *   safe, if r and s point to the same memory region
 ***********************************************************/
void apply_propagator_constant_cvc_vertex ( fermion_propagator_type *s, fermion_propagator_type *r, int mu, int fbwd, double U[18],  const unsigned int N ) {

  if ( fbwd == 0 ) {
    /**************************************************************
     * FORWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,mu,U)
{
#endif
    unsigned int ix;
    fermion_propagator_type fp1, fp2;
    fermion_propagator_type s_ = NULL, r_ = NULL;

    create_fp(&fp1);
    create_fp(&fp2);

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
    
      s_ = s[ix];
      r_ = r[ix];
      
      /* ============================================ */
      /* =============== direction mu =============== */

      /* s = r_ U ( g_mu - 1 ) */
      _fp_eq_fp_ti_gamma(fp1, mu, r_);
      _fp_eq_fp_mi_fp(fp2, fp1, r_ );
      _fp_eq_fp_ti_cm(fp1, U, fp2);
      _fp_eq_fp_ti_re(s_, fp1, 0.5);

    }  /* end of loop on ix over VOLUME / 2 */

    free_fp(&fp1);
    free_fp(&fp2);
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
 } else if ( fbwd == 1 ) {
    /**************************************************************
     * BACKWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,mu,U)
{
#endif
    unsigned int ix;
    fermion_propagator_type fp1, fp2;
    fermion_propagator_type s_ = NULL, r_ = NULL;

    create_fp(&fp1);
    create_fp(&fp2);
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
    
      s_ = s[ix];
      r_ = r[ix];
      
      /* ============================================ */
      /* =============== direction mu =============== */

      /* s = r_ U^+ ( g_mu + 1 ) */
      _fp_eq_fp_ti_gamma(fp1, mu, r_ );
      _fp_eq_fp_pl_fp(fp2, r_, fp1);
      _fp_eq_fp_ti_cm_dagger(fp1, U, fp2);
      _fp_eq_fp_ti_re(s_, fp1, 0.5);
    }  /* end of loop on ix over VOLUME / 2 */

    free_fp(&fp1);
    free_fp(&fp2);
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
 }  /* end of if fbwd = 0 or 1 */

}  /* end of apply_cvc_vertex_propagator_eo */


/********************************************************************
 * prop and source full spinor fields
 ********************************************************************/
int Q_invert (double*prop, double*source, double*gauge_field, double mass, int op_id) {
#ifdef HAVE_TMLQCD_LIBWRAPPER
  const size_t sizeof_eo_spinor_field_with_halo = _GSI(VOLUME+RAND)/2;

  int exitstatus;
  double *eo_spinor_work[3];

  eo_spinor_work[0]  = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  eo_spinor_work[1]  = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  eo_spinor_work[2]  = (double*)malloc( sizeof_eo_spinor_field_with_halo );

  spinor_field_lexic2eo (source, eo_spinor_work[0], eo_spinor_work[1] );

  Q_eo_SchurDecomp_Ainv (eo_spinor_work[0], eo_spinor_work[1], eo_spinor_work[0], eo_spinor_work[1], gauge_field, mass, eo_spinor_work[2]);
  
  // FIXME: this function does not exist in tmLQCD 
  exitstatus = 0;//tmLQCD_invert_eo(eo_spinor_work[2], eo_spinor_work[1], op_id);
  if(exitstatus != 0) {
    fprintf(stderr, "[Q_clover_invert] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
    return(1);
  }
  Q_eo_SchurDecomp_Binv (eo_spinor_work[0], eo_spinor_work[2], eo_spinor_work[0], eo_spinor_work[2], gauge_field, mass, eo_spinor_work[1]);
  spinor_field_eo2lexic (prop, eo_spinor_work[0], eo_spinor_work[2] );
  free( eo_spinor_work[0] );
  free( eo_spinor_work[1] );
  free( eo_spinor_work[2] );

  return(0);
#else
  if( g_cart_id == 0 ) fprintf(stderr, "[Q_invert] Error, no inverter\n");
  return(2);
#endif
}  /* Q_invert */


/***********************************************************
 * apply cov. derivative to a spinor field
 *
 *   IN: r_in --- source spinor
 *   IN: mu   --- direction \in {0,1,2,3}
 *   IN: fbwd --- forward 0 / backward 1,
 *   IN: gauge field
 *   OUT: s   --- target spinor
 *
 *   safe, if s == r_in
 ***********************************************************/

int spinor_field_eq_cov_deriv_spinor_field ( double * const s, double * const r_in, int const mu, int const fbwd, double * const gauge_field ) {

  const unsigned int N = VOLUME;

  double * r = (double*) malloc ( _GSI( (VOLUME+RAND) ) * sizeof(double) );
  if ( r == NULL ) {
    fprintf ( stderr, "[spinor_field_eq_cov_deriv_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__ );
    return ( 1 );
  }
  memcpy ( r, r_in, _GSI(VOLUME)*sizeof(double) );

#ifdef HAVE_MPI
  xchange_field( r );
#endif

  if ( fbwd == 0 ) {
    /**************************************************************
     * FORWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int ix = 0; ix < N; ix++ ) {
    
      double * const r_ = r + _GSI(ix);

      double * const s_ = s + _GSI(ix);
      
      /* ============================================ */
      /* =============== direction mu =============== */
      double * const U_fwd = gauge_field+_GGI(ix,mu);

      /* ix_fwd */
      unsigned int const ix_fwd = g_iup[ix][mu];

      double * const r_fwd_ = r + _GSI(ix_fwd);

      /* s_ = U_fwd r_fwd_ */
      _fv_eq_cm_ti_fv( s_, U_fwd, r_fwd_ );
      /* s_ = s_ - r_ */
      _fv_mi_eq_fv( s_, r_ );

  }  /* end of loop on ix over VOLUME */

 } else if ( fbwd == 1 ) {
    /**************************************************************
     * BACKWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int ix = 0; ix < N; ix++ ) {
    
      double * const r_ = r + _GSI(ix);

      double * const s_ = s + _GSI(ix);
      
      /* ============================================ */
      /* =============== direction mu =============== */
      double * const U_bwd = gauge_field+_GGI( g_idn[ix][mu], mu);

      /* ix_bwd */
      unsigned int const ix_bwd = g_idn[ix][mu];

      double * const r_bwd_ = r + _GSI(ix_bwd);

      /* s_ = U_bwd^+ r_bwd_ */
      _fv_eq_cm_dag_ti_fv( s_, U_bwd, r_bwd_ );
      /* s_ = r_ - s_ */
      _fv_eq_fv_mi_fv ( s_, r_, s_ )

  }  /* end of loop on ix over VOLUME */

 }  /* end of if fbwd = 0 or 1 */

  free ( r );
  return ( 0 );
}  /* end of spinor_field_eq_cov_deriv_spinor_field */

/***********************************************************/
/***********************************************************/

/***********************************************************
 * apply cov. displacement to a spinor field
 *
 *   IN: r_in --- source spinor
 *   IN: mu   --- direction \in {0,1,2,3}
 *   IN: fbwd --- forward 0 / backward 1,
 *   IN: gauge field
 *   OUT: s   --- target spinor
 *
 *   safe, if s == r_in
 ***********************************************************/

int spinor_field_eq_cov_displ_spinor_field ( double * const s, double * const r_in, int const mu, int const fbwd, double * const gauge_field ) {

  const unsigned int N = VOLUME;

  double * r = (double*) malloc ( _GSI( (VOLUME+RAND) ) * sizeof(double) );
  if ( r == NULL ) {
    fprintf ( stderr, "[spinor_field_eq_cov_displ_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__ );
    return ( 1 );
  }
  memcpy ( r, r_in, _GSI(VOLUME)*sizeof(double) );

#ifdef HAVE_MPI
  xchange_field( r );
#endif

  if ( fbwd == 0 ) {
    /**************************************************************
     * FORWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int ix = 0; ix < N; ix++ ) {
    
      double * const s_ = s + _GSI(ix);
      
      /* ============================================ */
      /* =============== direction mu =============== */
      double * const U_fwd = gauge_field+_GGI(ix,mu);

      /* ix_fwd */
      unsigned int const ix_fwd = g_iup[ix][mu];

      double * const r_fwd_ = r + _GSI(ix_fwd);

      /* s_ = U_fwd r_fwd_ */
      _fv_eq_cm_ti_fv( s_, U_fwd, r_fwd_ );

  }  /* end of loop on ix over VOLUME */

 } else if ( fbwd == 1 ) {
    /**************************************************************
     * BACKWARD
     **************************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( unsigned int ix = 0; ix < N; ix++ ) {
    
      double * const s_ = s + _GSI(ix);
      
      /* ============================================ */
      /* =============== direction mu =============== */
      double * const U_bwd = gauge_field+_GGI( g_idn[ix][mu], mu);

      /* ix_bwd */
      unsigned int const ix_bwd = g_idn[ix][mu];

      double * const r_bwd_ = r + _GSI(ix_bwd);

      /* s_ = U_bwd^+ r_bwd_ */
      _fv_eq_cm_dag_ti_fv( s_, U_bwd, r_bwd_ );

  }  /* end of loop on ix over VOLUME */

 }  /* end of if fbwd = 0 or 1 */

  free ( r );
  return ( 0 );
}  /* end of spinor_field_eq_cov_displ_spinor_field */

/***********************************************************/
/***********************************************************/

}  /* end of namespace cvc */
