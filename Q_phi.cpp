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
#ifndef HAVE_OPENMP
void Q_phi_tbc(double *xi, double *phi) {
  int it, ix, iy, iz;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;

  double _1_2_kappa = 0.5 / g_kappa;

  for(it = 0; it < T; it++) {
  for(ix = 0; ix < LX; ix++) {
  for(iy = 0; iy < LY; iy++) {
  for(iz = 0; iz < LZ; iz++) {

      index_s = g_ipt[it][ix][iy][iz];

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
  }
  }
  }
/****************************************
 * call xchange_field in calling process
 *
#ifdef HAVE_MPI
  xchange_field(xi);
#endif
 *
 ****************************************/

}
#else
void Q_phi_tbc(double *xi, double *phi) {
  int ix;
  int index_s; 
  double *xi_, *phi_, *U_;

  double _1_2_kappa = 0.5 / g_kappa;

// #pragma omp parallel for private(ix,index_s,phi_,xi_,spinor1,spinor2,U_,SU3_1)  shared(phi,xi,T,_1_2_kappa,g_gauge_field,g_iup,g_idn, co_phase_up)
#pragma omp parallel private(ix,index_s,phi_,xi_,U_)  shared(phi,xi,_1_2_kappa,g_gauge_field,g_iup,g_idn, co_phase_up, VOLUME)
{
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  int threadid = omp_get_thread_num();
  int num_threads = omp_get_num_threads();

  for(ix = threadid; ix < VOLUME; ix+=num_threads)
  // for(ix = 0; ix < g_num_threads; ix++)
  {
#if 0
      // TEST
      index_s = omp_get_thread_num();
      fprintf(stdout, "# [Q_phi_tbc] thread%.4d number of threads = %d\n", index_s, omp_get_num_threads());
      fprintf(stdout, "# [Q_phi_tbc] thread%.4d address of spinor1 = %lu\n", index_s, spinor1);
      fprintf(stdout, "# [Q_phi_tbc] thread%.4d address of spinor2 = %lu\n", index_s, spinor2);

      // TEST
      fprintf(stdout, "# [Q_phi_tbc] thread%.4d ix = %d\n", threadid, ix);
#endif

      index_s = ix;

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

}  // end of parallel region
}
#endif  /* of ifndef HAVE_OPENMP */
#if 0
/* this has been moved to cvc_utils.cpp */
void g5_phi(double *phi, unsigned int N) {
#ifdef HAVE_OPENMP
#pragma omp parallel shared(phi, N)
{
#endif
  int ix;
  double spinor1[24];
  double *phi_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
      phi_ = phi + _GSI(ix);
/*
      _fv_eq_gamma_ti_fv(spinor1, 5, phi_);
      _fv_eq_fv(phi_, spinor1);
*/
      _fv_ti_eq_g5(phi_);
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
}  /* end of g5_phi */
#endif

/**********************************************************************
 * Hopping 
 * - computes xi = H phi, where H is the Dirac Wilson hopping matrix
 * - off-diagonal part as:
 *   - kappa sum over mu [ (1 - gamma_mu) U_mu(x) phi(x+mu) + 
 *      (1 + gamma_mu) U_mu(x-mu)^+ phi(x-mu) ]
 * - in the hopping parameter representation
 * - _NO_ field exchange in Hopping
 **********************************************************************/

void Hopping(double *xi, double *phi) {
  int it, ix, iy, iz;
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  int index_s; 
  double *xi_, *phi_, *U_;

  for(it = 0; it < T; it++) {
  for(ix = 0; ix < LX; ix++) {
  for(iy = 0; iy < LY; iy++) {
  for(iz = 0; iz < LZ; iz++) {

      index_s = g_ipt[it][ix][iy][iz];

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

      /* Multiplication with -kappa (hopping parameter repr.) */
      /* _fv_ti_eq_re(xi_, -g_kappa); */
      _fv_ti_eq_re(xi_, -0.5);

  }
  }
  }
  }
  
}  /* end of Hopping */

/*****************************************************
 * gamma5_BH4_gamma5 
 * - calculates xi = gamma5 (B^+ H)^4 gamma5 phi
 * - _NOTE_ : B^+H is indep. of repr.
 *****************************************************/

void gamma5_BdagH4_gamma5 (double *xi, double *phi, double *work) {

  int ix, i;
  double spinor1[24];
  double _2kappamu = 2 * g_kappa * g_mu;

  /* multiply original source (phi) with gamma_5, save in xi */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(&xi[_GSI(ix)], 5, &phi[_GSI(ix)]);
  }

  /* apply B^+ H four times 
   * status: source = xi from last step, dest = work
   */
  for(i=0; i<2; i++) {
    /* apply the hopping matrix */
    xchange_field(xi);
    Hopping(work, xi);
    /* apply B^+ */
    mul_one_pm_imu_inv(work, -1., VOLUME);

    /* apply the hopping matrix */
    xchange_field(work);
    Hopping(xi, work);
    /* apply B^+ */
    mul_one_pm_imu_inv(xi, -1., VOLUME);

  }

  /* final step: multiply with gamma_5 and */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(spinor1, 5, &xi[_GSI(ix)]);
    _fv_eq_fv(&xi[_GSI(ix)], spinor1);
  }
}

/***********************************
 * mul_one_pm_imu_inv
 *
 ***********************************/
void mul_one_pm_imu_inv (double *phi, double sign, int V) {

  int ix;
  double spinor1[24], spinor2[24];
  double _2kappamu = 2. * g_kappa * g_mu;
  double norminv = 1. / (1. + _2kappamu*_2kappamu);

  for(ix=0; ix<V; ix++) {
    _fv_eq_gamma_ti_fv(spinor1, 5, &phi[_GSI(ix)]);
    _fv_eq_fv_ti_im(spinor2, spinor1, -sign*_2kappamu);
    _fv_eq_fv_pl_fv(spinor1, &phi[_GSI(ix)], spinor2);
    _fv_eq_fv_ti_re(&phi[_GSI(ix)], spinor1, norminv);
  }
}

/*****************************************************
 * BH3
 * - calculates xi = (B^+ H)^3 phi
 * - B = A^{-1}, A = 1 + i gamma_5 \tilde{\mu}
 * - _NOTE_ : BH is indep. of repr.
 *****************************************************/

void BH3 (double *xi, double *phi) {

  /*************************************************
   * apply B H three times 
   *************************************************/
  /* 1st application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 2nd application of the hopping matrix */
  Hopping(phi, xi);
  mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 3rd application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);
}

/*****************************************************
 * BH5
 * - calculates xi = (B H)^5 phi
 * - B = A^{-1}, A = 1 + i gamma_5 \tilde{\mu}
 * - _NOTE_ : BH is indep. of repr.
 *****************************************************/

void BH5 (double *xi, double *phi) {

  /*************************************************
   * apply B H three times 
   *************************************************/
  /* 1st application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 2nd application of the hopping matrix */
  Hopping(phi, xi);
  mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 3rd application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 4th application of the hopping matrix */
  Hopping(phi, xi);
  mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 5th application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

}

/*****************************************************
 * BH
 * - calculates xi = (B H) phi
 * - B = A^{-1}, A = 1 + i gamma_5 \tilde{\mu}
 * - _NOTE_ : BH is indep. of repr.
 *****************************************************/

void BH (double *xi, double *phi) {

  /*************************************************
   * apply B H 
   *************************************************/
  /* application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);
}

/*****************************************************
 * BH2
 * - calculates phi = (B H)^2 phi via xi
 * - B = A^{-1}, A = 1 + i gamma_5 \tilde{\mu}
 * - _NOTE_ : BH is indep. of repr.
 *****************************************************/

void BH2 (double *xi, double *phi) {

  /*************************************************
   * apply B H 
   *************************************************/
  /* application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);
  Hopping(phi, xi);
  mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);
}

/*****************************************************
 * BH7
 * - calculates xi = (B H)^7 phi
 * - B = A^{-1}, A = 1 + i gamma_5 \tilde{\mu}
 * - _NOTE_ : BH is indep. of repr.
 *****************************************************/

void BH7 (double *xi, double *phi) {

  /*************************************************
   * apply B H sevev times 
   *************************************************/
  /* 1st application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 2nd application of the hopping matrix */
  Hopping(phi, xi);
  mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 3rd application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 4th application of the hopping matrix */
  Hopping(phi, xi);
  mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 5th application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 6th application of the hopping matrix */
  Hopping(phi, xi);
  mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 7th application of the hopping matrix */
  Hopping(xi, phi);
  mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

}

/*****************************************************
 * BHn
 * - calculates xi = (B H)^n phi
 * - B = A^{-1}, A = 1 + i gamma_5 \tilde{\mu}
 * - _NOTE_ : BH is indep. of repr.
 *****************************************************/

void BHn (double *xi, double *phi, int n) {

  int m, r, i;

  /*************************************************
   * apply B H n times
   *************************************************/
  m = n / 2;
  r = n % 2;

  for(i=0; i<m; i++) {
    Hopping(xi, phi);
    mul_one_pm_imu_inv(xi, +1., VOLUME);
    xchange_field(xi);

    Hopping(phi, xi);
    mul_one_pm_imu_inv(phi, +1., VOLUME);
    xchange_field(phi);
  }

  if(r==1) {
    Hopping(xi, phi);
    mul_one_pm_imu_inv(xi, +1., VOLUME);
    xchange_field(xi);
  } else {
    memcpy((void*)xi, (void*)phi, 24*VOLUMEPLUSRAND*sizeof(double));
  }

}

/**************************************************************************************
 * Qf5 
 * - computes xi = gamma_5 Q_f phi 
 *   where Q_f is the tm Dirac operator with twisted boundary conditions.
 **************************************************************************************/
#ifndef HAVE_OPENMP
void Qf5(double *xi, double *phi, double mutm) {
  int it, ix, iy, iz;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24], spinor3[24];
  double *xi_, *phi_, *U_;

  double _1_2_kappa = 0.5 / g_kappa;

  for(it = 0; it < T; it++) {
  for(ix = 0; ix < LX; ix++) {
  for(iy = 0; iy < LY; iy++) {
  for(iz = 0; iz < LZ; iz++) {

      index_s = g_ipt[it][ix][iy][iz];

      xi_ = spinor3;

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
      _fv_eq_fv_ti_im(spinor2, spinor1, mutm);
      _fv_pl_eq_fv(xi_, spinor2);


      _fv_eq_gamma_ti_fv(xi+_GSI(index_s), 5, xi_);
  }
  }
  }
  }

}
#else
void Qf5(double *xi, double *phi, double mutm) {
  int ix;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24], spinor3[24];
  double *xi_, *phi_, *U_;

  double _1_2_kappa = 0.5 / g_kappa;


#pragma omp parallel for private(ix,index_s,phi_,xi_,spinor1,spinor2,U_,SU3_1)  shared(phi,xi,T,_1_2_kappa,g_gauge_field,g_iup,g_idn, co_phase_up)
  for(ix = 0; ix < VOLUME; ix++) {

      index_s = ix;

      xi_ = spinor3;

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
      _fv_eq_fv_ti_im(spinor2, spinor1, mutm);
      _fv_pl_eq_fv(xi_, spinor2);


      _fv_eq_gamma_ti_fv(xi+_GSI(index_s), 5, xi_);
  }
}
#endif

/********************
 Q_phi
 - computes xi = Q phi, where Q is the tm Dirac operator 
 - should be compatible with Carsten's implementation of the Dirac operator 
   in hmc's invert programme
 - Q is written as Q = m_0 + i mu gamma_5 + ...
   (not in the hopping parameter representation)
 - off-diagonal part as:
   -1/2 sum over mu [ (1 - gamma_mu) U_mu(x) phi(x+mu) + 
       (1 + gamma_mu) U_mu(x-mu)^+ phi(x-mu) ]
 - diagonal part as: 1 / (2 kappa) phi(x) + i mu gamma_5 \phi(x)
*/

void Q_phi(double *xi, double *phi, const double mutm) {
  int it, ix, iy, iz;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;

  double _1_2_kappa = 0.5 / g_kappa;

  for(it = 0; it < T; it++) {
  for(ix = 0; ix < LX; ix++) {
  for(iy = 0; iy < LY; iy++) {
  for(iz = 0; iz < LZ; iz++) {

      index_s = g_ipt[it][ix][iy][iz];

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
      _fv_eq_fv_ti_im(spinor2, spinor1, mutm);
      _fv_pl_eq_fv(xi_, spinor2);
  }
  }
  }
  }
/****************************************
 * call xchange_field in calling process
 *
#ifdef HAVE_MPI
  xchange_field(xi);
#endif
 *
 ****************************************/

}

/****************************************
 * same as Q_phi_tbc but withou the
 *   (i mu g5) - term
 ****************************************/

void Q_Wilson_phi_tbc(double *xi, double *phi) {
  int it, ix, iy, iz;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;

  double _1_2_kappa = 0.5 / g_kappa;

  for(it = 0; it < T; it++) {
  for(ix = 0; ix < LX; ix++) {
  for(iy = 0; iy < LY; iy++) {
  for(iz = 0; iz < LZ; iz++) {

      index_s = g_ipt[it][ix][iy][iz];

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
		    
  }}}}

}


/****************************************
 * Q_Wilson_phi without twisted boundary
 * condition;
 * explicit implementation of antiperiodicity
 * in t-direction
 ****************************************/
void Q_Wilson_phi(double *xi, double *phi) {
  int it, ix, iy, iz;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;
  double phase_neg, phase_pos;
  double _1_2_kappa = 0.5 / g_kappa;

  for(it = 0; it < T; it++) {
    phase_neg = it==0 && g_proc_coords[0]==0  ? -1. : +1.;
    phase_pos = it==T-1 && g_proc_coords[0]==g_nproc_t-1 ? -1. : +1.;
  for(ix = 0; ix < LX; ix++) {
  for(iy = 0; iy < LY; iy++) {
  for(iz = 0; iz < LZ; iz++) {

      index_s = g_ipt[it][ix][iy][iz];

      xi_ = xi + _GSI(index_s);

      _fv_eq_zero(xi_);

      /* Negative t-direction. */
      phi_ = phi + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][0], 0);

      _cm_eq_cm_ti_re(SU3_1, U_, phase_neg);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive t-direction. */
      phi_ = phi + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 0);

      _cm_eq_cm_ti_re(SU3_1, U_, phase_pos);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Negative x-direction. */
      phi_ = phi + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][1], 1);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive x-direction. */
      phi_ = phi + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 1);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative y-direction. */
      phi_ = phi + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][2], 2);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Positive y-direction. */
      phi_ = phi + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 2);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative z-direction. */
      phi_ = phi + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][3], 3);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive z-direction. */
      phi_ = phi + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);
 
      U_ = g_gauge_field + _GGI(index_s, 3);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Multiplication with -1/2. */

      _fv_ti_eq_re(xi_, -0.5);

      /* Diagonal elements. */

      phi_ = phi + _GSI(index_s);
		    
      _fv_eq_fv_ti_re(spinor1, phi_, _1_2_kappa);
      _fv_pl_eq_fv(xi_, spinor1);
		    
  }}}}

}

/*************************************************************
 * xi = gamma_5 x D_W phi
 *************************************************************/
void Q_g5_Wilson_phi(double *xi, double *phi) {
  int it, ix, iy, iz;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;
  double phase_neg, phase_pos;
  double _1_2_kappa = 0.5 / g_kappa;

  for(it = 0; it < T; it++) {
    phase_neg = it==0   ? -1. : +1.;
    phase_pos = it==T-1 ? -1. : +1.;
  for(ix = 0; ix < LX; ix++) {
  for(iy = 0; iy < LY; iy++) {
  for(iz = 0; iz < LZ; iz++) {

      index_s = g_ipt[it][ix][iy][iz];

      xi_ = xi + _GSI(index_s);

      _fv_eq_zero(xi_);

      /* Negative t-direction. */
      phi_ = phi + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][0], 0);

      _cm_eq_cm_ti_re(SU3_1, U_, phase_neg);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive t-direction. */
      phi_ = phi + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 0);

      _cm_eq_cm_ti_re(SU3_1, U_, phase_pos);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Negative x-direction. */
      phi_ = phi + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][1], 1);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive x-direction. */
      phi_ = phi + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 1);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative y-direction. */
      phi_ = phi + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][2], 2);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Positive y-direction. */
      phi_ = phi + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 2);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative z-direction. */
      phi_ = phi + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][3], 3);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive z-direction. */
      phi_ = phi + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);
 
      U_ = g_gauge_field + _GGI(index_s, 3);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Multiplication with -1/2. */

      _fv_ti_eq_re(xi_, -0.5);

      /* Diagonal elements. */

      phi_ = phi + _GSI(index_s);
		    
      _fv_eq_fv_ti_re(spinor1, phi_, _1_2_kappa);
      _fv_pl_eq_fv(xi_, spinor1);

      // multiply with g5
      _fv_eq_gamma_ti_fv(spinor1, 5, xi_);
      _fv_eq_fv(xi_, spinor1);
  }}}}

}

/****************************************
 * Q_Wilson_phi without any explicit
 * boundary conditon
 ****************************************/
void Q_Wilson_phi_nobc(double *xi, double *phi) {
  int it, ix, iy, iz;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;
  double _1_2_kappa = 0.5 / g_kappa;

  for(it = 0; it < T; it++) {
  for(ix = 0; ix < LX; ix++) {
  for(iy = 0; iy < LY; iy++) {
  for(iz = 0; iz < LZ; iz++) {

      index_s = g_ipt[it][ix][iy][iz];

      xi_ = xi + _GSI(index_s);

      _fv_eq_zero(xi_);

      /* Negative t-direction. */
      phi_ = phi + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][0], 0);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive t-direction. */
      phi_ = phi + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 0);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Negative x-direction. */
      phi_ = phi + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][1], 1);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive x-direction. */
      phi_ = phi + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 1);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative y-direction. */
      phi_ = phi + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][2], 2);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Positive y-direction. */
      phi_ = phi + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 2);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative z-direction. */
      phi_ = phi + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][3], 3);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive z-direction. */
      phi_ = phi + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);
 
      U_ = g_gauge_field + _GGI(index_s, 3);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Multiplication with -1/2. */

      _fv_ti_eq_re(xi_, -0.5);

      /* Diagonal elements. */

      phi_ = phi + _GSI(index_s);
		    
      _fv_eq_fv_ti_re(spinor1, phi_, _1_2_kappa);
      _fv_pl_eq_fv(xi_, spinor1);
		    
  }}}}

}

#ifdef HAVE_OPENMP
/****************************************
 * Q_Wilson_phi_threads
 * explicit of antiperiodic boundary
 * conditions in t-direction
 ****************************************/
void Q_Wilson_phi_threads(double *xi, double *phi) {
  int it, ix;
  int index_s; 
  double SU3_1[18];
  double spinor1[24], spinor2[24];
  double *xi_, *phi_, *U_;
  double phase_neg, phase_pos;
  double _1_2_kappa = 0.5 / g_kappa;
  unsigned int V3 = LX*LY*LZ;

#pragma omp parallel for private(it,ix,index_s,phi_,xi_,spinor1,spinor2,U_,SU3_1,phase_pos,phase_neg)  shared(phi,xi,T,V3,_1_2_kappa,g_gauge_field,g_iup,g_idn)
  for(it=0; it < T; it++) {
    phase_neg = it==0   ? -1. : +1.;
    phase_pos = it==T-1 ? -1. : +1.;

    for(ix = 0; ix < V3; ix++) {

      index_s = it*V3+ix;

      xi_ = xi + _GSI(index_s);

      _fv_eq_zero(xi_);

      /* Negative t-direction. */
      phi_ = phi + _GSI(g_idn[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][0], 0);

      _cm_eq_cm_ti_re(SU3_1, U_, phase_neg);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive t-direction. */
      phi_ = phi + _GSI(g_iup[index_s][0]);

      _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 0);

      _cm_eq_cm_ti_re(SU3_1, U_, phase_pos);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Negative x-direction. */
      phi_ = phi + _GSI(g_idn[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][1], 1);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive x-direction. */
      phi_ = phi + _GSI(g_iup[index_s][1]);

      _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 1);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative y-direction. */
      phi_ = phi + _GSI(g_idn[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][2], 2);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Positive y-direction. */
      phi_ = phi + _GSI(g_iup[index_s][2]);

      _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(index_s, 2);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);


      /* Negative z-direction. */
      phi_ = phi + _GSI(g_idn[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_pl_eq_fv(spinor1, phi_);

      U_ = g_gauge_field + _GGI(g_idn[index_s][3], 3);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Positive z-direction. */
      phi_ = phi + _GSI(g_iup[index_s][3]);

      _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
      _fv_mi(spinor1);
      _fv_pl_eq_fv(spinor1, phi_);
 
      U_ = g_gauge_field + _GGI(index_s, 3);

      _cm_eq_cm(SU3_1, U_);
      _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
      _fv_pl_eq_fv(xi_, spinor2);

      /* Multiplication with -1/2. */

      _fv_ti_eq_re(xi_, -0.5);

      /* Diagonal elements. */

      phi_ = phi + _GSI(index_s);
		    
      _fv_eq_fv_ti_re(spinor1, phi_, _1_2_kappa);
      _fv_pl_eq_fv(xi_, spinor1);

    }
  }

}
#endif



/****************************************
 * Q_DW_Wilson_phi
 * 
 * explicit implementation of antiperiodicity
 * in t-direction
 ****************************************/
void Q_DW_Wilson_4d_phi(double *xi, double *phi) {

  unsigned int ix, is, index_s; 
  unsigned int VOL3 = LX*LY*LZ;
  double xi_[24], *phi_, *U_, SU3_1[18];
  double spinor1[24], spinor2[24];
#if (defined HAVE_QUDA) && ( (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) )
  int tproc_dir = 3;
#else
  int tproc_dir = 0;
#endif
  double psign = ( g_proc_coords[tproc_dir] == g_nproc_t-1 ) ? -1. : 1.;
  double nsign = ( g_proc_coords[tproc_dir] == 0           ) ? -1. : 1.;
 
  for(is=0; is<L5; is++) {

    index_s = is * VOLUME;
  // ----------------------------------------------------------------------------
  // timeslice no. 0
  for(ix = 0; ix < VOL3; ix++) {

    _fv_eq_zero(xi_);

    // Negative t-direction
    // phi_ = phi + _GSI(_G5DI(is, g_idn[ix][0]) );
    phi_ = phi + _GSI( g_idn_5d[index_s][0] );

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI( g_idn[ix][0], 0);

    _cm_eq_cm_ti_re(SU3_1, U_, nsign);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive t-direction
    //phi_ = phi + _GSI( _G5DI(is,g_iup[ix][0]) );
    phi_ = phi + _GSI(g_iup_5d[index_s][0] );

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 0);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Negative x-direction
    // phi_ = phi + _GSI(_G5DI(is,g_idn[ix][1]) );
    phi_ = phi + _GSI(g_idn_5d[index_s][1] );

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][1], 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive x-direction
    //phi_ = phi + _GSI(_G5DI(is, g_iup[ix][1]) );
    phi_ = phi + _GSI(g_iup_5d[index_s][1] );

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative y-direction
    // phi_ = phi + _GSI(_G5DI(is,g_idn[ix][2]));
    phi_ = phi + _GSI(g_idn_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][2], 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Positive y-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][2]));
    phi_ = phi + _GSI(g_iup_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative z-direction
    // phi_ = phi + _GSI(_G5DI(is,g_idn[ix][3]));
    phi_ = phi + _GSI(g_idn_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][3], 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive z-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][3]));
    phi_ = phi + _GSI(g_iup_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);
 
    U_ = g_gauge_field + _GGI(ix, 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Multiplication with -1/2

    _fv_ti_eq_re(xi_, -0.5);

    _fv_pl_eq_fv(xi+_GSI(index_s), xi_);
    index_s++;

  }  // of loop on ix


  // ----------------------------------------------------------------------------
  // timeslices no. 1,..., T-2
  for(; ix < (T-1)*VOL3; ix++) {

    _fv_eq_zero(xi_);

    // Negative t-direction. 
    // phi_ = phi + _GSI(_G5DI(is,g_idn[ix][0]));
    phi_ = phi + _GSI(g_idn_5d[index_s][0]);

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][0], 0);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive t-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][0]));
    phi_ = phi + _GSI(g_iup_5d[index_s][0]);

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 0);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Negative x-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][1]));
    phi_ = phi + _GSI(g_idn_5d[index_s][1]);

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][1], 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive x-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][1]));
    phi_ = phi + _GSI(g_iup_5d[index_s][1]);

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative y-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][2]));
    phi_ = phi + _GSI(g_idn_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][2], 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Positive y-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][2]));
    phi_ = phi + _GSI(g_iup_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative z-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][3]));
    phi_ = phi + _GSI(g_idn_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][3], 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive z-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][3]));
    phi_ = phi + _GSI(g_iup_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);
 
    U_ = g_gauge_field + _GGI(ix, 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Multiplication with -1/2. 
    _fv_ti_eq_re(xi_, -0.5);

    _fv_pl_eq_fv(xi+_GSI(index_s), xi_);
    index_s++;
  }  // of loop on ix, intermediate timeslices

  // ----------------------------------------------------------------------------
  // timeslice no. T-1  
  for(; ix < VOLUME; ix++) {

    _fv_eq_zero(xi_);

    // Negative t-direction
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][0]));
    phi_ = phi + _GSI(g_idn_5d[index_s][0]);

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][0], 0);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive t-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][0]));
    phi_ = phi + _GSI(g_iup_5d[index_s][0]);

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 0);

    _cm_eq_cm_ti_re(SU3_1, U_, psign);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Negative x-direction
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][1]));
    phi_ = phi + _GSI(g_idn_5d[index_s][1]);

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][1], 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive x-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][1]));
    phi_ = phi + _GSI(g_iup_5d[index_s][1]);

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative y-direction
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][2]));
    phi_ = phi + _GSI(g_idn_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][2], 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Positive y-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][2]));
    phi_ = phi + _GSI(g_iup_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative z-direction
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][3]));
    phi_ = phi + _GSI(g_idn_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(g_idn[ix][3], 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive z-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][3]));
    phi_ = phi + _GSI(g_iup_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_mi(spinor1);
    _fv_pl_eq_fv(spinor1, phi_);
 
    U_ = g_gauge_field + _GGI(ix, 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Multiplication with -1/2

    _fv_ti_eq_re(xi_, -0.5);

    _fv_pl_eq_fv(xi+_GSI(index_s), xi_);
    index_s++;
  }  // of loop on ix, last timeslice

  }  // of loop on is

  return;
}

void Q_DW_Wilson_dag_4d_phi(double *xi, double *phi) {

  unsigned int ix, is, index_s; 
  unsigned int VOL3 = LX*LY*LZ;
  double xi_[24], *phi_, *U_, SU3_1[18];
  double spinor1[24], spinor2[24];
#if (defined HAVE_QUDA) && ( (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) )
  int tproc_dir = 3;
#else
  int tproc_dir = 0;
#endif
  double psign = ( g_proc_coords[tproc_dir] == g_nproc_t-1 ) ? -1. : 1.;
  double nsign = ( g_proc_coords[tproc_dir] == 0           ) ? -1. : 1.;
 
  for(is=0; is<L5; is++) {

    index_s = is * VOLUME;
  // ----------------------------------------------------------------------------
  // timeslice no. 0
  for(ix = 0; ix < VOL3; ix++) {

    _fv_eq_zero(xi_);

    // Negative t-direction
    // phi_ = phi + _GSI(_G5DI(is, g_idn[ix][0]) );
    phi_ = phi + _GSI( g_idn_5d[index_s][0] );

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_, spinor1);

    U_ = g_gauge_field + _GGI( g_idn[ix][0], 0);

    _cm_eq_cm_ti_re(SU3_1, U_, nsign);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive t-direction
    //phi_ = phi + _GSI( _G5DI(is,g_iup[ix][0]) );
    phi_ = phi + _GSI(g_iup_5d[index_s][0] );

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 0);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Negative x-direction
    // phi_ = phi + _GSI(_G5DI(is,g_idn[ix][1]) );
    phi_ = phi + _GSI(g_idn_5d[index_s][1] );

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_, spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][1], 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive x-direction
    //phi_ = phi + _GSI(_G5DI(is, g_iup[ix][1]) );
    phi_ = phi + _GSI(g_iup_5d[index_s][1] );

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative y-direction
    // phi_ = phi + _GSI(_G5DI(is,g_idn[ix][2]));
    phi_ = phi + _GSI(g_idn_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_, spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][2], 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Positive y-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][2]));
    phi_ = phi + _GSI(g_iup_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative z-direction
    // phi_ = phi + _GSI(_G5DI(is,g_idn[ix][3]));
    phi_ = phi + _GSI(g_idn_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_, spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][3], 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive z-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][3]));
    phi_ = phi + _GSI(g_iup_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_pl_eq_fv(spinor1, phi_);
 
    U_ = g_gauge_field + _GGI(ix, 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Multiplication with -1/2

    _fv_ti_eq_re(xi_, -0.5);

    _fv_pl_eq_fv(xi+_GSI(index_s), xi_);
    index_s++;

  }  // of loop on ix


  // ----------------------------------------------------------------------------
  // timeslices no. 1,..., T-2
  for(; ix < (T-1)*VOL3; ix++) {

    _fv_eq_zero(xi_);

    // Negative t-direction. 
    // phi_ = phi + _GSI(_G5DI(is,g_idn[ix][0]));
    phi_ = phi + _GSI(g_idn_5d[index_s][0]);

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_, spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][0], 0);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive t-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][0]));
    phi_ = phi + _GSI(g_iup_5d[index_s][0]);

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 0);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Negative x-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][1]));
    phi_ = phi + _GSI(g_idn_5d[index_s][1]);

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_, spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][1], 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive x-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][1]));
    phi_ = phi + _GSI(g_iup_5d[index_s][1]);

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative y-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][2]));
    phi_ = phi + _GSI(g_idn_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_, spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][2], 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Positive y-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][2]));
    phi_ = phi + _GSI(g_iup_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative z-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][3]));
    phi_ = phi + _GSI(g_idn_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_, spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][3], 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive z-direction. 
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][3]));
    phi_ = phi + _GSI(g_iup_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_pl_eq_fv(spinor1, phi_);
 
    U_ = g_gauge_field + _GGI(ix, 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Multiplication with -1/2. 
    _fv_ti_eq_re(xi_, -0.5);

    _fv_pl_eq_fv(xi+_GSI(index_s), xi_);
    index_s++;
  }  // of loop on ix, intermediate timeslices

  // ----------------------------------------------------------------------------
  // timeslice no. T-1  
  for(; ix < VOLUME; ix++) {

    _fv_eq_zero(xi_);

    // Negative t-direction
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][0]));
    phi_ = phi + _GSI(g_idn_5d[index_s][0]);

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_,spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][0], 0);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive t-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][0]));
    phi_ = phi + _GSI(g_iup_5d[index_s][0]);

    _fv_eq_gamma_ti_fv(spinor1, 0, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 0);

    _cm_eq_cm_ti_re(SU3_1, U_, psign);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Negative x-direction
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][1]));
    phi_ = phi + _GSI(g_idn_5d[index_s][1]);

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_,spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][1], 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive x-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][1]));
    phi_ = phi + _GSI(g_iup_5d[index_s][1]);

    _fv_eq_gamma_ti_fv(spinor1, 1, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 1);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative y-direction
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][2]));
    phi_ = phi + _GSI(g_idn_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_,spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][2], 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Positive y-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][2]));
    phi_ = phi + _GSI(g_iup_5d[index_s][2]);

    _fv_eq_gamma_ti_fv(spinor1, 2, phi_);
    _fv_pl_eq_fv(spinor1, phi_);

    U_ = g_gauge_field + _GGI(ix, 2);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);


    // Negative z-direction
    //phi_ = phi + _GSI(_G5DI(is,g_idn[ix][3]));
    phi_ = phi + _GSI(g_idn_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_eq_fv_mi_fv(spinor1, phi_,spinor1);

    U_ = g_gauge_field + _GGI(g_idn[ix][3], 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_dag_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Positive z-direction
    //phi_ = phi + _GSI(_G5DI(is,g_iup[ix][3]));
    phi_ = phi + _GSI(g_iup_5d[index_s][3]);

    _fv_eq_gamma_ti_fv(spinor1, 3, phi_);
    _fv_pl_eq_fv(spinor1, phi_);
 
    U_ = g_gauge_field + _GGI(ix, 3);

    _cm_eq_cm(SU3_1, U_);
    _fv_eq_cm_ti_fv(spinor2, SU3_1, spinor1);
    _fv_pl_eq_fv(xi_, spinor2);

    // Multiplication with -1/2

    _fv_ti_eq_re(xi_, -0.5);

    _fv_pl_eq_fv(xi+_GSI(index_s), xi_);
    index_s++;
  }  // of loop on ix, last timeslice

  }  // of loop on is

  return;
}

void Q_DW_Wilson_dag_5th_phi(double *xi, double *phi) {

  unsigned int ix, index_s, is; 
  double spinor1[24], spinor2[24];
  double *xi_, *phi_;

  // 5-slice no 0
  index_s = 0;
  for(ix=0;ix<VOLUME;ix++) {
    xi_  = xi  + _GSI(index_s);
    phi_ = phi + _GSI(index_s);
#if (defined RIGHTHANDED_FWD)
    _fv_pl_eq_PLi_fv_ti_re(xi_, phi+_GSI((L5-1)*VOLUME+ix), g_m0);
    _fv_mi_eq_PRe_fv(xi_, phi+_GSI(index_s+VOLUME));
#elif (defined RIGHTHANDED_BWD)  
    _fv_pl_eq_PRe_fv_ti_re(xi_, phi+_GSI((L5-1)*VOLUME+ix), g_m0);
    _fv_mi_eq_PLi_fv(xi_, phi+_GSI(index_s+VOLUME));
#endif
    index_s++;
  }

  // 5-slice no 1,...,L5-2
  for(is=1;is<L5-1;is++) {
  for(ix=0;ix<VOLUME;ix++) {
    xi_  = xi  + _GSI(index_s);
    phi_ = phi + _GSI(index_s);

#if (defined RIGHTHANDED_FWD)
    _fv_mi_eq_PLi_fv(xi_, phi+_GSI(index_s - VOLUME));
    _fv_mi_eq_PRe_fv(xi_, phi+_GSI(index_s + VOLUME));
#elif (defined RIGHTHANDED_BWD)  
    _fv_mi_eq_PRe_fv(xi_, phi+_GSI(index_s - VOLUME));
    _fv_mi_eq_PLi_fv(xi_, phi+_GSI(index_s + VOLUME));
#endif
    index_s++;
  }
  }

  // 5-slice no L5-1
  for(ix=0;ix<VOLUME;ix++) {
    xi_  = xi  + _GSI(index_s);
    phi_ = phi + _GSI(index_s);

#if (defined RIGHTHANDED_FWD)
    _fv_mi_eq_PLi_fv(xi_, phi+_GSI(index_s - VOLUME));
    _fv_pl_eq_PRe_fv_ti_re(xi_, phi+_GSI(ix), g_m0);
#elif (defined RIGHTHANDED_BWD)  
    _fv_mi_eq_PRe_fv(xi_, phi+_GSI(index_s - VOLUME));
    _fv_pl_eq_PLi_fv_ti_re(xi_, phi+_GSI(ix), g_m0);
#endif
    index_s++;
  }
  return;
}

void Q_DW_Wilson_5th_phi(double *xi, double *phi) {

  unsigned int ix, index_s, is; 
  double spinor1[24], spinor2[24];
  double *xi_, *phi_;

  // 5-slice no 0
  index_s = 0;
  for(ix=0;ix<VOLUME;ix++) {
    xi_  = xi  + _GSI(index_s);
    phi_ = phi + _GSI(index_s);

#if (defined RIGHTHANDED_FWD)
    _fv_pl_eq_PRe_fv_ti_re(xi_, phi+_GSI((L5-1)*VOLUME+ix), g_m0);
    _fv_mi_eq_PLi_fv(xi_, phi+_GSI(index_s+VOLUME));
#elif (defined RIGHTHANDED_BWD)  
    _fv_pl_eq_PLi_fv_ti_re(xi_, phi+_GSI((L5-1)*VOLUME+ix), g_m0);
    _fv_mi_eq_PRe_fv(xi_, phi+_GSI(index_s+VOLUME));
#endif
    index_s++;
  }

  // 5-slice no 1,...,L5-2
  for(is=1;is<L5-1;is++) {
  for(ix=0;ix<VOLUME;ix++) {
    xi_  = xi  + _GSI(index_s);
    phi_ = phi + _GSI(index_s);

#if (defined RIGHTHANDED_FWD)
    _fv_mi_eq_PRe_fv(xi_, phi+_GSI(index_s - VOLUME));
    _fv_mi_eq_PLi_fv(xi_, phi+_GSI(index_s + VOLUME));
#elif (defined RIGHTHANDED_BWD)  
    _fv_mi_eq_PLi_fv(xi_, phi+_GSI(index_s - VOLUME));
    _fv_mi_eq_PRe_fv(xi_, phi+_GSI(index_s + VOLUME));
#endif
    index_s++;
  }
  }

  // 5-slice no L5-1
  for(ix=0;ix<VOLUME;ix++) {
    xi_  = xi  + _GSI(index_s);
    phi_ = phi + _GSI(index_s);

#if (defined RIGHTHANDED_FWD)
    _fv_mi_eq_PRe_fv(xi_, phi+_GSI(index_s - VOLUME));
    _fv_pl_eq_PLi_fv_ti_re(xi_, phi+_GSI(ix), g_m0);
#elif (defined RIGHTHANDED_BWD)  
    _fv_mi_eq_PLi_fv(xi_, phi+_GSI(index_s - VOLUME));
    _fv_pl_eq_PRe_fv_ti_re(xi_, phi+_GSI(ix), g_m0);
#endif
    index_s++;
  }

  return;
}

void Q_DW_Wilson_phi(double *xi, double *phi) {
#if !(defined RIGHTHANDED_BWD) && !(defined RIGHTHANDED_FWD)
  EXIT_WITH_MSG(1, "[Q_DW_Wilson_phi] Error, chiral projectors undefined\n");
#else
  double _1_2_kappa = 0.5 / g_kappa5d;
  unsigned int ix;

  for(ix=0;ix<VOLUME*L5;ix++) {
    _fv_eq_fv_ti_re(xi+_GSI(ix), phi+_GSI(ix), _1_2_kappa);
    //_fv_eq_zero(xi+_GSI(ix));
  }

  Q_DW_Wilson_4d_phi(xi, phi);
  Q_DW_Wilson_5th_phi(xi, phi);
  return;
#endif
}

void Q_DW_Wilson_dag_phi(double *xi, double *phi) {
#if !(defined RIGHTHANDED_BWD) && !(defined RIGHTHANDED_FWD)
  EXIT_WITH_MSG(1, "[Q_DW_Wilson_dag_phi] Error, chiral projectors undefined\n");
#else
  double _1_2_kappa = 0.5 / g_kappa5d;
  unsigned int ix;

  for(ix=0;ix<VOLUME*L5;ix++) {
    _fv_eq_fv_ti_re(xi+_GSI(ix), phi+_GSI(ix), _1_2_kappa);
    //_fv_eq_zero(xi+_GSI(ix));
  }

  Q_DW_Wilson_dag_4d_phi(xi, phi);
  Q_DW_Wilson_dag_5th_phi(xi, phi);
  return;
#endif
}

/*********************************************
 * q(x) = P^-/L psi(0,x) + P^+/R psi(L5-1,x)
 * - s and t can be the same field
 *********************************************/
void spinor_5d_to_4d(double*s, double*t) {
  unsigned int ix, iy;

  for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
    _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
    _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#endif
  }

  for(ix=0, iy=(L5-1)*VOLUME;ix<VOLUME;ix++, iy++) {
#if (defined RIGHTHANDED_FWD)
    _fv_pl_eq_PRe_fv(s+_GSI(ix), t+_GSI(iy));
#elif (defined RIGHTHANDED_BWD)  
    _fv_pl_eq_PLi_fv(s+_GSI(ix), t+_GSI(iy));
#endif
  }
 return;
}

/*********************************************
 * psi(0, x)    = P^+/R q(x)
 * psi(L5-1, x) = P^-/L q(x)a
 * - s and t can be the same field
 *********************************************/
void spinor_4d_to_5d(double *s, double*t) {
  unsigned int ix, iy;

  for(ix=0, iy=(L5-1)*VOLUME;ix<VOLUME;ix++, iy++) {
#if (defined RIGHTHANDED_FWD)
    _fv_eq_PLi_fv(s+_GSI(iy), t+_GSI(ix) );
#elif (defined RIGHTHANDED_BWD)  
    _fv_eq_PRe_fv(s+_GSI(iy), t+_GSI(ix) );
#endif
  }

  for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
    _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
    _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#endif
  }

  for(ix=VOLUME;ix<(L5-1)*VOLUME;ix++) {
    _fv_eq_zero(s+_GSI(ix));
  }

 return;
}


/*********************************************
 * like spinor_5d_to_4d but with additional
 *   choice of sign
 * - sign = +1 gives result of spinor_5d_to_4d
 *********************************************/
void spinor_5d_to_4d_sign(double*s, double*t, int isign) {
  unsigned int ix, iy;

  if(isign == +1) {
    for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#endif
    }
    for(ix=0, iy=(L5-1)*VOLUME;ix<VOLUME;ix++, iy++) {
#if (defined RIGHTHANDED_FWD)
      _fv_pl_eq_PRe_fv(s+_GSI(ix), t+_GSI(iy));
#elif (defined RIGHTHANDED_BWD)  
      _fv_pl_eq_PLi_fv(s+_GSI(ix), t+_GSI(iy));
#endif
    }
  } else if (isign == -1) {
    for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#endif
    }
    for(ix=0, iy=(L5-1)*VOLUME;ix<VOLUME;ix++, iy++) {
#if (defined RIGHTHANDED_FWD)
      _fv_pl_eq_PLi_fv(s+_GSI(ix), t+_GSI(iy));
#elif (defined RIGHTHANDED_BWD)  
      _fv_pl_eq_PRe_fv(s+_GSI(ix), t+_GSI(iy));
#endif
    }
  }
 return;
}

/*********************************************
 * like spinor_4d_to_5d but with additional
 *   choice of sign
 * - sign = +1 gives result of spinor_4d_to_5d
 *********************************************/
void spinor_4d_to_5d_sign(double *s, double*t, int isign) {
  unsigned int ix, iy;

  if(isign == +1) {
    for(ix=0, iy=(L5-1)*VOLUME;ix<VOLUME;ix++, iy++) {
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PLi_fv(s+_GSI(iy), t+_GSI(ix) );
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PRe_fv(s+_GSI(iy), t+_GSI(ix) );
#endif
    }

    for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#endif
    }
  } else if (isign == -1) {
    for(ix=0, iy=(L5-1)*VOLUME;ix<VOLUME;ix++, iy++) {
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PRe_fv(s+_GSI(iy), t+_GSI(ix) );
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PLi_fv(s+_GSI(iy), t+_GSI(ix) );
#endif
    }

    for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#endif
    }
  }

  for(ix=VOLUME;ix<(L5-1)*VOLUME;ix++) {
    _fv_eq_zero(s+_GSI(ix));
  }

 return;
}

/**********************************************
 * undo 4d -> 5d
 **********************************************/
void spinor_4d_to_5d_inv(double*s, double*t) {
  unsigned int ix, iy;
#if 0
  for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
    _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
    _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#endif
  }

  for(ix=0, iy=(L5-1)*VOLUME;ix<VOLUME;ix++, iy++) {
#if (defined RIGHTHANDED_FWD)
    _fv_pl_eq_PLi_fv(s+_GSI(ix), t+_GSI(iy));
#elif (defined RIGHTHANDED_BWD)  
    _fv_pl_eq_PRe_fv(s+_GSI(ix), t+_GSI(iy));
#endif
  }
#endif
  for(ix=0, iy=(L5-1)*VOLUME;ix<VOLUME;ix++, iy++) {
    _fv_eq_fv_pl_fv(s+_GSI(ix), t+_GSI(ix), t+_GSI(iy));
  }
 return;
}

/*********************************************
 * psi(0, x)    = P^+/R q(x)
 * psi(L5-1, x) = P^-/L q(x)a
 * - s and t can be the same field
 *********************************************/
void spinor_4d_to_5d_threaded(double *s, double*t, int threadid, int nthreads) {
  unsigned int ix, iy, i3, is, it;
  unsigned int VOL3 = LX*LY*LZ;

  for(it=threadid; it<T; it+=nthreads) {
  for(i3=0; i3<VOL3; i3++) {
    ix = it * VOL3 + i3;
    iy= (L5-1)*VOLUME + ix;

#if (defined RIGHTHANDED_FWD)
    _fv_eq_PLi_fv(s+_GSI(iy), t+_GSI(ix) );
#elif (defined RIGHTHANDED_BWD)  
    _fv_eq_PRe_fv(s+_GSI(iy), t+_GSI(ix) );
#endif
  }}

  for(it=threadid; it<T; it+=nthreads) {
  for(i3=0; i3<VOL3; i3++) {
    ix = it * VOL3 + i3;
#if (defined RIGHTHANDED_FWD)
    _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
    _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#endif
  }}

  for(is=1; is<(L5-1); is++) {
  for(it=threadid; it<T;it+=nthreads) {
  for(i3=0; i3<VOL3; i3++) {
    _fv_eq_zero(s+_GSI(ix));
  }}}

 return;
}

/*********************************************
 * like spinor_4d_to_5d but with additional
 *   choice of sign and threaded
 *********************************************/
void spinor_4d_to_5d_sign_threaded(double *s, double*t, int isign, int threadid, int nthreads) {
  unsigned int ix, iy, it, i3, is;
  unsigned int VOL3 = LX*LY*LZ;

  if(isign == +1) {
    for(it=threadid; it<T; it+=nthreads) {
    for(i3=0; i3<VOL3; i3++) {
      ix = it * VOL3 + i3;
      iy=(L5-1)*VOLUME + ix;
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PLi_fv(s+_GSI(iy), t+_GSI(ix) );
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PRe_fv(s+_GSI(iy), t+_GSI(ix) );
#endif
    }}

    for(it=threadid; it<T; it+=nthreads) {
    for(i3=0; i3<VOL3; i3++) {
      ix = it * VOL3 + i3;
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#endif
    }}
  } else if (isign == -1) {
    for(it=threadid; it<T; it+=nthreads) {
    for(i3=0; i3<VOL3; i3++) {
      ix = it * VOL3 + i3;
      iy = (L5-1)*VOLUME + ix;
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PRe_fv(s+_GSI(iy), t+_GSI(ix) );
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PLi_fv(s+_GSI(iy), t+_GSI(ix) );
#endif
    }}

    for(it=threadid; it<T; it+=nthreads) {
    for(i3=0; i3<VOL3; i3++) {
      ix = it * VOL3 + i3;
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(ix));
#endif
    }}
  }

  for(is=1; is<(L5-1); is++) {
  for(it=threadid; it<T; it+=nthreads) {
  for(i3=0; i3<VOL3; i3++) {
    ix = (is*T + it) * VOL3 + i3;
    _fv_eq_zero(s+_GSI(ix));
  }}}

 return;
}


/*********************************************
 * like spinor_5d_to_4d, but
 *   project out L5/2 and L5/2-1
 * - s and t can be the same field
 *********************************************/
void spinor_5d_to_4d_L5h(double*s, double*t) {
  unsigned int ix;
  unsigned int shift  = (L5/2)  *VOLUME;
  unsigned int shift2 = (L5/2-1)*VOLUME;


  for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
    _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(shift + ix));
#elif (defined RIGHTHANDED_BWD)  
    _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(shift + ix));
#endif
  }

  for(ix=0; ix<VOLUME; ix++) {
#if (defined RIGHTHANDED_FWD)
    _fv_pl_eq_PRe_fv(s+_GSI(ix), t+_GSI(shift2 + ix));
#elif (defined RIGHTHANDED_BWD)  
    _fv_pl_eq_PLi_fv(s+_GSI(ix), t+_GSI(shift2 + ix));
#endif
  }
 return;
}


/*********************************************
 * like spinor_5d_to_4d but with additional
 *   choice of sign and projection planes L5, L5/2-1
 * - sign = +1 gives result of spinor_5d_to_4d
 *********************************************/
void spinor_5d_to_4d_L5h_sign(double*s, double*t, int isign) {
  unsigned int ix;
  unsigned int shift  = (L5/2  )*VOLUME;
  unsigned int shift2 = (L5/2+1)*VOLUME;

  if(isign == +1) {
    for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(shift+ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(shift+ix));
#endif
    }
    for(ix=0; ix<VOLUME; ix++) {
#if (defined RIGHTHANDED_FWD)
      _fv_pl_eq_PRe_fv(s+_GSI(ix), t+_GSI(shift2 + ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_pl_eq_PLi_fv(s+_GSI(ix), t+_GSI(shift2 + ix));
#endif
    }
  } else if (isign == -1) {
    for(ix=0;ix<VOLUME;ix++) {
#if (defined RIGHTHANDED_FWD)
      _fv_eq_PRe_fv(s+_GSI(ix), t+_GSI(shift + ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_eq_PLi_fv(s+_GSI(ix), t+_GSI(shift + ix));
#endif
    }
    for(ix=0; ix<VOLUME; ix++) {
#if (defined RIGHTHANDED_FWD)
      _fv_pl_eq_PLi_fv(s+_GSI(ix), t+_GSI(shift2 + ix));
#elif (defined RIGHTHANDED_BWD)  
      _fv_pl_eq_PRe_fv(s+_GSI(ix), t+_GSI(shift2 + ix));
#endif
    }
  }
 return;
}

/***********************************************************
 * M_eo and M_oe
 *   input: r, EO = 0/ 1 for eo / oe
 *   output: s
 ***********************************************************/

void Hopping_eo(double *s, double *r, double *gauge_field, int EO) {

  unsigned int ix, ix_lexic;
  unsigned int N  = VOLUME / 2;
  unsigned int N2 = (VOLUME+RAND) / 2;
  unsigned int ix_fwd, ix_bwd;
  double *U_fwd = NULL, *U_bwd = NULL;
  double sp1[24], sp2[24];
  double *s_ = NULL, *r_fwd_ = NULL, *r_bwd_ = NULL;
  double V_fwd[18], V_bwd[18];
  int threadid=0, nthreads = g_num_threads;

#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix,threadid,ix_lexic,ix_fwd,ix_bwd,U_fwd,U_bwd,V_fwd,V_bwd,r_fwd_,r_bwd_,sp1,sp2,s_) firstprivate(nthreads,N,N2) shared(s,r,gauge_field,co_phase_up,EO)
{
  threadid = omp_get_thread_num();
#endif
  for(ix = threadid; ix < N; ix += nthreads) {
    
    s_ = s + _GSI(ix);
      
    _fv_eq_zero(s_);

    /* ix is an even / odd point */
    ix_lexic = g_eo2lexic[ix + (unsigned int)(EO * N2)];

    /* =========================================== */
    /* =============== direction 0 =============== */
    U_fwd = gauge_field+_GGI(ix_lexic,0);
    _cm_eq_cm_ti_co(V_fwd, U_fwd, &co_phase_up[0]);

    U_bwd = gauge_field+_GGI( g_idn[ix_lexic][0],0);
    _cm_eq_cm_ti_co(V_bwd, U_bwd, &co_phase_up[0]);

    /* ix_fwd and ix_bwd are odd / even points */
    ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][0] ];
    ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][0] ];

    r_fwd_ = r + _GSI(ix_fwd);
    r_bwd_ = r + _GSI(ix_bwd);

    /* s += U_fwd ( 1 - g0 ) r_fwd */
    _fv_eq_gamma_ti_fv(sp1, 0, r_fwd_);
    _fv_eq_fv_mi_fv(sp2, r_fwd_, sp1);
    _fv_eq_cm_ti_fv(sp1, V_fwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* s += U_bwd^+ ( 1 + g0 ) r_bwd */
    _fv_eq_gamma_ti_fv(sp1, 0, r_bwd_);
    _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
    _fv_eq_cm_dag_ti_fv(sp1, V_bwd, sp2);
    _fv_pl_eq_fv(s_, sp1);


    /* =========================================== */
    /* =============== direction 1 =============== */
    U_fwd = gauge_field+_GGI(ix_lexic,1);
    _cm_eq_cm_ti_co(V_fwd, U_fwd, &co_phase_up[1]);

    U_bwd = gauge_field+_GGI( g_idn[ix_lexic][1],1);
    _cm_eq_cm_ti_co(V_bwd, U_bwd, &co_phase_up[1]);

    /* ix_fwd and ix_bwd are odd points */
    ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][1] ];
    ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][1] ];

    r_fwd_ = r + _GSI(ix_fwd);
    r_bwd_ = r + _GSI(ix_bwd);

    /* s += U_fwd ( 1 - g1 ) r_fwd */
    _fv_eq_gamma_ti_fv(sp1, 1, r_fwd_);
    _fv_eq_fv_mi_fv(sp2, r_fwd_, sp1);
    _fv_eq_cm_ti_fv(sp1, V_fwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* s += U_bwd^+ ( 1 + g1 ) r_bwd */
    _fv_eq_gamma_ti_fv(sp1, 1, r_bwd_);
    _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
    _fv_eq_cm_dag_ti_fv(sp1, V_bwd, sp2);
    _fv_pl_eq_fv(s_, sp1);
    
    /* =========================================== */
    /* =============== direction 2 =============== */
    U_fwd = gauge_field+_GGI(ix_lexic,2);
    _cm_eq_cm_ti_co(V_fwd, U_fwd, &co_phase_up[2]);

    U_bwd = gauge_field+_GGI( g_idn[ix_lexic][2],2);
    _cm_eq_cm_ti_co(V_bwd, U_bwd, &co_phase_up[2]);

    /* ix_fwd and ix_bwd are odd points */
    ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][2] ];
    ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][2] ];

    r_fwd_ = r + _GSI(ix_fwd);
    r_bwd_ = r + _GSI(ix_bwd);

    /* s += U_fwd ( 1 - g2 ) r_fwd */
    _fv_eq_gamma_ti_fv(sp1, 2, r_fwd_);
    _fv_eq_fv_mi_fv(sp2, r_fwd_, sp1);
    _fv_eq_cm_ti_fv(sp1, V_fwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* s += U_bwd^+ ( 1 + g2 ) r_bwd */
    _fv_eq_gamma_ti_fv(sp1, 2, r_bwd_);
    _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
    _fv_eq_cm_dag_ti_fv(sp1, V_bwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* =========================================== */
    /* =============== direction 3 =============== */
    U_fwd = gauge_field+_GGI(ix_lexic,3);
    _cm_eq_cm_ti_co(V_fwd, U_fwd, &co_phase_up[3]);

    U_bwd = gauge_field+_GGI( g_idn[ix_lexic][3],3);
    _cm_eq_cm_ti_co(V_bwd, U_bwd, &co_phase_up[3]);

    /* ix_fwd and ix_bwd are odd points */
    ix_fwd = g_lexic2eosub[ g_iup[ix_lexic][3] ];
    ix_bwd = g_lexic2eosub[ g_idn[ix_lexic][3] ];

    r_fwd_ = r + _GSI(ix_fwd);
    r_bwd_ = r + _GSI(ix_bwd);

    /* s += U_fwd ( 1 - g3 ) r_fwd */
    _fv_eq_gamma_ti_fv(sp1, 3, r_fwd_);
    _fv_eq_fv_mi_fv(sp2, r_fwd_, sp1);
    _fv_eq_cm_ti_fv(sp1, V_fwd, sp2);
    _fv_pl_eq_fv(s_, sp1);

    /* s += U_bwd^+ ( 1 + g3 ) r_bwd */
    _fv_eq_gamma_ti_fv(sp1, 3, r_bwd_);
    _fv_eq_fv_pl_fv(sp2, r_bwd_, sp1);
    _fv_eq_cm_dag_ti_fv(sp1, V_bwd, sp2);
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
  const int nthreads   = g_num_threads;
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

  const int nthreads = g_num_threads;
  const unsigned int N = VOLUME / 2;
  const size_t sizeof_field = _GSI(N) * sizeof(double);

  unsigned int ix;
  double *s_ = NULL, *s_aux_ = NULL;
  double sp1[24];
  int threadid=0;

#if 0
  /* s_aux = M_oe M_ee^-1 M_eo r */
  xchange_eo_field(r, 1);
  Hopping_eo(s_aux, r, gauge_field, 0);
  M_zz_inv(s, s_aux, mass);


  /* xchange before next application of M_oe */
  /* NOTE: s exchanged as even field */
  xchange_eo_field(s, 0);
  Hopping_eo(s_aux, s, gauge_field, 1);

  /* s = M_oo r */
  M_zz(s, r, mass);
 
#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix,threadid,sp1,s_,s_aux_) shared(s,s_aux)
{
  threadid = omp_get_thread_num();
#endif
  for(ix = threadid; ix < N; ix+=nthreads) {
    s_ = s + _GSI(ix);
    s_aux_ = s_aux + _GSI(ix);
    
    /* sp1 = s - s_aux = ( M_oo - M_oe M_ee^-1 M_eo ) r */
    _fv_eq_fv_mi_fv(sp1, s_, s_aux_);

    /* s = g5 sp1 */
    _fv_eq_gamma_ti_fv(s_, 5, sp1);

  }  /* end of  */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
#endif  /* of if 0 */

  /* s_aux = <- r */
  memcpy(s_aux, r, sizeof_field);
  xchange_eo_field(s_aux, 1);
  /* s <- M_eo s_aux */
  Hopping_eo(s, s_aux, gauge_field, 0);
  /* s <- M^-1 s */
  M_zz_inv(s, s, mass);

  /* xchange before next application of M_oe */
  /* NOTE: s exchanged as even field */
  /* s_aux <- s */
  memcpy(s_aux, s, sizeof_field);
  xchange_eo_field(s_aux, 0);
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

  unsigned int ix;
  unsigned int N = VOLUME / 2;

  xchange_eo_field(e_old, 0);
  /* e_new = M_ee e_old + M_eo o_old */
  Hopping_eo(e_new, o_old, gauge_field, 0);
  /* aux = M_ee e_old */
  M_zz (aux, e_old, mass);

  /* e_new = e_new + aux = M_ee e_old  M_eo_o_old */
  for(ix=0; ix < N; ix++) {
    _fv_pl_eq_fv(e_new+_GSI(ix), aux+_GSI(ix));
  }

  xchange_eo_field(o_old, 1);
  /* o_new = M_oo o_old + M_oe e_old */
  Hopping_eo(o_new, e_old, gauge_field, 1);
  /* aux = M_oo o_old*/
  M_zz (aux, o_old, mass);
  /* o_new  = o_new + aux = M_oe e_old + M_oo o_old */
  for(ix=0; ix < N; ix++) {
    _fv_pl_eq_fv(o_new+_GSI(ix), aux+_GSI(ix));
  }

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
  double *o_new_=NULL, *e_new_=NULL, *aux_=NULL, *o_old_=NULL;

  /* aux <- e_old */
  memcpy(aux, e_old, sizeof_field);
  /* exchange even field aux */
  xchange_eo_field(aux, 0);
  /* o_new = M_oe aux = M_oe e_old */
  Hopping_eo(e_new, aux, gauge_field, 1);
  /* o_new <- g5 e_new + o_old  = g5 M_oe e_old + o_old */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset, o_new_, e_new_, aux_, o_old_)
#endif
  for(ix=0; ix<N; ix++) {
    offset = _GSI(ix);
    o_new_ = o_new + offset;
    e_new_ = e_new + offset;
    o_old_ = o_old + offset;
    aux_   = aux + offset;
    _fv_ti_eq_g5(e_new_);
    _fv_eq_fv_pl_fv( o_new_, o_old_, e_new_);
  }

  /* e_new = M_zz aux = M_zz e_old */
  M_zz (e_new, aux, mass);
  /* e_new = g5 aux */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(e_new_)
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
#pragma omp parallel for private(ix)
#endif
  /* aux = g5 e_old */
  for(ix=0; ix<N; ix++) {
    _fv_eq_gamma_ti_fv(aux+_GSI(ix), 5, e_old+_GSI(ix));
  }

  /* aux <- M_ee^-1 aux = M_ee^-1 g5 e_old */
  M_zz_inv (aux, aux, mass);

  /* xchange aux (even) field */
  xchange_eo_field(aux,0);

  /* e_new = M_oe aux  */
  Hopping_eo(e_new, aux, gauge_field, 1);

#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix,iix,spinor1)
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

  unsigned int ix;
  unsigned int N = VOLUME / 2;
  
  xchange_eo_field(o_old,1);
  /* aux = M_eo o_old */
  Hopping_eo(aux, o_old, gauge_field, 0);
  /* e_new = M_ee^(-1) aux */
  M_zz_inv(e_new, aux, mass);
  /* e_new += e_old */
  for(ix=0; ix<N; ix++) {
    _fv_pl_eq_fv( e_new+_GSI(ix), e_old+_GSI(ix));
  }

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
  unsigned int ix;

  spinor_field_eq_spinor_field_ti_re (aux, o_old, twokappa, N);

  xchange_eo_field(aux, 1);
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
  xchange_eo_field(odd, 1);
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

#if 0
DOES NOT WORK LIKE THAT
/********************************************************************
 * X_oe    = -M_oe M_ee^-1,    mu > 0
 * Xbar_oe = -Mbar_oo^-1 M_oe, mu < 0
 * the input field is always even, the output field is always odd
 ********************************************************************/
void X_oe (double *odd, double *even, double mu, double *gauge_field) {

  const int nthreads = g_num_threads;
  const unsigned int N = VOLUME/2;
  const double mutilde = 2. * g_kappa * mu;
  const double a_re = -2. * g_kappa / ( 1 + mutilde * mutilde);
  const double a_im = -a_re * mutilde;

  double *ptr, sp[24];
  int threadid = 0;
  unsigned int ix;

#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(ix,threadid,ptr,sp) shared(odd)
{
  threadid = omp_get_thread_num();
#endif
  /* -M_ee^-1 */
  for(ix = threadid; ix < N; ix+=nthreads) {
    ptr = even + _GSI(ix);
    _fv_eq_fv(sp, ptr);
    _fv_eq_a_pl_ib_g5_ti_fv(ptr, sp, a_re, a_im);
  }  /* end of loop in ix = 0, N-1 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  /* M_oe */
  xchange_eo_field(even, 0);
  Hopping_eo(odd, even, gauge_field, 1);
}  /* end of X_oe */
#endif

/********************************************************************
 * C_with_Xeo
 * - apply C = g5 M_oo + g5 M_oe X_eo
 ********************************************************************/
void C_with_Xeo (double *r, double *s, double *gauge_field, double mu, double *r_aux) {

  unsigned int ix, iix;
  unsigned int N = VOLUME / 2;
  double a_re = 1./(2. * g_kappa);
  double a_im = mu;

  /* xchange_eo_field(s, 1); */
  X_eo (r_aux, s, mu, gauge_field);

  xchange_eo_field(r_aux, 0);
  Hopping_eo(r, r_aux, gauge_field, 1);

  for(ix=0; ix<N; ix++) {
    iix = _GSI(ix);
    _fv_ti_eq_g5 (r+iix);
    _fv_pl_eq_a_g5_pl_ib_ti_fv(r+iix, s+iix, a_re, a_im);
  }

}  /* C_with_Xeo */

/********************************************************************
 * C_with_Xeo
 * - apply C = g5 M_oo + g5 M_oe X_eo
 *
 * output t
 * input s = X_eo v
 *       t = v
 *       r = auxilliary
 ********************************************************************/
void C_from_Xeo (double *t, double *s, double *r, double *gauge_field, double mu) {

  const int nthreads = g_num_threads;
  const double a_re = 1./(2. * g_kappa);
  const double a_im = mu;
  const unsigned int N = VOLUME / 2;

  unsigned int ix, iix;
  int threadid = 0;

  xchange_eo_field(s, 0);
  /*r = M_oe s = -M_oe M_ee^-1 M_eo v */
  Hopping_eo(r, s, gauge_field, 1);

#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) private(threadid,ix,iix) shared(r,t)
{
  threadid = omp_get_thread_num();
#endif
  /*  */
  for(ix = threadid; ix < N; ix += nthreads) {
    iix = _GSI(ix);
    _fv_ti_eq_g5 (r+iix);
    _fv_pl_eq_a_g5_pl_ib_ti_fv(r+iix, t+iix, a_re, a_im);
    _fv_eq_fv(t+iix, r+iix);
  }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* C_from_Xeo */

}  /* end of namespace cvc */
