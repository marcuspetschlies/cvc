/********************
 * Q_phi.c
 ********************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef MPI
#  include <mpi.h>
#endif
#ifdef OPENMP
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
#ifndef OPENMP
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
#ifdef MPI
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
#endif  // of ifndef OPENMP

#ifndef OPENMP
void g5_phi(double *phi) {
  double spinor1[24];
  int ix;
  double *phi_;

  for(ix = 0; ix < VOLUME; ix++)
  {
      phi_ = phi + _GSI(ix);
      _fv_eq_gamma_ti_fv(spinor1, 5, phi_);
      _fv_eq_fv(phi_, spinor1);
  }

}  // end of g5_phi

#else
void g5_phi(double *phi) {
#pragma omp parallel shared(phi, VOLUME)
{
  double spinor1[24];
  int threadid = omp_get_thread_num();
  int num_threads = omp_get_num_threads();
  int ix;
  double *phi_;

  // TEST
  // fprintf(stdout, "# [g5_phi] thread%.4d number of threads %d\n", threadid, num_threads);

  for(ix = threadid; ix < VOLUME; ix+=num_threads)
  {
      phi_ = phi + _GSI(ix);
      _fv_eq_gamma_ti_fv(spinor1, 5, phi_);
      _fv_eq_fv(phi_, spinor1);
  }

}  // end of parallel region
}  // end of g5_phi

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
      _fv_ti_eq_re(xi_, -g_kappa);

  }
  }
  }
  }
  
}

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
#ifndef OPENMP
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
#ifdef MPI
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

#ifdef OPENMP
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
#if (defined HAVE_QUDA) && ( (defined PARALLELTX) || (defined PARALLELTXY) )
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
#if (defined HAVE_QUDA) && ( (defined PARALLELTX) || (defined PARALLELTXY) )
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

}
