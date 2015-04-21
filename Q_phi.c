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
#include "cvc_complex.h"
#include "global.h"
#include "cvc_linalg.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "Q_phi.h"

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
    cvc_mul_one_pm_imu_inv(work, -1., VOLUME);

    /* apply the hopping matrix */
    xchange_field(work);
    Hopping(xi, work);
    /* apply B^+ */
    cvc_mul_one_pm_imu_inv(xi, -1., VOLUME);

  }

  /* final step: multiply with gamma_5 and */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(spinor1, 5, &xi[_GSI(ix)]);
    _fv_eq_fv(&xi[_GSI(ix)], spinor1);
  }
}

/***********************************
 * cvc_mul_one_pm_imu_inv
 *
 ***********************************/
void cvc_mul_one_pm_imu_inv (double *phi, double sign, int V) {

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
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 2nd application of the hopping matrix */
  Hopping(phi, xi);
  cvc_mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 3rd application of the hopping matrix */
  Hopping(xi, phi);
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
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
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 2nd application of the hopping matrix */
  Hopping(phi, xi);
  cvc_mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 3rd application of the hopping matrix */
  Hopping(xi, phi);
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 4th application of the hopping matrix */
  Hopping(phi, xi);
  cvc_mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 5th application of the hopping matrix */
  Hopping(xi, phi);
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
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
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
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
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);
  Hopping(phi, xi);
  cvc_mul_one_pm_imu_inv(phi, +1., VOLUME);  
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
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 2nd application of the hopping matrix */
  Hopping(phi, xi);
  cvc_mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 3rd application of the hopping matrix */
  Hopping(xi, phi);
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 4th application of the hopping matrix */
  Hopping(phi, xi);
  cvc_mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 5th application of the hopping matrix */
  Hopping(xi, phi);
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
  xchange_field(xi);

  /* 6th application of the hopping matrix */
  Hopping(phi, xi);
  cvc_mul_one_pm_imu_inv(phi, +1., VOLUME);  
  xchange_field(phi);

  /* 7th application of the hopping matrix */
  Hopping(xi, phi);
  cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);  
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
    cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);
    xchange_field(xi);

    Hopping(phi, xi);
    cvc_mul_one_pm_imu_inv(phi, +1., VOLUME);
    xchange_field(phi);
  }

  if(r==1) {
    Hopping(xi, phi);
    cvc_mul_one_pm_imu_inv(xi, +1., VOLUME);
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
