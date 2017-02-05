#ifndef _Q_PHI_H
#define _Q_PHI_H

/********************/

/* Computes xi = Q phi, where Q is the light tm Dirac operator with twisted boundary conditions. */

namespace cvc {

void Q_phi_tbc(double *xi, double *phi);

void Hopping(double *xi, double *phi);
void Hopping_eo(double *s, double *r, double *gauge_field, int EO);

void M_zz (double*s, double*r, double mass);
void M_zz_inv (double*s, double*r, double mass);
  
void C_oo (double*s, double*r, double *gauge_field, double mass, double *s_aux);


void gamma5_BdagH4_gamma5 (double *xi, double *phi, double *work);
void mul_one_pm_imu_inv (double *phi, double sign, int V);
void BH  (double *xi, double *phi);
void BH2 (double *xi, double *phi);
void BH3 (double *xi, double *phi);
void BH5 (double *xi, double *phi);
void BH7 (double *xi, double *phi);
void Hopping_rec(double *xi, double *phi, int xd, int xs, int l, int deg, int *steps);
void init_trace_coeff(double **tcf, double **tcb, int ***loop_tab, int deg, int *N);
void Hopping_iter(double *truf, double *trub, double *tcf, double *tcb, int xd, int mu, int deg, int nloopi, int **loop_tab);
void BHn (double *xi, double *phi, int n);
void test_hpem(void);
void test_cm_eq_cm_dag_ti_cm(void);

void Qf5(double *xi, double *phi, double mutm);

void Q_phi(double *xi, double *phi, const double mutm);

void Q_phi_eo (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);

void Q_eo_SchurDecomp_A (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);
void Q_eo_SchurDecomp_Ainv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);
void Q_eo_SchurDecomp_B (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);
void Q_eo_SchurDecomp_Binv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux);


void Q_Wilson_phi_tbc(double *xi, double *phi);
void Q_Wilson_phi(double *xi, double *phi);
void Q_g5_Wilson_phi(double *xi, double *phi);
void Q_Wilson_phi_nobc(double *xi, double *phi);
#ifdef HAVE_OPENMP
void Q_Wilson_phi_threads(double *xi, double *phi);
#endif

/* functions for DWF */
void Q_DW_Wilson_4d_phi(double *xi, double *phi);
void Q_DW_Wilson_dag_4d_phi(double *xi, double *phi);
void Q_DW_Wilson_5th_phi(double *xi, double *phi);
void Q_DW_Wilson_dag_5th_phi(double *xi, double *phi);
void Q_DW_Wilson_phi(double *xi, double *phi);
void Q_DW_Wilson_dag_phi(double *xi, double *phi);
void spinor_4d_to_5d(double *s, double*t);
void spinor_5d_to_4d(double *s, double*t);
void spinor_4d_to_5d_sign(double *s, double*t, int isign);
void spinor_5d_to_4d_sign(double *s, double*t, int isign);
void spinor_4d_to_5d_inv(double*s, double*t);

void spinor_4d_to_5d_threaded(double *s, double*t, int threadid, int nthreads);
void spinor_4d_to_5d_sign_threaded(double *s, double*t, int isign, int threadid, int nthreads);

void spinor_5d_to_4d_L5h(double*s, double*t);
void spinor_5d_to_4d_L5h_sign(double*s, double*t, int isign);

void X_eo (double *even, double *odd, double mu, double *gauge_field);
void X_oe (double *odd, double *even, double mu, double *gauge_field);

void C_with_Xeo (double *r, double *s, double *gauge_field, double mu, double *r_aux);

void C_from_Xeo (double *t, double *s, double *r, double *gauge_field, double mu);

void apply_cvc_vertex_eo(double *s, double *r, int mu, int fbwd, double *gauge_field, int EO);

void apply_cvc_vertex(double *s, double *r, int mu, int fbwd, double *gauge_field);

}  /* end of namespace cvc */
#endif
