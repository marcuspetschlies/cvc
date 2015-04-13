#ifndef _Q_PHI_H
#define _Q_PHI_H

/********************/

/* Computes xi = Q phi, where Q is the light tm Dirac operator with twisted boundary conditions. */

void Q_phi_tbc(double *xi, double *phi);
void Hopping(double *xi, double *phi);
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
#endif
