#ifndef _HYP_SMEAR_H
#define _HYP_SMEAR_H

namespace cvc {

void init_tab_inv(void);

int hyp_smear_step( double *u_out, double *u_in, double A[3], double accu, unsigned int imax);
int hyp_smear_step_3d( double *u_out, double *u_in, double A[3], double accu, unsigned int imax);

int hyp_smear (double *u, unsigned int N, double accu, unsigned int imax);
int hyp_smear_3d (double *u, unsigned int N, double *A, double accu, unsigned int imax);
}
#endif
