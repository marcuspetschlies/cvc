#ifndef _HYP_SMEAR_H
#define _HYP_SMEAR_H

namespace cvc {

void init_tab_inv(void);

int hyp_smear_step( double * const u_out, double * const u_in, double const A[3], double const accu, unsigned int const imax);

int hyp_smear_step_3d( double * const u_out, double * const u_in, double const A[2], double const accu, unsigned int const imax);


int hyp_smear (double *u, unsigned int N, double accu, unsigned int imax);

int hyp_smear_3d (double * const u, unsigned int const N, double * const A, double const accu, unsigned int const imax);

}
#endif
