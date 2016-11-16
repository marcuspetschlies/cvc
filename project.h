#ifndef _PROJECT_SOURCE_H
#define _PROJECT_SOURCE_H

namespace cvc {

int project_spinor_field(double *s, double * r, int parallel, double *V, int num, unsigned int N);
int project_propagator_field(double *s, double * r, int parallel, double *V, int num1, int num2, unsigned int N);
int project_reduce_from_propagator_field(double *p, double * r, double *V, int num1, int num2, unsigned int N);
int project_expand_to_propagator_field(double *s, double *p, double *V, int num1, int num2, unsigned int N);
int momentum_projection (double*V, double *W, unsigned int nv, int momentum_number, int (*momentum_list)[3]);
int momentum_projection2 (double*V, double *W, unsigned int nv, int momentum_number, int (*momentum_list)[3], int gshift[3]);


}  /* end of namespace cvc */

#endif
