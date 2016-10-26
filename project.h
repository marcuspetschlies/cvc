#ifndef _PROJECT_SOURCE_H
#define _PROJECT_SOURCE_H

namespace cvc {

int project_spinor_field(double *s, double * r, int parallel, double *V, int num, unsigned int N);
int project_propagator_field(double *s, double * r, int parallel, double *V, int num1, int num2, unsigned int N);
int project_reduce_from_propagator_field(double *p, double * r, double *V, int num1, int num2, unsigned int N);
int project_expand_to_propagator_field(double *s, double *p, double *V, int num1, int num2, unsigned int N);

}  /* end of namespace cvc */

#endif
