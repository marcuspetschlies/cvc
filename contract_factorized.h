#ifndef _CONTRACT_FACTORIZED_H
#define _CONTRACT_FACTORIZED_H

namespace cvc {

int contract_v1 (double **v1, double *phi, fermion_propagator_type *prop1, unsigned int N );

int contract_v2 (double **v2, double *phi, fermion_propagator_type *prop1, fermion_propagator_type *prop2, unsigned int N );

int contract_v3  (double **v3, double*phi, fermion_propagator_type*prop, unsigned int N );

}  /* end of namespace cvc */
#endif

