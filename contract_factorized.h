#ifndef _CONTRACT_FACTORIZED_H
#define _CONTRACT_FACTORIZED_H

namespace cvc {

int contract_v1 (double **v1, double *phi, fermion_propagator_type *prop1, unsigned int N );

int contract_v2 (double **v2, double *phi, fermion_propagator_type *prop1, fermion_propagator_type *prop2, unsigned int N );

int contract_v3  (double **v3, double*phi, fermion_propagator_type*prop, unsigned int N );

int contract_v2_from_v1 (double **v2, double **v1, fermion_propagator_type *prop, unsigned int N );

int contract_v4 (double **v4, double *phi, fermion_propagator_type *prop1, fermion_propagator_type *prop2, unsigned int N );

int contract_v5 (double **v5, fermion_propagator_type *prop1, fermion_propagator_type *prop2, fermion_propagator_type *prop3, unsigned int N );

int contract_v6 (double **v6, fermion_propagator_type *prop1, fermion_propagator_type *prop2, fermion_propagator_type *prop3, unsigned int N );

/* int contract_vn_momentum_projection (double***vp, double**vx, int n, int (*momentum_list)[3], int momentum_number); */
int contract_vn_momentum_projection (double *** const vp, double ** const vx, int const n, const int (* const momentum_list)[3], int const momentum_number);


/* int contract_vn_write_aff (double ***vp, int n, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc ); */
int contract_vn_write_aff (double *** const vp, int const n, struct AffWriter_s*affw, char*tag, const int (* const momentum_list)[3], int const momentum_number, int const io_proc );


}  /* end of namespace cvc */
#endif

