#ifndef _LAPHS_UTILS_H
#define _LAPHS_UTILS_H

#include "laphs.h"

namespace cvc {


void init_perambulator (perambulator_type *peram);
int alloc_perambulator (perambulator_type *peram, \
                       int time_src_number, int spin_src_number, int evec_src_number,\
                       int time_snk_number, int spin_snk_number, int evec_snk_number, int color_snk_number, \
                       char *quark_type, char *snk_type, int irnd );

void fini_perambulator (perambulator_type *peram);
int print_perambulator_info (perambulator_type *peram);

/*  */

void init_eigensystem (eigensystem_type *es);
int alloc_eigensystem (eigensystem_type *es, unsigned int nt, unsigned nv);
void fini_eigensystem (eigensystem_type *es);
int print_eigensystem_info (eigensystem_type *es);
int test_eigensystem (eigensystem_type *es, double* gauge_field);

/*  */
void init_randomvector (randomvector_type *rv);
int alloc_randomvector (randomvector_type *rv, int nt, int ns, int nv);
void fini_randomvector (randomvector_type *rv);
void print_randomvector_info (randomvector_type *rv);
void print_randomvector (randomvector_type *rv, FILE*fp);

int project_randomvector (randomvector_type *prv, randomvector_type *rv, int bt, int bs, int bv);;

int fv_eq_eigensystem_ti_randomvector (double*s, eigensystem_type *v, randomvector_type*r);
int perambulator_eq_eigensystem_dag_ti_fv (perambulator_type*p, eigensystem_type*v, double*s, int it_src, int is_src, int iv_src);

int rotate_perambulator(perambulator_type*p, int sign);

void init_tripleV (tripleV_type *s, int nt, int np, int nv);
void fini_tripleV ( tripleV_type*s);

int reduce_triple_eigensystem_timeslice (tripleV_type *tripleV, eigensystem_type*es, int x0, int imom, double*momentum_phase);
}  /* end of namespace cvc */

#endif
