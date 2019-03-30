#ifndef _GAMMA_H
#define _GAMMA_H

namespace cvc {

typedef struct {
  double _Complex v[16]; 
  double _Complex *m[4];
  int id;
  double s;
} gamma_matrix_type;


void gamma_matrix_init ( gamma_matrix_type *g);
void gamma_matrix_zero ( gamma_matrix_type *g);
void gamma_matrix_fill ( gamma_matrix_type *g);
void gamma_matrix_printf (gamma_matrix_type *g, char*name, FILE*ofs);
void gamma_matrix_mult ( gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3 );

void gamma_matrix_transposed (gamma_matrix_type *g, gamma_matrix_type *p);
void gamma_matrix_adjoint ( gamma_matrix_type *g, gamma_matrix_type *p);

void gamma_matrix_norm (double *norm, gamma_matrix_type *g);

void gamma_matrix_eq_gamma_matrix_mi_gamma_matrix (gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3);

void gamma_matrix_eq_gamma_matrix_pl_gamma_matrix_ti_re (gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3, double r);

void gamma_matrix_eq_gamma_matrix_ti_gamma_matrix ( gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3 );

void gamma_matrix_eq_gamma_matrix_transposed (gamma_matrix_type *g, gamma_matrix_type *p);
void gamma_matrix_eq_gamma_matrix_adjoint ( gamma_matrix_type *g, gamma_matrix_type *p);

void init_gamma_matrix (void);

void gamma_matrix_set ( gamma_matrix_type *g, int id, double s );

void gamma_matrix_eq_gamma_matrix_pl_gamma_matrix_ti_co (gamma_matrix_type *g1, gamma_matrix_type *g2, gamma_matrix_type *g3, double _Complex c );

void gamma_matrix_ukqcd_binary ( gamma_matrix_type * const g, int const n  );

}
#endif
