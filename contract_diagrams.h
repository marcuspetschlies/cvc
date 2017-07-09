#ifndef _CONTRACT_DIAGRAMS_H
#define _CONTRACT_DIAGRAMS_H

#include "gamma.h"

namespace cvc {
#if 0
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int perm[3], unsigned int N, int init );
#endif  /* of if 0 */
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int perm[4], unsigned int N, int init );

int contract_diagram_oet_v2_gamma_v3 ( double _Complex **vdiag, double _Complex ***v2, double _Complex ***v3, gamma_matrix_type goet, gamma_matrix_type g, int perm[4], unsigned int N, int init );

#if 0
void contract_b1 (double _Complex ***b1, double _Complex **v3, **double v2, gamma_matrix_type g);

void contract_b2 (double _Complex ***b2, double _Complex **v3, **double v2, gamma_matrix_type g);
#endif  /* end of if 0 */


static inline void zm_4x4_array_transposed (double _Complex *r, double _Complex *s ) {

  r[ 0] = s[ 0];
  r[ 1] = s[ 4];
  r[ 2] = s[ 8];
  r[ 3] = s[12];
  r[ 4] = s[ 1];
  r[ 5] = s[ 5];
  r[ 6] = s[ 9];
  r[ 7] = s[13];
  r[ 8] = s[ 2];
  r[ 9] = s[ 6];
  r[10] = s[10];
  r[11] = s[14];
  r[12] = s[ 3];
  r[13] = s[ 7];
  r[14] = s[11];
  r[15] = s[15];
}  /* end of zm_4x4_array_transposed */

}  /* end of namespace cvc */
#endif
