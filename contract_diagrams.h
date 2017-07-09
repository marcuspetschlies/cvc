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
}  /* end of namespace cvc */
#endif
