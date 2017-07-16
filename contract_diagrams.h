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

int match_momentum_id ( int **pid, int **m1, int **m2, int N1, int N2 );

int correlator_add_baryon_boundary_phase ( double _Complex ***sp, int tsrc);

int correlator_add_source_phase ( double _Complex ***sp, int p[3], int source_coords[3], unsigned int N );

int correlator_spin_parity_projection (double _Complex ***sp_out, double _Complex ***sp_in, double c, unsigned N);

int correlator_spin_projection (double _Complex ***sp_out, double _Complex ***sp_in, double c, unsigned N);

int reorder_to_absolute_time (double _Complex ***sp_out, double _Complex ***sp_in, int tsrc, int dir, unsigned N);

}  /* end of namespace cvc */
#endif
