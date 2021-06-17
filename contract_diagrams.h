#ifndef _CONTRACT_DIAGRAMS_H
#define _CONTRACT_DIAGRAMS_H

#include "gamma.h"

namespace cvc {
#if 0
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int perm[3], unsigned int N, int init );
#endif  /* of if 0 */
int contract_diagram_v2_gamma_v3 ( double _Complex **vdiag, double _Complex **v2, double _Complex **v3, gamma_matrix_type g, int const perm[4], unsigned int const N, int const init );

int contract_diagram_oet_v2_gamma_v3 ( double _Complex **vdiag, double _Complex ***v2, double _Complex ***v3, gamma_matrix_type goet, gamma_matrix_type g, int const perm[4], unsigned int const N, int const init );

#if 0
void contract_b1 (double _Complex ***b1, double _Complex **v3, **double v2, gamma_matrix_type g);

void contract_b2 (double _Complex ***b2, double _Complex **v3, **double v2, gamma_matrix_type g);
#endif  /* end of if 0 */

int match_momentum_id ( int **pid, int **m1, int **m2, int N1, int N2 );

int * get_conserved_momentum_id ( int (*p1)[3], int const n1, int const p2[3], int (*p3)[3], int const n3 );

int get_momentum_id ( int const p1[3], int (* const p2)[3], unsigned int const N );

int correlator_add_baryon_boundary_phase ( double _Complex *** const sp, int const tsrc, int const dir, int const N );

int correlator_add_source_phase ( double _Complex ***sp, int const p[3], int const source_coords[3], unsigned int const N );

int correlator_spin_parity_projection (double _Complex ***sp_out, double _Complex ***sp_in, double const c, unsigned const N);

int correlator_spin_projection (double _Complex ***sp_out, double _Complex ***sp_in, int const i, int const k, double const a, double const b, unsigned int const N);

int reorder_to_absolute_time (double _Complex ***sp_out, double _Complex ***sp_in, int const tsrc, int const dir, unsigned int const N);

int reorder_to_relative_time (double _Complex ***sp_out, double _Complex ***sp_in, int const tsrc, int const dir, unsigned int const N);

int contract_diagram_zm4x4_field_ti_co_field ( double _Complex ***sp_out, double _Complex ***sp_in, double _Complex *c_in, unsigned int N);

int contract_diagram_zm4x4_field_eq_zm4x4_field_ti_co ( double _Complex *** const sp_out, double _Complex *** const sp_in, double _Complex const c_in, unsigned int const N);

int contract_diagram_zm4x4_field_ti_eq_co ( double _Complex *** const sp_out, double _Complex const c_in, unsigned int const N);

int contract_diagram_zm4x4_field_ti_eq_re ( double _Complex *** const sp, double const r_in, unsigned int const N);

int contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( double _Complex *** const sp_out, double _Complex *** const sp_in, unsigned int const N);

int contract_diagram_sample (double _Complex ***diagram, double _Complex ***xi, double _Complex ***phi, int const nsample, int const perm[4], gamma_matrix_type C, int const nT );

int contract_diagram_sample_oet (double _Complex ***diagram, double _Complex ***xi, double _Complex ***phi, gamma_matrix_type goet, int const perm[4], gamma_matrix_type C, int const nT );

int contract_diagram_write_aff (double _Complex***diagram, struct AffWriter_s*affw, char*aff_tag, int const tstart, int const dt, int const fbwd, int const io_proc );

int contract_diagram_write_aff_sst (double _Complex***diagram, struct AffWriter_s*affw, char*aff_tag, int const tstart, int const dt, int const fbwd, int const io_proc );

int contract_diagram_write_scalar_aff (double _Complex*diagram, struct AffWriter_s*affw, char*aff_tag, int const tstart, int const dt, int const fbwd, int const io_proc );

int contract_diagram_write_fp ( double _Complex*** const diagram, FILE *fp, char*tag, int const tstart, unsigned int const dt, int const fbwd );

int contract_diagram_key_suffix ( char * const suffix, int const gf2, int const pf2[3], int const gf11, int const gf12, int const pf1[3], int const gi2, int const pi2[3], int const gi11, int const gi12, int const pi1[3], int const sx[4] );

int contract_diagram_key_suffix_from_type ( char * key_suffix, twopoint_function_type * p );

int contract_diagram_zm4x4_field_mul_gamma_lr ( double _Complex *** const sp_out, double _Complex *** const sp_in, gamma_matrix_type const gl, gamma_matrix_type const gr, unsigned int const N );

int contract_diagram_read_key_qlua (
  double _Complex **fac, // output
  char const *prefix,    // key prefix
  int const gi,          // sequential gamma id
  int const pi[3],       // sequential momenta
  int const gsx[4],      // source coords
  int const isample,     // number of sample
  char const * vtype,    // contraction type
  int const gf,          // vertex gamma
  int const pf[3],       // vertex momentum
  struct AffReader_s *affr,  // AFF reader 
  int const N,           // length of data key ( will be mostly T_global )
  int const ncomp        // length of data key ( will be mostly T_global )
);

int contract_diagram_read_oet_key_qlua (
  double _Complex ***fac, // output
  char const *prefix,     // key prefix
  int const pi[3],        // sequential momenta
  int const gsx[4],       // source coords
  char const * vtype,     // contraction type
  int const gf,           // vertex gamma
  int const pf[3],        // vertex momentum
  struct AffReader_s *affr,  // AFF reader 
  int const N,            // length of data key ( will be mostly T_global )
  int const ncomp         // components
);

double _Complex contract_diagram_get_correlator_phase ( char * const type, int const gi11, int const gi12, int const gi2, int const gf11, int const gf12, int const gf2 );
 

int contract_diagram_zm4x4_field_pl_eq_zm4x4_field ( double _Complex *** const s, double _Complex *** const r, unsigned int const N);

int contract_diagram_zm4x4_field_eq_zm4x4_field_pl_zm4x4_field_ti_co ( double _Complex *** const s, double _Complex *** const r, double _Complex *** const p, double _Complex const z, unsigned int const N);

void contract_diagram_mat_op_ti_zm4x4_field_ti_mat_op ( double _Complex *** const sp_out, double _Complex ** const R1, char const op1 , double _Complex *** const sp_in, double _Complex ** const R2, char const op2, unsigned int const N );

int contract_diagram_finalize ( double _Complex *** const diagram, char * const diagram_type, int const sx[4], int const p[3], 
    int const gf11_id, int const gf12_id, int const gf12_sign, int const gf2_id,
    int const gi11_id, int const gi12_id, int const gi12_sign, int const gi2_id,
    unsigned int const N );


int contract_diagram_co_eq_tr_zm4x4_field ( double _Complex * const r, double _Complex *** const diagram, unsigned int const N );

}  // end of namespace cvc
#endif
