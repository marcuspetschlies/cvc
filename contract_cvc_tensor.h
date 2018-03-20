#ifndef _CONTRACT_CVC_TENSOR_H
#define _CONTRACT_CVC_TENSOR_H
/****************************************************
 * contract_cvc_tensor.h
 *
 * Fri Mar 10 14:39:30 CET 2017
 *
 ****************************************************/

namespace cvc {

void init_contract_cvc_tensor_usource(double *gauge_field, int source_coords[4], complex *phase);

void contract_cvc_tensor_eo ( double *conn_e, double *conn_o, double *contact_term, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o , double*gauge_field );

void co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (complex *w, fermion_propagator_type *r, fermion_propagator_type *s, double sign, unsigned int N);

void cvc_tensor_eo_subtract_contact_term (double**eo_tensor, double*contact_term, int gsx[4], int have_source );

int cvc_tensor_eo_momentum_projection (double****tensor_tp, double**tensor_eo, int (*momentum_list)[3], int momentum_number);

int cvc_tensor_tp_write_to_aff_file (double***cvc_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc );

int cvc_tensor_eo_check_wi_position_space (double **tensor_eo);

int contract_vdag_gloc_spinor_field (double**prop_list_e, double**prop_list_o, int nsf, double**V, int numV, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2] );

int apply_constant_cvc_vertex_at_source (double**s, int mu, int fbwd, const unsigned int N );

int contract_vdag_cvc_spinor_field (double**prop_list_e, double**prop_list_o, int nsf, double**V, int numV, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

int vdag_w_reduce_write (double _Complex ***contr, double _Complex **V, double _Complex **W, int dimV, int dimW, char *aff_path, struct AffWriter_s *affw, struct AffNode_s *affn, int io_proc, double _Complex **V_ts, double _Complex **W_ts, double *mcontr_buffer);

int contract_vdag_gloc_w_blocked (double**V, int numV, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

int contract_vdag_gloc_phi_blocked (double**V, double**Phi, int numV, int numPhi, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

int contract_vdag_cvc_w_blocked (double**V, int numV, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

int contract_vdag_cvc_phi_blocked (double**V, double**Phi, int numV, int numPhi, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

int contract_write_to_aff_file (double **c_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc );

int contract_local_local_2pt_eo ( double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o,
    int *gamma_sink_list, int gamma_sink_num, int*gamma_source_list, int gamma_source_num, int (*momentum_list)[3], int momentum_number,  struct AffWriter_s*affw, char*tag,
    int io_proc );

int contract_local_cvc_2pt_eo ( double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o,
  int *gamma_sink_list, int gamma_sink_num, int (*momentum_list)[3], int momentum_number,  struct AffWriter_s*affw, char*tag,
  int io_proc );

int cvc_tensor_eo_write_contact_term_to_aff_file ( double *contact_term, struct AffWriter_s*affw, char *tag, int io_proc );

void co_field_pl_eq_tr_propagator_field (complex *w, fermion_propagator_type *r, double sign, unsigned int N);

void co_field_pl_eq_tr_propagator_field_conj (complex *w, fermion_propagator_type *r, double sign, unsigned int N);

void contract_cvc_loop_eo ( double ***loop, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o , double*gauge_field );

void contract_cvc_loop_eo_lma ( double ***loop, double**eo_evecs_field, double *eo_evecs_norm, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2]);

void contract_cvc_loop_eo_lma_wi ( double **wi, double**eo_evecs_field, double *eo_evecs_norm, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2]);

int cvc_loop_eo_check_wi_position_space_lma ( double ***wwi, double ***loop_lma, double **eo_evecs_field, double *evecs_norm, int nev, double *gauge_field, double **mzz[2], double **mzzinv[2]  );

int cvc_loop_eo_momentum_projection (double****loop_tp, double***loop_eo, int (*momentum_list)[3], int momentum_number);

int cvc_loop_tp_write_to_aff_file (double***cvc_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc );

int cvc_loop_eo_check_wi_momentum_space_lma ( double **wi, double ***loop_lma, int (*momentum_list)[3], int momentum_number  );

int vdag_w_spin_color_reduction ( double ***contr, double**V, double**W, int dimV, int dimW, int t );

int vdag_w_momentum_projection ( double _Complex ***contr_p, double ***contr_x, int dimV, int dimW, int (*momentum_list)[3], int momentum_number, int t, int ieo, int mu );

int vdag_w_write_to_aff_file (
  double _Complex *** const contr_tp, unsigned int const nv, unsigned int const nw,
  struct AffWriter_s*affw,
  char * const tag,
  int (* const momentum_list)[3], unsigned int const momentum_number,
  int const io_proc
);


int contract_cvc_tensor_eo_lm_factors ( double**eo_evecs_field, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2],
    struct AffWriter_s **affw, char*tag,
    int (*momentum_list)[3], int momentum_number, int io_proc, int block_length );

}  /* end of namespace cvc */

#endif
