#ifndef _CONTRACT_CVC_TENSOR_H
#define _CONTRACT_CVC_TENSOR_H
/****************************************************
 * contract_cvc_tensor.h
 *
 * Fri Mar 10 14:39:30 CET 2017
 *
 ****************************************************/

namespace cvc {

void init_contract_cvc_tensor_usource(double *gauge_field, int source_coords[4]);

void contract_cvc_tensor_eo ( double *conn_e, double *conn_o, double *contact_term, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o , double*gauge_field );

void co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (complex *w, fermion_propagator_type *r, fermion_propagator_type *s, double sign, unsigned int N);

void cvc_tensor_eo_subtract_contact_term (double**eo_tensor, double*contact_term, int gsx[4], int have_source );

int cvc_tensor_eo_momentum_projection (double****tensor_tp, double**tensor_eo, int (*momentum_list)[3], int momentum_number);

int cvc_tensor_tp_write_to_aff_file (double***cvc_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc );

int cvc_tensor_eo_check_wi_position_space (double **tensor_eo);

}  /* end of namespace cvc */

#endif
