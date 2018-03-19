#ifndef _CONTRACT_CVC_TENSOR_MIXED_H
#define _CONTRACT_CVC_TENSOR_MIXED_H

/****************************************************
 * contract_cvc_tensor_mixed.h
 ****************************************************/

namespace cvc {

/************************************************************
 * calculate gsp using t-blocks
 ************************************************************/
int contract_vdag_gloc_spinor_field (
    double**prop_list_e, 
    double**prop_list_o, 
    int nsf,
    double**V, 
    int numV, 
    int momentum_number, 
    int (*momentum_list)[3], 
    int gamma_id_number, 
    int*gamma_id_list, 
    struct AffWriter_s*affw, 
    char*tag, int io_proc, 
    double*gauge_field, 
    double **mzz[2], 
    double**mzzinv[2] );

/************************************************************
 * calculate V^+ cvc-vertex S
 ************************************************************/
int contract_vdag_cvc_spinor_field (double**prop_list_e, double**prop_list_o, int nsf, double**V, int numV, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

}  // end of namespace cvc

#endif
