/***************************************************
 * gsp.h                                     *
 ***************************************************/
#ifndef _GSP_H
#define _GSP_H
namespace cvc {

#if 0

int gsp_calculate_v_dag_gamma_p_w(double**V, double**W, int num, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, char*tag, int symmetric);

int gsp_calculate_v_dag_gamma_p_xi_block(double**V, double*W, int num, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, char*tag);

int gsp_calculate_v_dag_gamma_p_w_block_asym(double*gsp_out, double**V, double**W, int numV, int numW, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, char*tag, int eo);

int gsp_calculate_v_w_block_asym(double*gsp_out, double**V, double**W, unsigned int numV, unsigned int numW);

#endif

/***************************************************/
/***************************************************/

int gsp_calculate_v_dag_gamma_p_w_block(double**V, int numV, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, char*prefix, char*tag, int io_proc, \
       double *gauge_field, double **mzz[2], double **mzzinv[2] );


/***************************************************/
/***************************************************/

}  /* end of namespace cvc */
#endif
