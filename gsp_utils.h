/***************************************************
 * gsp_utils.h                                     *
 ***************************************************/
#ifndef _GSP_UTILS_H
#define _GSP_UTILS_H
namespace cvc {

int gsp_read_node (double ***gsp, int num, int momentum[3], int gamma_id, char*tag, int timeslice);

int gsp_write_eval(double *eval, int num, char*tag);

int gsp_read_eval(double **eval, int num, char*tag);

/***************************************************/
/***************************************************/

}  /* end of namespace cvc */
#endif
