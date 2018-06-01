/***************************************************
 * gsp_utils.h                                     *
 ***************************************************/
#ifndef _GSP_UTILS_H
#define _GSP_UTILS_H
namespace cvc {

int gsp_read_node (double ***gsp, int num, int momentum[3], int gamma_id, char*tag, int timeslice);

int gsp_write_eval(double * const eval, unsigned int const num, char * const filename_prefix, char * const tag);

int gsp_read_eval(double ** eval, unsigned int num, char * const filename_prefix, char * const tag);

int gsp_read_cvc_node (
  double _Complex ** const fac,
  unsigned int const numV,
  unsigned int const block_length,
  int const momentum[3],
  char * const type,
  int const mu,
  char * const prefix,
  char * const tag,
  int const timeslice,
  unsigned int const nt
);

int gsp_ft_p0_shift ( double _Complex * const s_out, double _Complex * const s_in, int const pvec[3], int const mu , int const nu, int const sign );

int gsp_read_cvc_mee_node (
  double _Complex **** const fac,
  unsigned int const numV,
  int const momentum[3],
  char * const prefix,
  char * const tag,
  int const timeslice
);

int gsp_read_cvc_mee_ct_node (
  double _Complex *** const fac,
  unsigned int const numV,
  int const momentum[3],
  char * const prefix,
  char * const tag,
  int const timeslice
);

int gsp_read_cvc_ct_node (
  double _Complex ** const fac,
  unsigned int const numV,
  char * const prefix,
  char * const tag,
  int const timeslice
);

/***************************************************/
/***************************************************/

}  // end of namespace cvc
#endif
