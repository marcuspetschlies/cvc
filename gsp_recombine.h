/***************************************************
 * gsp_recombine.h                                     *
 ***************************************************/
#ifndef _GSP_RECOMBINE_H
#define _GSP_RECOMBINE_H
namespace cvc {

void gsp_tr_mat_weight ( double _Complex * const r , double _Complex *** const s , double * const w, int const numV, int const N );

void gsp_tr_mat_weight_mat_weight ( double _Complex * const r , 
  double _Complex *** const s1 ,
  double * const w1,
  double _Complex *** const s2 ,
  double * const w2, 
  int const numV, int const N
);

/***************************************************/
/***************************************************/

}  // end of namespace cvc
#endif
