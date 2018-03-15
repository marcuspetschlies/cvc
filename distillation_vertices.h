#ifndef _DISTILLATION_VERTICES_H
#define _DISTILLATION_VERTICES_H

/***************************************************
 * distillation_vertices.h
 ***************************************************/

namespace cvc {

/***********************************************************************************************/
/***********************************************************************************************/

/***********************************************************************************************
 * calculate V^+ D^d_k V, where
 ***********************************************************************************************/
int distillation_vertex_displacement ( double**V, int numV, int momentum_number, int (*momentum_list)[3], char*prefix, char*tag, int io_proc, double *gauge_field, int timeslice );
 
int distillation_vertex_vdagw ( double ** const vv, double ** const V, double ** const W, int const numV, int const numW );

/***********************************************************************************************/
/***********************************************************************************************/

}  /* end of namespace cvc */

#endif
