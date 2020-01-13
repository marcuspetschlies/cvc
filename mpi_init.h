#ifndef _MPI_INIT_H
#define _MPI_INIT_H

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

namespace cvc
{

#  ifdef HAVE_MPI
extern MPI_Datatype gauge_point;
extern MPI_Datatype spinor_point;
extern MPI_Datatype contraction_point;



extern MPI_Datatype gauge_time_slice_cont;
extern MPI_Datatype spinor_time_slice_cont;
extern MPI_Datatype eo_spinor_time_slice_cont;
extern MPI_Datatype contraction_time_slice_cont;
extern MPI_Datatype eo_propagator_time_slice_cont;

#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ

/* gauge slices */
extern MPI_Datatype gauge_x_slice_vector;
extern MPI_Datatype gauge_x_slice_cont;
extern MPI_Datatype gauge_x_subslice_cont;

extern MPI_Datatype gauge_y_slice_vector;
extern MPI_Datatype gauge_y_subslice_cont;
extern MPI_Datatype gauge_y_slice_cont;

extern MPI_Datatype gauge_z_slice_vector;
extern MPI_Datatype gauge_z_subslice_cont;
extern MPI_Datatype gauge_z_slice_cont;
/* spinor slices */

extern MPI_Datatype spinor_x_slice_vector;
extern MPI_Datatype spinor_x_slice_cont;
extern MPI_Datatype spinor_x_subslice_cont;

extern MPI_Datatype spinor_y_slice_vector;
extern MPI_Datatype spinor_y_subslice_cont;
extern MPI_Datatype spinor_y_slice_cont;

extern MPI_Datatype spinor_z_slice_vector;
extern MPI_Datatype spinor_z_subslice_cont;
extern MPI_Datatype spinor_z_slice_cont;

/* slices for even-odd spinors */

extern MPI_Datatype eo_spinor_x_slice_vector;
extern MPI_Datatype eo_spinor_x_slice_cont;
extern MPI_Datatype eo_spinor_x_subslice_cont;

extern MPI_Datatype eo_spinor_y_slice_vector;
extern MPI_Datatype eo_spinor_y_subslice_cont;
extern MPI_Datatype eo_spinor_y_slice_cont;

extern MPI_Datatype eo_spinor_z_subslice_cont;
extern MPI_Datatype eo_spinor_z_slice_cont;
extern MPI_Datatype eo_spinor_z_odd_fwd_slice_struct;
extern MPI_Datatype eo_spinor_z_even_fwd_slice_struct;
extern MPI_Datatype eo_spinor_z_odd_bwd_slice_struct;
extern MPI_Datatype eo_spinor_z_even_bwd_slice_struct;

/* slices for even-odd propagator */

extern MPI_Datatype eo_propagator_x_slice_vector;
extern MPI_Datatype eo_propagator_x_slice_cont;
extern MPI_Datatype eo_propagator_x_subslice_cont;

extern MPI_Datatype eo_propagator_y_slice_vector;
extern MPI_Datatype eo_propagator_y_subslice_cont;
extern MPI_Datatype eo_propagator_y_slice_cont;

extern MPI_Datatype eo_propagator_z_subslice_cont;
extern MPI_Datatype eo_propagator_z_slice_cont;
extern MPI_Datatype eo_propagator_z_odd_fwd_slice_struct;
extern MPI_Datatype eo_propagator_z_even_fwd_slice_struct;
extern MPI_Datatype eo_propagator_z_odd_bwd_slice_struct;
extern MPI_Datatype eo_propagator_z_even_bwd_slice_struct;


/* edges for gauge field */

extern MPI_Datatype gauge_xt_edge_vector;
extern MPI_Datatype gauge_xt_edge_cont;
extern MPI_Datatype gauge_yt_edge_vector;
extern MPI_Datatype gauge_yt_edge_cont;
extern MPI_Datatype gauge_yx_edge_vector;
extern MPI_Datatype gauge_yx_edge_cont;

extern MPI_Datatype gauge_zt_edge_vector;
extern MPI_Datatype gauge_zt_edge_cont;
extern MPI_Datatype gauge_zx_edge_vector;
extern MPI_Datatype gauge_zx_edge_cont;
extern MPI_Datatype gauge_zy_edge_vector;
extern MPI_Datatype gauge_zy_edge_cont;

/* edges for spinor fields */

extern MPI_Datatype spinor_xt_edge_vector;
extern MPI_Datatype spinor_xt_edge_cont;

extern MPI_Datatype spinor_yt_edge_vector;
extern MPI_Datatype spinor_yt_edge_cont;

extern MPI_Datatype spinor_yx_edge_vector;
extern MPI_Datatype spinor_yx_edge_cont;

extern MPI_Datatype spinor_zt_edge_vector;
extern MPI_Datatype spinor_zt_edge_cont;

extern MPI_Datatype spinor_zx_edge_vector;
extern MPI_Datatype spinor_zx_edge_cont;

extern MPI_Datatype spinor_zy_edge_vector;
extern MPI_Datatype spinor_zy_edge_cont;


/* slices for contraction fields  */
extern MPI_Datatype contraction_x_slice_vector;
extern MPI_Datatype contraction_x_subslice_cont;
extern MPI_Datatype contraction_x_slice_cont;

extern MPI_Datatype contraction_y_slice_vector;
extern MPI_Datatype contraction_y_subslice_cont;
extern MPI_Datatype contraction_y_slice_cont;

extern MPI_Datatype contraction_z_slice_vector;
extern MPI_Datatype contraction_z_subslice_cont;
extern MPI_Datatype contraction_z_slice_cont;



#  endif
#  endif

void mpi_init(int argc, char *argv[]);
void mpi_fini_datatypes (void);

void mpi_init_xchange_contraction(int N);
void mpi_fini_xchange_contraction(void);

void mpi_init_xchange_eo_spinor (void);
void mpi_fini_xchange_eo_spinor (void);

void mpi_init_xchange_eo_propagator(void);
void mpi_fini_xchange_eo_propagator(void);


typedef struct {
  int N;
#ifdef HAVE_MPI
  MPI_Datatype point;

  MPI_Datatype time_slice_cont;

#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ

  MPI_Datatype x_slice_vector;
  MPI_Datatype x_subslice_cont;
  MPI_Datatype x_slice_cont;

  MPI_Datatype y_slice_vector;
  MPI_Datatype y_subslice_cont;
  MPI_Datatype y_slice_cont;

  MPI_Datatype z_slice_vector;
  MPI_Datatype z_subslice_cont;
  MPI_Datatype z_slice_cont;

  MPI_Datatype xt_edge_vector;
  MPI_Datatype xt_edge_cont;
  MPI_Datatype yt_edge_vector;
  MPI_Datatype yt_edge_cont;
  MPI_Datatype yx_edge_vector;
  MPI_Datatype yx_edge_cont;
  MPI_Datatype zt_edge_vector;
  MPI_Datatype zt_edge_cont;
  MPI_Datatype zx_edge_vector;
  MPI_Datatype zx_edge_cont;
  MPI_Datatype zy_edge_vector;
  MPI_Datatype zy_edge_cont;
#endif
#endif

} xchanger_type;

void mpi_init_xchanger ( xchanger_type *x, int N);
void mpi_fini_xchanger ( xchanger_type *x);
void mpi_xchanger ( double *phi, xchanger_type *p );


}  /* end of namespace cvc */
#endif
