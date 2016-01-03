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
extern MPI_Datatype contraction_time_slice_cont;

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

/* edges */

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

void mpi_init_xchange_contraction(int N);
void mpi_fini_xchange_contraction(void);
}
#endif
