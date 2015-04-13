#ifndef _MPI_INIT_H
#define _MPI_INIT_H

#  ifdef MPI
#    include <mpi.h>
#  endif

#  ifdef MPI
extern MPI_Datatype gauge_point;
extern MPI_Datatype gauge_time_slice_cont;
extern MPI_Datatype spinor_time_slice_cont;

#  if defined PARALLELTX || defined PARALLELTXY
extern MPI_Datatype gauge_x_slice_vector;
extern MPI_Datatype gauge_x_slice_cont;
extern MPI_Datatype gauge_x_subslice_cont;

extern MPI_Datatype gauge_y_slice_vector;
extern MPI_Datatype gauge_y_subslice_cont;
extern MPI_Datatype gauge_y_slice_cont;

extern MPI_Datatype spinor_x_slice_vector;
extern MPI_Datatype spinor_x_slice_cont;
extern MPI_Datatype spinor_x_subslice_cont;

extern MPI_Datatype spinor_y_slice_vector;
extern MPI_Datatype spinor_y_subslice_cont;
extern MPI_Datatype spinor_y_slice_cont;

extern MPI_Datatype gauge_xt_edge_vector;
extern MPI_Datatype gauge_xt_edge_cont;
extern MPI_Datatype gauge_yt_edge_vector;
extern MPI_Datatype gauge_yt_edge_cont;
extern MPI_Datatype gauge_yx_edge_vector;
extern MPI_Datatype gauge_yx_edge_cont;


#  endif
#  endif

void mpi_init(int argc, char *argv[]);

#endif
