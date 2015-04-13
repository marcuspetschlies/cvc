#include <stdlib.h>
#include <stdio.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include "cvc_complex.h"
#include "global.h"
#include "mpi_init.h"

#ifdef MPI
MPI_Datatype gauge_point;
MPI_Datatype spinor_point;
MPI_Datatype gauge_time_slice_cont;
MPI_Datatype spinor_time_slice_cont;
#  if defined PARALLELTX || defined PARALLELTXY
MPI_Datatype gauge_x_slice_vector;
MPI_Datatype gauge_x_subslice_cont;
MPI_Datatype gauge_x_slice_cont;

MPI_Datatype gauge_y_slice_vector;
MPI_Datatype gauge_y_subslice_cont;
MPI_Datatype gauge_y_slice_cont;

MPI_Datatype spinor_x_slice_vector;
MPI_Datatype spinor_x_subslice_cont;
MPI_Datatype spinor_x_slice_cont;

MPI_Datatype spinor_y_slice_vector;
MPI_Datatype spinor_y_subslice_cont;
MPI_Datatype spinor_y_slice_cont;

MPI_Datatype gauge_xt_edge_vector;
MPI_Datatype gauge_xt_edge_cont;
MPI_Datatype gauge_yt_edge_vector;
MPI_Datatype gauge_yt_edge_cont;
MPI_Datatype gauge_yx_edge_vector;
MPI_Datatype gauge_yx_edge_cont;
#  endif

#endif

void mpi_init(int argc,char *argv[]) {

#if (defined PARALLELTX || defined PARALLELTXY) && !(defined MPI)
  exit(555);
#endif

#ifdef MPI
#if !(defined PARALLELTX || defined PARALLELTXY)

  int reorder=1;
  int namelen;
  int dims[1], periods[1]={1};
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
  MPI_Get_processor_name(processor_name, &namelen);

  /* determine the neighbours in +/-t-direction */
  dims[0] = g_nproc;
  g_nproc_t = g_nproc;
  g_nproc_x = 1;
  g_nproc_y = 1;

  MPI_Dims_create(g_nproc, 1, dims);

  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &g_cart_grid);
  MPI_Comm_rank(g_cart_grid, &g_cart_id);
  MPI_Cart_coords(g_cart_grid, g_cart_id, 1, g_proc_coords);

  MPI_Cart_shift(g_cart_grid, 0, 1, &g_nb_t_dn, &g_nb_t_up);

  g_nb_list[0] = g_nb_t_up;
  g_nb_list[1] = g_nb_t_dn;
  g_nb_list[2] = g_cart_id;
  g_nb_list[3] = g_cart_id;
  g_nb_list[4] = g_cart_id;
  g_nb_list[5] = g_cart_id;
  g_nb_list[6] = g_cart_id;
  g_nb_list[7] = g_cart_id;

  MPI_Type_contiguous(72, MPI_DOUBLE, &gauge_point);
  MPI_Type_commit(&gauge_point);
  MPI_Type_contiguous(LX*LY*LZ, gauge_point, &gauge_time_slice_cont);
  MPI_Type_commit(&gauge_time_slice_cont);

  MPI_Type_contiguous(LX*LY*LZ*24, MPI_DOUBLE, &spinor_time_slice_cont);
  MPI_Type_commit(&spinor_time_slice_cont);

  T = T_global / g_nproc;
  Tstart = g_proc_coords[0] * T;
  LX_global = LX;
  LXstart   = 0;
  LY_global = LY;
  LYstart   = 0;
  g_ts_id   = 0;
  g_xs_id   = 0;
  g_ts_nb_up = 0;
  g_ts_nb_dn = 0;

  fprintf(stdout, "# [%2d] MPI parameters:\n"\
                  "# [%2d] g_nproc   = %3d\n"\
		  "# [%2d] g_proc_id = %3d\n"\
		  "# [%2d] g_cart_id = %3d\n"\
		  "# [%2d] g_nb_t_up = %3d\n"\
		  "# [%2d] g_nb_t_dn = %3d\n"\
                  "# [%2d] g_nb_list[0] = %3d\n"\
		  "# [%2d] g_nb_list[1] = %3d\n",
		  g_cart_id, g_cart_id, g_nproc,
		  g_cart_id, g_proc_id,
		  g_cart_id, g_cart_id,
		  g_cart_id, g_nb_t_up,
		  g_cart_id, g_nb_t_dn,
		  g_cart_id, g_nb_list[0],
		  g_cart_id, g_nb_list[1]);

#else /* PARALLELTX || PARALLELTXY defined */

  int reorder=1;
  int namelen;
  int dims[4], periods[4]={1,1,1,1};
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
  MPI_Get_processor_name(processor_name, &namelen);

  /* determine the neighbours in +/-t-direction */
  g_nproc_t = g_nproc / ( g_nproc_x * g_nproc_y );
  g_nproc_z = 1;
  dims[0] = g_nproc_t;
  dims[1] = g_nproc_x;
  dims[2] = g_nproc_y;
  dims[3] = g_nproc_z;

  MPI_Cart_create(MPI_COMM_WORLD, 4, dims, periods, reorder, &g_cart_grid);
  MPI_Comm_rank(g_cart_grid, &g_cart_id);
  MPI_Cart_coords(g_cart_grid, g_cart_id, 4, g_proc_coords);

  T        = T_global / g_nproc_t;
  Tstart   = g_proc_coords[0] * T;

  LX_global = LX;
  LX        = LX_global / g_nproc_x;
  LXstart   = g_proc_coords[1] * LX;

  LY_global = LY;
  LY        = LY_global / g_nproc_y;
  LYstart   = g_proc_coords[2] * LY;

  MPI_Cart_shift(g_cart_grid, 0, 1, &g_nb_t_dn, &g_nb_t_up);
  MPI_Cart_shift(g_cart_grid, 1, 1, &g_nb_x_dn, &g_nb_x_up);
  MPI_Cart_shift(g_cart_grid, 2, 1, &g_nb_y_dn, &g_nb_y_up);

  g_nb_list[0] = g_nb_t_up;
  g_nb_list[1] = g_nb_t_dn;

  g_nb_list[2] = g_nb_x_up;
  g_nb_list[3] = g_nb_x_dn;

  g_nb_list[4] = g_nb_y_up;
  g_nb_list[5] = g_nb_y_dn;

  g_nb_list[6] = g_cart_id;
  g_nb_list[7] = g_cart_id;

  MPI_Type_contiguous(72, MPI_DOUBLE, &gauge_point);
  MPI_Type_commit(&gauge_point);

  MPI_Type_contiguous(24, MPI_DOUBLE, &spinor_point);
  MPI_Type_commit(&spinor_point);

  MPI_Type_contiguous(LX*LY*LZ, gauge_point, &gauge_time_slice_cont);
  MPI_Type_commit(&gauge_time_slice_cont);

  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LY*LZ, gauge_point, &gauge_x_subslice_cont);
  MPI_Type_commit(&gauge_x_subslice_cont);

  MPI_Type_contiguous(T*LY*LZ, gauge_point, &gauge_x_slice_cont);
  MPI_Type_commit(&gauge_x_slice_cont);

  MPI_Type_vector(T, 1, LX, gauge_x_subslice_cont, &gauge_x_slice_vector);
  MPI_Type_commit(&gauge_x_slice_vector);

  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LX*LZ, gauge_point, &gauge_y_subslice_cont);
  MPI_Type_commit(&gauge_y_subslice_cont);

  MPI_Type_contiguous(T*LX*LZ, gauge_point, &gauge_y_slice_cont);
  MPI_Type_commit(&gauge_y_slice_cont);

  MPI_Type_vector(T*LX, LZ, LY*LZ, gauge_point, &gauge_y_slice_vector);
  MPI_Type_commit(&gauge_y_slice_vector);

  /* ======================================================================== */

  MPI_Type_contiguous(LX*LY*LZ, spinor_point, &spinor_time_slice_cont);
  MPI_Type_commit(&spinor_time_slice_cont);

  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LY*LZ, spinor_point, &spinor_x_subslice_cont);
  MPI_Type_commit(&spinor_x_subslice_cont);

  MPI_Type_vector(T, 1, LX, spinor_x_subslice_cont, &spinor_x_slice_vector);
  MPI_Type_commit(&spinor_x_slice_vector);

  MPI_Type_contiguous(T*LY*LZ, spinor_point, &spinor_x_slice_cont);
  MPI_Type_commit(&spinor_x_slice_cont);
  
  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LX*LZ, spinor_point, &spinor_y_subslice_cont);
  MPI_Type_commit(&spinor_y_subslice_cont);

  MPI_Type_vector(T*LX, LZ, LY*LZ, spinor_point, &spinor_y_slice_vector);
  MPI_Type_commit(&spinor_y_slice_vector);

  MPI_Type_contiguous(T*LX*LZ, spinor_point, &spinor_y_slice_cont);
  MPI_Type_commit(&spinor_y_slice_cont);
  
  /* ========= the edges ==================================================== */

  /* --------- x-t-edge ----------------------------------------------------- */
  MPI_Type_contiguous(2*LY*LZ, gauge_point, &gauge_xt_edge_cont);
  MPI_Type_commit(&gauge_xt_edge_cont);

  MPI_Type_vector(2, 1, T, gauge_x_subslice_cont, &gauge_xt_edge_vector);
  MPI_Type_commit(&gauge_xt_edge_vector);

  /* --------- y-t-edge ----------------------------------------------------- */

  MPI_Type_contiguous(2*LX*LZ, gauge_point, &gauge_yt_edge_cont);
  MPI_Type_commit(&gauge_yt_edge_cont);

  MPI_Type_vector(2, 1, T, gauge_y_subslice_cont, &gauge_yt_edge_vector);
  MPI_Type_commit(&gauge_yt_edge_vector);

  /* --------- y-x-edge ----------------------------------------------------- */

  MPI_Type_contiguous( 2*T*LZ, gauge_point, &gauge_yx_edge_cont);
  MPI_Type_commit(&gauge_yx_edge_cont);

  MPI_Type_vector(2*T, LZ, LX*LZ, gauge_point, &gauge_yx_edge_vector);
  MPI_Type_commit(&gauge_yx_edge_vector);

  /* --------- sub lattices ------------------------------------------------- */

  dims[0]=0; dims[1]=1; dims[2]=1; dims[3]=1;
  MPI_Cart_sub (g_cart_grid, dims, &g_ts_comm);
  MPI_Comm_size(g_ts_comm, &g_ts_nproc);
  MPI_Comm_rank(g_ts_comm, &g_ts_id);

  MPI_Cart_shift(g_ts_comm, 0, 1, &g_ts_nb_dn, &g_ts_nb_up);
  g_ts_nb_x_up = g_ts_nb_up;
  g_ts_nb_x_dn = g_ts_nb_dn;

  MPI_Cart_shift(g_ts_comm, 1, 1, &g_ts_nb_y_dn, &g_ts_nb_y_up);

  /* ------------------------------------------------------------------------ */

  dims[0]=1; dims[1]=0; dims[2]=0; dims[3]=0;
  MPI_Cart_sub (g_cart_grid, dims, &g_xs_comm);
  MPI_Comm_size(g_xs_comm, &g_xs_nproc);
  MPI_Comm_rank(g_xs_comm, &g_xs_id);

  fprintf(stdout, "# [%2d] MPI parameters:\n"\
                  "# [%2d] g_nproc   = %3d\n"\
                  "# [%2d] g_nproc_t = %3d\n"\
                  "# [%2d] g_nproc_x = %3d\n"\
                  "# [%2d] g_nproc_y = %3d\n"\
		  "# [%2d] g_proc_id = %3d\n"\
		  "# [%2d] g_cart_id = %3d\n"\
                  "# [%2d] g_nproc_t = %3d\n"\
                  "# [%2d] g_nproc_x = %3d\n"\
		  "# [%2d] g_nb_t_up = %3d\n"\
		  "# [%2d] g_nb_t_dn = %3d\n"\
		  "# [%2d] g_nb_x_up = %3d\n"\
		  "# [%2d] g_nb_x_dn = %3d\n"\
		  "# [%2d] g_nb_y_up = %3d\n"\
		  "# [%2d] g_nb_y_dn = %3d\n"\
                  "# [%2d] g_nb_list[0] = %3d\n"\
		  "# [%2d] g_nb_list[1] = %3d\n"\
                  "# [%2d] g_nb_list[2] = %3d\n"\
		  "# [%2d] g_nb_list[3] = %3d\n"\
                  "# [%2d] g_nb_list[4] = %3d\n"\
		  "# [%2d] g_nb_list[5] = %3d\n"\
                  "# [%2d] g_ts_nproc   = %3d\n"\
		  "# [%2d] g_ts_id      = %3d\n"\
                  "# [%2d] g_xs_nproc   = %3d\n"\
		  "# [%2d] g_xs_id      = %3d\n"\
		  "# [%2d] g_ts_nb_x_up = %3d\n"\
		  "# [%2d] g_ts_nb_x_dn = %3d\n"\
		  "# [%2d] g_ts_nb_y_up = %3d\n"\
		  "# [%2d] g_ts_nb_y_dn = %3d\n",\
		  g_cart_id, g_cart_id, g_nproc,
                  g_cart_id, g_nproc_t,
                  g_cart_id, g_nproc_x,
                  g_cart_id, g_nproc_y,
		  g_cart_id, g_proc_id,
		  g_cart_id, g_cart_id,
                  g_cart_id, g_nproc_t, 
                  g_cart_id, g_nproc_x,
		  g_cart_id, g_nb_t_up,
		  g_cart_id, g_nb_t_dn,
		  g_cart_id, g_nb_x_up,
		  g_cart_id, g_nb_x_dn,
		  g_cart_id, g_nb_y_up,
		  g_cart_id, g_nb_y_dn,
		  g_cart_id, g_nb_list[0],
		  g_cart_id, g_nb_list[1],
		  g_cart_id, g_nb_list[2],
		  g_cart_id, g_nb_list[3],
		  g_cart_id, g_nb_list[4],
		  g_cart_id, g_nb_list[5],
                  g_cart_id, g_ts_nproc,
                  g_cart_id, g_ts_id,
                  g_cart_id, g_xs_nproc,
                  g_cart_id, g_xs_id,
		  g_cart_id, g_ts_nb_x_up,
		  g_cart_id, g_ts_nb_x_dn,
		  g_cart_id, g_ts_nb_y_up,
		  g_cart_id, g_ts_nb_y_dn);

#endif  /* of ifdef PARALLELTX || PARALLELTXY */

#else  /* MPI not defined */
  g_nproc = 1;
  g_nproc_t = 1;
  g_nproc_x = 1;
  g_nproc_y = 1;
  g_proc_id = 0;
  g_cart_id = 0;
  g_ts_id   = 0;
  g_xs_id   = 0;
  g_nb_t_up = 0;
  g_nb_t_dn = 0;
  g_nb_x_up = 0;
  g_nb_x_dn = 0;
  g_nb_y_up = 0;
  g_nb_y_dn = 0;
  g_ts_nb_up = 0;
  g_ts_nb_dn = 0;
  g_nb_list[0] = 0;
  g_nb_list[1] = 0;
  g_nb_list[2] = 0;
  g_nb_list[3] = 0;
  g_nb_list[4] = 0;
  g_nb_list[5] = 0;
  g_nb_list[6] = 0;
  g_nb_list[7] = 0;
  T         = T_global;
  Tstart    = 0;
  LX_global = LX;
  LXstart   = 0;
  LY_global = LY;
  LYstart   = 0;
  fprintf(stdout, "# [%2d] MPI parameters:\n"\
                  "# [%2d] g_nproc   = %3d\n"\
		  "# [%2d] g_proc_id = %3d\n"\
		  "# [%2d] g_cart_id = %3d\n"\
		  "# [%2d] g_nb_t_up = %3d\n"\
		  "# [%2d] g_nb_t_dn = %3d\n"\
                  "# [%2d] g_nb_list[0] = %3d\n"\
		  "# [%2d] g_nb_list[1] = %3d\n",
		  g_cart_id, g_cart_id, g_nproc,
		  g_cart_id, g_proc_id,
		  g_cart_id, g_cart_id,
		  g_cart_id, g_nb_t_up,
		  g_cart_id, g_nb_t_dn,
		  g_cart_id, g_nb_list[0],
		  g_cart_id, g_nb_list[1]);
#endif  /* of ifdef MPI */

}
