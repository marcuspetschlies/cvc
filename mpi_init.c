

#include <stdlib.h>
#include <stdio.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#include "cvc_complex.h"
#include "global.h"
#include "mpi_init.h"

namespace cvc {

#ifdef HAVE_MPI
MPI_Datatype gauge_point;
MPI_Datatype spinor_point;
MPI_Datatype contraction_point;

MPI_Datatype gauge_time_slice_cont;
MPI_Datatype spinor_time_slice_cont;

MPI_Datatype contraction_time_slice_cont;
#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ

/* gauge slices */
MPI_Datatype gauge_x_slice_vector;
MPI_Datatype gauge_x_subslice_cont;
MPI_Datatype gauge_x_slice_cont;

MPI_Datatype gauge_y_slice_vector;
MPI_Datatype gauge_y_subslice_cont;
MPI_Datatype gauge_y_slice_cont;

MPI_Datatype gauge_z_slice_vector;
MPI_Datatype gauge_z_subslice_cont;
MPI_Datatype gauge_z_slice_cont;

/* gauge edges */

/* x - t - edges */
MPI_Datatype gauge_xt_edge_vector;
MPI_Datatype gauge_xt_edge_cont;

/* y - t - edges */
MPI_Datatype gauge_yt_edge_vector;
MPI_Datatype gauge_yt_edge_cont;

/* y - x - edges */
MPI_Datatype gauge_yx_edge_vector;
MPI_Datatype gauge_yx_edge_cont;

/* z - t - edges */
MPI_Datatype gauge_zt_edge_vector;
MPI_Datatype gauge_zt_edge_cont;

/* z - x - edges */
MPI_Datatype gauge_zx_edge_vector;
MPI_Datatype gauge_zx_edge_cont;

/* z - y - edges */
MPI_Datatype gauge_zy_edge_vector;
MPI_Datatype gauge_zy_edge_cont;


/* spinor slices */

MPI_Datatype spinor_x_slice_vector;
MPI_Datatype spinor_x_subslice_cont;
MPI_Datatype spinor_x_slice_cont;

MPI_Datatype spinor_y_slice_vector;
MPI_Datatype spinor_y_subslice_cont;
MPI_Datatype spinor_y_slice_cont;

MPI_Datatype spinor_z_slice_vector;
MPI_Datatype spinor_z_subslice_cont;
MPI_Datatype spinor_z_slice_cont;

/* contraction slices */

MPI_Datatype contraction_x_slice_vector;
MPI_Datatype contraction_x_subslice_cont;
MPI_Datatype contraction_x_slice_cont;

MPI_Datatype contraction_y_slice_vector;
MPI_Datatype contraction_y_subslice_cont;
MPI_Datatype contraction_y_slice_cont;

MPI_Datatype contraction_z_slice_vector;
MPI_Datatype contraction_z_subslice_cont;
MPI_Datatype contraction_z_slice_cont;

#  endif

#endif

void mpi_init(int argc,char *argv[]) {

#ifdef HAVE_MPI
  int reorder=1;
  int namelen;
  int dims[4], periods[4]={1,1,1,1};
  char processor_name[MPI_MAX_PROCESSOR_NAME];
#endif

#if (defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ) && !(defined HAVE_MPI)
  exit(555);
#endif

#if (defined HAVE_QUDA) && (defined HAVE_TMLQCD_LIBWRAPPER)
  exit(556);
#endif


#ifdef HAVE_TMLQCD_LIBWRAPPER

#ifdef HAVE_MPI
  g_nproc   = g_tmLQCD_mpi.nproc;
  g_nproc_t = g_tmLQCD_mpi.nproc_t;
  g_nproc_x = g_tmLQCD_mpi.nproc_x;
  g_nproc_y = g_tmLQCD_mpi.nproc_y;
  g_nproc_z = g_tmLQCD_mpi.nproc_z;
  g_proc_id = g_tmLQCD_mpi.proc_id;
  g_cart_id = g_tmLQCD_mpi.cart_id;
  g_proc_coords[0] = g_tmLQCD_mpi.proc_coords[0];
  g_proc_coords[1] = g_tmLQCD_mpi.proc_coords[1];
  g_proc_coords[2] = g_tmLQCD_mpi.proc_coords[2];
  g_proc_coords[3] = g_tmLQCD_mpi.proc_coords[3];
  MPI_Comm_dup(g_tmLQCD_mpi.cart_grid, &g_cart_grid);
#endif

#ifdef OPENMP
  g_num_threads = g_tmLQCD_mpi.omp_num_threads;
#endif

#endif  /* of HAVE_TMLQCD_LIBWRAPPER */

#ifdef HAVE_MPI

/****************************************************************
 * MPI defined 
 ****************************************************************/

#if !(defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ)

  /****************************************************************
   *  ( PARALLELTX || PARALLELTXY || PARALLELTXYZ ) NOT defined
   ****************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
#endif
  MPI_Get_processor_name(processor_name, &namelen);

  /* determine the neighbours in +/-t-direction */

#ifndef HAVE_TMLQCD_LIBWRAPPER
  g_nproc_t = g_nproc;
  g_nproc_x = 1;
  g_nproc_y = 1;
  g_nproc_z = 1;
#endif
  dims[0] = g_nproc_t;
  dims[1] = g_nproc_x;
  dims[2] = g_nproc_y;
  dims[3] = g_nproc_z;

#ifndef HAVE_TMLQCD_LIBWRAPPER
  MPI_Dims_create(g_nproc, 4, dims);

  MPI_Cart_create(MPI_COMM_WORLD, 4, dims, periods, reorder, &g_cart_grid);
  MPI_Comm_rank(g_cart_grid, &g_cart_id);
  MPI_Cart_coords(g_cart_grid, g_cart_id, 4, g_proc_coords);
#endif

  MPI_Cart_shift(g_cart_grid, 0, 1, &g_nb_t_dn, &g_nb_t_up);
  MPI_Cart_shift(g_cart_grid, 1, 1, &g_nb_x_dn, &g_nb_x_up);
  MPI_Cart_shift(g_cart_grid, 2, 1, &g_nb_y_dn, &g_nb_y_up);
  MPI_Cart_shift(g_cart_grid, 3, 1, &g_nb_z_dn, &g_nb_z_up);


  g_nb_list[0] = g_nb_t_up;
  g_nb_list[1] = g_nb_t_dn;

  g_nb_list[2] = g_nb_x_up;
  g_nb_list[3] = g_nb_x_dn;

  g_nb_list[4] = g_nb_y_up;
  g_nb_list[5] = g_nb_y_dn;

  g_nb_list[6] = g_nb_z_up;
  g_nb_list[7] = g_nb_z_dn;

  MPI_Type_contiguous(72, MPI_DOUBLE, &gauge_point);
  MPI_Type_commit(&gauge_point);
  MPI_Type_contiguous(LX*LY*LZ, gauge_point, &gauge_time_slice_cont);
  MPI_Type_commit(&gauge_time_slice_cont);

  MPI_Type_contiguous(LX*LY*LZ*24, MPI_DOUBLE, &spinor_time_slice_cont);
  MPI_Type_commit(&spinor_time_slice_cont);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  LX = g_tmLQCD_lat.LX ;
  LY = g_tmLQCD_lat.LY ;
  LZ = g_tmLQCD_lat.LZ;
  T  = g_tmLQCD_lat.T;
  LX_global = LX * g_tmLQCD_mpi.nproc_x;
  LY_global = LY * g_tmLQCD_mpi.nproc_y;
  LZ_global = LZ * g_tmLQCD_mpi.nproc_z;
  T_global  = T  * g_tmLQCD_mpi.nproc_t;
#else

  T = T_global / g_nproc;
  LX_global = LX;
  LY_global = LY;
  LZ_global = LZ;

#endif
  Tstart = g_proc_coords[0] * T;
  LXstart   = 0;
  LYstart   = 0;
  LZstart   = 0;

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
		  "# [%2d] g_nb_x_up = %3d\n"\
		  "# [%2d] g_nb_x_dn = %3d\n"\
		  "# [%2d] g_nb_y_up = %3d\n"\
		  "# [%2d] g_nb_y_dn = %3d\n"\
		  "# [%2d] g_nb_z_up = %3d\n"\
		  "# [%2d] g_nb_z_dn = %3d\n"\
                  "# [%2d] g_nb_list[0] = %3d\n"\
                  "# [%2d] g_nb_list[1] = %3d\n"\
                  "# [%2d] g_nb_list[2] = %3d\n"\
                  "# [%2d] g_nb_list[3] = %3d\n"\
                  "# [%2d] g_nb_list[4] = %3d\n"\
                  "# [%2d] g_nb_list[5] = %3d\n"\
                  "# [%2d] g_nb_list[6] = %3d\n"\
		  "# [%2d] g_nb_list[7] = %3d\n",
		  g_cart_id, g_cart_id, g_nproc,
		  g_cart_id, g_proc_id,
		  g_cart_id, g_cart_id,
		  g_cart_id, g_nb_t_up,
		  g_cart_id, g_nb_t_dn,
		  g_cart_id, g_nb_x_up,
		  g_cart_id, g_nb_x_dn,
		  g_cart_id, g_nb_y_up,
		  g_cart_id, g_nb_y_dn,
		  g_cart_id, g_nb_z_up,
		  g_cart_id, g_nb_z_dn,
		  g_cart_id, g_nb_list[0],
		  g_cart_id, g_nb_list[1],
		  g_cart_id, g_nb_list[2],
		  g_cart_id, g_nb_list[3],
		  g_cart_id, g_nb_list[4],
		  g_cart_id, g_nb_list[5],
		  g_cart_id, g_nb_list[6],
		  g_cart_id, g_nb_list[7]);

#else
  /*********************************************************
   *  PARALLELTX || PARALLELTXY || PARALLELTXYZ defined
   *********************************************************/

#ifndef HAVE_QUDA

#ifndef HAVE_TMLQCD_LIBWRAPPER
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
#endif
  MPI_Get_processor_name(processor_name, &namelen);

  /* determine the neighbours in +/-t-direction */
#ifndef HAVE_TMLQCD_LIBWRAPPER
  g_nproc_t = g_nproc / ( g_nproc_x * g_nproc_y );
  g_nproc_z = 1;
#endif
  dims[0] = g_nproc_t;
  dims[1] = g_nproc_x;
  dims[2] = g_nproc_y;
  dims[3] = g_nproc_z;

#ifndef HAVE_TMLQCD_LIBWRAPPER
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

  LZ_global = LZ;
  LZ        = LZ_global / g_nproc_z;
  LZstart   = g_proc_coords[3] * LZ;
#else

  LX = g_tmLQCD_lat.LX ;
  LY = g_tmLQCD_lat.LY ;
  LZ = g_tmLQCD_lat.LZ;
  T  = g_tmLQCD_lat.T;

  LX_global = LX * g_tmLQCD_mpi.nproc_x;
  LY_global = LY * g_tmLQCD_mpi.nproc_y;
  LZ_global = LZ * g_tmLQCD_mpi.nproc_z;
  T_global  = T  * g_tmLQCD_mpi.nproc_t;
#endif

  Tstart   = g_proc_coords[0] * T;
  LXstart  = g_proc_coords[1] * LX;
  LYstart  = g_proc_coords[2] * LY;
  LZstart  = g_proc_coords[3] * LZ;

  MPI_Cart_shift(g_cart_grid, 0, 1, &g_nb_t_dn, &g_nb_t_up);
  MPI_Cart_shift(g_cart_grid, 1, 1, &g_nb_x_dn, &g_nb_x_up);
  MPI_Cart_shift(g_cart_grid, 2, 1, &g_nb_y_dn, &g_nb_y_up);
  MPI_Cart_shift(g_cart_grid, 3, 1, &g_nb_z_dn, &g_nb_z_up);

  g_nb_list[0] = g_nb_t_up;
  g_nb_list[1] = g_nb_t_dn;

  g_nb_list[2] = g_nb_x_up;
  g_nb_list[3] = g_nb_x_dn;

  g_nb_list[4] = g_nb_y_up;
  g_nb_list[5] = g_nb_y_dn;

  g_nb_list[6] = g_nb_z_up;
  g_nb_list[7] = g_nb_z_dn;

  MPI_Type_contiguous(72, MPI_DOUBLE, &gauge_point);
  MPI_Type_commit(&gauge_point);

  MPI_Type_contiguous(24, MPI_DOUBLE, &spinor_point);
  MPI_Type_commit(&spinor_point);

  MPI_Type_contiguous(LX*LY*LZ, gauge_point, &gauge_time_slice_cont);
  MPI_Type_commit(&gauge_time_slice_cont);

  /* ------------------------------------------------------------------------ */
  /* x slices */

  MPI_Type_contiguous(LY*LZ, gauge_point, &gauge_x_subslice_cont);
  MPI_Type_commit(&gauge_x_subslice_cont);

  MPI_Type_contiguous(T*LY*LZ, gauge_point, &gauge_x_slice_cont);
  MPI_Type_commit(&gauge_x_slice_cont);

  MPI_Type_vector(T, 1, LX, gauge_x_subslice_cont, &gauge_x_slice_vector);
  MPI_Type_commit(&gauge_x_slice_vector);

  /* ------------------------------------------------------------------------ */
  /* y slices */

  MPI_Type_contiguous(LX*LZ, gauge_point, &gauge_y_subslice_cont);
  MPI_Type_commit(&gauge_y_subslice_cont);

  MPI_Type_contiguous(T*LX*LZ, gauge_point, &gauge_y_slice_cont);
  MPI_Type_commit(&gauge_y_slice_cont);

  MPI_Type_vector(T*LX, LZ, LY*LZ, gauge_point, &gauge_y_slice_vector);
  MPI_Type_commit(&gauge_y_slice_vector);

  /* ------------------------------------------------------------------------ */
  /* z slices */

  MPI_Type_contiguous(LX*LY, gauge_point, &gauge_z_subslice_cont);
  MPI_Type_commit(&gauge_z_subslice_cont);

  MPI_Type_contiguous(T*LX*LY, gauge_point, &gauge_z_slice_cont);
  MPI_Type_commit(&gauge_z_slice_cont);

  MPI_Type_vector(T*LX*LY, 1, LZ, gauge_point, &gauge_z_slice_vector);
  MPI_Type_commit(&gauge_z_slice_vector);


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
  
  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LX*LY, spinor_point, &spinor_z_subslice_cont);
  MPI_Type_commit(&spinor_z_subslice_cont);

  MPI_Type_vector(T*LX*LY, 1, LZ, spinor_point, &spinor_z_slice_vector);
  MPI_Type_commit(&spinor_z_slice_vector);

  MPI_Type_contiguous(T*LX*LY, spinor_point, &spinor_z_slice_cont);
  MPI_Type_commit(&spinor_z_slice_cont);

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


  /* --------- z-t-edge ----------------------------------------------------- */

  MPI_Type_contiguous( 2*LX*LY, gauge_point, &gauge_zt_edge_cont);
  MPI_Type_commit(&gauge_zt_edge_cont);

  MPI_Type_vector(2, 1, T, gauge_z_subslice_cont, &gauge_zt_edge_vector);
  MPI_Type_commit(&gauge_zt_edge_vector);

  /* --------- z-x-edge ----------------------------------------------------- */

  MPI_Type_contiguous( 2*T*LY, gauge_point, &gauge_zx_edge_cont);
  MPI_Type_commit(&gauge_zx_edge_cont);

  MPI_Type_vector(2*T, LY, LX*LY, gauge_point, &gauge_zx_edge_vector);
  MPI_Type_commit(&gauge_zx_edge_vector);

  /* --------- z-y-edge ----------------------------------------------------- */

  MPI_Type_contiguous( 2*T*LX, gauge_point, &gauge_zy_edge_cont);
  MPI_Type_commit(&gauge_zy_edge_cont);

  MPI_Type_vector(2*T*LX, 1, LY, gauge_point, &gauge_zy_edge_vector);
  MPI_Type_commit(&gauge_zy_edge_vector);

  /* --------- sub lattices ------------------------------------------------- */

  dims[0]=0; dims[1]=1; dims[2]=1; dims[3]=1;
  MPI_Cart_sub (g_cart_grid, dims, &g_ts_comm);
  MPI_Comm_size(g_ts_comm, &g_ts_nproc);
  MPI_Comm_rank(g_ts_comm, &g_ts_id);

  MPI_Cart_shift(g_ts_comm, 0, 1, &g_ts_nb_dn, &g_ts_nb_up);
  g_ts_nb_x_up = g_ts_nb_up;
  g_ts_nb_x_dn = g_ts_nb_dn;

  MPI_Cart_shift(g_ts_comm, 1, 1, &g_ts_nb_y_dn, &g_ts_nb_y_up);
  MPI_Cart_shift(g_ts_comm, 2, 1, &g_ts_nb_z_dn, &g_ts_nb_z_up);

  /* ------------------------------------------------------------------------ */

  dims[0]=1; dims[1]=0; dims[2]=1; dims[3]=1;
  MPI_Cart_sub (g_cart_grid, dims, &g_xs_comm);
  MPI_Comm_size(g_xs_comm, &g_xs_nproc);
  MPI_Comm_rank(g_xs_comm, &g_xs_id);

  fprintf(stdout, "# [%2d] MPI parameters:\n"\
                  "# [%2d] g_nproc   = %3d\n"\
                  "# [%2d] g_nproc_t = %3d\n"\
                  "# [%2d] g_nproc_x = %3d\n"\
                  "# [%2d] g_nproc_y = %3d\n"\
                  "# [%2d] g_nproc_z = %3d\n"\
		  "# [%2d] g_proc_id = %3d\n"\
		  "# [%2d] g_cart_id = %3d\n"\
		  "# [%2d] g_nb_t_up = %3d\n"\
		  "# [%2d] g_nb_t_dn = %3d\n"\
		  "# [%2d] g_nb_x_up = %3d\n"\
		  "# [%2d] g_nb_x_dn = %3d\n"\
		  "# [%2d] g_nb_y_up = %3d\n"\
		  "# [%2d] g_nb_y_dn = %3d\n"\
		  "# [%2d] g_nb_z_up = %3d\n"\
		  "# [%2d] g_nb_z_dn = %3d\n"\
                  "# [%2d] g_nb_list[0] = %3d\n"\
		  "# [%2d] g_nb_list[1] = %3d\n"\
                  "# [%2d] g_nb_list[2] = %3d\n"\
		  "# [%2d] g_nb_list[3] = %3d\n"\
                  "# [%2d] g_nb_list[4] = %3d\n"\
		  "# [%2d] g_nb_list[5] = %3d\n"\
                  "# [%2d] g_nb_list[6] = %3d\n"\
		  "# [%2d] g_nb_list[7] = %3d\n"\
                  "# [%2d] g_ts_nproc   = %3d\n"\
		  "# [%2d] g_ts_id      = %3d\n"\
                  "# [%2d] g_xs_nproc   = %3d\n"\
		  "# [%2d] g_xs_id      = %3d\n"\
		  "# [%2d] g_ts_nb_x_up = %3d\n"\
		  "# [%2d] g_ts_nb_x_dn = %3d\n"\
		  "# [%2d] g_ts_nb_y_up = %3d\n"\
		  "# [%2d] g_ts_nb_y_dn = %3d\n"\
		  "# [%2d] g_ts_nb_z_up = %3d\n"\
		  "# [%2d] g_ts_nb_z_dn = %3d\n",\
		  g_cart_id, g_cart_id, g_nproc,
                  g_cart_id, g_nproc_t,
                  g_cart_id, g_nproc_x,
                  g_cart_id, g_nproc_y,
                  g_cart_id, g_nproc_z,
		  g_cart_id, g_proc_id,
		  g_cart_id, g_cart_id,
		  g_cart_id, g_nb_t_up,
		  g_cart_id, g_nb_t_dn,
		  g_cart_id, g_nb_x_up,
		  g_cart_id, g_nb_x_dn,
		  g_cart_id, g_nb_y_up,
		  g_cart_id, g_nb_y_dn,
		  g_cart_id, g_nb_z_up,
		  g_cart_id, g_nb_z_dn,
		  g_cart_id, g_nb_list[0],
		  g_cart_id, g_nb_list[1],
		  g_cart_id, g_nb_list[2],
		  g_cart_id, g_nb_list[3],
		  g_cart_id, g_nb_list[4],
		  g_cart_id, g_nb_list[5],
		  g_cart_id, g_nb_list[6],
		  g_cart_id, g_nb_list[7],
                  g_cart_id, g_ts_nproc,
                  g_cart_id, g_ts_id,
                  g_cart_id, g_xs_nproc,
                  g_cart_id, g_xs_id,
		  g_cart_id, g_ts_nb_x_up,
		  g_cart_id, g_ts_nb_x_dn,
		  g_cart_id, g_ts_nb_y_up,
		  g_cart_id, g_ts_nb_y_dn,
		  g_cart_id, g_ts_nb_z_up,
		  g_cart_id, g_ts_nb_z_dn);
#else  // HAVE_QUDA


  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
  MPI_Get_processor_name(processor_name, &namelen);

  // determine the neighbours in +/-t-direction
  g_nproc_t = g_nproc / ( g_nproc_x * g_nproc_y );
  g_nproc_z = 1;
  dims[0] = g_nproc_z;
  dims[1] = g_nproc_y;
  dims[2] = g_nproc_x;
  dims[3] = g_nproc_t;

  MPI_Cart_create(MPI_COMM_WORLD, 4, dims, periods, reorder, &g_cart_grid);
  MPI_Comm_rank(g_cart_grid, &g_cart_id);
  MPI_Cart_coords(g_cart_grid, g_cart_id, 4, g_proc_coords);

  T        = T_global / g_nproc_t;
  Tstart   = g_proc_coords[3] * T;

  LX_global = LX;
  LX        = LX_global / g_nproc_x;
  LXstart   = g_proc_coords[2] * LX;

  LY_global = LY;
  LY        = LY_global / g_nproc_y;
  LYstart   = g_proc_coords[1] * LY;

  LZ_global = LZ;
  LZ        = LZ_global / g_nproc_z;
  LZstart   = g_proc_coords[0] * LZ;

  MPI_Cart_shift(g_cart_grid, 0, 1, &g_nb_z_dn, &g_nb_z_up);
  MPI_Cart_shift(g_cart_grid, 1, 1, &g_nb_y_dn, &g_nb_y_up);
  MPI_Cart_shift(g_cart_grid, 2, 1, &g_nb_x_dn, &g_nb_x_up);
  MPI_Cart_shift(g_cart_grid, 3, 1, &g_nb_t_dn, &g_nb_t_up);

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

  dims[0]=1; dims[1]=1; dims[2]=1; dims[3]=0;
  MPI_Cart_sub (g_cart_grid, dims, &g_ts_comm);
  MPI_Comm_size(g_ts_comm, &g_ts_nproc);
  MPI_Comm_rank(g_ts_comm, &g_ts_id);

  MPI_Cart_shift(g_ts_comm, 2, 1, &g_ts_nb_dn, &g_ts_nb_up);
  g_ts_nb_x_up = g_ts_nb_up;
  g_ts_nb_x_dn = g_ts_nb_dn;

  MPI_Cart_shift(g_ts_comm, 1, 1, &g_ts_nb_y_dn, &g_ts_nb_y_up);

  /* ------------------------------------------------------------------------ */

  dims[0]=0; dims[1]=0; dims[2]=0; dims[3]=1;
  MPI_Cart_sub (g_cart_grid, dims, &g_xs_comm);
  MPI_Comm_size(g_xs_comm, &g_xs_nproc);
  MPI_Comm_rank(g_xs_comm, &g_xs_id);

  fprintf(stdout, "# [%2d] MPI parameters:\n"\
                  "# [%2d] g_nproc   = %3d\n"\
                  "# [%2d] g_nproc_t = %3d\n"\
                  "# [%2d] g_nproc_x = %3d\n"\
                  "# [%2d] g_nproc_y = %3d\n"\
                  "# [%2d] g_nproc_z = %3d\n"\
		  "# [%2d] g_proc_id = %3d\n"\
		  "# [%2d] g_cart_id = %3d\n"\
		  "# [%2d] g_nb_t_up = %3d\n"\
		  "# [%2d] g_nb_t_dn = %3d\n"\
		  "# [%2d] g_nb_x_up = %3d\n"\
		  "# [%2d] g_nb_x_dn = %3d\n"\
		  "# [%2d] g_nb_y_up = %3d\n"\
		  "# [%2d] g_nb_y_dn = %3d\n"\
		  "# [%2d] g_nb_z_up = %3d\n"\
		  "# [%2d] g_nb_z_dn = %3d\n"\
                  "# [%2d] g_nb_list[0] = %3d\n"\
		  "# [%2d] g_nb_list[1] = %3d\n"\
                  "# [%2d] g_nb_list[2] = %3d\n"\
		  "# [%2d] g_nb_list[3] = %3d\n"\
                  "# [%2d] g_nb_list[4] = %3d\n"\
		  "# [%2d] g_nb_list[5] = %3d\n"\
                  "# [%2d] g_nb_list[6] = %3d\n"\
		  "# [%2d] g_nb_list[7] = %3d\n"\
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
                  g_cart_id, g_nproc_z,
		  g_cart_id, g_proc_id,
		  g_cart_id, g_cart_id,
		  g_cart_id, g_nb_t_up,
		  g_cart_id, g_nb_t_dn,
		  g_cart_id, g_nb_x_up,
		  g_cart_id, g_nb_x_dn,
		  g_cart_id, g_nb_y_up,
		  g_cart_id, g_nb_y_dn,
		  g_cart_id, g_nb_z_up,
		  g_cart_id, g_nb_z_dn,
		  g_cart_id, g_nb_list[0],
		  g_cart_id, g_nb_list[1],
		  g_cart_id, g_nb_list[2],
		  g_cart_id, g_nb_list[3],
		  g_cart_id, g_nb_list[4],
		  g_cart_id, g_nb_list[5],
		  g_cart_id, g_nb_list[6],
		  g_cart_id, g_nb_list[7],
                  g_cart_id, g_ts_nproc,
                  g_cart_id, g_ts_id,
                  g_cart_id, g_xs_nproc,
                  g_cart_id, g_xs_id,
		  g_cart_id, g_ts_nb_x_up,
		  g_cart_id, g_ts_nb_x_dn,
		  g_cart_id, g_ts_nb_y_up,
		  g_cart_id, g_ts_nb_y_dn);

#endif  /* of ifndef HAVE_QUDA */

#endif  /* of ifdef PARALLELTX || PARALLELTXY || PARALLELTXYZ */

  fprintf(stdout, "[mpi_init] proc%.2d one host %s\n", g_cart_id, processor_name);
#else

/****************************************************************
 * MPI NOT defined 
 ****************************************************************/

  g_nproc = 1;
  g_nproc_t = 1;
  g_nproc_x = 1;
  g_nproc_y = 1;
  g_nproc_z = 1;
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
  LZ_global = LZ;
  LZstart   = 0;
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
#endif  /* of ifdef HAVE_MPI */

}  /* end of mpi_init */

void mpi_init_xchange_contraction(int N) {

  MPI_Type_contiguous(2*N, MPI_DOUBLE, &contraction_point);
  MPI_Type_commit(&contraction_point);

  /* ======================================================================== */

  MPI_Type_contiguous(LX*LY*LZ, contraction_point, &contraction_time_slice_cont);
  MPI_Type_commit(&contraction_time_slice_cont);

  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LY*LZ, contraction_point, &contraction_x_subslice_cont);
  MPI_Type_commit(&contraction_x_subslice_cont);

  MPI_Type_vector(T, 1, LX, contraction_x_subslice_cont, &contraction_x_slice_vector);
  MPI_Type_commit(&contraction_x_slice_vector);

  MPI_Type_contiguous(T*LY*LZ, contraction_point, &contraction_x_slice_cont);
  MPI_Type_commit(&contraction_x_slice_cont);
  
  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LX*LZ, contraction_point, &contraction_y_subslice_cont);
  MPI_Type_commit(&contraction_y_subslice_cont);

  MPI_Type_vector(T*LX, LZ, LY*LZ, contraction_point, &contraction_y_slice_vector);
  MPI_Type_commit(&contraction_y_slice_vector);

  MPI_Type_contiguous(T*LX*LZ, contraction_point, &contraction_y_slice_cont);
  MPI_Type_commit(&contraction_y_slice_cont);
  
  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LX*LY, contraction_point, &contraction_z_subslice_cont);
  MPI_Type_commit(&contraction_z_subslice_cont);

  MPI_Type_vector(T*LX*LY, 1, LZ, contraction_point, &contraction_z_slice_vector);
  MPI_Type_commit(&contraction_z_slice_vector);

  MPI_Type_contiguous(T*LX*LY, contraction_point, &contraction_z_slice_cont);
  MPI_Type_commit(&contraction_z_slice_cont);

}  /* end of mpi_init_xchange_contraction */

void mpi_fini_xchange_contraction (void) {

  MPI_Type_free(&contraction_z_slice_cont);
  MPI_Type_free(&contraction_z_slice_vector);
  MPI_Type_free(&contraction_z_subslice_cont);
  MPI_Type_free(&contraction_y_slice_cont);
  MPI_Type_free(&contraction_y_slice_vector);
  MPI_Type_free(&contraction_y_subslice_cont);
  MPI_Type_free(&contraction_x_slice_cont);
  MPI_Type_free(&contraction_x_slice_vector);
  MPI_Type_free(&contraction_x_subslice_cont);
  MPI_Type_free(&contraction_time_slice_cont);

}

}  /* end of namespace cvc */
