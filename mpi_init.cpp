

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
MPI_Datatype propagator_point;
MPI_Datatype contraction_point;

MPI_Datatype gauge_time_slice_cont;
MPI_Datatype spinor_time_slice_cont;

MPI_Datatype eo_spinor_time_slice_cont;

MPI_Datatype contraction_time_slice_cont;

MPI_Datatype eo_propagator_time_slice_cont;

#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)

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

/* edges for spinor fields */

/* spinor x-t-edge */
MPI_Datatype spinor_xt_edge_vector;
MPI_Datatype spinor_xt_edge_cont;

/* spinor y-t-edge */
MPI_Datatype spinor_yt_edge_vector;
MPI_Datatype spinor_yt_edge_cont;

/* spinor y-x-edge */
MPI_Datatype spinor_yx_edge_vector;
MPI_Datatype spinor_yx_edge_cont;

/* spinor z-t-edge */
MPI_Datatype spinor_zt_edge_vector;
MPI_Datatype spinor_zt_edge_cont;

/* spinor z-x-edge */
MPI_Datatype spinor_zx_edge_vector;
MPI_Datatype spinor_zx_edge_cont;

/* spinor z-y-edge */
MPI_Datatype spinor_zy_edge_vector;
MPI_Datatype spinor_zy_edge_cont;

/* slices of even-odd spinors */

MPI_Datatype eo_spinor_x_slice_vector;
MPI_Datatype eo_spinor_x_subslice_cont;
MPI_Datatype eo_spinor_x_slice_cont;

MPI_Datatype eo_spinor_y_slice_vector;
MPI_Datatype eo_spinor_y_subslice_cont;
MPI_Datatype eo_spinor_y_slice_cont;

MPI_Datatype eo_spinor_z_subslice_cont;
MPI_Datatype eo_spinor_z_slice_cont;
MPI_Datatype eo_spinor_z_odd_fwd_slice_struct;
MPI_Datatype eo_spinor_z_even_fwd_slice_struct;
MPI_Datatype eo_spinor_z_odd_bwd_slice_struct;
MPI_Datatype eo_spinor_z_even_bwd_slice_struct;

/* slices of even-odd propagators */

MPI_Datatype eo_propagator_x_slice_vector;
MPI_Datatype eo_propagator_x_subslice_cont;
MPI_Datatype eo_propagator_x_slice_cont;

MPI_Datatype eo_propagator_y_slice_vector;
MPI_Datatype eo_propagator_y_subslice_cont;
MPI_Datatype eo_propagator_y_slice_cont;

MPI_Datatype eo_propagator_z_subslice_cont;
MPI_Datatype eo_propagator_z_slice_cont;
MPI_Datatype eo_propagator_z_odd_fwd_slice_struct;
MPI_Datatype eo_propagator_z_even_fwd_slice_struct;
MPI_Datatype eo_propagator_z_odd_bwd_slice_struct;
MPI_Datatype eo_propagator_z_even_bwd_slice_struct;

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
#endif  /* of ifdef HAVE_MPI */

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
  if(g_cart_id == 0) fprintf(stdout, "# [mpi_init] copying data from tmLQCD_mpi struct\n");
#endif

#ifdef HAVE_OPENMP
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

  MPI_Type_contiguous(LX*LY*LZ*24/2, MPI_DOUBLE, &eo_spinor_time_slice_cont);
  MPI_Type_commit(&eo_spinor_time_slice_cont);

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

  if(g_verbose > 0 ) {
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
  }

#else  /* of if !(defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ) */

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
  g_nproc_t = g_nproc / ( g_nproc_x * g_nproc_y * g_nproc_z );
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

  MPI_Type_contiguous(12, spinor_point, &propagator_point);
  MPI_Type_commit(&propagator_point);

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

  MPI_Type_contiguous(LX*LY*LZ/2, spinor_point, &eo_spinor_time_slice_cont);
  MPI_Type_commit(&eo_spinor_time_slice_cont);

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


  /* ========= the gauge edges ==================================================== */

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
  /* TEST */
  /* MPI_Type_vector(2, 1, T-1, gauge_z_subslice_cont, &gauge_zt_edge_vector); */
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

  /* ------------------------------------------------------------------------ */
  /* ------------------------------------------------------------------------ */
  /* ------------------------------------------------------------------------ */

  /* ========= the spinor edges ==================================================== */

  /* --------- x-t-edge ----------------------------------------------------- */
  MPI_Type_contiguous(2*LY*LZ, spinor_point, &spinor_xt_edge_cont);
  MPI_Type_commit(&spinor_xt_edge_cont);

  MPI_Type_vector(2, 1, T, spinor_x_subslice_cont, &spinor_xt_edge_vector);
  MPI_Type_commit(&spinor_xt_edge_vector);

  /* --------- y-t-edge ----------------------------------------------------- */

  MPI_Type_contiguous(2*LX*LZ, spinor_point, &spinor_yt_edge_cont);
  MPI_Type_commit(&spinor_yt_edge_cont);

  MPI_Type_vector(2, 1, T, spinor_y_subslice_cont, &spinor_yt_edge_vector);
  MPI_Type_commit(&spinor_yt_edge_vector);

  /* --------- y-x-edge ----------------------------------------------------- */

  MPI_Type_contiguous( 2*T*LZ, spinor_point, &spinor_yx_edge_cont);
  MPI_Type_commit(&spinor_yx_edge_cont);

  MPI_Type_vector(2*T, LZ, LX*LZ, spinor_point, &spinor_yx_edge_vector);
  MPI_Type_commit(&spinor_yx_edge_vector);


  /* --------- z-t-edge ----------------------------------------------------- */

  MPI_Type_contiguous( 2*LX*LY, spinor_point, &spinor_zt_edge_cont);
  MPI_Type_commit(&spinor_zt_edge_cont);

  MPI_Type_vector(2, 1, T, spinor_z_subslice_cont, &spinor_zt_edge_vector);
  /* TEST */
  /* MPI_Type_vector(2, 1, T-1, spinor_z_subslice_cont, &spinor_zt_edge_vector); */
  MPI_Type_commit(&spinor_zt_edge_vector);

  /* --------- z-x-edge ----------------------------------------------------- */

  MPI_Type_contiguous( 2*T*LY, spinor_point, &spinor_zx_edge_cont);
  MPI_Type_commit(&spinor_zx_edge_cont);

  MPI_Type_vector(2*T, LY, LX*LY, spinor_point, &spinor_zx_edge_vector);
  MPI_Type_commit(&spinor_zx_edge_vector);

  /* --------- z-y-edge ----------------------------------------------------- */

  MPI_Type_contiguous( 2*T*LX, spinor_point, &spinor_zy_edge_cont);
  MPI_Type_commit(&spinor_zy_edge_cont);

  MPI_Type_vector(2*T*LX, 1, LY, spinor_point, &spinor_zy_edge_vector);
  MPI_Type_commit(&spinor_zy_edge_vector);



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

  dims[0]=1; dims[1]=0; dims[2]=0; dims[3]=0;
  MPI_Cart_sub (g_cart_grid, dims, &g_tr_comm);
  MPI_Comm_size(g_tr_comm, &g_tr_nproc);
  MPI_Comm_rank(g_tr_comm, &g_tr_id);

  /* ------------------------------------------------------------------------ */

  dims[0]=1; dims[1]=0; dims[2]=1; dims[3]=1;
  MPI_Cart_sub (g_cart_grid, dims, &g_xs_comm);
  MPI_Comm_size(g_xs_comm, &g_xs_nproc);
  MPI_Comm_rank(g_xs_comm, &g_xs_id);

  if( g_verbose > 0 ) {
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
  }
#else  // of ifndef HAVE_QUDA


  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
  MPI_Get_processor_name(processor_name, &namelen);

  // determine the neighbours in +/-t-direction
  g_nproc_t = g_nproc / ( g_nproc_x * g_nproc_y * g_nproc_z);
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

  MPI_Type_contiguous(LX*LY*LZ/2, spinor_point, &eo_spinor_time_slice_cont);
  MPI_Type_commit(&eo_spinor_time_slice_cont);

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

  if( g_verbose > 0  ) {
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
  }

#endif  /* of ifndef HAVE_QUDA */

#endif  /* of ifdef PARALLELTX || PARALLELTXY || PARALLELTXYZ */

  fprintf(stdout, "# [mpi_init] proc%.2d on host %s\n", g_cart_id, processor_name);
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

  if( g_verbose > 0 ) {
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
  }
#endif  /* of ifdef HAVE_MPI */

}  /* end of mpi_init */

/******************************************************************************/

void mpi_init_xchange_eo_spinor(void) {

#ifdef HAVE_MPI


#if (defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ)
  int count, i;
  int *blocklengths = NULL;
  int x0, x1, x2, x3, ix, iix;
  MPI_Aint *displacements = NULL;
  MPI_Datatype *datatypes = NULL;
  MPI_Aint size_of_spinor_point;
  char errorString[400];
  int errorStringLength;
#endif


  /* ========= eo spinor slices ============================================= */

  /* ---------- t direction ------------------------------------------------- */
  MPI_Type_contiguous(LX*LY*LZ/2, spinor_point, &eo_spinor_time_slice_cont);
  MPI_Type_commit(&eo_spinor_time_slice_cont);


#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  /* ---------- x direction ------------------------------------------------- */
  MPI_Type_contiguous(T*LY*LZ/2, spinor_point, &eo_spinor_x_slice_cont);
  MPI_Type_commit(&eo_spinor_x_slice_cont);

  MPI_Type_contiguous(LY*LZ/2, spinor_point, &eo_spinor_x_subslice_cont);
  MPI_Type_commit(&eo_spinor_x_subslice_cont);

  MPI_Type_vector(T, 1, LX, eo_spinor_x_subslice_cont, &eo_spinor_x_slice_vector);
  MPI_Type_commit(&eo_spinor_x_slice_vector);

  /* ---------- y direction ------------------------------------------------- */
  MPI_Type_contiguous(T*LX*LZ/2, spinor_point, &eo_spinor_y_slice_cont);
  MPI_Type_commit(&eo_spinor_y_slice_cont);
  
  MPI_Type_contiguous(LZ/2, spinor_point, &eo_spinor_y_subslice_cont);
  MPI_Type_commit(&eo_spinor_y_subslice_cont);

  MPI_Type_vector(T*LX, 1, LY, eo_spinor_y_subslice_cont, &eo_spinor_y_slice_vector);
  MPI_Type_commit(&eo_spinor_y_slice_vector);

/*
 * TEST
  MPI_Type_vector(T*LX, LZ/2, LY*LZ/2, spinor_point, &eo_spinor_y_slice_vector);
  MPI_Type_commit(&eo_spinor_y_slice_vector);
*/

  /* ---------- z direction ------------------------------------------------- */
  MPI_Type_contiguous(T*LX*LY/2, spinor_point, &eo_spinor_z_slice_cont);
  MPI_Type_commit(&eo_spinor_z_slice_cont);

/*
  MPI_Type_contiguous(LX*LY, spinor_point, &eo_spinor_z_subslice_cont);
*/

  MPI_Type_vector( LY/2, 1, LZ, spinor_point, &eo_spinor_z_subslice_cont );
  MPI_Type_commit(&eo_spinor_z_subslice_cont);

/*
  MPI_Type_vector(T*LX/2, 1, LY*LZ, eo_spinor_z_subslice_cont, &eo_spinor_z_slice_vector);
  MPI_Type_commit(&eo_spinor_z_slice_vector);
*/
/*
  int MPI_Type_struct(int count, 
                      const int *array_of_blocklengths,
                      const MPI_Aint *array_of_displacements,
                      const MPI_Datatype *array_of_types,
                      MPI_Datatype *newtype)
*/


  count = T * LX * LY / 2;
  MPI_Type_extent(spinor_point, &size_of_spinor_point);
  if(g_cart_id == 0) fprintf(stdout, "# [mpi_init] size_of_spinor_point = %lu\n", size_of_spinor_point);

  if( (blocklengths = (int*)malloc(count*sizeof(int)) ) == NULL ) {
    fprintf(stderr, "[mpi_init] Error, count not allocate blocklengths\n");
    EXIT(12);
  }

  if( (displacements = (MPI_Aint*)malloc(count*sizeof(MPI_Aint)) ) == NULL ) {
    fprintf(stderr, "[mpi_init] Error, count not allocate displacements\n");
    EXIT(13);
  }

  if( (datatypes = (MPI_Datatype*)malloc(count*sizeof(MPI_Datatype)) ) == NULL ) {
    fprintf(stderr, "[mpi_init] Error, count not allocate datatypes\n");
    EXIT(14);
  }

  /* odd, backward z == 0 */

  x3 = 0;
  i = 0;
  for(x0=0; x0 < T;  x0++) {
  for(x1=0; x1 < LX; x1++) {
  for(x2=0; x2 < LY; x2++) {
        ix  = g_ipt[x0][x1][x2][x3];
        iix = g_lexic2eosub[ix];
        if(g_iseven[ix]) continue;

        datatypes    [i] = spinor_point;
        displacements[i] = (MPI_Aint)iix * size_of_spinor_point;
        blocklengths [i] = 1;

        i++;
      }
    }
  }

  /* TEST */
/*
  if(g_cart_id == 0) {
    fprintf(stdout, "# [mpi_init] odd fwd, count = %d\n", i);
    for(i=0; i<count; i++) {
      fprintf(stdout, "\t%3d%2d%8lu\n", i, blocklengths[i], displacements[i] / size_of_spinor_point);
    }
  }
*/


  i = MPI_Type_struct(count, blocklengths, displacements, datatypes, &eo_spinor_z_odd_bwd_slice_struct );
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_struct, status was %s\n", errorString);
    EXIT(15);
  }
  i = MPI_Type_commit(&eo_spinor_z_odd_bwd_slice_struct);
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_commit, status was %s\n", errorString);
    EXIT(15);
  }

  /* even, backward z == 0 */

  x3 = 0;
  i = 0;
  for(x0=0; x0 < T;  x0++) {
  for(x1=0; x1 < LX; x1++) {
  for(x2=0; x2 < LY; x2++) {
    ix  = g_ipt[x0][x1][x2][x3];
    iix = g_lexic2eosub[ix];
    if(!g_iseven[ix]) continue;

    datatypes    [i] = spinor_point;
    displacements[i] = (MPI_Aint)iix * size_of_spinor_point;
    blocklengths [i] = 1;

    i++;
  }}}

  i = MPI_Type_struct(count, blocklengths, displacements, datatypes, &eo_spinor_z_even_bwd_slice_struct );
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_struct, status was %s\n", errorString);
    EXIT(16);
  }
  i = MPI_Type_commit(&eo_spinor_z_even_bwd_slice_struct);
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_commit, status was %s\n", errorString);
    EXIT(17);
  }


  /* odd, forward z == LZ-1 */

  x3 = LZ-1;
  i = 0;
  for(x0=0; x0 < T;  x0++) {
  for(x1=0; x1 < LX; x1++) {
  for(x2=0; x2 < LY; x2++) {
    ix  = g_ipt[x0][x1][x2][x3];
    iix = g_lexic2eosub[ix];
    if(g_iseven[ix]) continue;

    datatypes    [i] = spinor_point;
    displacements[i] = (MPI_Aint)iix * size_of_spinor_point;
    blocklengths [i] = 1;

    i++;
  }}}

  i = MPI_Type_struct(count, blocklengths, displacements, datatypes, &eo_spinor_z_odd_fwd_slice_struct );
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_struct, status was %s\n", errorString);
    EXIT(18);
  }
  
  i = MPI_Type_commit(&eo_spinor_z_odd_fwd_slice_struct);
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_commit, status was %s\n", errorString);
    EXIT(19);
  }



  /* even, forward z == LZ-1 */

  x3 = LZ-1;
  i = 0;
  for(x0=0; x0 < T;  x0++) {
  for(x1=0; x1 < LX; x1++) {
  for(x2=0; x2 < LY; x2++) {
    ix  = g_ipt[x0][x1][x2][x3];
    iix = g_lexic2eosub[ix];
    if(!g_iseven[ix]) continue;

    datatypes    [i] = spinor_point;
    displacements[i] = (MPI_Aint)iix * size_of_spinor_point;
    blocklengths [i] = 1;

    i++;
  }}}

  i = MPI_Type_struct(count, blocklengths, displacements, datatypes, &eo_spinor_z_even_fwd_slice_struct );
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_struct, status was %s\n", errorString);
    EXIT(20);
  }

  i = MPI_Type_commit(&eo_spinor_z_even_fwd_slice_struct);
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_commit, status was %s\n", errorString);
    EXIT(21);
  }

#if 0
#endif  /* of if 0 */

  if(datatypes     != NULL) free( datatypes );
  if(displacements != NULL) free( displacements );
  if(blocklengths  != NULL) free( blocklengths );

#endif  /* of if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) */
#endif  /* of ifdef HAVE_MPI */


}   /* mpi_init_xchange_eo_spinor */

void mpi_fini_xchange_eo_spinor (void) {
#ifdef HAVE_MPI
  MPI_Type_free(&eo_spinor_time_slice_cont);

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  MPI_Type_free(&eo_spinor_x_slice_vector);
  MPI_Type_free(&eo_spinor_x_subslice_cont);
  MPI_Type_free(&eo_spinor_x_slice_cont);

  MPI_Type_free(&eo_spinor_y_slice_vector);
  MPI_Type_free(&eo_spinor_y_subslice_cont);
  MPI_Type_free(&eo_spinor_y_slice_cont);


  MPI_Type_free(&eo_spinor_z_slice_cont);
  MPI_Type_free(&eo_spinor_z_odd_fwd_slice_struct);
  MPI_Type_free(&eo_spinor_z_even_fwd_slice_struct);
  MPI_Type_free(&eo_spinor_z_odd_bwd_slice_struct);
  MPI_Type_free(&eo_spinor_z_even_bwd_slice_struct);
  MPI_Type_free(&eo_spinor_z_subslice_cont);

#endif
#endif
}  /* mpi_fini_xchange_eo_spinor */

  /******************************************************************************
   * init exchange data types for a fermion propagator field
   ******************************************************************************/
void mpi_init_xchange_eo_propagator(void) {

#ifdef HAVE_MPI

#if (defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ)
  int count, i;
  int *blocklengths = NULL;
  int x0, x1, x2, x3, ix, iix;
  MPI_Aint *displacements = NULL;
  MPI_Datatype *datatypes = NULL;
  MPI_Aint size_of_propagator_point;
  char errorString[400];
  int errorStringLength;
#endif


  /* ========= eo propagator slices ============================================= */

  /* ---------- t direction ------------------------------------------------- */
  MPI_Type_contiguous(LX*LY*LZ/2, propagator_point, &eo_propagator_time_slice_cont);
  MPI_Type_commit(&eo_propagator_time_slice_cont);


#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  /* ---------- x direction ------------------------------------------------- */
  MPI_Type_contiguous(T*LY*LZ/2, propagator_point, &eo_propagator_x_slice_cont);
  MPI_Type_commit(&eo_propagator_x_slice_cont);

  MPI_Type_contiguous(LY*LZ/2, propagator_point, &eo_propagator_x_subslice_cont);
  MPI_Type_commit(&eo_propagator_x_subslice_cont);

  MPI_Type_vector(T, 1, LX, eo_propagator_x_subslice_cont, &eo_propagator_x_slice_vector);
  MPI_Type_commit(&eo_propagator_x_slice_vector);

  /* ---------- y direction ------------------------------------------------- */
  MPI_Type_contiguous(T*LX*LZ/2, propagator_point, &eo_propagator_y_slice_cont);
  MPI_Type_commit(&eo_propagator_y_slice_cont);
  
  MPI_Type_contiguous(LZ/2, propagator_point, &eo_propagator_y_subslice_cont);
  MPI_Type_commit(&eo_propagator_y_subslice_cont);

  MPI_Type_vector(T*LX, 1, LY, eo_propagator_y_subslice_cont, &eo_propagator_y_slice_vector);
  MPI_Type_commit(&eo_propagator_y_slice_vector);

  /* ---------- z direction ------------------------------------------------- */
  MPI_Type_contiguous(T*LX*LY/2, propagator_point, &eo_propagator_z_slice_cont);
  MPI_Type_commit(&eo_propagator_z_slice_cont);

  MPI_Type_vector( LY/2, 1, LZ, propagator_point, &eo_propagator_z_subslice_cont );
  MPI_Type_commit(&eo_propagator_z_subslice_cont);


  count = T * LX * LY / 2;
  MPI_Type_extent(propagator_point, &size_of_propagator_point);
  if(g_cart_id == 0) fprintf(stdout, "# [mpi_init] size_of_propagator_point = %lu\n", size_of_propagator_point);

  if( (blocklengths = (int*)malloc(count*sizeof(int)) ) == NULL ) {
    fprintf(stderr, "[mpi_init] Error, count not allocate blocklengths\n");
    EXIT(12);
  }

  if( (displacements = (MPI_Aint*)malloc(count*sizeof(MPI_Aint)) ) == NULL ) {
    fprintf(stderr, "[mpi_init] Error, count not allocate displacements\n");
    EXIT(13);
  }

  if( (datatypes = (MPI_Datatype*)malloc(count*sizeof(MPI_Datatype)) ) == NULL ) {
    fprintf(stderr, "[mpi_init] Error, count not allocate datatypes\n");
    EXIT(14);
  }

  /* odd, backward z == 0 */

  x3 = 0;
  i = 0;
  for(x0=0; x0 < T;  x0++) {
  for(x1=0; x1 < LX; x1++) {
  for(x2=0; x2 < LY; x2++) {
        ix  = g_ipt[x0][x1][x2][x3];
        iix = g_lexic2eosub[ix];
        if(g_iseven[ix]) continue;

        datatypes    [i] = propagator_point;
        displacements[i] = (MPI_Aint)iix * size_of_propagator_point;
        blocklengths [i] = 1;

        i++;
      }
    }
  }

  i = MPI_Type_struct(count, blocklengths, displacements, datatypes, &eo_propagator_z_odd_bwd_slice_struct );
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_struct, status was %s\n", errorString);
    EXIT(15);
  }
  i = MPI_Type_commit(&eo_propagator_z_odd_bwd_slice_struct);
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_commit, status was %s\n", errorString);
    EXIT(15);
  }

  /* even, backward z == 0 */

  x3 = 0;
  i = 0;
  for(x0=0; x0 < T;  x0++) {
  for(x1=0; x1 < LX; x1++) {
  for(x2=0; x2 < LY; x2++) {
    ix  = g_ipt[x0][x1][x2][x3];
    iix = g_lexic2eosub[ix];
    if(!g_iseven[ix]) continue;

    datatypes    [i] = propagator_point;
    displacements[i] = (MPI_Aint)iix * size_of_propagator_point;
    blocklengths [i] = 1;

    i++;
  }}}

  i = MPI_Type_struct(count, blocklengths, displacements, datatypes, &eo_propagator_z_even_bwd_slice_struct );
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_struct, status was %s\n", errorString);
    EXIT(16);
  }
  i = MPI_Type_commit(&eo_propagator_z_even_bwd_slice_struct);
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_commit, status was %s\n", errorString);
    EXIT(17);
  }


  /* odd, forward z == LZ-1 */

  x3 = LZ-1;
  i = 0;
  for(x0=0; x0 < T;  x0++) {
  for(x1=0; x1 < LX; x1++) {
  for(x2=0; x2 < LY; x2++) {
    ix  = g_ipt[x0][x1][x2][x3];
    iix = g_lexic2eosub[ix];
    if(g_iseven[ix]) continue;

    datatypes    [i] = propagator_point;
    displacements[i] = (MPI_Aint)iix * size_of_propagator_point;
    blocklengths [i] = 1;

    i++;
  }}}

  i = MPI_Type_struct(count, blocklengths, displacements, datatypes, &eo_propagator_z_odd_fwd_slice_struct );
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_struct, status was %s\n", errorString);
    EXIT(18);
  }
  
  i = MPI_Type_commit(&eo_propagator_z_odd_fwd_slice_struct);
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_commit, status was %s\n", errorString);
    EXIT(19);
  }



  /* even, forward z == LZ-1 */

  x3 = LZ-1;
  i = 0;
  for(x0=0; x0 < T;  x0++) {
  for(x1=0; x1 < LX; x1++) {
  for(x2=0; x2 < LY; x2++) {
    ix  = g_ipt[x0][x1][x2][x3];
    iix = g_lexic2eosub[ix];
    if(!g_iseven[ix]) continue;

    datatypes    [i] = propagator_point;
    displacements[i] = (MPI_Aint)iix * size_of_propagator_point;
    blocklengths [i] = 1;

    i++;
  }}}

  i = MPI_Type_struct(count, blocklengths, displacements, datatypes, &eo_propagator_z_even_fwd_slice_struct );
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_struct, status was %s\n", errorString);
    EXIT(20);
  }

  i = MPI_Type_commit(&eo_propagator_z_even_fwd_slice_struct);
  if(i != MPI_SUCCESS ) {
    MPI_Error_string(i, errorString, &errorStringLength);
    fprintf(stderr, "[mpi_init] Error from MPI_Type_commit, status was %s\n", errorString);
    EXIT(21);
  }

  if(datatypes     != NULL) free( datatypes );
  if(displacements != NULL) free( displacements );
  if(blocklengths  != NULL) free( blocklengths );

#endif  /* of if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) */
#endif  /* of ifdef HAVE_MPI */


}   /* mpi_init_xchange_eo_propagator */

void mpi_fini_xchange_eo_propagator (void) {
#ifdef HAVE_MPI
  MPI_Type_free(&eo_propagator_time_slice_cont);

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  MPI_Type_free(&eo_propagator_x_slice_vector);
  MPI_Type_free(&eo_propagator_x_subslice_cont);
  MPI_Type_free(&eo_propagator_x_slice_cont);

  MPI_Type_free(&eo_propagator_y_slice_vector);
  MPI_Type_free(&eo_propagator_y_subslice_cont);
  MPI_Type_free(&eo_propagator_y_slice_cont);


  MPI_Type_free(&eo_propagator_z_slice_cont);
  MPI_Type_free(&eo_propagator_z_odd_fwd_slice_struct);
  MPI_Type_free(&eo_propagator_z_even_fwd_slice_struct);
  MPI_Type_free(&eo_propagator_z_odd_bwd_slice_struct);
  MPI_Type_free(&eo_propagator_z_even_bwd_slice_struct);
  MPI_Type_free(&eo_propagator_z_subslice_cont);
#endif
#endif  /* of ifdef HAVE_MPI */

}  /* mpi_fini_xchange_eo_propagator */




/******************************************************************************/

void mpi_init_xchange_contraction(int N) {
#ifdef HAVE_MPI
  MPI_Type_contiguous(N, MPI_DOUBLE, &contraction_point);
  MPI_Type_commit(&contraction_point);

  /* ======================================================================== */

  MPI_Type_contiguous(LX*LY*LZ, contraction_point, &contraction_time_slice_cont);
  MPI_Type_commit(&contraction_time_slice_cont);

  /* ------------------------------------------------------------------------ */
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
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
#endif  /* of if defined PARALLELT* */
#endif
}  /* end of mpi_init_xchange_contraction */

/****************************************************************
 * finalize / deallocate exchange data types for contraction
 ****************************************************************/
void mpi_fini_xchange_contraction (void) {
#ifdef HAVE_MPI
  MPI_Type_free(&contraction_time_slice_cont);
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  MPI_Type_free(&contraction_z_slice_cont);
  MPI_Type_free(&contraction_z_slice_vector);
  MPI_Type_free(&contraction_z_subslice_cont);
  MPI_Type_free(&contraction_y_slice_cont);
  MPI_Type_free(&contraction_y_slice_vector);
  MPI_Type_free(&contraction_y_subslice_cont);
  MPI_Type_free(&contraction_x_slice_cont);
  MPI_Type_free(&contraction_x_slice_vector);
  MPI_Type_free(&contraction_x_subslice_cont);
#endif
#endif
}  /* end of mpi_fini_xchange_contraction */

/****************************************************************
 * finalize / deallocate datatypes
 ****************************************************************/
void mpi_fini_datatypes (void) {
#ifdef HAVE_MPI
  MPI_Type_free(&gauge_point);
  MPI_Type_free(&gauge_time_slice_cont);
  MPI_Type_free(&spinor_point);
  MPI_Type_free(&spinor_time_slice_cont);
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  MPI_Type_free(&gauge_x_slice_cont);
  MPI_Type_free(&gauge_x_slice_vector);
  MPI_Type_free(&gauge_x_subslice_cont);
  MPI_Type_free(&gauge_xt_edge_cont);
  MPI_Type_free(&gauge_xt_edge_vector);
  MPI_Type_free(&gauge_y_slice_cont);
  MPI_Type_free(&gauge_y_slice_vector);
  MPI_Type_free(&gauge_y_subslice_cont);
  MPI_Type_free(&gauge_yt_edge_cont);
  MPI_Type_free(&gauge_yt_edge_vector);
  MPI_Type_free(&gauge_yx_edge_cont);
  MPI_Type_free(&gauge_yx_edge_vector);
  MPI_Type_free(&gauge_z_slice_cont);
  MPI_Type_free(&gauge_z_slice_vector);
  MPI_Type_free(&gauge_z_subslice_cont);
  MPI_Type_free(&gauge_zt_edge_cont);
  MPI_Type_free(&gauge_zt_edge_vector);
  MPI_Type_free(&gauge_zx_edge_cont);
  MPI_Type_free(&gauge_zx_edge_vector);
  MPI_Type_free(&gauge_zy_edge_cont);
  MPI_Type_free(&gauge_zy_edge_vector);

  MPI_Type_free(&spinor_x_slice_cont);
  MPI_Type_free(&spinor_x_slice_vector);
  MPI_Type_free(&spinor_x_subslice_cont);
  MPI_Type_free(&spinor_y_slice_cont);
  MPI_Type_free(&spinor_y_slice_vector);
  MPI_Type_free(&spinor_y_subslice_cont);
  MPI_Type_free(&spinor_z_slice_cont);
  MPI_Type_free(&spinor_z_slice_vector);
  MPI_Type_free(&spinor_z_subslice_cont);

  MPI_Type_free(&spinor_xt_edge_cont);
  MPI_Type_free(&spinor_xt_edge_vector);
  MPI_Type_free(&spinor_yt_edge_cont);
  MPI_Type_free(&spinor_yt_edge_vector);
  MPI_Type_free(&spinor_yx_edge_cont);
  MPI_Type_free(&spinor_yx_edge_vector);
  MPI_Type_free(&spinor_zt_edge_cont);
  MPI_Type_free(&spinor_zt_edge_vector);
  MPI_Type_free(&spinor_zx_edge_cont);
  MPI_Type_free(&spinor_zx_edge_vector);
  MPI_Type_free(&spinor_zy_edge_cont);
  MPI_Type_free(&spinor_zy_edge_vector);

#endif
#endif
}  /* end of mpi_fini_data_types */


/******************************************************************************
 *
 ******************************************************************************/
void mpi_init_xchanger (xchanger_type *x, int N ) {
#ifdef HAVE_MPI
  x->N = N;

  MPI_Type_contiguous(N, MPI_DOUBLE, &x->point);
  MPI_Type_commit(&x->point);

  /* ======================================================================== */

  MPI_Type_contiguous(LX*LY*LZ, x->point, &x->time_slice_cont);
  MPI_Type_commit(&x->time_slice_cont);

  /* ------------------------------------------------------------------------ */
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  MPI_Type_contiguous(LY*LZ, x->point, &x->x_subslice_cont);
  MPI_Type_commit(&x->x_subslice_cont);

  MPI_Type_vector(T, 1, LX, x->x_subslice_cont, &x->x_slice_vector);
  MPI_Type_commit(&x->x_slice_vector);

  MPI_Type_contiguous(T*LY*LZ, x->point, &x->x_slice_cont);
  MPI_Type_commit(&x->x_slice_cont);

  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LX*LZ, x->point, &x->y_subslice_cont);
  MPI_Type_commit(&x->y_subslice_cont);

  MPI_Type_vector(T*LX, LZ, LY*LZ, x->point, &x->y_slice_vector);
  MPI_Type_commit(&x->y_slice_vector);

  MPI_Type_contiguous(T*LX*LZ, x->point, &x->y_slice_cont);
  MPI_Type_commit(&x->y_slice_cont);

  /* ------------------------------------------------------------------------ */

  MPI_Type_contiguous(LX*LY, x->point, &x->z_subslice_cont);
  MPI_Type_commit(&x->z_subslice_cont);

  MPI_Type_vector(T*LX*LY, 1, LZ, x->point, &x->z_slice_vector);
  MPI_Type_commit(&x->z_slice_vector);

  MPI_Type_contiguous(T*LX*LY, x->point, &x->z_slice_cont);
  MPI_Type_commit(&x->z_slice_cont);

  /******************************************************************************
   * edges
   ******************************************************************************/
  MPI_Type_contiguous(2*LY*LZ, x->point, &x->xt_edge_cont);
  MPI_Type_commit(&x->xt_edge_cont);
  MPI_Type_vector(2, 1, T, x->x_subslice_cont, &x->xt_edge_vector);
  MPI_Type_commit(&x->xt_edge_vector);

  MPI_Type_contiguous(2*LX*LZ, x->point, &x->yt_edge_cont);
  MPI_Type_commit(&x->yt_edge_cont);
  MPI_Type_vector(2, 1, T, x->y_subslice_cont, &x->yt_edge_vector);
  MPI_Type_commit(&x->yt_edge_vector);

  MPI_Type_contiguous( 2*T*LZ, x->point, &x->yx_edge_cont);
  MPI_Type_commit(&x->yx_edge_cont);
  MPI_Type_vector(2*T, LZ, LX*LZ, x->point, &x->yx_edge_vector);
  MPI_Type_commit(&x->yx_edge_vector);

  MPI_Type_contiguous( 2*LX*LY, x->point, &x->zt_edge_cont);
  MPI_Type_commit(&x->zt_edge_cont);
  MPI_Type_vector(2, 1, T, x->z_subslice_cont, &x->zt_edge_vector);
  MPI_Type_commit(&x->zt_edge_vector);

  MPI_Type_contiguous( 2*T*LY, x->point, &x->zx_edge_cont);
  MPI_Type_commit(&x->zx_edge_cont);
  MPI_Type_vector(2*T, LY, LX*LY, x->point, &x->zx_edge_vector);
  MPI_Type_commit(&x->zx_edge_vector);

  MPI_Type_contiguous( 2*T*LX, x->point, &x->zy_edge_cont);
  MPI_Type_commit(&x->zy_edge_cont);
  MPI_Type_vector(2*T*LX, 1, LY, x->point, &x->zy_edge_vector);
  MPI_Type_commit(&x->zy_edge_vector);

#endif  /* of if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) */
#endif
}  /* end of mpi_init_xchanger */

/******************************************************************************/
/******************************************************************************/

/******************************************************************************
 *
 ******************************************************************************/
void mpi_fini_xchanger ( xchanger_type *x) {
#ifdef HAVE_MPI
  x->N = 0;

  MPI_Type_free( &x->point           );
  MPI_Type_free( &x->time_slice_cont );
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  MPI_Type_free( &x->z_slice_cont    );
  MPI_Type_free( &x->z_slice_vector  );
  MPI_Type_free( &x->z_subslice_cont );
  MPI_Type_free( &x->y_slice_cont    );
  MPI_Type_free( &x->y_slice_vector  );
  MPI_Type_free( &x->y_subslice_cont );
  MPI_Type_free( &x->x_slice_cont    );
  MPI_Type_free( &x->x_slice_vector  );
  MPI_Type_free( &x->x_subslice_cont );

  MPI_Type_free( &x->xt_edge_cont    );
  MPI_Type_free( &x->xt_edge_vector  );
  MPI_Type_free( &x->yt_edge_cont    );
  MPI_Type_free( &x->yt_edge_vector  );
  MPI_Type_free( &x->yx_edge_cont    );
  MPI_Type_free( &x->yx_edge_vector  );
  MPI_Type_free( &x->zt_edge_cont    );
  MPI_Type_free( &x->zt_edge_vector  );
  MPI_Type_free( &x->zx_edge_cont    );
  MPI_Type_free( &x->zx_edge_vector  );
  MPI_Type_free( &x->zy_edge_cont    );
  MPI_Type_free( &x->zy_edge_vector  );
#endif
#endif
}  /* mpi_fini_xchanger */

/******************************************************************************/
/******************************************************************************/

/******************************************************************************
 * general xchanger, including edges
 ******************************************************************************/
void mpi_xchanger ( double *phi, xchanger_type *p ) {

#ifdef HAVE_MPI
  int cntr=0;
  int N = p->N;

  MPI_Request request[120];
  MPI_Status status[120];

  MPI_Isend(&phi[0],        1, p->time_slice_cont, g_nb_t_dn, 83, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*VOLUME], 1, p->time_slice_cont, g_nb_t_up, 83, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(T-1)*LX*LY*LZ], 1, p->time_slice_cont, g_nb_t_up, 84, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(T+1)*LX*LY*LZ], 1, p->time_slice_cont, g_nb_t_dn, 84, g_cart_grid, &request[cntr]);
  cntr++;

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  MPI_Isend(&phi[0],                             1, p->x_slice_vector, g_nb_x_dn, 85, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*LX*LY*LZ)],         1, p->x_slice_cont,   g_nb_x_up, 85, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(LX-1)*LY*LZ],                1, p->x_slice_vector, g_nb_x_up, 86, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, p->x_slice_cont,   g_nb_x_dn, 86, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  cntr = 0;

  MPI_Isend(&phi[N*(VOLUME+2*LX*LY*LZ)],             1, p->xt_edge_vector, g_nb_t_dn, 87, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND)],                   1, p->xt_edge_cont, g_nb_t_up, 87, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(VOLUME+2*LX*LY*LZ+(T-1)*LY*LZ)], 1, p->xt_edge_vector, g_nb_t_up, 88, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+2*LY*LZ)],           1, p->xt_edge_cont, g_nb_t_dn, 88, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXY || (defined PARALLELTXYZ)
  MPI_Isend(&phi[0],                                       1, p->y_slice_vector, g_nb_y_dn, 89, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)],         1, p->y_slice_cont, g_nb_y_up, 89, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(LY-1)*LZ],                             1, p->y_slice_vector, g_nb_y_up, 90, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+T*LX*LZ)], 1, p->y_slice_cont, g_nb_y_dn, 90, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  cntr = 0;

  MPI_Isend(&phi[N*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)],             1, p->yt_edge_vector, g_nb_t_dn, 91, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*LY*LZ)],                     1, p->yt_edge_cont, g_nb_t_up, 91, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(T-1)*LX*LZ)], 1, p->yt_edge_vector, g_nb_t_up, 92, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*LY*LZ+2*LX*LZ)],             1, p->yt_edge_cont, g_nb_t_dn, 92, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)],           1, p->yx_edge_vector, g_nb_x_dn, 93, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*LY*LZ+4*LX*LZ)],           1, p->yx_edge_cont, g_nb_x_up, 93, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(LX-1)*LZ)], 1, p->yx_edge_vector, g_nb_x_up, 94, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*LY*LZ+4*LX*LZ+2*T*LZ)],    1, p->yx_edge_cont, g_nb_x_dn, 94, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXYZ
  /* boundary faces */
  MPI_Isend(&phi[0],                                                     1, p->z_slice_vector, g_nb_z_dn, 95, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) )],        1, p->z_slice_cont, g_nb_z_up, 95, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(LZ-1)],                                              1, p->z_slice_vector, g_nb_z_up, 96, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*(LX*LY*LZ + T*LY*LZ + T*LX*LZ) + T*LX*LY)], 1, p->z_slice_cont, g_nb_z_dn, 96, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  /* boundary edges */

  cntr = 0;

  /* z-t edges */
  MPI_Isend(&phi[N*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )          )], 1, p->zt_edge_vector, g_nb_t_dn,  97, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ)               )], 1, p->zt_edge_cont,   g_nb_t_up,  97, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(VOLUME+2*(LX*LY*LZ + T*LY*LZ + T*LX*LZ)+(T-1)*LX*LY)], 1, p->zt_edge_vector, g_nb_t_up,  98, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ) + 2*LX*LY     )], 1, p->zt_edge_cont,   g_nb_t_dn,  98, g_cart_grid, &request[cntr]);
  cntr++;

  /* z-x edges */
  MPI_Isend(&phi[N*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )             )], 1, p->zx_edge_vector, g_nb_x_dn, 99, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY)          )], 1, p->zx_edge_cont,   g_nb_x_up, 99, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) + (LX-1)*LY )], 1, p->zx_edge_vector, g_nb_x_up, 100, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY) + 2*T*LY )], 1, p->zx_edge_cont,   g_nb_x_dn, 100, g_cart_grid, &request[cntr]);
  cntr++;

  /* z-y edges */
  MPI_Isend(&phi[N*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )                    )], 1, p->zy_edge_vector, g_nb_y_dn, 101, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY + T*LY)          )], 1, p->zy_edge_cont,   g_nb_y_up, 101, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) + (LY-1)           )], 1, p->zy_edge_vector, g_nb_y_up, 102, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY + T*LY) + 2*T*LX )], 1, p->zy_edge_cont,   g_nb_y_dn, 102, g_cart_grid, &request[cntr]);
  cntr++;

#endif

  MPI_Waitall(cntr, request, status);

#endif
}  /* end of mpi_xchanger */


}  /* end of namespace cvc */
