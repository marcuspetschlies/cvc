/***************************************************
 * cvc_utils.c                                     *
 ***************************************************/
 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "cvc_complex.h"
#include "global.h"
#include "ilinalg.h"
#include "cvc_geometry.h"
#include "mpi_init.h"
#include "propagator_io.h"
#include "get_index.h"
#include "read_input_parser.h"
#include "cvc_utils.h"
#include "ranlxd.h"
#include "Q_phi.h"
#include "scalar_products.h"

namespace cvc {

void EV_Hermitian_3x3_Matrix(double *M, double *lambda);

/*****************************************************
 * read the input file
 *****************************************************/
int read_input (char *filename) {

    
  FILE *fs;

  if((void*)(fs = fopen(filename, "r"))==NULL) {
#ifdef HAVE_MPI
     MPI_Abort(MPI_COMM_WORLD, 10);
     MPI_Finalize();
#endif
     exit(101);
  }

  fscanf(fs, "%d", &T_global);
  fscanf(fs, "%d", &LX);
  fscanf(fs, "%d", &LY);
  fscanf(fs, "%d", &LZ);
   
  fscanf(fs, "%d", &Nconf);
  fscanf(fs, "%lf", &g_kappa);
  fscanf(fs, "%lf", &g_mu);

  fscanf(fs, "%d", &g_sourceid);
  fscanf(fs, "%d", &g_sourceid2);
  fscanf(fs, "%d", &Nsave);
  
  fscanf(fs, "%d", &format);

  fscanf(fs, "%lf", &BCangle[0]);
  fscanf(fs, "%lf", &BCangle[1]);
  fscanf(fs, "%lf", &BCangle[2]);
  fscanf(fs, "%lf", &BCangle[3]);

  fscanf(fs, "%s", filename_prefix);
  fscanf(fs, "%s", filename_prefix2);
  fscanf(fs, "%s", gaugefilename_prefix);

  fscanf(fs, "%d", &g_resume);
  fscanf(fs, "%d", &g_subtract);

  fscanf(fs, "%d", &g_source_location);

  fclose(fs);

  return(0);
}


/*****************************************************
 * allocate gauge field
 *****************************************************/
int alloc_gauge_field(double **gauge, const int V) {
  *gauge = (double*)calloc(72*V, sizeof(double));
  if((void*)gauge == NULL) {
    fprintf(stderr, "could not allocate memory for gauge\n");
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 10);
    MPI_Finalize();
#endif
    exit(102);
  }
  return(0);
}

/*****************************************************
 * allocate gauge field
 *****************************************************/
int alloc_gauge_field_dbl(double **gauge, const int N) {
  *gauge = (double*)calloc(N, sizeof(double));
  if((void*)gauge == NULL) {
    fprintf(stderr, "could not allocate memory for gauge\n");
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 10);
    MPI_Finalize();
#endif
    exit(102);
  }
  return(0);
}

int alloc_gauge_field_flt(float **gauge, const int N) {
  *gauge = (float*)calloc(N, sizeof(float));
  if((void*)gauge == NULL) {
    fprintf(stderr, "could not allocate memory for gauge\n");
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 10);
    MPI_Finalize();
#endif
    exit(102);
  }
  return(0);
}

/*****************************************************
 * allocate spinor fields
 *****************************************************/
int alloc_spinor_field(double **s, const int V) {
  (*s) = (double*)calloc(24*V, sizeof(double));
  if((void*)(*s) == NULL) {
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 10);
    MPI_Finalize();
#endif
    exit(103);
  }
  return(0);
}

int alloc_spinor_field_dbl(double **s, const int N) {
  (*s) = (double*)calloc(N, sizeof(double));
  if((void*)(*s) == NULL) {
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 10);
    MPI_Finalize();
#endif
    exit(103);
  }
  return(0);
}

int alloc_spinor_field_flt(float **s, const int N) {
  (*s) = (float*)calloc(N, sizeof(float));
  if((void*)(*s) == NULL) {
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 10);
    MPI_Finalize();
#endif
    exit(103);
  }
  return(0);
}

/*****************************************************
 * exchange the global gauge field g_gauge_field
 *****************************************************/

void xchange_gauge() {

#ifdef HAVE_MPI
  int cntr=0;
  MPI_Request request[120];
  MPI_Status status[120];

  MPI_Isend(&g_gauge_field[0],         1, gauge_time_slice_cont, g_nb_t_dn, 83, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*VOLUME], 1, gauge_time_slice_cont, g_nb_t_up, 83, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(T-1)*LX*LY*LZ], 1, gauge_time_slice_cont, g_nb_t_up, 84, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(T+1)*LX*LY*LZ], 1, gauge_time_slice_cont, g_nb_t_dn, 84, g_cart_grid, &request[cntr]);
  cntr++;

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  MPI_Isend(&g_gauge_field[0],                              1, gauge_x_slice_vector, g_nb_x_dn, 85, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ)],         1, gauge_x_slice_cont,   g_nb_x_up, 85, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(LX-1)*LY*LZ],                1, gauge_x_slice_vector, g_nb_x_up, 86, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, gauge_x_slice_cont,   g_nb_x_dn, 86, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  cntr = 0;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ)], 1, gauge_xt_edge_vector, g_nb_t_dn, 87, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND)], 1, gauge_xt_edge_cont, g_nb_t_up, 87, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ+(T-1)*LY*LZ)], 1, gauge_xt_edge_vector, g_nb_t_up, 88, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+2*LY*LZ)], 1, gauge_xt_edge_cont, g_nb_t_dn, 88, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXY || (defined PARALLELTXYZ)
  MPI_Isend(&g_gauge_field[0], 1, gauge_y_slice_vector, g_nb_y_dn, 89, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_y_slice_cont, g_nb_y_up, 89, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(LY-1)*LZ], 1, gauge_y_slice_vector, g_nb_y_up, 90, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+T*LX*LZ)], 1, gauge_y_slice_cont, g_nb_y_dn, 90, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  cntr = 0;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_yt_edge_vector, g_nb_t_dn, 91, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*LY*LZ)], 1, gauge_yt_edge_cont, g_nb_t_up, 91, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(T-1)*LX*LZ)], 1, gauge_yt_edge_vector, g_nb_t_up, 92, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*LY*LZ+2*LX*LZ)], 1, gauge_yt_edge_cont, g_nb_t_dn, 92, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_yx_edge_vector, g_nb_x_dn, 93, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*LY*LZ+4*LX*LZ)], 1, gauge_yx_edge_cont, g_nb_x_up, 93, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(LX-1)*LZ)], 1, gauge_yx_edge_vector, g_nb_x_up, 94, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*LY*LZ+4*LX*LZ+2*T*LZ)], 1, gauge_yx_edge_cont, g_nb_x_dn, 94, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXYZ
  /* boundary faces */
  MPI_Isend(&g_gauge_field[0], 1, gauge_z_slice_vector, g_nb_z_dn, 95, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) )], 1, gauge_z_slice_cont, g_nb_z_up, 95, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(LZ-1)], 1, gauge_z_slice_vector, g_nb_z_up, 96, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+2*(LX*LY*LZ + T*LY*LZ + T*LX*LZ) + T*LX*LY)], 1, gauge_z_slice_cont, g_nb_z_dn, 96, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);


  /* boundary edges */

  cntr = 0;

  /* z-t edges */
  MPI_Isend(&g_gauge_field[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )          )], 1, gauge_zt_edge_vector, g_nb_t_dn,  97, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ)               )], 1, gauge_zt_edge_cont,   g_nb_t_up,  97, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*(LX*LY*LZ + T*LY*LZ + T*LX*LZ)+(T-1)*LX*LY)], 1, gauge_zt_edge_vector, g_nb_t_up,  98, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ) + 2*LX*LY     )], 1, gauge_zt_edge_cont,   g_nb_t_dn,  98, g_cart_grid, &request[cntr]);
  cntr++;

  /* z-x edges */
  MPI_Isend(&g_gauge_field[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )             )], 1, gauge_zx_edge_vector, g_nb_x_dn, 99, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY)          )], 1, gauge_zx_edge_cont,   g_nb_x_up, 99, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) + (LX-1)*LY )], 1, gauge_zx_edge_vector, g_nb_x_up, 100, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY) + 2*T*LY )], 1, gauge_zx_edge_cont,   g_nb_x_dn, 100, g_cart_grid, &request[cntr]);
  cntr++;

  /* z-y edges */
  MPI_Isend(&g_gauge_field[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )                    )], 1, gauge_zy_edge_vector, g_nb_y_dn, 101, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY + T*LY)          )], 1, gauge_zy_edge_cont,   g_nb_y_up, 101, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&g_gauge_field[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) + (LY-1)           )], 1, gauge_zy_edge_vector, g_nb_y_up, 102, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&g_gauge_field[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY + T*LY) + 2*T*LX )], 1, gauge_zy_edge_cont,   g_nb_y_dn, 102, g_cart_grid, &request[cntr]);
  cntr++;

#endif

  MPI_Waitall(cntr, request, status);
/*
  int i;
  for(i=0; i<8; i++) {
    fprintf(stdout, "# [%2d] status no. %d: MPI_SOURCE=%d, MPI_TAG=%d, MPI_ERROR=%d, _count=%d, _cancelled=%d\n", 
      g_cart_id, i, status[i].MPI_SOURCE, status[i].MPI_TAG, status[i].MPI_ERROR, status[i]._count, status[i]._cancelled);
  }
*/
#endif
}  /* end of xchange_gauge */



/*****************************************************
 * exchange any globally defined gauge field
 *****************************************************/
void xchange_gauge_field(double *gfield) {

#ifdef HAVE_MPI
  int cntr=0;
  MPI_Request request[120];
  MPI_Status status[120];

  MPI_Isend(&gfield[0],         1, gauge_time_slice_cont, g_nb_t_dn, 83, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*VOLUME], 1, gauge_time_slice_cont, g_nb_t_up, 83, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(T-1)*LX*LY*LZ], 1, gauge_time_slice_cont, g_nb_t_up, 84, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(T+1)*LX*LY*LZ], 1, gauge_time_slice_cont, g_nb_t_dn, 84, g_cart_grid, &request[cntr]);
  cntr++;

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  MPI_Isend(&gfield[0],                              1, gauge_x_slice_vector, g_nb_x_dn, 85, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*LX*LY*LZ)],         1, gauge_x_slice_cont,   g_nb_x_up, 85, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(LX-1)*LY*LZ],                1, gauge_x_slice_vector, g_nb_x_up, 86, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, gauge_x_slice_cont,   g_nb_x_dn, 86, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  cntr = 0;

  MPI_Isend(&gfield[72*(VOLUME+2*LX*LY*LZ)], 1, gauge_xt_edge_vector, g_nb_t_dn, 87, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND)], 1, gauge_xt_edge_cont, g_nb_t_up, 87, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(VOLUME+2*LX*LY*LZ+(T-1)*LY*LZ)], 1, gauge_xt_edge_vector, g_nb_t_up, 88, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+2*LY*LZ)], 1, gauge_xt_edge_cont, g_nb_t_dn, 88, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXY || (defined PARALLELTXYZ)
  MPI_Isend(&gfield[0], 1, gauge_y_slice_vector, g_nb_y_dn, 89, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_y_slice_cont, g_nb_y_up, 89, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(LY-1)*LZ], 1, gauge_y_slice_vector, g_nb_y_up, 90, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+T*LX*LZ)], 1, gauge_y_slice_cont, g_nb_y_dn, 90, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  cntr = 0;

  MPI_Isend(&gfield[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_yt_edge_vector, g_nb_t_dn, 91, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*LY*LZ)], 1, gauge_yt_edge_cont, g_nb_t_up, 91, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(T-1)*LX*LZ)], 1, gauge_yt_edge_vector, g_nb_t_up, 92, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*LY*LZ+2*LX*LZ)], 1, gauge_yt_edge_cont, g_nb_t_dn, 92, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_yx_edge_vector, g_nb_x_dn, 93, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*LY*LZ+4*LX*LZ)], 1, gauge_yx_edge_cont, g_nb_x_up, 93, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(LX-1)*LZ)], 1, gauge_yx_edge_vector, g_nb_x_up, 94, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*LY*LZ+4*LX*LZ+2*T*LZ)], 1, gauge_yx_edge_cont, g_nb_x_dn, 94, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXYZ
  /* boundary faces */
  MPI_Isend(&gfield[0], 1, gauge_z_slice_vector, g_nb_z_dn, 95, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) )], 1, gauge_z_slice_cont, g_nb_z_up, 95, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(LZ-1)], 1, gauge_z_slice_vector, g_nb_z_up, 96, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*(LX*LY*LZ + T*LY*LZ + T*LX*LZ) + T*LX*LY)], 1, gauge_z_slice_cont, g_nb_z_dn, 96, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);


  /* boundary edges */

  cntr = 0;

  /* z-t edges */
  MPI_Isend(&gfield[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )          )], 1, gauge_zt_edge_vector, g_nb_t_dn,  97, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ)               )], 1, gauge_zt_edge_cont,   g_nb_t_up,  97, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(VOLUME+2*(LX*LY*LZ + T*LY*LZ + T*LX*LZ)+(T-1)*LX*LY)], 1, gauge_zt_edge_vector, g_nb_t_up,  98, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ) + 2*LX*LY     )], 1, gauge_zt_edge_cont,   g_nb_t_dn,  98, g_cart_grid, &request[cntr]);
  cntr++;

  /* z-x edges */
  MPI_Isend(&gfield[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )             )], 1, gauge_zt_edge_vector, g_nb_x_dn, 99, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY)          )], 1, gauge_zt_edge_cont,   g_nb_x_up, 99, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) + (LX-1)*LY )], 1, gauge_zx_edge_vector, g_nb_x_up, 100, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY) + 2*T*LY )], 1, gauge_zx_edge_cont,   g_nb_x_dn, 100, g_cart_grid, &request[cntr]);
  cntr++;

  /* z-y edges */
  MPI_Isend(&gfield[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ )                    )], 1, gauge_zy_edge_vector, g_nb_y_dn, 101, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY + T*LY)          )], 1, gauge_zy_edge_cont,   g_nb_y_up, 101, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(VOLUME+2*( LX*LY*LZ + T*LY*LZ + T*LX*LZ ) + (LY-1)           )], 1, gauge_zy_edge_vector, g_nb_y_up, 102, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*(LY*LZ + LX*LZ + T*LZ + LX*LY + T*LY) + 2*T*LX )], 1, gauge_zy_edge_cont,   g_nb_y_dn, 102, g_cart_grid, &request[cntr]);
  cntr++;

#endif

  MPI_Waitall(cntr, request, status);
/*
  int i;
  for(i=0; i<8; i++) {
    fprintf(stdout, "# [%2d] status no. %d: MPI_SOURCE=%d, MPI_TAG=%d, MPI_ERROR=%d, _count=%d, _cancelled=%d\n", 
      g_cart_id, i, status[i].MPI_SOURCE, status[i].MPI_TAG, status[i].MPI_ERROR, status[i]._count, status[i]._cancelled);
  }
*/
#endif
}  /* end of xchange_gauge_field */


/*****************************************************
 * exchange a spinor field
 *****************************************************/
void xchange_field(double *phi) {
#ifdef HAVE_MPI
  int cntr=0;

  MPI_Request request[120];
  MPI_Status status[120];

  MPI_Isend(&phi[0],                 1, spinor_time_slice_cont, g_nb_t_dn, 83, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*VOLUME],         1, spinor_time_slice_cont, g_nb_t_up, 83, g_cart_grid, &request[cntr]);
  cntr++;
  
  MPI_Isend(&phi[24*(T-1)*LX*LY*LZ], 1, spinor_time_slice_cont, g_nb_t_up, 84, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(T+1)*LX*LY*LZ], 1, spinor_time_slice_cont, g_nb_t_dn, 84, g_cart_grid, &request[cntr]);
  cntr++;
#if (defined PARALLELTX) || (defined PARALLELTXY)  || (defined PARALLELTXYZ) 
 
  /* x - boundary faces */
  MPI_Isend(&phi[0],                              1, spinor_x_slice_vector, g_nb_x_dn, 85, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*LX*LY*LZ)],         1, spinor_x_slice_cont,   g_nb_x_up, 85, g_cart_grid, &request[cntr]);
  cntr++;
  
  MPI_Isend(&phi[24*(LX-1)*LY*LZ],                1, spinor_x_slice_vector, g_nb_x_up, 86, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, spinor_x_slice_cont,   g_nb_x_dn, 86, g_cart_grid, &request[cntr]);
  cntr++;
#endif
#if defined PARALLELTXY || (defined PARALLELTXYZ) 
  /* y - boundary faces */
  MPI_Isend(&phi[0],                                        1, spinor_y_slice_vector, g_nb_y_dn, 87, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*(LX*LY*LZ+T*LY*LZ))],         1, spinor_y_slice_cont,   g_nb_y_up, 87, g_cart_grid, &request[cntr]);
  cntr++;
  
  MPI_Isend(&phi[24*(LY-1)*LZ],                             1, spinor_y_slice_vector, g_nb_y_up, 88, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*(LX*LY*LZ+T*LY*LZ)+T*LX*LZ)], 1, spinor_y_slice_cont,   g_nb_y_dn, 88, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if (defined PARALLELTXYZ) 

  /* z - boundary faces */

  MPI_Isend(&phi[0],                                                1, spinor_z_slice_vector, g_nb_z_dn, 89, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*(LX*LY*LZ+T*LY*LZ+T*LX*LZ))],         1, spinor_z_slice_cont,   g_nb_z_up, 89, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[24*(LZ-1)],                                        1, spinor_z_slice_vector, g_nb_z_up, 90, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*(LX*LY*LZ+T*LY*LZ+T*LX*LZ)+T*LX*LY)], 1, spinor_z_slice_cont,   g_nb_z_dn, 90, g_cart_grid, &request[cntr]);
  cntr++;

#endif

  MPI_Waitall(cntr, request, status);
#endif
}  /* xchange_field */

/*****************************************************
 * measure the plaquette value
 *****************************************************/
void plaquette(double *pl) {

  int ix, mu, nu; 
  double s[18], t[18], u[18], pl_loc;
  complex w;
  double linksum[2], ls[2];

  pl_loc=0;

  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<3; mu++) {
    for(nu=mu+1; nu<4; nu++) {
      _cm_eq_cm_ti_cm(s, &g_gauge_field[72*ix+mu*18], &g_gauge_field[72*g_iup[ix][mu]+18*nu]);
      _cm_eq_cm_ti_cm(t, &g_gauge_field[72*ix+nu*18], &g_gauge_field[72*g_iup[ix][nu]+18*mu]);
      _cm_eq_cm_ti_cm_dag(u, s, t);
      _co_eq_tr_cm(&w, u);
      pl_loc += w.re;
    }
    }
  }


#ifdef HAVE_MPI
  MPI_Reduce(&pl_loc, pl, 1, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
  *pl = pl_loc;
#endif
  *pl = *pl / ((double)T_global * (double)(LX*g_nproc_x) * (double)(LY*g_nproc_y) * (double)(LZ*g_nproc_z) * 18.);

  linksum[0] = 0.;
  linksum[1] = 0.;
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      for(nu=0; nu<9; nu++) {
        linksum[0] += g_gauge_field[_GGI(ix,mu)+2*nu  ];
        linksum[1] += g_gauge_field[_GGI(ix,mu)+2*nu+1];
      }
    }
  }
#ifdef HAVE_MPI
  MPI_Reduce(linksum, ls, 2, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
  ls[0] = linksum[0];
  ls[1] = linksum[1];
#endif
  if(g_cart_id==0) {
    fprintf(stdout, "# [plaquette] measured linksuj value = %25.16e + I %25.16e\n", ls[0], ls[1]);
  }

}

/*****************************************************
 * measure the plaquette value
 *****************************************************/
void plaquette2(double *pl, double*gfield) {

  int ix, mu, nu; 
  double s[18], t[18], u[18], pl_loc;
  complex w;

  pl_loc=0;

  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<3; mu++) {
    for(nu=mu+1; nu<4; nu++) {
      _cm_eq_cm_ti_cm(s, &gfield[72*ix+mu*18], &gfield[72*g_iup[ix][mu]+18*nu]);
      _cm_eq_cm_ti_cm(t, &gfield[72*ix+nu*18], &gfield[72*g_iup[ix][nu]+18*mu]);
      _cm_eq_cm_ti_cm_dag(u, s, t);
      _co_eq_tr_cm(&w, u);
      pl_loc += w.re;
    }
    }
  }

#ifdef HAVE_MPI
  MPI_Reduce(&pl_loc, pl, 1, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
  *pl = pl_loc;
#endif
  *pl = *pl / ((double)T_global * (double)(LX*g_nproc_x) * (double)(LY*g_nproc_y) * (double)(LZ*g_nproc_z) * 18.);
}

/*****************************************************
 * write contractions to file
 *****************************************************/
int write_contraction (double *s, int *nsource, char *filename, int Nmu, 
                       int write_ascii, int append) {

  int x0, x1, x2, x3, ix, mu, i;
  unsigned long int count=0;
  int ti[2], lvol;
  FILE *ofs;
#ifdef HAVE_MPI
  double *buffer;
  MPI_Status status;
#endif

  if(g_cart_id==0) {
    if(append==1) ofs = fopen(filename, "a");
    else ofs = fopen(filename, "w");
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "[write_contraction] Error, could not open file %s for writing\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(104);
    }
    if(write_ascii == 1) { /* write in ASCII format */
      /* write own part */
      fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n",
        Nconf, T_global, LX, LY, LZ, g_kappa, g_mu);
      for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix = g_ipt[x0][x1][x2][x3];
        fprintf(ofs, "# t=%3d, x=%3d, y=%3d, z=%3d\n", x0, x1, x2, x3);
	for(mu=0; mu<Nmu; mu++) {
	  fprintf(ofs, "%3d%25.16e%25.16e\n", mu, s[2*(Nmu*ix+mu)], s[2*(Nmu*ix+mu)+1]);
	}
      }
      }
      }
      }
    }
    else if(write_ascii == 2) { /* inner loop ix */
      fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n",
        Nconf, T_global, LX, LY, LZ, g_kappa, g_mu);
      for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix = g_ipt[x0][x1][x2][x3];
        fprintf(ofs, "# t=%3d, x=%3d, y=%3d, z=%3d\n", x0, x1, x2, x3);
	for(mu=0; mu<Nmu; mu++) {
	  fprintf(ofs, "%3d%25.16e%25.16e\n", mu, s[_GWI(mu,ix,VOLUME)], s[_GWI(mu,ix,VOLUME)+1]);
	}
      }
      }
      }
      }
    }
    else if(write_ascii == 0) {
      ix = -1;
      for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix++;
        for(mu=0; mu<Nmu; mu++) {
          fwrite(s+_GWI(mu,ix,VOLUME), sizeof(double), 2, ofs);
        }
      }
      }
      }
      }
    }
#ifdef HAVE_MPI
#  if !(defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ)
    for(i=1; i<g_nproc; i++) {
      MPI_Recv(ti, 2, MPI_INT, i, 100+i, g_cart_grid, &status);
      count = (unsigned long int)Nmu * (unsigned long int)(ti[0]*LX*LY*LZ) * 2;
      if((void*)(buffer = (double*)calloc(count, sizeof(double))) == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
	MPI_Finalize();
	exit(105);
      }
      MPI_Recv(buffer, count, MPI_DOUBLE, i, 200+i, g_cart_grid, &status);
      if(write_ascii == 1) { /* write in ASCII format */
        /* write part from process i */
	ix = -1;
        for(x0=0; x0<ti[0]; x0++) {
        for(x1=0; x1<LX; x1++) {
        for(x2=0; x2<LY; x2++) {
        for(x3=0; x3<LZ; x3++) {
          ix++;
          fprintf(ofs, "# t=%3d, x=%3d, y=%3d, z=%3d\n", x0+ti[1], x1, x2, x3);
  	  for(mu=0; mu<Nmu; mu++) {
	    fprintf(ofs, "%3d%25.16e%25.16e\n", mu, 
	      buffer[2*(Nmu*ix+mu)], buffer[2*(Nmu*ix+mu)+1]);
	  }
        }
        }
        }
        }
      }
      else if(write_ascii == 2) { /* inner loop ix */
	ix = -1;
        for(x0=0; x0<ti[0]; x0++) {
        for(x1=0; x1<LX; x1++) {
        for(x2=0; x2<LY; x2++) {
        for(x3=0; x3<LZ; x3++) {
          ix++;
          fprintf(ofs, "# t=%3d, x=%3d, y=%3d, z=%3d\n", x0+ti[1], x1, x2, x3);
  	  for(mu=0; mu<Nmu; mu++) {
	    fprintf(ofs, "%3d%25.16e%25.16e\n", mu, 
	      buffer[_GWI(mu,ix,VOLUME)], buffer[_GWI(mu,ix,VOLUME)+1]);
	  }
        }
        }
        }
        }
      }
      else if (write_ascii == 0) {
        lvol = ti[0]*LX*LY*LZ;
        fprintf(stdout, "lvol = %d\n", lvol);
	ix = -1;
        for(x0=0; x0<ti[0]; x0++) {
        for(x1=0; x1<LX; x1++) {
        for(x2=0; x2<LY; x2++) {
        for(x3=0; x3<LZ; x3++) {
          ix++;
  	  for(mu=0; mu<Nmu; mu++) {
            fwrite(buffer+_GWI(mu,ix,lvol), sizeof(double), 2, ofs);
          }
        }
        }
        }
        }
      }
      free(buffer);
    }
#  endif
#endif
    if(nsource!=(int*)NULL) {
      if(write_ascii==0) fwrite(nsource, sizeof(int), 1, ofs);
      if(write_ascii==1) {
        fprintf(ofs, "#count:\n");
	fprintf(ofs, "%d\n", *nsource);
      }
    }
    fclose(ofs);
  } /* end of if g_cart_id == 0 */
#ifdef HAVE_MPI
  else {
    for(i=1; i<g_nproc; i++) {
      if(g_cart_id==i) {
        ti[0] = T;
	ti[1] = Tstart;
        count = (unsigned long int)Nmu * (unsigned long int)VOLUME * 2;
        MPI_Send(ti, 2, MPI_INT, 0, 100+i, g_cart_grid);
        MPI_Send(s, count, MPI_DOUBLE, 0, 200+i, g_cart_grid);
      }
    }
  }
#endif
  return(0);

}

/*****************************************************
 * read contractions from file
 *****************************************************/
int read_contraction(double *s, int *nsource, char *filename, int Nmu) {
#if defined PARALLELTXYZ
  fprintf(stderr, "[read_contraction] Error, not implemented for 4-dim parallel\n");
  return(1);
#endif

  unsigned long int shift=0, count=0;
  int ix, mu, iy;
  double buffer[128];
  FILE *ofs = (FILE*)NULL;

  ofs = fopen(filename, "r");
  if(ofs==(FILE*)NULL) {
    fprintf(stderr, "[read_contraction] Error, could not open file %s for reading\n", filename);
    return(106);
  }
  if(format==2) {
    if(g_cart_id==0) fprintf(stdout, "# [read_contraction] Reading contraction data in format %d\n", format);
#ifdef HAVE_MPI
#  if ! ( defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ )  /* T-parallel, seek by shifting timeslices */
    shift = (unsigned long int)Tstart * (unsigned long int)LX*LY*LZ * Nmu * 2 * sizeof(double);
    fseek(ofs, shift, SEEK_SET);
#  endif
#endif
    count= Nmu * 2;
    for(ix=0; ix<VOLUME; ix++) { 
      if( fread(buffer, sizeof(double), count, ofs) != count) return(107);
      for(mu=0; mu<Nmu; mu++) {
        iy = index_conv(16*ix+mu, 2);
        s[iy  ] = buffer[2*mu  ];
        s[iy+1] = buffer[2*mu+1];
      } 
    }
  } else {
#ifdef HAVE_MPI
#  if ! ( defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ )
    /* shift = Volume of previous nodes * Nmu * 2(complex=2doubles) * 
               size of double */
    shift = (unsigned long int)Tstart * (unsigned long int)LX*LY*LZ * Nmu * 2 * sizeof(double);
    fseek(ofs, shift, SEEK_SET);
#  endif
#endif
    count= Nmu * 2;
    for(ix=0; ix<VOLUME; ix++) { 
      if( fread(buffer, sizeof(double), count, ofs) != count) return(107);
      for(mu=0; mu<Nmu; mu++) {
        s[_GWI(mu,ix,VOLUME)  ] = buffer[2*mu  ];
        s[_GWI(mu,ix,VOLUME)+1] = buffer[2*mu+1];
      } 
    }
    if(nsource != (int*)NULL) {
#ifdef HAVE_MPI
#  if ! ( defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ )

      /* shift to position EOF - one integer */
      fseek(ofs, -sizeof(int), SEEK_END);
#  endif
#endif
      if( fread(nsource, sizeof(int), 1, ofs)  != 1) return(108);
    }
  } /* of format != 2 */
  fclose(ofs);

  return(0);
}

/*****************************************************
 * init the gamma matrix permutations and signs
 *****************************************************/
void init_gamma(void) {
/*
   The gamma matrices.

   Standard choice (as in gwc):

   gamma_0:

   |  0  0 -1  0 |
   |  0  0  0 -1 |
   | -1  0  0  0 |
   |  0 -1  0  0 |

   gamma_1:

   |  0  0  0 -i |
   |  0  0 -i  0 |
   |  0 +i  0  0 |
   | +i  0  0  0 |

   gamma_2:

   |  0  0  0 -1 |
   |  0  0 +1  0 |
   |  0 +1  0  0 |
   | -1  0  0  0 |

   gamma_3:
   |  0  0 -i  0 |
   |  0  0  0 +i |
   | +i  0  0  0 |
   |  0 -i  0  0 |

   Permutation of the eight elements of a spinor (re0, im0, re1, im1, ...).

   the sequence is:
   0, 1, 2, 3, id, 5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
id 0  1  2  3   4  5    6    7    8    9   10   11   12   13   14   15
*/

/* the gamma matrix index */
gamma_permutation[0][0] = 12;
gamma_permutation[0][1] = 13;
gamma_permutation[0][2] = 14;
gamma_permutation[0][3] = 15;
gamma_permutation[0][4] = 16;
gamma_permutation[0][5] = 17;
gamma_permutation[0][6] = 18;
gamma_permutation[0][7] = 19;
gamma_permutation[0][8] = 20;
gamma_permutation[0][9] = 21;
gamma_permutation[0][10] = 22;
gamma_permutation[0][11] = 23;
gamma_permutation[0][12] = 0;
gamma_permutation[0][13] = 1;
gamma_permutation[0][14] = 2;
gamma_permutation[0][15] = 3;
gamma_permutation[0][16] = 4;
gamma_permutation[0][17] = 5;
gamma_permutation[0][18] = 6;
gamma_permutation[0][19] = 7;
gamma_permutation[0][20] = 8;
gamma_permutation[0][21] = 9;
gamma_permutation[0][22] = 10;
gamma_permutation[0][23] = 11;
gamma_permutation[1][0] = 19;
gamma_permutation[1][1] = 18;
gamma_permutation[1][2] = 21;
gamma_permutation[1][3] = 20;
gamma_permutation[1][4] = 23;
gamma_permutation[1][5] = 22;
gamma_permutation[1][6] = 13;
gamma_permutation[1][7] = 12;
gamma_permutation[1][8] = 15;
gamma_permutation[1][9] = 14;
gamma_permutation[1][10] = 17;
gamma_permutation[1][11] = 16;
gamma_permutation[1][12] = 7;
gamma_permutation[1][13] = 6;
gamma_permutation[1][14] = 9;
gamma_permutation[1][15] = 8;
gamma_permutation[1][16] = 11;
gamma_permutation[1][17] = 10;
gamma_permutation[1][18] = 1;
gamma_permutation[1][19] = 0;
gamma_permutation[1][20] = 3;
gamma_permutation[1][21] = 2;
gamma_permutation[1][22] = 5;
gamma_permutation[1][23] = 4;
gamma_permutation[2][0] = 18;
gamma_permutation[2][1] = 19;
gamma_permutation[2][2] = 20;
gamma_permutation[2][3] = 21;
gamma_permutation[2][4] = 22;
gamma_permutation[2][5] = 23;
gamma_permutation[2][6] = 12;
gamma_permutation[2][7] = 13;
gamma_permutation[2][8] = 14;
gamma_permutation[2][9] = 15;
gamma_permutation[2][10] = 16;
gamma_permutation[2][11] = 17;
gamma_permutation[2][12] = 6;
gamma_permutation[2][13] = 7;
gamma_permutation[2][14] = 8;
gamma_permutation[2][15] = 9;
gamma_permutation[2][16] = 10;
gamma_permutation[2][17] = 11;
gamma_permutation[2][18] = 0;
gamma_permutation[2][19] = 1;
gamma_permutation[2][20] = 2;
gamma_permutation[2][21] = 3;
gamma_permutation[2][22] = 4;
gamma_permutation[2][23] = 5;
gamma_permutation[3][0] = 13;
gamma_permutation[3][1] = 12;
gamma_permutation[3][2] = 15;
gamma_permutation[3][3] = 14;
gamma_permutation[3][4] = 17;
gamma_permutation[3][5] = 16;
gamma_permutation[3][6] = 19;
gamma_permutation[3][7] = 18;
gamma_permutation[3][8] = 21;
gamma_permutation[3][9] = 20;
gamma_permutation[3][10] = 23;
gamma_permutation[3][11] = 22;
gamma_permutation[3][12] = 1;
gamma_permutation[3][13] = 0;
gamma_permutation[3][14] = 3;
gamma_permutation[3][15] = 2;
gamma_permutation[3][16] = 5;
gamma_permutation[3][17] = 4;
gamma_permutation[3][18] = 7;
gamma_permutation[3][19] = 6;
gamma_permutation[3][20] = 9;
gamma_permutation[3][21] = 8;
gamma_permutation[3][22] = 11;
gamma_permutation[3][23] = 10;
gamma_permutation[4][0] = 0;
gamma_permutation[4][1] = 1;
gamma_permutation[4][2] = 2;
gamma_permutation[4][3] = 3;
gamma_permutation[4][4] = 4;
gamma_permutation[4][5] = 5;
gamma_permutation[4][6] = 6;
gamma_permutation[4][7] = 7;
gamma_permutation[4][8] = 8;
gamma_permutation[4][9] = 9;
gamma_permutation[4][10] = 10;
gamma_permutation[4][11] = 11;
gamma_permutation[4][12] = 12;
gamma_permutation[4][13] = 13;
gamma_permutation[4][14] = 14;
gamma_permutation[4][15] = 15;
gamma_permutation[4][16] = 16;
gamma_permutation[4][17] = 17;
gamma_permutation[4][18] = 18;
gamma_permutation[4][19] = 19;
gamma_permutation[4][20] = 20;
gamma_permutation[4][21] = 21;
gamma_permutation[4][22] = 22;
gamma_permutation[4][23] = 23;
gamma_permutation[5][0] = 0;
gamma_permutation[5][1] = 1;
gamma_permutation[5][2] = 2;
gamma_permutation[5][3] = 3;
gamma_permutation[5][4] = 4;
gamma_permutation[5][5] = 5;
gamma_permutation[5][6] = 6;
gamma_permutation[5][7] = 7;
gamma_permutation[5][8] = 8;
gamma_permutation[5][9] = 9;
gamma_permutation[5][10] = 10;
gamma_permutation[5][11] = 11;
gamma_permutation[5][12] = 12;
gamma_permutation[5][13] = 13;
gamma_permutation[5][14] = 14;
gamma_permutation[5][15] = 15;
gamma_permutation[5][16] = 16;
gamma_permutation[5][17] = 17;
gamma_permutation[5][18] = 18;
gamma_permutation[5][19] = 19;
gamma_permutation[5][20] = 20;
gamma_permutation[5][21] = 21;
gamma_permutation[5][22] = 22;
gamma_permutation[5][23] = 23;
gamma_permutation[6][0] = 12;
gamma_permutation[6][1] = 13;
gamma_permutation[6][2] = 14;
gamma_permutation[6][3] = 15;
gamma_permutation[6][4] = 16;
gamma_permutation[6][5] = 17;
gamma_permutation[6][6] = 18;
gamma_permutation[6][7] = 19;
gamma_permutation[6][8] = 20;
gamma_permutation[6][9] = 21;
gamma_permutation[6][10] = 22;
gamma_permutation[6][11] = 23;
gamma_permutation[6][12] = 0;
gamma_permutation[6][13] = 1;
gamma_permutation[6][14] = 2;
gamma_permutation[6][15] = 3;
gamma_permutation[6][16] = 4;
gamma_permutation[6][17] = 5;
gamma_permutation[6][18] = 6;
gamma_permutation[6][19] = 7;
gamma_permutation[6][20] = 8;
gamma_permutation[6][21] = 9;
gamma_permutation[6][22] = 10;
gamma_permutation[6][23] = 11;
gamma_permutation[7][0] = 19;
gamma_permutation[7][1] = 18;
gamma_permutation[7][2] = 21;
gamma_permutation[7][3] = 20;
gamma_permutation[7][4] = 23;
gamma_permutation[7][5] = 22;
gamma_permutation[7][6] = 13;
gamma_permutation[7][7] = 12;
gamma_permutation[7][8] = 15;
gamma_permutation[7][9] = 14;
gamma_permutation[7][10] = 17;
gamma_permutation[7][11] = 16;
gamma_permutation[7][12] = 7;
gamma_permutation[7][13] = 6;
gamma_permutation[7][14] = 9;
gamma_permutation[7][15] = 8;
gamma_permutation[7][16] = 11;
gamma_permutation[7][17] = 10;
gamma_permutation[7][18] = 1;
gamma_permutation[7][19] = 0;
gamma_permutation[7][20] = 3;
gamma_permutation[7][21] = 2;
gamma_permutation[7][22] = 5;
gamma_permutation[7][23] = 4;
gamma_permutation[8][0] = 18;
gamma_permutation[8][1] = 19;
gamma_permutation[8][2] = 20;
gamma_permutation[8][3] = 21;
gamma_permutation[8][4] = 22;
gamma_permutation[8][5] = 23;
gamma_permutation[8][6] = 12;
gamma_permutation[8][7] = 13;
gamma_permutation[8][8] = 14;
gamma_permutation[8][9] = 15;
gamma_permutation[8][10] = 16;
gamma_permutation[8][11] = 17;
gamma_permutation[8][12] = 6;
gamma_permutation[8][13] = 7;
gamma_permutation[8][14] = 8;
gamma_permutation[8][15] = 9;
gamma_permutation[8][16] = 10;
gamma_permutation[8][17] = 11;
gamma_permutation[8][18] = 0;
gamma_permutation[8][19] = 1;
gamma_permutation[8][20] = 2;
gamma_permutation[8][21] = 3;
gamma_permutation[8][22] = 4;
gamma_permutation[8][23] = 5;
gamma_permutation[9][0] = 13;
gamma_permutation[9][1] = 12;
gamma_permutation[9][2] = 15;
gamma_permutation[9][3] = 14;
gamma_permutation[9][4] = 17;
gamma_permutation[9][5] = 16;
gamma_permutation[9][6] = 19;
gamma_permutation[9][7] = 18;
gamma_permutation[9][8] = 21;
gamma_permutation[9][9] = 20;
gamma_permutation[9][10] = 23;
gamma_permutation[9][11] = 22;
gamma_permutation[9][12] = 1;
gamma_permutation[9][13] = 0;
gamma_permutation[9][14] = 3;
gamma_permutation[9][15] = 2;
gamma_permutation[9][16] = 5;
gamma_permutation[9][17] = 4;
gamma_permutation[9][18] = 7;
gamma_permutation[9][19] = 6;
gamma_permutation[9][20] = 9;
gamma_permutation[9][21] = 8;
gamma_permutation[9][22] = 11;
gamma_permutation[9][23] = 10;
gamma_permutation[10][0] = 7;
gamma_permutation[10][1] = 6;
gamma_permutation[10][2] = 9;
gamma_permutation[10][3] = 8;
gamma_permutation[10][4] = 11;
gamma_permutation[10][5] = 10;
gamma_permutation[10][6] = 1;
gamma_permutation[10][7] = 0;
gamma_permutation[10][8] = 3;
gamma_permutation[10][9] = 2;
gamma_permutation[10][10] = 5;
gamma_permutation[10][11] = 4;
gamma_permutation[10][12] = 19;
gamma_permutation[10][13] = 18;
gamma_permutation[10][14] = 21;
gamma_permutation[10][15] = 20;
gamma_permutation[10][16] = 23;
gamma_permutation[10][17] = 22;
gamma_permutation[10][18] = 13;
gamma_permutation[10][19] = 12;
gamma_permutation[10][20] = 15;
gamma_permutation[10][21] = 14;
gamma_permutation[10][22] = 17;
gamma_permutation[10][23] = 16;
gamma_permutation[11][0] = 6;
gamma_permutation[11][1] = 7;
gamma_permutation[11][2] = 8;
gamma_permutation[11][3] = 9;
gamma_permutation[11][4] = 10;
gamma_permutation[11][5] = 11;
gamma_permutation[11][6] = 0;
gamma_permutation[11][7] = 1;
gamma_permutation[11][8] = 2;
gamma_permutation[11][9] = 3;
gamma_permutation[11][10] = 4;
gamma_permutation[11][11] = 5;
gamma_permutation[11][12] = 18;
gamma_permutation[11][13] = 19;
gamma_permutation[11][14] = 20;
gamma_permutation[11][15] = 21;
gamma_permutation[11][16] = 22;
gamma_permutation[11][17] = 23;
gamma_permutation[11][18] = 12;
gamma_permutation[11][19] = 13;
gamma_permutation[11][20] = 14;
gamma_permutation[11][21] = 15;
gamma_permutation[11][22] = 16;
gamma_permutation[11][23] = 17;
gamma_permutation[12][0] = 1;
gamma_permutation[12][1] = 0;
gamma_permutation[12][2] = 3;
gamma_permutation[12][3] = 2;
gamma_permutation[12][4] = 5;
gamma_permutation[12][5] = 4;
gamma_permutation[12][6] = 7;
gamma_permutation[12][7] = 6;
gamma_permutation[12][8] = 9;
gamma_permutation[12][9] = 8;
gamma_permutation[12][10] = 11;
gamma_permutation[12][11] = 10;
gamma_permutation[12][12] = 13;
gamma_permutation[12][13] = 12;
gamma_permutation[12][14] = 15;
gamma_permutation[12][15] = 14;
gamma_permutation[12][16] = 17;
gamma_permutation[12][17] = 16;
gamma_permutation[12][18] = 19;
gamma_permutation[12][19] = 18;
gamma_permutation[12][20] = 21;
gamma_permutation[12][21] = 20;
gamma_permutation[12][22] = 23;
gamma_permutation[12][23] = 22;
gamma_permutation[13][0] = 1;
gamma_permutation[13][1] = 0;
gamma_permutation[13][2] = 3;
gamma_permutation[13][3] = 2;
gamma_permutation[13][4] = 5;
gamma_permutation[13][5] = 4;
gamma_permutation[13][6] = 7;
gamma_permutation[13][7] = 6;
gamma_permutation[13][8] = 9;
gamma_permutation[13][9] = 8;
gamma_permutation[13][10] = 11;
gamma_permutation[13][11] = 10;
gamma_permutation[13][12] = 13;
gamma_permutation[13][13] = 12;
gamma_permutation[13][14] = 15;
gamma_permutation[13][15] = 14;
gamma_permutation[13][16] = 17;
gamma_permutation[13][17] = 16;
gamma_permutation[13][18] = 19;
gamma_permutation[13][19] = 18;
gamma_permutation[13][20] = 21;
gamma_permutation[13][21] = 20;
gamma_permutation[13][22] = 23;
gamma_permutation[13][23] = 22;
gamma_permutation[14][0] = 6;
gamma_permutation[14][1] = 7;
gamma_permutation[14][2] = 8;
gamma_permutation[14][3] = 9;
gamma_permutation[14][4] = 10;
gamma_permutation[14][5] = 11;
gamma_permutation[14][6] = 0;
gamma_permutation[14][7] = 1;
gamma_permutation[14][8] = 2;
gamma_permutation[14][9] = 3;
gamma_permutation[14][10] = 4;
gamma_permutation[14][11] = 5;
gamma_permutation[14][12] = 18;
gamma_permutation[14][13] = 19;
gamma_permutation[14][14] = 20;
gamma_permutation[14][15] = 21;
gamma_permutation[14][16] = 22;
gamma_permutation[14][17] = 23;
gamma_permutation[14][18] = 12;
gamma_permutation[14][19] = 13;
gamma_permutation[14][20] = 14;
gamma_permutation[14][21] = 15;
gamma_permutation[14][22] = 16;
gamma_permutation[14][23] = 17;
gamma_permutation[15][0] = 7;
gamma_permutation[15][1] = 6;
gamma_permutation[15][2] = 9;
gamma_permutation[15][3] = 8;
gamma_permutation[15][4] = 11;
gamma_permutation[15][5] = 10;
gamma_permutation[15][6] = 1;
gamma_permutation[15][7] = 0;
gamma_permutation[15][8] = 3;
gamma_permutation[15][9] = 2;
gamma_permutation[15][10] = 5;
gamma_permutation[15][11] = 4;
gamma_permutation[15][12] = 19;
gamma_permutation[15][13] = 18;
gamma_permutation[15][14] = 21;
gamma_permutation[15][15] = 20;
gamma_permutation[15][16] = 23;
gamma_permutation[15][17] = 22;
gamma_permutation[15][18] = 13;
gamma_permutation[15][19] = 12;
gamma_permutation[15][20] = 15;
gamma_permutation[15][21] = 14;
gamma_permutation[15][22] = 17;
gamma_permutation[15][23] = 16;

/* the gamma matrix sign */

gamma_sign[0][0] = -1;
gamma_sign[0][1] = -1;
gamma_sign[0][2] = -1;
gamma_sign[0][3] = -1;
gamma_sign[0][4] = -1;
gamma_sign[0][5] = -1;
gamma_sign[0][6] = -1;
gamma_sign[0][7] = -1;
gamma_sign[0][8] = -1;
gamma_sign[0][9] = -1;
gamma_sign[0][10] = -1;
gamma_sign[0][11] = -1;
gamma_sign[0][12] = -1;
gamma_sign[0][13] = -1;
gamma_sign[0][14] = -1;
gamma_sign[0][15] = -1;
gamma_sign[0][16] = -1;
gamma_sign[0][17] = -1;
gamma_sign[0][18] = -1;
gamma_sign[0][19] = -1;
gamma_sign[0][20] = -1;
gamma_sign[0][21] = -1;
gamma_sign[0][22] = -1;
gamma_sign[0][23] = -1;
gamma_sign[1][0] = +1;
gamma_sign[1][1] = -1;
gamma_sign[1][2] = +1;
gamma_sign[1][3] = -1;
gamma_sign[1][4] = +1;
gamma_sign[1][5] = -1;
gamma_sign[1][6] = +1;
gamma_sign[1][7] = -1;
gamma_sign[1][8] = +1;
gamma_sign[1][9] = -1;
gamma_sign[1][10] = +1;
gamma_sign[1][11] = -1;
gamma_sign[1][12] = -1;
gamma_sign[1][13] = +1;
gamma_sign[1][14] = -1;
gamma_sign[1][15] = +1;
gamma_sign[1][16] = -1;
gamma_sign[1][17] = +1;
gamma_sign[1][18] = -1;
gamma_sign[1][19] = +1;
gamma_sign[1][20] = -1;
gamma_sign[1][21] = +1;
gamma_sign[1][22] = -1;
gamma_sign[1][23] = +1;
gamma_sign[2][0] = -1;
gamma_sign[2][1] = -1;
gamma_sign[2][2] = -1;
gamma_sign[2][3] = -1;
gamma_sign[2][4] = -1;
gamma_sign[2][5] = -1;
gamma_sign[2][6] = +1;
gamma_sign[2][7] = +1;
gamma_sign[2][8] = +1;
gamma_sign[2][9] = +1;
gamma_sign[2][10] = +1;
gamma_sign[2][11] = +1;
gamma_sign[2][12] = +1;
gamma_sign[2][13] = +1;
gamma_sign[2][14] = +1;
gamma_sign[2][15] = +1;
gamma_sign[2][16] = +1;
gamma_sign[2][17] = +1;
gamma_sign[2][18] = -1;
gamma_sign[2][19] = -1;
gamma_sign[2][20] = -1;
gamma_sign[2][21] = -1;
gamma_sign[2][22] = -1;
gamma_sign[2][23] = -1;
gamma_sign[3][0] = +1;
gamma_sign[3][1] = -1;
gamma_sign[3][2] = +1;
gamma_sign[3][3] = -1;
gamma_sign[3][4] = +1;
gamma_sign[3][5] = -1;
gamma_sign[3][6] = -1;
gamma_sign[3][7] = +1;
gamma_sign[3][8] = -1;
gamma_sign[3][9] = +1;
gamma_sign[3][10] = -1;
gamma_sign[3][11] = +1;
gamma_sign[3][12] = -1;
gamma_sign[3][13] = +1;
gamma_sign[3][14] = -1;
gamma_sign[3][15] = +1;
gamma_sign[3][16] = -1;
gamma_sign[3][17] = +1;
gamma_sign[3][18] = +1;
gamma_sign[3][19] = -1;
gamma_sign[3][20] = +1;
gamma_sign[3][21] = -1;
gamma_sign[3][22] = +1;
gamma_sign[3][23] = -1;
gamma_sign[4][0] = +1;
gamma_sign[4][1] = +1;
gamma_sign[4][2] = +1;
gamma_sign[4][3] = +1;
gamma_sign[4][4] = +1;
gamma_sign[4][5] = +1;
gamma_sign[4][6] = +1;
gamma_sign[4][7] = +1;
gamma_sign[4][8] = +1;
gamma_sign[4][9] = +1;
gamma_sign[4][10] = +1;
gamma_sign[4][11] = +1;
gamma_sign[4][12] = +1;
gamma_sign[4][13] = +1;
gamma_sign[4][14] = +1;
gamma_sign[4][15] = +1;
gamma_sign[4][16] = +1;
gamma_sign[4][17] = +1;
gamma_sign[4][18] = +1;
gamma_sign[4][19] = +1;
gamma_sign[4][20] = +1;
gamma_sign[4][21] = +1;
gamma_sign[4][22] = +1;
gamma_sign[4][23] = +1;
gamma_sign[5][0] = +1;
gamma_sign[5][1] = +1;
gamma_sign[5][2] = +1;
gamma_sign[5][3] = +1;
gamma_sign[5][4] = +1;
gamma_sign[5][5] = +1;
gamma_sign[5][6] = +1;
gamma_sign[5][7] = +1;
gamma_sign[5][8] = +1;
gamma_sign[5][9] = +1;
gamma_sign[5][10] = +1;
gamma_sign[5][11] = +1;
gamma_sign[5][12] = -1;
gamma_sign[5][13] = -1;
gamma_sign[5][14] = -1;
gamma_sign[5][15] = -1;
gamma_sign[5][16] = -1;
gamma_sign[5][17] = -1;
gamma_sign[5][18] = -1;
gamma_sign[5][19] = -1;
gamma_sign[5][20] = -1;
gamma_sign[5][21] = -1;
gamma_sign[5][22] = -1;
gamma_sign[5][23] = -1;
gamma_sign[6][0] = +1;
gamma_sign[6][1] = +1;
gamma_sign[6][2] = +1;
gamma_sign[6][3] = +1;
gamma_sign[6][4] = +1;
gamma_sign[6][5] = +1;
gamma_sign[6][6] = +1;
gamma_sign[6][7] = +1;
gamma_sign[6][8] = +1;
gamma_sign[6][9] = +1;
gamma_sign[6][10] = +1;
gamma_sign[6][11] = +1;
gamma_sign[6][12] = -1;
gamma_sign[6][13] = -1;
gamma_sign[6][14] = -1;
gamma_sign[6][15] = -1;
gamma_sign[6][16] = -1;
gamma_sign[6][17] = -1;
gamma_sign[6][18] = -1;
gamma_sign[6][19] = -1;
gamma_sign[6][20] = -1;
gamma_sign[6][21] = -1;
gamma_sign[6][22] = -1;
gamma_sign[6][23] = -1;
gamma_sign[7][0] = -1;
gamma_sign[7][1] = +1;
gamma_sign[7][2] = -1;
gamma_sign[7][3] = +1;
gamma_sign[7][4] = -1;
gamma_sign[7][5] = +1;
gamma_sign[7][6] = -1;
gamma_sign[7][7] = +1;
gamma_sign[7][8] = -1;
gamma_sign[7][9] = +1;
gamma_sign[7][10] = -1;
gamma_sign[7][11] = +1;
gamma_sign[7][12] = -1;
gamma_sign[7][13] = +1;
gamma_sign[7][14] = -1;
gamma_sign[7][15] = +1;
gamma_sign[7][16] = -1;
gamma_sign[7][17] = +1;
gamma_sign[7][18] = -1;
gamma_sign[7][19] = +1;
gamma_sign[7][20] = -1;
gamma_sign[7][21] = +1;
gamma_sign[7][22] = -1;
gamma_sign[7][23] = +1;
gamma_sign[8][0] = +1;
gamma_sign[8][1] = +1;
gamma_sign[8][2] = +1;
gamma_sign[8][3] = +1;
gamma_sign[8][4] = +1;
gamma_sign[8][5] = +1;
gamma_sign[8][6] = -1;
gamma_sign[8][7] = -1;
gamma_sign[8][8] = -1;
gamma_sign[8][9] = -1;
gamma_sign[8][10] = -1;
gamma_sign[8][11] = -1;
gamma_sign[8][12] = +1;
gamma_sign[8][13] = +1;
gamma_sign[8][14] = +1;
gamma_sign[8][15] = +1;
gamma_sign[8][16] = +1;
gamma_sign[8][17] = +1;
gamma_sign[8][18] = -1;
gamma_sign[8][19] = -1;
gamma_sign[8][20] = -1;
gamma_sign[8][21] = -1;
gamma_sign[8][22] = -1;
gamma_sign[8][23] = -1;
gamma_sign[9][0] = -1;
gamma_sign[9][1] = +1;
gamma_sign[9][2] = -1;
gamma_sign[9][3] = +1;
gamma_sign[9][4] = -1;
gamma_sign[9][5] = +1;
gamma_sign[9][6] = +1;
gamma_sign[9][7] = -1;
gamma_sign[9][8] = +1;
gamma_sign[9][9] = -1;
gamma_sign[9][10] = +1;
gamma_sign[9][11] = -1;
gamma_sign[9][12] = -1;
gamma_sign[9][13] = +1;
gamma_sign[9][14] = -1;
gamma_sign[9][15] = +1;
gamma_sign[9][16] = -1;
gamma_sign[9][17] = +1;
gamma_sign[9][18] = +1;
gamma_sign[9][19] = -1;
gamma_sign[9][20] = +1;
gamma_sign[9][21] = -1;
gamma_sign[9][22] = +1;
gamma_sign[9][23] = -1;
gamma_sign[10][0] = +1;
gamma_sign[10][1] = -1;
gamma_sign[10][2] = +1;
gamma_sign[10][3] = -1;
gamma_sign[10][4] = +1;
gamma_sign[10][5] = -1;
gamma_sign[10][6] = +1;
gamma_sign[10][7] = -1;
gamma_sign[10][8] = +1;
gamma_sign[10][9] = -1;
gamma_sign[10][10] = +1;
gamma_sign[10][11] = -1;
gamma_sign[10][12] = -1;
gamma_sign[10][13] = +1;
gamma_sign[10][14] = -1;
gamma_sign[10][15] = +1;
gamma_sign[10][16] = -1;
gamma_sign[10][17] = +1;
gamma_sign[10][18] = -1;
gamma_sign[10][19] = +1;
gamma_sign[10][20] = -1;
gamma_sign[10][21] = +1;
gamma_sign[10][22] = -1;
gamma_sign[10][23] = +1;
gamma_sign[11][0] = -1;
gamma_sign[11][1] = -1;
gamma_sign[11][2] = -1;
gamma_sign[11][3] = -1;
gamma_sign[11][4] = -1;
gamma_sign[11][5] = -1;
gamma_sign[11][6] = +1;
gamma_sign[11][7] = +1;
gamma_sign[11][8] = +1;
gamma_sign[11][9] = +1;
gamma_sign[11][10] = +1;
gamma_sign[11][11] = +1;
gamma_sign[11][12] = +1;
gamma_sign[11][13] = +1;
gamma_sign[11][14] = +1;
gamma_sign[11][15] = +1;
gamma_sign[11][16] = +1;
gamma_sign[11][17] = +1;
gamma_sign[11][18] = -1;
gamma_sign[11][19] = -1;
gamma_sign[11][20] = -1;
gamma_sign[11][21] = -1;
gamma_sign[11][22] = -1;
gamma_sign[11][23] = -1;
gamma_sign[12][0] = +1;
gamma_sign[12][1] = -1;
gamma_sign[12][2] = +1;
gamma_sign[12][3] = -1;
gamma_sign[12][4] = +1;
gamma_sign[12][5] = -1;
gamma_sign[12][6] = -1;
gamma_sign[12][7] = +1;
gamma_sign[12][8] = -1;
gamma_sign[12][9] = +1;
gamma_sign[12][10] = -1;
gamma_sign[12][11] = +1;
gamma_sign[12][12] = -1;
gamma_sign[12][13] = +1;
gamma_sign[12][14] = -1;
gamma_sign[12][15] = +1;
gamma_sign[12][16] = -1;
gamma_sign[12][17] = +1;
gamma_sign[12][18] = +1;
gamma_sign[12][19] = -1;
gamma_sign[12][20] = +1;
gamma_sign[12][21] = -1;
gamma_sign[12][22] = +1;
gamma_sign[12][23] = -1;
gamma_sign[13][0] = -1;
gamma_sign[13][1] = +1;
gamma_sign[13][2] = -1;
gamma_sign[13][3] = +1;
gamma_sign[13][4] = -1;
gamma_sign[13][5] = +1;
gamma_sign[13][6] = +1;
gamma_sign[13][7] = -1;
gamma_sign[13][8] = +1;
gamma_sign[13][9] = -1;
gamma_sign[13][10] = +1;
gamma_sign[13][11] = -1;
gamma_sign[13][12] = -1;
gamma_sign[13][13] = +1;
gamma_sign[13][14] = -1;
gamma_sign[13][15] = +1;
gamma_sign[13][16] = -1;
gamma_sign[13][17] = +1;
gamma_sign[13][18] = +1;
gamma_sign[13][19] = -1;
gamma_sign[13][20] = +1;
gamma_sign[13][21] = -1;
gamma_sign[13][22] = +1;
gamma_sign[13][23] = -1;
gamma_sign[14][0] = -1;
gamma_sign[14][1] = -1;
gamma_sign[14][2] = -1;
gamma_sign[14][3] = -1;
gamma_sign[14][4] = -1;
gamma_sign[14][5] = -1;
gamma_sign[14][6] = +1;
gamma_sign[14][7] = +1;
gamma_sign[14][8] = +1;
gamma_sign[14][9] = +1;
gamma_sign[14][10] = +1;
gamma_sign[14][11] = +1;
gamma_sign[14][12] = -1;
gamma_sign[14][13] = -1;
gamma_sign[14][14] = -1;
gamma_sign[14][15] = -1;
gamma_sign[14][16] = -1;
gamma_sign[14][17] = -1;
gamma_sign[14][18] = +1;
gamma_sign[14][19] = +1;
gamma_sign[14][20] = +1;
gamma_sign[14][21] = +1;
gamma_sign[14][22] = +1;
gamma_sign[14][23] = +1;
gamma_sign[15][0] = -1;
gamma_sign[15][1] = +1;
gamma_sign[15][2] = -1;
gamma_sign[15][3] = +1;
gamma_sign[15][4] = -1;
gamma_sign[15][5] = +1;
gamma_sign[15][6] = -1;
gamma_sign[15][7] = +1;
gamma_sign[15][8] = -1;
gamma_sign[15][9] = +1;
gamma_sign[15][10] = -1;
gamma_sign[15][11] = +1;
gamma_sign[15][12] = -1;
gamma_sign[15][13] = +1;
gamma_sign[15][14] = -1;
gamma_sign[15][15] = +1;
gamma_sign[15][16] = -1;
gamma_sign[15][17] = +1;
gamma_sign[15][18] = -1;
gamma_sign[15][19] = +1;
gamma_sign[15][20] = -1;
gamma_sign[15][21] = +1;
gamma_sign[15][22] = -1;
gamma_sign[15][23] = +1;

}

int printf_gauge_field(double *gauge, FILE *ofs) {

  int i, start_t=0;
  int x0, x1, x2, x3, ix;
  int y0, y1, y2, y3;
 
  if( (ofs == (FILE*)NULL) || (gauge==(double*)NULL) ) return(107);

#ifdef HAVE_MPI
  start_t = 1;
#endif

  for(x0=-start_t; x0<T+start_t; x0++) {
  for(x1= 0; x1<LX;  x1++) {
  for(x2= 0; x2<LX;  x2++) {
  for(x3= 0; x3<LX;  x3++) {
    y0=x0; y1=x1; y2=x2; y3=x3;
    if(x0==-1) y0=T+1;
    for(i=0; i<4; i++) {
      ix=18*( 4*g_ipt[y0][y1][y2][y3] + i);
      fprintf(ofs, "[%2d] # t=%3d,  x=%3d,  y=%3d,  z=%3d,  mu=%3d\n",
        g_cart_id, x0, x1, x2, x3, i);
      fprintf(ofs, "[%2d] (%12.5e)+(%12.5e)\t(%12.5e)+(%12.5e)\t(%12.5e)+(%12.5e)\n"\
                   "[%2d] (%12.5e)+(%12.5e)\t(%12.5e)+(%12.5e)\t(%12.5e)+(%12.5e)\n"\
                   "[%2d] (%12.5e)+(%12.5e)\t(%12.5e)+(%12.5e)\t(%12.5e)+(%12.5e)\n",
                    g_cart_id,
                    gauge[ix+ 0], gauge[ix+ 1],
                    gauge[ix+ 2], gauge[ix+ 3],
                    gauge[ix+ 4], gauge[ix+ 5],
                    g_cart_id,
                    gauge[ix+ 6], gauge[ix+ 7],
                    gauge[ix+ 8], gauge[ix+ 9],
                    gauge[ix+10], gauge[ix+11],
                    g_cart_id,
                    gauge[ix+12], gauge[ix+13],
                    gauge[ix+14], gauge[ix+15],
                    gauge[ix+16], gauge[ix+17]);
    }
  }}}}

  return(0);
}  /* end of printf_gauge_field */

void printf_cm( double*A, char*name, FILE*file) {

  fprintf(file, "%s <- array(0, dim=c(3,3))\n", name);
  _cm_fprintf(A, name,file);

}

int printf_SU3_link (double *u, FILE*ofs) {
  int i;
  fprintf(ofs, "# [printf_SU3_link] SU(3) link\n");
  for(i=0; i<9; i++) {
    fprintf(ofs, "\t%3d%3d%25.16e%25.16e\n", i/3, i%3, u[2*i], u[2*i+1]);
  }
  return(0);
}  // end of printf_SU3_link

int printf_spinor_field(double *s, int print_halo, FILE *ofs) {

  int i, start_valuet=0, start_valuex=0, start_valuey=0, start_valuez=0;
  int x0, x1, x2, x3, ix;
  int y0, y1, y2, y3;
  int z0, z1, z2, z3;
  int boundary;

  if( (ofs == (FILE*)NULL) || (s==(double*)NULL) ) return(108);

  if(print_halo) {
#ifdef HAVE_MPI
  start_valuet = 1;

#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ
  start_valuex = 1;
#  else
  start_valuex = 0;
# endif

#  if defined PARALLELTXY || defined PARALLELTXYZ
  start_valuey = 1;
#  else
  start_valuey = 0;
# endif

#  if defined PARALLELTXYZ
  start_valuez = 1;
#  else
  start_valuez = 0;
# endif

#else
  start_valuet = 0;
#endif
  }

  for(x0= -start_valuet; x0 < T +start_valuet; x0++) {
  for(x1= -start_valuex; x1 < LX+start_valuex; x1++) {
  for(x2= -start_valuey; x2 < LY+start_valuey; x2++) {
  for(x3= -start_valuez; x3 < LZ+start_valuez; x3++) {
    boundary =  (x0==-1 || x0==T) + (x1==-1 || x1==LX) + (x2==-1 || x2==LY) + (x3==-1 || x3==LZ);
    if(boundary>1) continue;
    y0=x0; y1=x1; y2=x2; y3=x3;
    if(x0==-1) y0=T+1;
    if(x1==-1) y1=LX+1;
    if(x2==-1) y2=LY+1;
    if(x3==-1) y3=LZ+1;

    ix = _GSI(g_ipt[y0][y1][y2][y3]);
    // fprintf(ofs, "# [%2d] t=%3d, x=%3d, y=%3d, z=%3d\n", g_cart_id, x0, x1, x2, x3);
    z0 = x0 + g_proc_coords[0] * T;
    z1 = x1 + g_proc_coords[1] * LX;
    z2 = x2 + g_proc_coords[2] * LY;
    z3 = x3 + g_proc_coords[3] * LZ;
    fprintf(ofs, "# [%2d] t=%3d, x=%3d, y=%3d, z=%3d\n", g_cart_id, z0, z1, z2, z3);
/*
    fprintf(ofs, "%18.9e %18.9e\t%18.9e %18.9e\t%18.9e %18.9e\n"\
                 "%18.9e %18.9e\t%18.9e %18.9e\t%18.9e %18.9e\n"\
                 "%18.9e %18.9e\t%18.9e %18.9e\t%18.9e %18.9e\n"\
                 "%18.9e %18.9e\t%18.9e %18.9e\t%18.9e %18.9e\n",
                 s[ix   ], s[ix+ 1], s[ix+ 2], s[ix+ 3], s[ix+ 4], s[ix+ 5],
                 s[ix +6], s[ix+ 7], s[ix+ 8], s[ix+ 9], s[ix+10], s[ix+11],
                 s[ix+12], s[ix+13], s[ix+14], s[ix+15], s[ix+16], s[ix+17],
                 s[ix+18], s[ix+19], s[ix+20], s[ix+21], s[ix+22], s[ix+23]);
*/
    for(i=0; i<12; i++) {
      fprintf(ofs, "s[%2d,%2d] <- %18.9e + %18.9e*1.i\n", ix/24+1, i+1, s[ix+2*i], s[ix+2*i+1]);
    }
  }}}}

  return(0);
}  /* end of printf_spinor_field */

/*******************************************************************
 * print an even-odd spinor field
 *******************************************************************/
int printf_eo_spinor_field(double *s, int use_even, int print_halo, FILE *ofs) {

  int i, start_valuet=0, start_valuex=0, start_valuey=0, start_valuez=0;
  int x0, x1, x2, x3, ix;
  int y0, y1, y2, y3;
  int z0, z1, z2, z3;
  int boundary;
  int jx;

  if( (ofs == (FILE*)NULL) || (s==(double*)NULL) ) return(108);

  if(print_halo > 0) {
#ifdef HAVE_MPI
    start_valuet = 1;

#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ
    start_valuex = 1;
#  else
    start_valuex = 0;
# endif

#  if defined PARALLELTXY || defined PARALLELTXYZ
    start_valuey = 1;
#  else
    start_valuey = 0;
# endif

#  if defined PARALLELTXYZ
    start_valuez = 1;
#  else
    start_valuez = 0;
# endif

#else
    start_valuet = 0;
#endif
  }  /* end of if print halo */

  if(g_cart_id == 0) fprintf(stdout, "# [printf_eo_spinor_field] start values = (%d, %d, %d, %d)\n", start_valuet, start_valuex, start_valuey, start_valuez);

  for(x0= -start_valuet; x0 < T +start_valuet; x0++) {
  for(x1= -start_valuex; x1 < LX+start_valuex; x1++) {
  for(x2= -start_valuey; x2 < LY+start_valuey; x2++) {
  for(x3= -start_valuez; x3 < LZ+start_valuez; x3++) {
    boundary =  (x0==-1 || x0==T) + (x1==-1 || x1==LX) + (x2==-1 || x2==LY) + (x3==-1 || x3==LZ);
    if(boundary>1) continue;
    y0=x0; y1=x1; y2=x2; y3=x3;
    if(x0==-1) y0=T+1;
    if(x1==-1) y1=LX+1;
    if(x2==-1) y2=LY+1;
    if(x3==-1) y3=LZ+1;

    ix = g_ipt[y0][y1][y2][y3];

    jx = g_lexic2eosub[ix];

    if( use_even != g_iseven[ix] ) continue;

    // fprintf(ofs, "# [%2d] t=%3d, x=%3d, y=%3d, z=%3d\n", g_cart_id, x0, x1, x2, x3);
    z0 = (x0 + g_proc_coords[0] * T  + T_global  ) % T_global ;
    z1 = (x1 + g_proc_coords[1] * LX + LX_global ) % LX_global;
    z2 = (x2 + g_proc_coords[2] * LY + LY_global ) % LY_global;
    z3 = (x3 + g_proc_coords[3] * LZ + LZ_global ) % LZ_global;
    /* fprintf(ofs, "# t=%d, x=%d, y=%d, z=%d %8d %3d\n", z0, z1, z2, z3, jx, boundary); */
    fprintf(ofs, "# t=%d x=%d y=%d z=%d lt=%d lx=%d ly=%d lz=%d ieo=%d b=%d\n", z0, z1, z2, z3, y0, y1, y2, y3, jx, boundary);

    for(i=0; i<12; i++) {
      fprintf(ofs, "\t%3d %18.9e %18.9e\n", i, s[_GSI(jx)+2*i], s[_GSI(jx)+2*i+1]);
    }
  }}}}

  return(0);
}  /* end of printf_eo_spinor_field */

int printf_spinor_field_5d(double *s, FILE *ofs) {

  int i, start_valuet=0, start_valuex=0, start_valuey=0;
  int x0, x1, x2, x3, ix, is;
  int gx0, gx1, gx2, gx3;
  int y0, y1, y2, y3;
  int boundary;

  if( (ofs == (FILE*)NULL) || (s==(double*)NULL) ) return(108);
#ifdef HAVE_MPI
  start_valuet = 1;
#ifdef PARALLELTX
  start_valuex = 1;
#elif defined PARALLELTXY
  start_valuex = 1;
  start_valuey = 1;
#endif
#endif
  for(is=0;is<L5;is++) {
    for(x0=-start_valuet; x0<T +start_valuet; x0++) {
    for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
    for(x2=-start_valuey; x2<LY+start_valuey; x2++) {
    for(x3= 0; x3<LZ;  x3++) {
      boundary = 0;
      if( x0==-1 || x0==T ) boundary++;
      if( x1==-1 || x1==LX) boundary++;
      if( x2==-1 || x2==LY) boundary++;
      if(boundary>1) continue;

      y0=x0; y1=x1; y2=x2; y3=x3;
      if(x0==-1) y0= T  + 1;
      if(x1==-1) y1= LX + 1;
      if(x2==-1) y2= LY + 1;

      gx0 = x0 + g_proc_coords[0] * T;
      gx1 = x1 + g_proc_coords[1] * LX;
      gx2 = x2 + g_proc_coords[2] * LY;
      gx3 = x3 + g_proc_coords[3] * LZ;

      ix = _GSI(g_ipt_5d[is][y0][y1][y2][y3]);
      fprintf(ofs, "# [%2d] s=%3d, t=%3d, x=%3d, y=%3d, z=%3d; %3d\n", g_cart_id, is, gx0, gx1, gx2, gx3, ix/24);
/*
      fprintf(ofs, "%18.9e %18.9e\t%18.9e %18.9e\t%18.9e %18.9e\n"\
                   "%18.9e %18.9e\t%18.9e %18.9e\t%18.9e %18.9e\n"\
                   "%18.9e %18.9e\t%18.9e %18.9e\t%18.9e %18.9e\n"\
                   "%18.9e %18.9e\t%18.9e %18.9e\t%18.9e %18.9e\n",
                   s[ix   ], s[ix+ 1], s[ix+ 2], s[ix+ 3], s[ix+ 4], s[ix+ 5],
                   s[ix +6], s[ix+ 7], s[ix+ 8], s[ix+ 9], s[ix+10], s[ix+11],
                   s[ix+12], s[ix+13], s[ix+14], s[ix+15], s[ix+16], s[ix+17],
                   s[ix+18], s[ix+19], s[ix+20], s[ix+21], s[ix+22], s[ix+23]);
*/
      for(i=0; i<12; i++) {
        fprintf(ofs, "s[%2d,%2d,%2d] <- %18.9e + %18.9e*1.i\n", is+1, g_ipt[y0][y1][y2][y3]+1, i+1, s[ix+2*i], s[ix+2*i+1]);
      }
    }}}}  // of z,y,x,t
  }       // of is

  return(0);
}

void  TraceAB(complex *w, double A[12][24], double B[12][24]) {

  int i, j;

  w->re = 0.; w->im = 0.;
  for(i=0; i<12; i++) {
  for(j=0; j<12; j++) {
    w->re += A[i][2*j]   * B[j][2*i] - A[i][2*j+1] * B[j][2*i+1];
    w->im += A[i][2*j+1] * B[j][2*i] + A[i][2*j]   * B[j][2*i+1];
  }
  }
}

void  TraceAdagB(complex *w, double A[12][24], double B[12][24]) {

  int i, j;

  w->re = 0.; w->im = 0.;
  for(i=0; i<12; i++) {
  for(j=0; j<12; j++) {
    w->re +=  A[j][2*i  ] * B[j][2*i] + A[j][2*i+1] * B[j][2*i+1];
    w->im += -A[j][2*i+1] * B[j][2*i] + A[j][2*i  ] * B[j][2*i+1];
  }
  }
}

/* make random gauge transformation g */
void init_gauge_trafo(double **g, double heat) {

  int ix, i;
  double ran[12], inorm, u[18], v[18], w[18];
#ifdef HAVE_MPI 
  int cntr;
  MPI_Request request[5];
  MPI_Status status[5];
#endif

  if(g_cart_id==0) fprintf(stdout, "# [init_gauge_trafo] initialising random gauge transformation\n");

  *g = (double*)calloc(18*VOLUMEPLUSRAND, sizeof(double));
  if(*g==(double*)NULL) {
    fprintf(stderr, "[init_gauge_trafo] not enough memory\n");
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(1);
  }
    
  /* set g to random field */
  for(ix=0; ix<VOLUME; ix++) {
    u[ 0] = 0.; u[ 1] = 0.; u[ 2] = 0.; u[ 3] = 0.; u[ 4] = 0.; u[ 5] = 0.;
    u[ 6] = 0.; u[ 7] = 0.; u[ 8] = 0.; u[ 9] = 0.; u[10] = 0.; u[11] = 0.;
    u[12] = 0.; u[13] = 0.; u[14] = 0.; u[15] = 0.; u[16] = 0.; u[17] = 0.;
    v[ 0] = 0.; v[ 1] = 0.; v[ 2] = 0.; v[ 3] = 0.; v[ 4] = 0.; v[ 5] = 0.;
    v[ 6] = 0.; v[ 7] = 0.; v[ 8] = 0.; v[ 9] = 0.; v[10] = 0.; v[11] = 0.; 
    v[12] = 0.; v[13] = 0.; v[14] = 0.; v[15] = 0.; v[16] = 0.; v[17] = 0.;
    w[ 0] = 0.; w[ 1] = 0.; w[ 2] = 0.; w[ 3] = 0.; w[ 4] = 0.; w[ 5] = 0.;
    w[ 6] = 0.; w[ 7] = 0.; w[ 8] = 0.; w[ 9] = 0.; w[10] = 0.; w[11] = 0.; 
    w[12] = 0.; w[13] = 0.; w[14] = 0.; w[15] = 0.; w[16] = 0.; w[17] = 0.;

/*
    ran[ 0]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 1]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 2]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[ 3]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[ 4]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[ 5]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 6]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 7]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 8]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 9]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[10]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[11]=((double)rand()) / ((double)RAND_MAX+1.0);
*/
    ranlxd(ran, 12);
    ran[0] = 1.0 + (ran[0]-0.5)*heat;
    ran[1] = (ran[1]-0.5)*heat;
    ran[2] = (ran[2]-0.5)*heat;
    ran[3] = (ran[3]-0.5)*heat;
    inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

    u[ 0] = ran[0]*inorm; u[1] = ran[1]*inorm;
    u[ 2] = ran[2]*inorm; u[3] = ran[3]*inorm;
    u[ 6] = -u[2]; u[7] =  u[3];
    u[ 8] =  u[0]; u[9] = -u[1];
    u[16] = 1.;

    ran[0] = 1.0 + (ran[4]-0.5)*heat;
    ran[1] = (ran[5]-0.5)*heat;
    ran[2] = (ran[6]-0.5)*heat;
    ran[3] = (ran[7]-0.5)*heat;
    inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

    v[0] = 1.;
    v[ 8] = ran[0]*inorm; v[ 9] = ran[1]*inorm;
    v[14] = ran[2]*inorm; v[15] = ran[3]*inorm;
    v[10] = -v[14]; v[11] =  v[15];
    v[16] =  v[ 8]; v[17] = -v[ 9];

    _cm_eq_cm_ti_cm(&(*g)[18*ix], u, v);


    ran[0] = 1.0 + (ran[8]-0.5)*heat;
    ran[1] = (ran[9]-0.5)*heat;
    ran[2] = (ran[10]-0.5)*heat;
    ran[3] = (ran[11]-0.5)*heat;
    inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

    w[8] = 1.;
    w[ 0] = ran[0]*inorm; w[ 1] = ran[1]*inorm;
    w[12] = ran[2]*inorm; w[13] = ran[3]*inorm;
    w[ 4] = -w[12]; w[ 5] =  w[13];
    w[16] =  w[ 0]; w[17] = -w[ 1];

    _cm_eq_cm(u, &(*g)[18*ix]);
    _cm_eq_cm_ti_cm(&(*g)[18*ix], u, w);
  }

  /* exchange the field */

/*
  fprintf(stdout, "\n# [init_gauge_trafo] The gauge trafo field:\n");
  fprintf(stdout, "\n\tg <- array(0., dim=c(%d,%d,%d))\n", VOLUME, 3,3);
  for(ix=0; ix<VOLUME; ix++) {
    for(i=0;i<9;i++) {
      fprintf(stdout, "\tg[%6d,%d,%d] <- %25.16e + %25.16e*1.i\n", ix+1, i/3+1, i%3+1, (*g)[2*(9*ix+i)], (*g)[2*(9*ix+i)+1]);
    }
  }
*/
#ifdef HAVE_MPI

  cntr = 0;

  MPI_Isend(&(*g)[0], 18*LX*LY*LZ, MPI_DOUBLE, g_nb_t_dn, 83, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&(*g)[18*VOLUME], 18*LX*LY*LZ, MPI_DOUBLE, g_nb_t_up, 83, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&(*g)[18*(T-1)*LX*LY*LZ], 18*LX*LY*LZ, MPI_DOUBLE, g_nb_t_up, 84, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&(*g)[18*(T+1)*LX*LY*LZ], 18*LX*LY*LZ, MPI_DOUBLE, g_nb_t_dn, 84, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);
#endif

/*
  fprintf(stdout, "Writing gauge transformation to std out\n");
  const char *gauge_format="[(%25.16e +i*(%25.16e)), (%25.16e +i*(%25.16e)), (%25.16e +i*(%25.16e));"\
                          " (%25.16e +i*(%25.16e)), (%25.16e +i*(%25.16e)), (%25.16e +i*(%25.16e));"\
                         " (%25.16e +i*(%25.16e)), (%25.16e +i*(%25.16e)), (%25.16e +i*(%25.16e))]\n";
  for(ix=0; ix<N; ix++) {
   fprintf(stdout, "ix=%3d\n", ix);
   fprintf(stdout, gauge_format, 
     g[18*ix+ 0], g[18*ix+ 1], g[18*ix+ 2], g[18*ix+ 3], g[18*ix+ 4], g[18*ix+ 5], 
     g[18*ix+ 6], g[18*ix+ 7], g[18*ix+ 8], g[18*ix+ 9], g[18*ix+10], g[18*ix+11], 
     g[18*ix+12], g[18*ix+13], g[18*ix+14], g[18*ix+15], g[18*ix+16], g[18*ix+17]);
  }
*/
}

void apply_gt_gauge(double *g) {

  int ix, mu;
  double u[18];
  FILE* ofs;

  if(g_cart_id==0) fprintf(stdout, "applying gauge transformation to gauge field\n");
/*
  ofs = fopen("gauge_field_before", "w");
  printf_gauge_field(g_gauge_field, ofs);
  fclose(ofs);
*/
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      _cm_eq_cm_ti_cm(u, &g[18*ix], &g_gauge_field[_GGI(ix,mu)]);
      _cm_eq_cm_ti_cm_dag(&g_gauge_field[_GGI(ix, mu)], u, &g[18*g_iup[ix][mu]]);
    }
  }
  xchange_gauge();
/*
  ofs = fopen("gauge_field_after", "w");
  printf_gauge_field(g_gauge_field, ofs);
  fclose(ofs);
*/
}

/* apply gt to propagator; (is,ic) = (spin, colour)-index */
void apply_gt_prop(double *g, double *phi, int is, int ic, int mu, char *basename, int source_location) {
#ifndef HAVE_MPI
  int ix, ix1[5], k;
  double psi1[24], psi2[24], psi3[24], *work[3];
  complex co[3];
  char filename[200];

  /* allocate memory for work spinor fields */
  alloc_spinor_field(&work[0], VOLUME);
  alloc_spinor_field(&work[1], VOLUME);
  alloc_spinor_field(&work[2], VOLUME);

  if(format==0) {
    sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", basename, Nconf, mu, 3*is+0);
    read_lime_spinor(work[0], filename, 0);
    sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", basename, Nconf, mu, 3*is+1);
    read_lime_spinor(work[1], filename, 0);
    sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", basename, Nconf, mu, 3*is+2);
    read_lime_spinor(work[2], filename, 0);
  }
  else if(format==4) {
    sprintf(filename, "%s.%.4d.%.2d.inverted", basename, Nconf, 3*is+0);
    read_lime_spinor(work[0], filename, 0);
    sprintf(filename, "%s.%.4d.%.2d.inverted", basename, Nconf, 3*is+1);
    read_lime_spinor(work[1], filename, 0);
    sprintf(filename, "%s.%.4d.%.2d.inverted", basename, Nconf, 3*is+2);
    read_lime_spinor(work[2], filename, 0);
  }

  /* apply g to propagators from the left */
  for(k=0; k<3; k++) {
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_cm_ti_fv(psi1, &g[18*ix], &work[k][_GSI(ix)]);
      _fv_eq_fv(&work[k][_GSI(ix)], psi1);
    }
  }

  /* apply g to propagators from the right */
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_zero(psi1);
    _fv_eq_zero(psi2);
    _fv_eq_zero(psi3);

    if(mu==4) {
      co[0].re =  g[18*source_location + 6*ic+0];
      co[0].im = -g[18*source_location + 6*ic+1];
      co[1].re =  g[18*source_location + 6*ic+2];
      co[1].im = -g[18*source_location + 6*ic+3];
      co[2].re =  g[18*source_location + 6*ic+4];
      co[2].im = -g[18*source_location + 6*ic+5];
    }
    else {
      co[0].re =  g[18*g_iup[source_location][mu] + 6*ic+0];
      co[0].im = -g[18*g_iup[source_location][mu] + 6*ic+1];
      co[1].re =  g[18*g_iup[source_location][mu] + 6*ic+2];
      co[1].im = -g[18*g_iup[source_location][mu] + 6*ic+3];
      co[2].re =  g[18*g_iup[source_location][mu] + 6*ic+4];
      co[2].im = -g[18*g_iup[source_location][mu] + 6*ic+5];
    }

    _fv_eq_fv_ti_co(psi1, &work[0][_GSI(ix)], &co[0]);
    _fv_eq_fv_ti_co(psi2, &work[1][_GSI(ix)], &co[1]);
    _fv_eq_fv_ti_co(psi3, &work[2][_GSI(ix)], &co[2]);

    _fv_eq_fv_pl_fv(&phi[_GSI(ix)], psi1, psi2);
    _fv_pl_eq_fv(&phi[_GSI(ix)], psi3);
  }

  free(work[0]); free(work[1]); free(work[2]);
#endif
}

void get_filename(char *filename, const int nu, const int sc, const int sign) {

  int isx[4], Lsize[4];

//  if(format==2 || format==3) {
    isx[0] =  g_source_location/(LX_global * LY_global * LZ_global);
    isx[1] = (g_source_location%(LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
    isx[2] = (g_source_location%(LY_global * LZ_global)) / LZ_global;
    isx[3] = (g_source_location%LZ_global);
    Lsize[0] = T_global;
    Lsize[1] = LX_global;
    Lsize[2] = LY_global;
    Lsize[3] = LZ_global;
//  }

  if(format==0) {  // format from invert
    if(sign==1) { sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix,  Nconf, nu, sc); }
    else if(sign==-1) { sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix2, Nconf, nu, sc); }
    else if(sign == 0) {
      if(nu!=4) isx[nu] = (isx[nu]+1)%Lsize[nu];
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.inverted", 
          filename_prefix, Nconf, isx[0], isx[1], isx[2], isx[3], sc);
    }
  }
  else if(format==2) {  // Dru/Xu format
    if(nu!=4) isx[nu] = (isx[nu]+1)%Lsize[nu];
    if(sign==1) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2ds%.2dc%.2d.pmass.%s.inverted", 
        filename_prefix, Nconf, isx[0], isx[1], isx[2], isx[3], sc/3, sc%3, filename_prefix2);
    }
    else {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2ds%.2dc%.2d.nmass.%s.inverted", 
        filename_prefix, Nconf, isx[0], isx[1], isx[2], isx[3], sc/3, sc%3, filename_prefix2);
    }
  }
  else if(format==3) {  // Dru/Xu format
    if(nu!=4) isx[nu] = (isx[nu]+1)%Lsize[nu];
    if(sign==1) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.pmass.%s.%.2d.inverted", 
        filename_prefix, Nconf, isx[0], isx[1], isx[2], isx[3], filename_prefix2, sc);
    }
    else {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.nmass.%s.%.2d.inverted", 
        filename_prefix, Nconf, isx[0], isx[1], isx[2], isx[3], filename_prefix2, sc);
    }
  }
  else if(format==4) {  // my format for lvc
    if(sign==1) {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sc);
    }
    else {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix2, Nconf, sc);
    }
  }
  // else if(format==1) {  // GWC format }
}

int wilson_loop(complex *w, const int xstart, const int dir, const int Ldir) {

  int ix, i;
  double U_[18], V_[18], *u1=(double*)NULL, *u2=(double*)NULL, *u3=(double*)NULL;
  complex tr, phase;

  if(dir==0) {
    ix=g_iup[xstart][dir];
    _cm_eq_cm_ti_cm(V_, g_gauge_field+_GGI(xstart, dir), g_gauge_field+_GGI(ix, dir));
    u1=U_; u2=V_;
    for(i=2; i<Ldir; i++) {
      ix = g_iup[ix][dir];
      u3=u1; u1=u2; u2=u3;
      _cm_eq_cm_ti_cm(u2, u1, g_gauge_field+_GGI(ix,dir));
    }
#ifdef HAVE_MPI
    if(g_cart_id==0) {
      fprintf(stderr, "MPI version _NOT_ yet implemented;\n");
      return(1);
    }
#endif
    _co_eq_tr_cm(&tr, u2);
  } else {
    ix=g_iup[xstart][dir];
    _cm_eq_cm_ti_cm(V_, g_gauge_field+_GGI(xstart, dir), g_gauge_field+_GGI(ix, dir));
    u1=U_; u2=V_;
    for(i=2; i<Ldir; i++) {
      ix = g_iup[ix][dir];
      u3=u1; u1=u2; u2=u3;
      _cm_eq_cm_ti_cm(u2, u1, g_gauge_field+_GGI(ix,dir));
    }
    _co_eq_tr_cm(&tr, u2);
  }

  phase.re = cos( M_PI * BCangle[dir] );
  phase.im = sin( M_PI * BCangle[dir] );

  _co_eq_co_ti_co(w, &tr, &phase);

  return(0);
 
}

int IRand(int min, int max) {
  return( min + (int)( ((double)(max-min+1)) * ((double)(rand())) /
                      ((double)(RAND_MAX) + 1.0) ) );
}

double Random_Z2() {
  if(IRand(0, 1) == 0)
    return(+1.0 / sqrt(2.0));
  return( -1.0 / sqrt(2.0));
}  /* end of Random_Z2 */

int ranz2(double * y, unsigned int NRAND) {
  const double sqrt2inv = 1. / sqrt(2.);
  unsigned int k;

  ranlxd(y, NRAND);

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(k=0; k<NRAND; k++) {
    y[k] = (double)(2 * (int)(y[k]>=0.5) - 1) * sqrt2inv;
  }
  return(0);
}  /* end of ranz2 */

/********************************************************
 * random_gauge_field
 *
 * NOTE: we assume the random number generator has
 * been properly initialized in the case of MPI usage;
 * we cease to have g_cart_id 0 do all the sampling
 * and sending
 ********************************************************/
void random_gauge_field(double *gfield, double h) {

  int mu, ix;
  double buffer[72], *gauge_point[4];
#ifdef HAVE_MPI
  int iproc, tgeom[2], tag, t;
  double *gauge_ts;
  MPI_Status mstatus;
#endif

  gauge_point[0] = buffer;
  gauge_point[1] = buffer + 18;
  gauge_point[2] = buffer + 36;
  gauge_point[3] = buffer + 54;

/*  if(g_cart_id==0) { */
    for(ix=0; ix<VOLUME; ix++) {
      random_gauge_point(gauge_point, h);
      memcpy((void*)(gfield + _GGI(ix,0)), (void*)buffer, 72*sizeof(double));
    }
/*  } */
#if 0
#ifdef HAVE_MPI
  if(g_cart_id==0) {
    if( (gauge_ts = (double*)malloc(72*LX*LY*LZ*sizeof(double))) == (double*)NULL ) {
       MPI_Abort(MPI_COMM_WORLD, 1);
       MPI_Finalize();
       exit(101);
    }
  }
  tgeom[0] = Tstart;
  tgeom[1] = T;
  for(iproc=1; iproc<g_nproc; iproc++) {
    if(g_cart_id==0) {
      tag = 2*iproc;
      MPI_Recv((void*)tgeom, 2, MPI_INT, iproc, tag, g_cart_grid, &mstatus);
      for(t=0; t<tgeom[1]; t++) {
        for(ix=0; ix<LX*LY*LZ; ix++) {
          random_gauge_point(gauge_point, h);
          memcpy((void*)(gauge_ts + _GGI(ix,0)), (void*)buffer, 72*sizeof(double));
        }
        tag = 2 * ( t * g_nproc + iproc ) + 1;
        MPI_Send((void*)gauge_ts, 72*LX*LY*LZ, MPI_DOUBLE, iproc, tag, g_cart_grid);

      }
    }
    if(g_cart_id==iproc) {
      tag = 2*iproc;
      MPI_Send((void*)tgeom, 2, MPI_INT, 0, tag, g_cart_grid);
      for(t=0; t<T; t++) {
        tag = 2 * ( t * g_nproc + iproc ) + 1;
        MPI_Recv((void*)(gfield + _GGI(g_ipt[t][0][0][0],0)), 72*LX*LY*LZ, MPI_DOUBLE, 0, tag, g_cart_grid, &mstatus);
      }
    }
    MPI_Barrier(g_cart_grid);
  }
  if(g_cart_id==0) free(gauge_ts);
#endif  
#endif  /* of if 0 */
}  /* end of random_gauge_field */

/********************************************************
 * random spinor field
 ********************************************************/
void random_spinor_field (double *s, unsigned int V)  {

  const double norm = 1. / sqrt(2.);
  unsigned int ix;
  double r, phi;

  ranlxd(s, 24*V);

  if(g_noise_type == 1) {
    /* Gaussian noise */

    for(ix=0; ix<12*V; ix++) {
      /* Box-Muller method */
      r = sqrt( -2.*log( s[2*ix] ) );
      phi = 2. * M_PI * s[2*ix+1];
      s[2*ix  ] = r * cos(phi);
      s[2*ix+1] = r * sin(phi);
    }
  } else if(g_noise_type == 2) {
    /* Z2 x Z2 */
    for(ix=0; ix<24*V; ix++) {
      r = 2. * (double)( s[ix] < 0.5 ) - 1.;
      s[ix] = r * norm;
    }
  }

}  /* end of random_spinor_field */


/******************************************************
 * read_pimn
 ******************************************************/
int read_pimn(double *pimn, const int read_flag) {

  char filename[800];
  int iostat, ix, mu, nu, iix, np;
  double ratime, retime, buff[32], *buff2=(double*)NULL;

  FILE *ofs;
#ifndef HAVE_MPI
  /* read the data file */
  if(format==0 || format==1) {
    if(Nsave>=0) {
      sprintf(filename, "%s.%.4d.%.4d", filename_prefix, Nconf, Nsave);
    }
    else {
      sprintf(filename, "%s.%.4d", filename_prefix, Nconf);
    }
  }
  else {
    sprintf(filename, "%s", filename_prefix);
  }

  if((void*)(ofs = fopen(filename, "r"))==NULL) {
    fprintf(stderr, "could not open file %s for reading\n", filename);
    return(-1);
  }

  ratime = clock() / CLOCKS_PER_SEC;
  if(read_flag==0) {
    if(format==1) {
      fprintf(stdout, "reading of binary data from file %s\n", filename);
      for(ix=0; ix<VOLUME; ix++) {
        iostat = fread(buff, sizeof(double), 32, ofs);
        if(iostat != 32) {
          fprintf(stderr, "could not read proper amount of data\n");
          return(-3);
        }
        /* fprintf(stdout, "ix = %d\n", ix); */
        for(mu=0; mu<16; mu++) {
          pimn[_GWI(mu,ix,VOLUME)  ] = buff[2*mu  ];
          pimn[_GWI(mu,ix,VOLUME)+1] = buff[2*mu+1];
        }
      }
    }
    else if(format>100) {
      np = format - 100; /* number of processes */
      fprintf(stdout, "reconstructing old version of io-output with np=%d\n", np);
      buff2 = (double*)malloc(VOLUME/np*32*sizeof(double));
      for(nu=0; nu<np; nu++) {
        fread(buff2, sizeof(double), (VOLUME/np)*32, ofs);
        for(mu=0; mu<16; mu++) {
          for(ix=0; ix<VOLUME/np; ix++) {
            pimn[_GWI(mu,ix+nu*(VOLUME/np),VOLUME)  ] = buff2[_GWI(mu,ix,VOLUME/np)  ];
            pimn[_GWI(mu,ix+nu*(VOLUME/np),VOLUME)+1] = buff2[_GWI(mu,ix,VOLUME/np)+1];
          }
        }
      }
      free(buff2);
    }
    else if(format==0 || format==2) {
      fprintf(stdout, "buffered reading of binary data from file %s in format %2d\n", filename, format);
      for(ix=0; ix<VOLUME; ix++) {
        iostat = fread(buff, sizeof(double), 32, ofs);
        if(iostat != 32) {
          fprintf(stderr, "could not read proper amount of data\n");
          return(-3);
        }
        for(mu=0; mu<16; mu++) {
          iix = index_conv(16*ix+mu,format);
          /* fprintf(stdout, "Dru's index: %8d\t my index: %8d\n", 16*ix+mu, iix); */
          pimn[iix  ] = buff[2*mu  ];
          pimn[iix+1] = buff[2*mu+1];
        }
      }
    }
  }
  else if(read_flag==1 && format==1) {
    /* reading from ascii file only in my format for testing purposes */
    fprintf(stdout, "reading in format 1 from ascii file %s\n", filename);
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<16; mu++) {
        fscanf(ofs, "%lf%lf", pimn+_GWI(mu,ix,VOLUME), pimn+_GWI(mu,ix,VOLUME)+1);
      }
    }
  }
  retime = clock() / CLOCKS_PER_SEC;
  fprintf(stdout, "done in %e seconds\n", retime-ratime);
  fclose(ofs);
  return(0);
#else
  return(-1);
#endif
}


/*********************************************
 * random_cm
 *********************************************/
void random_cm(double *A, double heat) {

  double ran[12], inorm, u[18], v[18], w[18], aux[18];

  u[ 0] = 0.; u[ 1] = 0.; u[ 2] = 0.; u[ 3] = 0.; u[ 4] = 0.; u[ 5] = 0.;
  u[ 6] = 0.; u[ 7] = 0.; u[ 8] = 0.; u[ 9] = 0.; u[10] = 0.; u[11] = 0.;
  u[12] = 0.; u[13] = 0.; u[14] = 0.; u[15] = 0.; u[16] = 0.; u[17] = 0.;
  v[ 0] = 0.; v[ 1] = 0.; v[ 2] = 0.; v[ 3] = 0.; v[ 4] = 0.; v[ 5] = 0.;
  v[ 6] = 0.; v[ 7] = 0.; v[ 8] = 0.; v[ 9] = 0.; v[10] = 0.; v[11] = 0.; 
  v[12] = 0.; v[13] = 0.; v[14] = 0.; v[15] = 0.; v[16] = 0.; v[17] = 0.;
  w[ 0] = 0.; w[ 1] = 0.; w[ 2] = 0.; w[ 3] = 0.; w[ 4] = 0.; w[ 5] = 0.;
  w[ 6] = 0.; w[ 7] = 0.; w[ 8] = 0.; w[ 9] = 0.; w[10] = 0.; w[11] = 0.; 
  w[12] = 0.; w[13] = 0.; w[14] = 0.; w[15] = 0.; w[16] = 0.; w[17] = 0.;

  ranlxd(ran,12);

  ran[0] = 1.0 + (ran[0]-0.5)*heat;
  ran[1] = (ran[1]-0.5)*heat;
  ran[2] = (ran[2]-0.5)*heat;
  ran[3] = (ran[3]-0.5)*heat;
  inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

  u[ 0] = ran[0]*inorm; u[1] = ran[1]*inorm;
  u[ 2] = ran[2]*inorm; u[3] = ran[3]*inorm;
  u[ 6] = -u[2]; u[7] =  u[3];
  u[ 8] =  u[0]; u[9] = -u[1];
  u[16] = 1.;

  ran[0] = 1.0 + (ran[4]-0.5)*heat;
  ran[1] = (ran[5]-0.5)*heat;
  ran[2] = (ran[6]-0.5)*heat;
  ran[3] = (ran[7]-0.5)*heat;
  inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

  v[0] = 1.;
  v[ 8] = ran[0]*inorm; v[ 9] = ran[1]*inorm;
  v[14] = ran[2]*inorm; v[15] = ran[3]*inorm;
  v[10] = -v[14]; v[11] =  v[15];
  v[16] =  v[ 8]; v[17] = -v[ 9];

  _cm_eq_cm_ti_cm(aux, u, v);

  ran[0] = 1.0 + (ran[8]-0.5)*heat;
  ran[1] = (ran[9]-0.5)*heat;
  ran[2] = (ran[10]-0.5)*heat;
  ran[3] = (ran[11]-0.5)*heat;
  inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

  w[8] = 1.;
  w[ 0] = ran[0]*inorm; w[ 1] = ran[1]*inorm;
  w[12] = ran[2]*inorm; w[13] = ran[3]*inorm;
  w[ 4] = -w[12]; w[ 5] =  w[13];
  w[16] =  w[ 0]; w[17] = -w[ 1];

  _cm_eq_cm_ti_cm(A, aux, w);
}  /* end of random_cm */


/*********************************************
 * random_gauge_point
 * - now with factors from 3 SU(2) subgroups
 *********************************************/
void random_gauge_point(double **gauge_point, double heat) {

  int mu;
  double ran[12], inorm, u[18], v[18], w[18], aux[18];


  for(mu=0; mu<4; mu++) {
    u[ 0] = 0.; u[ 1] = 0.; u[ 2] = 0.; u[ 3] = 0.; u[ 4] = 0.; u[ 5] = 0.;
    u[ 6] = 0.; u[ 7] = 0.; u[ 8] = 0.; u[ 9] = 0.; u[10] = 0.; u[11] = 0.;
    u[12] = 0.; u[13] = 0.; u[14] = 0.; u[15] = 0.; u[16] = 0.; u[17] = 0.;
    v[ 0] = 0.; v[ 1] = 0.; v[ 2] = 0.; v[ 3] = 0.; v[ 4] = 0.; v[ 5] = 0.;
    v[ 6] = 0.; v[ 7] = 0.; v[ 8] = 0.; v[ 9] = 0.; v[10] = 0.; v[11] = 0.; 
    v[12] = 0.; v[13] = 0.; v[14] = 0.; v[15] = 0.; v[16] = 0.; v[17] = 0.;
    w[ 0] = 0.; w[ 1] = 0.; w[ 2] = 0.; w[ 3] = 0.; w[ 4] = 0.; w[ 5] = 0.;
    w[ 6] = 0.; w[ 7] = 0.; w[ 8] = 0.; w[ 9] = 0.; w[10] = 0.; w[11] = 0.; 
    w[12] = 0.; w[13] = 0.; w[14] = 0.; w[15] = 0.; w[16] = 0.; w[17] = 0.;

    ranlxd(ran,12);
/*
    ran[ 0]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 1]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 2]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[ 3]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[ 4]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[ 5]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 6]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 7]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[ 8]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[ 9]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[10]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[11]=((double)rand()) / ((double)RAND_MAX+1.0);
*/

    ran[0] = 1.0 + (ran[0]-0.5)*heat;
    ran[1] = (ran[1]-0.5)*heat;
    ran[2] = (ran[2]-0.5)*heat;
    ran[3] = (ran[3]-0.5)*heat;
    inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

    u[ 0] = ran[0]*inorm; u[1] = ran[1]*inorm;
    u[ 2] = ran[2]*inorm; u[3] = ran[3]*inorm;
    u[ 6] = -u[2]; u[7] =  u[3];
    u[ 8] =  u[0]; u[9] = -u[1];
    u[16] = 1.;

    ran[0] = 1.0 + (ran[4]-0.5)*heat;
    ran[1] = (ran[5]-0.5)*heat;
    ran[2] = (ran[6]-0.5)*heat;
    ran[3] = (ran[7]-0.5)*heat;
    inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

    v[0] = 1.;
    v[ 8] = ran[0]*inorm; v[ 9] = ran[1]*inorm;
    v[14] = ran[2]*inorm; v[15] = ran[3]*inorm;
    v[10] = -v[14]; v[11] =  v[15];
    v[16] =  v[ 8]; v[17] = -v[ 9];

    _cm_eq_cm_ti_cm(aux, u, v);

    ran[0] = 1.0 + (ran[8]-0.5)*heat;
    ran[1] = (ran[9]-0.5)*heat;
    ran[2] = (ran[10]-0.5)*heat;
    ran[3] = (ran[11]-0.5)*heat;
    inorm = 1.0 / sqrt(ran[0]*ran[0] + ran[1]*ran[1] + ran[2]*ran[2] + ran[3]*ran[3]);

    w[8] = 1.;
    w[ 0] = ran[0]*inorm; w[ 1] = ran[1]*inorm;
    w[12] = ran[2]*inorm; w[13] = ran[3]*inorm;
    w[ 4] = -w[12]; w[ 5] =  w[13];
    w[16] =  w[ 0]; w[17] = -w[ 1];

    _cm_eq_cm_ti_cm(gauge_point[mu], aux, w);
  }
}  /* end of random_gauge_point */


/********************************************************
 * random_gauge_field2
 ********************************************************/
void random_gauge_field2(double *gfield) {

  int mu, ix, i;
  double norm;
  complex u[3], v[3], w[3], z[3], pr;
#ifdef HAVE_MPI
  int iproc, tgeom[2], tag, t;
  double *gauge_ts;
  MPI_Status mstatus;
#endif


  if(g_cart_id==0) {
    for(ix=0; ix<VOLUME; ix++) {
      for(mu=0; mu<4; mu++) {
        u[0].re = (double)rand() / ((double)RAND_MAX + 1.);
        u[0].im = (double)rand() / ((double)RAND_MAX + 1.);
        u[1].re = (double)rand() / ((double)RAND_MAX + 1.);
        u[1].im = (double)rand() / ((double)RAND_MAX + 1.);
        u[2].re = (double)rand() / ((double)RAND_MAX + 1.);
        u[2].im = (double)rand() / ((double)RAND_MAX + 1.);

        v[0].re = (double)rand() / ((double)RAND_MAX + 1.);
        v[0].im = (double)rand() / ((double)RAND_MAX + 1.);
        v[1].re = (double)rand() / ((double)RAND_MAX + 1.);
        v[1].im = (double)rand() / ((double)RAND_MAX + 1.);
        v[2].re = (double)rand() / ((double)RAND_MAX + 1.);
        v[2].im = (double)rand() / ((double)RAND_MAX + 1.);
     
        norm = 1. / sqrt( u[0].re*u[0].re + u[0].im*u[0].im + u[1].re*u[1].re + u[1].im*u[1].im + u[2].re*u[2].re + u[2].im*u[2].im );
        _co_ti_eq_re(u+0, norm);
        _co_ti_eq_re(u+1, norm);
        _co_ti_eq_re(u+2, norm);

        pr.re = 0.; pr.im = 0.;
        _co_pl_eq_co_ti_co_conj(&pr, v+0, u+0);
        _co_pl_eq_co_ti_co_conj(&pr, v+1, u+1);
        _co_pl_eq_co_ti_co_conj(&pr, v+2, u+2);
        
        _co_eq_co_ti_co(z+0, u+0, &pr);
        _co_eq_co_ti_co(z+1, u+1, &pr);
        _co_eq_co_ti_co(z+2, u+2, &pr);

        _co_mi_eq_co(v+0, z+0);
        _co_mi_eq_co(v+1, z+1);
        _co_mi_eq_co(v+2, z+2);

        norm = 1. / sqrt( v[0].re*v[0].re + v[0].im*v[0].im + v[1].re*v[1].re + v[1].im*v[1].im + v[2].re*v[2].re + v[2].im*v[2].im );
        _co_ti_eq_re(v+0, norm);
        _co_ti_eq_re(v+1, norm);
        _co_ti_eq_re(v+2, norm);

        _co_eq_co_ti_co(w+0, u+1, v+2);
        _co_eq_co_ti_co(w+1, u+2, v+0);
        _co_eq_co_ti_co(w+2, u+0, v+1);

        _co_mi_eq_co_ti_co(w+0, u+2, v+1); w[0].im *= -1.;
        _co_mi_eq_co_ti_co(w+1, u+0, v+2); w[1].im *= -1.;
        _co_mi_eq_co_ti_co(w+2, u+1, v+0); w[2].im *= -1.;

/*
        fprintf(stdout, "u = c(%25.16e+%25.16ei, %25.16e+%25.16ei, %25.16e+%25.16ei)\n", u[0].re, u[0].im, u[1].re, u[1].im, u[2].re, u[2].im);
        fprintf(stdout, "v = c(%25.16e+%25.16ei, %25.16e+%25.16ei, %25.16e+%25.16ei)\n", v[0].re, v[0].im, v[1].re, v[1].im, v[2].re, v[2].im);
        fprintf(stdout, "w = c(%25.16e+%25.16ei, %25.16e+%25.16ei, %25.16e+%25.16ei)\n", w[0].re, w[0].im, w[1].re, w[1].im, w[2].re, w[2].im);
        fprintf(stdout, "==============================================================");
*/
        for(i=0; i<3; i++) {
          gfield[_GGI(ix,mu) +   2*i  ] =  u[i].re;
          gfield[_GGI(ix,mu) +   2*i+1] =  u[i].im;
          gfield[_GGI(ix,mu) + 6+2*i  ] =  v[i].re;
          gfield[_GGI(ix,mu) + 6+2*i+1] =  v[i].im;
          gfield[_GGI(ix,mu) +12+2*i  ] =  w[i].re;
          gfield[_GGI(ix,mu) +12+2*i+1] =  w[i].im;
        }
      }
    }
  }
#ifdef HAVE_MPI
  if(g_cart_id==0) {
    if( (gauge_ts = (double*)malloc(72*LX*LY*LZ*sizeof(double))) == (double*)NULL ) {
       MPI_Abort(MPI_COMM_WORLD, 1);
       MPI_Finalize();
       exit(101);
    }
  }
  tgeom[0] = Tstart;
  tgeom[1] = T;
  for(iproc=1; iproc<g_nproc; iproc++) {
    if(g_cart_id==0) {
      tag = 2*iproc;
      MPI_Recv((void*)tgeom, 2, MPI_INT, iproc, tag, g_cart_grid, &mstatus);
      for(t=0; t<tgeom[1]; t++) {
        for(ix=0; ix<LX*LY*LZ; ix++) {
          for(mu=0; mu<4; mu++) {
            u[0].re = (double)rand() / ((double)RAND_MAX + 1.);
            u[0].im = (double)rand() / ((double)RAND_MAX + 1.);
            u[1].re = (double)rand() / ((double)RAND_MAX + 1.);
            u[1].im = (double)rand() / ((double)RAND_MAX + 1.);
            u[2].re = (double)rand() / ((double)RAND_MAX + 1.);
            u[2].im = (double)rand() / ((double)RAND_MAX + 1.);

            v[0].re = (double)rand() / ((double)RAND_MAX + 1.);
            v[0].im = (double)rand() / ((double)RAND_MAX + 1.);
            v[1].re = (double)rand() / ((double)RAND_MAX + 1.);
            v[1].im = (double)rand() / ((double)RAND_MAX + 1.);
            v[2].re = (double)rand() / ((double)RAND_MAX + 1.);
            v[2].im = (double)rand() / ((double)RAND_MAX + 1.);
     
            norm = 1. / sqrt( u[0].re*u[0].re + u[0].im*u[0].im + u[1].re*u[1].re + u[1].im*u[1].im + u[2].re*u[2].re + u[2].im*u[2].im );
            _co_ti_eq_re(u+0, norm);
            _co_ti_eq_re(u+1, norm);
            _co_ti_eq_re(u+2, norm);
  
            pr.re = 0.; pr.im = 0.;
            _co_pl_eq_co_ti_co_conj(&pr, v+0, u+0);
            _co_pl_eq_co_ti_co_conj(&pr, v+1, u+1);
            _co_pl_eq_co_ti_co_conj(&pr, v+2, u+2);
        
            _co_eq_co_ti_co(z+0, u+0, &pr);
            _co_eq_co_ti_co(z+1, u+1, &pr);
            _co_eq_co_ti_co(z+2, u+2, &pr);

            _co_mi_eq_co(v+0, z+0);
            _co_mi_eq_co(v+1, z+1);
            _co_mi_eq_co(v+2, z+2);

            norm = 1. / sqrt( v[0].re*v[0].re + v[0].im*v[0].im + v[1].re*v[1].re + v[1].im*v[1].im + v[2].re*v[2].re + v[2].im*v[2].im );
            _co_ti_eq_re(v+0, norm);
            _co_ti_eq_re(v+1, norm);
            _co_ti_eq_re(v+2, norm);

            _co_eq_co_ti_co(w+0, u+1, v+2);
            _co_eq_co_ti_co(w+1, u+2, v+0);
            _co_eq_co_ti_co(w+2, u+0, v+1);

            _co_mi_eq_co_ti_co(w+0, u+2, v+1); w[0].im *= -1.;
            _co_mi_eq_co_ti_co(w+1, u+0, v+2); w[1].im *= -1.;
            _co_mi_eq_co_ti_co(w+2, u+1, v+0); w[2].im *= -1.;
        
            for(i=0; i<3; i++) {
              gauge_ts[_GGI(ix,mu) +   2*i  ] =  u[i].re;
              gauge_ts[_GGI(ix,mu) +   2*i+1] =  u[i].im;
              gauge_ts[_GGI(ix,mu) + 6+2*i  ] =  v[i].re;
              gauge_ts[_GGI(ix,mu) + 6+2*i+1] =  v[i].im;
              gauge_ts[_GGI(ix,mu) +12+2*i  ] =  w[i].re;
              gauge_ts[_GGI(ix,mu) +12+2*i+1] =  w[i].im;
            }
          }
        }
        tag = 2 * ( t * g_nproc + iproc ) + 1;
        MPI_Send((void*)gauge_ts, 72*LX*LY*LZ, MPI_DOUBLE, iproc, tag, g_cart_grid);

      }
    }
    if(g_cart_id==iproc) {
      tag = 2*iproc;
      MPI_Send((void*)tgeom, 2, MPI_INT, 0, tag, g_cart_grid);
      for(t=0; t<T; t++) {
        tag = 2 * ( t * g_nproc + iproc ) + 1;
        MPI_Recv((void*)(gfield + _GGI(g_ipt[t][0][0][0],0)), 72*LX*LY*LZ, MPI_DOUBLE, 0, tag, g_cart_grid, &mstatus);
      }
    }
    MPI_Barrier(g_cart_grid);
  }
  if(g_cart_id==0) free(gauge_ts);
#endif  
}

/***************************************************
 * Generates a random integer between min and max.
 ***************************************************/

/*************************************************************************
 * pseudo random number generator for
 * Gaussian distribution
 * uses Box-Muller method
 * cf. Press, Teukolsky, Vetterling, Flannery: Numerical Receipes in C++. 
 * Second Edition. Cambridge University Press, 2002
 *************************************************************************/

int rangauss (double * y1, unsigned int NRAND) {

  const double TWO_MPI = 2. * M_PI;
  const unsigned int nrandh = NRAND/2;
  unsigned int k, k2, k2p1;
  double x1;

  if(NRAND%2 != 0) {
    fprintf(stderr, "ERROR, NRAND must be an even number\n");
    return(1);
  }

  /* fill the complete field y1 */
  ranlxd(y1,NRAND);

#ifdef HAVE_OPEMP
#pragma omp parallel for private(k2,k2p1,x1)
#endif
  for(k=0; k<nrandh; k++) {
    k2   = 2*k;
    k2p1 = k2+1;

    x1       = sqrt( -2. * log(y1[k2]) );
    y1[k2]   = x1 * cos( TWO_MPI * y1[k2p1] );
    y1[k2p1] = x1 * sin( TWO_MPI * y1[k2p1] );
  }  /* end of loop on nrandh */
  return(0);
}

/*************************************************************************
 *
 * projection of smeared colour matrices to SU(3)
 * 
 *************************************************************************/

#ifdef F_
#define _F(s) s##_
#else
#define _F(s) s
#endif

extern "C" int _F(ilaenv)(int *ispec, char name[], char opts[], int *n1, int *n2, int *n3, int *n4);

extern "C" void _F(zheev)(char *jobz, char *uplo, int *n, double a[], int *lda, double w[], double work[], int *lwork, double *rwork, int *info);

/*****************************************************************************
 * Computes the eigenvalues and the eigenvectors of a hermitian 3x3 matrix.
 *
 * n = size of the matrix
 * M = hermitean 3x3 matrix (only the upper triangle is needed);
 *
 *   M[ 0] + i M[ 1]   M[ 6] + i M[ 7]   M[12] + i M[13]
 *   xxx               M[ 8] + i M[ 9]   M[14] + i M[15]
 *   xxx               xxx               M[16] + i M[17]
 *
 *   at the end of the computation the eigenvectors are stored here
 *
 *   v[0] = (lambda[ 0] + i lambda[ 1]   lambda[2] + i lambda[3]   ...
 *   v[1] = (lambda[ 6] + i lambda[ 7]   ...
 *   v[2] = (lambda[12] + i lambda[13]   ...
 *
 * lambda = eigenvalues sorted in ascending order
 *****************************************************************************/
void EV_Hermitian_3x3_Matrix(double *M, double *lambda) {
  int n = 3, lwork, info;
  double *work=NULL, rwork[9];

/* 
  int one = 1;
  int m_one = -1;
  fprintf(stdout, "# calling ilaenv to get optimized size of work\n");
*/
  lwork = 102;
/* 
  lwork = (2 + _F(ilaenv)(&one, "zhetrd", "VU", &n, &m_one, &m_one, &m_one)) * 3;
  fprintf(stdout, "# finished ilaenv with lwork = %d\n", lwork); 
*/

  work = (double *)malloc(lwork * 2 * sizeof(double));

  _F(zheev)("V", "U", &n, M, &n, lambda, work, &lwork, rwork, &info);

  free(work);
}

/*****************************************************************************
 *
 * A = Proj_SU3(A).
 *
 * Projects a color matrix on SU(3).
 *
 * A' = A / sqrt(A^\dagger A)
 *
 * P_{SU(3)}(A) = A' / det(A')^{1/3}
 *****************************************************************************/
void cm_proj(double *A) {
  double d1;
  double M1[18], M2[18], M3[18];
  double lambda[3], phi;
  complex det, de1_3_cc;

  /* Compute A^\dagger A. */
  _cm_eq_cm_dag_ti_cm(M1, A, A);

  /* Compute the eigenvalues and the eigenvectors of A^\dagger A.
   *
   * Transpose A^\dagger A (this is needed to call a Fortan/Lapack function to
   * compute the eigenvectors and eigenvalues). */

  M1[ 1] = -M1[ 1];
  M1[ 3] = -M1[ 3];
  M1[ 5] = -M1[ 5];

  M1[ 7] = -M1[ 7];
  M1[ 9] = -M1[ 9];
  M1[11] = -M1[11];

  M1[13] = -M1[13];
  M1[15] = -M1[15];
  M1[17] = -M1[17];

  EV_Hermitian_3x3_Matrix(M1, lambda);

/*  fprintf(stderr, "lambda = (%+6.3lf  , %+6.3lf , %+6.3lf).\n", lambda[0], lambda[1], lambda[2]); */

  if(lambda[0] <= 0.000000000001 || lambda[1] <= 0.000000000001 || lambda[2] <= 0.000000000001) {
    fprintf(stderr, "lambda = (%+6.3f  , %+6.3f , %+6.3f).\n", lambda[0], lambda[1], lambda[2]);
    fprintf(stderr, "Error: inline void SU3_proj(...\n");
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(102);
  }

  /* Compute "T^\dagger". */
  M1[ 1] = -M1[ 1];
  M1[ 3] = -M1[ 3];
  M1[ 5] = -M1[ 5];

  M1[ 7] = -M1[ 7];
  M1[ 9] = -M1[ 9];
  M1[11] = -M1[11];

  M1[13] = -M1[13];
  M1[15] = -M1[15];
  M1[17] = -M1[17];

  /* Compute "T D^{-1/2}". */
  _cm_eq_cm_dag(M2, M1);

  d1 = 1.0 / sqrt(lambda[0]);
  M2[ 0] *= d1;
  M2[ 1] *= d1;
  M2[ 6] *= d1;
  M2[ 7] *= d1;
  M2[12] *= d1;
  M2[13] *= d1;

  d1 = 1.0 / sqrt(lambda[1]);
  M2[ 2] *= d1;
  M2[ 3] *= d1;
  M2[ 8] *= d1;
  M2[ 9] *= d1;
  M2[14] *= d1;
  M2[15] *= d1;

  d1 = 1.0 / sqrt(lambda[2]);
  M2[ 4] *= d1;
  M2[ 5] *= d1;
  M2[10] *= d1;
  M2[11] *= d1;
  M2[16] *= d1;
  M2[17] *= d1;


  /* Compute "T D^{-1/2} T^\dagger". */
  _cm_eq_cm_ti_cm(M3, M2, M1);

  /* Compute A'. */
  _cm_eq_cm_ti_cm(M1, A, M3);

  /* Divide by det(A')^{1/3}. */
  _co_eq_det_cm(&det, M1);
  phi = atan2(det.im, det.re) / 3.0;

  de1_3_cc.re = +cos(phi);
  de1_3_cc.im = -sin(phi);

  _cm_eq_cm_ti_co(A, M1, &de1_3_cc);
}

void reunit(double *A) {
#ifndef _SQR
#define _SQR(_a) ((_a)*(_a))
#endif
   double v1[6], v2[6], v3[6];
   double l1, l2, l3;

   v1[0] = A[0]; v1[1] = A[1];
   v1[2] = A[2]; v1[3] = A[3];
   v1[4] = A[4]; v1[5] = A[5];

   /* normalize v1 */
   l1 = sqrt( _SQR( v1[0] ) + _SQR( v1[1] ) + _SQR( v1[2] ) + _SQR( v1[3] ) + _SQR( v1[4] ) + _SQR( v1[5] ) );
   l1 = 1. / l1;
   v1[0] *= l1; v1[1] *= l1; v1[2] *= l1; v1[3] *= l1; v1[4] *= l1; v1[5] *= l1; 

   /* orthognormalize v2 with respect to v1 */
   v2[0] = A[ 6]; v2[1] = A[ 7];
   v2[2] = A[ 8]; v2[3] = A[ 9];
   v2[4] = A[10]; v2[5] = A[11];
   l2 =  v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3] + v1[4] * v2[4] + v1[5] * v2[5];
   l3 = -v1[1] * v2[0] + v1[0] * v2[1] - v1[3] * v2[2] + v1[2] * v2[3] - v1[5] * v2[4] + v1[4] * v2[5];

   v2[0] -= l2 * v1[0] - l3 * v1[1];
   v2[1] -= l2 * v1[1] + l3 * v1[0];
   v2[2] -= l2 * v1[2] - l3 * v1[3];
   v2[3] -= l2 * v1[3] + l3 * v1[2];
   v2[4] -= l2 * v1[4] - l3 * v1[5];
   v2[5] -= l2 * v1[5] + l3 * v1[4];

   l2 = sqrt( _SQR( v2[0] ) + _SQR( v2[1] ) + _SQR( v2[2] ) + _SQR( v2[3] ) + _SQR( v2[4] ) + _SQR( v2[5] ) );
   l2 = 1. / l2;
   v2[0] *= l2; v2[1] *= l2; v2[2] *= l2; v2[3] *= l2; v2[4] *= l2; v2[5] *= l2; 

   /* v3 from v1 x v2 */

   v3[0] =  ( v1[2] * v2[4] - v1[3] * v2[5] - v1[4] * v2[2] + v1[5] * v2[3] );
   v3[1] = -( v1[2] * v2[5] + v1[3] * v2[4] - v1[4] * v2[3] - v1[5] * v2[2] ); 
   v3[2] =  ( v1[4] * v2[0] - v1[5] * v2[1] - v1[0] * v2[4] + v1[1] * v2[5] );
   v3[3] = -( v1[4] * v2[1] + v1[5] * v2[0] - v1[0] * v2[5] - v1[1] * v2[4] );
   v3[4] =  ( v1[0] * v2[2] - v1[1] * v2[3] - v1[2] * v2[0] + v1[3] * v2[1] );
   v3[5] = -( v1[0] * v2[3] + v1[1] * v2[2] - v1[2] * v2[1] - v1[3] * v2[0] );


   A[ 0] = v1[0]; A[ 1] = v1[1]; A[ 2] = v1[2]; A[ 3] = v1[3]; A[ 4] = v1[4]; A[ 5] = v1[5];
   A[ 6] = v2[0]; A[ 7] = v2[1]; A[ 8] = v2[2]; A[ 9] = v2[3]; A[10] = v2[4]; A[11] = v2[5];
   A[12] = v3[0]; A[13] = v3[1]; A[14] = v3[2]; A[15] = v3[3]; A[16] = v3[4]; A[17] = v3[5];
 
}  /* end of reunit*/

/*****************************************************************************************************************
 * projection step for cm_proj_iterate
 *****************************************************************************************************************/
void su3_proj_step (double *A, double *B) {

  const double COLOR_EPS = 1.e-15;
  double U[18], V[18];
  double r0, r1, r2, r3, r_l;
  double a0, a1, a2, a3;
  int i, j, ii_re, ii_im, jj_re, jj_im, ij_re, ij_im, ji_re, ji_im;

  _cm_eq_cm_ti_cm_dag(U, A, B);
  for(i=1; i<3; i++) {

    ii_re = 8*i;
    ii_im = 8*i+1;
    for(j=0; j<i; j++) {
      jj_re = 8*j;
      jj_im = 8*j+1;
      ij_re = 2*(3*i+j);
      ij_im = 2*(3*i+j)+1;
      ji_re = 2*(3*j+i);
      ji_im = 2*(3*j+i)+1;

      r0 = U[ii_re] + U[jj_re];
      r3 = U[ii_im] - U[jj_im];

      r2 = U[ij_re] - U[ji_re];
      r1 = U[ij_im] + U[ji_im];

      r_l = sqrt( r0*r0 + r1*r1 + r2*r2 + r3*r3 );

      /* fprintf(stdout, "# [] r = %e, %e, %e, %e\t%e\n", r0, r1, r2, r3, r_l); */
      if(r_l > COLOR_EPS) {
        r_l = 1. / r_l;
        a0 =  r0 * r_l;
        a1 = -r1 * r_l;
        a2 = -r2 * r_l;
        a3 = -r3 * r_l;
      } else {
        a0 = 1.; a1 = 0.; a2 = 0.; a3 = 0.;
      }
      /* fprintf(stdout, "# [] a = %e, %e, %e, %e\n", a0, a1, a2, a3); */
      _cm_eq_id(V);
      
      V[ii_re] =  a0;
      V[ii_im] =  a3;

      V[jj_re] =  a0;
      V[jj_im] = -a3;

      V[ij_re] =  a2;
      V[ij_im] =  a1;

      V[ji_re] = -a2;
      V[ji_im] =  a1;

      _cm_eq_cm_ti_cm(U, V, A);
      _cm_eq_cm(A, U);
    }
  }
}  /* end of su3_proj_step */

/*****************************************************************************************************************
 * iterative projection of GL(3, C) on SU(3)
 *****************************************************************************************************************/
void cm_proj_iterate(double *A, double *B, int maxiter, double tol) {
  const double ONETHIRD = 0.333333333333333333333333333333333;
  double conver = 1.;
  int iter=0;
  double tr_old, tr_new;
  double U[18];
  complex w;

  _cm_eq_cm(A, B);
  _cm_eq_cm_ti_cm_dag(U, A, B);
  _co_eq_tr_cm(&w, U);
  tr_new = w.re * ONETHIRD;

  for(iter=0; iter<maxiter && conver>tol; iter++) {
    tr_old = tr_new;
    su3_proj_step(A, B);
    reunit(A);

    /* TEST */
    /* fprintf(stdout, "# [cm_proj_iterate] A[%d]\n", iter);
    _cm_fprintf(A, stdout);
    */

    _cm_eq_cm_ti_cm_dag(U, A, B);
    _co_eq_tr_cm(&w, U);
    tr_new = w.re * ONETHIRD;
    conver = fabs((tr_new - tr_old)/tr_old);
  }
  if(conver > tol || iter == maxiter) {
    fprintf(stderr, "[cm_proj_iterate] Error, projection did not converge\n");
  }
}  /* end of cm_proj_iterate */

/*****************************************************************************************************************
 * contraction of meson 2-point functions
 *****************************************************************************************************************/
void contract_twopoint(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c) {

  int x0, ix, iix, psource[4], tt=0, isimag, mu, c, j;
  int VOL3 = LX*LY*LZ;
  double ssource[4], spinor1[24], spinor2[24];
  complex w;

  psource[0] = gamma_permutation[idsource][ 0] / 6;
  psource[1] = gamma_permutation[idsource][ 6] / 6;
  psource[2] = gamma_permutation[idsource][12] / 6;
  psource[3] = gamma_permutation[idsource][18] / 6;
  isimag = gamma_permutation[idsource][ 0] % 2;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  ssource[0] =  gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]];
  ssource[1] =  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]];
  ssource[2] =  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]];
  ssource[3] =  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]];
/*
  fprintf(stdout, "__________________________________\n");
  fprintf(stdout, "isource=%d, idsink=%d, p[0] = %d\n", idsource, idsink, psource[0]);
  fprintf(stdout, "isource=%d, idsink=%d, p[1] = %d\n", idsource, idsink, psource[1]);
  fprintf(stdout, "isource=%d, idsink=%d, p[2] = %d\n", idsource, idsink, psource[2]);
  fprintf(stdout, "isource=%d, idsink=%d, p[3] = %d\n", idsource, idsink, psource[3]);
*/

/*  if(g_cart_id==0) fprintf(stdout, "# %3d %3d ssource = %e\t%e\t%e\t%e\n", idsource, idsink,
    ssource[0], ssource[1], ssource[2], ssource[3]); */

  for(x0=0; x0<T; x0++) {
    for(ix=0; ix<VOL3; ix++) {
      iix = x0*VOL3 + ix;
      for(mu=0; mu<4; mu++) {
        for(c=0; c<n_c; c++) {
/*
           if(g_cart_id==0 && (iix==0 || iix==111)) {
             fprintf(stdout, "iix=%4d, c=%d, mu=%d, idsource=%d, idsink=%d\n", iix, c, mu, idsource, idsink); 
             for(j=0; j<12; j++) 
               fprintf(stdout, "phi = %e +I %e\n", phi[mu*n_c+c][_GSI(iix)+2*j], phi[mu*n_c+c][_GSI(iix)+2*j+1]);
           }
*/
          _fv_eq_gamma_ti_fv(spinor1, idsink, phi[mu*n_c+c]+_GSI(iix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, chi[psource[mu]*n_c+c]+_GSI(iix), spinor2);

          if( !isimag ) {
            contr[2*tt  ] += ssource[mu]*w.re;
            contr[2*tt+1] += ssource[mu]*w.im;
          } else {
/*
            contr[2*tt  ] += ssource[mu]*w.im;
            contr[2*tt+1] += ssource[mu]*w.re;
*/
            contr[2*tt  ] +=  ssource[mu]*w.im;
            contr[2*tt+1] += -ssource[mu]*w.re;
          }
/*          if(g_cart_id==0) fprintf(stdout, "# source[%2d, %2d] = %25.16e +I %25.16e\n", mu, tt, ssource[mu]*w.re, ssource[mu]*w.im); */
        }
      }
    }
    tt++;
  }
}

/*****************************************************************************************************************
 * contraction of meson 2-point functions with sink momentum
 *****************************************************************************************************************/
#ifndef HAVE_OPENMP
void contract_twopoint_snk_momentum(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int* snk_mom) {

  int x0, x1, x2, x3, ix, iix, psource[4], tt=0, isimag, mu, c, j;
  int VOL3 = LX*LY*LZ;
  int sx0=0, sx1=0, sx2=0, sx3=0;
  double ssource[4], spinor1[24], spinor2[24];
  double phase, cphase, sphase, px, py, pz;
  double tmp[2], re, im;
  complex w;

  psource[0] = gamma_permutation[idsource][ 0] / 6;
  psource[1] = gamma_permutation[idsource][ 6] / 6;
  psource[2] = gamma_permutation[idsource][12] / 6;
  psource[3] = gamma_permutation[idsource][18] / 6;
  isimag = gamma_permutation[idsource][ 0] % 2;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  ssource[0] =  gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]];
  ssource[1] =  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]];
  ssource[2] =  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]];
  ssource[3] =  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]];
/*
  fprintf(stdout, "__________________________________\n");
  fprintf(stdout, "isource=%d, idsink=%d, p[0] = %d\n", idsource, idsink, psource[0]);
  fprintf(stdout, "isource=%d, idsink=%d, p[1] = %d\n", idsource, idsink, psource[1]);
  fprintf(stdout, "isource=%d, idsink=%d, p[2] = %d\n", idsource, idsink, psource[2]);
  fprintf(stdout, "isource=%d, idsink=%d, p[3] = %d\n", idsource, idsink, psource[3]);
*/

/*  if(g_cart_id==0) fprintf(stdout, "# %3d %3d ssource = %e\t%e\t%e\t%e\n", idsource, idsink,
    ssource[0], ssource[1], ssource[2], ssource[3]); */

  if(g_source_type==0) { // point souce
    iix = g_source_location;
    sx0 = iix / (LX_global * LY_global * LZ_global);
    iix -= sx0 * LX_global * LY_global * LZ_global;
    sx1 = iix / (LY_global * LZ_global);
    iix -= sx1 * LY_global * LZ_global;
    sx2 = iix / (LZ_global);
    sx3 = iix - sx2 * LZ_global;
    //fprintf(stdout, "# [contract_twopoint_snk_momentum] global source coordinates = (%d, %d, %d, %d)\n", sx0, sx1, sx2, sx3);
  }

  px = 2.*M_PI*(double)snk_mom[0]/(double)LX_global;
  py = 2.*M_PI*(double)snk_mom[1]/(double)LY_global;
  pz = 2.*M_PI*(double)snk_mom[2]/(double)LZ_global;
  // fprintf(stdout, "# [contract_twopoint_snk_momentum] p = (%e, %e, %e)\n", px, py, pz);

  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      iix = g_ipt[x0][x1][x2][x3];
      phase = (double)(x1+g_proc_coords[1]*LX - sx1)*px + (double)(x2+g_proc_coords[2]*LY - sx2)*py + (double)(x3+g_proc_coords[3]*LZ - sx3)*pz;
      cphase = cos( phase );
      sphase = sin( phase );

      tmp[0] = 0.; tmp[1] = 0.;
      for(mu=0; mu<4; mu++) {
        for(c=0; c<n_c; c++) {
/*
           if(g_cart_id==0 && (iix==0 || iix==111)) {
             fprintf(stdout, "iix=%4d, c=%d, mu=%d, idsource=%d, idsink=%d\n", iix, c, mu, idsource, idsink); 
             for(j=0; j<12; j++) 
               fprintf(stdout, "phi = %e +I %e\n", phi[mu*n_c+c][_GSI(iix)+2*j], phi[mu*n_c+c][_GSI(iix)+2*j+1]);
           }
*/
          _fv_eq_gamma_ti_fv(spinor1, idsink, phi[mu*n_c+c]+_GSI(iix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, chi[psource[mu]*n_c+c]+_GSI(iix), spinor2);

          tmp[0] += ssource[mu]*w.re;
          tmp[1] += ssource[mu]*w.im;

/*          if(g_cart_id==0) fprintf(stdout, "# source[%2d, %2d] = %25.16e +I %25.16e\n", mu, tt, ssource[mu]*w.re, ssource[mu]*w.im); */
        }  // of color  index
      }    // of spinor index

      // multiply by momentum phase factor and add to correlator
      re = tmp[0]*cphase - tmp[1]*sphase;
      im = tmp[0]*sphase + tmp[1]*cphase;
      //fprintf(stdout, "\tephase<-%e+%e*1.i; tmp<-%e+%e*1.i; z<-%e+%e*1.i\n", cphase, sphase, tmp[0], tmp[1], re, im);
      if( !isimag ) {
        contr[2*tt  ] += re;
        contr[2*tt+1] += im;
      } else {
        contr[2*tt  ] +=  im;
        contr[2*tt+1] += -re;
      }
    }}}  // of x3, x2, x1
    tt++;
  }      // of x0
}
#else
void contract_twopoint_snk_momentum(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int* snk_mom) {

  int x0, iix, psource[4], isimag;
  int sx0=0, sx1=0, sx2=0, sx3=0;
  double ssource[4];
  double px, py, pz;

  psource[0] = gamma_permutation[idsource][ 0] / 6;
  psource[1] = gamma_permutation[idsource][ 6] / 6;
  psource[2] = gamma_permutation[idsource][12] / 6;
  psource[3] = gamma_permutation[idsource][18] / 6;
  isimag = gamma_permutation[idsource][ 0] % 2;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  ssource[0] =  gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]];
  ssource[1] =  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]];
  ssource[2] =  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]];
  ssource[3] =  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]];
/*
  fprintf(stdout, "__________________________________\n");
  fprintf(stdout, "isource=%d, idsink=%d, p[0] = %d\n", idsource, idsink, psource[0]);
  fprintf(stdout, "isource=%d, idsink=%d, p[1] = %d\n", idsource, idsink, psource[1]);
  fprintf(stdout, "isource=%d, idsink=%d, p[2] = %d\n", idsource, idsink, psource[2]);
  fprintf(stdout, "isource=%d, idsink=%d, p[3] = %d\n", idsource, idsink, psource[3]);
*/

/*  if(g_cart_id==0) fprintf(stdout, "# %3d %3d ssource = %e\t%e\t%e\t%e\n", idsource, idsink,
    ssource[0], ssource[1], ssource[2], ssource[3]); */

  if(g_source_type==0) { // point souce
    iix = g_source_location;
    sx0 = iix / (LX_global * LY_global * LZ_global);
    iix -= sx0 * LX_global * LY_global * LZ_global;
    sx1 = iix / (LY_global * LZ_global);
    iix -= sx1 * LY_global * LZ_global;
    sx2 = iix / (LZ_global);
    sx3 = iix - sx2 * LZ_global;
    //fprintf(stdout, "# [contract_twopoint_snk_momentum] global source coordinates = (%d, %d, %d, %d)\n", sx0, sx1, sx2, sx3);
  }

  px = 2.*M_PI*(double)snk_mom[0]/(double)LX_global;
  py = 2.*M_PI*(double)snk_mom[1]/(double)LY_global;
  pz = 2.*M_PI*(double)snk_mom[2]/(double)LZ_global;
  // fprintf(stdout, "# [contract_twopoint_snk_momentum] p = (%e, %e, %e)\n", px, py, pz);

#pragma omp parallel private(x0, iix) shared(T,LX,LY,LZ, sx0,sx1,sx2,sx3, px, py, pz, isimag, ssource, psource, contr, chi, phi, n_c, snk_mom)
{
  int x1, x2, x3, ix, tt=0, mu, c, j;
  double phase, cphase, sphase;
  double tmp[2], re, im;
  complex w;
  double spinor1[24], spinor2[24];

  int threadid = omp_get_thread_num();
  int num_threads = omp_get_num_threads();

  // TEST
  // fprintf(stdout, "# [contract_twopoint_snk_momentum] thread%.4d number of threads %d\n", threadid, num_threads);

  tt = threadid;
  for(x0=threadid; x0<T; x0+=num_threads) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      iix = g_ipt[x0][x1][x2][x3];
      // phase = (double)(x1-sx1)*px + (double)(x2-sx2)*py + (double)(x3-sx3)*pz;
      phase = (double)(x1+g_proc_coords[1]*LX - sx1)*px + (double)(x2+g_proc_coords[2]*LY - sx2)*py + (double)(x3+g_proc_coords[3]*LZ - sx3)*pz;
      cphase = cos( phase );
      sphase = sin( phase );

      tmp[0] = 0.; tmp[1] = 0.;
      for(mu=0; mu<4; mu++) {
        for(c=0; c<n_c; c++) {
/*
           if(g_cart_id==0 && (iix==0 || iix==111)) {
             fprintf(stdout, "iix=%4d, c=%d, mu=%d, idsource=%d, idsink=%d\n", iix, c, mu, idsource, idsink); 
             for(j=0; j<12; j++) 
               fprintf(stdout, "phi = %e +I %e\n", phi[mu*n_c+c][_GSI(iix)+2*j], phi[mu*n_c+c][_GSI(iix)+2*j+1]);
           }
*/
          _fv_eq_gamma_ti_fv(spinor1, idsink, phi[mu*n_c+c]+_GSI(iix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, chi[psource[mu]*n_c+c]+_GSI(iix), spinor2);

          tmp[0] += ssource[mu]*w.re;
          tmp[1] += ssource[mu]*w.im;

/*          if(g_cart_id==0) fprintf(stdout, "# source[%2d, %2d] = %25.16e +I %25.16e\n", mu, tt, ssource[mu]*w.re, ssource[mu]*w.im); */
        }  // of color  index
      }    // of spinor index

      // multiply by momentum phase factor and add to correlator
      re = tmp[0]*cphase - tmp[1]*sphase;
      im = tmp[0]*sphase + tmp[1]*cphase;
      //fprintf(stdout, "\tephase<-%e+%e*1.i; tmp<-%e+%e*1.i; z<-%e+%e*1.i\n", cphase, sphase, tmp[0], tmp[1], re, im);
      if( !isimag ) {
        contr[2*tt  ] += re;
        contr[2*tt+1] += im;
      } else {
        contr[2*tt  ] +=  im;
        contr[2*tt+1] += -re;
      }
    }}}  // of x3, x2, x1
    tt += num_threads;
  }      // of x0
}  // end of parallel region
}
#endif


/*****************************************************************************************************************
 * contraction of meson 2-point functions with sink momentum and for a time interval
 * - 0 <= tmin, tmax <= T-1
 *****************************************************************************************************************/
#ifndef HAVE_OPENMP
void contract_twopoint_snk_momentum_trange(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int* snk_mom, int tmin, int tmax) {

  int x0, x1, x2, x3, ix, iix, psource[4], tt=0, isimag, mu, c, j;
  int VOL3 = LX*LY*LZ;
  int sx0=0, sx1=0, sx2=0, sx3=0;
  double ssource[4], spinor1[24], spinor2[24];
  double phase, cphase, sphase, px, py, pz;
  double tmp[2], re, im;
  complex w;

  psource[0] = gamma_permutation[idsource][ 0] / 6;
  psource[1] = gamma_permutation[idsource][ 6] / 6;
  psource[2] = gamma_permutation[idsource][12] / 6;
  psource[3] = gamma_permutation[idsource][18] / 6;
  isimag = gamma_permutation[idsource][ 0] % 2;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  ssource[0] =  gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]];
  ssource[1] =  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]];
  ssource[2] =  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]];
  ssource[3] =  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]];
/*
  fprintf(stdout, "__________________________________\n");
  fprintf(stdout, "isource=%d, idsink=%d, p[0] = %d\n", idsource, idsink, psource[0]);
  fprintf(stdout, "isource=%d, idsink=%d, p[1] = %d\n", idsource, idsink, psource[1]);
  fprintf(stdout, "isource=%d, idsink=%d, p[2] = %d\n", idsource, idsink, psource[2]);
  fprintf(stdout, "isource=%d, idsink=%d, p[3] = %d\n", idsource, idsink, psource[3]);
*/

/*  if(g_cart_id==0) fprintf(stdout, "# %3d %3d ssource = %e\t%e\t%e\t%e\n", idsource, idsink,
    ssource[0], ssource[1], ssource[2], ssource[3]); */

  if(g_source_type==0) { // point souce
    iix = g_source_location;
    sx0 = iix / (LX_global * LY_global * LZ_global);
    iix -= sx0 * LX_global * LY_global * LZ_global;
    sx1 = iix / (LY_global * LZ_global);
    iix -= sx1 * LY_global * LZ_global;
    sx2 = iix / (LZ_global);
    sx3 = iix - sx2 * LZ_global;
    //fprintf(stdout, "# [contract_twopoint_snk_momentum] global source coordinates = (%d, %d, %d, %d)\n", sx0, sx1, sx2, sx3);
  }

  px = 2.*M_PI*(double)snk_mom[0]/(double)LX_global;
  py = 2.*M_PI*(double)snk_mom[1]/(double)LY_global;
  pz = 2.*M_PI*(double)snk_mom[2]/(double)LZ_global;
  // fprintf(stdout, "# [contract_twopoint_snk_momentum] p = (%e, %e, %e)\n", px, py, pz);

  tt = 0;
  for(x0=tmin; x0<=tmax; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      iix = g_ipt[x0][x1][x2][x3];
      //phase = (double)(x1-sx1)*px + (double)(x2-sx2)*py + (double)(x3-sx3)*pz;
      phase = (double)(x1+g_proc_coords[1]*LX - sx1)*px + (double)(x2+g_proc_coords[2]*LY - sx2)*py + (double)(x3+g_proc_coords[3]*LZ - sx3)*pz;

      cphase = cos( phase );
      sphase = sin( phase );

      tmp[0] = 0.; tmp[1] = 0.;
      for(mu=0; mu<4; mu++) {
        for(c=0; c<n_c; c++) {
/*
           if(g_cart_id==0 && (iix==0 || iix==111)) {
             fprintf(stdout, "iix=%4d, c=%d, mu=%d, idsource=%d, idsink=%d\n", iix, c, mu, idsource, idsink); 
             for(j=0; j<12; j++) 
               fprintf(stdout, "phi = %e +I %e\n", phi[mu*n_c+c][_GSI(iix)+2*j], phi[mu*n_c+c][_GSI(iix)+2*j+1]);
           }
*/
          _fv_eq_gamma_ti_fv(spinor1, idsink, phi[mu*n_c+c]+_GSI(iix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, chi[psource[mu]*n_c+c]+_GSI(iix), spinor2);

          tmp[0] += ssource[mu]*w.re;
          tmp[1] += ssource[mu]*w.im;

/*          if(g_cart_id==0) fprintf(stdout, "# source[%2d, %2d] = %25.16e +I %25.16e\n", mu, tt, ssource[mu]*w.re, ssource[mu]*w.im); */
        }  // of color  index
      }    // of spinor index

      // multiply by momentum phase factor and add to correlator
      re = tmp[0]*cphase - tmp[1]*sphase;
      im = tmp[0]*sphase + tmp[1]*cphase;
      //fprintf(stdout, "\tephase<-%e+%e*1.i; tmp<-%e+%e*1.i; z<-%e+%e*1.i\n", cphase, sphase, tmp[0], tmp[1], re, im);
      if( !isimag ) {
        contr[2*tt  ] += re;
        contr[2*tt+1] += im;
      } else {
        contr[2*tt  ] +=  im;
        contr[2*tt+1] += -re;
      }
    }}}  // of x3, x2, x1
    tt++;
  }      // of x0
}
#else
void contract_twopoint_snk_momentum_trange(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int* snk_mom, int tmin, int tmax) {

  int x0, iix, tt, psource[4], isimag;
  int sx0=0, sx1=0, sx2=0, sx3=0;
  double ssource[4];
  double px, py, pz;
  double *contr_threads=NULL;
  int num_threads;

  psource[0] = gamma_permutation[idsource][ 0] / 6;
  psource[1] = gamma_permutation[idsource][ 6] / 6;
  psource[2] = gamma_permutation[idsource][12] / 6;
  psource[3] = gamma_permutation[idsource][18] / 6;
  isimag = gamma_permutation[idsource][ 0] % 2;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  ssource[0] =  gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]];
  ssource[1] =  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]];
  ssource[2] =  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]];
  ssource[3] =  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]];
/*
  fprintf(stdout, "__________________________________\n");
  fprintf(stdout, "isource=%d, idsink=%d, p[0] = %d\n", idsource, idsink, psource[0]);
  fprintf(stdout, "isource=%d, idsink=%d, p[1] = %d\n", idsource, idsink, psource[1]);
  fprintf(stdout, "isource=%d, idsink=%d, p[2] = %d\n", idsource, idsink, psource[2]);
  fprintf(stdout, "isource=%d, idsink=%d, p[3] = %d\n", idsource, idsink, psource[3]);
*/

/*  if(g_cart_id==0) fprintf(stdout, "# %3d %3d ssource = %e\t%e\t%e\t%e\n", idsource, idsink,
    ssource[0], ssource[1], ssource[2], ssource[3]); */

  if(g_source_type==0) { // point souce
    iix = g_source_location;
    sx0 = iix / (LX_global * LY_global * LZ_global);
    iix -= sx0 * LX_global * LY_global * LZ_global;
    sx1 = iix / (LY_global * LZ_global);
    iix -= sx1 * LY_global * LZ_global;
    sx2 = iix / (LZ_global);
    sx3 = iix - sx2 * LZ_global;
    //fprintf(stdout, "# [contract_twopoint_snk_momentum] source coordinates = (%d, %d, %d, %d)\n", sx0, sx1, sx2, sx3);
  }

  px = 2.*M_PI*(double)snk_mom[0]/(double)LX_global;
  py = 2.*M_PI*(double)snk_mom[1]/(double)LY_global;
  pz = 2.*M_PI*(double)snk_mom[2]/(double)LZ_global;
  // fprintf(stdout, "# [contract_twopoint_snk_momentum] p = (%e, %e, %e)\n", px, py, pz);

#pragma omp parallel shared(num_threads)
{
  if(omp_get_thread_num() == 0) {
    num_threads = omp_get_num_threads();
  }
}

  contr_threads = (double*)malloc(2 * num_threads * sizeof(double));

  tt = 0;
  for(x0=tmin; x0<=tmax; x0++) {

    memset(contr_threads, 0, 2*num_threads * sizeof(double));

#pragma omp parallel private(iix) shared(T,LX,LY,LZ, x0, sx0,sx1,sx2,sx3, px, py, pz, isimag, ssource, psource, contr, chi, phi, n_c, snk_mom, num_threads, tmin, tmax, contr_threads)
{
    int x1, x2, x3, ix, mu, c, j;
    double phase, cphase, sphase;
    double tmp[2], re, im;
    complex w;
    double spinor1[24], spinor2[24];
    unsigned int VOL3 = LX*LY*LZ;
    unsigned int LLYZ = LY * LZ;
    int nts = tmax - tmin + 1;

    int threadid = omp_get_thread_num();

    // TEST
    // fprintf(stdout, "# [contract_twopoint_snk_momentum] thread%.4d number of threads %d\n", threadid, num_threads);

    for(ix=threadid; ix<VOL3; ix+=num_threads) {


      x1 = ix / LLYZ;
      iix = ix - LLYZ * x1;
      x2 = iix / LZ;
      x3 = iix - LZ * x2;

      //phase = (double)(x1-sx1)*px + (double)(x2-sx2)*py + (double)(x3-sx3)*pz;
      phase = (double)(x1+g_proc_coords[1]*LX - sx1)*px + (double)(x2+g_proc_coords[2]*LY - sx2)*py + (double)(x3+g_proc_coords[3]*LZ - sx3)*pz;

      cphase = cos( phase );
      sphase = sin( phase );

      iix = x0 * VOL3 + ix;

      tmp[0] = 0.; tmp[1] = 0.;
      for(mu=0; mu<4; mu++) {
        for(c=0; c<n_c; c++) {

          _fv_eq_gamma_ti_fv(spinor1, idsink, phi[mu*n_c+c]+_GSI(iix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, chi[psource[mu]*n_c+c]+_GSI(iix), spinor2);

          tmp[0] += ssource[mu]*w.re;
          tmp[1] += ssource[mu]*w.im;

        }  // of color  index
      }    // of spinor index

      // multiply by momentum phase factor and add to correlator
      re = tmp[0]*cphase - tmp[1]*sphase;
      im = tmp[0]*sphase + tmp[1]*cphase;

      if( !isimag ) {
        contr_threads[2*threadid  ] += re;
        contr_threads[2*threadid+1] += im;
      } else {
        contr_threads[2*threadid  ] +=  im;
        contr_threads[2*threadid+1] += -re;
      }
    }  // of ix
}      // end of parallel region

    for(iix=0; iix<num_threads; iix++) {
      contr[2*tt  ] += contr_threads[2*iix  ];
      contr[2*tt+1] += contr_threads[2*iix+1];
    }
    tt++;
  }      // of x0

  free(contr_threads);
}  /* contract_twopoint_snk_momentum_trange */
#endif

/******************************************************************************
 * contract_twopoint_xdep
 *   contr - contraction field (2*VOLUME), out
 *   idsource - gamma id at source, in
 *   idsink - gamma id at sink, in
 *   chi - backward propagator, in
 *   phi - forward propagator, in
 *   n_c - number of colors, in
 *   stride - stride for contr, in
 *   factor - normalization of contractions, in
 *   prec - precision type, 64 for double precision, single precision else
 *
 ******************************************************************************/
void contract_twopoint_xdep(void*contr, const int idsource, const int idsink, void*chi, void*phi, int n_c, int stride, double factor, size_t prec) {

  const int psource[4] = { gamma_permutation[idsource][ 0] / 6,
                           gamma_permutation[idsource][ 6] / 6,
                           gamma_permutation[idsource][12] / 6,
                           gamma_permutation[idsource][18] / 6 };
  const int isimag = gamma_permutation[idsource][ 0] % 2;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  const double ssource[4] =  { gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]],
                  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]],
                  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]],
                  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]] };
/*
 * if( g_cart_id == 0 ) {
    fprintf(stdout, "__________________________________\n");
    fprintf(stdout, "isource=%d, idsink=%d, p[0] = %d, s[0] = %e\n", idsource, idsink, psource[0], ssource[0]);
    fprintf(stdout, "isource=%d, idsink=%d, p[1] = %d, s[1] = %e\n", idsource, idsink, psource[1], ssource[1]);
    fprintf(stdout, "isource=%d, idsink=%d, p[2] = %d, s[2] = %e\n", idsource, idsink, psource[2], ssource[2]);
    fprintf(stdout, "isource=%d, idsink=%d, p[3] = %d, s[3] = %e\n", idsource, idsink, psource[3], ssource[3]);
    fprintf(stdout, "isource=%d, idsink=%d, factor = %e\n", idsource, idsink, factor);

    fprintf(stdout, "# %3d %3d ssource = %e\t%e\t%e\t%e\n", idsource, idsink,
        ssource[0], ssource[1], ssource[2], ssource[3]);
  }
*/

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  int mu, c, j;
  unsigned int ix, iix;
  double  spinor1[24], spinor2[24];
  complex w;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix=0; ix<VOLUME; ix++) {
    iix = ix * stride;

    for(mu=0; mu<4; mu++) {
      for(c=0; c<n_c; c++) {

        if(prec==64) {
          _fv_eq_gamma_ti_fv(spinor1, idsink, (double*)(((double**)phi)[mu*n_c+c])+_GSI(ix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, (double*)(((double**)chi)[psource[mu]*n_c+c])+_GSI(ix), spinor2);

          if( !isimag ) {
            ((double*)contr)[2*iix  ] += factor * ssource[mu] * w.re;
            ((double*)contr)[2*iix+1] += factor * ssource[mu] * w.im;
          } else {
            ((double*)contr)[2*iix  ] +=  factor * ssource[mu] * w.im;
            ((double*)contr)[2*iix+1] += -factor * ssource[mu] * w.re;
          }
        } else {
          _fv_eq_gamma_ti_fv(spinor1, idsink, (float*)(((float**)phi)[mu*n_c+c])+_GSI(ix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, (float*)(((float**)chi)[psource[mu]*n_c+c])+_GSI(ix), spinor2);

          if( !isimag ) {
            ((float*)contr)[2*iix  ] += factor * ssource[mu] * w.re;
            ((float*)contr)[2*iix+1] += factor * ssource[mu] * w.im;
          } else {
            ((float*)contr)[2*iix  ] +=  factor * ssource[mu] * w.im;
            ((float*)contr)[2*iix+1] += -factor * ssource[mu] * w.re;
          }
        }
/*
        if(g_cart_id==0) 
          fprintf(stdout, "# source[%2d, %2d] = %25.16e +I %25.16e\n", mu, tt, 
            ssource[mu]*w.re, ssource[mu]*w.im);
        if(idsink==idsource && (idsink==1 || idsink==2 || idsink==3) && phi!=chi ) {
          fprintf(stdout, "%8d%3d%3d\t%e, %e\n",ix,mu,c, 
            ssource[mu] * w.re ,ssource[mu] * w.im  );
        }
*/
        
      }  /* of c */
    }  /* of mu */
  }  /* of ix */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
}  /* end of contract_twopoint_xdep */


/******************************************************************************
 * contract_twopoint_xdep_timeslice
 ******************************************************************************/
void contract_twopoint_xdep_timeslice(void*contr, const int idsource, const int idsink, void*chi, void*phi, int n_c, int stride, double factor, size_t prec) {

  int ix, iix, psource[4], isimag, mu, c, j;
  int VOL3 = LX*LY*LZ;
  double ssource[4], spinor1[24], spinor2[24];
  complex w;

  psource[0] = gamma_permutation[idsource][ 0] / 6;
  psource[1] = gamma_permutation[idsource][ 6] / 6;
  psource[2] = gamma_permutation[idsource][12] / 6;
  psource[3] = gamma_permutation[idsource][18] / 6;
  isimag = gamma_permutation[idsource][ 0] % 2;
  // sign from the source gamma matrix; the minus sign
  //   in the lower two lines is the action of gamma_5
  ssource[0] =  gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]];
  ssource[1] =  gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]];
  ssource[2] =  gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]];
  ssource[3] =  gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]];
/*
  fprintf(stdout, "__________________________________\n");
  fprintf(stdout, "isource=%d, idsink=%d, p[0] = %d\n", idsource, idsink, psource[0]);
  fprintf(stdout, "isource=%d, idsink=%d, p[1] = %d\n", idsource, idsink, psource[1]);
  fprintf(stdout, "isource=%d, idsink=%d, p[2] = %d\n", idsource, idsink, psource[2]);
  fprintf(stdout, "isource=%d, idsink=%d, p[3] = %d\n", idsource, idsink, psource[3]);
*/

/*  if(g_cart_id==0) fprintf(stdout, "# %3d %3d ssource = %e\t%e\t%e\t%e\n", idsource, idsink,
    ssource[0], ssource[1], ssource[2], ssource[3]); */

  //fprintf(stdout, "\n# [contract_twopoint_xdep] ix mu c re im\n");
  for(ix=0; ix<VOL3; ix++) {
    iix = ix * stride;

    for(mu=0; mu<4; mu++) {
      for(c=0; c<n_c; c++) {
/*
         if(g_cart_id==0 && (iix==0 || iix==111)) {
           fprintf(stdout, "iix=%4d, c=%d, mu=%d, idsource=%d, idsink=%d\n", iix, c, mu, idsource, idsink); 
           for(j=0; j<12; j++) 
             fprintf(stdout, "phi = %e +I %e\n", phi[mu*n_c+c][_GSI(iix)+2*j], phi[mu*n_c+c][_GSI(iix)+2*j+1]);
         }
*/
        if(prec==64) {
          _fv_eq_gamma_ti_fv(spinor1, idsink, (double*)(((double**)phi)[mu*n_c+c])+_GSI(ix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, (double*)(((double**)chi)[psource[mu]*n_c+c])+_GSI(ix), spinor2);
        } else {
          _fv_eq_gamma_ti_fv(spinor1, idsink, (float*)(((float**)phi)[mu*n_c+c])+_GSI(ix));
          _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
          _co_eq_fv_dag_ti_fv(&w, (float*)(((float**)chi)[psource[mu]*n_c+c])+_GSI(ix), spinor2);
        }

        if( !isimag ) {
          if(prec==64) {
            ((double*)contr)[2*iix  ] += factor * ssource[mu] * w.re;
            ((double*)contr)[2*iix+1] += factor * ssource[mu] * w.im;
          } else {
            ((float*)contr)[2*iix  ] += factor * ssource[mu] * w.re;
            ((float*)contr)[2*iix+1] += factor * ssource[mu] * w.im;
          }
        } else {
          if(prec==64) {
            ((double*)contr)[2*iix  ] +=  factor * ssource[mu] * w.im;
            ((double*)contr)[2*iix+1] += -factor * ssource[mu] * w.re;
          } else {
            ((float*)contr)[2*iix  ] +=  factor * ssource[mu] * w.im;
            ((float*)contr)[2*iix+1] += -factor * ssource[mu] * w.re;
          }
        }
        //if(g_cart_id==0) 
        //  fprintf(stdout, "# source[%2d, %2d] = %25.16e +I %25.16e\n", mu, tt, 
        //    ssource[mu]*w.re, ssource[mu]*w.im);
        //if(idsink==idsource && (idsink==1 || idsink==2 || idsink==3) && phi!=chi ) {
        //  fprintf(stdout, "%8d%3d%3d\t%e, %e\n",ix,mu,c, 
        //    ssource[mu] * w.re ,ssource[mu] * w.im  );
        //}
        
      }  // of c
    }  // of mu
  }  // of ix
}

/*******************************************
 * decompress the gauge field 
 *******************************************/
int decompress_gauge(double*gauge_aux, float*gauge_field_flt) {

  int it, ix, itmp, itmp2, mu, iix;
  int VOL3 = LX*LY*LZ;
  const unsigned int bytes = 4*sizeof(float);
  double a1[2], a2[2], a3[2], b1[2], c1[2], theta_a1, theta_c1;
  double U_[18], V_[18], dtmp, dtmp2;
  float gtmp[24];

  /*********************************************************
   * convert compressed float gauge field back to double
   * - first the spatial links in the whole volume
   *********************************************************/ 
  for(ix=0; ix<VOLUME; ix++) {
    /* collect the parts of the 3 spatial links */
    memcpy((void*)gtmp   , gauge_field_flt+4*ix, bytes);
    memcpy((void*)(gtmp+ 4), gauge_field_flt+4*ix+ 4*VOLUME, bytes);
    memcpy((void*)(gtmp+ 8), gauge_field_flt+4*ix+ 8*VOLUME, bytes);
    memcpy((void*)(gtmp+12), gauge_field_flt+4*ix+12*VOLUME, bytes);
    memcpy((void*)(gtmp+16), gauge_field_flt+4*ix+16*VOLUME, bytes);
    memcpy((void*)(gtmp+20), gauge_field_flt+4*ix+20*VOLUME, bytes);
    itmp2 = 72*ix;
    _cm_eq_id(gauge_aux+itmp2);
    for(mu=0; mu<3; mu++) {
      _cm_eq_zero(U_);
      _cm_eq_zero(V_);
      itmp = itmp2 + (mu+1)*18;
      iix  = mu*8;
      a2[0]    = (double)gtmp[iix  ];
      a2[1]    = (double)gtmp[iix+1];
      a3[0]    = (double)gtmp[iix+2];
      a3[1]    = (double)gtmp[iix+3];
      b1[0]    = (double)gtmp[iix+4];
      b1[1]    = (double)gtmp[iix+5];
      theta_a1 = (double)gtmp[iix+6];
      theta_c1 = (double)gtmp[iix+7];
      dtmp = a2[0]*a2[0] + a2[1]*a2[1] + a3[0]*a3[0] + a3[1]*a3[1];
      dtmp2 = sqrt(1. - dtmp);
      a1[0] = dtmp2 * cos(theta_a1);
      a1[1] = dtmp2 * sin(theta_a1);
      V_[12] =  sqrt(dtmp);
      dtmp2 = 1./V_[12];
      dtmp = sqrt(1. - a1[0]*a1[0] - a1[1]*a1[1] - b1[0]*b1[0] - b1[1]*b1[1] );
      c1[0] = dtmp * cos(theta_c1);
      c1[1] = dtmp * sin(theta_c1);
      U_[ 0] = 1.;
      U_[ 8] =  c1[0]*dtmp2;
      U_[ 9] = -c1[1]*dtmp2;
      U_[10] =  b1[0]*dtmp2;
      U_[11] =  b1[1]*dtmp2;
      U_[14] = -b1[0]*dtmp2;
      U_[15] =  b1[1]*dtmp2;
      U_[16] =  c1[0]*dtmp2;
      U_[17] =  c1[1]*dtmp2;
      V_[ 0] =  a1[0];
      V_[ 1] =  a1[1];
      V_[ 2] =  a2[0];
      V_[ 3] =  a2[1];
      V_[ 4] =  a3[0];
      V_[ 5] =  a3[1];
      V_[ 8] = -a3[0]*dtmp2;
      V_[ 9] =  a3[1]*dtmp2;
      V_[10] =  a2[0]*dtmp2;
      V_[11] = -a2[1]*dtmp2;
      V_[14] = -(a1[0]*a2[0] + a1[1]*a2[1])*dtmp2;
      V_[15] = -(a1[0]*a2[1] - a1[1]*a2[0])*dtmp2;
      V_[16] = -(a1[0]*a3[0] + a1[1]*a3[1])*dtmp2;
      V_[17] = -(a1[0]*a3[1] - a1[1]*a3[0])*dtmp2;
      _cm_eq_cm_ti_cm(gauge_aux+itmp, U_, V_);
    }
  }

  for(ix=0; ix<VOL3; ix++) {
    itmp = 24*VOLUME;
    memcpy((void*)gtmp, gauge_field_flt+itmp+4*ix, bytes);
    memcpy((void*)(gtmp+4), gauge_field_flt+itmp+4*ix+ 4*VOL3, bytes);
    itmp2 = 72*((T-1)*VOL3+ix);
    _cm_eq_zero(U_);
    _cm_eq_zero(V_);
    itmp = itmp2;
    a2[0]    = (double)gtmp[0];
    a2[1]    = (double)gtmp[1];
    a3[0]    = (double)gtmp[2];
    a3[1]    = (double)gtmp[3];
    b1[0]    = (double)gtmp[4];
    b1[1]    = (double)gtmp[5];
    theta_a1 = (double)gtmp[6];
    theta_c1 = (double)gtmp[7];
    dtmp = a2[0]*a2[0] + a2[1]*a2[1] + a3[0]*a3[0] + a3[1]*a3[1];
    dtmp2 = sqrt(1. - dtmp);
    a1[0] = dtmp2 * cos(theta_a1);
    a1[1] = dtmp2 * sin(theta_a1);
    V_[12] =  sqrt( dtmp );
    dtmp2 = 1. / V_[12];
    dtmp = sqrt(1. - a1[0]*a1[0] - a1[1]*a1[1] - b1[0]*b1[0] - b1[1]*b1[1] );
    c1[0] = dtmp * cos(theta_c1);
    c1[1] = dtmp * sin(theta_c1);
    U_[ 0] = 1.;
    U_[ 8] =  c1[0]*dtmp2;
    U_[ 9] = -c1[1]*dtmp2;
    U_[10] =  b1[0]*dtmp2;
    U_[11] =  b1[1]*dtmp2;
    U_[14] = -b1[0]*dtmp2;
    U_[15] =  b1[1]*dtmp2;
    U_[16] =  c1[0]*dtmp2;
    U_[17] =  c1[1]*dtmp2;
    V_[ 0] =  a1[0];
    V_[ 1] =  a1[1];
    V_[ 2] =  a2[0];
    V_[ 3] =  a2[1];
    V_[ 4] =  a3[0];
    V_[ 5] =  a3[1];
    V_[ 8] = -a3[0]*dtmp2;
    V_[ 9] =  a3[1]*dtmp2;
    V_[10] =  a2[0]*dtmp2;
    V_[11] = -a2[1]*dtmp2;
    V_[14] = -(a1[0]*a2[0] + a1[1]*a2[1])*dtmp2;
    V_[15] = -(a1[0]*a2[1] - a1[1]*a2[0])*dtmp2;
    V_[16] = -(a1[0]*a3[0] + a1[1]*a3[1])*dtmp2;
    V_[17] = -(a1[0]*a3[1] - a1[1]*a3[0])*dtmp2;
    _cm_eq_cm_ti_cm(gauge_aux+itmp, U_, V_);
  }

  return(0);
}

/**************************************************
 * compress the gauge field
 **************************************************/
int compress_gauge(float*gauge_field_flt, double *gauge_aux) {

  unsigned int it, ix, iix, itmp, itmp2;
  unsigned int VOL3 = LX*LY*LZ;

  /* links in spatial directions in timeslices 0,...,T-1 */
  for(ix=0; ix<VOLUME; ix++) {
    itmp2 = ix*72;

    itmp  = itmp2+18;
    iix = 4*ix;
    gauge_field_flt[iix  ] = (float)gauge_aux[itmp+ 2];
    gauge_field_flt[iix+1] = (float)gauge_aux[itmp+ 3];
    gauge_field_flt[iix+2] = (float)gauge_aux[itmp+ 4];
    gauge_field_flt[iix+3] = (float)gauge_aux[itmp+ 5];

    iix = 4*ix +  4*VOLUME;
    gauge_field_flt[iix  ] = (float)gauge_aux[itmp+ 6];
    gauge_field_flt[iix+1] = (float)gauge_aux[itmp+ 7];
    gauge_field_flt[iix+2] = atan2(gauge_aux[itmp+ 1], gauge_aux[itmp   ]);
    gauge_field_flt[iix+3] = atan2(gauge_aux[itmp+13], gauge_aux[itmp+12]);

    itmp  = itmp2+36;
    iix = 4*ix +  8*VOLUME;
    gauge_field_flt[iix  ] = (float)gauge_aux[itmp+ 2];
    gauge_field_flt[iix+1] = (float)gauge_aux[itmp+ 3];
    gauge_field_flt[iix+2] = (float)gauge_aux[itmp+ 4];
    gauge_field_flt[iix+3] = (float)gauge_aux[itmp+ 5];

    iix = 4*ix + 12*VOLUME;
    gauge_field_flt[iix  ] = (float)gauge_aux[itmp+ 6];
    gauge_field_flt[iix+1] = (float)gauge_aux[itmp+ 7];
    gauge_field_flt[iix+2] = atan2(gauge_aux[itmp+ 1], gauge_aux[itmp   ]);
    gauge_field_flt[iix+3] = atan2(gauge_aux[itmp+13], gauge_aux[itmp+12]);

    itmp  = itmp2+54;
    iix = 4*ix + 16*VOLUME;
    gauge_field_flt[iix  ] = (float)gauge_aux[itmp+ 2];
    gauge_field_flt[iix+1] = (float)gauge_aux[itmp+ 3];
    gauge_field_flt[iix+2] = (float)gauge_aux[itmp+ 4];
    gauge_field_flt[iix+3] = (float)gauge_aux[itmp+ 5];

    iix = 4*ix + 20*VOLUME;
    gauge_field_flt[iix  ] = (float)gauge_aux[itmp+ 6];
    gauge_field_flt[iix+1] = (float)gauge_aux[itmp+ 7];
    gauge_field_flt[iix+2] = atan2(gauge_aux[itmp+ 1], gauge_aux[itmp   ]);
    gauge_field_flt[iix+3] = atan2(gauge_aux[itmp+13], gauge_aux[itmp+12]);
  }

  /* links in time direction of the timeslice T-1 */
  for(ix=0; ix<VOL3; ix++) {
    itmp2 = ((T-1)*VOL3+ix)*72;
    itmp  = itmp2;
    iix = 24*VOLUME + 4*ix;
    gauge_field_flt[iix  ] = (float)gauge_aux[itmp+ 2];
    gauge_field_flt[iix+1] = (float)gauge_aux[itmp+ 3];
    gauge_field_flt[iix+2] = (float)gauge_aux[itmp+ 4];
    gauge_field_flt[iix+3] = (float)gauge_aux[itmp+ 5];

    iix = 24*VOLUME + 4*ix+4*VOL3;
    gauge_field_flt[iix  ] = (float)gauge_aux[itmp+ 6];
    gauge_field_flt[iix+1] = (float)gauge_aux[itmp+ 7];
    gauge_field_flt[iix+2] = atan2(gauge_aux[itmp+ 1], gauge_aux[itmp   ]);
    gauge_field_flt[iix+3] = atan2(gauge_aux[itmp+13], gauge_aux[itmp+12]);
  }
  return(0);
}

/***********************************************************************
 * set temporal gauge transform field
 ***********************************************************************/
int set_temporal_gauge(double*gauge_transform) {
  int ix, iix, count;
  int VOL3 = LX*LY*LZ;

  for(ix=0; ix<18*VOLUME; ix++) gauge_transform[ix] = 0.;
  for(ix=0; ix<18*VOL3; ix+=18) {
    gauge_transform[ix   ] = 1.;
    gauge_transform[ix+ 8] = 1.;
    gauge_transform[ix+16] = 1.;
  }
  count=0;
  for(ix=18*VOL3; ix<36*VOL3; ix+=18) {
    memcpy((void*)(gauge_transform+ix), (void*)(g_gauge_field+count), 18*sizeof(double));
    count += 72;
  }

  count = 72*VOL3;
  iix   = 18*VOL3;
  for(ix=36*VOL3; ix<18*VOLUME; ix+=18) {
    _cm_eq_cm_ti_cm(gauge_transform+ix, gauge_transform+iix, g_gauge_field+count);
    iix+=18;
    count+=72;
  }
  return(0);
}

/****************************************************************************************
 ****************************************************************************************
 **
 ** apply gauge transform
 **
 ****************************************************************************************
 ****************************************************************************************/
int apply_gauge_transform(double*gauge_new, double*gauge_transform, double*gauge_old){

  int ix, itmp;
  double U_[18], ratime, retime;
  ratime = (double)clock() / CLOCKS_PER_SEC;
  for(ix=0; ix<VOLUME; ix++) {
    itmp=72*ix;
    _cm_eq_cm_ti_cm(U_, gauge_transform+18*ix, gauge_old+itmp);
    _cm_eq_cm_ti_cm_dag(gauge_new+itmp,    U_, gauge_transform+g_iup[ix][0]*18);

    _cm_eq_cm_ti_cm(U_, gauge_transform+18*ix, gauge_old+itmp+18);
    _cm_eq_cm_ti_cm_dag(gauge_new+itmp+18, U_, gauge_transform+g_iup[ix][1]*18);

    _cm_eq_cm_ti_cm(U_, gauge_transform+18*ix, gauge_old+itmp+36);
    _cm_eq_cm_ti_cm_dag(gauge_new+itmp+36, U_, gauge_transform+g_iup[ix][2]*18);

    _cm_eq_cm_ti_cm(U_, gauge_transform+18*ix, gauge_old+itmp+54);
    _cm_eq_cm_ti_cm_dag(gauge_new+itmp+54, U_, gauge_transform+g_iup[ix][3]*18);
  }
  retime = (double)clock() / CLOCKS_PER_SEC;
  if(g_cart_id==0) fprintf(stdout, "# time for gauge transform: %e seconds\n", retime-ratime);
  return(0);
}

/******************************************************************
 * initialize rng; write first state to file
 * - method as in
 *   tmLQCD/start.c, function start_ranlux
 ******************************************************************/
int init_rng_stat_file (unsigned int seed, char*filename) {

  int c, i, j, iproc;
  int *rng_state=NULL;
#ifdef HAVE_MPI
  unsigned int *buffer = NULL;
#endif
  unsigned int step = g_proc_coords[0] * g_nproc_x * g_nproc_y * g_nproc_z 
                    + g_proc_coords[1] *             g_nproc_y * g_nproc_z 
                    + g_proc_coords[2] *                         g_nproc_z 
                    + g_proc_coords[3];
  unsigned int max_seed = 2147483647 / g_nproc;
  unsigned int l_seed = ( seed + step * max_seed ) % 2147483647;

  char *default_filename = "rng_stat.out";
  FILE*ofs=NULL;

  if(l_seed == 0) l_seed++; /* why zero not allowed? */

  fprintf(stdout, "# [init_rng_stat_file] proc%.4d l_seed = %u, max_seed = %u, step = %u\n", g_cart_id, l_seed, max_seed, step);

#ifdef HAVE_MPI
  /* check all seeds */
  buffer = (unsigned int*)malloc(g_nproc * sizeof(unsigned int));
  if(buffer == NULL) {
    fprintf(stderr, "[init_rng_stat_file] Error from malloc\n");
    return(1);
  }
  MPI_Gather(&l_seed, 1, MPI_UNSIGNED, buffer, 1, MPI_UNSIGNED, 0, g_cart_grid);  
  if(g_cart_id == 0) {
    for(i=0; i<g_nproc-1; i++) {
      for(j=i+1; j<g_nproc; j++) {
        if(buffer[i] == buffer[j]) {
          fprintf(stderr, "[init_rng_stat_file] Error, two seeds (%u, %u) are equal\n", buffer[i], buffer[j]);
          EXIT(1);
        }
      }
    }
  }  /* of if g_cart_id == 0 */
  free(buffer);
#endif

  fprintf(stdout, "# [init_rng_stat_file] proc%.4d ranldxd: using seed %u and level 2\n", g_cart_id, l_seed);
  rlxd_init(2, l_seed);

  c = rlxd_size();
  if( (rng_state = (int*)malloc(c*sizeof(int))) == (int*)NULL ) {
    fprintf(stderr, "Error, could not save the random number generator state\n");
    return(102);
  }
  rlxd_get(rng_state);

  for(iproc=0; iproc<g_nproc; iproc++) {
    if(filename == NULL) { filename = default_filename; }
    if(iproc == g_cart_id) {
      ofs = iproc == 0 ? fopen(filename, "w") : fopen(filename, "a");
      if( ofs == (FILE*)NULL ) {
        fprintf(stderr, "[init_rng_stat_file] Error, could not save the random number generator state\n");
        EXIT(103);
      }
      /* fprintf(stdout, "# [init_rng_stat_file] writing rng state to file %s\n", filename); */
      fprintf(ofs, "# proc %3d %3d %3d %3d\n", g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);
      for(i=0; i<c; i++) fprintf(ofs, "%d\n", rng_state[i]);
      fclose(ofs);
    }
#ifdef HAVE_MPI
    MPI_Barrier(g_cart_grid);
#endif
  }

  free(rng_state);
  return(0);
}  /* end of init_rng_stat_file */

int sync_rng_state(int id, int reset) {
#ifdef HAVE_MPI
  int c;
  int *rng_state=NULL;
  if(g_cart_id==0) fprintf(stdout, "# [sync_rng_state] synchronize rng states with current state at process %d\n", id);
  c = rlxd_size();
  rng_state = (int*)malloc(c*sizeof(int));
  if(g_cart_id == id) { rlxd_get(rng_state); }
  MPI_Barrier(g_cart_grid);
  MPI_Bcast(rng_state, c, MPI_INT, id, g_cart_grid);
  rlxd_reset(rng_state);
  free(rng_state);
#endif
  if(reset) { 
    if(g_cart_id==0) fprintf(stdout, "# [sync_rng_state] set global rng state to current state at process %d\n", id);
    rlxd_get(g_rng_state);
  }
  return(0);
}

/******************************************************************
 * initialize rng; write first state to array
 ******************************************************************/
int init_rng_state (int seed, int **rng_state) {

  int c;
  if(g_cart_id==0) fprintf(stdout, "# [init_rng_state] ranldxd: using seed %d and level 2\n", seed);
  rlxd_init(2, seed);

  if(*rng_state != NULL) {
    fprintf(stderr, "[] Error, rng_state not NULL\n");
    return(103);
  }
  c = rlxd_size();
  if( (*rng_state = (int*)malloc(c*sizeof(int))) == (int*)NULL ) {
    fprintf(stderr, "[init_rng_state] Error, could not save the random number generator state\n");
    return(102);
  }
  rlxd_get(*rng_state);
#ifdef HAVE_MPI
  MPI_Bcast(*rng_state, c, MPI_INT, 0, g_cart_grid);
  rlxd_reset(*rng_state);
#endif
  return(0);
}

int fini_rng_state (int **rng_state) {
  if(*rng_state != NULL) {
    free(*rng_state);
    *rng_state = NULL;
  }
  return(0);
}


// create, free spin propagator field
spinor_propagator_type *create_sp_field(size_t N) {
  size_t i, j;
  unsigned int count;
  spinor_propagator_type *fp;
  fp = (spinor_propagator_type *) calloc(N, sizeof(spinor_propagator_type));
  if( fp == NULL) return(NULL);

  fp[0] = (spinor_propagator_type)calloc(N * g_sv_dim, sizeof(double*));
  if ( fp[0] == NULL) {
    free( fp );
    return(NULL);
  }
  for(i=1;i<N;i++) fp[i] = fp[i-1] + g_sv_dim;
 
  fp[0][0] = (double*)calloc(N * g_sv_dim * g_sv_dim * 2, sizeof(double));
  if ( fp[0][0] == NULL) {
    free( fp[0] );
    free( fp );
    return(NULL);
  }
  count=0;
  for(j=0;j<N;j++) {
    for(i=0;i<g_sv_dim; i++) {
      if(count>0) {
        fp[j][i] = fp[0][0] + count * g_sv_dim * 2;
      }
      count++;
    }
  }

  return( fp );
}

void free_sp_field(spinor_propagator_type **fp) {
  int i;
  if( *fp != NULL ) {
    if( (*fp)[0] != NULL) {
      if(fp[0][0] != NULL ) {
        free( (*fp)[0][0] );
        (*fp)[0][0] = NULL;
      }
      free( (*fp)[0] );
      (*fp)[0] = NULL;
    }
    free( *fp );
    *fp = NULL;
  }
  return;
}

// create, free fermion propagator field
fermion_propagator_type *create_fp_field(size_t N) {
  size_t i, j;
  unsigned int count;
  fermion_propagator_type *fp;
  fp = (fermion_propagator_type *) calloc(N, sizeof(fermion_propagator_type));
  if( fp == NULL) return(NULL);

  fp[0] = (fermion_propagator_type)calloc(N * g_fv_dim, sizeof(double*));
  if ( fp[0] == NULL) {
    free( fp );
    return(NULL);
  }
  for(i=1;i<N;i++) fp[i] = fp[i-1] + g_fv_dim;
 
  fp[0][0] = (double*)calloc(N * g_fv_dim * g_fv_dim * 2, sizeof(double));
  if ( fp[0][0] == NULL) {
    free( fp[0] );
    free( fp );
    return(NULL);
  }
  count=0;
  for(j=0;j<N;j++) {
    for(i=0;i<g_fv_dim; i++) {
      if(count>0) {
        fp[j][i] = fp[0][0] + count * g_fv_dim * 2;
      }
      count++;
    }
  }

  return( fp );
}

void free_fp_field(fermion_propagator_type **fp) {
  int i;
  if( *fp != NULL ) {
    if( (*fp)[0] != NULL) {
      if(fp[0][0] != NULL ) {
        free( (*fp)[0][0] );
        (*fp)[0][0] = NULL;
      }
      free( (*fp)[0] );
      (*fp)[0] = NULL;
    }
    free( *fp );
    *fp = NULL;
  }
  return;
}

int unit_gauge_field(double*g, unsigned int N) {
  unsigned int i, mu;
  for(i=0;i<N;i++) {
    for(mu=0;mu<4;mu++) {
      _cm_eq_id( g+_GGI(i , mu) );
    }
  }
  return(0);
}

/*****************************************************
 * write fp_prop in R-format
 *****************************************************/
void printf_fp(fermion_propagator_type f, char*name, FILE*ofs) {
  int i,j;
  FILE *my_ofs = ofs==NULL ? stdout : ofs;
  fprintf(my_ofs, "# [] fermion propagator point:\n");
  fprintf(my_ofs, "\t%s <- array(0., dim=c(%d, %d))\n", name, g_fv_dim, g_fv_dim);
  for(i=0;i<g_fv_dim;i++) {
  for(j=0;j<g_fv_dim;j++) {
    //fprintf(my_ofs, "\t%s[%2d,%2d] <- %25.16e + %25.16e*1.i\n", name, i+1,j+1,f[i][2*j], f[i][2*j+1]);
    fprintf(my_ofs, "\t%s[%2d,%2d] <- %25.16e + %25.16e*1.i\n", name, i+1,j+1,f[j][2*i], f[j][2*i+1]);
  }}
}

/*****************************************************
 * write sp_prop in R-format
 *****************************************************/
void printf_sp(spinor_propagator_type f, char*name, FILE*ofs) {
  int i,j;
  FILE *my_ofs = ofs==NULL ? stdout : ofs;
  if(name[0] =='#') {
    fprintf(my_ofs, "%s\n", name);
    for(i=0;i<g_sv_dim;i++) {
    for(j=0;j<g_sv_dim;j++) {
      fprintf(my_ofs, "\t%3d%3d%25.16e%25.16e\n", i, j, f[j][2*i], f[j][2*i+1]);
    }}
  } else {
    fprintf(my_ofs, "# [printf_sp] spinor propagator point:\n");
    fprintf(my_ofs, "%s <- array(0, dim=c(%d, %d))\n", name, g_sv_dim, g_sv_dim);
    for(i=0;i<g_sv_dim;i++) {
    for(j=0;j<g_sv_dim;j++) {
      // fprintf(my_ofs, "\t%s[%2d,%2d] <- %25.16e + %25.16e*1.i\n", name, i+1,j+1,f[i][2*j], f[i][2*j+1]);
      fprintf(my_ofs, "\t%s[%2d,%2d] <- %25.16e + %25.16e*1.i\n", name, i+1,j+1,f[j][2*i], f[j][2*i+1]);
    }}
  }
}

/*****************************************************
 * norm of sp_prop
 * - defined as tr(f * f^dagger)
 *****************************************************/
void norm2_sp(spinor_propagator_type f, double*res) {
  int i,j;
  double re, im, tres=0.;

  for(j=0;j<g_sv_dim;j++) {
  for(i=0;i<g_sv_dim;i++) {
    re = f[j][2*i  ]; 
    im = f[j][2*i+1];
    tres += re*re + im*im;
  }}
  *res = tres;
}

/*****************************************************
 * write more general contractions to file
 *****************************************************/
int write_contraction2 (double *s, char *filename, int Nmu, unsigned int items, int write_ascii, int append) {

#ifdef HAVE_MPI
  fprintf(stderr, "[write_contractin2] no MPI version\n");
  return(1);
#else
  int x0, x1, x2, x3, mu, i;
  unsigned int ix;
  unsigned long int count=0;
  int ti[2], lvol;
  FILE *ofs;

  if(g_cart_id==0) {
    if(append==1) ofs = fopen(filename, "a");
    else ofs = fopen(filename, "w");
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "[write_contraction2] could not open file %s for writing\n", filename);
      exit(104);
    }
    if(write_ascii == 1) { /* write in ASCII format */
      /* write own part */
      fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n",
        Nconf, T_global, LX, LY, LZ, g_kappa, g_mu);
      for(ix=0; ix<items; ix++) {
        fprintf(ofs, "# ix=%6d\n", ix);
	for(mu=0; mu<Nmu; mu++) {
	  fprintf(ofs, "%3d%25.16e%25.16e\n", mu, s[2*(Nmu*ix+mu)], s[2*(Nmu*ix+mu)+1]);
	}
      }
    } else if(write_ascii == 2) { /* inner loop ix */
      fprintf(ofs, "# %6d%3d%3d%3d%3d%12.7f%12.7f\n",
        Nconf, T_global, LX, LY, LZ, g_kappa, g_mu);
      for(ix=0; ix<items; ix++) {
        fprintf(ofs, "# ix=%6d\n", ix);
	for(mu=0; mu<Nmu; mu++) {
	  fprintf(ofs, "%3d%25.16e%25.16e\n", mu, s[_GWI(mu,ix,items)], s[_GWI(mu,ix,items)+1]);
	}
      }
    } else if(write_ascii == 0) {
      for(ix=0; ix<items; ix++) {
        for(mu=0; mu<Nmu; mu++) {
          fwrite(s+_GWI(mu,ix,items), sizeof(double), 2, ofs);
        }
      }
    }
    fclose(ofs);
  } /* end of if g_cart_id == 0 */
  return(0);
#endif
}

void check_F_SU3(float*g, float*res) {
  float u[18], v[18];
  _cm_eq_cm_ti_cm_dag(u, g, g);
  u[ 0] -= 1.;
  u[ 8] -= 1.;
  u[16] -= 1.;

  *res =
    + u[ 0]*u[ 0] + u[ 1]*u[ 1] +u[ 2]*u[ 2] +u[ 3]*u[ 3] +u[ 4]*u[ 4] +u[ 5]*u[ 5]
    + u[ 6]*u[ 6] + u[ 7]*u[ 7] +u[ 8]*u[ 8] +u[ 9]*u[ 9] +u[10]*u[10] +u[11]*u[11]
    + u[12]*u[12] + u[13]*u[13] +u[14]*u[14] +u[15]*u[15] +u[16]*u[16] +u[17]*u[17];
  return;
}

void check_error(int status, char*myname, int*success, int exitstatus) {
  int suc = success != NULL ? *success : 0;
  char msg[400];
  if(status != suc) {
    if(myname!=NULL) {
      sprintf(msg, "[check_error] Error from %s; status was %d\n", myname, status);
    } else {
      sprintf(msg, "[check_error] Error; status was %d", status);
    }
    if(g_cart_id==0) fprintf(stderr, "%s", msg);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, exitstatus);
    MPI_Finalize();
#endif
    exit(exitstatus);
  }
  return;
}


unsigned int lexic2eot_5d (unsigned int is, unsigned int ix) {
  int eoflag;
  unsigned int shift;
  eoflag = ( is + g_iseven[ix]+1) % 2 == 0;
  shift  = (unsigned int)(is*VOLUME/2 + ( eoflag ? 0 : L5*VOLUME/2 ) );
  return( shift + ( g_iseven[ix] ? g_lexic2eot[ix] : (g_lexic2eot[ix] - VOLUME/2) ) );
}


int shift_spinor_field (double *s, double *r, int *d) {
#ifndef HAVE_MPI
  int x0, x1, x2, x3, y0, y1, y2, y3;
  unsigned int ix, iy;
  for(x0=0; x0<T; x0++) {
    y0 = (x0 + d[0] + T) % T ;
  for(x1=0; x1<LX; x1++) {
    y1 = (x1 + d[1] + LX) % LX;
  for(x2=0; x2<LY; x2++) {
    y2 = (x2 + d[2] + LY) % LY;
  for(x3=0; x3<LZ; x3++) {
    y3 = (x3 + d[3] + LZ) % LZ;
    ix = g_ipt[x0][x1][x2][x3];
    iy = g_ipt[y0][y1][y2][y3];
    _fv_eq_fv(s+_GSI(iy), r+_GSI(ix));
  }}}}
#else
  fprintf(stderr, "[shift_spinor_field] Error, MPI version not implemented\n");
  EXIT(1);
#endif
  return(0);
}  // end of shift_spinor_field

/***********************************************************************
 * check_source ()
 *
 * - for point sources: apply the Dirac oprator once, subtract 1.0 at
 *   the source location and calculate the norm of the spinor field
 *   (cf. apply_Dtm.c)
 * - assumes that sf has been exchanged beforehand
 ***********************************************************************/
void check_source(double *sf, double*work, double mass, unsigned int glocation, int sc) {

  int src0, src1, src2, src3;
  int lsrc0, lsrc1, lsrc2, lsrc3;
  int src_proc_coords[4];
  int src_proc_id=0;
  unsigned int llocation;

  double norm1, norm2;

  // global source coordinates
  src0 = glocation / (LX*g_nproc_x*LY*g_nproc_y*LZ*g_nproc_z);
  src2 = glocation - (LX*g_nproc_x*LY*g_nproc_y*LZ*g_nproc_z) * src0;
  src1 = src2 / (LY*g_nproc_y*LZ*g_nproc_z);
  src3 = src2 - (LY*g_nproc_y*LZ*g_nproc_z) * src1;
  src2 = src3 / (LZ*g_nproc_z);
  src3 -= (LZ*g_nproc_z)*src2;
  if(g_cart_id==0) fprintf(stdout, "# [check_source] global source location %d = (%d, %d, %d, %d)\n",
      glocation, src0, src1, src2, src3);

  // local source coordinates
  lsrc0 = src0 % T;
  lsrc1 = src1 % LX;
  lsrc2 = src2 % LY;
  lsrc3 = src3 % LZ;
  llocation = g_ipt[lsrc0][lsrc1][lsrc2][lsrc3];
  if(g_cart_id==0) fprintf(stdout, "# [check_source] local source location (%d, %d, %d, %d) = %u\n", lsrc0, lsrc1, lsrc2, lsrc3, llocation);

  // coordinates of process with source location
  src_proc_coords[0] = src0 / T;
  src_proc_coords[1] = src1 / LX;
  src_proc_coords[2] = src2 / LY;
  src_proc_coords[3] = src3 / LZ;

#ifdef HAVE_MPI
  MPI_Cart_rank(g_cart_grid, src_proc_coords, &src_proc_id);
#endif
  if(g_cart_id==0) fprintf(stdout, "# [check_source] source location process coordintates (%d, %d, %d, %d) = %d\n",
      src_proc_coords[0], src_proc_coords[1], src_proc_coords[2], src_proc_coords[3], src_proc_id);

  xchange_field(sf);

  Q_phi(work, sf, g_gauge_field, mass);
  if(src_proc_id == g_cart_id) {
    work[_GSI(llocation)+2*sc] -= 1.0;
  }

  spinor_scalar_product_re(&norm1, sf, sf, VOLUME);
  spinor_scalar_product_re(&norm2, work, work, VOLUME);

  if(g_cart_id==0) {
    fprintf(stdout, "# [check_source] norm of solution    = %e\n", norm1);
    fprintf(stdout, "# [check_source] norm of A x - delta = %e\n", norm2);
  }
}


/***********************************************************
 * Method based on Givens' rotations, as used by Urs Wenger
 * - copied from tmLQCD
 * - difference to cm_proj_iterate at least unitarization step
 ***********************************************************/
void reunitarize_Givens_rotations(double *omega) {
  int iter;
  double w[18], rot[18], tmp[18];
  double trace_old, trace_new;
  complex s0, s1;
  double scale;
  const int maxIter = 200;

  _cm_eq_id(w);
  trace_old = omega[0] + omega[8] + omega[16];

  for (iter = 0; iter < maxIter; ++iter)
  {
      /* TEST */
      /* fprintf(stdout, "# [reunitarize_Givens_rotations] (%6d) first omega\n", iter);
      _cm_fprintf(omega, stdout);
      */

    /* Givens' rotation 01 */
    /* s0 = omega_00 + conj( omega_11 ) */
      s0.re = omega[0] + omega[8];
      s0.im = omega[1] - omega[9];

    /* s1 = omega_01 - conj( omega_10 ) */
      s1.re = omega[2] - omega[6];
      s1.im = omega[3] + omega[7];

      scale = 1.0 / sqrt( s0.re * s0.re + s0.im * s0.im + s1.re * s1.re + s1.im * s1.im ); 
      s0.re *= scale;
      s0.im *= scale;
      s1.re *= scale;
      s1.im *= scale;
      
      /* Projecting */
      _cm_eq_id(rot);
      rot[ 0] =  s0.re;
      rot[ 1] =  s0.im;
      rot[ 8] =  s0.re;
      rot[ 9] = -s0.im;
      rot[ 2] =  s1.re;
      rot[ 3] =  s1.im;
      rot[ 6] = -s1.re;
      rot[ 7] =  s1.im;

      /* TEST */
      /* fprintf(stdout, "# [reunitarize_Givens_rotations] (%6d) first rotation\n", iter);
      _cm_fprintf(rot, stdout);
      */

      _cm_eq_cm_ti_cm(tmp, rot, w);
      _cm_eq_cm(w, tmp);
      _cm_eq_cm_ti_cm_dag(tmp, omega, rot);
      _cm_eq_cm(omega, tmp);

      /* TEST */
      /* fprintf(stdout, "# [reunitarize_Givens_rotations] (%6d) second omega\n", iter);
      _cm_fprintf(omega, stdout);
      */

    /* Givens' rotation 12 */
      s0.re = omega[ 8] + omega[16];
      s0.im = omega[ 9] - omega[17];
      s1.re = omega[10] - omega[14];
      s1.im = omega[11] + omega[15];
      scale = 1.0 / sqrt( s0.re * s0.re + s0.im * s0.im + s1.re * s1.re + s1.im * s1.im );

      s0.re *= scale;
      s0.im *= scale;
      s1.re *= scale;
      s1.im *= scale;

      /* Projecting */
      _cm_eq_id(rot);
      rot[ 8] =  s0.re;
      rot[ 9] =  s0.im;
      rot[16] =  s0.re;
      rot[17] = -s0.im;
      rot[10] =  s1.re;
      rot[11] =  s1.im;
      rot[14] = -s1.re;
      rot[15] =  s1.im;

      /* TEST */
      /* fprintf(stdout, "# [reunitarize_Givens_rotations] (%6d) second rotation\n", iter);
      _cm_fprintf(rot, stdout);
      */
      
      _cm_eq_cm_ti_cm(tmp, rot, w);
      _cm_eq_cm(w, tmp);
      _cm_eq_cm_ti_cm_dag(tmp, omega, rot);
      _cm_eq_cm(omega, tmp);

      /* TEST */
      /* fprintf(stdout, "# [reunitarize_Givens_rotations] (%6d) third omega\n", iter);
      _cm_fprintf(omega, stdout);
      */

    /* Givens' rotation 20 */
      s0.re = omega[16] + omega[ 0];
      s0.im = omega[17] - omega[ 1];
      s1.re = omega[12] - omega[ 4];
      s1.im = omega[13] + omega[ 5];

      scale = 1.0 / sqrt( s0.re * s0.re + s0.im * s0.im + s1.re * s1.re + s1.im * s1.im );

      s0.re *= scale;
      s0.im *= scale;
      s1.re *= scale;
      s1.im *= scale;

      /* Projecting */
      _cm_eq_id(rot);
      rot[16] =  s0.re;
      rot[17] =  s0.im;
      rot[ 0] =  s0.re;
      rot[ 1] = -s0.im;
      rot[12] =  s1.re;
      rot[13] =  s1.im;
      rot[ 4] = -s1.re;
      rot[ 5] =  s1.im;

      /* TEST */
      /* fprintf(stdout, "# [reunitarize_Givens_rotations] (%6d) third rotation\n", iter);
      _cm_fprintf(rot, stdout);
      */

      _cm_eq_cm_ti_cm(tmp, rot, w);
      _cm_eq_cm(w, tmp);
      _cm_eq_cm_ti_cm_dag(tmp, omega, rot);
      _cm_eq_cm(omega, tmp);

    trace_new = omega[0] + omega[8] + omega[16];
    /* fprintf(stdout, "# [reunitarize_Givens_rotations] trace_new[%d] = %25.16e\n", iter, trace_new); */

    if (trace_new - trace_old < 1e-15)
    /* if (fabs( trace_new - trace_old ) < 1e-15) */
      break;
    trace_old = trace_new;
  }
  _cm_eq_cm(omega, w);
}  /* end of reunitarize_Givens_rotations */



/*****************************************************
 * exchange a contractinos
 *   phi point to contraction field
 *   N number of real elements per site
 *   ONLY EXCHANGE BOUNDARY FACES
 *****************************************************/
void xchange_contraction(double *phi, int N) {
#ifdef HAVE_MPI
  int cntr=0;

  MPI_Request request[120];
  MPI_Status status[120];

  MPI_Isend(&phi[0],                 1, contraction_time_slice_cont, g_nb_t_dn, 83, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*VOLUME],         1, contraction_time_slice_cont, g_nb_t_up, 83, g_cart_grid, &request[cntr]);
  cntr++;
  
  MPI_Isend(&phi[N*(T-1)*LX*LY*LZ], 1, contraction_time_slice_cont, g_nb_t_up, 84, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(T+1)*LX*LY*LZ], 1, contraction_time_slice_cont, g_nb_t_dn, 84, g_cart_grid, &request[cntr]);
  cntr++;

#if (defined PARALLELTX) || (defined PARALLELTXY)  || (defined PARALLELTXYZ) 
  /* x - boundary faces */
  MPI_Isend(&phi[0],                              1, contraction_x_slice_vector, g_nb_x_dn, 85, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*LX*LY*LZ)],         1, contraction_x_slice_cont,   g_nb_x_up, 85, g_cart_grid, &request[cntr]);
  cntr++;
  
  MPI_Isend(&phi[N*(LX-1)*LY*LZ],                1, contraction_x_slice_vector, g_nb_x_up, 86, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, contraction_x_slice_cont,   g_nb_x_dn, 86, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXY || (defined PARALLELTXYZ) 
  /* y - boundary faces */
  MPI_Isend(&phi[0],                                        1, contraction_y_slice_vector, g_nb_y_dn, 87, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*(LX*LY*LZ+T*LY*LZ))],         1, contraction_y_slice_cont,   g_nb_y_up, 87, g_cart_grid, &request[cntr]);
  cntr++;
  
  MPI_Isend(&phi[N*(LY-1)*LZ],                             1, contraction_y_slice_vector, g_nb_y_up, 88, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*(LX*LY*LZ+T*LY*LZ)+T*LX*LZ)], 1, contraction_y_slice_cont,   g_nb_y_dn, 88, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if (defined PARALLELTXYZ) 
  /* z - boundary faces */
  MPI_Isend(&phi[0],                                                1, contraction_z_slice_vector, g_nb_z_dn, 89, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*(LX*LY*LZ+T*LY*LZ+T*LX*LZ))],         1, contraction_z_slice_cont,   g_nb_z_up, 89, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[N*(LZ-1)],                                        1, contraction_z_slice_vector, g_nb_z_up, 90, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[N*(VOLUME+2*(LX*LY*LZ+T*LY*LZ+T*LX*LZ)+T*LX*LY)], 1, contraction_z_slice_cont,   g_nb_z_dn, 90, g_cart_grid, &request[cntr]);
  cntr++;
#endif

  MPI_Waitall(cntr, request, status);
#endif  /* of ifdef HAVE_MPI */
}  /* xchange_contraction */


/***********************************************************
 * spinor_field_lexic2eo
 * - decompose lexicorgraphic spinor field in even and odd part
 ***********************************************************/
void spinor_field_lexic2eo (double *r_lexic, double*r_e, double *r_o) {

  unsigned int ix, ix_e2lexic, ix_o2lexic;
  unsigned int N = (VOLUME+RAND)/2;

  if(r_e != NULL) {
    /* for(ix=0; ix<N; ix++) */
    for(ix=0; ix < VOLUME/2; ix++)
    {
      ix_e2lexic = g_eo2lexic[ix];
      /* fprintf(stdout, "# [spinor_field_lexic2eo] ix = %u, ix_e2lexic = %u\n", ix, ix_e2lexic); */
      _fv_eq_fv(r_e + _GSI(ix), r_lexic+_GSI(ix_e2lexic) );
    }  /* end of loop on ix over (VOLUME+RAND)/2 */
  }

  if(r_o != NULL) {
    /* for(ix=0; ix<N; ix++) */
    for(ix=0; ix<VOLUME/2; ix++)
    {
      ix_o2lexic = g_eo2lexic[ix+N];
      /* fprintf(stdout, "# [spinor_field_lexic2eo] ix = %u, ix_o2lexic = %u\n", ix, ix_o2lexic); */
      _fv_eq_fv(r_o + _GSI(ix), r_lexic+_GSI(ix_o2lexic) );
    }  /* end of loop on ix over (VOLUME+RAND)/2 */
  }
}  /* end of spinor_field_lexic2eo */

/***********************************************************
 * spinor_field_eo2lexic
 * - compose lexicographic spinor field from even and odd part
 ***********************************************************/
void spinor_field_eo2lexic (double *r_lexic, double*r_e, double *r_o) {

  unsigned int ix, ix_e2lexic, ix_o2lexic;
  unsigned int N = (VOLUME+RAND)/2;

  if(r_e != NULL) {
    /* for(ix=0; ix<N; ix++) */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix_e2lexic)
#endif
    for(ix=0; ix<VOLUME/2; ix++)
    {
      ix_e2lexic = g_eo2lexic[ix];
      _fv_eq_fv(  r_lexic+_GSI(ix_e2lexic) , r_e + _GSI(ix) );
    }  /* end of loop on ix */
  }

  if(r_o != NULL) {
    /* for(ix=0; ix<N; ix++) */
#ifdef HAVE_OPENMP
#pragma omp parallel for private(ix_o2lexic)
#endif
    for(ix=0; ix<VOLUME/2; ix++)
    {
      ix_o2lexic = g_eo2lexic[ix+N];
      _fv_eq_fv(  r_lexic+_GSI(ix_o2lexic) , r_o + _GSI(ix) );
    }  /* end of loop on ix */
  }
}  /* end of spinor_field_eo2lexic */





/*****************************************************
 * exchange an odd spinor field
 *   input: phi - even or odd spinor field
 *          eo  - flag for even (0) and odd (1)
 *****************************************************/
void xchange_eo_field(double *phi, int eo) {
#ifdef HAVE_MPI
  const unsigned int Vhalf = VOLUME / 2;
  const int LZh = LZ / 2;
  /* =================================================== */
  const unsigned int Tslice   =      LX * LY * LZ / 2;
  const unsigned int Xslice   =  T *      LY * LZ / 2;
  const unsigned int Yslice   =  T * LX *      LZ / 2;
  const unsigned int Zslice   =  T * LX * LY      / 2;
  /* =================================================== */
  const unsigned int TXslice  =           LY * LZ / 2;
  const unsigned int TXYslice =                LZ / 2;
  /* =================================================== */
  int cntr=0;
/*
  int i, error_string_length;
  char error_string[400];
*/

  const unsigned int Zshift_start = eo ? LZ / 2 : 0;
  const unsigned int Zshift_end   = eo ? 0 : LZ / 2;

  MPI_Request request[220];
  MPI_Status status[220];

  /* t - boundary faces */
  MPI_Isend(&phi[0],                                          1, eo_spinor_time_slice_cont, g_nb_t_dn, 183, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*Vhalf],                                   1, eo_spinor_time_slice_cont, g_nb_t_up, 183, g_cart_grid, &request[cntr]);
  cntr++;
 
  MPI_Isend(&phi[24*(T-1)*Tslice],                            1, eo_spinor_time_slice_cont, g_nb_t_up, 184, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(T+1)*Tslice],                            1, eo_spinor_time_slice_cont, g_nb_t_dn, 184, g_cart_grid, &request[cntr]);
  cntr++;

#if (defined PARALLELTX) || (defined PARALLELTXY)  || (defined PARALLELTXYZ) 
 
  /* x - boundary faces */
  MPI_Isend(&phi[0],                                          1, eo_spinor_x_slice_vector, g_nb_x_dn, 185, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(Vhalf+2*Tslice)],                        1, eo_spinor_x_slice_cont,   g_nb_x_up, 185, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[24*(Tslice-TXslice)],                        1, eo_spinor_x_slice_vector, g_nb_x_up, 186, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(Vhalf+2*Tslice+Xslice)],                 1, eo_spinor_x_slice_cont,   g_nb_x_dn, 186, g_cart_grid, &request[cntr]);
  cntr++;
#endif


#if defined PARALLELTXY || (defined PARALLELTXYZ) 

  /* y - boundary faces */
  MPI_Isend(&phi[0],                                          1, eo_spinor_y_slice_vector, g_nb_y_dn, 187, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice))],               1, eo_spinor_y_slice_cont,   g_nb_y_up, 187, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[24*( TXslice - TXYslice)],                   1, eo_spinor_y_slice_vector, g_nb_y_up, 188, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice)+Yslice)],        1, eo_spinor_y_slice_cont,   g_nb_y_dn, 188, g_cart_grid, &request[cntr]);
  cntr++;

#endif


#if (defined PARALLELTXYZ) 

  /* z - boundary faces */


#if 0
  /* 1st half z boundary, backward */
  MPI_Isend(&phi[24*Zshift_start],                            1, eo_spinor_z_slice_vector, g_nb_z_dn, 189, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice+Yslice))],        1, eo_spinor_z_slice_cont,   g_nb_z_up, 189, g_cart_grid, &request[cntr]);
  cntr++;

  /* 1st half z boundary, forward */
  MPI_Isend(&phi[24*Zshift_start],                            1, eo_spinor_z_slice_vector, g_nb_z_dn, 190, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice+Yslice))],        1, eo_spinor_z_slice_cont,   g_nb_z_up, 190, g_cart_grid, &request[cntr]);
  cntr++;

  /* 2nd half z boundary, forward */
  MPI_Isend(&phi[24*(LZh-1+Zshift_end)],                      1, eo_spinor_z_slice_vector, g_nb_z_up, 191, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice+Yslice)+Zslice)], 1, eo_spinor_z_slice_cont,   g_nb_z_dn, 191, g_cart_grid, &request[cntr]);
  cntr++;

  /* 2nd half z boundary, backward */
  MPI_Isend(&phi[24*(LZh-1+Zshift_end)],                      1, eo_spinor_z_slice_vector, g_nb_z_up, 192, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice+Yslice)+Zslice)], 1, eo_spinor_z_slice_cont,   g_nb_z_dn, 192, g_cart_grid, &request[cntr]);
  cntr++;
#endif


  if ( eo == 0 ) {
    /* even field */

    MPI_Isend(&phi[0],                                          1, eo_spinor_z_even_bwd_slice_struct, g_nb_z_dn, 189, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice+Yslice))],        1, eo_spinor_z_slice_cont,            g_nb_z_up, 189, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Isend(&phi[0],                                          1, eo_spinor_z_even_fwd_slice_struct, g_nb_z_up, 190, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice+Yslice)+Zslice)], 1, eo_spinor_z_slice_cont,            g_nb_z_dn, 190, g_cart_grid, &request[cntr]);
    cntr++;

  } else {
    /* odd field */

    MPI_Isend(&phi[0],                                          1, eo_spinor_z_odd_bwd_slice_struct,  g_nb_z_dn, 189, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice+Yslice))],        1, eo_spinor_z_slice_cont,            g_nb_z_up, 189, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Isend(&phi[0],                                          1, eo_spinor_z_odd_fwd_slice_struct,  g_nb_z_up, 190, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Irecv(&phi[24*(Vhalf+2*(Tslice+Xslice+Yslice)+Zslice)], 1, eo_spinor_z_slice_cont,            g_nb_z_dn, 190, g_cart_grid, &request[cntr]);
    cntr++;

  }
#if 0
#endif  /* of if 0 */

#endif  /* of if defined PARALLELTXYZ */

  /* fprintf(stdout, "# [xchange_eo_field] proc%.4d starting MPI_Waitall\n", g_cart_id); */

  MPI_Waitall(cntr, request, status);

/* TEST
  if(g_cart_id == 0) {
    for(i=0; i<cntr; i++) {
      MPI_Error_string(status[i].MPI_ERROR, error_string, &error_string_length);
      fprintf(stdout, "# [xchange_eo_field] %3d %s\n", i,  error_string);
    }
  }
*/

#endif
}  /* xchange_eo_field */


/*****************************************************
 * exchange an eo propagator field
 *   input: fp  - even or odd propagator field
 *          eo  - flag for even (0) and odd (1)
 *          dir - direction in which to exchange or -1 for
 *                all directions
 *****************************************************/
void xchange_eo_propagator ( fermion_propagator_type *fp, int eo, int dir) {
#ifdef HAVE_MPI
  const unsigned int Vhalf = VOLUME / 2;
  const int LZh = LZ / 2;
  /* =================================================== */
  const unsigned int Tslice   =      LX * LY * LZ / 2;
  const unsigned int Xslice   =  T *      LY * LZ / 2;
  const unsigned int Yslice   =  T * LX *      LZ / 2;
  const unsigned int Zslice   =  T * LX * LY      / 2;
  /* =================================================== */
  const unsigned int TXslice  =           LY * LZ / 2;
  const unsigned int TXYslice =                LZ / 2;
  /* =================================================== */
  int cntr=0;
/*
  int i, error_string_length;
  char error_string[400];
*/

  const unsigned int Zshift_start = eo ? LZ / 2 : 0;
  const unsigned int Zshift_end   = eo ? 0 : LZ / 2;

  double *phi = &(fp[0][0][0]);

  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [xchange_eo_propagator] exchanging eo propagator field %d in direction %d\n", eo, dir);
  }

  MPI_Request request[220];
  MPI_Status status[220];

  if( dir == -1 || dir == 0) {
    /* t - boundary faces */
    MPI_Isend(&phi[0],                                          1, eo_propagator_time_slice_cont, g_nb_t_dn, 183, g_cart_grid, &request[cntr]);
    cntr++;
    MPI_Irecv(&phi[288*Vhalf],                                   1, eo_propagator_time_slice_cont, g_nb_t_up, 183, g_cart_grid, &request[cntr]);
    cntr++;
 
    MPI_Isend(&phi[288*(T-1)*Tslice],                            1, eo_propagator_time_slice_cont, g_nb_t_up, 184, g_cart_grid, &request[cntr]);
    cntr++;
    MPI_Irecv(&phi[288*(T+1)*Tslice],                            1, eo_propagator_time_slice_cont, g_nb_t_dn, 184, g_cart_grid, &request[cntr]);
    cntr++;

  }
#if (defined PARALLELTX) || (defined PARALLELTXY)  || (defined PARALLELTXYZ) 
 
  if(dir == -1 || dir == 1) {
    /* x - boundary faces */
    MPI_Isend(&phi[0],                                          1, eo_propagator_x_slice_vector, g_nb_x_dn, 185, g_cart_grid, &request[cntr]);
    cntr++;
    MPI_Irecv(&phi[288*(Vhalf+2*Tslice)],                        1, eo_propagator_x_slice_cont,   g_nb_x_up, 185, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Isend(&phi[288*(Tslice-TXslice)],                        1, eo_propagator_x_slice_vector, g_nb_x_up, 186, g_cart_grid, &request[cntr]);
    cntr++;
    MPI_Irecv(&phi[288*(Vhalf+2*Tslice+Xslice)],                 1, eo_propagator_x_slice_cont,   g_nb_x_dn, 186, g_cart_grid, &request[cntr]);
    cntr++;
  }
#endif


#if defined PARALLELTXY || (defined PARALLELTXYZ) 

  if(dir == -1 || dir == 2) {
    /* y - boundary faces */
    MPI_Isend(&phi[0],                                          1, eo_propagator_y_slice_vector, g_nb_y_dn, 187, g_cart_grid, &request[cntr]);
    cntr++;
    MPI_Irecv(&phi[288*(Vhalf+2*(Tslice+Xslice))],               1, eo_propagator_y_slice_cont,   g_nb_y_up, 187, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Isend(&phi[288*( TXslice - TXYslice)],                   1, eo_propagator_y_slice_vector, g_nb_y_up, 188, g_cart_grid, &request[cntr]);
    cntr++;
    MPI_Irecv(&phi[288*(Vhalf+2*(Tslice+Xslice)+Yslice)],        1, eo_propagator_y_slice_cont,   g_nb_y_dn, 188, g_cart_grid, &request[cntr]);
    cntr++;
  }
#endif


#if (defined PARALLELTXYZ) 

  if( dir == -1 || dir == 3 ) {
    /* z - boundary faces */

    if ( eo == 0 ) {
      /* even field */
 
      MPI_Isend(&phi[0],                                          1, eo_propagator_z_even_bwd_slice_struct, g_nb_z_dn, 189, g_cart_grid, &request[cntr]);
      cntr++;

      MPI_Irecv(&phi[288*(Vhalf+2*(Tslice+Xslice+Yslice))],        1, eo_propagator_z_slice_cont,            g_nb_z_up, 189, g_cart_grid, &request[cntr]);
      cntr++;

      MPI_Isend(&phi[0],                                          1, eo_propagator_z_even_fwd_slice_struct, g_nb_z_up, 190, g_cart_grid, &request[cntr]);
      cntr++;

      MPI_Irecv(&phi[288*(Vhalf+2*(Tslice+Xslice+Yslice)+Zslice)], 1, eo_propagator_z_slice_cont,            g_nb_z_dn, 190, g_cart_grid, &request[cntr]);
      cntr++;

    } else {
      /* odd field */

      MPI_Isend(&phi[0],                                          1, eo_propagator_z_odd_bwd_slice_struct,  g_nb_z_dn, 189, g_cart_grid, &request[cntr]);
      cntr++;

      MPI_Irecv(&phi[288*(Vhalf+2*(Tslice+Xslice+Yslice))],        1, eo_propagator_z_slice_cont,            g_nb_z_up, 189, g_cart_grid, &request[cntr]);
      cntr++;

      MPI_Isend(&phi[0],                                          1, eo_propagator_z_odd_fwd_slice_struct,  g_nb_z_up, 190, g_cart_grid, &request[cntr]);
      cntr++;

      MPI_Irecv(&phi[288*(Vhalf+2*(Tslice+Xslice+Yslice)+Zslice)], 1, eo_propagator_z_slice_cont,            g_nb_z_dn, 190, g_cart_grid, &request[cntr]);
      cntr++;

    }
  }

#endif  /* of if defined PARALLELTXYZ */

  /* fprintf(stdout, "# [xchange_eo_field] proc%.4d starting MPI_Waitall\n", g_cart_id); */

  MPI_Waitall(cntr, request, status);

/* TEST
  if(g_cart_id == 0) {
    for(i=0; i<cntr; i++) {
      MPI_Error_string(status[i].MPI_ERROR, error_string, &error_string_length);
      fprintf(stdout, "# [xchange_eo_field] %3d %s\n", i,  error_string);
    }
  }
*/

#endif
}  /* xchange_eo_propagator */


/***********************************************************
 * unpack lexic to odd,odd
 * - decompose lexicorgraphic spinor field into 2 odd parts
 *   by shifting in 0-direction
 ***********************************************************/
void spinor_field_unpack_lexic2eo (double *r_lexic, double*r_o1, double *r_o2) {

  unsigned int ix, iy;
  unsigned int N     = (VOLUME+RAND) / 2;
  unsigned int Vhalf =  VOLUME       / 2;

  xchange_field(r_lexic);
  /* even part to shift direction 0 to odd r_o1 */
  for(iy=0; iy<Vhalf; iy++) {
    ix = g_idn[ g_eo2lexic[iy + N ] ][0];
    _fv_eq_fv(r_o1+_GSI(iy), r_lexic+_GSI(ix) );
  }

  /* odd part to odd r_o2 */
  for(iy=0; iy<Vhalf; iy++) {
    ix = g_eo2lexic[iy + N ];
    _fv_eq_fv(r_o2+_GSI(iy), r_lexic+_GSI(ix) );
  }
}  /* end of spinor_field_unpack_lexic2eo */


/***********************************************************
 * r = s + c * t
 ***********************************************************/
void spinor_field_eq_spinor_field_pl_spinor_field_ti_re(double*r, double*s, double *t, double c, unsigned int N) {

  unsigned int ix, offset;
  double *rr, *ss, *tt;

#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset,rr,ss,tt) shared(r,s,t,c,N)
#endif
  for(ix = 0; ix < N; ix++) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    tt = t + offset;
    _fv_eq_fv_pl_fv_ti_re(rr, ss, tt, c);
  }
}  /* end of spinor_field_eq_spinor_field_pl_spinor_field_ti_re */

/***********************************************************
 * r -= c * s
 ***********************************************************/
void spinor_field_mi_eq_spinor_field_ti_re(double*r, double*s, double c, unsigned int N) {

  unsigned int ix, offset;
  double *rr, *ss;

#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset,rr,ss) shared(r,s,c,N)
#endif
  for(ix = 0; ix < N; ix++) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    _fv_mi_eq_fv_ti_re(rr, ss, c);
  }
}  /* end of spinor_field_mi_eq_spinor_field_ti_re */

/* r *= c */
void spinor_field_ti_eq_re (double *r, double c, unsigned int N) {

  unsigned int ix;
  double *rr;
#ifdef HAVE_OPENMP
#pragma omp parallel for private(rr) shared(r,c,N)
#endif
  for(ix = 0; ix < N; ix++ ) {
    rr = r + _GSI(ix);
    _fv_ti_eq_re(rr, c);
  }
}  /* end of spinor_field_ti_eq_re */

/* r = s * c */
void spinor_field_eq_spinor_field_ti_re (double *r, double *s, double c, unsigned int N) {

  unsigned int ix, offset;
  double *rr, *ss;
#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset,rr,ss) shared(r,s,c,N)
#endif
  for(ix = 0; ix < N; ix++ ) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    _fv_eq_fv_ti_re(rr, ss, c);
  }
}  /* end of spinor_field_eq_spinor_field_ti_re */

/****************************************************************************
 * d = || r - s ||
 ****************************************************************************/
void spinor_field_norm_diff (double*d, double *r, double *s, unsigned int N) {

  const int nthreads = g_num_threads;
  const int sincr    = _GSI(nthreads);

  unsigned int ix, iix;
  int threadid = 0;
  double daccum=0., daccumt, sp1[24];
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel default(shared) private(threadid,ix,iix,daccumt,sp1) shared(d,r,s,N,daccum)
{
  threadid = omp_get_thread_num();
#endif
  daccumt = 0.;
  iix = _GSI(threadid);
  for(ix = threadid; ix < N; ix += nthreads ) {
    _fv_eq_fv_mi_fv(sp1, r+iix, s+iix);
    _re_pl_eq_fv_dag_ti_fv(daccumt,sp1,sp1);
    iix += sincr;
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
  daccum += daccumt;

  /* TEST */
  /* fprintf(stdout, "# [spinor_field_norm_diff] proc%.4d thread%.2d daccumt = %25.16e\n", g_cart_id, threadid, daccumt); */

  omp_unset_lock(&writelock);
}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#else
  daccum = daccumt;
#endif

  /* TEST */
  /* fprintf(stdout, "# [spinor_field_norm_diff] proc%.4d daccum = %25.16e\n", g_cart_id, daccum); */

#ifdef HAVE_MPI
  daccumt = 0.;
  MPI_Allreduce(&daccum, &daccumt, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  /* TEST */
  /* fprintf(stdout, "# [spinor_field_norm_diff] proc%.4d full d = %25.16e\n", g_cart_id, daccumt); */
  *d = sqrt(daccumt);
#else
  *d = sqrt( daccum );
#endif
}  /* end of spinor_field_eq_spinor_field_ti_re */

/***********************************************************
 * r += s
 ***********************************************************/
void spinor_field_pl_eq_spinor_field(double*r, double*s, unsigned int N) {

  unsigned int ix, offset;
  double *rr, *ss;
#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset,rr,ss) shared(r,s,N)
#endif
  for(ix = 0; ix < N; ix++) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    _fv_pl_eq_fv(rr, ss);
  }
}  /* end of spinor_field_pl_eq_spinor_field_ti_re */


/***********************************************************
 * r = s - t
 ***********************************************************/
void spinor_field_eq_spinor_field_mi_spinor_field(double*r, double*s, double*t, unsigned int N) {

  unsigned int ix, offset;
  double *rr, *ss, *tt;
#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset,rr,ss,tt) shared(r,s,t,N)
#endif
  for(ix = 0; ix < N; ix++ ) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    tt = t + offset;
    _fv_eq_fv_mi_fv(rr, ss, tt);
  }
}  /* end of spinor_field_eq_spinor_field_mi_spinor_field */

/***********************************************************
 * r = s + t
 ***********************************************************/
void spinor_field_eq_spinor_field_pl_spinor_field(double*r, double*s, double*t, unsigned int N) {

  unsigned int ix, offset;
  double *rr, *ss, *tt;
#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset,rr,ss,tt) shared(r,s,t,N)
#endif
  for(ix = 0; ix < N; ix++ ) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    tt = t + offset;
    _fv_eq_fv_pl_fv(rr, ss, tt);
  }
}  /* end of spinor_field_eq_spinor_field_pl_spinor_field */

/***********************************************************
 * r = gamma[ gid ] x s
 * r and s can be same memory region
 ***********************************************************/
void spinor_field_eq_gamma_ti_spinor_field(double*r, int gid, double*s, unsigned int N) {

  unsigned int ix, offset;
  double *rr, *ss;
  if(r != s) {
#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset,rr,ss) shared(r,s,N)
#endif
    for(ix = 0; ix < N; ix++ ) {
      offset = _GSI(ix);
      rr = r + offset;
      ss = s + offset;
      _fv_eq_gamma_ti_fv(rr, gid, ss);
    }
  } else {
#ifdef HAVE_OPENMP
#pragma omp parallel private(offset,rr,ss) shared(r,s,N)
{
#endif
    double spinor1[24];
#ifdef HAVE_OPENMP
#pragma omp parallel for 
#endif
    for(ix = 0; ix < N; ix++ ) {
      offset = _GSI(ix);
      rr = r + offset;
      ss = s + offset;
      _fv_eq_gamma_ti_fv(spinor1, gid, ss);
      _fv_eq_fv(rr, spinor1);
    }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  }
}  /* end of spinor_field_eq_gamma_ti_spinor_field */

void g5_phi(double *phi, unsigned int N) {
#ifdef HAVE_OPENMP
#pragma omp parallel shared(phi, N)
{
#endif
  int ix;
  double spinor1[24];
  double *phi_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    phi_ = phi + _GSI(ix);
    /*
    _fv_eq_gamma_ti_fv(spinor1, 5, phi_);
    _fv_eq_fv(phi_, spinor1);
     */
    _fv_ti_eq_g5(phi_);
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
}  /* end of g5_phi */


/********************************************
 * check Ward-identity in position space
 * for conn
 ********************************************/

int check_cvc_wi_position_space (double *conn) {

  int nu;
  int exitstatus;
  double ratime, retime;

  /********************************************
   * check the Ward identity in position space 
   ********************************************/
  ratime = _GET_TIME;
#ifdef HAVE_MPI
  const unsigned int VOLUMEplusRAND = VOLUME + RAND;
  const unsigned int stride = VOLUMEplusRAND;
  double *conn_buffer = (double*)malloc(32*VOLUMEplusRAND*sizeof(double));
  if(conn_buffer == NULL)  {
    fprintf(stderr, "# [check_cvc_wi_position_space] Error from malloc\n");
    return(1);
  }

  for(nu=0; nu<16; nu++) {
    memcpy(conn_buffer+2*nu*VOLUMEplusRAND, conn+2*nu*VOLUME, 2*VOLUME*sizeof(double));
    xchange_contraction(conn_buffer+2*nu*VOLUMEplusRAND, 2);
  }
#else
  const unsigned int stride = VOLUME;
  double *conn_buffer = conn;
#endif

  if(g_cart_id == 0) fprintf(stdout, "# [check_cvc_wi_position_space] checking Ward identity in position space\n");
  for(nu=0; nu<4; nu++) {
    double norm = 0.;
    complex w;
    unsigned int ix;
    for(ix=0; ix<VOLUME; ix++ ) {
      w.re = conn_buffer[_GWI(4*0+nu,ix          ,stride)  ] + conn_buffer[_GWI(4*1+nu,ix          ,stride)  ]
           + conn_buffer[_GWI(4*2+nu,ix          ,stride)  ] + conn_buffer[_GWI(4*3+nu,ix          ,stride)  ]
           - conn_buffer[_GWI(4*0+nu,g_idn[ix][0],stride)  ] - conn_buffer[_GWI(4*1+nu,g_idn[ix][1],stride)  ]
           - conn_buffer[_GWI(4*2+nu,g_idn[ix][2],stride)  ] - conn_buffer[_GWI(4*3+nu,g_idn[ix][3],stride)  ];

      w.im = conn_buffer[_GWI(4*0+nu,ix          ,stride)+1] + conn_buffer[_GWI(4*1+nu,ix          ,stride)+1]
           + conn_buffer[_GWI(4*2+nu,ix          ,stride)+1] + conn_buffer[_GWI(4*3+nu,ix          ,stride)+1]
           - conn_buffer[_GWI(4*0+nu,g_idn[ix][0],stride)+1] - conn_buffer[_GWI(4*1+nu,g_idn[ix][1],stride)+1]
           - conn_buffer[_GWI(4*2+nu,g_idn[ix][2],stride)+1] - conn_buffer[_GWI(4*3+nu,g_idn[ix][3],stride)+1];
      
      norm += w.re*w.re + w.im*w.im;
    }
#ifdef HAVE_MPI
    double dtmp = norm;
    exitstatus = MPI_Allreduce(&dtmp, &norm, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[check_cvc_wi_position_space] Error from MPI_Allreduce, status was %d\n", exitstatus);
      return(2);
    }
#endif
    if(g_cart_id == 0) fprintf(stdout, "# [check_cvc_wi_position_space] WI nu = %2d norm = %25.16e\n", nu, norm);
  }  /* end of loop on nu */
#ifdef HAVE_MPI
  free(conn_buffer);
#endif
  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [check_cvc_wi_position_space] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);
}  /* end of check_cvc_wi_position_space */

/***************************************************
 * rotate fermion propagator field from
 * twisted to physical basis (depending on sign)
 *
 * safe, if s = r
 ***************************************************/
int fermion_propagator_field_tm_rotation(fermion_propagator_type *s, fermion_propagator_type *r, int sign, int fermion_type, unsigned int N) {

  if( fermion_type == _TM_FERMION ) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    unsigned int ix;
    fermion_propagator_type fp1, fp2, s_, r_;
  
    create_fp(&fp1);
    create_fp(&fp2);

#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix=0; ix<N; ix++) {

      s_ = s[ix];
      r_ = r[ix];

      /* flavor rotation */
  
      _fp_eq_rot_ti_fp(fp1, r_, sign, fermion_type, fp2);
      _fp_eq_fp_ti_rot(s_, fp1, sign, fermion_type, fp2);

    }  /* end of loop on ix */

    free_fp(&fp1);
    free_fp(&fp2);
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  } else if ( fermion_type == _WILSON_FERMION && s != r) {
    size_t bytes = N * g_fv_dim * g_fv_dim * 2 * sizeof(double);
    memcpy(s[0][0], r[0][0], bytes);
  }
  return(0);
}  /* end of fermion_propagator_field_tm_rotation */

/***************************************************
 * rotate spinor field from
 * twisted to physical basis (depending on sign)
 *
 * safe, if s = r
 ***************************************************/
int spinor_field_tm_rotation(double*s, double*r, int sign, int fermion_type, unsigned int N) {

  const double norm = 1. / sqrt(2.);

  if( fermion_type == _TM_FERMION ) {
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
    unsigned int ix;
    double sp1[_GSI(1)], sp2[_GSI(1)], *s_, *r_;
  
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix=0; ix<N; ix++) {

      s_ = s + _GSI(ix);
      r_ = r + _GSI(ix);

      /* flavor rotation */
      _fv_eq_gamma_ti_fv(sp1, 5, r_);
      _fv_eq_fv_ti_im(sp2, sp1, (double)sign );
      _fv_eq_fv_pl_fv(sp1, r_, sp2);
      _fv_eq_fv_ti_re(s_, sp1, norm);
    }  /* end of loop on ix */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  } else if ( fermion_type == _WILSON_FERMION  && s != r ) {
    memcpy(s, r, _GSI(N)*sizeof(double));
  }
  return(0);
}  /* end of fermion_propagator_field_tm_rotation */

/***************************************************
 * spinor fields to fermion propagator points
 ***************************************************/
int assign_fermion_propagaptor_from_spinor_field (fermion_propagator_type *s, double**prop_list, unsigned int N) {

  if(s[0][0] == prop_list[0] ) {
    fprintf(stderr, "[assign_fermion_propagaptor_from_spinor_field] Error, input fields have same address\n");
    return(1);
  }

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  unsigned int ix;
  fermion_propagator_type s_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix=0; ix<N; ix++) {
    s_ = s[ix];
    _assign_fp_point_from_field(s_, prop_list, ix);
  }
#ifdef HAVE_OPENMP
}
#endif
  return(0);
}  /* end of assign_fermion_propagaptor_from_spinor_field */

/***************************************************
 * fermion propagator points to spinor fields
 ***************************************************/
int assign_spinor_field_from_fermion_propagaptor (double**prop_list, fermion_propagator_type *s, unsigned int N) {

  if(s[0][0] == prop_list[0]) {
    fprintf(stderr, "[assign_spinor_field_from_fermion_propagaptor] Error, input fields have same address\n");
    return(1);
  }
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  unsigned int ix;
  fermion_propagator_type s_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix=0; ix<N; ix++) {
    s_ = s[ix];
    _assign_fp_field_from_point(prop_list, s_, ix);
  }
#ifdef HAVE_OPENMP
}
#endif
  return(0);
}  /* end of assign_spinor_field_from_fermion_propagaptor */

/***************************************************
 * component of fermion propagator points
 * to spinor fields
 ***************************************************/
int assign_spinor_field_from_fermion_propagaptor_component (double*spinor_field, fermion_propagator_type *s, int icomp, unsigned int N) {

  if(s[0][0] == spinor_field) {
    fprintf(stderr, "[assign_spinor_field_from_fermion_propagaptor] Error, input fields have same address\n");
    return(1);
  }
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  unsigned int ix;
  fermion_propagator_type s_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix=0; ix<N; ix++) {
    s_ = s[ix];
    _assign_fp_field_from_point_component (&(spinor_field), s_, icomp, ix);
  }
#ifdef HAVE_OPENMP
}
#endif
  return(0);
}  /* end of assign_spinor_field_from_fermion_propagaptor_component */

/***********************************************************
 * r = s * c
 * safe, if r = s
 ***********************************************************/
void spinor_field_eq_spinor_field_ti_real_field (double*r, double*s, double *c, unsigned int N) {

  unsigned int ix, offset;
  double *rr, *ss, cc;

#ifdef HAVE_OPENMP
#pragma omp parallel for private(offset,rr,ss,cc) shared(r,s,c,N)
#endif
  for(ix = 0; ix < N; ix++) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    cc = c[ix];
    _fv_eq_fv_ti_re(rr, ss, cc);
  }
}  /* end of spinor_field_eq_spinor_field_ti_real_field */

/***********************************************************
 * r = s * c
 * safe, if r = s
 ***********************************************************/
void spinor_field_eq_spinor_field_ti_complex_field (double*r, double*s, double *c, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel shared(r,s,c,N)
{
#endif
  unsigned int ix, offset;
  double *rr, *ss;
  complex w;
  double sp1[_GSI(1)];

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    w.re = c[2*ix  ];
    w.im = c[2*ix+1];
    _fv_eq_fv_ti_co(sp1, ss, &w);
    _fv_eq_fv(rr, sp1);
  }
#ifdef HAVE_OPENMP
}
#endif
}  /* end of spinor_field_eq_spinor_field_ti_complex_field */

/***********************************************************
 * r = s * w
 * safe, if r = s
 ***********************************************************/
void spinor_field_eq_spinor_field_ti_co (double*r, double*s, complex w, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel shared(r,s,w,N)
{
#endif
  unsigned int ix, offset;
  double *rr, *ss;
  double sp1[_GSI(1)];

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    _fv_eq_fv_ti_co(sp1, ss, &w);
    _fv_eq_fv(rr, sp1);
  }
#ifdef HAVE_OPENMP
}
#endif
}  /* end of spinor_field_eq_spinor_field_ti_co */


/***********************************************************
 * r = s * w
 * safe, if r = s
 ***********************************************************/
void spinor_field_pl_eq_spinor_field_ti_co (double*r, double*s, complex w, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel shared(r,s,w,N)
{
#endif
  unsigned int ix, offset;
  double *rr, *ss;
  double sp1[_GSI(1)];

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    offset = _GSI(ix);
    rr = r + offset;
    ss = s + offset;
    _fv_eq_fv_ti_co(sp1, ss, &w);
    _fv_pl_eq_fv(rr, sp1);
  }
#ifdef HAVE_OPENMP
}
#endif
}  /* end of spinor_field_pl_eq_spinor_field_ti_co */


}  /* end of namespace cvc */
