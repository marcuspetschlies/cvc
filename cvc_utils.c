/***************************************************
 * cvc_utils.c                                     *
 ***************************************************/
 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include "cvc_complex.h"
#include "global.h"
#include "cvc_linalg.h"
#include "cvc_geometry.h"
#include "mpi_init.h"
#include "default_input_values.h"
#include "propagator_io.h"
#include "get_index.h"
#include "read_input_parser.h"
#include "cvc_utils.h"
#include "ranlxd.h"

void EV_Hermitian_3x3_Matrix(double *M, double *lambda);

/*****************************************************
 * read the input file
 *****************************************************/
int cvc_read_input (char *filename) {

    
  FILE *fs;

  if((void*)(fs = fopen(filename, "r"))==NULL) {
#ifdef MPI
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
#ifdef MPI
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
#ifdef MPI
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
#ifdef MPI
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
    fprintf(stderr, "could not allocate memory for spinor fields\n");
#ifdef MPI
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
    fprintf(stderr, "could not allocate memory for spinor fields (dbl)\n");
#ifdef MPI
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
#ifdef MPI
    MPI_Abort(MPI_COMM_WORLD, 10);
    MPI_Finalize();
#endif
    exit(103);
  }
  return(0);
}

/*****************************************************
 * exchange the global gauge field cvc_gauge_field
 *****************************************************/

void xchange_gauge() {

#ifdef MPI
  int cntr=0;
  MPI_Request request[120];
  MPI_Status status[120];

  MPI_Isend(&cvc_gauge_field[0],         1, gauge_time_slice_cont, g_nb_t_dn, 83, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*VOLUME], 1, gauge_time_slice_cont, g_nb_t_up, 83, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&cvc_gauge_field[72*(T-1)*LX*LY*LZ], 1, gauge_time_slice_cont, g_nb_t_up, 84, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(T+1)*LX*LY*LZ], 1, gauge_time_slice_cont, g_nb_t_dn, 84, g_cart_grid, &request[cntr]);
  cntr++;

#if (defined PARALLELTX) || (defined PARALLELTXY)
  MPI_Isend(&cvc_gauge_field[0],                              1, gauge_x_slice_vector, g_nb_x_dn, 85, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ)],         1, gauge_x_slice_cont,   g_nb_x_up, 85, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&cvc_gauge_field[72*(LX-1)*LY*LZ],                1, gauge_x_slice_vector, g_nb_x_up, 86, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, gauge_x_slice_cont,   g_nb_x_dn, 86, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  cntr = 0;

  MPI_Isend(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ)], 1, gauge_xt_edge_vector, g_nb_t_dn, 87, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+RAND)], 1, gauge_xt_edge_cont, g_nb_t_up, 87, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ+(T-1)*LY*LZ)], 1, gauge_xt_edge_vector, g_nb_t_up, 88, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+RAND+2*LY*LZ)], 1, gauge_xt_edge_cont, g_nb_t_dn, 88, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXY
  MPI_Isend(&cvc_gauge_field[0], 1, gauge_y_slice_vector, g_nb_y_dn, 89, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_y_slice_cont, g_nb_y_up, 89, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&cvc_gauge_field[72*(LY-1)*LZ], 1, gauge_y_slice_vector, g_nb_y_up, 90, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+T*LX*LZ)], 1, gauge_y_slice_cont, g_nb_y_dn, 90, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);

  cntr = 0;

  MPI_Isend(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_yt_edge_vector, g_nb_t_dn, 91, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+RAND+4*LY*LZ)], 1, gauge_yt_edge_cont, g_nb_t_up, 91, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(T-1)*LX*LZ)], 1, gauge_yt_edge_vector, g_nb_t_up, 92, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+RAND+4*LY*LZ+2*LX*LZ)], 1, gauge_yt_edge_cont, g_nb_t_dn, 92, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)], 1, gauge_yx_edge_vector, g_nb_x_dn, 93, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+RAND+4*LY*LZ+4*LX*LZ)], 1, gauge_yx_edge_cont, g_nb_x_up, 93, g_cart_grid, &request[cntr]);
  cntr++;

  MPI_Isend(&cvc_gauge_field[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(LX-1)*LZ)], 1, gauge_yx_edge_vector, g_nb_x_up, 94, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&cvc_gauge_field[72*(VOLUME+RAND+4*LY*LZ+4*LX*LZ+2*T*LZ)], 1, gauge_yx_edge_cont, g_nb_x_dn, 94, g_cart_grid, &request[cntr]);
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
}

/*****************************************************
 * exchange any globally defined gauge field
 *****************************************************/
void xchange_gauge_field(double *gfield) {
#ifdef MPI
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

#if (defined PARALLELTX) || (defined PARALLELTXY)
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

#if defined PARALLELTXY
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

  MPI_Waitall(cntr, request, status);
#endif
}

/**************************************************************
 * exchange any gauge field defined on a timeslice communicator
 **************************************************************/
void xchange_gauge_field_timeslice(double *gfield) {
#if (defined MPI) && ( (defined PARALLELTX) || (defined PARALLELTXY) )

  int cntr=0;
  MPI_Request request[12];
  MPI_Status status[12];

  MPI_Isend(&gfield[0],                              1, gauge_x_slice_vector, g_ts_nb_x_dn, 83, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*LX*LY*LZ)],         1, gauge_x_slice_cont,   g_ts_nb_x_up, 83, g_ts_comm, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(LX-1)*LY*LZ],                1, gauge_x_slice_vector, g_ts_nb_x_up, 84, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, gauge_x_slice_cont,   g_ts_nb_x_dn, 84, g_ts_comm, &request[cntr]);
  cntr++;

#ifdef PARALLELTXY
  MPI_Isend(&gfield[0],                                        1, gauge_y_slice_vector, g_ts_nb_y_dn, 85, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*(LX*LY*LZ+T*LY*LZ))],         1, gauge_y_slice_cont,   g_ts_nb_y_up, 85, g_ts_comm, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(LY-1)*LZ],                             1, gauge_y_slice_vector, g_ts_nb_y_up, 86, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+2*(LX*LY*LZ+T*LY*LZ)+T*LX*LZ)], 1, gauge_y_slice_cont,   g_ts_nb_y_dn, 86, g_ts_comm, &request[cntr]);
  cntr++;

  MPI_Waitall(cntr, request, status);
  cntr = 0;

  MPI_Isend(&gfield[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ)],           1, gauge_yx_edge_vector, g_ts_nb_x_dn, 93, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*LY*LZ+4*LX*LZ)],           1, gauge_yx_edge_cont, g_ts_nb_x_up, 93, g_ts_comm, &request[cntr]);
  cntr++;

  MPI_Isend(&gfield[72*(VOLUME+2*LX*LY*LZ+2*T*LY*LZ+(LX-1)*LZ)], 1, gauge_yx_edge_vector, g_ts_nb_x_up, 94, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&gfield[72*(VOLUME+RAND+4*LY*LZ+4*LX*LZ+2*T*LZ)],    1, gauge_yx_edge_cont, g_ts_nb_x_dn, 94, g_ts_comm, &request[cntr]);
  cntr++;
#endif

  MPI_Waitall(cntr, request, status);

#endif
}

/*****************************************************
 * exchange a spinor field
 *****************************************************/
void xchange_field(double *phi) {

#ifdef MPI
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
#if (defined PARALLELTX) || (defined PARALLELTXY)
  MPI_Isend(&phi[0],                              1, spinor_x_slice_vector, g_nb_x_dn, 85, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*LX*LY*LZ)],         1, spinor_x_slice_cont,   g_nb_x_up, 85, g_cart_grid, &request[cntr]);
  cntr++;
  
  MPI_Isend(&phi[24*(LX-1)*LY*LZ],                1, spinor_x_slice_vector, g_nb_x_up, 86, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, spinor_x_slice_cont,   g_nb_x_dn, 86, g_cart_grid, &request[cntr]);
  cntr++;
#endif

#if defined PARALLELTXY
  MPI_Isend(&phi[0],                                        1, spinor_y_slice_vector, g_nb_y_dn, 87, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*(LX*LY*LZ+T*LY*LZ))],         1, spinor_y_slice_cont,   g_nb_y_up, 87, g_cart_grid, &request[cntr]);
  cntr++;
  
  MPI_Isend(&phi[24*(LY-1)*LZ],                             1, spinor_y_slice_vector, g_nb_y_up, 88, g_cart_grid, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*(LX*LY*LZ+T*LY*LZ)+T*LX*LZ)], 1, spinor_y_slice_cont,   g_nb_y_dn, 88, g_cart_grid, &request[cntr]);
  cntr++;
#endif
  MPI_Waitall(cntr, request, status);
#endif
}

/****************************************************************
 * exchange a spinor field defined on an a timeslice communicator
 ****************************************************************/
void xchange_field_timeslice(double *phi) {
#if (defined PARALLELTX) || (defined PARALLELTXY)
  int cntr=0;
  MPI_Request request[12];
  MPI_Status status[12];

  MPI_Isend(&phi[0],                              1, spinor_x_slice_vector, g_ts_nb_x_dn, 83, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*LX*LY*LZ)],         1, spinor_x_slice_cont,   g_ts_nb_x_up, 83, g_ts_comm, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[24*(LX-1)*LY*LZ],                1, spinor_x_slice_vector, g_ts_nb_x_up, 84, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*LX*LY*LZ+T*LY*LZ)], 1, spinor_x_slice_cont,   g_ts_nb_x_dn, 84, g_ts_comm, &request[cntr]);
  cntr++;

#if defined PARALLELTXY
  MPI_Isend(&phi[0],                                        1, spinor_y_slice_vector, g_ts_nb_y_dn, 85, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*(LX*LY*LZ+T*LY*LZ))],         1, spinor_y_slice_cont,   g_ts_nb_y_up, 85, g_ts_comm, &request[cntr]);
  cntr++;

  MPI_Isend(&phi[24*(LY-1)*LZ],                             1, spinor_y_slice_vector, g_ts_nb_y_up, 86, g_ts_comm, &request[cntr]);
  cntr++;
  MPI_Irecv(&phi[24*(VOLUME+2*(LX*LY*LZ+T*LY*LZ)+T*LX*LZ)], 1, spinor_y_slice_cont,   g_ts_nb_y_dn, 86, g_ts_comm, &request[cntr]);
  cntr++;
#endif

  MPI_Waitall(cntr, request, status);
#endif
}

/*****************************************************
 * measure the plaquette value
 *****************************************************/
void plaquette(double *pl) {

  int ix, mu, nu; 
  double s[18], t[18], u[18], pl_loc;
  complex w;

  pl_loc=0;

  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<3; mu++) {
    for(nu=mu+1; nu<4; nu++) {
      _cm_eq_cm_ti_cm(s, &cvc_gauge_field[72*ix+mu*18], &cvc_gauge_field[72*g_iup[ix][mu]+18*nu]);
      _cm_eq_cm_ti_cm(t, &cvc_gauge_field[72*ix+nu*18], &cvc_gauge_field[72*g_iup[ix][nu]+18*mu]);
      _cm_eq_cm_ti_cm_dag(u, s, t);
      _co_eq_tr_cm(&w, u);
      pl_loc += w.re;
    }
    }
  }

#ifdef MPI
  MPI_Reduce(&pl_loc, pl, 1, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
  *pl = pl_loc;
#endif
  *pl = *pl / ((double)T_global * (double)(LX*g_nproc_x) * (double)(LY*g_nproc_y) * (double)LZ * 18.);
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

#ifdef MPI
  MPI_Reduce(&pl_loc, pl, 1, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid);
#else
  *pl = pl_loc;
#endif
  *pl = *pl / ((double)T_global * (double)(LX*g_nproc_x) * (double)(LY*g_nproc_y) * (double)LZ * 18.);
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
#ifdef MPI
  double *buffer;
  MPI_Status status;
#endif

  if(g_cart_id==0) {
    if(append==1) ofs = fopen(filename, "a");
    else ofs = fopen(filename, "w");
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "could not open file %s for writing\n", filename);
#ifdef MPI
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
#ifdef MPI
#  if !(defined PARALLELTX || defined PARALLELTXY)
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
#ifdef MPI
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
  
  unsigned long int shift=0, count=0;
  int ix, mu, iy;
  double buffer[128];
  FILE *ofs = (FILE*)NULL;

  ofs = fopen(filename, "r");
  if(ofs==(FILE*)NULL) {
    fprintf(stderr, "could not open file %s for reading\n", filename);
    return(106);
  }
  if(format==2) {
    if(g_cart_id==0) fprintf(stdout, "# Reading contraction data in format %d\n", format);
#ifdef MPI
#  ifndef PARALLELTX
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
#ifdef MPI
#  ifndef PARALLELTX
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
#ifdef MPI
#  ifndef PARALLELTX
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

#ifdef MPI
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
                    cvc_gauge_field[ix+ 0], cvc_gauge_field[ix+ 1],
                    cvc_gauge_field[ix+ 2], cvc_gauge_field[ix+ 3],
                    cvc_gauge_field[ix+ 4], cvc_gauge_field[ix+ 5],
                    g_cart_id,
                    cvc_gauge_field[ix+ 6], cvc_gauge_field[ix+ 7],
                    cvc_gauge_field[ix+ 8], cvc_gauge_field[ix+ 9],
                    cvc_gauge_field[ix+10], cvc_gauge_field[ix+11],
                    g_cart_id,
                    cvc_gauge_field[ix+12], cvc_gauge_field[ix+13],
                    cvc_gauge_field[ix+14], cvc_gauge_field[ix+15],
                    cvc_gauge_field[ix+16], cvc_gauge_field[ix+17]);
    }
  }
  }
  }
  }

  return(0);
}

int printf_spinor_field(double *s, FILE *ofs) {

  int i, start_valuet, start_valuex;
  int x0, x1, x2, x3, ix;
  int y0, y1, y2, y3;

  if( (ofs == (FILE*)NULL) || (s==(double*)NULL) ) return(108);
#ifdef MPI
  start_valuet = 1;
#  ifdef PARALLELTX
  start_valuex = 1;
#  else
  start_valuex = 0;
# endif
#else
  start_valuet = 0;
#endif

  for(x0=-start_valuet; x0<T +start_valuet; x0++) {
  for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
    if( (x0==-1 || x0==T) && (x1==-1 || x1==LX)) continue;
  for(x2= 0; x2<LX;  x2++) {
  for(x3= 0; x3<LX;  x3++) {
    y0=x0; y1=x1; y2=x2; y3=x3;
    if(x0==-1) y0=T+1;
    ix = _GSI(g_ipt[y0][y1][y2][y3]);
    fprintf(ofs, "# [%2d] t=%3d, x=%3d, y=%3d, z=%3d\n", g_cart_id, x0, x1, x2, x3);
    fprintf(ofs, "%16.7e %16.7e\t%16.7e %16.7e\t%16.7e %16.7e\n"\
                 "%16.7e %16.7e\t%16.7e %16.7e\t%16.7e %16.7e\n"\
                 "%16.7e %16.7e\t%16.7e %16.7e\t%16.7e %16.7e\n"\
                 "%16.7e %16.7e\t%16.7e %16.7e\t%16.7e %16.7e\n",
                 s[ix   ], s[ix+ 1], s[ix+ 2], s[ix+ 3], s[ix+ 4], s[ix+ 5],
                 s[ix +6], s[ix+ 7], s[ix+ 8], s[ix+ 8], s[ix+10], s[ix+11],
                 s[ix+12], s[ix+13], s[ix+14], s[ix+15], s[ix+16], s[ix+17],
                 s[ix+18], s[ix+19], s[ix+20], s[ix+21], s[ix+22], s[ix+23]);
  }
  }
  }
  }

  return(0);
}

void set_default_input_values(void) {

  T_global    = _default_T_global;
  Tstart      = _default_Tstart;
  LX          = _default_LX;
  LXstart     = _default_LXstart;
  LY          = _default_LY;
  LYstart     = _default_LYstart;
  LZ          = _default_LZ;
  g_nproc_t   = _default_nproc_t;
  g_nproc_x   = _default_nproc_x;
  g_nproc_y   = _default_nproc_y;
  g_nproc_z   = _default_nproc_z;
  g_ts_id     = _default_ts_id;
  g_xs_id     = _default_xs_id;
/*  g_ys_id     = _default_ys_id; */
  g_proc_coords[0] = 0;
  g_proc_coords[1] = 0;
  g_proc_coords[2] = 0;
  g_proc_coords[3] = 0;

  Nconf       = _default_Nconf;
  g_kappa     = _default_kappa;
  g_mu        = _default_mu;
  g_musigma   = _default_musigma;
  g_mudelta   = _default_mudelta;
  g_sourceid  = _default_sourceid;
  g_sourceid2 = _default_sourceid2;
  g_sourceid_step = _default_sourceid_step;
  Nsave       = _default_Nsave;
  format      = _default_format;
  BCangle[0]  = _default_BCangleT;
  BCangle[1]  = _default_BCangleX;
  BCangle[2]  = _default_BCangleY;
  BCangle[3]  = _default_BCangleZ;
  g_resume    = _default_resume;
  g_subtract  = _default_subtract;
  g_source_location = _default_source_location;
  strcpy(filename_prefix,      _default_filename_prefix);
  strcpy(filename_prefix2,     _default_filename_prefix2);
  strcpy(gaugefilename_prefix, _default_gaugefilename_prefix);
  g_gaugeid   = _default_gaugeid; 
  g_gaugeid2  = _default_gaugeid2;
  g_gauge_step= _default_gauge_step;
  g_seed      = _default_seed;
  g_noise_type= _default_noise_type;
  g_source_type= _default_source_type;
  solver_precision = _default_solver_precision;
  niter_max = _default_niter_max;
  hpe_order_min = _default_hpe_order_min;
  hpe_order_max = _default_hpe_order_max;
  hpe_order = _default_hpe_order;
 
  g_cutradius = _default_cutradius;
  g_cutangle  = _default_cutangle;
  g_cutdir[0] = _default_cutdirT;
  g_cutdir[1] = _default_cutdirX;
  g_cutdir[2] = _default_cutdirY;
  g_cutdir[3] = _default_cutdirZ;
  g_rmin      = _default_rmin;
  g_rmax      = _default_rmax;

  avgT        = _default_avgT;
  avgL        = _default_avgL;

  model_dcoeff_re = _default_model_dcoeff_re;
  model_dcoeff_im = _default_model_dcoeff_im;
  model_mrho      = _default_model_mrho;
  ft_rmax[0]      = _default_ft_rmax;
  ft_rmax[1]      = _default_ft_rmax;
  ft_rmax[2]      = _default_ft_rmax;
  ft_rmax[3]      = _default_ft_rmax;
  g_prop_normsqr  = _default_prop_normsqr; 
  g_qhatsqr_min  = _default_qhatsqr_min;
  g_qhatsqr_max  = _default_qhatsqr_max;

  Nlong        = _default_Nlong;
  N_ape        = _default_N_ape;
  N_Jacobi     = _default_N_Jacobi;
  alpha_ape    = _default_alpha_ape;
  kappa_Jacobi = _default_kappa_Jacobi;
  g_source_timeslice  = _default_source_timeslice;
  g_no_extra_masses = _default_no_extra_masses;
  g_no_light_masses = _default_no_light_masses;
  g_no_strange_masses = _default_no_strange_masses;
  g_local_local       = _default_local_local;
  g_local_smeared     = _default_local_smeared;
  g_smeared_local     = _default_smeared_local;
  g_smeared_smeared   = _default_smeared_smeared;
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

  int ix;
  double ran[8], inorm, u[18], v[18];
#ifdef MPI 
  int cntr;
  MPI_Request request[5];
  MPI_Status status[5];
#endif

  if(g_cart_id==0) fprintf(stdout, "initialising random gauge transformation\n");

  *g = (double*)calloc(18*VOLUMEPLUSRAND, sizeof(double));
  if(*g==(double*)NULL) {
    fprintf(stderr, "not enough memory\n");
#ifdef MPI
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

    ran[0]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[1]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[2]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[3]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[4]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[5]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[6]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[7]=((double)rand()) / ((double)RAND_MAX+1.0);

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
  }

  /* exchange the field */
  fprintf(stdout, "\nThe gauge trafo field:\n");
  for(ix=0; ix<9*VOLUME; ix++) fprintf(stdout, "%6d%25.16e%25.16e\n", ix, (*g)[2*ix], (*g)[2*ix+1]);
#ifdef MPI

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
  printf_gauge_field(cvc_gauge_field, ofs);
  fclose(ofs);
*/
  for(ix=0; ix<VOLUME; ix++) {
    for(mu=0; mu<4; mu++) {
      _cm_eq_cm_ti_cm(u, &g[18*ix], &cvc_gauge_field[_GGI(ix,mu)]);
      _cm_eq_cm_ti_cm_dag(&cvc_gauge_field[_GGI(ix, mu)], u, &g[18*g_iup[ix][mu]]);
    }
  }
  xchange_gauge();
/*
  ofs = fopen("gauge_field_after", "w");
  printf_gauge_field(cvc_gauge_field, ofs);
  fclose(ofs);
*/
}

/* apply gt to propagator; (is,ic) = (spin, colour)-index */
void apply_gt_prop(double *g, double *phi, int is, int ic, int mu, char *basename, int source_location) {
#ifndef MPI
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

  if(format==2 || format==3) {
    isx[0] =  g_source_location/(LX*LY*LZ);
    isx[1] = (g_source_location%(LX*LY*LZ)) / (LY*LZ);
    isx[2] = (g_source_location%(LY*LZ)) / LZ;
    isx[3] = (g_source_location%LZ);
    Lsize[0] = T_global;
    Lsize[1] = LX;
    Lsize[2] = LY;
    Lsize[3] = LZ;
  }

  if(format==0) {
    /* format from invert */
    if(sign==1) {
      sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, nu, sc);
    }
    else {
      sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix2, Nconf, nu, sc);
    }
  }
  else if(format==2) {
    /* Dru/Xu format */
    if(nu!=4) isx[nu] = (isx[nu]+1)%Lsize[nu];
    if(sign==1) {
      sprintf(filename, "conf.%.4d.t%.2dx%.2dy%.2dz%.2ds%.2dc%.2d.pmass.%s.inverted", 
        Nconf, isx[0], isx[1], isx[2], isx[3], sc/3, sc%3, filename_prefix);
    }
    else {
      sprintf(filename, "conf.%.4d.t%.2dx%.2dy%.2dz%.2ds%.2dc%.2d.nmass.%s.inverted", 
        Nconf, isx[0], isx[1], isx[2], isx[3], sc/3, sc%3, filename_prefix);
    }
  }
  else if(format==3) {
    /* Dru/Xu format */
    if(nu!=4) isx[nu] = (isx[nu]+1)%Lsize[nu];
    if(sign==1) {
      sprintf(filename, "conf.%.4d.t%.2dx%.2dy%.2dz%.2d.pmass.%s.%.2d.inverted", 
        Nconf, isx[0], isx[1], isx[2], isx[3], filename_prefix, sc);
    }
    else {
      sprintf(filename, "conf.%.4d.t%.2dx%.2dy%.2dz%.2d.nmass.%s.%.2d.inverted", 
        Nconf, isx[0], isx[1], isx[2], isx[3], filename_prefix, sc);
    }
  }
  else if(format==4) {
    /* my format for lvc */
    if(sign==1) {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix, Nconf, sc);
    }
    else {
      sprintf(filename, "%s.%.4d.%.2d.inverted", filename_prefix2, Nconf, sc);
    }
  }
  else if(format==1) {
    /* GWC format */
  }
}

int wilson_loop(complex *w, const int xstart, const int dir, const int Ldir) {

  int ix, i;
  double U_[18], V_[18], *u1=(double*)NULL, *u2=(double*)NULL, *u3=(double*)NULL;
  complex tr, phase;

  if(dir==0) {
    ix=g_iup[xstart][dir];
    _cm_eq_cm_ti_cm(V_, cvc_gauge_field+_GGI(xstart, dir), cvc_gauge_field+_GGI(ix, dir));
    u1=U_; u2=V_;
    for(i=2; i<Ldir; i++) {
      ix = g_iup[ix][dir];
      u3=u1; u1=u2; u2=u3;
      _cm_eq_cm_ti_cm(u2, u1, cvc_gauge_field+_GGI(ix,dir));
    }
#ifdef MPI
    if(g_cart_id==0) {
      fprintf(stderr, "MPI version _NOT_ yet implemented;\n");
      return(1);
    }
#endif
    _co_eq_tr_cm(&tr, u2);
  } else {
    ix=g_iup[xstart][dir];
    _cm_eq_cm_ti_cm(V_, cvc_gauge_field+_GGI(xstart, dir), cvc_gauge_field+_GGI(ix, dir));
    u1=U_; u2=V_;
    for(i=2; i<Ldir; i++) {
      ix = g_iup[ix][dir];
      u3=u1; u1=u2; u2=u3;
      _cm_eq_cm_ti_cm(u2, u1, cvc_gauge_field+_GGI(ix,dir));
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
}

int ranz2(double * y, int NRAND) {
  int k;
  double r[2];
  double sqrt2inv = 1. / sqrt(2.);

  if(NRAND%2 != 0) {
    fprintf(stderr, "ERROR, NRAND must be an even number\n");
    return(1);
  }

  for(k=0; k<NRAND/2; k++) {
    cvc_ranlxd(r,2);
    y[2*k  ] = (double)(2 * (int)(r[0]>=0.5) - 1) * sqrt2inv;
    y[2*k+1] = (double)(2 * (int)(r[1]>=0.5) - 1) * sqrt2inv;
  }
  return(0);
}

/********************************************************
 * random_gauge_field
 ********************************************************/
void cvc_random_gauge_field(double *gfield, double h) {

  int mu, ix;
  double buffer[72], *gauge_point[4];
#ifdef MPI
  int iproc, tgeom[2], tag, t;
  double *gauge_ts;
  MPI_Status mstatus;
#endif

  gauge_point[0] = buffer;
  gauge_point[1] = buffer + 18;
  gauge_point[2] = buffer + 36;
  gauge_point[3] = buffer + 54;

  if(g_cart_id==0) {
    for(ix=0; ix<VOLUME; ix++) {
      random_gauge_point(gauge_point, h);
      memcpy((void*)(gfield + _GGI(ix,0)), (void*)buffer, 72*sizeof(double));
    }
  }
#ifdef MPI
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
}

/******************************************************
 * read_pimn
 ******************************************************/
int read_pimn(double *pimn, const int read_flag) {

  char filename[800];
  int iostat, ix, mu, nu, iix, np;
  double ratime, retime, buff[32], *buff2=(double*)NULL;

  FILE *ofs;
#ifndef MPI
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
 * random_gauge_point
 *********************************************/
void random_gauge_point(double **gauge_point, double heat) {

  int mu;
  double ran[8], inorm, u[18], v[18];


  for(mu=0; mu<4; mu++) {
    u[ 0] = 0.; u[ 1] = 0.; u[ 2] = 0.; u[ 3] = 0.; u[ 4] = 0.; u[ 5] = 0.;
    u[ 6] = 0.; u[ 7] = 0.; u[ 8] = 0.; u[ 9] = 0.; u[10] = 0.; u[11] = 0.;
    u[12] = 0.; u[13] = 0.; u[14] = 0.; u[15] = 0.; u[16] = 0.; u[17] = 0.;
    v[ 0] = 0.; v[ 1] = 0.; v[ 2] = 0.; v[ 3] = 0.; v[ 4] = 0.; v[ 5] = 0.;
    v[ 6] = 0.; v[ 7] = 0.; v[ 8] = 0.; v[ 9] = 0.; v[10] = 0.; v[11] = 0.; 
    v[12] = 0.; v[13] = 0.; v[14] = 0.; v[15] = 0.; v[16] = 0.; v[17] = 0.;

    ran[0]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[1]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[2]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[3]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[4]=((double)rand()) / ((double)RAND_MAX+1.0); 
    ran[5]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[6]=((double)rand()) / ((double)RAND_MAX+1.0);
    ran[7]=((double)rand()) / ((double)RAND_MAX+1.0);

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

    _cm_eq_cm_ti_cm(gauge_point[mu], u, v);
  }


}


/********************************************************
 * random_gauge_field2
 ********************************************************/
void cvc_random_gauge_field2(double *gfield) {

  int mu, ix, i;
  double norm;
  complex u[3], v[3], w[3], z[3], pr;
#ifdef MPI
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
#ifdef MPI
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

int init_hpe_fields(int ***loop_tab, int ***sigma_tab, int ***shift_start, double **tcf, double **tcb) {
  int i;
  if(loop_tab    != (int***)NULL) { for(i=0; i<HPE_MAX_ORDER; i++) loop_tab[i]    = (int**)NULL; }
  if(sigma_tab   != (int***)NULL) { for(i=0; i<HPE_MAX_ORDER; i++) sigma_tab[i]   = (int**)NULL; }
  if(shift_start != (int***)NULL) { for(i=0; i<HPE_MAX_ORDER; i++) shift_start[i] = (int**)NULL; }
  if(tcf != (double**)NULL) { for(i=0; i<HPE_MAX_ORDER; i++) tcf[i] = (double*)NULL; }
  if(tcb != (double**)NULL) { for(i=0; i<HPE_MAX_ORDER; i++) tcb[i] = (double*)NULL; }
  return(0);
}

int free_hpe_fields(int ***loop_tab, int ***sigma_tab, int ***shift_start, double **tcf, double **tcb) {
  int i;
  if(loop_tab    != (int***)NULL) { 
    for(i=0; i<HPE_MAX_ORDER; i++) {
      if(loop_tab[i]  != (int**)NULL) { free(loop_tab[i][0]); free(loop_tab[i]); }
    }
  }
  if(sigma_tab   != (int***)NULL) { 
    for(i=0; i<HPE_MAX_ORDER; i++) {
      if(sigma_tab[i] != (int**)NULL) { free(sigma_tab[i][0]); free(sigma_tab[i]); }
    }
  }
  if(shift_start    != (int***)NULL) { 
    for(i=0; i<HPE_MAX_ORDER; i++) {
      if(shift_start[i]!=(int**)NULL) { free(shift_start[i][0]); free(shift_start[i]); }
    }
  }
  if(tcf != (double**)NULL) { 
    for(i=0; i<HPE_MAX_ORDER; i++) {
      if(tcf[i]!=(double*)NULL) { free(tcf[i]); }
    }
  }
  if(tcb != (double**)NULL) { 
    for(i=0; i<HPE_MAX_ORDER; i++) {
      if(tcb[i]!=(double*)NULL) { free(tcb[i]); }
    }
  }
  return(0);
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

int rangauss (double * y1, int NRAND) {

  int k;
  int nrandh;
  double x1, x2;

  if(NRAND%2 != 0) {
    fprintf(stderr, "ERROR, NRAND must be an even number\n");
    return(1);
  }

  nrandh = NRAND/2;

  for(k=0; k<nrandh; k++) {
/*
    x1 = (1. + (double)rand() ) / ( (double)(RAND_MAX) + 1. );
    x2 = (double)rand() / ( (double)(RAND_MAX) + 1. );
*/
    cvc_ranlxd(&x1,1);
    cvc_ranlxd(&x2,1);
    y1[2*k  ] = sqrt(-2.*log(x1)) * cos(2*M_PI*x2);
    y1[2*k+1] = sqrt(-2.*log(x1)) * sin(2*M_PI*x2);
  }
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

int _F(ilaenv)(int *ispec, char name[], char opts[], int *n1, int *n2, int *n3, int *n4);

void _F(zheev)(char *jobz, char *uplo, int *n, double a[], int *lda, double w[], double work[], int *lwork, double *rwork, int *info);

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
    fprintf(stderr, "lambda = (%+6.3lf  , %+6.3lf , %+6.3lf).\n", lambda[0], lambda[1], lambda[2]);
    fprintf(stderr, "Error: inline void SU3_proj(...\n");
#ifdef MPI
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

/******************************************************************************
 * contract_twopoint_xdep
 ******************************************************************************/
void contract_twopoint_xdep(double *contr, const int idsource, const int idsink, double **chi, double **phi, int n_c, int stride, double factor) {

  int ix, iix, psource[4], isimag, mu, c, j;
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

  //fprintf(stdout, "\n# [contract_twopoint_xdep] ix mu c re im\n");
  for(ix=0; ix<VOLUME; ix++) {
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
        _fv_eq_gamma_ti_fv(spinor1, idsink, phi[mu*n_c+c]+_GSI(ix));
        _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
        _co_eq_fv_dag_ti_fv(&w, chi[psource[mu]*n_c+c]+_GSI(ix), spinor2);

        if( !isimag ) {
          *(contr+2*iix  ) += factor * ssource[mu] * w.re;
          *(contr+2*iix+1) += factor * ssource[mu] * w.im;
        } else {
/*
            contr[2*tt  ] += ssource[mu]*w.im;
            contr[2*tt+1] += ssource[mu]*w.re;
*/
          *(contr+2*iix  ) +=  factor * ssource[mu] * w.im;
          *(contr+2*iix+1) += -factor * ssource[mu] * w.re;
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
    memcpy((void*)(gauge_transform+ix), (void*)(cvc_gauge_field+count), 18*sizeof(double));
    count += 72;
  }

  count = 72*VOL3;
  iix   = 18*VOL3;
  for(ix=36*VOL3; ix<18*VOLUME; ix+=18) {
    _cm_eq_cm_ti_cm(gauge_transform+ix, gauge_transform+iix, cvc_gauge_field+count);
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

