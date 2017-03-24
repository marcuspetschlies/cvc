/****************************************************
 * test_sp_mom.cpp
 *
 * Di 7. Jun 14:09:06 CEST 2016
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define MAIN_PROGRAM

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "set_default.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "laplace_linalg.h"
#include "hyp_smear.h"
#include "Q_phi.h"
#include "scalar_products.h"
#include "ranlxd.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c;
  /* int status; */
  int i;
  int filename_set = 0;
  int ix, iix;
  int isource_momentum, isource_gamma_id;
  int x0, x1, x2, x3;
  int no_eo_fields;

  double dtmp;
  double *phase_e=NULL, *phase_o = NULL;
  double *buffer=NULL, *buffer2 = NULL;
  complex w, w2, w3;
  complex *sp_le_e = NULL, *sp_le_o = NULL, *sp_eo_e = NULL, *sp_eo_o = NULL;

  char filename[200];

  double **eo_spinor_field=NULL;
  double ratime, retime;
  unsigned int Vhalf;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_cart_id==0) fprintf(stdout, "# [test_sp_mom] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_sp_mom] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  status = tmLQCD_invert_init(argc, argv, 1);
  if(status != 0) {
    EXIT(14);
  }
  status = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(status != 0) {
    EXIT(15);
  }
  status = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(status != 0) {
    EXIT(16);
  }
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize geometry */
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_sp_mom] Error from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();


  Vhalf = VOLUME / 2;

  /***********************************************
   * allocate spinor fields
   ***********************************************/
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  no_eo_fields = 5;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields; i++) alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);

  phase_e = (double*)malloc(Vhalf*2*sizeof(double));
  phase_o = (double*)malloc(Vhalf*2*sizeof(double));
  if(phase_e == NULL || phase_o == NULL) {
    fprintf(stderr, "[test_sp_mom] Error from malloc\n");
    EXIT(14);
  }

  buffer = (double*)malloc(2*Vhalf*sizeof(double));
  if(buffer == NULL)  {
    fprintf(stderr, "[test_sp_mom] Error from malloc\n");
    EXIT(19);
  }

  g_seed = 10000 + g_cart_id;
  rlxd_init(2, g_seed);

  /* set the spinor field */
  rangauss (g_spinor_field[0], VOLUME*24);
  rangauss (g_spinor_field[1], VOLUME*24);

  spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[0], eo_spinor_field[1]);
  spinor_field_lexic2eo (g_spinor_field[1], eo_spinor_field[2], eo_spinor_field[3]);


  for(isource_momentum=0; isource_momentum < g_source_momentum_number; isource_momentum++) {

    g_source_momentum[0] = g_source_momentum_list[isource_momentum][0];
    g_source_momentum[1] = g_source_momentum_list[isource_momentum][1];
    g_source_momentum[2] = g_source_momentum_list[isource_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [test_sp_mom] using source momentum = (%d, %d, %d)\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    }

    /* make phase field in eo ordering */
    for(x0=0; x0<T; x0++) {
      for(x1=0; x1<LX; x1++) {
      for(x2=0; x2<LY; x2++) {
      for(x3=0; x3<LZ; x3++) {
        ix  = g_ipt[x0][x1][x2][x3];
        iix = g_lexic2eosub[ix];
        dtmp = 2. * M_PI * (
            (x1 + g_proc_coords[1]*LX) * g_source_momentum[0] / (double)LX_global +
            (x2 + g_proc_coords[2]*LY) * g_source_momentum[1] / (double)LY_global +
            (x3 + g_proc_coords[3]*LZ) * g_source_momentum[2] / (double)LZ_global );
        if(g_iseven[ix]) {
          phase_e[2*iix  ] = cos(dtmp);
          phase_e[2*iix+1] = sin(dtmp);
        } else {
          phase_o[2*iix  ] = cos(dtmp);
          phase_o[2*iix+1] = sin(dtmp);
        }
      }}}
    }

   sp_eo_e = (complex*)malloc(T*sizeof(complex));
   sp_eo_o = (complex*)malloc(T*sizeof(complex));
   sp_le_e = (complex*)malloc(T*sizeof(complex));
   sp_le_o = (complex*)malloc(T*sizeof(complex));
   if (  sp_eo_e == NULL || sp_eo_o == NULL ||  sp_le_e == NULL || sp_le_o == NULL ) {
     fprintf(stderr, "[test_sp] Error from malloc\n");
     EXIT(1);
   }

    for(isource_gamma_id=0; isource_gamma_id < g_source_gamma_id_number; isource_gamma_id++) {

      ratime = _GET_TIME;
      eo_spinor_dag_gamma_spinor ((complex*)buffer, eo_spinor_field[0], g_source_gamma_id_list[isource_gamma_id], eo_spinor_field[2]);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [test_sp_mom] time for eo_spinor_dag_gamma_spinor = %e\n", retime - ratime);

      ratime = _GET_TIME;
      eo_gsp_momentum_projection (sp_eo_e, (complex*)buffer, (complex*)phase_e, 0);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [test_sp_mom] time for eo_gsp_momentum_projection = %e\n", retime - ratime);

      ratime = _GET_TIME;
      eo_spinor_dag_gamma_spinor((complex*)buffer, eo_spinor_field[1], g_source_gamma_id_list[isource_gamma_id], eo_spinor_field[3]);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [test_sp_mom] time for eo_spinor_dag_gamma_spinor = %e\n", retime - ratime);

      ratime = _GET_TIME;
      eo_gsp_momentum_projection (sp_eo_o, (complex*)buffer, (complex*)phase_o, 1);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [test_sp_mom] time for eo_gsp_momentum_projection = %e\n", retime - ratime);


      /* lexic calculation */
      memset(sp_le_e, 0, T*sizeof(complex));
      memset(sp_le_o, 0, T*sizeof(complex));

      for(x0=0; x0<T; x0++) {
        for(x1=0; x1<LX; x1++) {
          for(x2=0; x2<LY; x2++) {
            for(x3=0; x3<LZ; x3++) {
              double spinor1[24];
              dtmp = 2 * M_PI * ( (x1+g_proc_coords[1]*LX) * g_source_momentum[0] / (double)LX_global +
                                  (x2+g_proc_coords[2]*LY) * g_source_momentum[1] / (double)LY_global +
                                  (x3+g_proc_coords[3]*LZ) * g_source_momentum[2] / (double)LZ_global );
              w2.re = cos(dtmp);
              w2.im = sin(dtmp);

              ix = g_ipt [x0][x1][x2][x3];
              _fv_eq_gamma_ti_fv(spinor1, g_source_gamma_id_list[isource_gamma_id], g_spinor_field[1]+_GSI(ix));
              _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), spinor1);
              _co_eq_co_ti_co(&w3, &w, &w2);

              if(g_iseven[ix]) {
                sp_le_e[x0].re += w3.re;
                sp_le_e[x0].im += w3.im;
              } else {
                sp_le_o[x0].re += w3.re;
                sp_le_o[x0].im += w3.im;
              }
            }
          }
        }
      }
#ifdef HAVE_MPI
      buffer2 = (double*)malloc(2*T*sizeof(double));
      if ( buffer2 == NULL ) {
        fprintf(stderr, "[test_sp_mom] Error from malloc\n");
        EXIT(2);
      }
      memcpy(buffer2, sp_le_e, 2*T*sizeof(double));
      MPI_Allreduce(buffer2, sp_le_e, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);

      memcpy(buffer2, sp_le_o, 2*T*sizeof(double));
      MPI_Allreduce(buffer2, sp_le_o, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);

      free(buffer2);
#endif

      /* compare */
      for(x0 = 0; x0 < T; x0++) {
        x1 = x0 + g_proc_coords[0] * T;
        fprintf(stdout, "e %2d\tp %3d %3d %3d\tg %2d\t%3d\t%25.16e%25.16e\t%25.16e%25.16e\n", g_cart_id, 
            g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], 
            g_source_gamma_id_list[isource_gamma_id], x1,
            sp_le_e[x0].re, sp_le_e[x0].im, sp_eo_e[x0].re, sp_eo_e[x0].im);
      }

      for(x0 = 0; x0 < T; x0++) {
        x1 = x0 + g_proc_coords[0] * T;
        fprintf(stdout, "o %2d\tp %3d %3d %3d\tg %2d\t%3d\t%25.16e%25.16e\t%25.16e%25.16e\n", g_cart_id,
            g_source_momentum[0], g_source_momentum[1], g_source_momentum[2],
            g_source_gamma_id_list[isource_gamma_id], x1, sp_le_o[x0].re, sp_le_o[x0].im, sp_eo_o[x0].re, sp_eo_o[x0].im);
      }

   }  /* end of loop on source gamma id */
  }   /* end of loop on source momenta */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  if(phase_e != NULL) free(phase_e);
  if(phase_o != NULL) free(phase_o);
  if(buffer  != NULL) free(buffer);

  if(g_gauge_field != NULL) free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_sp_mom] %s# [test_sp_mom] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_sp_mom] %s# [test_sp_mom] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

