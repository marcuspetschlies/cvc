/****************************************************
 * calculate_gsp.cpp
 *
 * Mo 30. Mai 17:32:05 CEST 2016
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
#include "Q_phi.h"
#include "scalar_products.h"
#include "gsp.h"

using namespace cvc;


const int g_gamma_adjoint_sign[16] = {
  /*   the sequence is:
   *        0, 1, 2, 3, id, 5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *          id 0  1  2  3   4  5    6    7    8    9   10   11   12   13   14   15
   *          */
       1, 1, 1, 1,  1, 1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1
};



void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status;
  int i, k, x0, ievecs, kevecs;
  int filename_set = 0;
  int isource_momentum, isource_gamma_id;
  int threadid, nthreads;
  int no_eo_fields;

  int evecs_num=0;
  double *evecs_eval = NULL;

  double norm, dtmp;
  double evecs_lambda;
  double *phase_e=NULL, *phase_o = NULL;
  double 
    *****gsp_V = NULL, *****gsp_W = NULL, *****gsp_XeobarV = NULL, *****gsp_XeoW = NULL;
  double *****gsp_buffer = NULL;
  double *buffer=NULL, *buffer2 = NULL;
  complex w;

  double plaq=0.;
  int verbose = 0;
  char filename[200];

  FILE *ofs=NULL;
  size_t items, bytes;

  double **eo_spinor_field=NULL, *eo_spinor_work=NULL, *eo_spinor_work2 = NULL, *eo_spinor_work3=NULL;
  double ratime, retime, momentum_ratime, momentum_retime, gamma_ratime, gamma_retime;
  unsigned int Vhalf;

  int check_eigenvectors = 0;

  // in order to be able to initialise QMP if QPhiX is used, we need
  // to allow tmLQCD to intialise MPI via QMP
  // the input file has to be read 
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_init_parallel_and_read_input(argc, argv, 1, "invert.input");
#else
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
#endif

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
  uint32_t aff_buffer_size;
#endif


  while ((c = getopt(argc, argv, "ch?vf:n:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'n':
      evecs_num = atoi(optarg);
      fprintf(stdout, "# [calculate_gsp] dimension of eigenspace set to%d\n", evecs_num);
      break;
    case 'c':
      check_eigenvectors = 1;
      fprintf(stdout, "# [calculate_gsp] check eigenvectors set to %d\n", check_eigenvectors);
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
  if(g_cart_id==0) fprintf(stdout, "# [calculate_gsp] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [calculate_gsp] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  status = tmLQCD_invert_init(argc, argv, 1, 0);
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
    fprintf(stderr, "[calculate_gsp] Error from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [calculate_gsp] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[calculate_gsp] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif




  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[calculate_gsp] eigenspace dimension is 0\n");
    EXIT(1);
  }
  evecs_eval = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[calculate_gsp] Error from malloc\n");
    EXIT(117);
  }

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [calculate_gsp] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
#ifdef HAVE_MPI
  xchange_gauge();
#endif
  } else if(strcmp(gaugefilename_prefix, "NA") != 0) {
    /* status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq); */
    /* status = read_ildg_nersc_gauge_field(g_gauge_field, filename); */
    status = read_lime_gauge_field_doubleprec(filename);
    /* status = read_nersc_gauge_field(g_gauge_field, filename, &plaq); */
#ifdef HAVE_MPI
    xchange_gauge();
#endif
    if(status != 0) {
      fprintf(stderr, "[calculate_gsp] Error, could not read gauge field\n");
      EXIT(11);
    }

    /* measure the plaquette */
    if(g_cart_id==0) fprintf(stdout, "# [calculate_gsp] read plaquette value 1st field: %25.16e\n", plaq);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# [calculate_gsp] measured plaquette value 1st field: %25.16e\n", plaq);
  } else {
    if(g_cart_id == 0) fprintf(stderr, "# [calculate_gsp] need gauge field\n");
    EXIT(20);
  }

  /***********************************************
   * allocate spinor fields
   ***********************************************/
  no_fields = 1;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  /* no_eo_fields = 2*evecs_num + 3; */
  no_eo_fields = evecs_num + 3;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields; i++) alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);
  eo_spinor_work  = eo_spinor_field[no_eo_fields - 3];
  eo_spinor_work2 = eo_spinor_field[no_eo_fields - 2];
  eo_spinor_work3 = eo_spinor_field[no_eo_fields - 1];

  /***********************************************
   * allocate gsp space
   ***********************************************/
  status = gsp_init (&gsp_V, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[calculate_gsp] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  status = gsp_init (&gsp_W, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[calculate_gsp] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  status = gsp_init (&gsp_XeobarV, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[calculate_gsp] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  status = gsp_init (&gsp_XeoW, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[calculate_gsp] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }


  /***********************************************
   * read eo eigenvectors
   ***********************************************/
  for(ievecs = 0; ievecs<evecs_num; ievecs+=2) {
    
    sprintf(filename, "%s%.5d", filename_prefix, ievecs);
    if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] reading C_oo_sym eigenvector from file %s\n", filename);

    status = read_lime_spinor(g_spinor_field[0], filename, 0);
    if( status != 0) {
      fprintf(stderr, "[calculate_gsp] Error from read_lime_spinor, status was %d\n", status);
      EXIT(1);
    }

    ratime = _GET_TIME;
    spinor_field_unpack_lexic2eo (g_spinor_field[0], eo_spinor_field[ievecs], eo_spinor_field[ievecs+1]);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for unpacking = %e\n", retime - ratime);

  }  /* end of loop on evecs number */

  if (check_eigenvectors) {

    /***********************************************
     * check eigenvector equation
     ***********************************************/
    for(ievecs = 0; ievecs<evecs_num; ievecs++) {
  
      ratime = _GET_TIME;
#ifdef HAVE_MPI
      xchange_eo_field(eo_spinor_field[ievecs], 1);
#endif
      C_oo(eo_spinor_work, eo_spinor_field[ievecs], g_gauge_field, -g_mu, eo_spinor_work3);
#ifdef HAVE_MPI
      xchange_eo_field( eo_spinor_work, 1);
#endif
      C_oo(eo_spinor_work2, eo_spinor_work, g_gauge_field,  g_mu, eo_spinor_work3);
  
      norm = 4 * g_kappa * g_kappa;
      spinor_field_ti_eq_re (eo_spinor_work2, norm, Vhalf);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time to apply C_oo_sym = %e\n", retime - ratime);
  
      /* determine eigenvalue */
      spinor_scalar_product_re(&norm,  eo_spinor_field[ievecs], eo_spinor_field[ievecs], Vhalf);
      spinor_scalar_product_co(&w,  eo_spinor_field[ievecs], eo_spinor_work2, Vhalf);
      evecs_lambda = w.re / norm;
      if(g_cart_id == 0) {
        fprintf(stdout, "# [calculate_gsp] estimated eigenvalue(%d) = %25.16e\n", ievecs, evecs_lambda);
      }
  
      /* check evec equation */
      spinor_field_mi_eq_spinor_field_ti_re(eo_spinor_work2, eo_spinor_field[ievecs], evecs_lambda, Vhalf);

      spinor_scalar_product_re(&norm, eo_spinor_work2, eo_spinor_work2, Vhalf);
      if(g_cart_id == 0) {
        fprintf(stdout, "# [calculate_gsp] eigenvector(%d) || A x - lambda x || = %16.7e\n", ievecs, sqrt(norm) );
      }
    }  /* end of loop on evecs_num */

  }  /* end of if check_eigenvectors */


  /***********************************************
   * even/odd phase field for Fourier phase
   ***********************************************/
  phase_e = (double*)malloc(Vhalf*2*sizeof(double));
  phase_o = (double*)malloc(Vhalf*2*sizeof(double));
  if(phase_e == NULL || phase_o == NULL) {
    fprintf(stderr, "[calculate_gsp] Error from malloc\n");
    EXIT(14);
  }

  buffer = (double*)malloc(2*Vhalf*sizeof(double));
  if(buffer == NULL)  {
    fprintf(stderr, "[calculate_gsp] Error from malloc\n");
    EXIT(19);
  }

  buffer2 = (double*)malloc(2*T*sizeof(double));
  if(buffer2 == NULL)  {
    fprintf(stderr, "[calculate_gsp] Error from malloc\n");
    EXIT(20);
  }


  /***********************************************************************************************
   ***********************************************************************************************
   **
   ** calculate gsp_V
   **
   ***********************************************************************************************
   ***********************************************************************************************/

  /***********************************************
   * output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {

    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [calculate_gsp] using aff version %s\n", aff_status_str);

    
    sprintf(filename, "gsp_V.%.4d.aff", Nconf);
    fprintf(stdout, "# [calculate_gsp] writing correlator data from file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_reader, status was %s\n", aff_status_str);
      EXIT(102);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[calculate_gsp] Error, aff writer is not initialized\n");
      EXIT(103);
    }

    if(g_cart_id == 0) {
      aff_buffer = (double _Complex*)malloc(2*evecs_num*evecs_num*sizeof(double _Complex));
      if(aff_buffer == NULL) {
        fprintf(stderr, "[calculate_gsp] Error from malloc\n");
        EXIT(22);
      }
    }

  }
#endif

  for(isource_momentum=0; isource_momentum < g_source_momentum_number; isource_momentum++) {

    g_source_momentum[0] = g_source_momentum_list[isource_momentum][0];
    g_source_momentum[1] = g_source_momentum_list[isource_momentum][1];
    g_source_momentum[2] = g_source_momentum_list[isource_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [calculate_gsp] using source momentum = (%d, %d, %d)\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    }
    momentum_ratime = _GET_TIME;

    /* make phase field in eo ordering */
    gsp_make_eo_phase_field (phase_e, phase_o, g_source_momentum);

    for(isource_gamma_id=0; isource_gamma_id < g_source_gamma_id_number; isource_gamma_id++) {
      if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] using source gamma id %d\n", g_source_gamma_id_list[isource_gamma_id]);
      gamma_ratime = _GET_TIME;

      /* loop on eigenvectors */
      for(ievecs = 0; ievecs<evecs_num; ievecs++) {
        for(kevecs = ievecs; kevecs<evecs_num; kevecs++) {

          ratime = _GET_TIME;
          eo_spinor_dag_gamma_spinor((complex*)buffer, eo_spinor_field[ievecs], g_source_gamma_id_list[isource_gamma_id], eo_spinor_field[kevecs]);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_spinor_dag_gamma_spinor = %e seconds\n", retime - ratime);

          ratime = _GET_TIME;
          eo_gsp_momentum_projection ((complex*)buffer2, (complex*)buffer, (complex*)phase_o, 1);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_gsp_momentum_projection = %e seconds\n", retime - ratime);

          for(x0=0; x0<T; x0++) {
            memcpy(gsp_V[isource_momentum][isource_gamma_id][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
          }

        }
      }

      /* gsp[k,i] = sigma_Gamma gsp[i,k]^+ */
      for(x0=0; x0<T; x0++) {
        for(ievecs = 0; ievecs<evecs_num-1; ievecs++) {
          for(kevecs = ievecs+1; kevecs<evecs_num; kevecs++) {
            gsp_V[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs  ] =  g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp_V[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs  ];
            gsp_V[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs+1] = -g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp_V[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs+1];
          }
        }
      }

      /***********************************************
       * write gsp to disk
       ***********************************************/

#ifdef HAVE_MPI

      status = gsp_init (&gsp_buffer, 1, 1, T_global, evecs_num);
      if(gsp_buffer == NULL) {
        fprintf(stderr, "[calculate_gsp] Error from gsp_init\n");
        EXIT(18);
      }
      k = 2*T*evecs_num*evecs_num; /* number of items to be sent and received */

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)

      fprintf(stdout, "# [calculate_gsp] proc%.2d g_tr_id = %d; g_tr_nproc =%d\n", g_cart_id, g_tr_id, g_tr_nproc);
      MPI_Allgather(gsp_V[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, g_tr_comm);
  
#else
      /* collect at 0 from all times */
      MPI_Gather(gsp_V[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, 0, g_cart_grid);
#endif

#else
      gsp_buffer = gsp_V;
#endif  /* of ifdef HAVE_MPI */

      if(g_cart_id == 0) {
#ifdef HAVE_LHPC_AFF

        for(x0=0; x0<T_global; x0++) {
          sprintf(aff_buffer_path, "/px%.2dpy%.2dpz%.2d/g%.2d/t%.2d", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], g_source_gamma_id_list[isource_gamma_id], x0);
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] current aff path = %s\n", aff_buffer_path);

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          items = evecs_num*evecs_num;
          memcpy(aff_buffer, gsp_buffer[0][0][x0][0], 2*items*sizeof(double));
          status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items); 
          if(status != 0) {
            fprintf(stderr, "[calculate_gsp] Error  from aff_node_put_double, status was %d\n", status);
            EXIT(104);
          }
        }  /* end of loop on x0 */
#else

        sprintf(filename, "gsp_V.%.4d.px%.2dpy%.2dpz%.2d.g%.2d", 
            Nconf, g_source_momentum[0], g_source_momentum[1], g_source_momentum[2],
            g_source_gamma_id_list[isource_gamma_id]);
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[calculate_gsp] Error, could not open file %s for writing\n", filename);
          EXIT(103);
        }
        items = 2 * (size_t)T * evecs_num*evecs_num;
        if( fwrite(gsp_buffer[0][0][0][0], sizeof(double), items, ofs) != items ) {
          fprintf(stderr, "[calculate_gsp] Error, could not write proper amount of data to file %s\n", filename);
          EXIT(104);
        }
        fclose(ofs);

#endif
      }  /* end of if g_cart_id == 0 */


#ifdef HAVE_MPI
      gsp_fini(&gsp_buffer);
#else
      gsp_buffer = NULL;
#endif

     gamma_retime = _GET_TIME;
     if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for gamma id %d = %e seconds\n", g_source_gamma_id_list[isource_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on source gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for momentum (%d, %d, %d) = %e seconds\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], momentum_retime - momentum_ratime);

  }   /* end of loop on source momenta */

  /***********************************************
   * close the output files
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(104);
    }
  }
#endif


  /***********************************************************************************************
   ***********************************************************************************************
   **
   ** calculate gsp_XeobarV
   **
   ***********************************************************************************************
   ***********************************************************************************************/

  /***********************************************
   * output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {

    sprintf(filename, "gsp_XeobarV.%.4d.aff", Nconf);
    fprintf(stdout, "# [calculate_gsp] writing correlator data from file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_reader, status was %s\n", aff_status_str);
      EXIT(102);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[calculate_gsp] Error, aff writer is not initialized\n");
      EXIT(103);
    }
  }
#endif

  /***********************************************
   * calculate Xeobar V
   ***********************************************/
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    X_eo (eo_spinor_field[evecs_num+ievecs], eo_spinor_field[ievecs], -g_mu, g_gauge_field);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for X_eo = %e seconds\n", retime-ratime);
  }
  
  for(isource_momentum=0; isource_momentum < g_source_momentum_number; isource_momentum++) {

    g_source_momentum[0] = g_source_momentum_list[isource_momentum][0];
    g_source_momentum[1] = g_source_momentum_list[isource_momentum][1];
    g_source_momentum[2] = g_source_momentum_list[isource_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [calculate_gsp] using source momentum = (%d, %d, %d)\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    }
    momentum_ratime = _GET_TIME;

    /* make phase field in eo ordering */
    gsp_make_eo_phase_field (phase_e, phase_o, g_source_momentum);

    for(isource_gamma_id=0; isource_gamma_id < g_source_gamma_id_number; isource_gamma_id++) {
      if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] using source gamma id %d\n", g_source_gamma_id_list[isource_gamma_id]);
      gamma_ratime = _GET_TIME;

      /* loop on eigenvectors */
      for(ievecs = 0; ievecs<evecs_num; ievecs++) {
        for(kevecs = ievecs; kevecs<evecs_num; kevecs++) {

          ratime = _GET_TIME;
          eo_spinor_dag_gamma_spinor((complex*)buffer, eo_spinor_field[evecs_num+ievecs], g_source_gamma_id_list[isource_gamma_id], eo_spinor_field[evecs_num+kevecs]);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_spinor_dag_gamma_spinor = %e\n", retime - ratime);

          ratime = _GET_TIME;
          eo_gsp_momentum_projection ((complex*)buffer2, (complex*)buffer, (complex*)phase_o, 1);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_gsp_momentum_projection = %e\n", retime - ratime);

          for(x0=0; x0<T; x0++) {
            memcpy(gsp_XeobarV[isource_momentum][isource_gamma_id][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
          }

        }
      }

      /* gsp[k,i] = sigma_Gamma gsp[i,k]^+ */
      for(x0=0; x0<T; x0++) {
        for(ievecs = 0; ievecs<evecs_num-1; ievecs++) {
          for(kevecs = ievecs+1; kevecs<evecs_num; kevecs++) {
            gsp_XeobarV[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs  ] =  g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp_XeobarV[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs  ];
            gsp_XeobarV[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs+1] = -g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp_XeobarV[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs+1];
          }
        }
      }

      /***********************************************
       * write gsp to disk
       ***********************************************/
#ifdef HAVE_MPI
      status = gsp_init (&gsp_buffer, 1, 1, T_global, evecs_num);
      if(gsp_buffer == NULL) {
        fprintf(stderr, "[calculate_gsp] Error from gsp_init\n");
        EXIT(18);
      }
      k = 2*T*evecs_num*evecs_num; /* number of items to be sent and received */

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)

      fprintf(stdout, "# [calculate_gsp] proc%.2d g_tr_id = %d; g_tr_nproc =%d\n", g_cart_id, g_tr_id, g_tr_nproc);
      MPI_Allgather(gsp_XeobarV[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, g_tr_comm);
  
#else
      /* collect at 0 from all times */
      MPI_Gather(gsp_XeobarV[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, 0, g_cart_grid);
#endif

#else
      gsp_buffer = gsp_XeobarV;
#endif  /* of ifdef HAVE_MPI */

      if(g_cart_id == 0) {
#ifdef HAVE_LHPC_AFF

        for(x0=0; x0<T_global; x0++) {
          sprintf(aff_buffer_path, "/px%.2dpy%.2dpz%.2d/g%.2d/t%.2d", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], g_source_gamma_id_list[isource_gamma_id], x0);
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] current aff path = %s\n", aff_buffer_path);

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          items = evecs_num*evecs_num;
          memcpy(aff_buffer, gsp_buffer[0][0][x0][0], 2*items*sizeof(double));
          status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items); 
          if(status != 0) {
            fprintf(stderr, "[calculate_gsp] Error  from aff_node_put_double, status was %d\n", status);
            EXIT(104);
          }
        }  /* end of loop on x0 */
#else

        sprintf(filename, "gsp_XeobarV.%.4d.px%.2dpy%.2dpz%.2d.g%.2d", 
            Nconf, g_source_momentum[0], g_source_momentum[1], g_source_momentum[2],
            g_source_gamma_id_list[isource_gamma_id]);
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[calculate_gsp] Error, could not open file %s for writing\n", filename);
          EXIT(103);
        }
        items = 2 * (size_t)T * evecs_num*evecs_num;
        if( fwrite(gsp_buffer[0][0][0][0], sizeof(double), items, ofs) != items ) {
          fprintf(stderr, "[calculate_gsp] Error, could not write proper amount of data to file %s\n", filename);
          EXIT(104);
        }
        fclose(ofs);

#endif
      }  /* end of if g_cart_id == 0 */

#ifdef HAVE_MPI
      gsp_fini(&gsp_buffer);
#else
      gsp_buffer = NULL;
#endif

     gamma_retime = _GET_TIME;
     if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for gamma id %d = %e seconds\n", g_source_gamma_id_list[isource_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on source gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for momentum (%d, %d, %d) = %e seconds\n",
       g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], momentum_retime - momentum_ratime);

  }   /* end of loop on source momenta */

  /***********************************************
   * close the output files
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(104);
    }
  }
  /* if(aff_status_str != NULL) free(aff_status_str); */
#endif

  /***********************************************************************************************
   ***********************************************************************************************
   **
   ** calculate gsp_W
   **
   ***********************************************************************************************
   ***********************************************************************************************/

  /***********************************************
   * output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {

    sprintf(filename, "gsp_W.%.4d.aff", Nconf);
    fprintf(stdout, "# [calculate_gsp] writing correlator data from file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_reader, status was %s\n", aff_status_str);
      EXIT(102);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[calculate_gsp] Error, aff writer is not initialized\n");
      EXIT(103);
    }
  }
#endif

  /***********************************************
   * calculate W
   ***********************************************/
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    C_from_Xeo (eo_spinor_work, eo_spinor_field[evecs_num+ievecs], eo_spinor_field[ievecs], g_gauge_field, -g_mu);
    /* memcpy( eo_spinor_field[ievecs], eo_spinor_work, 24*Vhalf*sizeof(double) ); */
    spinor_scalar_product_re(&norm, eo_spinor_work, eo_spinor_work, Vhalf);
    evecs_eval[ievecs] = norm  * 4.*g_kappa*g_kappa;
    norm = 1./sqrt( norm );
    spinor_field_eq_spinor_field_ti_re (eo_spinor_field[ievecs],  eo_spinor_work, norm, Vhalf);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for C_from_Xeo = %e seconds\n", retime-ratime);
  }

  /* TEST */
  if(g_cart_id == 0) {
    for(ievecs = 0; ievecs<evecs_num; ievecs++) {
      fprintf(stdout, "# [calculate_gsp] eval %4d %25.16e\n", ievecs, evecs_eval[ievecs]);
    }
  }
  
  for(isource_momentum=0; isource_momentum < g_source_momentum_number; isource_momentum++) {

    g_source_momentum[0] = g_source_momentum_list[isource_momentum][0];
    g_source_momentum[1] = g_source_momentum_list[isource_momentum][1];
    g_source_momentum[2] = g_source_momentum_list[isource_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [calculate_gsp] using source momentum = (%d, %d, %d)\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    }
    momentum_ratime = _GET_TIME;

    /* make phase field in eo ordering */
    gsp_make_eo_phase_field (phase_e, phase_o, g_source_momentum);

    for(isource_gamma_id=0; isource_gamma_id < g_source_gamma_id_number; isource_gamma_id++) {
      if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] using source gamma id %d\n", g_source_gamma_id_list[isource_gamma_id]);
      gamma_ratime = _GET_TIME;

      /* loop on eigenvectors */
      for(ievecs = 0; ievecs<evecs_num; ievecs++) {
        for(kevecs = ievecs; kevecs<evecs_num; kevecs++) {

          ratime = _GET_TIME;
          eo_spinor_dag_gamma_spinor((complex*)buffer, eo_spinor_field[ievecs], g_source_gamma_id_list[isource_gamma_id], eo_spinor_field[kevecs]);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_spinor_dag_gamma_spinor = %e\n", retime - ratime);

          ratime = _GET_TIME;
          eo_gsp_momentum_projection ((complex*)buffer2, (complex*)buffer, (complex*)phase_o, 1);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_gsp_momentum_projection = %e\n", retime - ratime);

          for(x0=0; x0<T; x0++) {
            memcpy(gsp_W[isource_momentum][isource_gamma_id][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
          }

        }
      }

      /* gsp[k,i] = sigma_Gamma gsp[i,k]^+ */
      for(x0=0; x0<T; x0++) {
        for(ievecs = 0; ievecs<evecs_num-1; ievecs++) {
          for(kevecs = ievecs+1; kevecs<evecs_num; kevecs++) {
            gsp_W[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs  ] =  g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp_W[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs  ];
            gsp_W[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs+1] = -g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp_W[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs+1];
          }
        }
      }

      /***********************************************
       * write gsp to disk
       ***********************************************/
#ifdef HAVE_MPI
      status = gsp_init (&gsp_buffer, 1, 1, T_global, evecs_num);
      if(gsp_buffer == NULL) {
        fprintf(stderr, "[calculate_gsp] Error from gsp_init\n");
        EXIT(18);
      }
      k = 2*T*evecs_num*evecs_num; /* number of items to be sent and received */

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)

      fprintf(stdout, "# [calculate_gsp] proc%.2d g_tr_id = %d; g_tr_nproc =%d\n", g_cart_id, g_tr_id, g_tr_nproc);
      MPI_Allgather(gsp_W[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, g_tr_comm);
  
#else
      /* collect at 0 from all times */
      MPI_Gather(gsp_W[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, 0, g_cart_grid);
#endif

#else
      gsp_buffer = gsp_W;
#endif  /* of ifdef HAVE_MPI */

      if(g_cart_id == 0) {
#ifdef HAVE_LHPC_AFF

        for(x0=0; x0<T_global; x0++) {
          sprintf(aff_buffer_path, "/px%.2dpy%.2dpz%.2d/g%.2d/t%.2d", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], g_source_gamma_id_list[isource_gamma_id], x0);
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] current aff path = %s\n", aff_buffer_path);

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          items = evecs_num*evecs_num;
          memcpy(aff_buffer, gsp_buffer[0][0][x0][0], 2*items*sizeof(double));
          status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items); 
          if(status != 0) {
            fprintf(stderr, "[calculate_gsp] Error  from aff_node_put_double, status was %d\n", status);
            EXIT(104);
          }
        }  /* end of loop on x0 */
#else

        sprintf(filename, "gsp_W.%.4d.px%.2dpy%.2dpz%.2d.g%.2d", 
            Nconf, g_source_momentum[0], g_source_momentum[1], g_source_momentum[2],
            g_source_gamma_id_list[isource_gamma_id]);
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[calculate_gsp] Error, could not open file %s for writing\n", filename);
          EXIT(103);
        }
        items = 2 * (size_t)T * evecs_num*evecs_num;
        if( fwrite(gsp_buffer[0][0][0][0], sizeof(double), items, ofs) != items ) {
          fprintf(stderr, "[calculate_gsp] Error, could not write proper amount of data to file %s\n", filename);
          EXIT(104);
        }
        fclose(ofs);

#endif
      }  /* end of if g_cart_id == 0 */

#ifdef HAVE_MPI
      gsp_fini(&gsp_buffer);
#else
      gsp_buffer = NULL;
#endif

     gamma_retime = _GET_TIME;
     if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for gamma id %d = %e seconds\n", g_source_gamma_id_list[isource_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on source gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for momentum (%d, %d, %d) = %e seconds\n",
       g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], momentum_retime - momentum_ratime);

  }   /* end of loop on source momenta */

  /***********************************************
   * close the output files
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(104);
    }
  }
  /* if(aff_status_str != NULL) free(aff_status_str); */
#endif


  /***********************************************************************************************
   ***********************************************************************************************
   **
   ** calculate gsp_XeoW
   **
   ***********************************************************************************************
   ***********************************************************************************************/

  /***********************************************
   * output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {

    sprintf(filename, "gsp_XeoW.%.4d.aff", Nconf);
    fprintf(stdout, "# [calculate_gsp] writing correlator data from file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_reader, status was %s\n", aff_status_str);
      EXIT(102);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[calculate_gsp] Error, aff writer is not initialized\n");
      EXIT(103);
    }
  }
#endif

  /***********************************************
   * calculate Xeobar V
   ***********************************************/
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    X_eo (eo_spinor_field[evecs_num+ievecs], eo_spinor_field[ievecs], g_mu, g_gauge_field);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for X_eo = %e seconds\n", retime-ratime);
  }
  
  for(isource_momentum=0; isource_momentum < g_source_momentum_number; isource_momentum++) {

    g_source_momentum[0] = g_source_momentum_list[isource_momentum][0];
    g_source_momentum[1] = g_source_momentum_list[isource_momentum][1];
    g_source_momentum[2] = g_source_momentum_list[isource_momentum][2];

    if(g_cart_id == 0) {
      fprintf(stdout, "# [calculate_gsp] using source momentum = (%d, %d, %d)\n", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2]);
    }
    momentum_ratime = _GET_TIME;

    /* make phase field in eo ordering */
    gsp_make_eo_phase_field (phase_e, phase_o, g_source_momentum);

    for(isource_gamma_id=0; isource_gamma_id < g_source_gamma_id_number; isource_gamma_id++) {
      if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] using source gamma id %d\n", g_source_gamma_id_list[isource_gamma_id]);
      gamma_ratime = _GET_TIME;

      /* loop on eigenvectors */
      for(ievecs = 0; ievecs<evecs_num; ievecs++) {
        for(kevecs = ievecs; kevecs<evecs_num; kevecs++) {

          ratime = _GET_TIME;
          eo_spinor_dag_gamma_spinor((complex*)buffer, eo_spinor_field[evecs_num+ievecs], g_source_gamma_id_list[isource_gamma_id], eo_spinor_field[evecs_num+kevecs]);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_spinor_dag_gamma_spinor = %e\n", retime - ratime);

          ratime = _GET_TIME;
          eo_gsp_momentum_projection ((complex*)buffer2, (complex*)buffer, (complex*)phase_o, 1);
          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for eo_gsp_momentum_projection = %e\n", retime - ratime);

          for(x0=0; x0<T; x0++) {
            memcpy(gsp_XeoW[isource_momentum][isource_gamma_id][x0][ievecs] + 2*kevecs, buffer2+2*x0, 2*sizeof(double));
          }

        }
      }

      /* gsp[k,i] = sigma_Gamma gsp[i,k]^+ */
      for(x0=0; x0<T; x0++) {
        for(ievecs = 0; ievecs<evecs_num-1; ievecs++) {
          for(kevecs = ievecs+1; kevecs<evecs_num; kevecs++) {
            gsp_XeoW[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs  ] =  g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp_XeoW[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs  ];
            gsp_XeoW[isource_momentum][isource_gamma_id][x0][kevecs][2*ievecs+1] = -g_gamma_adjoint_sign[g_source_gamma_id_list[isource_gamma_id]] * gsp_XeoW[isource_momentum][isource_gamma_id][x0][ievecs][2*kevecs+1];
          }
        }
      }

      /***********************************************
       * write gsp to disk
       ***********************************************/
#ifdef HAVE_MPI
      status = gsp_init (&gsp_buffer, 1, 1, T_global, evecs_num);
      if(gsp_buffer == NULL) {
        fprintf(stderr, "[calculate_gsp] Error from gsp_init\n");
        EXIT(18);
      }
      k = 2*T*evecs_num*evecs_num; /* number of items to be sent and received */

#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)

      fprintf(stdout, "# [calculate_gsp] proc%.2d g_tr_id = %d; g_tr_nproc =%d\n", g_cart_id, g_tr_id, g_tr_nproc);
      MPI_Allgather(gsp_XeoW[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, g_tr_comm);
  
#else
      /* collect at 0 from all times */
      MPI_Gather(gsp_XeoW[isource_momentum][isource_gamma_id][0][0], k, MPI_DOUBLE, gsp_buffer[0][0][0][0], k, MPI_DOUBLE, 0, g_cart_grid);
#endif

#else
      gsp_buffer = gsp_XeoW;
#endif  /* of ifdef HAVE_MPI */

      if(g_cart_id == 0) {
#ifdef HAVE_LHPC_AFF

        for(x0=0; x0<T_global; x0++) {
          sprintf(aff_buffer_path, "/px%.2dpy%.2dpz%.2d/g%.2d/t%.2d", g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], g_source_gamma_id_list[isource_gamma_id], x0);
          if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] current aff path = %s\n", aff_buffer_path);

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          items = evecs_num*evecs_num;
          memcpy(aff_buffer, gsp_buffer[0][0][x0][0], 2*items*sizeof(double));
          status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items); 
          if(status != 0) {
            fprintf(stderr, "[calculate_gsp] Error  from aff_node_put_double, status was %d\n", status);
            EXIT(104);
          }
        }  /* end of loop on x0 */
#else

        sprintf(filename, "gsp_XeoW.%.4d.px%.2dpy%.2dpz%.2d.g%.2d", 
            Nconf, g_source_momentum[0], g_source_momentum[1], g_source_momentum[2],
            g_source_gamma_id_list[isource_gamma_id]);
        ofs = fopen(filename, "w");
        if(ofs == NULL) {
          fprintf(stderr, "[calculate_gsp] Error, could not open file %s for writing\n", filename);
          EXIT(103);
        }
        items = 2 * (size_t)T * evecs_num*evecs_num;
        if( fwrite(gsp_buffer[0][0][0][0], sizeof(double), items, ofs) != items ) {
          fprintf(stderr, "[calculate_gsp] Error, could not write proper amount of data to file %s\n", filename);
          EXIT(104);
        }
        fclose(ofs);

#endif
      }  /* end of if g_cart_id == 0 */

#ifdef HAVE_MPI
      gsp_fini(&gsp_buffer);
#else
      gsp_buffer = NULL;
#endif

     gamma_retime = _GET_TIME;
     if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for gamma id %d = %e seconds\n", g_source_gamma_id_list[isource_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on source gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [calculate_gsp] time for momentum (%d, %d, %d) = %e seconds\n",
       g_source_momentum[0], g_source_momentum[1], g_source_momentum[2], momentum_retime - momentum_ratime);

  }   /* end of loop on source momenta */

  /***********************************************
   * close the output files
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[calculate_gsp] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(104);
    }
  }
  /* if(aff_status_str != NULL) free(aff_status_str); */
#endif


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  if(phase_e != NULL) free(phase_e);
  if(phase_o != NULL) free(phase_o);
  if(buffer  != NULL) free(buffer);
  if(buffer2 != NULL) free(buffer2);
  if(evecs_eval != NULL) free(evecs_eval);

  if(g_gauge_field != NULL) free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);

  gsp_fini(&gsp_V);
  gsp_fini(&gsp_XeobarV);
  gsp_fini(&gsp_W);
  gsp_fini(&gsp_XeoW);

#ifdef HAVE_LHPC_AFF
  if(aff_buffer != NULL) free(aff_buffer);
#endif

  free_geometry();


#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [calculate_gsp] %s# [calculate_gsp] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [calculate_gsp] %s# [calculate_gsp] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

