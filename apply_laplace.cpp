/****************************************************
 * apply_laplace.c
 *
 * Sun May 31 17:05:41 CEST 2015
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

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "laplace_linalg.h"
#include "hyp_smear.h"

using namespace cvc {

void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu, nu, status, sid;
  int i, j, ncon=-1, ir, is, ic, id, idx;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  int threadid, nthreads;
  int ev_number = 0;
  double dtmp[4], norm, norm2, norm3;

  double plaq=0.;
  double *gauge_field_smeared = NULL;
  int verbose = 0;
  char filename[200];
  FILE *ofs=NULL;
  double **ev_field=NULL, *ev_field2=NULL, *ev_values=NULL, *ev_field3=NULL, *ev_phases=NULL;
  unsigned int VOL3;
  double v1[6], v2[6];
  size_t items, bytes;
  complex w, w1, w2;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:n:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'n':
      ev_number = atoi(optarg);
      fprintf(stdout, "# [apply_laplace] number of eigenvalues = %d\n", ev_number);
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
  if(g_cart_id==0) fprintf(stdout, "# [apply_laplace] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "# [apply_laplace] T and L's must be set\n");
    usage();
  }

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize T etc. */
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",\
		  g_cart_id, g_cart_id, T_global, g_cart_id, T, g_cart_id, Tstart,
		             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
		             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "[apply_laplace] Error from init_geometry\n");
    exit(101);
  }

  geometry();

  VOL3 = LX*LY*LZ;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [apply_laplace] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    status = read_lime_gauge_field_doubleprec(filename);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
  }
  if(status != 0) {
    fprintf(stderr, "[apply_laplace] Error, could not read gauge field\n");
    exit(11);
  }
  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# [apply_laplace] read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [apply_laplace] measured plaquette value 1st field: %25.16e\n", plaq);

#if 0
  /* smear the gauge field */
  status = hyp_smear_3d (g_gauge_field, N_hyp, alpha_hyp, 0, 0);
  if(status != 0) {
    fprintf(stderr, "[apply_laplace] Error from hyp_smear_3d, status was %d\n", status);
    EXIT(7);
  }

  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [apply_laplace] measured plaquette value ofter hyp smearing = %25.16e\n", plaq);

  sprintf(filename, "%s_hyp.%.4d", gaugefilename_prefix, Nconf);
  fprintf(stdout, "# [apply_laplace] writing hyp-smeared gauge field to file %s\n", filename);

  status = write_lime_gauge_field(filename, plaq, Nconf, 64);
  if(status != 0) {
    fprintf(stderr, "[apply_lapace] Error friom write_lime_gauge_field, status was %d\n", status);
    EXIT(7);
  }
#endif

  ev_field = (double**)malloc(ev_number*sizeof(double*));
  if(ev_field == NULL) {
    fprintf(stderr, "[apply_laplace] Error from malloc\n");
    EXIT(1);
  }

  ev_field[0]  = (double*)malloc(ev_number * 6 * VOL3*sizeof(double));
  if(ev_field[0] == NULL) {
    fprintf(stderr, "[apply_laplace] Error from malloc\n");
    EXIT(2);
  }
  for(i=1; i<ev_number; i++) ev_field[i] = ev_field[i-1] + 6*VOL3;

  ev_field2 = (double*)malloc(6*VOL3*sizeof(double));
  if(ev_field2 == NULL) {
    fprintf(stderr, "[apply_laplace] Error from malloc\n");
    EXIT(3);
  }

  ev_field3 = (double*)malloc(6*VOL3*sizeof(double));
  if(ev_field3 == NULL) {
    fprintf(stderr, "[apply_laplace] Error from malloc\n");
    EXIT(6);
  }

  ev_values = (double*)malloc(ev_number*sizeof(double));
  if(ev_values == NULL) {
    fprintf(stderr, "[apply_laplace] Error from malloc\n");
    EXIT(5);
  }

  ev_phases = (double*)malloc(ev_number*sizeof(double));
  if(ev_phases == NULL) {
    fprintf(stderr, "[apply_laplace] Error from malloc\n");
    EXIT(5);
  }

  /* for(x0=0; x0<T; x0++) */
  for(x0=0; x0<1; x0++)
  {
    sprintf(filename, "%s/eigenvectors.%.4d.%.3d", filename_prefix, Nconf, x0);
    fprintf(stdout, "# [apply_laplace] reading eigenvectors from file %s\n", filename);
    ofs = fopen(filename, "r");
    if(ofs == NULL) {
      fprintf(stderr, "[apply_laplace] Error from fopen for filename %s\n", filename);
      EXIT(4);
    }
    items = fread (ev_field[0], sizeof(double), (size_t)ev_number*VOL3*6, ofs );
    if(items != (size_t)ev_number*VOL3*6) {
      fprintf(stderr, "[apply_laplace] Error, number of items read is %lu\n", items);
      EXIT(5);
    }
    fclose(ofs);
    ofs = NULL;

    sprintf(filename, "%s/eigenvalues.%.4d.%.3d", filename_prefix, Nconf, x0);
    fprintf(stdout, "# [apply_laplace] reading eigenvalues from file %s\n", filename);
    ofs = fopen(filename, "r");
    if(ofs == NULL) {
      fprintf(stderr, "[apply_laplace] Error from fopen for filename %s\n", filename);
      EXIT(4);
    }
    items = fread (ev_values, sizeof(double), (size_t)ev_number, ofs );
    if(items != (size_t)ev_number) {
      fprintf(stderr, "[apply_laplace] Error, number of items read is %lu\n", items);
      EXIT(5);
    }
    fclose(ofs);
    ofs = NULL;
    for(i=0; i<ev_number; i++) {
      fprintf(stdout, "# [apply_laplace] ev[%d,%d] <- %e\n", x0+1, i+1, ev_values[i]);
    }

    sprintf(filename, "%s/phases.%.4d.%.3d", filename_prefix2, Nconf, x0);
    fprintf(stdout, "# [apply_laplace] reading eigenvalues from file %s\n", filename);
    ofs = fopen(filename, "r");
    if(ofs == NULL) {
      fprintf(stderr, "[apply_laplace] Error from fopen for filename %s\n", filename);
      EXIT(4);
    }
    items = fread (ev_phases, sizeof(double), (size_t)ev_number, ofs );
    if(items != (size_t)ev_number) {
      fprintf(stderr, "[apply_laplace] Error, number of items read is %lu\n", items);
      EXIT(5);
    }
    fclose(ofs);
    ofs = NULL;
    for(i=0; i<ev_number; i++) {
      fprintf(stdout, "# [apply_laplace] arg[%d,%d] <- %e\n", x0+1, i+1, ev_phases[i]);
    }

    for(i=0; i<ev_number; i++)
    /* for(i=0; i<5; i++) */
    {
      dtmp[0]= 0.;
      dtmp[1]= 0.;
      for(ix=0; ix<VOL3; ix++) {
        _co_eq_cv_dag_ti_cv(&w, ev_field[i]+_GVI(ix), ev_field[i]+_GVI(ix));
        dtmp[0] += w.re;
        dtmp[1] += w.im;
      }
      fprintf(stdout, "# [apply_laplac3] norm of vector no. %d = %e + %ei\n", i, dtmp[0], dtmp[1]);

      /* apply laplace operator */
      cv_eq_laplace_cv(ev_field2, g_gauge_field,  ev_field[i], x0);

      dtmp[0] = 0;
      dtmp[1] = 0;
      dtmp[2] = 0;
      dtmp[3] = 0;
      
      for(ix=0; ix<VOL3; ix++) {
        for(mu=0; mu<3; mu++) {
          w1.re = ev_field[i][_GVI(ix)+2*mu  ];
          w1.im = ev_field[i][_GVI(ix)+2*mu+1];
          w2.re = ev_field2[_GVI(ix)+2*mu  ];
          w2.im = ev_field2[_GVI(ix)+2*mu+1];
          _co_eq_co_ti_co_inv(&w, &w2, &w1);
          dtmp[0] += w.re;
          dtmp[1] += w.re * w.re;
          dtmp[2] += w.im;
          dtmp[3] += w.im * w.im;
          /* fprintf(stdout, "# [apply_laplace] %6d %3d %25.16e %25.16e %25.16e %25.16e %25.16e %25.16e\n", ix, mu, w1.re, w1.im, w2.re, w2.im, w.re, w.im); */
        }
      }

      dtmp[0] /= (double)(VOL3*3);
      dtmp[1] /= (double)(VOL3*3);
      dtmp[2] /= (double)(VOL3*3);
      dtmp[3] /= (double)(VOL3*3);
      fprintf(stdout, "# [apply_laplace] ev[%d, %d] dtmp  %25.16e ( %25.16e )  %25.16e ( %25.16e ) \n", x0, i, dtmp[0], dtmp[1], dtmp[2], dtmp[3]);

      dtmp[1] = sqrt( fabs(dtmp[1] - dtmp[0]*dtmp[0]) / (double)(VOL3-1) );

      dtmp[3] = sqrt( fabs(dtmp[3] - dtmp[2]*dtmp[2]) / (double)(VOL3-1) );

      fprintf(stdout, "# [apply_laplace] ev[%d, %d] = %e ( %e ) + %e ( %e ) i\n", x0, i, dtmp[0], dtmp[1], dtmp[2], dtmp[3]);

#if 0
      /* multply with eigenvalue */
      for(ix=0; ix<VOL3; ix++) {
        _cv_eq_cv_ti_re(ev_field3+_GVI(ix), ev_field[i]+_GVI(ix), ev_values[i]);
      }
#endif
      /* for(ix=0; ix<VOL3; ix++)*/
      for(ix=0; ix<1; ix++)
      {
        fprintf(stdout, "# [apply_laplace] ix = %u\n", ix);
        _cv_printf(ev_field[i]+_GVI(ix) , "\tv ", stdout);
        _cv_printf(ev_field2+_GVI(ix) , "\tDv", stdout);
        /* _cv_printf(ev_field3+_GVI(ix) , "lv", stdout); */

      }

#if 0
      norm  = 0.;
      norm2 = 0.;
      for(ix=0; ix<VOL3; ix++) {
        _co_eq_cv_dag_ti_cv(&w, ev_field2+_GVI(ix), ev_field2+_GVI(ix));
        norm2 += w.re;

        _cv_mi_eq_cv(ev_field2+_GVI(ix), ev_field3+_GVI(ix));

        _co_eq_cv_dag_ti_cv(&w, ev_field2+_GVI(ix), ev_field2+_GVI(ix));
        norm += w.re;

      }

      fprintf(stdout, "# [apply_lapace] x0=%2d ev=%3d norm / %e / %e / %e\n", x0, i, norm, norm2, norm3);
#endif       
    }  /* end of loop on eigenvectors */

  }  /* of loop on timeslices */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  if(ev_field != NULL) {
    if(ev_field[0] != NULL) free(ev_field[0]);
    free(ev_field);
  }
  if(ev_field2 != NULL) free(ev_field2);
  if(ev_field3 != NULL) free(ev_field3);
  if(ev_values != NULL) free(ev_values);
  if(ev_phases != NULL) free(ev_phases);

  free(g_gauge_field);
  free_geometry();

  g_the_time = time(NULL);
  fprintf(stdout, "# [apply_laplace] %s# [apply_laplace] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [apply_laplace] %s# [apply_laplace] end fo run\n", ctime(&g_the_time));
  fflush(stderr);


#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return(0);
}

}
