/****************************************************
 * test_vdagw.cpp
 *
 * Do 22. Mär 12:56:24 CET 2018
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
#include <complex.h>
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
#include "table_init_d.h"
#include "table_init_z.h"
#include "contract_cvc_tensor.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status; 
  int filename_set = 0;
  char filename[200];


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
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
  if(g_cart_id==0) fprintf(stdout, "# [test_vdagw] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_xspace] calling tmLQCD wrapper init functions\n");

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

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_vdagw] Error from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  int exitstatus;
  unsigned int const Vhalf = VOLUME / 2;
  unsigned int const VOL3half = LX*LY*LZ / 2;

  unsigned int const nv = 11;
  unsigned int const nw =  7;

  double ** v = init_2level_dtable ( nv, _GSI( (VOLUME+RAND)/2 ) );
  double ** w = init_2level_dtable ( nw, _GSI( (VOLUME+RAND)/2 ) );

  double *** contr = init_3level_dtable ( nv, nw, 2*VOL3half );

  g_seed = 10000 + g_cart_id;
  rlxd_init(2, g_seed);

  /* set the spinor field */
  for ( unsigned int i = 0; i < nv; i++ ) {
    rangauss ( v[i], _GSI(Vhalf) );
#ifdef HAVE_MPI
    xchange_eo_field( v[i], 0 );
#endif
  }

  for ( unsigned int i = 0; i < nw; i++ ) {
    rangauss ( w[i], _GSI(Vhalf) );
#ifdef HAVE_MPI
    xchange_eo_field( w[i], 0 );
#endif
  }

  for ( int t = 0; t <= T; t++ ) {
    exitstatus = vdag_w_spin_color_reduction ( contr, v, w, nv, nw, t );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_vdagw] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(102);
    }

    for ( unsigned int i = 0; i < nv; i++ ) {

    for ( unsigned int k = 0; k < nw; k++ ) {

      double norm = 0.;
      for ( int x1 = 0; x1 < LX; x1++ ) {
      for ( int x2 = 0; x2 < LY; x2++ ) {
      for ( int x3 = 0; x3 < LZ; x3++ ) {

        unsigned int const ix = t==T ? g_iup[g_ipt[t-1][x1][x2][x3]][0] : g_ipt[t][x1][x2][x3];
        unsigned int const ixeosub = g_lexic2eosub[ix];
        if ( ! g_iseven[ix] ) continue;
        unsigned int const iix = _GSI( ixeosub );
        complex z;
        _co_eq_fv_dag_ti_fv ( &z, v[i]+ iix, w[k] + iix );
        unsigned int const iz = g_lexic2eosub[ g_ipt[0][x1][x2][x3] ];
        double const dtmp[2] = {
          z.re - contr[i][k][2*iz ],
          z.im - contr[i][k][2*iz+1]
        };
        norm += dtmp[0] * dtmp[0] + dtmp[1] * dtmp[1];

        // if ( t == T ) 
        //  fprintf ( stdout, "# [test_vdagw] proc%.4d x %3d %3d %3d %3d i %2d k %2d z %25.16e %25.16e c %25.16e %25.16e\n", g_cart_id, 
        //    t + g_proc_coords[0]*T, x1 + g_proc_coords[1]*LX, x2 + g_proc_coords[2]*LY, x3 + g_proc_coords[3]*LZ,
        //    i, k, z.re, z.im, contr[i][k][2*iz ], contr[i][k][2*iz+1] );
      }}}

      norm = sqrt ( norm );

      fprintf ( stdout, "# [test_vdagw] proc%.4d t %3d i %2d k %2d diff %25.16e\n", g_cart_id, t + g_proc_coords[0]*T, i, k, norm );

    }}
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  fini_2level_dtable ( &v );
  fini_2level_dtable ( &w );

  fini_3level_dtable ( &contr );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_vdagw] %s# [test_vdagw] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_vdagw] %s# [test_vdagw] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

