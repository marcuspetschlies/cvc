/****************************************************
 * test_sp2.cpp
 *
 * So 5. Jun 18:15:21 CEST 2016
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

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status; 
  int filename_set = 0;
  char filename[200];

  double ratime, retime;

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
  if(g_cart_id==0) fprintf(stdout, "# [test_sp2] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_sp2] Error from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  unsigned int const Vhalf = VOLUME / 2;
  unsigned int const VOL3  = LX*LY*LZ;

  /* init and allocate spinor fields */
  double ** g_spinor_field = init_2level_dtable ( 2, _GSI( (VOLUME+RAND) ) );
  if ( g_spinor_field == NULL ) {
    EXIT(11);
  }

  double ** eo_spinor_field = init_2level_dtable ( 4, _GSI( (VOLUME+RAND)/2 ) );
  if ( eo_spinor_field == NULL ) {
    EXIT(12);
  }

  g_seed = 10000 + g_cart_id;
  rlxd_init(2, g_seed);

  /* set the spinor field */
  rangauss (g_spinor_field[0], VOLUME*24);
  rangauss (g_spinor_field[1], VOLUME*24);

  spinor_field_lexic2eo (g_spinor_field[0], eo_spinor_field[0], eo_spinor_field[1]);
  spinor_field_lexic2eo (g_spinor_field[1], eo_spinor_field[2], eo_spinor_field[3]);

  double _Complex * sp_eo_e = init_1level_ztable ( T );
  double _Complex * sp_eo_o = init_1level_ztable ( T );
  double _Complex * sp_le_e = init_1level_ztable ( T );
  double _Complex * sp_le_o = init_1level_ztable ( T );
  if (  sp_eo_e == NULL || sp_eo_o == NULL ||  sp_le_e == NULL || sp_le_o == NULL ) {
    fprintf(stderr, "[test_sp2] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__);
    EXIT(1);
  }

  /* t-dependent scalar product in eo ordering */
  ratime = _GET_TIME;
  eo_spinor_spatial_scalar_product_co( sp_eo_e, eo_spinor_field[0], eo_spinor_field[2], 0);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_sp2] eo_sp time = %e\n", retime-ratime);

  ratime = _GET_TIME;
  eo_spinor_spatial_scalar_product_co(sp_eo_o, eo_spinor_field[1], eo_spinor_field[3], 1);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_sp2] eo_sp time = %e\n", retime-ratime);

  /* t-dependent scalar product in lexic ordering */
  memset(sp_le_e, 0, T*sizeof( double _Complex));
  memset(sp_le_o, 0, T*sizeof( double _Complex));

  for( int x0=0; x0<T; x0++) {
    for( int x1=0; x1<LX; x1++) {
    for( int x2=0; x2<LY; x2++) {
    for( int x3=0; x3<LZ; x3++) {
      unsigned int const ix = g_ipt[x0][x1][x2][x3];
      complex w;
      _co_eq_fv_dag_ti_fv(&w, g_spinor_field[0]+_GSI(ix), g_spinor_field[1]+_GSI(ix));
      if(g_iseven[ix]) {
        sp_le_e[x0] += w.re + w.im * I;
      } else {
        sp_le_o[x0] += w.re + w.im * I;
      }
    }}}
  }

#ifdef HAVE_MPI
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  double * const buffer = (double*)malloc(2*T*sizeof(double));
  if ( buffer == NULL ) {
    fprintf(stderr, "[test_sp2] Error from malloc\n");
    EXIT(2);
  }
  memcpy(buffer, sp_le_e, 2*T*sizeof(double));
  MPI_Allreduce(buffer, sp_le_e, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);

  memcpy(buffer, sp_le_o, 2*T*sizeof(double));
  MPI_Allreduce(buffer, sp_le_o, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);

  free(buffer);
#  endif
#endif

  // compare
  if ( g_ts_id == 0 ) {
    for( int x0 = 0; x0 < T; x0++) {
      int x1 = x0 + g_proc_coords[0] * T;
      fprintf(stdout, "e %2d\t%3d\t%25.16e%25.16e\t%25.16e%25.16e   %25.16e\n", g_cart_id, x1,
         creal( sp_le_e[x0] ), cimag ( sp_le_e[x0] ), creal( sp_eo_e[x0] ), cimag ( sp_eo_e[x0] ),
         cabs( sp_le_o[x0] - sp_eo_o[x0]) );
    }

    for( int x0 = 0; x0 < T; x0++) {
      int x1 = x0 + g_proc_coords[0] * T;
      fprintf(stdout, "o %2d\t%3d\t%25.16e%25.16e\t%25.16e%25.16e   %25.16e\n", g_cart_id, x1,
         creal(sp_le_o[x0]), cimag(sp_le_o[x0]), creal(sp_eo_o[x0]), cimag(sp_eo_o[x0]),
         cabs(sp_le_o[x0] - sp_eo_o[x0]) );
    }
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/


  fini_2level_dtable ( &g_spinor_field );
  fini_2level_dtable ( &eo_spinor_field );

  fini_1level_ztable ( &sp_eo_e );
  fini_1level_ztable ( &sp_eo_o );
  fini_1level_ztable ( &sp_le_e );
  fini_1level_ztable ( &sp_le_o );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_sp2] %s# [test_sp2] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_sp2] %s# [test_sp2] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

