/****************************************************
 * apply_Dtm
 *
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

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#ifdef __cplusplus
}
#endif


#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "table_init_d.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "scalar_products.h"
#include "clover.h"

using namespace cvc;

void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, exitstatus;
  int filename_set = 0;
  char filename[200];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;

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
  if(filename_set==0) strcpy(filename, "apply_Dtm.input");
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);


#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [apply_Dtm] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    EXIT(14);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(15);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(16);
  }
#endif

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
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
    fprintf(stderr, "Error from init_geometry\n");
    exit(101);
  }

  geometry();

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [apply_Dtm] reading gauge field from file %s\n", filename);
      read_lime_gauge_field_doubleprec(filename);
    } else {
      /* initialize unit matrices */
      if(g_cart_id==0) fprintf(stdout, "\n# [apply_Dtm] initializing unit matrices\n");
      for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
        _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
      }
    }
#else
   Nconf = g_tmLQCD_lat.nstore;
   if(g_cart_id== 0) fprintf(stdout, "[apply_Dtm] Nconf = %d\n", Nconf);

   exitstatus = tmLQCD_read_gauge(Nconf);
   if(exitstatus != 0) {
     EXIT(3);
   }

   exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
   if(exitstatus != 0) {
     EXIT(4);
   }
   if(&g_gauge_field == NULL) {
     fprintf(stderr, "[apply_Dtm] Error, &g_gauge_field is NULL\n");
     EXIT(5);
   }
#endif

#ifdef HAVE_MPI
   xchange_gauge();
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[apply_Dtm] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************
   * measure the plaquette
   ***********************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[apply_Dtm] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }


  no_fields = 2;
  g_spinor_field = init_2level_dtable ( no_fields, _GSI(VOLUME+RAND) );
  if (  g_spinor_field == NULL ) {
    fprintf ( stderr, "# [apply_Dtm] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  double * zero_gauge_field = init_1level_dtable ( 72*(VOLUMEPLUSRAND) );
  memset ( zero_gauge_field, 0., 72*VOLUME*sizeof(double) );

  for ( int i =0; i< 12; i++ )  {
    /****************************************
     * read read the spinor fields
     ****************************************/

    sprintf ( filename, "%s.%.4d.t%dx%dy%dz%d.%d.inverted", filename_prefix, Nconf, 
        g_source_coords_list[0][0],
        g_source_coords_list[0][1],
        g_source_coords_list[0][2],
        g_source_coords_list[0][3], i );

    if(g_cart_id==0) fprintf(stdout, "# [apply_Dtm] Reading prop. from file %s\n", filename_prefix);
    if( read_lime_spinor(g_spinor_field[0], filename, 0) != 0 ) {
      fprintf(stderr, "[apply_Dtm] Error, could not read file %s\n", filename_prefix);
      EXIT(9);
    }


    /* Q_phi ( g_spinor_field[1], g_spinor_field[0], gauge_field_with_phase, g_mu ); */
    Q_phi ( g_spinor_field[1], g_spinor_field[0], zero_gauge_field, g_mu );

    sprintf ( filename, "D_%s.%.4d.t%dx%dy%dz%d.%d.inverted.ascii", filename_prefix, Nconf, 
        g_source_coords_list[0][0],
        g_source_coords_list[0][1],
        g_source_coords_list[0][2],
        g_source_coords_list[0][3], i );
    FILE * ofs = fopen( filename, "w");
    exitstatus = printf_spinor_field ( g_spinor_field[1], 0, ofs );
    fclose ( ofs );

  }

  fini_1level_dtable ( &zero_gauge_field );
#if 0
  spinor_scalar_product_re(&norm2, g_spinor_field[0], g_spinor_field[0], VOLUME);
  fprintf(stdout, "# [apply_Dtm] propagator norm = %e\n", sqrt(norm2));

  spinor_scalar_product_re(&norm2, g_spinor_field[1], g_spinor_field[1], VOLUME);
  fprintf(stdout, "# [apply_Dtm] source norm = %e\n", sqrt(norm2));
#endif

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();

  if ( g_gauge_field          != NULL ) free ( g_gauge_field );
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );

  fini_2level_dtable ( &g_spinor_field );

  fini_clover ( &mzz, &mzzinv );

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [apply_Dtm] %s# [apply_Dtm] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [apply_Dtm] %s# [apply_Dtm] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }
  return(0);
}

