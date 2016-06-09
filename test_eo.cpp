/****************************************************
 * test_eo.cpp 
 *
 * Mi 16. Mär 09:50:59 CET 2016
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
#ifdef OPENMP
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
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "ranlxd.h"
#include "invert_Qtm.h"


using namespace cvc;

int main(int argc, char **argv) {
  
  int c, exitstatus;
  int i, j;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int ix_even, ix_odd, y0, y1, y2, y3;
  /* int start_valuet=0, start_valuex=0, start_valuey=0; */
  /* int threadid, nthreads; */
  /* double diff1, diff2; */
  double plaq=0;
  double spinor1[24], spinor2[24];
  complex w, w2;
  int verbose = 0;
  char filename[200];
  FILE *ofs=NULL;
  double norm, norm2;
  unsigned int Vhalf;
  double **eo_spinor_field = NULL;
  int no_eo_fields;


#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      exit(0);
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "apply_Dtm.input");
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);


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
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;


  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [test_invert] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
    if(exitstatus != 0) { EXIT(4); }

  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_invert] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }


#ifdef HAVE_MPI
   xchange_gauge();
#endif


  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value 1st field: %25.16e\n", plaq);

  no_fields=3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  no_eo_fields = 4;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields; i++) alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);

  g_seed = 10000 + g_cart_id;
  rlxd_init(2, g_seed);

  /* set the spinor field */
  rangauss (eo_spinor_field[0], VOLUME*12);
#ifdef HAVE_MPI
  xchange_eo_field( eo_spinor_field[0], 1);
#endif

  for(ix_even = 0; ix_even < Vhalf; ix_even++) {
    ix_odd = g_lexic2eosub[g_iup[g_eo2lexic[ix_even]][0] ];
    _fv_eq_fv(eo_spinor_field[1]+_GSI(ix_even), eo_spinor_field[0]+_GSI(ix_odd));
  }

  sprintf(filename, "spinor_up.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  for(ix=0; ix<Vhalf; ix++) {
    ix_even = g_eo2lexic[ix];
    x0 = ix_even                  / (LX*LY*LZ) + g_proc_coords[0]*T;
    x1 = ( ix_even % (LX*LY*LZ) ) / (   LY*LZ) + g_proc_coords[1]*LX;
    x2 = ( ix_even % (   LY*LZ) ) / (      LZ) + g_proc_coords[2]*LY;
    x3 = ( ix_even % (      LZ) )              + g_proc_coords[3]*LZ;
    ix_odd  = g_eo2lexic[ix+(VOLUME+RAND)/2];
    y0 = ix_odd                  / (LX*LY*LZ) + g_proc_coords[0]*T;
    y1 = ( ix_odd % (LX*LY*LZ) ) / (   LY*LZ) + g_proc_coords[1]*LX;
    y2 = ( ix_odd % (   LY*LZ) ) / (      LZ) + g_proc_coords[2]*LY;
    y3 = ( ix_odd % (      LZ) )              + g_proc_coords[3]*LZ;
    fprintf(ofs, "# eo = %8d lexic e = %8d = %2d %2d %2d %2d\t lexic o =%8d =  %2d %2d %2d %2d\n", ix,
        ix_even, x0, x1, x2, x3, ix_odd,  y0, y1, y2, y3 );
      for(i=0; i<12; i++) {
      fprintf(ofs, "\t%3d%3d\t%16.7e%16.7e\t%16.7e%16.7e\n", i/3, i%3,
          eo_spinor_field[1][_GSI(ix)+2*i], eo_spinor_field[1][_GSI(ix)+2*i+1],
          eo_spinor_field[0][_GSI(ix)+2*i], eo_spinor_field[0][_GSI(ix)+2*i+1]);
    }
  }
  fclose(ofs);

#ifdef HAVE_MPI
  xchange_eo_field( eo_spinor_field[1], 0);
#endif
  for(ix_odd = 0; ix_odd < Vhalf; ix_odd++) {
    ix_even = g_lexic2eosub[g_idn[g_eo2lexic[ix_odd+(VOLUME+RAND)/2]][0] ];
    _fv_eq_fv(eo_spinor_field[0]+_GSI(ix_odd), eo_spinor_field[1]+_GSI(ix_even));
  }

  sprintf(filename, "spinor_dn.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  for(ix=0; ix<Vhalf; ix++) {
    ix_even = g_eo2lexic[ix];
    x0 = ix_even                  / (LX*LY*LZ) + g_proc_coords[0]*T;
    x1 = ( ix_even % (LX*LY*LZ) ) / (   LY*LZ) + g_proc_coords[1]*LX;
    x2 = ( ix_even % (   LY*LZ) ) / (      LZ) + g_proc_coords[2]*LY;
    x3 = ( ix_even % (      LZ) )              + g_proc_coords[3]*LZ;
    ix_odd  = g_eo2lexic[ix+(VOLUME+RAND)/2];
    y0 = ix_odd                  / (LX*LY*LZ) + g_proc_coords[0]*T;
    y1 = ( ix_odd % (LX*LY*LZ) ) / (   LY*LZ) + g_proc_coords[1]*LX;
    y2 = ( ix_odd % (   LY*LZ) ) / (      LZ) + g_proc_coords[2]*LY;
    y3 = ( ix_odd % (      LZ) )              + g_proc_coords[3]*LZ;
    fprintf(ofs, "# eo = %8d lexic e = %8d = %2d %2d %2d %2d\t lexic o =%8d =  %2d %2d %2d %2d\n", ix,
        ix_even, x0, x1, x2, x3, ix_odd,  y0, y1, y2, y3 );
      for(i=0; i<12; i++) {
      fprintf(ofs, "\t%3d%3d\t%16.7e%16.7e\t%16.7e%16.7e\n", i/3, i%3,
          eo_spinor_field[1][_GSI(ix)+2*i], eo_spinor_field[1][_GSI(ix)+2*i+1],
          eo_spinor_field[0][_GSI(ix)+2*i], eo_spinor_field[0][_GSI(ix)+2*i+1]);
    }
  }
  fclose(ofs);

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  free_geometry();
  if(g_spinor_field != NULL) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field);
  }
  if(eo_spinor_field != NULL) {
    for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
    free(eo_spinor_field);
  }



  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}
