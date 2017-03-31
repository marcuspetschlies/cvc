/****************************************************
 * piN2piN.c
 * 
 * Thu Dec  1 15:14:13 CET 2016
 *
 * PURPOSE:
 *   pi N - pi N 2-point function contractions
 *   with point-source propagators, sequential
 *   propagators and stochastic propagagtors
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
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

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "matrix_init.h"
#include "project.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "contract_baryon.h"

#include <string>
#include <iostream>
#include <iomanip>

using namespace cvc;


/**
 * class for time measurement
 */

class TimeMeas{

private:
  double t;
  std::string secname;

public:

  TimeMeas() {};

  void begin(std::string name){
    end();
    t = _GET_TIME;
    secname = name;
  }

  void end(){
    if(secname == "") return;
    int world_rank = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif
    if(world_rank != 0) return;
    double dt = _GET_TIME - t;
    std::cout << "# [GlobTimeMeas] (" << g_proc_coords[0] << "," << g_proc_coords[1] << "," << g_proc_coords[2] << "," << g_proc_coords[3] << ") section " << secname << " finished in " << std::setprecision(4) << ((float)dt) << "s" << std::endl;
    secname = "";
    t = 0;
  }


};

/*
 * Functions to reduce output
 * */
int mpi_fprintf(FILE* stream, const char * format, ...){
  int world_rank = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif
  if(world_rank == 0){
    va_list arg;
    int done;

    va_start(arg,format);
    done = vfprintf(stream,format,arg);
    va_end(arg);

    return done;
  }
  return 0;
}

int mpi_printf(const char * format, ...){
  int world_rank = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif
  if(world_rank == 0){
    va_list arg;
    int done;

    va_start(arg,format);
    done = vfprintf(stdout,format,arg);
    va_end(arg);

    return done;
  }
  return 0;
}

/************************************************************************************
 * determine all stochastic source timeslices needed; make a source timeslice list
 ************************************************************************************/
int **stochastic_source_timeslice_lookup_table;
int *stochastic_source_timeslice_list;
int stochastic_source_timeslice_number;

int get_stochastic_source_timeslices (void) {
  int tmp_list[T_global];
  int t, i_src, i_snk;
  int i_coherent;

  for(t = 0; t<T_global; t++) { tmp_list[t] = -1; }

  i_snk = 0;
  for(i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];
    for(i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + i_coherent * ( T_global / g_coherent_source_number) ) % T_global;
      for(t = 0; t<=g_src_snk_time_separation; t++) {
        int t_snk = ( t_coherent + t ) % T_global;
        if( tmp_list[t_snk] == -1 ) {
          tmp_list[t_snk] = i_snk;
          i_snk++;
        }
      }
    }  /* of loop on coherent source timeslices */
  }    /* of loop on base source timeslices */
  if(g_cart_id == 0) { fprintf(stdout, "# [get_stochastic_source_timeslices] number of stochastic timeslices = %2d\n", i_snk); }

  stochastic_source_timeslice_number = i_snk;
  if(stochastic_source_timeslice_number == 0) {
    fprintf(stderr, "# [get_stochastic_source_timeslices] Error, stochastic_source_timeslice_number = 0\n");
    return(4);
  }

  stochastic_source_timeslice_list = (int*)malloc(i_snk*sizeof(int));
  if(stochastic_source_timeslice_list == NULL) {
    fprintf(stderr, "[get_stochastic_source_timeslices] Error from malloc\n");
    return(1);
  }

  i_src = g_source_location_number * g_coherent_source_number;
  stochastic_source_timeslice_lookup_table = (int**)malloc(i_src * sizeof(int*));
  if(stochastic_source_timeslice_lookup_table == NULL) {
    fprintf(stderr, "[get_stochastic_source_timeslices] Error from malloc\n");
    return(2);
  }

  stochastic_source_timeslice_lookup_table[0] = (int*)malloc( (g_src_snk_time_separation+1) * i_src * sizeof(int));
  if(stochastic_source_timeslice_lookup_table[0] == NULL) {
    fprintf(stderr, "[get_stochastic_source_timeslices] Error from malloc\n");
    return(3);
  }
  for(i_src=1; i_src<g_source_location_number*g_coherent_source_number; i_src++) {
    stochastic_source_timeslice_lookup_table[i_src] = stochastic_source_timeslice_lookup_table[i_src-1] + (g_src_snk_time_separation+1);
  }

  for(i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];
    for(i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {
      int i_prop = i_src * g_coherent_source_number + i_coherent;
      int t_coherent = ( t_base + i_coherent * ( T_global / g_coherent_source_number) ) % T_global;
      for(t = 0; t<=g_src_snk_time_separation; t++) {
        int t_snk = ( t_coherent + t ) % T_global;
        if( tmp_list[t_snk] != -1 ) {
          stochastic_source_timeslice_list[ tmp_list[t_snk] ] = t_snk;
          stochastic_source_timeslice_lookup_table[i_prop][t] = tmp_list[t_snk];
        }
      }
    }  /* of loop on coherent source timeslices */
  }    /* of loop on base source timeslices */

  if(g_cart_id == 0) {
    /* TEST */
    for(i_src = 0; i_src<g_source_location_number; i_src++) {
      int t_base = g_source_coords_list[i_src][0];
      for(i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {
        int i_prop = i_src * g_coherent_source_number + i_coherent;
        int t_coherent = ( t_base + i_coherent * ( T_global / g_coherent_source_number) ) % T_global;

        for(t = 0; t <= g_src_snk_time_separation; t++) {
          fprintf(stdout, "# [get_stochastic_source_timeslices] i_src = %d, i_prop = %d, t_src = %d, dt = %d, t_snk = %d, lookup table = %d\n",
              i_src, i_prop, t_coherent, t,
              stochastic_source_timeslice_list[ stochastic_source_timeslice_lookup_table[i_prop][t] ],
              stochastic_source_timeslice_lookup_table[i_prop][t]);
        }
      }
    }

    /* TEST */
    for(t=0; t<stochastic_source_timeslice_number; t++) {
      fprintf(stdout, "# [get_stochastic_source_timeslices] stochastic source timeslice no. %d is t = %d\n", t, stochastic_source_timeslice_list[t]);
    }
  }  /* end of if g_cart_id == 0 */
  return(0);
}  /* end of get_stochastic_source_timeslices */


/***********************************************************
 * determine source coordinates, find out, if source_location is in this process
 *   gcoords: global source coordinates (in)
 *   lcoords: local source coordinates (out)
 *   proc_id: source proc id (out)
 *   location: local lexic source location (out)
 ***********************************************************/
int get_point_source_info (int gcoords[4], int lcoords[4], int*proc_id) {
  /* local source coordinates */
  int source_proc_id = 0;
  int exitstatus;
#ifdef HAVE_MPI
  int source_proc_coords[4];
  source_proc_coords[0] = gcoords[0] / T;
  source_proc_coords[1] = gcoords[1] / LX;
  source_proc_coords[2] = gcoords[2] / LY;
  source_proc_coords[3] = gcoords[3] / LZ;
  exitstatus = MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  if(exitstatus !=  MPI_SUCCESS ) {
    fprintf(stderr, "[get_point_source_info] Error from MPI_Cart_rank, status was %d\n", exitstatus);
    EXIT(9);
  }
  if(source_proc_id == g_cart_id) {
    fprintf(stdout, "# [get_point_source_info] process %2d = (%3d,%3d,%3d,%3d) has source location\n", source_proc_id,
        source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }
#endif

  if(proc_id != NULL) *proc_id = source_proc_id;
  int x[4] = {-1,-1,-1,-1};
  /* local coordinates */
  if(g_cart_id == source_proc_id) {
    x[0] = gcoords[0] % T;
    x[1] = gcoords[1] % LX;
    x[2] = gcoords[2] % LY;
    x[3] = gcoords[3] % LZ;
    if(lcoords != NULL) {
      memcpy(lcoords,x,4*sizeof(int));
    }
  }
  return(0);
}  /* end of get_point_source_info */

/***********************************************************
 * usage function
 ***********************************************************/
void usage() {
  mpi_fprintf(stdout, "Code to perform contractions for piN 2-pt. function\n");
  mpi_fprintf(stdout, "Usage:    [options]\n");
  mpi_fprintf(stdout, "Options: -f input filename [default cvc.input]\n");
  mpi_fprintf(stdout, "         -h? this help\n");
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}
  
  
/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
  
  const int n_c=3;
  const int n_s=4;
  const int max_num_diagram = 6;


  int c, i, k, i_src, i_coherent, isample;
  int filename_set = 0;
  int exitstatus;
  int it, ir, is;
  int op_id_up= -1, op_id_dn = -1;
  int gsx[4], sx[4];
  int source_proc_id = 0;
  char filename[200];
  double ratime, retime;
  double plaq_m = 0., plaq_r = 0.;
  double *spinor_work[2];
  unsigned int VOL3;
  size_t sizeof_spinor_field = 0, sizeof_spinor_field_timeslice = 0;
  spinor_propagator_type **conn_X=NULL;
  double ****buffer=NULL;
  int io_proc = -1;
  int icomp, iseq_mom, iseq2_mom;
  double **propagator_list_up = NULL, **propagator_list_dn = NULL, **sequential_propagator_list = NULL, **stochastic_propagator_list = NULL,
         **stochastic_source_list = NULL;
  double *gauge_field_smeared = NULL;
  TimeMeas tm;

/*******************************************************************
 * Gamma components for the piN and Delta:
 *                                                                 */
// gamma_6 = 0_5
// gamma_4 = id
  const int num_component_piN_piN        = 9;
  int gamma_component_piN_piN[9][2]      = { {5, 5}, {4,4}, {6,6}, {5,4}, {5,6}, {4,5}, {4,6}, {6,5}, {6,4} };
  double gamma_component_sign_piN_piN[9] = {+1, +1, +1, +1, +1, +1, +1, +1, +1};

  const int num_component_N_N        = 9;
  int gamma_component_N_N[9][2]      = { {5, 5}, {4,4}, {6,6}, {5,4}, {5,6}, {4,5}, {4,6}, {6,5}, {6,4} };
  double gamma_component_sign_N_N[9] = {+1, +1, +1, +1, +1, +1, +1, +1, +1};

  const int num_component_D_D        = 9;
  int gamma_component_D_D[9][2]      = {{1,1}, {1,2}, {1,3}, {2,1}, {2,2}, {2,3}, {3,1}, {3,2}, {3,3}};
  double gamma_component_sign_D_D[9] = {+1.,-1.,+1.,+1,-1,+1,+1,-1,+1};

  const int num_component_piN_D        = 9;
  int gamma_component_piN_D[9][2]      = { {1, 5}, {2, 5}, {3, 5}, {1,4}, {2,4}, {3,4}, {1,6}, {2,6}, {3,6}};
  double gamma_component_sign_piN_D[9] = {+1., +1., +1., +1., +1., +1., +1., +1. ,+1.};

  int num_component_max = 9;
/*
 *******************************************************************/

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
  /*  uint32_t aff_buffer_size; */
#endif

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
  mpi_fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

  if(g_fermion_type == -1 ) {
    fprintf(stderr, "# [piN2piN] fermion_type must be set\n");
    exit(1);
  } else {
    mpi_fprintf(stdout, "# [piN2piN] using fermion type %d\n", g_fermion_type);
  }

#ifdef HAVE_TMLQCD_LIBWRAPPER

  mpi_fprintf(stdout, "# [piN2piN] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1, 0); */
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



#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  mpi_fprintf(stdout, "[piN2piN] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[piN2piN] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  VOL3 = LX*LY*LZ;
  sizeof_spinor_field = _GSI(VOLUME)*sizeof(double);
  sizeof_spinor_field_timeslice = _GSI(VOL3)*sizeof(double);


#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [piN2piN] proc%.4d tr%.4d is io process\n", g_cart_id, g_tr_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [piN2piN] proc%.4d tr%.4d is send process\n", g_cart_id, g_tr_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif



#ifndef HAVE_TMLQCD_LIBWRAPPER
  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  switch(g_gauge_file_format) {
    case 0:
      sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "reading gauge field from file %s\n", filename);
      exitstatus = read_lime_gauge_field_doubleprec(filename);
      break;
    case 1:
      sprintf(filename, "%s.%.5d", gaugefilename_prefix, Nconf);
      if(g_cart_id==0) fprintf(stdout, "\n# [piN2piN] reading gauge field from file %s\n", filename);
      exitstatus = read_nersc_gauge_field(g_gauge_field, filename, &plaq_r);
      break;
  }
  if(exitstatus != 0) {
    fprintf(stderr, "[piN2piN] Error, could not read gauge field\n");
    EXIT(21);
  }
#  ifdef HAVE_MPI
  xchange_gauge();
#  endif
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[piN2piN] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[piN2piN] Error, &g_gauge_field is NULL\n");
    EXIT(5);
  }
#endif

  /* measure the plaquette */
  plaquette(&plaq_m);
  if(g_cart_id==0) fprintf(stdout, "# [piN2piN] read plaquette value    : %25.16e\n", plaq_r);
  if(g_cart_id==0) fprintf(stdout, "# [piN2piN] measured plaquette value: %25.16e\n", plaq_m);


  /***********************************************
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {
    if(N_ape > 0 ) {
      alloc_gauge_field(&gauge_field_smeared, VOLUMEPLUSRAND);
      memcpy(gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
  
    } else {
      gauge_field_smeared = g_gauge_field;
    }
  }

  /***********************************************************
   * determine the stochastic source timeslices
   ***********************************************************/
  exitstatus = get_stochastic_source_timeslices();
  if(exitstatus != 0) {
    fprintf(stderr, "[piN2piN] Error from get_stochastic_source_timeslices, status was %d\n", exitstatus);
    EXIT(19);
  }

  /***********************************************************
   * allocate work spaces with halo
   ***********************************************************/
  alloc_spinor_field(&spinor_work[0], VOLUMEPLUSRAND);
  alloc_spinor_field(&spinor_work[1], VOLUMEPLUSRAND);

  /***********************************************************
   * allocate memory for the contractions
   **********************************************************/
  conn_X = (spinor_propagator_type**)malloc(max_num_diagram * sizeof(spinor_propagator_type*));
  for(i=0; i<max_num_diagram; i++) {
    conn_X[i] = create_sp_field( (size_t)VOLUME * num_component_max );
    if(conn_X[i] == NULL) {
      fprintf(stderr, "[piN2piN] Error, could not alloc conn_X\n");
      EXIT(2);
    }
  }

  /***********************************************************
   * set operator ids depending on fermion type
   ***********************************************************/
  if(g_fermion_type == _TM_FERMION) {
    op_id_up = 0;
    op_id_dn = 1;
  } else if(g_fermion_type == _WILSON_FERMION) {
    op_id_up = 0;
    op_id_dn = 0;
  }


  /***********************************************************
   * up-type propagator
   ***********************************************************/
  no_fields = g_coherent_source_number * g_source_location_number * n_s*n_c; /* up propagators at all base x coherent source locations */ 
  propagator_list_up = (double**)malloc(no_fields * sizeof(double*));
  propagator_list_up[0] = (double*)malloc(no_fields * sizeof_spinor_field);
  if(propagator_list_up[0] == NULL) {
    fprintf(stderr, "[piN2piN] Error from malloc\n");
    EXIT(44);
  }
  for(i=1; i<no_fields; i++) propagator_list_up[i] = propagator_list_up[i-1] + _GSI(VOLUME);

  if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] up-type inversion\n");
  for(i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];
    for(i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global; 
      int i_prop = i_src * g_coherent_source_number + i_coherent;
      gsx[0] = t_coherent;
      gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
      gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
      gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

      ratime = _GET_TIME;
      get_point_source_info (gsx, sx, &source_proc_id);

      for(is=0;is<n_s*n_c;is++) {
        memset(spinor_work[0], 0, sizeof_spinor_field);
        memset(spinor_work[1], 0, sizeof_spinor_field);
        if(source_proc_id == g_cart_id)  {
          spinor_work[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])+2*is] = 1.;
        }
        /* source-smear the point source */
        exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
        }

        exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(12);
        }

        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
        }

        /* sink-smear the point-source propagator */
        exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

        memcpy( propagator_list_up[i_prop*n_s*n_c + is], spinor_work[1], sizeof_spinor_field);
      }  /* end of loop on spin color */
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] time for up propagator = %e seconds\n", retime-ratime);

    }  /* end of loop on coherent source timeslices */
  }    /* end of loop on base source timeslices */

  /***********************************************************
   * dn-type propagator
   ***********************************************************/
  if(g_fermion_type == _TM_FERMION) {
    no_fields = g_coherent_source_number * g_source_location_number * n_s*n_c; /* dn propagators at all base x coherent source locations */ 
    propagator_list_dn = (double**)malloc(no_fields * sizeof( double*));
    propagator_list_dn[0] = (double*)malloc(no_fields * sizeof_spinor_field);
    if(propagator_list_dn[0] == NULL) {
      fprintf(stderr, "[piN2piN] Error from malloc\n");
      EXIT(45);
    }
    for(i=1; i<no_fields; i++) propagator_list_dn[i] = propagator_list_dn[i-1] + _GSI(VOLUME);

    if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] dn-type inversion\n");
    for(i_src = 0; i_src<g_source_location_number; i_src++) {
      int t_base = g_source_coords_list[i_src][0];
      for(i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
        int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global; 
        int i_prop = i_src * g_coherent_source_number + i_coherent;
        gsx[0] = t_coherent;
        gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
        gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
        gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

        ratime = _GET_TIME;
        get_point_source_info (gsx, sx, &source_proc_id);

        for(is=0;is<n_s*n_c;is++) {

          memset(spinor_work[0], 0, sizeof_spinor_field);
          memset(spinor_work[1], 0, sizeof_spinor_field);
          if(source_proc_id == g_cart_id)  {
            spinor_work[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])+2*is] = 1.;
          }

          /* source-smear the point source */
          exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[0], spinor_work[0], -1, g_fermion_type, VOLUME);
          }

          exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_dn, 0);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(12);
          }

          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[1], spinor_work[1], -1, g_fermion_type, VOLUME);
          }

          /* sink-smear the point-source propagator */
          exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);


          memcpy( propagator_list_dn[i_prop*n_s*n_c + is], spinor_work[1], sizeof_spinor_field);
        }
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] time for dn propagator = %e seconds\n", retime-ratime);
      } /* end of loop on coherent source timeslices */
    }  /* end of loop on base source timeslices */
  } else {
    propagator_list_dn  = propagator_list_up;
  }



  /***********************************************************
   ***********************************************************
   **
   ** sequential inversions
   **
   ***********************************************************
   ***********************************************************/

  /***********************************************************
   * allocate memory for the sequential propagators
   ***********************************************************/
  no_fields = g_source_location_number * g_seq_source_momentum_number * n_s*n_c;  /* sequential propagators at all base source locations */ 
  sequential_propagator_list = (double**)malloc(no_fields * sizeof(double*));
  sequential_propagator_list[0] = (double*)malloc(no_fields * sizeof_spinor_field);
  if( sequential_propagator_list[0] == NULL) {
    fprintf(stderr, "[piN2piN] Error from malloc\n");
    EXIT(46);
  }
  for(i=1; i<no_fields; i++) sequential_propagator_list[i] = sequential_propagator_list[i-1] + _GSI(VOLUME);

  /* loop on sequential source momenta */
  for(iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

    /***********************************************************
     * sequential propagator U^{-1} g5 exp(ip) D^{-1}: tfii
     ***********************************************************/
    if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] sequential inversion fpr pi2 = (%d, %d, %d)\n", 
        g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);

    double **prop_list = (double**)malloc(g_coherent_source_number * sizeof(double*));
    if(prop_list == NULL) {
      fprintf(stderr, "[piN2piN] Error from malloc\n");
      EXIT(43);
    }

    for(i_src=0; i_src<g_source_location_number; i_src++) {
  
      int i_prop = iseq_mom * g_source_location_number + i_src;

      gsx[0] = g_source_coords_list[i_src][0];
      gsx[1] = g_source_coords_list[i_src][1];
      gsx[2] = g_source_coords_list[i_src][2];
      gsx[3] = g_source_coords_list[i_src][3];

      ratime = _GET_TIME;
      for(is=0;is<n_s*n_c;is++) {


        /* extract spin-color source-component is from coherent source dn propagators */
        for(i=0; i<g_coherent_source_number; i++) {
          if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] using dn prop id %d / %d\n", (i_src * g_coherent_source_number + i), (i_src * g_coherent_source_number + i)*n_s*n_c + is);
          prop_list[i] = propagator_list_dn[(i_src * g_coherent_source_number + i)*n_s*n_c + is];
        }

        /* build sequential source */
        exitstatus = init_coherent_sequential_source(spinor_work[0], prop_list, gsx[0], g_coherent_source_number, g_seq_source_momentum_list[iseq_mom], 5);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN] Error from init_coherent_sequential_source, status was %d\n", exitstatus);
          EXIT(14);
        }

        /* source-smear the coherent source */
        exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

        /* tm-rotate sequential source */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
        }

        memset(spinor_work[1], 0, sizeof_spinor_field);
        /* invert */
        exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
        if(exitstatus != 0) {
          fprintf(stderr, "[piN2piN] Error from tmLQCD_invert, status was %d\n", exitstatus);
          EXIT(12);
        }

        /* tm-rotate at sink */
        if( g_fermion_type == _TM_FERMION ) {
          spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
        }

        /* sink-smear the coherent-source propagator */
        exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

        memcpy( sequential_propagator_list[i_prop*n_s*n_c + is], spinor_work[1], sizeof_spinor_field);

        if(g_write_sequential_propagator) { /* write sequential propagator to file */
          sprintf(filename, "seq_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.inverted",
              filename_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3], is,
              g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);
          if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] writing propagator to file %s\n", filename);
          exitstatus = write_propagator(spinor_work[1], filename, 0, 64);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN] Error from write_propagator, status was %d\n", exitstatus);
            EXIT(15);
          }
        }  /* end of if write sequential propagator */
      }  /* end of loop on spin-color component */
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] time for seq propagator = %e seconds\n", retime-ratime);


    }  /* end of loop on base source locations */
    free(prop_list);
  }  /* end of loop on sequential momentum list */


  for(i_src=0; i_src < g_source_location_number; i_src++ ) {
    int t_base = g_source_coords_list[i_src][0];

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_version();
      fprintf(stdout, "# [piN2piN] using aff version %s\n", aff_status_str);
    
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "B_B", Nconf, t_base );
      fprintf(stdout, "# [piN2piN] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      }
    
      if( (affn = aff_writer_root(affw)) == NULL ) {
        fprintf(stderr, "[piN2piN] Error, aff writer is not initialized\n");
        EXIT(5);
      }
  
      aff_buffer = (double _Complex*)malloc(T_global*g_sv_dim*g_sv_dim*sizeof(double _Complex));
        if(aff_buffer == NULL) {
        fprintf(stderr, "[piN2piN] Error from malloc\n");
        EXIT(6);
      }
    }  /* end of if io_proc == 2 */
#endif

  /******************************************************
   ******************************************************
   **
   ** contractions without stochastic propagators
   **
   ******************************************************
   ******************************************************/

    for(i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;
      int i_prop = i_src * g_coherent_source_number + i_coherent;
      gsx[0] = t_coherent;
      gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
      gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
      gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;
      get_point_source_info (gsx, sx, &source_proc_id);


      /* Delta - Delta 2pt */
      /* pi    - pi    2pt */
      /* Delta - pi N  3pt */

      /***********************
       ***********************
       **
       ** N     - N     2pt
       **
       ***********************
       ***********************/
      for(i=0; i<max_num_diagram; i++) { memset(conn_X[i][0][0], 0, 2*VOLUME*g_sv_dim*g_sv_dim*sizeof(double)); }

      exitstatus = contract_N_N (conn_X, &(propagator_list_up[i_prop*n_s*n_c]), &(propagator_list_dn[i_prop*n_s*n_c]) , num_component_N_N, gamma_component_N_N, gamma_component_sign_N_N);

      for(i=0; i<2; i++) {
        /* phase from quark field boundary condition */
        add_baryon_boundary_phase (conn_X[i], gsx[0], num_component_N_N);

        /* momentum projection */
        double ****connt = NULL;
        if( (exitstatus = init_4level_buffer(&connt, T, g_sink_momentum_number, num_component_N_N * g_sv_dim, 2*g_sv_dim) ) != 0 ) {
          fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
          EXIT(57);
        }
        for(it=0; it<T; it++) {
          exitstatus = momentum_projection2 (conn_X[i][it*VOL3*num_component_N_N][0], connt[it][0][0], num_component_N_N*g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, NULL );
        }
        /* add complex phase from source location and source momentum
         *   assumes momentum conservation 
         */
        //if( g_cart_id == 0 ) fprintf(stderr, "[piN2piN] Warning, add_source_phase called for N - N 2-point function\n");
        //add_source_phase (connt, NULL, NULL, &(gsx[1]), num_component_N_N);

        /* write to file */
       
        ratime = _GET_TIME;
#ifdef HAVE_MPI
        if(io_proc>0) {
          if( (exitstatus = init_4level_buffer(&buffer, T_global, g_sink_momentum_number, num_component_N_N*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
            fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
            EXIT(58);
          }
          k = T * g_sink_momentum_number * num_component_N_N * g_sv_dim * g_sv_dim * 2;
          exitstatus = MPI_Allgather(connt[0][0][0], k, MPI_DOUBLE, buffer[0][0][0], k, MPI_DOUBLE, g_tr_comm);
          if(exitstatus != MPI_SUCCESS) {
            fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
            EXIT(124);
          }
        }
#else
        buffer = connt;
#endif

        if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
          for(k=0; k<g_sink_momentum_number; k++) {
            for(icomp=0; icomp<num_component_N_N; icomp++) {

              sprintf(aff_buffer_path, "/%s/diag%d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                  "N-N", i,
                  g_sink_momentum_list[k][0],                g_sink_momentum_list[k][1],                g_sink_momentum_list[k][2],
                  gsx[0], gsx[1], gsx[2], gsx[3],
                  gamma_component_N_N[icomp][0], gamma_component_N_N[icomp][1]);

              fprintf(stdout, "# [piN2piN] current aff path = %s\n", aff_buffer_path);

              affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
              for(it=0; it<T_global; it++) {
                ir = ( it - gsx[0] + T_global ) % T_global;
                memcpy(aff_buffer + ir*g_sv_dim*g_sv_dim,  buffer[it][k][icomp*g_sv_dim] , g_sv_dim*g_sv_dim*sizeof(double _Complex) );
              }
              int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global*g_sv_dim*g_sv_dim);
              if(status != 0) {
                fprintf(stderr, "[piN2piN] Error from aff_node_put_double, status was %d\n", status);
                EXIT(81);
              }
            }  /* end of loop on components */
          }  /* end of loop on sink momenta */
#endif
        }  /* end of if io_proc == 2 */

#ifdef HAVE_MPI
        if(io_proc > 0) { fini_4level_buffer(&buffer); }
#endif


        fini_4level_buffer(&connt);
        retime = _GET_TIME;
        if( io_proc == 2 ) fprintf(stdout, "# [piN2piN] time for writing N-N = %e seconds\n", retime-ratime);
      }  /* end of loop on diagrams */

      /***********************
       ***********************
       **
       ** D - D 2-pt
       **
       ***********************
       ***********************/
      for(i=0; i<max_num_diagram; i++) { memset(conn_X[i][0][0], 0, 2*VOLUME*g_sv_dim*g_sv_dim*sizeof(double)); }

      exitstatus = contract_D_D (conn_X, &(propagator_list_up[i_prop*n_s*n_c]), &(propagator_list_dn[i_prop*n_s*n_c]),
         num_component_D_D, gamma_component_D_D, gamma_component_sign_D_D);

      for(i=0; i<6; i++) {
        /* phase from quark field boundary condition */
        add_baryon_boundary_phase (conn_X[i], gsx[0], num_component_D_D);

        /* momentum projection */
        double ****connt = NULL;
        if( (exitstatus = init_4level_buffer(&connt, T, g_sink_momentum_number, num_component_D_D * g_sv_dim, 2*g_sv_dim) ) != 0 ) {
          fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
          EXIT(59);
        }
        for(it=0; it<T; it++) {
          exitstatus = momentum_projection2 (conn_X[i][it*VOL3*num_component_D_D][0], connt[it][0][0], num_component_D_D*g_sv_dim*g_sv_dim,
              g_sink_momentum_number, g_sink_momentum_list, NULL );
        }
        /* add complex phase from source location and source momentum
         *   assumes momentum conservation 
         */
        add_source_phase (connt, NULL, NULL, &(gsx[1]), num_component_D_D);

        /* write to file */
       
        ratime = _GET_TIME;
#ifdef HAVE_MPI
        if(io_proc>0) {
          if( (exitstatus = init_4level_buffer(&buffer, T_global, g_sink_momentum_number, num_component_D_D*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
            fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
            EXIT(60);
          }
          k = T * g_sink_momentum_number * num_component_D_D * g_sv_dim * g_sv_dim * 2;
          exitstatus = MPI_Allgather(connt[0][0][0], k, MPI_DOUBLE, buffer[0][0][0], k, MPI_DOUBLE, g_tr_comm);
          if(exitstatus != MPI_SUCCESS) {
            fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
            EXIT(124);
          }
        }
#else
        buffer = connt;
#endif
    
        if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
          for(k=0; k<g_sink_momentum_number; k++) {

            for(icomp=0; icomp<num_component_D_D; icomp++) {

              sprintf(aff_buffer_path, "/%s/diag%d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                  "D-D", i,
                  g_sink_momentum_list[k][0],                g_sink_momentum_list[k][1],                g_sink_momentum_list[k][2],
                  gsx[0], gsx[1], gsx[2], gsx[3],
                  gamma_component_D_D[icomp][0], gamma_component_D_D[icomp][1]);

              fprintf(stdout, "# [piN2piN] current aff path = %s\n", aff_buffer_path);

              affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
              for(it=0; it<T_global; it++) {
                ir = ( it - gsx[0] + T_global ) % T_global;
                memcpy(aff_buffer + ir*g_sv_dim*g_sv_dim,  buffer[it][k][icomp*g_sv_dim] , g_sv_dim*g_sv_dim*sizeof(double _Complex) );
              }
              int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global*g_sv_dim*g_sv_dim);
              if(status != 0) {
                fprintf(stderr, "[piN2piN] Error from aff_node_put_double, status was %d\n", status);
                EXIT(81);
              }
            }  /* end of loop on components */
          }  /* end of loop on sink momenta */
#endif
        }  /* end of if io_proc == 2 */
    
#ifdef HAVE_MPI
        if(io_proc > 0) { fini_4level_buffer(&buffer); }
#endif

        fini_4level_buffer(&connt);
        retime = _GET_TIME;
        if( io_proc == 2 ) fprintf(stdout, "# [piN2piN] time for writing D-D = %e seconds\n", retime-ratime);
      }  /* end of loop on diagrams */

      /***********************
       ***********************
       **
       ** piN - D 2-pt
       **
       ***********************
       ***********************/

      /* loop on sequential source momenta */
      for(iseq_mom = 0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {
        int i_seq_prop = iseq_mom * g_source_location_number + i_src;


        for(i=0; i<max_num_diagram; i++) { memset(conn_X[i][0][0], 0, 2*VOLUME*g_sv_dim*g_sv_dim*sizeof(double)); }

        exitstatus = contract_piN_D (conn_X, &(propagator_list_up[i_prop*n_s*n_c]), &(propagator_list_dn[i_prop*n_s*n_c]), 
            &(sequential_propagator_list[i_seq_prop*n_s*n_c]), num_component_piN_D, gamma_component_piN_D, gamma_component_sign_piN_D);

        for(i=0; i<6; i++) {
          /* phase from quark field boundary condition */
          add_baryon_boundary_phase (conn_X[i], gsx[0], num_component_piN_D);

          /* momentum projection */
          double ****connt = NULL;
          if( (exitstatus = init_4level_buffer(&connt, T, g_sink_momentum_number, num_component_piN_D*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
            fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
            EXIT(61);
          }
          for(it=0; it<T; it++) {
            exitstatus = momentum_projection2 (conn_X[i][it*VOL3*num_component_piN_D][0], connt[it][0][0], 
                num_component_piN_D*g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, NULL );
          }
          /* add complex phase from source location and source momentum
           *   assumes momentum conservation 
           */
          add_source_phase (connt, g_seq_source_momentum_list[iseq_mom], NULL, &(gsx[1]), num_component_piN_D);

          /* write to file */
       
          ratime = _GET_TIME;
#ifdef HAVE_MPI
          if(io_proc>0) {
            if( (exitstatus = init_4level_buffer(&buffer, T_global, g_sink_momentum_number, num_component_piN_D*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
              fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
              EXIT(62);
            }
            k = T * g_sink_momentum_number * num_component_piN_D * g_sv_dim * g_sv_dim * 2;
            exitstatus = MPI_Allgather(connt[0][0][0], k, MPI_DOUBLE, buffer[0][0][0], k, MPI_DOUBLE, g_tr_comm);
            if(exitstatus != MPI_SUCCESS) {
              fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
              EXIT(124);
            }
          }
#else
          buffer = connt;
#endif
    
          if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
            for(k=0; k<g_sink_momentum_number; k++) {

              for(icomp=0; icomp<num_component_piN_D; icomp++) {

                sprintf(aff_buffer_path, "/%s/diag%d/pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                    "pixN-D", i,
                    g_seq_source_momentum_list[iseq_mom][0],   g_seq_source_momentum_list[iseq_mom][1],   g_seq_source_momentum_list[iseq_mom][2],
                    g_sink_momentum_list[k][0],                g_sink_momentum_list[k][1],                g_sink_momentum_list[k][2],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    gamma_component_piN_D[icomp][0], gamma_component_piN_D[icomp][1]);

                fprintf(stdout, "# [piN2piN] current aff path = %s\n", aff_buffer_path);

                affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
                for(it=0; it<T_global; it++) {
                  ir = ( it - gsx[0] + T_global ) % T_global;
                  memcpy(aff_buffer + ir*g_sv_dim*g_sv_dim,  buffer[it][k][icomp*g_sv_dim] , g_sv_dim*g_sv_dim*sizeof(double _Complex) );
                }
                int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global*g_sv_dim*g_sv_dim);
                if(status != 0) {
                  fprintf(stderr, "[piN2piN] Error from aff_node_put_complex, status was %d\n", status);
                  EXIT(82);
                }
              }  /* end of loop on components */
            }  /* end of loop on sink momenta */
#endif
          }  /* end of if io_proc == 2 */
    
#ifdef HAVE_MPI
          if(io_proc > 0) { fini_4level_buffer(&buffer); }
#endif

          fini_4level_buffer(&connt);
          retime = _GET_TIME;
          if( io_proc == 2 ) fprintf(stdout, "# [piN2piN] time for writing piN-D = %e seconds\n", retime-ratime);
        }  /* end of loop on diagrams */

      }  /* end of loop on sequential source momenta */       

#if 0
      /***********************
       ***********************
       **
       ** pi - pi 2-pt
       **
       ** we abuse conn_X here, which is an array for spin-propagator fields;
       ** but its entry conn_X(0,0,0) points to space several times VOLUME,
       ** which is sufficiently large for meson-meson contractions
       **
       ***********************
       ***********************/
      for(i=0; i<max_num_diagram; i++) { memset(conn_X[i][0][0], 0, 2*VOLUME*g_sv_dim*g_sv_dim*sizeof(double)); }
      double *conn_M = conn_X[0][0][0];
      contract_twopoint_xdep(conn_M, 5, 5, (void*)(&(propagator_list_up[i_prop*n_s*n_c])), (void*)(&(propagator_list_up[i_prop*n_s*n_c])), n_c, 1, 1., 64);

      double **connt = NULL;
      if( (exitstatus = init_2level_buffer(&connt, g_sink_momentum_number, 2*T) ) != 0 ) {
        fprintf(stderr, "[piN2piN] Error from init_2level_buffer, status was %d\n", exitstatus);
        EXIT(61);
      }

      /* momentum projection */
      exitstatus = momentum_projection ( conn_M, connt[0], T, g_sink_momentum_number, g_sink_momentum_list);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN] Error from momentum_projection, status was %d\n", exitstatus);
        EXIT(8);
      }

      /* write to file */

      ratime = _GET_TIME;
      double **buffer2 = NULL;
#ifdef HAVE_MPI
      if(io_proc>0) {
        if( (exitstatus = init_2level_buffer(&buffer2, g_sink_momentum_number, 2*T_global) ) != 0 ) {
          fprintf(stderr, "[piN2piN] Error from init_2level_buffer, status was %d\n", exitstatus);
          EXIT(62);
        }
        k = T * g_sink_momentum_number * 2;
        exitstatus = MPI_Allgather(connt[0], k, MPI_DOUBLE, buffer2[0], k, MPI_DOUBLE, g_tr_comm);
        if(exitstatus != MPI_SUCCESS) {
          fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
          EXIT(124);
        }
      }
#else
      buffer2 = connt;
#endif

      if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
        for(k=0; k<g_sink_momentum_number; k++) {

          sprintf(aff_buffer_path, "/%s/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
              "pi-pi",
              g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2],
              gsx[0], gsx[1], gsx[2], gsx[3], 5, 5);

          fprintf(stdout, "# [piN2piN] current aff path = %s\n", aff_buffer_path);

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          for(it=0; it<T_global; it++) {
            ir = ( it - gsx[0] + T_global ) % T_global;
            aff_buffer[ir] = buffer2[k][2*it]  + I * buffer2[k][2*it+1];
          }
          int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global);
          if(status != 0) {
            fprintf(stderr, "[piN2piN] Error from aff_node_put_complex, status was %d\n", status);
            EXIT(83);
          }

        }  /* end of loop on sink momenta */
#endif
      }  /* end of if io_proc == 2 */

#ifdef HAVE_MPI
      if(io_proc > 0) { fini_2level_buffer(&buffer2); }
#endif
      fini_2level_buffer(&connt);
#endif  /* of if 0 */

    }  /* end of loop on coherent source locations */

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(11);
      }
      if(aff_buffer != NULL) free(aff_buffer);
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  }  /* end of loop on base source locations */



  /******************************************************
   ******************************************************
   **
   ** stochastic inversions
   **  
   **  dn-type inversions
   ******************************************************
   ******************************************************/

  /******************************************************
   * initialize random number generator
   ******************************************************/
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[piN2piN] Error from init_rng_stat_file status was %d\n", exitstatus);
    EXIT(38);
  }

  /******************************************************
   * allocate memory for stochastic sources
   *   and propagators
   ******************************************************/
  stochastic_propagator_list    = (double**)malloc(g_nsample * sizeof(double*));
  stochastic_propagator_list[0] = (double* )malloc(g_nsample * sizeof_spinor_field);
  for(i=1; i<g_nsample; i++) stochastic_propagator_list[i] = stochastic_propagator_list[i-1] + _GSI(VOLUME);

  stochastic_source_list = (double**)malloc(g_nsample * sizeof(double*));
  stochastic_source_list[0] = (double*)malloc(g_nsample * sizeof_spinor_field);
  for(i=1; i<g_nsample; i++) stochastic_source_list[i] = stochastic_source_list[i-1] + _GSI(VOLUME);


  /* loop on stochastic samples */
  for(isample = 0; isample < g_nsample; isample++) {

    /* set a stochstic volume source */
    exitstatus = prepare_volume_source(stochastic_source_list[isample], VOLUME);
    if(exitstatus != 0) {
      fprintf(stderr, "[piN2piN] Error from prepare_volume_source, status was %d\n", exitstatus);
      EXIT(39);
    }

    memset( stochastic_propagator_list[isample], 0, sizeof_spinor_field);


    /* project to timeslices, invert */
    for(i_src = 0; i_src < stochastic_source_timeslice_number; i_src++) {

      
      /******************************************************
       * i_src is just a counter; we take the timeslices from
       * the list stochastic_source_timeslice_list, which are
       * in some order;
       * t_src should be used to address the fields
       ******************************************************/
      int t_src = stochastic_source_timeslice_list[i_src];
      memset(spinor_work[0], 0, sizeof_spinor_field);

      int have_source = ( g_proc_coords[0] == t_src / T );
      if( have_source ) {
        fprintf(stdout, "# [piN2piN] proc %4d = ( %d, %d, %d, %d) has t_src = %3d \n", g_cart_id, 
            g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3], t_src);
        /* this process copies timeslice t_src%T from source */
        unsigned int shift = _GSI(g_ipt[t_src%T][0][0][0]);
        memcpy(spinor_work[0]+shift, stochastic_source_list[isample]+shift, sizeof_spinor_field_timeslice );
      }

      /* tm-rotate stochastic source */
      if( g_fermion_type == _TM_FERMION ) {
        spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], -1, g_fermion_type, VOLUME);
      }

      memset(spinor_work[1], 0, sizeof_spinor_field);
      exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_dn, 0);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN] Error from tmLQCD_invert, status was %d\n", exitstatus);
        EXIT(12);
      }

      /* tm-rotate stochastic propagator at sink */
      if( g_fermion_type == _TM_FERMION ) {
        spinor_field_tm_rotation(spinor_work[1], spinor_work[1], -1, g_fermion_type, VOLUME);
      }

      /* copy only source timeslice from propagator */
      if(have_source) {
        unsigned int shift = _GSI(g_ipt[t_src%T][0][0][0]);
        memcpy( stochastic_propagator_list[isample]+shift, spinor_work[1]+shift, sizeof_spinor_field_timeslice);
      }

    }
 
    /* source-smear the stochastic source */
    exitstatus = Jacobi_Smearing(gauge_field_smeared, stochastic_source_list[isample], N_Jacobi, kappa_Jacobi);

    /* sink-smear the stochastic propagator */
    exitstatus = Jacobi_Smearing(gauge_field_smeared, stochastic_propagator_list[isample], N_Jacobi, kappa_Jacobi);

  }  /* end of loop on samples */

  /******************************************************
   ******************************************************
   **
   ** contractions using stochastic propagator
   **
   ** B and W diagrams
   ******************************************************
   ******************************************************/

  /* loop on base source locations */
  for(i_src = 0; i_src < g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

#ifdef HAVE_LHPC_AFF
    /***********************************************
     * open aff output file
     ***********************************************/
    
    if(io_proc == 2) {
      aff_status_str = (char*)aff_version();
      fprintf(stdout, "# [piN2piN] using aff version %s\n", aff_status_str);
    
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "MB_MB", Nconf, t_base );
      fprintf(stdout, "# [piN2piN] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      }
    
      if( (affn = aff_writer_root(affw)) == NULL ) {
        fprintf(stderr, "[piN2piN] Error, aff writer is not initialized\n");
        EXIT(5);
      }
  
      aff_buffer = (double _Complex*)malloc(T_global*g_sv_dim*g_sv_dim*sizeof(double _Complex));
      if(aff_buffer == NULL) {
        fprintf(stderr, "[piN2piN] Error from malloc\n");
        EXIT(6);
      }
    }  /* end of if io_proc == 2 */
#endif

    double **tffi_list=NULL, **pffii_list=NULL;
    if( (exitstatus = init_2level_buffer(&tffi_list, n_s*n_c, _GSI(VOLUME)) ) != 0 ) {
      fprintf(stderr, "[piN2piN] Error from init_2level_buffer, status was %d\n", exitstatus);
      EXIT(50);
    }
    if( (exitstatus = init_2level_buffer(&pffii_list, n_s*n_c, _GSI(VOLUME)) ) != 0 ) {
      fprintf(stderr, "[piN2piN] Error from init_2level_buffer, status was %d\n", exitstatus);
      EXIT(51);
    }

    /* loop on coherent source locations */
    for(i_coherent=0; i_coherent < g_coherent_source_number; i_coherent++) {

      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;
      int i_prop = i_src * g_coherent_source_number + i_coherent;
      gsx[0] = t_coherent;
      gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
      gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
      gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;
      get_point_source_info (gsx, sx, &source_proc_id);

      /* loop on pi2 */
      for(iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {
        if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] pi2 = (%d, %d, %d)\n",
            g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2]);

        int i_seq_prop = iseq_mom * g_source_location_number + i_src;


        /* loop on pf2 */
        for(iseq2_mom=0; iseq2_mom < g_seq2_source_momentum_number; iseq2_mom++) {
          if(g_cart_id == 0) fprintf(stdout, "# [piN2piN] pf2 = (%d, %d, %d)\n",
              g_seq2_source_momentum_list[iseq2_mom][0], g_seq2_source_momentum_list[iseq2_mom][1], g_seq2_source_momentum_list[iseq2_mom][2]);

          /* prepare the tffi propagator */
          exitstatus = prepare_seqn_stochastic_vertex_propagator_sliced3d (tffi_list, stochastic_propagator_list, stochastic_source_list,
              &(propagator_list_up[i_prop*n_s*n_c]), g_nsample, n_s*n_c, g_seq2_source_momentum_list[iseq2_mom], 5);

          /* prepare the pffii propagator */
          exitstatus = prepare_seqn_stochastic_vertex_propagator_sliced3d (pffii_list, stochastic_propagator_list, stochastic_source_list,
              &(sequential_propagator_list[i_seq_prop*n_s*n_c]), g_nsample, n_s*n_c, g_seq2_source_momentum_list[iseq2_mom], 5);


          /* contractions */

          ratime = _GET_TIME;

          for(i=0; i<max_num_diagram; i++) { memset(conn_X[i][0][0], 0, 2*VOLUME*g_sv_dim*g_sv_dim*sizeof(double)); }

          exitstatus = contract_piN_piN (conn_X,
              &(propagator_list_up[i_prop*n_s*n_c]), &(propagator_list_dn[i_prop*n_s*n_c]), 
              &(sequential_propagator_list[i_seq_prop*n_s*n_c]),
              tffi_list,
              pffii_list,
              num_component_piN_piN, gamma_component_piN_piN, gamma_component_sign_piN_piN);

          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN] Error from contract_piN_piN, status was %d\n", exitstatus);
            EXIT(41);
          }

          retime = _GET_TIME;
          if(g_cart_id == 0)  fprintf(stdout, "# [piN2piN] time for contractions = %e seconds\n", retime-ratime);
  
          for(i=0; i<6; i++) {
            /* phase from quark field boundary condition */
            add_baryon_boundary_phase (conn_X[i], gsx[0], num_component_piN_piN);

            /* momentum projection */
            double ****connt=NULL;
            if( (exitstatus = init_4level_buffer(&connt, T, g_sink_momentum_number, num_component_piN_piN*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
              fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
              EXIT(52);
            }
            for(it=0; it<T; it++) {
              exitstatus = momentum_projection2 (conn_X[i][it*VOL3*num_component_piN_piN][0], connt[it][0][0],
                  num_component_piN_piN*g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, NULL );
            }
            /* add complex phase from source location and source momentum
             *   assumes momentum conservation 
             */
            add_source_phase (connt, g_seq_source_momentum_list[iseq_mom], g_seq2_source_momentum_list[iseq2_mom], &(gsx[1]), num_component_piN_piN);

            /* write to file */
       
            ratime = _GET_TIME;
#ifdef HAVE_MPI
            if(io_proc>0) {
              if( (exitstatus = init_4level_buffer(&buffer, T_global, g_sink_momentum_number, num_component_piN_piN*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
                fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
                EXIT(53);
              }
              k = T * g_sink_momentum_number * num_component_piN_piN * g_sv_dim * g_sv_dim * 2;
              exitstatus = MPI_Allgather(connt[0][0][0], k, MPI_DOUBLE, buffer[0][0][0], k, MPI_DOUBLE, g_tr_comm);
              if(exitstatus != MPI_SUCCESS) {
                fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
                EXIT(124);
              }
            }
#else
            buffer = connt;
#endif
    
            if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
              for(k=0; k<g_sink_momentum_number; k++) {

                for(icomp=0; icomp<num_component_piN_piN; icomp++) {

                  sprintf(aff_buffer_path, "/%s/diag%d/pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                      "pixN-pixN", i,
                      g_seq_source_momentum_list[iseq_mom][0],   g_seq_source_momentum_list[iseq_mom][1],   g_seq_source_momentum_list[iseq_mom][2],
                      g_sink_momentum_list[k][0],                g_sink_momentum_list[k][1],                g_sink_momentum_list[k][2],
                      g_seq2_source_momentum_list[iseq2_mom][0], g_seq2_source_momentum_list[iseq2_mom][1], g_seq2_source_momentum_list[iseq2_mom][2],
                      gsx[0], gsx[1], gsx[2], gsx[3],
                      gamma_component_piN_piN[icomp][0], gamma_component_piN_piN[icomp][1]);

                  fprintf(stdout, "# [piN2piN] current aff path = %s\n", aff_buffer_path);

                  affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
                  for(it=0; it<T_global; it++) {
                    ir = ( it - gsx[0] + T_global ) % T_global;
                    memcpy(aff_buffer + ir*g_sv_dim*g_sv_dim,  buffer[it][k][icomp*g_sv_dim] , g_sv_dim*g_sv_dim*sizeof(double _Complex) );
                  }
                  int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global*g_sv_dim*g_sv_dim);
                  if(status != 0) {
                    fprintf(stderr, "[piN2piN] Error from aff_node_put_complex, status was %d\n", status);
                    EXIT(84);
                  }
                }  /* end of loop on components */
              }  /* end of loop on sink momenta */
#endif
            }  /* end of if io_proc == 2 */
    
#ifdef HAVE_MPI
            if(io_proc > 0) { fini_4level_buffer(&buffer); }
#endif

            fini_4level_buffer(&connt);

            retime = _GET_TIME;
            if( io_proc == 2 ) fprintf(stdout, "# [piN2piN] time for writing piN-piN BW = %e seconds\n", retime-ratime);
          }

        }  /* end of loop on pf2 */
      } /* end of loop on pi2 */

    }  /* end of loop on coherent source locations */

    fini_2level_buffer(&tffi_list);
    fini_2level_buffer(&pffii_list);
    
#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(11);
      }
      if(aff_buffer != NULL) free(aff_buffer);
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

  }  /* end of loop on base source locations */
   
  /* sequential propagator list not needed after this point */ 
  free( sequential_propagator_list[0]);
  free( sequential_propagator_list);
  /* stochast source and propagators will be used differently */
  free( stochastic_source_list[0] );
  free( stochastic_source_list );
  free( stochastic_propagator_list[0] );
  free( stochastic_propagator_list );



  /***********************************************
   ***********************************************
   **
   ** stochastic contractions using the 
   **   one-end-trick
   **
   ***********************************************
   ***********************************************/
  no_fields = 8;
  stochastic_propagator_list    = (double**)malloc( no_fields * sizeof(double*));
  stochastic_propagator_list[0] = (double* )malloc(no_fields * sizeof_spinor_field);
  for(i=1; i<no_fields; i++) stochastic_propagator_list[i] = stochastic_propagator_list[i-1] + _GSI(VOLUME);

  no_fields = 4;
  stochastic_source_list = (double**)malloc(no_fields * sizeof(double*));
  stochastic_source_list[0] = (double*)malloc(no_fields * sizeof_spinor_field);
  for(i=1; i < no_fields; i++) stochastic_source_list[i] = stochastic_source_list[i-1] + _GSI(VOLUME);


  /* loop on oet samples */
  for(isample=0; isample < g_nsample_oet; isample++) {

    double **pfifi_list = NULL;
    if( (exitstatus = init_2level_buffer(&pfifi_list, n_s*n_c, _GSI(VOLUME)) ) != 0 ) {
      fprintf(stderr, "[piN2piN] Error from init_2level_buffer, status was %d\n", exitstatus);
      EXIT(54);
    }

    for(i_src=0; i_src < g_source_location_number; i_src++) {
      int t_base = g_source_coords_list[i_src][0];

#ifdef HAVE_LHPC_AFF
      /***********************************************
       * open aff output file
       ***********************************************/
    
      if(io_proc == 2) {
        aff_status_str = (char*)aff_version();
        fprintf(stdout, "# [piN2piN] using aff version %s\n", aff_status_str);
    
        sprintf(filename, "%s.%.4d.sample%.2d.tsrc%.2d.aff", "piN_piN_oet", Nconf, isample, t_base );
        fprintf(stdout, "# [piN2piN] writing data to file %s\n", filename);
        affw = aff_writer(filename);
        aff_status_str = (char*)aff_writer_errstr(affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[piN2piN] Error from aff_writer, status was %s\n", aff_status_str);
          EXIT(4);
        }
    
        if( (affn = aff_writer_root(affw)) == NULL ) {
          fprintf(stderr, "[piN2piN] Error, aff writer is not initialized\n");
          EXIT(5);
        }
  
        aff_buffer = (double _Complex*)malloc(T_global*g_sv_dim*g_sv_dim*sizeof(double _Complex));
        if(aff_buffer == NULL) {
          fprintf(stderr, "[piN2piN] Error from malloc\n");
          EXIT(6);
        }
      }  /* end of if io_proc == 2 */
#endif

      for(i_coherent = 0; i_coherent < g_coherent_source_number; i_coherent++) {

        int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;
        int i_prop = i_src * g_coherent_source_number + i_coherent;
        gsx[0] = t_coherent;
        gsx[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
        gsx[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
        gsx[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;

        if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], NULL, 1)) != 0 ) {
          fprintf(stderr, "[piN2piN] Error from init_timeslice_source_oet, status was %d\n", exitstatus);
          EXIT(63);
        }

        /* zero-momentum propagator */
        for(i=0; i<4; i++) {
          memcpy(spinor_work[0], stochastic_source_list[i], sizeof_spinor_field);

          /* source-smearing stochastic momentum source */
          exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

          /* tm-rotate stochastic source */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
          }

          memset(spinor_work[1], 0, sizeof_spinor_field);
          exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(44);
          }

          /* tm-rotate stochastic propagator at sink */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
          }

          /* sink smearing stochastic propagator */
          exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);


          memcpy( stochastic_propagator_list[i], spinor_work[1], sizeof_spinor_field);
        }

        for(iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {

          if( (exitstatus = init_timeslice_source_oet(stochastic_source_list, gsx[0], g_seq_source_momentum_list[iseq_mom], 0) ) != 0 ) {
            fprintf(stderr, "[piN2piN] Error from init_timeslice_source_oet, status was %d\n", exitstatus);
            EXIT(64);
          }

          /* nonzero-momentum propagator */
          for(i=0; i<4; i++) {
            memcpy(spinor_work[0], stochastic_source_list[i], sizeof_spinor_field);

            /* source-smearing stochastic momentum source */
            exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[0], N_Jacobi, kappa_Jacobi);

            /* tm-rotate stochastic source */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], +1, g_fermion_type, VOLUME);
            }

            memset(spinor_work[1], 0, sizeof_spinor_field);
            exitstatus = tmLQCD_invert(spinor_work[1], spinor_work[0], op_id_up, 0);
            if(exitstatus != 0) {
              fprintf(stderr, "[piN2piN] Error from tmLQCD_invert, status was %d\n", exitstatus);
              EXIT(44);
            }

            /* tm-rotate stochastic propagator at sink */
            if( g_fermion_type == _TM_FERMION ) {
              spinor_field_tm_rotation(spinor_work[1], spinor_work[1], +1, g_fermion_type, VOLUME);
            }

            /* sink smearing stochastic propagator */
            exitstatus = Jacobi_Smearing(gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);

            memcpy( stochastic_propagator_list[4+i], spinor_work[1], sizeof_spinor_field);
          }

          /***********************
           ***********************
           **
           ** pi - pi 2-pt
           **
           ** we abuse conn_X here, which is an array for spin-propagator fields;
           ** but its entry conn_X(0,0,0) points to space several times VOLUME,
           ** which is sufficiently large for meson-meson contractions
           **
           ***********************
           ***********************/
          for(i=0; i<max_num_diagram; i++) { memset(conn_X[i][0][0], 0, 2*VOLUME*g_sv_dim*g_sv_dim*sizeof(double)); }
          double *conn_M = conn_X[0][0][0];
          contract_twopoint_xdep( (void*)conn_M, 5, 5, (void*)(&(stochastic_propagator_list[0])), (void*)(&(stochastic_propagator_list[4])), 1, 1, 1., 64);
    
          double **connt = NULL;
          if( (exitstatus = init_2level_buffer(&connt, T, 2*g_sink_momentum_number ) ) != 0 ) {
            fprintf(stderr, "[piN2piN] Error from init_2level_buffer, status was %d\n", exitstatus);
            EXIT(61);
          }
    
          /* momentum projection */
          exitstatus = momentum_projection3 ( conn_M, connt[0], T, g_sink_momentum_number, g_sink_momentum_list);
          if(exitstatus != 0) {
            fprintf(stderr, "[piN2piN] Error from momentum_projection, status was %d\n", exitstatus);
            EXIT(8);
          }
    
          /* write to file */
    
          ratime = _GET_TIME;
          double **buffer2 = NULL;
#ifdef HAVE_MPI
          if(io_proc>0) {
            if( (exitstatus = init_2level_buffer(&buffer2, T_global, 2*g_sink_momentum_number ) ) != 0 ) {
              fprintf(stderr, "[piN2piN] Error from init_2level_buffer, status was %d\n", exitstatus);
              EXIT(62);
            }
            k = T * g_sink_momentum_number * 2;
            exitstatus = MPI_Allgather(connt[0], k, MPI_DOUBLE, buffer2[0], k, MPI_DOUBLE, g_tr_comm);
            if(exitstatus != MPI_SUCCESS) {
              fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
              EXIT(124);
            }
          }
#else
          buffer2 = connt;
#endif
    
          if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
            for(k=0; k<g_sink_momentum_number; k++) {
    
              sprintf(aff_buffer_path, "/%s/pi1x%.2dpi1y%.2dpi1z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                  "m-m",
                  g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2],
                  g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2],
                  gsx[0], gsx[1], gsx[2], gsx[3], 5, 5);
    
              fprintf(stdout, "# [piN2piN] current aff path = %s\n", aff_buffer_path);
    
              affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
              for(it=0; it<T_global; it++) {
                ir = ( it - gsx[0] + T_global ) % T_global;
                aff_buffer[ir] = buffer2[it][2*k]  + I * buffer2[it][2*k+1];
              }
              int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global);
              if(status != 0) {
                fprintf(stderr, "[piN2piN] Error from aff_node_put_complex, status was %d\n", status);
                EXIT(85);
              }
    
            }  /* end of loop on sink momenta */
#endif
          }  /* end of if io_proc == 2 */
    
#ifdef HAVE_MPI
          if(io_proc > 0) { fini_2level_buffer(&buffer2); }
#endif
          fini_2level_buffer(&connt);
          retime = _GET_TIME;
          if( io_proc == 2 ) fprintf(stdout, "# [piN2piN] time for writing pi-pi = %e seconds\n", retime-ratime);

          /***********************************************
           ***********************************************
           **
           ** piN - piN
           ** Z-type diagram contraction
           **
           ***********************************************
           ***********************************************/
          /* loop on seq2 source momentum pf2 */
          for(iseq2_mom=0; iseq2_mom < g_seq2_source_momentum_number; iseq2_mom++) {

            /* prepare pfifi */
            exitstatus = prepare_seqn_stochastic_vertex_propagator_sliced3d_oet (pfifi_list, stochastic_propagator_list, &(stochastic_propagator_list[4]),
                    &(propagator_list_up[i_prop*n_s*n_c]), g_seq2_source_momentum_list[iseq2_mom], 5, 5);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN] Error from prepare_seqn_stochastic_vertex_propagator_sliced3d_oet, status was %d\n", exitstatus);
              EXIT(45);
            }

            /* contract 4 oet diagrams */

            for(i=0; i<max_num_diagram; i++) { memset(conn_X[i][0][0], 0, 2*VOLUME*g_sv_dim*g_sv_dim*sizeof(double)); }

            exitstatus = contract_piN_piN_oet ( conn_X, &(propagator_list_up[i_prop*n_s*n_c]), &(propagator_list_dn[i_prop*n_s*n_c]), 
                pfifi_list, num_component_piN_piN, gamma_component_piN_piN, gamma_component_sign_piN_piN);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN] Error from contract_piN_piN_oet, status was %d\n", exitstatus);
              EXIT(46);
            }
  
            /* write to file etc. */
            for(i=0; i<4; i++) {
              /* phase from quark field boundary condition */
              add_baryon_boundary_phase (conn_X[i], gsx[0], num_component_piN_piN);
  
              /* momentum projection */
              double ****connt = NULL;
              if( (exitstatus = init_4level_buffer(&connt, T, g_sink_momentum_number, num_component_piN_piN*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
                fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
                EXIT(55);
              }
  
              for(it=0; it<T; it++) {
                exitstatus = momentum_projection2 (conn_X[i][it*VOL3*num_component_piN_piN][0], connt[it][0][0],
                    num_component_piN_piN*g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, NULL );
              }
              /* add complex phase from source location and source momentum
               *   assumes momentum conservation 
               */
              add_source_phase (connt, g_seq_source_momentum_list[iseq_mom], g_seq2_source_momentum_list[iseq2_mom], &(gsx[1]), num_component_piN_piN);
  
              /* write to file */
         
              ratime = _GET_TIME;
#ifdef HAVE_MPI
              if(io_proc>0) {
                if( (exitstatus = init_4level_buffer(&buffer, T_global, g_sink_momentum_number, num_component_piN_piN*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
                  fprintf(stderr, "[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
                  EXIT(56);
                }
                k = T * g_sink_momentum_number * num_component_piN_piN * g_sv_dim * g_sv_dim * 2;
                exitstatus = MPI_Allgather(connt[0][0][0], k, MPI_DOUBLE, buffer[0][0][0], k, MPI_DOUBLE, g_tr_comm);
                if(exitstatus != MPI_SUCCESS) {
                  fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
                  EXIT(124);
                }
              }
#else
              buffer = connt;
#endif
      
              if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF
                for(k=0; k<g_sink_momentum_number; k++) {
  
                  for(icomp=0; icomp<num_component_piN_piN; icomp++) {
  
                    sprintf(aff_buffer_path, "/%s/sample%d/diag%d/pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                        "pixN-pixN", isample, i,
                        g_seq_source_momentum_list[iseq_mom][0],   g_seq_source_momentum_list[iseq_mom][1],   g_seq_source_momentum_list[iseq_mom][2],
                        g_sink_momentum_list[k][0],                g_sink_momentum_list[k][1],                g_sink_momentum_list[k][2],
                        g_seq2_source_momentum_list[iseq2_mom][0], g_seq2_source_momentum_list[iseq2_mom][1], g_seq2_source_momentum_list[iseq2_mom][2],
                        gsx[0], gsx[1], gsx[2], gsx[3],
                        gamma_component_piN_piN[icomp][0], gamma_component_piN_piN[icomp][1]);
  
                    fprintf(stdout, "# [piN2piN] current aff path = %s\n", aff_buffer_path);
  
                    affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
                    for(it=0; it<T_global; it++) {
                      ir = ( it - gsx[0] + T_global ) % T_global;
                      memcpy(aff_buffer + ir*g_sv_dim*g_sv_dim,  buffer[it][k][icomp*g_sv_dim] , g_sv_dim*g_sv_dim*sizeof(double _Complex) );
                    }
                    int status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)T_global*g_sv_dim*g_sv_dim);
                    if(status != 0) {
                      fprintf(stderr, "[piN2piN] Error from aff_node_put_double, status was %d\n", status);
                      EXIT(86);
                    }
                  }  /* end of loop on components */
                }  /* end of loop on sink momenta */
#endif
              }  /* end of if io_proc == 2 */
      
#ifdef HAVE_MPI
              if(io_proc > 0) { fini_4level_buffer(&buffer); }
#endif
  
              fini_4level_buffer(&connt);

              retime = _GET_TIME;
              if( io_proc == 2 ) fprintf(stdout, "# [piN2piN] time for writing piN-piN Z = %e seconds\n", retime-ratime);
            }  /* end of loop on diagrams */

          }  /* end of loop on seq2 source momentum pf2 */

        }  /* end of loop on sequential source momenta pi2 */
      }  /* end of loop on coherent sources */

#ifdef HAVE_LHPC_AFF
      if(io_proc == 2) {
        aff_status_str = (char*)aff_writer_close (affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[piN2piN] Error from aff_writer_close, status was %s\n", aff_status_str);
          EXIT(111);
        }
        if(aff_buffer != NULL) free(aff_buffer);
      }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    }  /* end of loop on base sources */

    fini_2level_buffer(&pfifi_list);
  }  /* end of loop on oet samples */



  /***********************************************
   * free gauge fields and spinor fields
   ***********************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  if(g_gauge_field != NULL) {
    free(g_gauge_field);
    g_gauge_field=(double*)NULL;
  }
#endif


  free(propagator_list_up[0]);
  free(propagator_list_up);
  if( g_fermion_type == _TM_FERMION ) {
    free(propagator_list_dn[0]);
    free(propagator_list_dn);
  }

  free( stochastic_source_list[0] );
  free( stochastic_source_list );

  free( stochastic_propagator_list[0] );
  free( stochastic_propagator_list );


  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();
  for(i=0; i<max_num_diagram; i++) { free_sp_field( &(conn_X[i]) ); }
  free(conn_X);

  if( N_Jacobi > 0 && N_ape > 0 ) {
    if( gauge_field_smeared != NULL ) free(gauge_field_smeared);
  }
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN] %s# [piN2piN] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN] %s# [piN2piN] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
