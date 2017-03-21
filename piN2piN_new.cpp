/****************************************************
 * piN2piN.cpp
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

#ifndef HAVE_TMLQCD_LIBWRAPPER
#error "need tmLQCD lib wrapper"
#endif

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

#include <sys/stat.h>

using namespace cvc;

#include "basic_types.h"

const int n_c = 3;
const int n_s = 4;

inline bool test_whether_pathname_exists(pathname_type pathname){
  struct stat buffer;
  return (stat(pathname,&buffer) == 0);
}

/*
 * Functions to reduce output
 * */
int mpi_fprintf(FILE* stream, const char * format, ...){
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
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
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
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

int write_to_stdout(const char * format, ...){
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
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

int write_to_stderr(const char * format, ...){
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if(world_rank == 0){
    va_list arg;
    int done;

    va_start(arg,format);
    done = vfprintf(stderr,format,arg);
    va_end(arg);

    return done;
  }
  return 0;
}

void print_spinor_field(const char *msg,double *spinor_field){
  int i;
  for(i = 0;i < 4;i++){
    write_to_stdout("%s: %d %e\n",msg,i,spinor_field[i]);
  }
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



// gamma componenets
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

int get_forward_complete_is_propagator_index(int i_src,int i_coherent){
  return i_src * g_coherent_source_number + i_coherent;
}

int get_sequential_complete_is_propagator_index(int iseq_mom,int i_src){
  return iseq_mom * g_source_location_number + i_src;
}

int get_sequential_propagator_index(int iseq_mom,int i_src,int is){
  return (iseq_mom * g_source_location_number + i_src)*n_s*n_c+is;
}

int get_forward_propagator_index(int i_src,int i_coherent,int is){
  return (i_src * g_coherent_source_number + i_coherent)*n_s*n_c+is;
}

void get_global_source_location(global_source_location_type *dest,int i_src){
  dest->x[0] = g_source_coords_list[i_src][0];
  dest->x[1] = g_source_coords_list[i_src][1];
  dest->x[2] = g_source_coords_list[i_src][2];
  dest->x[3] = g_source_coords_list[i_src][3];
}

void get_global_coherent_source_location(global_source_location_type *dest,int i_src,int i_coherent){
  int t_base = g_source_coords_list[i_src][0];
  int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global; 
  dest->x[0] = t_coherent;
  dest->x[1] = ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global;
  dest->x[2] = ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global;
  dest->x[3] = ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global;
}

void convert_global_to_local_source_location(local_source_location_type *dest,global_source_location_type src){
  get_point_source_info (src.x, dest->x, &dest->proc_id);
}

void get_local_coherent_source_location(local_source_location_type *dest,int i_src,int i_coherent){
  global_source_location_type gsl;
  get_global_coherent_source_location(&gsl,i_src,i_coherent);
  convert_global_to_local_source_location(dest,gsl);
}

void set_memory_for_Whick_Dirac_and_color_contractions_to_zero(program_instruction_type *program_instructions){
  int i;
  for(i=0; i<program_instructions->max_num_diagram; i++) { memset(program_instructions->conn_X[i][0][0], 0, 2*VOLUME*g_sv_dim*g_sv_dim*sizeof(double)); }
}

void get_sink_momentum(three_momentum_type *momentum,int i_sink_momentum){
  momentum->p[0] = g_sink_momentum_list[i_sink_momentum][0];
  momentum->p[1] = g_sink_momentum_list[i_sink_momentum][1];
  momentum->p[2] = g_sink_momentum_list[i_sink_momentum][2];
}

void get_seq_source_momentum(three_momentum_type *momentum,int iseq_mom){
  momentum->p[0] = g_seq_source_momentum_list[iseq_mom][0];
  momentum->p[1] = g_seq_source_momentum_list[iseq_mom][1];
  momentum->p[2] = g_seq_source_momentum_list[iseq_mom][2];
}

void set_three_momentum_to_three_momentum(three_momentum_type *dest,three_momentum_type src){
  dest->p[0] = src.p[0];
  dest->p[1] = src.p[1];
  dest->p[2] = src.p[2];
}

void set_three_momentum_to_zero(three_momentum_type *dest){
  dest->p[0] = 0;
  dest->p[1] = 0;
  dest->p[2] = 0;
}

void init_information_needed_for_source_phase_so_that_no_source_phase_is_computed(information_needed_for_source_phase_type *information_needed_for_source_phase){
  information_needed_for_source_phase->add_source_phase = false;
  set_three_momentum_to_zero(&information_needed_for_source_phase->pi2);
  set_three_momentum_to_zero(&information_needed_for_source_phase->pf2);
}

void init_information_needed_for_source_phase_so_that_source_phase_with_no_additional_momenta_is_computed(information_needed_for_source_phase_type *information_needed_for_source_phase){
  information_needed_for_source_phase->add_source_phase = true;
  set_three_momentum_to_zero(&information_needed_for_source_phase->pi2);
  set_three_momentum_to_zero(&information_needed_for_source_phase->pf2);
}

void init_information_needed_for_source_phase_so_that_source_phase_with_sequential_source_momentum_is_computed(information_needed_for_source_phase_type *information_needed_for_source_phase,three_momentum_type seq_source_momentum){
  information_needed_for_source_phase->add_source_phase = true;
  set_three_momentum_to_three_momentum(&information_needed_for_source_phase->pi2,seq_source_momentum);
  set_three_momentum_to_zero(&information_needed_for_source_phase->pf2);
}


void get_aff_key_for_N_N_contractions(pathname_type aff_key_to_write_contractions_to,int diagram,three_momentum_type sink_momentum,global_source_location_type gsl,int icomp){
  sprintf(aff_key_to_write_contractions_to, "/%s/diag%d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
    "N-N", diagram,
    sink_momentum.p[0],                sink_momentum.p[1],                sink_momentum.p[2],
    gsl.x[0], gsl.x[1], gsl.x[2], gsl.x[3],
    gamma_component_N_N[icomp][0], gamma_component_N_N[icomp][1]);
}

void store_contraction_under_aff_key(contraction_writer_type *contraction_writer,pathname_type aff_key_to_write_contractions_to,gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,global_source_location_type gsl,int i_sink_momentum,int icomp,int exit_code){
  write_to_stdout("# [piN2piN] current aff path = %s\n", aff_key_to_write_contractions_to);

  contraction_writer->affdir = aff_writer_mkpath(contraction_writer->affw, contraction_writer->affn, aff_key_to_write_contractions_to);
  int it;
  for(it=0; it<T_global; it++) {
    int ir = ( it - gsl.x[0] + T_global ) % T_global;
    memcpy(contraction_writer->aff_buffer + ir*g_sv_dim*g_sv_dim,  (*gathered_FT_WDc_contractions)[it][i_sink_momentum][icomp*g_sv_dim] , g_sv_dim*g_sv_dim*sizeof(double _Complex) );
  }
  int status = aff_node_put_complex (contraction_writer->affw, contraction_writer->affdir, contraction_writer->aff_buffer, (uint32_t)T_global*g_sv_dim*g_sv_dim);
  if(status != 0) {
    write_to_stderr("[piN2piN] Error from aff_node_put_double, status was %d\n", status);
    EXIT(exit_code);
  }
}

void store_N_N_contractions(contraction_writer_type *contraction_writer,gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,int diagram,global_source_location_type gsl,program_instruction_type *program_instructions,int exit_code){
  int k,icomp;
  if(program_instructions->io_proc == 2) {
    for(k=0; k<g_sink_momentum_number; k++) {
      for(icomp=0; icomp<num_component_N_N; icomp++) {
        pathname_type aff_key_to_write_contractions_to;
        three_momentum_type sink_momentum;
        get_sink_momentum(&sink_momentum,k);
        get_aff_key_for_N_N_contractions(aff_key_to_write_contractions_to,diagram,sink_momentum,gsl,icomp);
        store_contraction_under_aff_key(contraction_writer,aff_key_to_write_contractions_to,gathered_FT_WDc_contractions,gsl,k,icomp,exit_code);
      }
    }
  }
}

void add_baryon_boundary_phase_to_WDc_contractions(program_instruction_type *program_instructions,int diagram,int t_coherent,int num_component){
  add_baryon_boundary_phase(program_instructions->conn_X[diagram], t_coherent, num_component);
}

void init_FT_WDc_contractions(FT_WDc_contractions_type *FT_WDc_contractions,int num_component,int exit_code){
  (*FT_WDc_contractions) = NULL;
  int exitstatus;
  if( (exitstatus = init_4level_buffer(FT_WDc_contractions, T, g_sink_momentum_number, num_component * g_sv_dim, 2*g_sv_dim) ) != 0 ) {
    write_to_stderr("[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
    EXIT(exit_code);
  }  
}

void compute_fourier_transformation_on_local_lattice_from_WDc_contractions(FT_WDc_contractions_type *FT_WDc_contractions,program_instruction_type *program_instructions,int diagram,int num_component){
  int it,exitstatus;
  for(it=0; it<T; it++) {
    exitstatus = momentum_projection2 (program_instructions->conn_X[diagram][it*program_instructions->VOL3*num_component][0], (*FT_WDc_contractions)[it][0][0], num_component*g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, NULL );
  }
}

void init_gathered_FT_WDc_contractions(gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,int num_component,program_instruction_type *program_instructions,int exit_code){
  int exitstatus;
  if(program_instructions->io_proc>0) {
    (*gathered_FT_WDc_contractions) = NULL;
    if( (exitstatus = init_4level_buffer(gathered_FT_WDc_contractions, T_global, g_sink_momentum_number, num_component*g_sv_dim, 2*g_sv_dim) ) != 0 ) {
      write_to_stderr("[piN2piN] Error from init_4level_buffer, status was %d\n", exitstatus);
      EXIT(exit_code);
    }
  }
}

void gather_FT_WDc_contractions_on_timeline(gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,FT_WDc_contractions_type *FT_WDc_contractions,int num_component,program_instruction_type *program_instructions,int exit_code){
  int exitstatus;
  if(program_instructions->io_proc>0) {
    int k = T * g_sink_momentum_number * num_component * g_sv_dim * g_sv_dim * 2;
    exitstatus = MPI_Allgather((*FT_WDc_contractions)[0][0][0], k, MPI_DOUBLE, (*gathered_FT_WDc_contractions)[0][0][0], k, MPI_DOUBLE, g_tr_comm);
    if(exitstatus != MPI_SUCCESS) {
      write_to_stderr("[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
      EXIT(exit_code);
    }
  }
}

void set_gathered_FT_WDc_contractions_to_FT_WDc_contractions(gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,FT_WDc_contractions_type *FT_WDc_contractions){
  (*gathered_FT_WDc_contractions) = (*FT_WDc_contractions);
}

void exit_gathered_FT_WDc_contractions(gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,program_instruction_type *program_instructions){
  if(program_instructions->io_proc > 0) { fini_4level_buffer(gathered_FT_WDc_contractions); }
}

void exit_FT_WDc_contractions(FT_WDc_contractions_type *FT_WDc_contractions){
  fini_4level_buffer(FT_WDc_contractions);
}

void allocate_memory_for_ft_and_gathering(FT_WDc_contractions_type *FT_WDc_contractions,gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,int num_component,program_instruction_type *program_instructions,int exit_code_1,int exit_code_2){
    init_FT_WDc_contractions(FT_WDc_contractions,num_component,exit_code_1);
#ifdef HAVE_MPI
    init_gathered_FT_WDc_contractions(gathered_FT_WDc_contractions,num_component,program_instructions,exit_code_2);
#endif
}

void free_memory_for_ft_and_gathering(FT_WDc_contractions_type *FT_WDc_contractions,gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,program_instruction_type *program_instructions){
#ifdef HAVE_MPI
    exit_gathered_FT_WDc_contractions(gathered_FT_WDc_contractions,program_instructions);
#endif
    exit_FT_WDc_contractions(FT_WDc_contractions);
}

void add_source_phase_to_FT_WDc_contractions(FT_WDc_contractions_type *FT_WDc_contractions,information_needed_for_source_phase_type *information_needed_for_source_phase,global_source_location_type gsl,int num_component){
  add_source_phase((*FT_WDc_contractions), information_needed_for_source_phase->pi2.p,information_needed_for_source_phase->pf2.p, &(gsl.x[1]), num_component);
}

void compute_gathered_FT_WDc_contractions(FT_WDc_contractions_type *FT_WDc_contractions,gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,global_source_location_type gsl,int diagram,int num_component,program_instruction_type *program_instructions,information_needed_for_source_phase_type *information_needed_for_source_phase,int exit_code_1){
  add_baryon_boundary_phase_to_WDc_contractions(program_instructions,diagram,gsl.x[0],num_component);
  compute_fourier_transformation_on_local_lattice_from_WDc_contractions(FT_WDc_contractions,program_instructions,diagram,num_component);
  if(information_needed_for_source_phase->add_source_phase){
    add_source_phase_to_FT_WDc_contractions(FT_WDc_contractions,information_needed_for_source_phase,gsl,num_component);
  }
#ifdef HAVE_MPI
  gather_FT_WDc_contractions_on_timeline(gathered_FT_WDc_contractions,FT_WDc_contractions,num_component,program_instructions,exit_code_1);
#else
  set_gathered_FT_WDc_contractions_to_FT_WDc_contractions(gathered_FT_WDc_contractions,FT_WDc_contractions);
#endif
}

void compute_Whick_Dirac_and_color_contractions_for_N_N(int i_src,int i_coherent,program_instruction_type *program_instructions,forward_propagators_type *forward_propagators){
  int i_prop = get_forward_complete_is_propagator_index(i_src,i_coherent);
  int exitstatus = contract_N_N (program_instructions->conn_X, &(forward_propagators->propagator_list_up[i_prop*n_s*n_c]), &(forward_propagators->propagator_list_dn[i_prop*n_s*n_c]) , num_component_N_N, gamma_component_N_N, gamma_component_sign_N_N);
}

void compute_and_store_N_N_contractions_for_diagram(int diagram,global_source_location_type gsl,program_instruction_type *program_instructions,contraction_writer_type *contraction_writer){
  FT_WDc_contractions_type FT_WDc_contractions;
  gathered_FT_WDc_contractions_type gathered_FT_WDc_contractions;
  allocate_memory_for_ft_and_gathering(&FT_WDc_contractions,&gathered_FT_WDc_contractions,num_component_N_N,program_instructions,57,58);

  information_needed_for_source_phase_type information_needed_for_source_phase;
  init_information_needed_for_source_phase_so_that_no_source_phase_is_computed(&information_needed_for_source_phase);

  compute_gathered_FT_WDc_contractions(&FT_WDc_contractions,&gathered_FT_WDc_contractions,gsl,diagram,num_component_N_N,program_instructions,&information_needed_for_source_phase,124);
  store_N_N_contractions(contraction_writer,&gathered_FT_WDc_contractions,diagram,gsl,program_instructions,81);

  free_memory_for_ft_and_gathering(&FT_WDc_contractions,&gathered_FT_WDc_contractions,program_instructions);
}

void compute_and_store_N_N_contractions(int i_src,int i_coherent,global_source_location_type gsl,forward_propagators_type *forward_propagators,sequential_propagators_type *sequential_propagators,program_instruction_type *program_instructions,contraction_writer_type *contraction_writer){
  set_memory_for_Whick_Dirac_and_color_contractions_to_zero(program_instructions);

  compute_Whick_Dirac_and_color_contractions_for_N_N(i_src,i_coherent,program_instructions,forward_propagators);

  int diagram;
  for(diagram=0; diagram<2; diagram++){
    compute_and_store_N_N_contractions_for_diagram(diagram,gsl,program_instructions,contraction_writer);
  }
}

void compute_Whick_Dirac_and_color_contractions_for_D_D(int i_src,int i_coherent,program_instruction_type *program_instructions,forward_propagators_type *forward_propagators){
  int i_prop = get_forward_complete_is_propagator_index(i_src,i_coherent);
  int exitstatus = contract_D_D (program_instructions->conn_X, &(forward_propagators->propagator_list_up[i_prop*n_s*n_c]), &(forward_propagators->propagator_list_dn[i_prop*n_s*n_c]) , num_component_D_D, gamma_component_D_D, gamma_component_sign_D_D);
}

void get_aff_key_for_D_D_contractions(pathname_type aff_key_to_write_contractions_to,int diagram,three_momentum_type sink_momentum,global_source_location_type gsl,int icomp){
  sprintf(aff_key_to_write_contractions_to, "/%s/diag%d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
    "D-D", diagram,
    sink_momentum.p[0],                sink_momentum.p[1],                sink_momentum.p[2],
    gsl.x[0], gsl.x[1], gsl.x[2], gsl.x[3],
    gamma_component_D_D[icomp][0], gamma_component_D_D[icomp][1]);
}

void store_D_D_contractions(contraction_writer_type *contraction_writer,gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,int diagram,global_source_location_type gsl,program_instruction_type *program_instructions,int exit_code){
  int k,icomp;
  if(program_instructions->io_proc == 2) {
    for(k=0; k<g_sink_momentum_number; k++) {
      for(icomp=0; icomp<num_component_D_D; icomp++) {
        pathname_type aff_key_to_write_contractions_to;
        three_momentum_type sink_momentum;
        get_sink_momentum(&sink_momentum,k);
        get_aff_key_for_D_D_contractions(aff_key_to_write_contractions_to,diagram,sink_momentum,gsl,icomp);
        store_contraction_under_aff_key(contraction_writer,aff_key_to_write_contractions_to,gathered_FT_WDc_contractions,gsl,k,icomp,exit_code);
      }
    }
  }
}

void compute_and_store_D_D_contractions_for_diagram(int diagram,global_source_location_type gsl,program_instruction_type *program_instructions,contraction_writer_type *contraction_writer){
  FT_WDc_contractions_type FT_WDc_contractions;
  gathered_FT_WDc_contractions_type gathered_FT_WDc_contractions;
  allocate_memory_for_ft_and_gathering(&FT_WDc_contractions,&gathered_FT_WDc_contractions,num_component_D_D,program_instructions,59,60);

  information_needed_for_source_phase_type information_needed_for_source_phase;
  init_information_needed_for_source_phase_so_that_source_phase_with_no_additional_momenta_is_computed(&information_needed_for_source_phase);

  compute_gathered_FT_WDc_contractions(&FT_WDc_contractions,&gathered_FT_WDc_contractions,gsl,diagram,num_component_D_D,program_instructions,&information_needed_for_source_phase,124);
  store_D_D_contractions(contraction_writer,&gathered_FT_WDc_contractions,diagram,gsl,program_instructions,81);

  free_memory_for_ft_and_gathering(&FT_WDc_contractions,&gathered_FT_WDc_contractions,program_instructions);
}

void compute_and_store_D_D_contractions(int i_src,int i_coherent,global_source_location_type gsl,forward_propagators_type *forward_propagators,sequential_propagators_type *sequential_propagators,program_instruction_type *program_instructions,contraction_writer_type *contraction_writer){
  set_memory_for_Whick_Dirac_and_color_contractions_to_zero(program_instructions);

  compute_Whick_Dirac_and_color_contractions_for_D_D(i_src,i_coherent,program_instructions,forward_propagators);

  int diagram;
  for(diagram=0; diagram<6; diagram++){
    compute_and_store_D_D_contractions_for_diagram(diagram,gsl,program_instructions,contraction_writer);
  }
}

void compute_Whick_Dirac_and_color_contractions_for_piN_D(int iseq_mom,int i_src,int i_coherent,program_instruction_type *program_instructions,forward_propagators_type *forward_propagators,sequential_propagators_type *sequential_propagators){
  int i_prop = get_forward_complete_is_propagator_index(i_src,i_coherent);
  int i_seq_prop = get_sequential_complete_is_propagator_index(iseq_mom,i_src);
  int exitstatus = contract_piN_D (program_instructions->conn_X, &(forward_propagators->propagator_list_up[i_prop*n_s*n_c]), &(forward_propagators->propagator_list_dn[i_prop*n_s*n_c]), 
    &(sequential_propagators->propagator_list[i_seq_prop*n_s*n_c]), num_component_piN_D, gamma_component_piN_D, gamma_component_sign_piN_D);
}

void get_aff_key_for_piN_D_contractions(pathname_type aff_key_to_write_contractions_to,int diagram,three_momentum_type seq_source_momentum,three_momentum_type sink_momentum,global_source_location_type gsl,int icomp){
  sprintf(aff_key_to_write_contractions_to, "/%s/diag%d/pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
    "pixN-D", diagram,
    seq_source_momentum.p[0],   seq_source_momentum.p[1],   seq_source_momentum.p[2],
    sink_momentum.p[0],                sink_momentum.p[1],                sink_momentum.p[2],
    gsl.x[0], gsl.x[1], gsl.x[2], gsl.x[3],
    gamma_component_piN_D[icomp][0], gamma_component_piN_D[icomp][1]);
}

void store_piN_D_contractions(contraction_writer_type *contraction_writer,gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,int iseq_mom,int diagram,global_source_location_type gsl,program_instruction_type *program_instructions,int exit_code){
  int k,icomp;
  if(program_instructions->io_proc == 2) {
    for(k=0; k<g_sink_momentum_number; k++) {
      for(icomp=0; icomp<num_component_piN_D; icomp++) {
        pathname_type aff_key_to_write_contractions_to;
        three_momentum_type sink_momentum;
        get_sink_momentum(&sink_momentum,k);
        three_momentum_type seq_source_momentum;
        get_seq_source_momentum(&seq_source_momentum,iseq_mom);
        get_aff_key_for_piN_D_contractions(aff_key_to_write_contractions_to,diagram,seq_source_momentum,sink_momentum,gsl,icomp);
        store_contraction_under_aff_key(contraction_writer,aff_key_to_write_contractions_to,gathered_FT_WDc_contractions,gsl,k,icomp,exit_code);
      }
    }
  }
}

void compute_and_store_piN_D_contractions_for_diagram(int iseq_mom,int diagram,global_source_location_type gsl,program_instruction_type *program_instructions,contraction_writer_type *contraction_writer){
  FT_WDc_contractions_type FT_WDc_contractions;
  gathered_FT_WDc_contractions_type gathered_FT_WDc_contractions;
  allocate_memory_for_ft_and_gathering(&FT_WDc_contractions,&gathered_FT_WDc_contractions,num_component_piN_D,program_instructions,61,62);

  three_momentum_type seq_source_momentum;
  get_seq_source_momentum(&seq_source_momentum,iseq_mom);

  information_needed_for_source_phase_type information_needed_for_source_phase;
  init_information_needed_for_source_phase_so_that_source_phase_with_sequential_source_momentum_is_computed(&information_needed_for_source_phase,seq_source_momentum);
  //init_information_needed_for_source_phase_so_that_no_source_phase_is_computed(&information_needed_for_source_phase);

  compute_gathered_FT_WDc_contractions(&FT_WDc_contractions,&gathered_FT_WDc_contractions,gsl,diagram,num_component_piN_D,program_instructions,&information_needed_for_source_phase,124);
  store_piN_D_contractions(contraction_writer,&gathered_FT_WDc_contractions,iseq_mom,diagram,gsl,program_instructions,82);

  free_memory_for_ft_and_gathering(&FT_WDc_contractions,&gathered_FT_WDc_contractions,program_instructions);
}

void compute_and_store_piN_D_contractions(int iseq_mom,int i_src,int i_coherent,global_source_location_type gsl,forward_propagators_type *forward_propagators,sequential_propagators_type *sequential_propagators,program_instruction_type *program_instructions,contraction_writer_type *contraction_writer){
  set_memory_for_Whick_Dirac_and_color_contractions_to_zero(program_instructions);

  compute_Whick_Dirac_and_color_contractions_for_piN_D(iseq_mom,i_src,i_coherent,program_instructions,forward_propagators,sequential_propagators);

  int diagram;
  for(diagram=0; diagram<6; diagram++){
    compute_and_store_piN_D_contractions_for_diagram(iseq_mom,diagram,gsl,program_instructions,contraction_writer);
  }

}

void init_contraction_writer(contraction_writer_type *contraction_writer,const char *name,int i_src,int exit_code_1,int exit_code_2, int exit_code_3,program_instruction_type *program_instructions){
  contraction_writer->affw = NULL;
  contraction_writer->affn = NULL;
  contraction_writer->affdir = NULL;

  if(program_instructions->io_proc == 2) {
    global_source_location_type gsl;
    get_global_source_location(&gsl,i_src);

    char *aff_status_str = (char*)aff_version();
    write_to_stdout("# [piN2piN] using aff version %s\n", aff_status_str);
  
    pathname_type filename;
    sprintf(filename, "%s.%.4d.tsrc%.2d.aff", name, Nconf, gsl.x[0] );
    write_to_stdout("# [piN2piN] writing data to file %s\n", filename);
    if(test_whether_pathname_exists(filename)){
      write_to_stderr("# [piN2piN] destination file %s already exists\n", filename);
      EXIT(1);
    }
    contraction_writer->affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(contraction_writer->affw);
    if( aff_status_str != NULL ) {
      write_to_stderr("[piN2piN] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(exit_code_1);
    }
  
    if( (contraction_writer->affn = aff_writer_root(contraction_writer->affw)) == NULL ) {
      write_to_stderr("[piN2piN] Error, aff writer is not initialized\n");
      EXIT(exit_code_2);
    }

    contraction_writer->aff_buffer = (double _Complex*)malloc(T_global*g_sv_dim*g_sv_dim*sizeof(double _Complex));
      if(contraction_writer->aff_buffer == NULL) {
      write_to_stderr("[piN2piN] Error from malloc\n");
      EXIT(exit_code_3);
    }
  }
}

void exit_contraction_writer(contraction_writer_type *contraction_writer,int exit_code,program_instruction_type *program_instructions){
  if(program_instructions->io_proc == 2) {
    char *aff_status_str = (char*)aff_writer_close (contraction_writer->affw);
    if( aff_status_str != NULL ) {
      write_to_stderr("[piN2piN] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(exit_code);
    }
    if(contraction_writer->aff_buffer != NULL) free(contraction_writer->aff_buffer);
  }
}

void compute_and_store_correlators_which_need_only_forward_and_sequential_propagators(forward_propagators_type *forward_propagators,sequential_propagators_type *sequential_propagators,program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type *cvc_and_tmLQCD_information){

  int i_src;
  for(i_src=0; i_src < g_source_location_number; i_src++ ) {
    contraction_writer_type contraction_writer;

    init_contraction_writer(&contraction_writer,"B_B",i_src,4,5,6,program_instructions);

    int i_coherent;
    for(i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      global_source_location_type gsl;
      get_global_coherent_source_location(&gsl,i_src,i_coherent);
     
      compute_and_store_N_N_contractions(i_src,i_coherent,gsl,forward_propagators,sequential_propagators,program_instructions,&contraction_writer); 
      compute_and_store_D_D_contractions(i_src,i_coherent,gsl,forward_propagators,sequential_propagators,program_instructions,&contraction_writer); 

      int iseq_mom;
      for(iseq_mom = 0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {
        compute_and_store_piN_D_contractions(iseq_mom,i_src,i_coherent,gsl,forward_propagators,sequential_propagators,program_instructions,&contraction_writer); 
      }
    }

    exit_contraction_writer(&contraction_writer,11,program_instructions);
  }

}

void compute_and_store_correlators_which_need_stochastic_propagators(forward_propagators_type *forward_propagators,sequential_propagators_type *sequential_propagators,stochastic_propagators_type *stochastic_propagators,program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type *cvc_and_tmLQCD_information){


}

void compute_and_store_correlators_which_use_oet(forward_propagators_type *forward_propagators,sequential_propagators_type *sequential_propagators,program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type *cvc_and_tmLQCD_information){

}

/*****************************************************************************
* Allocate and free memory
*****************************************************************************/

void allocate_work_spaces(program_instruction_type *program_instructions){
  alloc_spinor_field(&program_instructions->spinor_work[0], VOLUMEPLUSRAND);
  alloc_spinor_field(&program_instructions->spinor_work[1], VOLUMEPLUSRAND);
}

void allocate_memory_for_the_contractions(program_instruction_type *program_instructions){
  int i;
  program_instructions->conn_X = (spinor_propagator_type**)malloc(program_instructions->max_num_diagram * sizeof(spinor_propagator_type*));
  for(i=0; i<program_instructions->max_num_diagram; i++) {
    program_instructions->conn_X[i] = create_sp_field( (size_t)VOLUME * num_component_max );
    if(program_instructions->conn_X[i] == NULL) {
      write_to_stderr("[piN2piN] Error, could not alloc conn_X\n");
      EXIT(2);
    }
  }
}

void allocate_memory_needed_for_the_correlator_computation_in_general(program_instruction_type *program_instructions){
  allocate_work_spaces(program_instructions);
  allocate_memory_for_the_contractions(program_instructions);
}

void free_work_spaces(program_instruction_type *program_instructions){
  // missing
}

void free_memory_for_the_contractions(program_instruction_type *program_instructions){
  int i;
  for(i=0; i<program_instructions->max_num_diagram; i++) { free_sp_field( &(program_instructions->conn_X[i]) ); }
  free(program_instructions->conn_X);
}

void free_memory_needed_for_the_correlator_computation_in_general(program_instruction_type *program_instructions){
  free_work_spaces(program_instructions);
  free_memory_for_the_contractions(program_instructions);
}

void allocate_memory_for_propagator_list(double ***propagator_list,int no_fields,int sizeof_spinor_field,int exit_code){
  int i;
  (*propagator_list) = (double**)malloc(no_fields * sizeof(double*));
  (*propagator_list)[0] = (double*)malloc(no_fields * sizeof_spinor_field);
  if((*propagator_list)[0] == NULL) {
    write_to_stderr("[piN2piN] Error from malloc\n");
    EXIT(exit_code);
  }
  for(i=1; i<no_fields; i++) (*propagator_list)[i] = (*propagator_list)[i-1] + _GSI(VOLUME);
}

void allocate_memory_for_forward_propagators(forward_propagators_type* forward_propagators,program_instruction_type *program_instructions){
  forward_propagators->no_fields = g_coherent_source_number * g_source_location_number * n_s*n_c; /* forward propagators at all base x coherent source locations */ 
  allocate_memory_for_propagator_list(&forward_propagators->propagator_list_up,forward_propagators->no_fields,program_instructions->sizeof_spinor_field,44);
  if(g_fermion_type == _TM_FERMION) {
    allocate_memory_for_propagator_list(&forward_propagators->propagator_list_dn,forward_propagators->no_fields,program_instructions->sizeof_spinor_field,45);
  }
}

void allocate_memory_for_sequential_propagators(sequential_propagators_type* sequential_propagators,program_instruction_type *program_instructions){
  sequential_propagators->no_fields = g_source_location_number * g_seq_source_momentum_number * n_s*n_c;
  allocate_memory_for_propagator_list(&sequential_propagators->propagator_list,sequential_propagators->no_fields,program_instructions->sizeof_spinor_field,46);
}

void free_memory_for_propagator_list(double **propagator_list){
  free(propagator_list[0]);
  free(propagator_list);
}

void free_memory_for_forward_propagators(forward_propagators_type* forward_propagators,program_instruction_type *program_instructions){
  free_memory_for_propagator_list(forward_propagators->propagator_list_up);
  if(g_fermion_type == _TM_FERMION) {
    free_memory_for_propagator_list(forward_propagators->propagator_list_dn);
  }
}

void free_memory_for_sequential_propagators(sequential_propagators_type* sequential_propagators,program_instruction_type *program_instructions){
  free_memory_for_propagator_list(sequential_propagators->propagator_list);
}


void set_spinor_field_to_zero(double* spinor_field,program_instruction_type *program_instructions){
  memset(spinor_field, 0, program_instructions->sizeof_spinor_field);
}

void copy_spinor_fields(double* dest,double* src,program_instruction_type *program_instructions){
  memcpy( dest, src, program_instructions->sizeof_spinor_field);
}

void set_spinor_field_to_point_source(double *spinor_field,local_source_location_type lsl,int is,program_instruction_type *program_instructions){
  set_spinor_field_to_zero(spinor_field,program_instructions);
  if(lsl.proc_id == g_cart_id)  {
    spinor_field[_GSI(g_ipt[lsl.x[0]][lsl.x[1]][lsl.x[2]][lsl.x[3]])+2*is] = 1.;
  }
}

void compute_inversion_with_tm_rotation_and_smearing(double* inversion,double* source,double* spinor_field_for_intermediate_storage,int op_id,int rotation_direction,program_instruction_type *program_instructions){
  int exitstatus;

  set_spinor_field_to_zero(spinor_field_for_intermediate_storage,program_instructions);
  
  /* source-smear the point source */
  exitstatus = Jacobi_Smearing(program_instructions->gauge_field_smeared, source, N_Jacobi, kappa_Jacobi);

  if( g_fermion_type == _TM_FERMION ) {
    spinor_field_tm_rotation(source, source, rotation_direction, g_fermion_type, VOLUME);
  }

  exitstatus = tmLQCD_invert(spinor_field_for_intermediate_storage, source, op_id, 0);
  if(exitstatus != 0) {
    write_to_stdout("[piN2piN] Error from tmLQCD_invert, status was %d\n", exitstatus);
    EXIT(12);
  }

  if( g_fermion_type == _TM_FERMION ) {
    spinor_field_tm_rotation(spinor_field_for_intermediate_storage, spinor_field_for_intermediate_storage, rotation_direction, g_fermion_type, VOLUME);
  }

  /* sink-smear the point-source propagator */
  exitstatus = Jacobi_Smearing(program_instructions->gauge_field_smeared, spinor_field_for_intermediate_storage, N_Jacobi, kappa_Jacobi);

  copy_spinor_fields(inversion,spinor_field_for_intermediate_storage,program_instructions);
}

void compute_forward_propagators_for_coherent_source_location(int i_src,int i_coherent,double **propagator_list,int op_id,int rotation_direction,program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type * cvc_and_tmLQCD_information){
  double ratime, retime;

  ratime = _GET_TIME;

  local_source_location_type lsl;
  get_local_coherent_source_location(&lsl,i_src,i_coherent);

  int is;
  for(is=0;is<n_s*n_c;is++) {  
    set_spinor_field_to_point_source(program_instructions->spinor_work[0],lsl,is,program_instructions);
    compute_inversion_with_tm_rotation_and_smearing(propagator_list[get_forward_propagator_index(i_src,i_coherent,is)],program_instructions->spinor_work[0],program_instructions->spinor_work[1],op_id,rotation_direction,program_instructions);
  }  /* end of loop on spin color */

  retime = _GET_TIME;

  write_to_stdout("# [piN2piN] time for up propagator = %e seconds\n", retime-ratime);
}

void compute_forward_up_propagators(forward_propagators_type* forward_propagators,program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type * cvc_and_tmLQCD_information){
  write_to_stdout("# [piN2piN] up-type inversion\n");
  int i_src,i_coherent;
  for(i_src = 0; i_src<g_source_location_number; i_src++) {
    for(i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      compute_forward_propagators_for_coherent_source_location(i_src,i_coherent,forward_propagators->propagator_list_up,program_instructions->op_id_up,+1,program_instructions,cvc_and_tmLQCD_information);
    }  /* end of loop on coherent source timeslices */
  }    /* end of loop on base source timeslices */
}

void compute_forward_down_propagators(forward_propagators_type* forward_propagators,program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type * cvc_and_tmLQCD_information){
  if(g_fermion_type == _TM_FERMION) {
    write_to_stdout("# [piN2piN] dn-type inversion\n");
    int i_src,i_coherent;
    for(i_src = 0; i_src<g_source_location_number; i_src++) {
      for(i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
        compute_forward_propagators_for_coherent_source_location(i_src,i_coherent,forward_propagators->propagator_list_dn,program_instructions->op_id_dn,-1,program_instructions,cvc_and_tmLQCD_information);
      }  /* end of loop on coherent source timeslices */
    }    /* end of loop on base source timeslices */
  }
  else{
    forward_propagators->propagator_list_dn  = forward_propagators->propagator_list_up;
  }
}

void compute_forward_propagators(forward_propagators_type* forward_propagators,program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type * cvc_and_tmLQCD_information){
  compute_forward_up_propagators(forward_propagators,program_instructions,cvc_and_tmLQCD_information);
  compute_forward_down_propagators(forward_propagators,program_instructions,cvc_and_tmLQCD_information);
}

void allocate_propagator_pointer_list(propagator_pointer_list_type *pointer_list,int size,int exit_code){
  (*pointer_list) = (double**)malloc(size * sizeof(double*));
  if((*pointer_list) == NULL) {
    write_to_stderr("[piN2piN] Error from malloc\n");
    EXIT(exit_code);
  }
}

void free_propagator_pointer_list(propagator_pointer_list_type pointer_list){
  free(pointer_list);
}

void set_spinor_field_to_sequential_source_from_coherent_down_propagators(double *sequential_source,three_momentum_type seq_source_momentum,int i_src,global_source_location_type gsl,int is,forward_propagators_type *forward_propagators,propagator_pointer_list_type pointers_to_coherent_source_forward_propagators,int exit_code){
  int exitstatus;
  /* extract spin-color source-component is from coherent source dn propagators */
  int i;
  for(i=0; i<g_coherent_source_number; i++) {
    write_to_stdout("# [piN2piN] using dn prop id %d / %d\n", get_forward_complete_is_propagator_index(i_src,i), get_forward_propagator_index(i_src,i,is));
    pointers_to_coherent_source_forward_propagators[i] = forward_propagators->propagator_list_dn[get_forward_propagator_index(i_src,i,is)];
  }

  /* build sequential source */
  exitstatus = init_coherent_sequential_source(sequential_source, pointers_to_coherent_source_forward_propagators, gsl.x[0], g_coherent_source_number, seq_source_momentum.p, 5);
  if(exitstatus != 0) {
    write_to_stderr("[piN2piN] Error from init_coherent_sequential_source, status was %d\n", exitstatus);
    EXIT(exit_code);
  }
}

void get_filename_for_sequential_propagator(pathname_type filename,global_source_location_type gsl,int is,three_momentum_type seq_source_momentum){
 sprintf(filename, "/storage/oehm/piN2piN_crosscheck/seq_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.inverted",
    filename_prefix, Nconf, gsl.x[0], gsl.x[1], gsl.x[2], gsl.x[3], is,
    seq_source_momentum.p[0], seq_source_momentum.p[1], seq_source_momentum.p[2]);
}

void get_filename_for_sequential_source(pathname_type filename,global_source_location_type gsl,int is,three_momentum_type seq_source_momentum){
 sprintf(filename, "/storage/oehm/piN2piN_crosscheck/seq_source_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.source",
    filename_prefix, Nconf, gsl.x[0], gsl.x[1], gsl.x[2], gsl.x[3], is,
    seq_source_momentum.p[0], seq_source_momentum.p[1], seq_source_momentum.p[2]);
}

void get_filename_for_sequential_propagator_with_g_cart_id(pathname_type filename,global_source_location_type gsl,int is,three_momentum_type seq_source_momentum){
 sprintf(filename, "/storage/oehm/piN2piN_crosscheck/seq_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.g_cart_id%d.inverted",
    filename_prefix, Nconf, gsl.x[0], gsl.x[1], gsl.x[2], gsl.x[3], is,
    seq_source_momentum.p[0], seq_source_momentum.p[1], seq_source_momentum.p[2],g_cart_id);
}

void get_filename_for_sequential_source_with_g_cart_id(pathname_type filename,global_source_location_type gsl,int is,three_momentum_type seq_source_momentum){
 sprintf(filename, "/storage/oehm/piN2piN_crosscheck/seq_source_%s.%.4d.t%.2dx%.2dy%.2dz%.2d.%.2d.qx%.2dqy%.2dqz%.2d.g_cart_id%d.source",
    filename_prefix, Nconf, gsl.x[0], gsl.x[1], gsl.x[2], gsl.x[3], is,
    seq_source_momentum.p[0], seq_source_momentum.p[1], seq_source_momentum.p[2],g_cart_id);
}


void write_spinor_field_to_file(double *spinor_field,int sizeof_spinor_field,pathname_type filename){
  FILE *f;

  if(test_whether_pathname_exists(filename)){
    write_to_stderr("File %s already exists\n",filename);
    EXIT(1);
  }

  f = fopen(filename,"w");
  if(f == NULL){
    write_to_stderr("could not open file %s for writing\n",filename);
    EXIT(1);
  }
  fwrite(spinor_field,sizeof(double),sizeof_spinor_field/sizeof(double),f);
  fclose(f);
}

void write_propagator_to_file(double *sequential_propagator_to_write,pathname_type filename,int exit_code){
  int exitstatus;

  if(test_whether_pathname_exists(filename)){
    write_to_stderr("File %s already exists\n",filename);
    EXIT(1);
  }

  write_to_stdout("# [piN2piN] writing propagator to file %s\n", filename);
  exitstatus = write_propagator(sequential_propagator_to_write, filename, 0, 64);
  if(exitstatus != 0) {
    write_to_stderr("[piN2piN] Error from write_propagator, status was %d\n", exitstatus);
    EXIT(exit_code);
  }
}

void write_sequential_propagator_to_file(double *sequential_propagator_to_write,global_source_location_type gsl,int is,three_momentum_type seq_source_momentum,int exit_code){
  pathname_type filename;
  get_filename_for_sequential_propagator(filename,gsl,is,seq_source_momentum);

  write_propagator_to_file(sequential_propagator_to_write,filename,exit_code);
}

void write_sequential_source_to_file_2ndversion(double *sequential_propagator_to_write,program_instruction_type *program_instructions,global_source_location_type gsl,int is,three_momentum_type seq_source_momentum){
  pathname_type filename;
  get_filename_for_sequential_source_with_g_cart_id(filename,gsl,is,seq_source_momentum);

  write_spinor_field_to_file(sequential_propagator_to_write,program_instructions->sizeof_spinor_field,filename);
}

void write_sequential_propagator_to_file_2ndversion(double *sequential_propagator_to_write,program_instruction_type *program_instructions,global_source_location_type gsl,int is,three_momentum_type seq_source_momentum){
  pathname_type filename;
  get_filename_for_sequential_propagator_with_g_cart_id(filename,gsl,is,seq_source_momentum);

  write_spinor_field_to_file(sequential_propagator_to_write,program_instructions->sizeof_spinor_field,filename);
}

void compute_sequential_propagators(sequential_propagators_type* sequential_propagators,forward_propagators_type *forward_propagators,program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type * cvc_and_tmLQCD_information){
  double ratime, retime;
  /* loop on sequential source momenta */
  int iseq_mom;
  for(iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {
    three_momentum_type seq_source_momentum;
    get_seq_source_momentum(&seq_source_momentum,iseq_mom);
    
    /***********************************************************
     * sequential propagator U^{-1} g5 exp(ip) D^{-1}: tfii
     ***********************************************************/
    write_to_stdout("# [piN2piN] sequential inversion fpr pi2 = (%d, %d, %d)\n",seq_source_momentum.p[0],seq_source_momentum.p[1],seq_source_momentum.p[2]); 

    propagator_pointer_list_type pointers_to_coherent_source_forward_propagators;
    allocate_propagator_pointer_list(&pointers_to_coherent_source_forward_propagators,g_coherent_source_number,43);

    int i_src;
    for(i_src=0; i_src<g_source_location_number; i_src++) {
      global_source_location_type gsl;
      get_global_source_location(&gsl,i_src);

      ratime = _GET_TIME;
      int is;
      for(is=0;is<n_s*n_c;is++) {
        set_spinor_field_to_sequential_source_from_coherent_down_propagators(program_instructions->spinor_work[0],seq_source_momentum,i_src,gsl,is,forward_propagators,pointers_to_coherent_source_forward_propagators,14);
        //write_sequential_source_to_file_2ndversion(program_instructions->spinor_work[0],program_instructions,gsl,is,seq_source_momentum);
        compute_inversion_with_tm_rotation_and_smearing(sequential_propagators->propagator_list[get_sequential_propagator_index(iseq_mom,i_src,is)],program_instructions->spinor_work[0],program_instructions->spinor_work[1],program_instructions->op_id_up,+1,program_instructions);
        if(g_write_sequential_propagator) {
          write_sequential_propagator_to_file(program_instructions->spinor_work[1],gsl,is,seq_source_momentum,15);
        }
        //write_sequential_propagator_to_file_2ndversion(program_instructions->spinor_work[1],program_instructions,gsl,is,seq_source_momentum);
      }
      retime = _GET_TIME;
      write_to_stdout("# [piN2piN] time for seq propagator = %e seconds\n", retime-ratime);
    }  

    free_propagator_pointer_list(pointers_to_coherent_source_forward_propagators);
  }
}

void set_operator_ids_depending_on_fermion_type(program_instruction_type *program_instructions){
  if(g_fermion_type == _TM_FERMION) {
    program_instructions->op_id_up = 0;
    program_instructions->op_id_dn = 1;
  } else if(g_fermion_type == _WILSON_FERMION) {
    program_instructions->op_id_up = 0;
    program_instructions->op_id_dn = 0;
  }
}

void compute_and_store_correlators(program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type *cvc_and_tmLQCD_information){

  allocate_memory_needed_for_the_correlator_computation_in_general(program_instructions);

  set_operator_ids_depending_on_fermion_type(program_instructions);

	forward_propagators_type forward_propagators;
	sequential_propagators_type sequential_propagators;

  allocate_memory_for_forward_propagators(&forward_propagators,program_instructions);
	compute_forward_propagators(&forward_propagators,program_instructions,cvc_and_tmLQCD_information);

  allocate_memory_for_sequential_propagators(&sequential_propagators,program_instructions);
	compute_sequential_propagators(&sequential_propagators,&forward_propagators,program_instructions,cvc_and_tmLQCD_information);

	compute_and_store_correlators_which_need_only_forward_and_sequential_propagators(&forward_propagators,&sequential_propagators,program_instructions,cvc_and_tmLQCD_information);

	stochastic_propagators_type stochastic_propagators;

	compute_and_store_correlators_which_need_stochastic_propagators(&forward_propagators,&sequential_propagators,&stochastic_propagators,program_instructions,cvc_and_tmLQCD_information);

  /* sequential propagator list not needed after this point */ 
  free_memory_for_sequential_propagators(&sequential_propagators,program_instructions);

	compute_and_store_correlators_which_use_oet(&forward_propagators,&sequential_propagators,program_instructions,cvc_and_tmLQCD_information);

  free_memory_for_forward_propagators(&forward_propagators,program_instructions);

  free_memory_needed_for_the_correlator_computation_in_general(program_instructions);

}

void init_tmLQCD(program_instruction_type *program_instructions){

  write_to_stdout("# [piN2piN] calling tmLQCD wrapper init functions\n");

  int exitstatus;
  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  /* exitstatus = tmLQCD_invert_init(program_instructions->argc, program_instructions->argv, 1, 0); */
  exitstatus = tmLQCD_invert_init(program_instructions->argc, program_instructions->argv, 1);
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

}

void init_cvc_MPI(program_instruction_type *program_instructions){
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[piN2piN] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(program_instructions->argc, program_instructions->argv);

}

void init_cvc(program_instruction_type *program_instructions){

  if(init_geometry() != 0) {
    fprintf(stderr, "[piN2piN] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  program_instructions->VOL3 = LX*LY*LZ;
  program_instructions->sizeof_spinor_field = _GSI(VOLUME)*sizeof(double);
  program_instructions->sizeof_spinor_field_timeslice = _GSI(program_instructions->VOL3)*sizeof(double);
}

void set_io_process(program_instruction_type *program_instructions){
#ifdef HAVE_MPI
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    program_instructions->io_proc = 2;
    fprintf(stdout, "# [piN2piN] proc%.4d tr%.4d is io process\n", g_cart_id, g_tr_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      program_instructions->io_proc = 1;
      fprintf(stdout, "# [piN2piN] proc%.4d tr%.4d is send process\n", g_cart_id, g_tr_id);
    } else {
      program_instructions->io_proc = 0;
    }
  }
#else
  program_instructions->io_proc = 2;
#endif

}

void init_cvc_and_tmLQCD(program_instruction_type *program_instructions,cvc_and_tmLQCD_information_type *cvc_and_tmLQCD_information){

  init_tmLQCD(program_instructions);

  init_cvc_MPI(program_instructions);

  init_cvc(program_instructions);

  set_io_process(program_instructions);
}

void read_gauge_field(program_instruction_type *program_instructions){
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[piN2piN] Nconf = %d\n", Nconf);

  int exitstatus = tmLQCD_read_gauge(Nconf);
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
}

void measure_the_plaquette(program_instruction_type *program_instructions){
  double plaq_m = 0., plaq_r = 0.;
  
  plaquette(&plaq_m);
  if(g_cart_id==0) write_to_stdout("# [piN2piN] read plaquette value    : %25.16e\n", plaq_r);
  if(g_cart_id==0) write_to_stdout("# [piN2piN] measured plaquette value: %25.16e\n", plaq_m); 
}

void smeare_gauge_field(program_instruction_type *program_instructions){
  int exitstatus;
   if( N_Jacobi > 0 ) {
    if(N_ape > 0 ) {
      alloc_gauge_field(&program_instructions->gauge_field_smeared, VOLUMEPLUSRAND);
      memcpy(program_instructions->gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));
      exitstatus = APE_Smearing(program_instructions->gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[piN2piN] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
  
    } else {
      program_instructions->gauge_field_smeared = g_gauge_field;
    }
  }

}

void init_gauge_field(program_instruction_type *program_instructions){
  read_gauge_field(program_instructions);
  measure_the_plaquette(program_instructions);
  smeare_gauge_field(program_instructions);
}

void determine_stoachastic_source_timeslices(program_instruction_type *program_instructions){
  int exitstatus = get_stochastic_source_timeslices();
  if(exitstatus != 0) {
    write_to_stderr("[piN2piN] Error from get_stochastic_source_timeslices, status was %d\n", exitstatus);
    EXIT(19);
  }
}

void execute_program_instructions(program_instruction_type *program_instructions){

	cvc_and_tmLQCD_information_type cvc_and_tmLQCD_information;

	init_cvc_and_tmLQCD(program_instructions,&cvc_and_tmLQCD_information);

  init_gauge_field(program_instructions);

  determine_stoachastic_source_timeslices(program_instructions);

	compute_and_store_correlators(program_instructions,&cvc_and_tmLQCD_information);

//	exit_cvc_and_tmLQCD(&cvc_and_tmLQCD_information);
}

void init_MPI(int argc,char** argv){
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
}

void exit_MPI(){
#ifdef HAVE_MPI
  MPI_Finalize();
#endif
}

/*
void init_gamma_component_information(gamma_component_information_type *gamma_component_information,int num_components,int gamma_component[][],double gamma_component_sign[]){
  gamma_component_information->num_component = num_components;
  gamma_component_information->gamma_component = alloc_2d_integer_array(num_components,2);
  copy_2d_integer_array(gamma_component->gamma_component,gamma_component);
  gamma_component_information->gamma_component_sign = alloc_1d_double_array(num_components);
  
}

void init_gamma_components(gamma_components_type *gamma_components){
  init_gamma_component_information(&gamma_components->piN_piN_component,9);
  init_gamma_component_information(&gamma_components->N_N_component,9);
  init_gamma_component_information(&gamma_components->D_D_component,9);
  init_gamma_component_information(&gamma_components->piN_D_component,9);
}

void init_program_instructions(program_instruction_type *program_instructions){
  program_instructions->filename_set = 1;
  
  init_gamma_components(&program_instructions->gamma_components);
}
*/

void print_usage() {
  write_to_stdout("Code to perform contractions for piN 2-pt. function\n");
  write_to_stdout("Usage:    [options]\n");
  write_to_stdout("Options: -f input filename [default cvc.input]\n");
  write_to_stdout("         -h? this help\n");
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

void init_program_instructions(int argc,char **argv,program_instruction_type *program_instructions){
  program_instructions->filename_set = 1;
  program_instructions->io_proc=-1;
  program_instructions->gauge_field_smeared = NULL;
  program_instructions->VOL3 = 0;
  program_instructions->sizeof_spinor_field = 0;
  program_instructions->sizeof_spinor_field_timeslice = 0;
  program_instructions->argc = argc;
  program_instructions->argv = argv;
  program_instructions->op_id_up= -1;
  program_instructions->op_id_dn = -1;
  program_instructions->source_proc_id = 0;
  program_instructions->conn_X=NULL;
  program_instructions->buffer=NULL;
  program_instructions->max_num_diagram = 6;
}

void read_command_line_arguments(int argc,char** argv,program_instruction_type *program_instructions){
	int c;

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(program_instructions->filename, optarg);
      program_instructions->filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      print_usage();
      break;
    }
  }
}

void read_cvc_input_file(program_instruction_type *program_instructions){
  if(program_instructions->filename_set==0) strcpy(program_instructions->filename, "cvc.input");
  write_to_stdout("# reading input from file %s\n", program_instructions->filename);
  read_input_parser(program_instructions->filename);
}

void set_default_values_for_program_instructions(program_instruction_type *program_instructions){
  if(g_fermion_type == -1 ) {
    write_to_stderr("# [piN2piN] fermion_type must be set\n");
    exit(1);
  } else {
    write_to_stdout("# [piN2piN] using fermion type %d\n", g_fermion_type);
  }
}

void read_command_line_argumens_and_input_file(int argc,char** argv,program_instruction_type *program_instructions){

  init_program_instructions(argc,argv,program_instructions);

  read_command_line_arguments(argc,argv,program_instructions);

  read_cvc_input_file(program_instructions);

  set_default_values_for_program_instructions(program_instructions);
}

int main(int argc,char** argv){

	program_instruction_type program_instructions;

	init_MPI(argc,argv);

  // this function initalizes the program_instructions
	read_command_line_argumens_and_input_file(argc,argv,&program_instructions);

  // calculate what should be calculated
	execute_program_instructions(&program_instructions);

	exit_MPI();

	return 0;
}


