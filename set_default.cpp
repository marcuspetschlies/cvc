/***************************************************
 * set_default
 ***************************************************/
 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include "cvc_complex.h"
#include "global.h"
#include "ilinalg.h"
#include "cvc_geometry.h"
#include "mpi_init.h"
#include "default_input_values.h"
#include "get_index.h"
#include "read_input_parser.h"

namespace cvc {

void set_default_input_values(void) {

  T_global    = _default_T_global;
  Tstart      = _default_Tstart;
  LX          = _default_LX;
  LXstart     = _default_LXstart;
  LY          = _default_LY;
  LYstart     = _default_LYstart;
  LZ          = _default_LZ;
  LZstart     = _default_LZstart;
  L5          = _default_L5;
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
  g_kappa5d   = _default_kappa5d;
  g_mu        = _default_mu;
  g_musigma   = _default_musigma;
  g_mudelta   = _default_mudelta;
  g_mubar     = _default_mubar;
  g_epsbar    = _default_epsbar;
  g_csw       = _default_csw;
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
  strcpy(filename_prefix3,     _default_filename_prefix3);

  strcpy(g_sequential_filename_prefix,      _default_sequential_filename_prefix);
  strcpy(g_sequential_filename_prefix2,     _default_sequential_filename_prefix2);

  strcpy(gaugefilename_prefix, _default_gaugefilename_prefix);
  strcpy(g_outfile_prefix, _default_outfile_prefix);
  g_gaugeid   = _default_gaugeid; 
  g_gaugeid2  = _default_gaugeid2;
  g_gauge_step= _default_gauge_step;
  g_seed      = _default_seed;
  g_noise_type= _default_noise_type;
  g_source_type= _default_source_type;
  solver_precision = _default_solver_precision;
  reliable_delta = _default_reliable_delta;
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

  N_hyp        = _default_N_hyp;
  alpha_hyp[0] = _default_alpha_hyp;
  alpha_hyp[1] = _default_alpha_hyp;
  alpha_hyp[2] = _default_alpha_hyp;


  g_source_timeslice  = _default_source_timeslice;

  g_sequential_source_timeslice  = _default_sequential_source_timeslice;
  g_sequential_source_timeslice_number = _default_sequential_source_timeslice_number;

  g_sequential_source_location_x = _default_sequential_source_location_x;
  g_sequential_source_location_y = _default_sequential_source_location_y;
  g_sequential_source_location_z = _default_sequential_source_location_z;


  g_no_extra_masses = _default_no_extra_masses;
  g_no_light_masses = _default_no_light_masses;
  g_no_strange_masses = _default_no_strange_masses;
  g_local_local       = _default_local_local;
  g_local_smeared     = _default_local_smeared;
  g_smeared_local     = _default_smeared_local;
  g_smeared_smeared   = _default_smeared_smeared;

  g_gpu_device_number = _default_gpu_device_number;
  g_gpu_per_node      = _default_gpu_per_node;

  g_coherent_source = _default_coherent_source;
  g_coherent_source_base  = _default_coherent_source_base;
  g_coherent_source_delta = _default_coherent_source_delta;
  g_gauge_file_format = _default_gauge_file_format;

  strcpy( g_rng_filename, _default_rng_filename );
  g_source_index[0] = _default_source_index_min;
  g_source_index[1] = _default_source_index_max;
  g_propagator_bc_type = _default_propagator_bc_type;
  g_propagator_gamma_basis = _default_propagator_gamma_basis;
  g_propagator_precision = _default_propagator_precision;
  g_write_source = _default_write_source;
  g_read_source = _default_read_source;

  g_write_propagator = _default_write_propagator;
  g_read_propagator = _default_read_propagator;

  g_read_sequential_propagator = _default_read_sequential_propagator;
  g_write_sequential_source = _default_write_sequential_source;
  g_write_sequential_propagator = _default_write_sequential_propagator;

  g_nsample     = _default_nsample;
  g_nsample_oet = _default_nsample_oet;
  g_sv_dim = 4;
  g_cv_dim = 3;

  g_fv_dim = g_sv_dim * g_cv_dim;
  g_fp_dim = g_fv_dim * g_fv_dim;
  g_cm_dim = g_cv_dim * g_cv_dim;
  g_num_threads = _default_num_threads;

  g_source_momentum[0] = _default_source_momentum_x;
  g_source_momentum[1] = _default_source_momentum_y;
  g_source_momentum[2] = _default_source_momentum_z;
  g_source_momentum_set = _default_source_momentum_set;
  g_source_momentum_number = _default_source_momentum_number;

  g_sink_momentum[0] = _default_sink_momentum_x;
  g_sink_momentum[1] = _default_sink_momentum_y;
  g_sink_momentum[2] = _default_sink_momentum_z;
  g_sink_momentum_set = _default_sink_momentum_set;
  g_sink_momentum_number = _default_sink_momentum_number;

  g_seq_source_momentum[0] = _default_seq_source_momentum_x;
  g_seq_source_momentum[1] = _default_seq_source_momentum_y;
  g_seq_source_momentum[2] = _default_seq_source_momentum_z;
  g_seq_source_momentum_set = _default_seq_source_momentum_set;
  g_seq_source_momentum_number = _default_seq_source_momentum_number;

  g_seq2_source_momentum[0]     = _default_seq_source_momentum_x;
  g_seq2_source_momentum[1]     = _default_seq_source_momentum_y;
  g_seq2_source_momentum[2]     = _default_seq_source_momentum_z;
  g_seq2_source_momentum_set    = _default_seq_source_momentum_set;
  g_seq2_source_momentum_number = _default_seq_source_momentum_number;

  g_rng_state = _default_rng_state;

  g_verbose = _default_verbose;

  g_m5 = _default_m5;
  g_m0 = _default_m0;

  g_cpu_prec = _default_cpu_prec;
  g_gpu_prec = _default_gpu_prec;
  g_gpu_prec_sloppy = _default_gpu_prec_sloppy;

  g_space_dilution_depth = _default_space_dilution_depth;

  g_mms_id = _default_mms_id;
  g_check_inversion = _default_check_inversion;

  g_src_snk_time_separation = _default_src_snk_time_separation;

  g_sequential_source_gamma_id = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 0] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 1] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 2] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 3] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 4] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 5] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 6] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 7] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 8] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[ 9] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[10] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[11] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[12] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[13] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[14] = _default_seq_source_gamma_id;
  g_sequential_source_gamma_id_list[15] = _default_seq_source_gamma_id;

  g_sequential_source_gamma_id_number = _default_seq_source_gamma_id_number;

  g_source_gamma_id_number = _default_source_gamma_id_number;
  g_source_gamma_id_list[ 0] = _default_source_gamma_id;
  g_source_gamma_id_list[ 1] = _default_source_gamma_id;
  g_source_gamma_id_list[ 2] = _default_source_gamma_id;
  g_source_gamma_id_list[ 3] = _default_source_gamma_id;
  g_source_gamma_id_list[ 4] = _default_source_gamma_id;
  g_source_gamma_id_list[ 5] = _default_source_gamma_id;
  g_source_gamma_id_list[ 6] = _default_source_gamma_id;
  g_source_gamma_id_list[ 7] = _default_source_gamma_id;
  g_source_gamma_id_list[ 8] = _default_source_gamma_id;
  g_source_gamma_id_list[ 9] = _default_source_gamma_id;
  g_source_gamma_id_list[10] = _default_source_gamma_id;
  g_source_gamma_id_list[11] = _default_source_gamma_id;
  g_source_gamma_id_list[12] = _default_source_gamma_id;
  g_source_gamma_id_list[13] = _default_source_gamma_id;
  g_source_gamma_id_list[14] = _default_source_gamma_id;
  g_source_gamma_id_list[15] = _default_source_gamma_id;

  g_twopoint_function_number = _default_twopoint_function_number;

  g_clover = NULL;
  g_mzz_up = NULL;
  g_mzz_dn = NULL;
  g_mzzinv_up = NULL;
  g_mzzinv_dn = NULL;

  g_fermion_type = _default_fermion_type;
  g_source_location_number = _default_source_location_number;

  g_coherent_source_number = _default_coherent_source_number;

  g_total_momentum_number = _default_total_momentum_number;

  g_gamma_f1_nucleon_number = _default_zero;
  g_gamma_f1_delta_number   = _default_zero;
  g_gamma_f2_number         = _default_zero;
  g_gamma_current_number    = _default_zero;

}  /* end of set_default_input_values */

}  /* end of namespace cvc */
