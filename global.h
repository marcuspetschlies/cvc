#ifndef _GLOBAL_H
#define _GLOBAL_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#ifdef  HAVE_MPI
#  include <mpi.h>
#endif
#include <math.h>
#include "types.h"
#include "ifftw.h"
#ifdef HAVE_TMLQCD_LIBWRAPPER
#include "tmLQCD.h"
#endif

#ifdef HAVE_LIBLEMON
#  ifndef LEMON_OFFSET_TYPE
#    define LEMON_OFFSET_TYPE MPI_Offset
/* #    define LEMON_OFFSET_TYPE uint64_t */
#  endif
#endif

#define _TM_FERMION        0
#define _WILSON_FERMION    1
#define _DW_WILSON_FERMION 2

#ifdef HAVE_MPI
#define EXIT(_i) { MPI_Abort(MPI_COMM_WORLD, (_i)); MPI_Finalize(); exit((_i)); }
#else
#define EXIT(_i) { exit(_i); }
#endif

#ifdef HAVE_MPI
#define EXIT_WITH_MSG(_i, _msg) {\
  if(g_cart_id==0) fprintf(stderr, "%s", _msg);\
  MPI_Abort(MPI_COMM_WORLD, (_i));\
  MPI_Finalize(); exit((_i));\
}
#else
#define EXIT_WITH_MSG(_i, _msg) { \
  fprintf(stderr, "%s", _msg);\
  exit(_i);\
}
#endif

#ifdef HAVE_MPI
#define _GET_TIME  MPI_Wtime()
#else
#define _GET_TIME ((double)clock() / CLOCKS_PER_SEC)
#endif


#if defined MAIN_PROGRAM
#  define EXTERN
#else
#  define EXTERN extern
#endif 

#include "cvc_complex.h"

#define _GSI(_ix) (24*(_ix))
#define _GVI(_ix) ( 6*(_ix))
#define _GGI(_ix,_mu) (18*(4*(_ix)+(_mu)))
#define _GJI(_ix,_mu) (2*(4*(_ix)+(_mu)))
#define _GSWI(_ix,_mu) (18*(6*(_ix)+(_mu)))

#define _GWI(_ix,_mu,_N) (2*((_N)*(_ix)+(_mu)))
/* #define _GWI(_ix,_mu,_N) (2 * ( 16 * (_mu) + (_ix) ) ) */



#define _GCI(_x,_k,_N) ( 2*( (_N) * (_x) + (_k) ) )

#define _G5DI(_is,_ix) ((_is)*VOLUME+(_ix))

#ifndef _Q2EPS
#  define _Q2EPS (5.e-14)
#endif
#ifndef M_PI
#define M_PI (3.141592653589793)
#endif

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

#define HPE_MAX_ORDER 11

#define _MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b) )
#define _MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b) )

#define _ONE_OVER_SQRT2 (0.707106781186548)

#define MAX_M_M_2PT_NUM 16
#define MAX_TWOPOINT_FUNCTION_NUM 20000
#define MAX_SOURCE_LOCATION_NUMBER 128

#define MAX_MOMENTUM_NUMBER 300
#define MAX_SEQUENTIAL_SOURCE_TIMESLICE_NUMBER 128

#define MAX_GAMMA_NUMBER 16

namespace cvc {

typedef struct momentum_info_struct {
  int filename_set;
  int number;
  int **list;
  char filename[200];
} momentum_info_type;

extern const char *g_gitversion;

EXTERN int T_global, LX_global, LY_global, LZ_global;
EXTERN int T, L, LX, LY, LZ, Tstart, LXstart, LYstart, LZstart, FFTW_LOC_VOLUME, L5;
EXTERN unsigned int VOLUME, RAND, EDGES, VOLUMEPLUSRAND;
EXTERN int Nconf;

EXTERN int **** g_ipt, *****g_ipt_5d;
EXTERN int ** g_iup, **g_iup_5d;
EXTERN int ** g_idn, **g_idn_5d;
EXTERN int *g_lexic2eo, *g_eo2lexic, *g_iseven, *g_isevent, *g_lexic2eot, *g_eot2lexic, *g_lexic2eosub;
EXTERN int **g_eosub2t;
/* EXTERN int ****g_eot2xyz; */
EXTERN int ****g_eosubt2coords;
EXTERN int *g_lexic2eo_5d, *g_eo2lexic_5d, *g_iseven_5d, *g_isevent_5d, *g_lexic2eot_5d, *g_eot2lexic_5d;
EXTERN int **g_eosub2sliced3d, ***g_sliced3d2eosub;


EXTERN double **g_spinor_field;

EXTERN double *g_gauge_field;

EXTERN double g_kappa, g_mu, g_musigma, g_mudelta, g_mubar, g_epsbar, g_m5, g_m0, g_kappa5d;

EXTERN int g_proc_id, g_nproc;
EXTERN int g_cart_id;
EXTERN int g_nb_list[8];
EXTERN int g_proc_coords[4];

#ifdef HAVE_MPI
EXTERN MPI_Comm g_cart_grid;
EXTERN MPI_Status status;
EXTERN MPI_Comm g_ts_comm, g_xs_comm;
EXTERN MPI_Comm g_tr_comm;
#endif

EXTERN int g_ts_id, g_ts_nproc;
EXTERN int g_tr_id, g_tr_nproc;
EXTERN int g_xs_id, g_xs_nproc;
EXTERN int g_nb_t_up, g_nb_t_dn;
EXTERN int g_nb_x_up, g_nb_x_dn;
EXTERN int g_nb_y_up, g_nb_y_dn;
EXTERN int g_nb_z_up, g_nb_z_dn;
EXTERN int g_ts_nb_up, g_ts_nb_dn;
EXTERN int g_ts_nb_x_up, g_ts_nb_x_dn;
EXTERN int g_ts_nb_y_up, g_ts_nb_y_dn;
EXTERN int g_ts_nb_z_up, g_ts_nb_z_dn;

EXTERN int g_nproc_t, g_nproc_x, g_nproc_y, g_nproc_z;

EXTERN int g_sourceid, g_sourceid2, g_sourceid_step, Nsave;
EXTERN int g_gaugeid, g_gaugeid2, g_gauge_step;

EXTERN char filename_prefix[200], filename_prefix2[200], filename_prefix3[200], gaugefilename_prefix[200], g_outfile_prefix[200], g_path_prefix[200];
EXTERN char g_sequential_filename_prefix[200], g_sequential_filename_prefix2[200];
EXTERN int format, rotate;
EXTERN double BCangle[4];

EXTERN int no_fields;

EXTERN complex co_phase_up[4];

EXTERN int gamma_permutation[16][24], gamma_sign[16][24];
EXTERN int perm_tab_3[6][3], perm_tab_4[24][4], perm_tab_3e[3][3], perm_tab_3o[3][3], perm_tab_4e[12][4], perm_tab_4o[12][4];
EXTERN double perm_tab_3_sign[6];

EXTERN int g_gamma_mult_table[16][16];
EXTERN double g_gamma_mult_sign[16][16];
EXTERN double g_gamma_adjoint_sign[16];
EXTERN double g_gamma_transposed_sign[16];

EXTERN int g_resume, g_subtract;
EXTERN int g_source_location, g_source_coords_list[MAX_SOURCE_LOCATION_NUMBER][4], g_source_location_number;

EXTERN unsigned int g_seed;
EXTERN int g_noise_type, g_source_type;

EXTERN double solver_precision, reliable_delta;
EXTERN int niter_max;
EXTERN int hpe_order_min, hpe_order_max, hpe_order;

EXTERN double g_cutangle, g_cutradius, g_rmin, g_rmax;
EXTERN int g_cutdir[4];

EXTERN int avgL, avgT;

EXTERN double model_dcoeff_re, model_dcoeff_im, model_mrho;
EXTERN double g_prop_normsqr;
EXTERN double g_qhatsqr_min, g_qhatsqr_max;

EXTERN int Nlong, N_ape, N_Jacobi;
EXTERN double alpha_ape, kappa_Jacobi;
EXTERN int g_source_timeslice, g_no_extra_masses, g_no_light_masses, g_no_strange_masses, \
         g_sequential_source_timeslice, g_sequential_source_location_x, g_sequential_source_location_y, g_sequential_source_location_z, \
         g_sequential_source_timeslice_list[MAX_SEQUENTIAL_SOURCE_TIMESLICE_NUMBER], g_sequential_source_timeslice_number;
EXTERN int N_hyp;
EXTERN double alpha_hyp[3];

EXTERN int g_local_local, g_local_smeared, g_smeared_local, g_smeared_smeared;
EXTERN int g_rotate_ETMC_UKQCD;
EXTERN time_t g_the_time;
EXTERN int g_propagator_position;

EXTERN int g_gpu_device_number, g_gpu_per_node;

EXTERN int g_coherent_source, g_coherent_source_base, g_coherent_source_delta;

EXTERN int g_gauge_file_format;

EXTERN char g_rng_filename[100];
EXTERN int g_source_index[2];
EXTERN int g_propagator_bc_type, g_propagator_gamma_basis;
EXTERN int g_propagator_precision;
EXTERN int g_write_source, g_read_source, g_write_propagator, g_read_propagator, g_read_sequential_propagator, g_write_sequential_source, g_write_sequential_propagator;
EXTERN int g_nsample, g_nsample_oet;
EXTERN int g_sv_dim, g_cv_dim, g_fv_dim, g_cm_dim, g_fp_dim;
EXTERN double g_as_over_a;
EXTERN int g_num_threads;
EXTERN int g_source_momentum[3], g_source_momentum_set, g_source_momentum_list[MAX_MOMENTUM_NUMBER][3], g_source_momentum_number;
EXTERN int g_sink_momentum[3], g_sink_momentum_set, g_sink_momentum_list[MAX_MOMENTUM_NUMBER][3], g_sink_momentum_number;
EXTERN int g_seq_source_momentum[3], g_seq_source_momentum_set, g_seq_source_momentum_list[MAX_MOMENTUM_NUMBER][3], g_seq_source_momentum_number;
EXTERN int g_seq2_source_momentum[3], g_seq2_source_momentum_set, g_seq2_source_momentum_list[MAX_MOMENTUM_NUMBER][3], g_seq2_source_momentum_number;
EXTERN int *g_rng_state;
EXTERN int g_verbose;
EXTERN int g_source_proc_id;
EXTERN int g_cpu_prec, g_gpu_prec, g_gpu_prec_sloppy;
EXTERN int g_inverter_type;
EXTERN char g_inverter_type_name[200];
EXTERN int g_space_dilution_depth;
EXTERN int g_mms_id;
EXTERN int g_check_inversion;

EXTERN int g_src_snk_time_separation, g_sequential_source_gamma_id, g_sequential_source_gamma_id_list[16], g_sequential_source_gamma_id_number;
EXTERN int g_source_gamma_id, g_source_gamma_id_list[16], g_source_gamma_id_number;

EXTERN double g_csw, *g_clover_term;

#ifdef HAVE_TMLQCD_LIBWRAPPER
EXTERN tmLQCD_mpi_params g_tmLQCD_mpi;
EXTERN tmLQCD_lat_params g_tmLQCD_lat;
EXTERN tmLQCD_deflator_params g_tmLQCD_defl;
#endif

EXTERN twopoint_function_type g_twopoint_function_list[MAX_TWOPOINT_FUNCTION_NUM];
EXTERN int g_twopoint_function_number;

EXTERN double **g_clover, **g_mzz_up, **g_mzz_dn, **g_mzzinv_up, **g_mzzinv_dn;

EXTERN int g_fermion_type;

EXTERN int g_coherent_source_number;

EXTERN int g_total_momentum_list[MAX_MOMENTUM_NUMBER][3], g_total_momentum_number;

EXTERN char g_gamma_f1_nucleon_list[MAX_GAMMA_NUMBER][100], g_gamma_f2_list[MAX_GAMMA_NUMBER][100], g_gamma_f1_delta_list[MAX_GAMMA_NUMBER][100], g_gamma_current_list[MAX_GAMMA_NUMBER][100];
EXTERN int g_gamma_f1_nucleon_number, g_gamma_f2_number, g_gamma_f1_delta_number, g_gamma_current_number;


}

#endif
