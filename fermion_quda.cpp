/***************************************************************************
 *
 * 
 *
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef _GFLOW_QUDA

#warning "including quda header file quda.h directly "
#include "quda.h"


#include "global.h"


namespace cvc {

/***************************************************************************
 * default settings, which worked so far
 ***************************************************************************/
void init_invert_param ( QudaInvertParam * const inv_param )
{
  /***************************************************************************
   * Begin of inv_param initialization
   ***************************************************************************/

  /* typedef struct QudaInvertParam_s
   */ 
    inv_param->struct_size = sizeof ( QudaInvertParam );

    inv_param->input_location  = QUDA_CPU_FIELD_LOCATION;
    inv_param->output_location = QUDA_CPU_FIELD_LOCATION;

    inv_param->dslash_type = QUDA_WILSON_DSLASH;
    inv_param->inv_type  = QUDA_CG_INVERTER;

    inv_param->mass = 0.; 
    inv_param->kappa = 0.;

    inv_param->m5 = 0.;
    inv_param->Ls = 0;

    /* inv_param->b_5[QUDA_MAX_DWF_LS]; */
    /* inv_param->c_5[QUDA_MAX_DWF_LS]; */

    inv_param->eofa_shift = 0.;
    inv_param->eofa_pm = 0.;
    inv_param->mq1 = 0.;
    inv_param->mq2 = 0.;
    inv_param->mq3 = 0.;

    inv_param->mu = 0.;
    inv_param->epsilon = 0.;

    inv_param->twist_flavor = QUDA_TWIST_NO;

    inv_param->laplace3D = -1; /**< omit this direction from laplace operator: x,y,z,t -> 0,1,2,3 (-1 is full 4D) */

    inv_param->tol = 0.;
    inv_param->tol_restart = 0.;
    inv_param->tol_hq = 0.;

    inv_param->compute_true_res = 0;
    inv_param->true_res = 0.;
    inv_param->true_res_hq = 0.;
    inv_param->maxiter = 0;
    inv_param->reliable_delta = 0.;
    inv_param->reliable_delta_refinement = 0.;
    inv_param->use_alternative_reliable = 0;
    inv_param->use_sloppy_partial_accumulator = 0;

    inv_param->solution_accumulator_pipeline = 0;

    inv_param->max_res_increase = 0;

    inv_param->max_res_increase_total = 0;

    inv_param->max_hq_res_increase = 0;

    inv_param->max_hq_res_restart_total = 0;

    inv_param->heavy_quark_check = 0;

    inv_param->pipeline = 0;

    inv_param->num_offset = 0;

    inv_param->num_src = 0;

    inv_param->num_src_per_sub_partition = 0;

    /* inv_param->split_grid[QUDA_MAX_DIM]; */

    inv_param->overlap = 0;

    /* inv_param->offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param->tol_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param->tol_hq_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param->true_res_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param->iter_res_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param->true_res_hq_offset[QUDA_MAX_MULTI_SHIFT]; */

    /* inv_param->residue[QUDA_MAX_MULTI_SHIFT]; */

    inv_param->compute_action = 0;

    /* inv_param->action[2] = {0., 0.}; */

    inv_param->solution_type = QUDA_MAT_SOLUTION;
    inv_param->solve_type = QUDA_DIRECT_SOLVE;
    inv_param->matpc_type = QUDA_MATPC_ODD_ODD;
    inv_param->dagger = QUDA_DAG_NO;
    inv_param->mass_normalization = QUDA_MASS_NORMALIZATION;
    inv_param->solver_normalization = QUDA_DEFAULT_NORMALIZATION;

    inv_param->preserve_source = QUDA_PRESERVE_SOURCE_YES;

    inv_param->cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param->cuda_prec = QUDA_DOUBLE_PRECISION;
    inv_param->cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param->cuda_prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param->cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
    inv_param->cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;

    inv_param->dirac_order = QUDA_DIRAC_ORDER;

    inv_param->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

    inv_param->clover_location = QUDA_CPU_FIELD_LOCATION;
    inv_param->clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param->clover_cuda_prec = QUDA_DOUBLE_PRECISION;
    inv_param->clover_cuda_prec_sloppy = QUDA_DOUBLE_PRECISION; 
    inv_param->clover_cuda_prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param->clover_cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
    inv_param->clover_cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;

    inv_param->clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param->use_init_guess = QUDA_USE_INIT_GUESS_NO; 

    inv_param->clover_csw =  0.;
    inv_param->clover_coeff = 0.;
    inv_param->clover_rho = 0.;

    inv_param->compute_clover_trlog = 0;
    /* inv_param->trlogA[2] = {0.,0.}; */

    inv_param->compute_clover = 0;
    inv_param->compute_clover_inverse = 0;
    inv_param->return_clover = 0;
    inv_param->return_clover_inverse = 0;

    inv_param->verbosity = QUDA_SILENT;  

    /* inv_param->sp_pad = 0; */
    /* inv_param->cl_pad = 0; */

    inv_param->iter = 0; 
    inv_param->gflops = 0.;
    inv_param->secs = 0.;

    inv_param->tune = QUDA_TUNE_YES;

    inv_param->Nsteps = 0;

    inv_param->gcrNkrylov = 0;

    inv_param->inv_type_precondition = QUDA_CG_INVERTER;

    inv_param->preconditioner = NULL;

    inv_param->deflation_op = NULL;

    inv_param->eig_param = NULL;

    inv_param->deflate = QUDA_BOOLEAN_FALSE;

    inv_param->dslash_type_precondition = QUDA_WILSON_DSLASH;

    inv_param->verbosity_precondition = QUDA_SILENT;

    inv_param->tol_precondition = 0.;

    inv_param->maxiter_precondition = 0;

    inv_param->omega = 0.;

    /* inv_param->ca_basis = QUDA_CHEBYSHEV_BASIS; */

    inv_param->ca_lambda_min = 0.;

    inv_param->ca_lambda_max = 0.;

    inv_param->precondition_cycle = 0;

    /* inv_param->schwarz_type = QUDA_ADDITIVE_SCHWARZ; */

    inv_param->residual_type = QUDA_L2_RELATIVE_RESIDUAL;

    inv_param->cuda_prec_ritz = QUDA_DOUBLE_PRECISION;
    
    inv_param->n_ev = 0;

    inv_param->max_search_dim = 0;

    inv_param->rhs_idx = 0;

    inv_param->deflation_grid = 0;

    inv_param->eigenval_tol = 0.;

    inv_param->eigcg_max_restarts = 0;

    inv_param->max_restart_num =0;

    inv_param->inc_tol = 0.;

    inv_param->make_resident_solution = false;

    inv_param->use_resident_solution = false;

    /* inv_param->chrono_make_resident; */

    /* inv_param->chrono_replace_last; */

    /* inv_param->chrono_use_resident; */

    /* inv_param->chrono_max_dim; */
    /* inv_param->chrono_index; */

    /* inv_param->chrono_precision;  */

    /* inv_param->extlib_type; */

    /* inv_param->native_blas_lapack = */

  /***************************************************************************
   * End of inv_param initialization
   ***************************************************************************/

  return;
}  /* end of init_invert_param */


/***************************************************************************
 * reordering fermion field to quda data layout
 *
 * in: s
 * out: r
 ***************************************************************************/
void spinor_field_cvc_to_quda ( double * const r, double * const s ) 
{
  size_t const sizeof_spinor_field = _GSI(VOLUME) * sizeof (double);
  size_t bytes = _GSI(1) * sizeof(double);

  double * aux = (double *)malloc ( sizeof_spinor_field );
  
  memcpy ( aux, s, sizeof_spinor_field );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix = 0; ix < VOLUME; ix++ )
  {
    int const x0 = g_lexic2coords[ix][0];
    int const x1 = g_lexic2coords[ix][1];
    int const x2 = g_lexic2coords[ix][2];
    int const x3 = g_lexic2coords[ix][3];
      
    /* index running t,z,y,x */
    unsigned int const j = x1 + LX * ( x2 + LY * ( x3 + LZ * x0 ) );
    /* index running t, x, y, z */
    unsigned int const k = x3 + LZ * ( x2 + LY * ( x1 + LX * x0 ) );

    int b = (x0+x1+x2+x3) & 1;
    unsigned int qidx = _GSI( b * VOLUME / 2 + j / 2 );

    memcpy( r + qidx, aux + _GSI(k), bytes );
  }
  
  free ( aux );

}  /* end of spinor_field_cvc_to_quda */


}  /* end of namespace cvc */

#endif  /* ifdef _GFLOW_QUDA */
