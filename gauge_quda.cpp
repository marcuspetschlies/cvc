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

int get_gauge_padding ( int x[4] )
{
  int pad = 0;
#ifdef HAVE_MPI
  int volume = x[0] * x[1] * x[2] * x[3];
  int face_size[4];
  for ( int dir=0; dir<4; ++dir )
  {
    face_size[dir] = ( volume / x[dir] ) / 2;
  }
  pad = face_size[0];
  for ( int dir = 1; dir < 4; dir++ )
    if ( face_size[dir] > pad ) pad = face_size[dir];
#endif
  return ( pad );
}  /* end of get_gauge_padding */


/***************************************************************************
 *
 ***************************************************************************/
void gauge_field_cvc_to_qdp ( double ** g_qdp, double * g_cvc )
{

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  size_t const bytes = 18 * sizeof(double);
  
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( int x0=0; x0<T; x0++ )
  {
    for( int x1=0; x1<LX; x1++ )
    {
      for( int x2=0; x2<LY; x2++ )
      {
        for( int x3=0; x3<LZ; x3++ ) 
        {
          /* index running t,z,y,x */
          unsigned int const j = x1 + LX * ( x2 + LY * ( x3 + LZ * x0 ) );
          /* index running t, x, y, z */
          unsigned int const k = x3 + LZ * ( x2 + LY * ( x1 + LX * x0 ) );

          int b = (x0+x1+x2+x3) & 1;
          int qidx = 18 * ( b * VOLUME / 2 + j / 2 );

          memcpy( &(g_qdp[0][qidx]), &(g_cvc[_GGI(k,1)]), bytes );
          memcpy( &(g_qdp[1][qidx]), &(g_cvc[_GGI(k,2)]), bytes );
          memcpy( &(g_qdp[2][qidx]), &(g_cvc[_GGI(k,3)]), bytes );
          memcpy( &(g_qdp[3][qidx]), &(g_cvc[_GGI(k,0)]), bytes );

        }
      }
    }
  }
#ifdef HAVE_OPENMP
} /* end of parallel region */
#endif
  return;
}  /* end of gauge_field_cvc_to_qdp */

/***************************************************************************
 *
 ***************************************************************************/
void gauge_field_qdp_to_cvc ( double * g_cvc, double ** g_qdp )
{

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  size_t const bytes = 18 * sizeof(double);
  
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( int x0=0; x0<T; x0++ )
  {
    for( int x1=0; x1<LX; x1++ )
    {
      for( int x2=0; x2<LY; x2++ )
      {
        for( int x3=0; x3<LZ; x3++ ) 
        {
          unsigned int const j = x1 + LX * ( x2 + LY * ( x3 + LZ * x0 ) );
          unsigned int const k = x3 + LZ * ( x2 + LY * ( x1 + LX * x0 ) );

          int b = (x0+x1+x2+x3) & 1;
          int qidx = 18 * ( b * VOLUME / 2 + j / 2 );

          memcpy( &(g_cvc[_GGI(k,1)]), &(g_qdp[0][qidx]), bytes );
          memcpy( &(g_cvc[_GGI(k,2)]), &(g_qdp[1][qidx]), bytes );
          memcpy( &(g_cvc[_GGI(k,3)]), &(g_qdp[2][qidx]), bytes );
          memcpy( &(g_cvc[_GGI(k,0)]), &(g_qdp[3][qidx]), bytes );
        }
      }
    }
  }
#ifdef HAVE_OPENMP
} /* end of parallel region */
#endif
  return;
}  /* end of gauge_field_cvc_to_qdp */



/***************************************************************************
 * Begin of gauge_param initialization
 ***************************************************************************/
void init_gauge_param (QudaGaugeParam * const gauge_param )
{

  gauge_param->struct_size = sizeof ( QudaGaugeParam );
 
  /* gauge_param->location = QUDA_CUDA_FIELD_LOCATION; */
  gauge_param->location = QUDA_CPU_FIELD_LOCATION;

  gauge_param->X[0] = LX;
  gauge_param->X[1] = LY;
  gauge_param->X[2] = LZ;
  gauge_param->X[3] = T;

  gauge_param->anisotropy    = 1.0;
  gauge_param->tadpole_coeff = 0.0;
  gauge_param->scale         = 0.0;

  // gauge_param->type = QUDA_FLOWED_LINKS;
  gauge_param->gauge_order = QUDA_QDP_GAUGE_ORDER;  /* expect *gauge[mu], even-odd, spacetime, row-column color */

  gauge_param->t_boundary = QUDA_PERIODIC_T; 

  gauge_param->cpu_prec = QUDA_DOUBLE_PRECISION;

  gauge_param->cuda_prec   = QUDA_DOUBLE_PRECISION;
  gauge_param->reconstruct = QUDA_RECONSTRUCT_NO;

  gauge_param->cuda_prec_sloppy   = QUDA_DOUBLE_PRECISION;
  gauge_param->reconstruct_sloppy = QUDA_RECONSTRUCT_NO;

  gauge_param->cuda_prec_refinement_sloppy   = QUDA_DOUBLE_PRECISION;
  gauge_param->reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;

  gauge_param->cuda_prec_precondition   = QUDA_DOUBLE_PRECISION;
  gauge_param->reconstruct_precondition = QUDA_RECONSTRUCT_NO;

  gauge_param->cuda_prec_eigensolver   = QUDA_DOUBLE_PRECISION;
  gauge_param->reconstruct_eigensolver = QUDA_RECONSTRUCT_NO;

  gauge_param->gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param->ga_pad = get_gauge_padding ( gauge_param->X );

  gauge_param->site_ga_pad = 0;

  gauge_param->staple_pad   = 0;
  gauge_param->llfat_ga_pad = 0;
  gauge_param->mom_ga_pad   = 0;
  
  gauge_param->staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  
  gauge_param->staggered_phase_applied = QUDA_STAGGERED_PHASE_NO;

  gauge_param->i_mu = 0.;

  gauge_param->overlap = 0;

  gauge_param->overwrite_mom = false;

  gauge_param->use_resident_gauge  = false;
  gauge_param->use_resident_mom    = false;
  gauge_param->make_resident_gauge = false;
  gauge_param->make_resident_mom   = false;
  gauge_param->return_result_gauge = false;
  gauge_param->return_result_mom   = false;

  gauge_param->gauge_offset = 0;
  gauge_param->mom_offset   = 0;

  /***************************************************************************
   * End of gauge_param initialization
   ***************************************************************************/

  return;
}  /* end of init_gauge_param */

}  /* end of namespace cvc */

#endif  /* ifdef _GFLOW_QUDA */
