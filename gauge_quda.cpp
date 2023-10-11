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

}  /* end of namespace cvc */
