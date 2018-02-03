#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#include <math.h>
#include "cvc_complex.h"
#include "global.h"
#include "cvc_utils.h"
#include "cvc_geometry_3d.h"

namespace cvc {

int *iup_3d=NULL, *idn_3d=NULL, *ipt_3d=NULL , **ipt_3d_=NULL;

unsigned long int get_index_3d(const int x, const int y, const int z)
{

  const unsigned long int VOL3 = LX*LY*LZ;
  unsigned long int xx, yy, zz, ix;

  xx = (x + LX) % LX;
  yy = (y + LY) % LY;
  zz = (z + LZ) % LZ;
  ix = (xx*LY+yy)*LZ+zz;

#ifdef HAVE_MPI
#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ
  if(x==LX) {
    ix = VOL3          + yy*LZ+zz;
  }
  if(x==-1) {
    ix = VOL3 +  LY*LZ + yy*LZ+zz;
  }
#  endif

#  if defined PARALLELTXY || defined PARALLELTXYZ
  if(y==LY) {
    ix = VOL3 + 2*( LY*LZ)         + xx*LZ+zz;
  }
  if(y==-1) {
    ix = VOL3 + 2*( LY*LZ) + LX*LZ + xx*LZ+zz;
  }
#  endif

#if defined PARALLELTXYZ
  if(z==LZ) {
    ix = VOL3 + 2*( LY*LZ + LX*LZ)         + xx*LY+yy;
  }
  if(z==-1) {
    ix = VOL3 + 2*( LY*LZ + LX*LZ) + LX*LY + xx*LY+yy;
  }
#endif

#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ

#  if defined PARALLELTXY || defined PARALLELTXYZ
  /* y-x-edges */
  if(x==LX) {
    if(y==LY) {
      ix = VOL3 + RAND3 +      + zz;
    }
    if(y==-1) {
      ix = VOL3 + RAND3 +   LZ + zz;
    }
  }
  if(x==-1) {
    if(y==LY) {
      ix = VOL3 + RAND3 + 2*LZ + zz;
    }
    if(y==-1) {
      ix = VOL3 + RAND3 + 3*LZ + zz;
    }
  }
#  if defined PARALLELTXYZ

  /* z-x-edges */

  if(x==LX) {
    if(z==LZ) {
      ix = VOL3 + RAND3 + 4*LZ        + yy;
    }
    if(z==-1) {
      ix = VOL3 + RAND3 + 4*LZ +   LY + yy;
    }
  }
  if(x==-1) {
    if(z==LZ) {
      ix = VOL3 + RAND3 + 4*LZ + 2*LY + yy;
    }
    if(z==-1) {
      ix = VOL3 + RAND3 + 4*LZ + 3*LY + yy;
    }
  }


  /* z-y-edges */
  if(y==LY) {
    if(z==LZ) {
      ix = VOL3 + RAND3 + 4*( LZ + LY )        + xx;
    }
    if(z==-1) {
      ix = VOL3 + RAND3 + 4*( LZ + LY ) +   LX + xx;
    }
  }
  if(y==-1) {
    if(z==LZ) {
      ix = VOL3 + RAND3 + 4*( LZ + LY ) + 2*LX + xx;
    }
    if(z==-1) {
      ix = VOL3 + RAND3 + 4*( LZ + LY ) + 3*LX + xx;
    }
  }


#  endif  /* of if defined PARALLELTXYZ */
#  endif  /* of if defined PARALLELTXY || defined PARALLELTXYZ */
#  endif  /* of if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ */
#endif
  return(ix);
}  /* end of get_index_3d */

void geometry_3d() {

  int x1, x2, x3;
  int y1, y2, y3, ix;
  int isboundary;
  int i_even, i_odd;

#ifdef HAVE_MPI

#  if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ
  int start_valuex = 1;
#  else
  int start_valuex = 0;
#  endif

#  if defined PARALLELTXY || defined PARALLELTXYZ
  int start_valuey = 1;
#  else
  int start_valuey = 0;
#  endif

#  if defined PARALLELTXYZ
  int start_valuez = 1;
#  else
  int start_valuez = 0;
#  endif

#else
  int start_valuex = 0;
  int start_valuey = 0;
  int start_valuez = 0;
#endif

  fprintf(stdout, "# [geometry_3d] start_value = ( %3d, %3d, %3d)\n", start_valuex, start_valuey, start_valuez);

  for(x1=-start_valuex; x1<LX+start_valuex; x1++) {
  for(x2=-start_valuey; x2<LY+start_valuey; x2++) {
  for(x3=-start_valuez; x3<LZ+start_valuez; x3++) {

    isboundary = 0;
    if(x1==-1 || x1==LX) isboundary++;
    if(x2==-1 || x2==LY) isboundary++;
    if(x3==-1 || x3==LZ) isboundary++;

    y1=x1; y2=x2; y3=x3;
    if(x1==-1) y1=LX+1;
    if(x2==-1) y2=LY+1;
    if(x3==-1) y3=LZ+1;

    if(isboundary > 2) {
      g_ipt_3d[y1][y2][y3] = -1;
      continue;
    }

    ix = get_index_3d( x1, x2, x3);

    g_ipt_3d[y1][y2][y3] = ix;

    g_iup_3d[ix][0] = get_index_3d( x1+1, x2, x3);
    g_iup_3d[ix][1] = get_index_3d( x1, x2+1, x3);
    g_iup_3d[ix][2] = get_index_3d( x1, x2, x3+1);

    g_idn_3d[ix][0] = get_index_3d( x1-1, x2, x3);
    g_idn_3d[ix][1] = get_index_3d( x1, x2-1, x3);
    g_idn_3d[ix][2] = get_index_3d( x1, x2, x3-1);

    /* is even / odd */

    for ( int x0 = 0; it < T; x0++ ) {
      g_iseven[x0][ix] = ( x0 + T *g_proc_coords[0] + x1 + LX*g_proc_coords[1] \
                     + x2 + LY*g_proc_coords[2] + x3 + LZ*g_proc_coords[3] ) % 2 == 0;
    }

  }}} // of x3, x2, x1

  for ( int x0 = 0; x0 < T; x0++ ) {

    i_even = 0; i_odd = 0;
    for(ix=0; ix<(VOL3+RAND3);ix++) {

      if(g_iseven_3d[x0][ix]) {
        g_lexic2eo_3d[x0][ix]     = i_even;
        g_lexic2eosub_3d[x0][ix]  = i_even;
        g_eo2lexic_3d[x0][i_even] = ix;
        i_even++;
      } else {
        g_lexic2eo_3d[x0][ix]     = i_odd + (VOL3+RAND3)/2;
        g_lexic2eosub_3d[x0][ix]  = i_odd;
        g_eo2lexic_3d[x0][i_odd + (VOL3+RAND3)/2] = ix;
        i_odd++;
      }
    }
  }

  return;
}  /* end of geometry_3d */

/************************************************************************/
/************************************************************************/

/************************************************************************
 *
 ************************************************************************/
int init_geometry_3d(void) {

  int ix = 0, V;
  int j;
  int dx = 0, dy = 0, dz = 0;
  unsigned int VOL3half;

  VOL3          = LX*LY*LZ;
  VOL3PLUSRAND3 = VOL3;
  RAND3         = 0;
  EDGES3        = 0;

#ifdef HAVE_MPI

#if defined PARALLELTX || defined PARALLELTXY || defined PARALLELTXYZ
  RAND3         += 2*LY*LZ;
  VOL3PLUSRAND3 += 2*LY*LZ;
  dx = 2;
#endif

#if defined PARALLELTXY || defined PARALLELTXYZ
  RAND           += 2*LX*LZ;
  EDGES          +=           4*LZ;
  VOLUMEPLUSRAND += 2*LX*LZ + 4*LZ;
  dy = 2;
#endif

#if defined PARALLELTXYZ
  RAND           += 2*LX*LY;
  EDGES          +=           4*LY + 4*LX;
  VOLUMEPLUSRAND += 2*LX*LY + 4*LY + 4*LX;
  dz = 2;
#endif

#endif  /* of ifdef HAVE_MPI */

  if(g_cart_id==0) {
    fprintf(stdout, "# [init_geometry_3d] (LX, LY, LZ) = (%d, %d, %d)\n", LX, LY, LZ);
    fprintf(stdout, "# [init_geometry_3d] VOL3   = %d\n", VOL3);
    fprintf(stdout, "# [init_geometry_3d] RAND3  = %d\n", RAND3);
    fprintf(stdout, "# [init_geometry_3d] EDGES3 = %d\n", EDGES3);
    fprintf(stdout, "# [init_geometry_3d] VOL3PLUSRAND3 = %d\n", VOL3PLUSRAND3 );
  }

  V = VOL3PLUSRAND3;

  VOL3half = VOL3 / 2;

  g_idn_3d = (int**)calloc(V, sizeof(int*));
  if((void*)g_idn_3d == NULL) return(1);

  idn_3d = (int*)calloc(3*V, sizeof(int));
  if((void*)idn_3d == NULL) return(2);

  g_iup_3d = (int**)calloc(V, sizeof(int*));
  if((void*)g_iup_3d == NULL) return(3);

  iup_3d = (int*)calloc(3*V, sizeof(int));
  if((void*)iup_3d==NULL) return(4);

  g_ipt_3d = (int***)calloc(LX+2, sizeof(int*));
  if((void*)g_ipt_3d == NULL) return(5);

  ipt_3d_ =  (int**)calloc((LX+2)*(LY+dy), sizeof(int*));
  if((void*)ipt_3d_ == NULL) return(7);

  ipt_3d =   (int*)calloc( (LX+dx)*(LY+dy)*(LZ+dz), sizeof(int));
  if((void*)ipt_3d == NULL) return(8);

 
  g_iup_3d[0] = iup_3d;
  g_idn_3d[0] = idn_3d;
  for(ix=1; ix<V; ix++) {
    g_iup_3d[ix] = g_iup_3d[ix-1] + 3;
    g_idn_3d[ix] = g_idn_3d[ix-1] + 3;
  }

  ipt_3d_[0]  = ipt_3d;

  g_ipt_3d[0] = ipt_3d_;

  for(ix=1; ix<(LX+dx)*(LY+dy); ix++) ipt_3d_[ix] = ipt_3d_[ix-1] + (LZ+dz);

  for(ix=1; ix<(LX+dx);         ix++) g_ipt_3d[ix] = g_ipt_3d[ix-1] + (LY+dy);


  g_iseven_3d = (int**)calloc(T, sizeof(int*));
  if(g_iseven_3d == NULL) return(12);
  g_iseven_3d[0] = (int*)calloc(T*V, sizeof(int));
  if(g_iseven_3d[0] == NULL) return(14);
  for ( ix = 1; ix < T; ix++ ) g_iseven_3d[ix] = g_iseven_3d[ix-1] + V;
  
  g_lexic2eo_3d = (int**)calloc(T, sizeof(int*));
  if(g_lexic2eo_3d == NULL) return(9);
  g_lexic2eo_3d[0] = (int*)calloc(T*V, sizeof(int));
  if(g_lexic2eo_3d[0] == NULL) return(9);
  for ( ix = 1; ix < T; ix++ ) g_lexic2eo_3d[ix] = g_lexic2eo_3d[ix-1] + V;

  g_eo2lexic_3d = (int**)calloc(T, sizeof(int*));
  if(g_eo2lexic_3d == NULL) return(9);
  g_eo2lexic_3d[0] = (int*)calloc(T*V, sizeof(int));
  if(g_eo2lexic_3d[0] == NULL) return(9);
  for ( ix = 1; ix < T; ix++ ) g_eo2lexic_3d[ix] = g_eo2lexic_3d[ix-1] + V;

  g_lexic2eosub_3d = (int**)calloc(T, sizeof(int*));
  if(g_lexic2eosub_3d == NULL) return(9);
  g_lexic2eosub_3d[0] = (int*)calloc(T*V, sizeof(int));
  if(g_lexic2eosub_3d[0] == NULL) return(9);
  for ( ix = 1; ix < T; ix++ ) g_lexic2eosub_3d[ix] = g_lexic2eosub_3d[ix-1] + V;

  /***********************************************************/
  /***********************************************************/

  return(0);
}  /* end of init_geometry_3d */

/***********************************************************/
/***********************************************************/

void free_geometry_3d() {

  free(idn_3d);
  free(iup_3d);
  free(ipt_3d);
  free(ipt_3d_);
  free(g_ipt_3d);
  free(g_idn_3d);
  free(g_iup_3d);
  free(g_iseven_3d[0]);
  free(g_iseven_3d);

  free(g_lexic2eo_3d[0]);
  free(g_lexic2eo_3d);
  free(g_lexic2eosub_3d[0]);
  free(g_lexic2eosub_3d);
  free(g_eo2lexic_3d[0]);
  free(g_eo2lexic_3d);

}  /* end of free_geometry_3d */

/***********************************************************/
/***********************************************************/

}  /* end of namespace cvc */
