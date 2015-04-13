/****************************************************
 * get_index.c
 *
 * Thu Oct  8 20:04:05 CEST 2009
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MPI
#  include <mpi.h>
#endif
#include <getopt.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"

unsigned long int get_indexf(const int t, const int x, const int y, const int z, const int mu, const int nu)
/* formats:
  - 0 : my format with 4*nu+mu, from cvc
  - 1 : my format with 4*mu+nu, from cvc_disc
  - 2 : Xu's format with formula  qt*L^3*16*2+qz*L^2*16*2+qy*L*16*2+qx*16*2+nu*4*2+mu*2
*/
{
  unsigned long int tt = (t+T_global)%T_global;
  unsigned long int xx = (x+LX)%LX;
  unsigned long int yy = (y+LY)%LY;
  unsigned long int zz = (z+LZ)%LZ;
  int perm_tab[4];
  
  if(format==0) {
    return( ( (4*nu+mu)*T_global*LX*LY*LZ + (((tt*LX+xx)*LY+yy)*LZ+zz) ) * 2);
  }
  else if(format==1) {
    return( ( (4*mu+nu)*T*LX*LY*LZ + (((tt*LX+xx)*LY+yy)*LZ+zz) ) * 2);
  }
  else if(format==2) {
    perm_tab[0]=3; perm_tab[1]=0; perm_tab[2]=1; perm_tab[3]=2;
    return( ((((tt*LZ+zz)*LY+yy)*LX+xx)*16 + perm_tab[mu]*4 + perm_tab[nu]) * 2);
  }
  else {
    fprintf(stderr, "Error: unrecongnized format\n");
  }
  return(-1);
}

unsigned long int index_conv(const unsigned long int ix, const int format)
/* convert formats 0 and 2 to my format 1 */
{
  unsigned long int tt, xx, yy, zz, iy;
  unsigned long int mu, nu;
  unsigned long int perm_tab[4];
  
  if(format==0) {
    mu = ix / VOLUME;
    iy = ix % VOLUME;
    nu = mu / 4; mu = mu % 4;
    return(_GWI(4*mu+nu,iy,VOLUME));
  }
  else if(format==2) {
    /* conversion from Xu's/Dru's format */
    perm_tab[0]=1; perm_tab[1]=2; perm_tab[2]=3; perm_tab[3]=0;
    iy = ix / 16;
    mu = ix % 16;
    nu = mu / 4;
    mu = mu % 4;
    tt = iy/(LX*LY*LZ);
    zz = (iy%(LX*LY*LZ))/(LX*LY);
    yy = (iy%(LX*LY))/LX;
    xx = iy%LX;
    iy = ((tt * LX + xx) * LY + yy) * LZ + zz;
    
/*    fprintf(stdout, "ix=%8d\tmu=%3d,nu=%3d,tt=%3d,xx=%3d,yy=%3d,zz=%3d\tiy=%8d\tnew=%9d\n", ix, mu, nu, tt,xx,yy,zz,iy, _GWI(perm_tab[mu]*4+perm_tab[nu],iy,VOLUME)); */

    return( _GWI(perm_tab[nu]*4+perm_tab[mu],iy,VOLUME) );
  }
  return(-1);
}

