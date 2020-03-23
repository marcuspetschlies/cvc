#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

#include "types.h"
#include "cvc_timer.h"
#include "global.h"
#include "default_input_values.h"
#include "io_utils.h"
#include "zm4x4.h"
#include "table_init_z.h"
#include "rotations.h"
#include "group_projection.h"
#include "contract_diagrams.h"
#include "twopoint_function_utils.h"

namespace cvc {

/* [sigma_gamma_adj_g0_dagger] Thu Jul 13 09:48:39 2017*/
  const int sigma_gamma_adj_g0_dagger[16]  = { 1,   -1,   -1,   -1,    1,   -1,    1,   -1,   -1,   -1,    1,    1,    1,   -1,   -1,   -1};
  const int sigma_gamma_imag[16]           = { 0,    1,    1,    1,    0,    1,    0,    1,    1,    1,    0,    0,    0,    1,    1,    1};
  const int sigma_Cgamma_adj_g0_dagger[16] = { 1,    1,   -1,    1,   -1,    1,    1,    1,   -1,    1,    1,   -1,    1,   -1,    1,   -1};

const int gamma_to_C_gamma[16][2] = {
  { 2,-1},
  { 9, 1},
  { 0, 1},
  { 7,-1},
  {11, 1},
  {14, 1},
  { 8,-1},
  { 3, 1},
  { 6, 1},
  { 1,-1},
  {13, 1},
  { 4,-1},
  {15,-1},
  {10,-1},
  { 5,-1},
  {12, 1} };


/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_init ( twopoint_function_type *p ) {

  p->pi1[0] = 0;
  p->pi1[1] = 0;
  p->pi1[2] = 0;
  p->pi2[0] = 0;
  p->pi2[1] = 0;
  p->pi2[2] = 0;
  p->pf1[0] = 0;
  p->pf1[1] = 0;
  p->pf1[2] = 0;
  p->pf2[0] = 0;
  p->pf2[1] = 0;
  p->pf2[2] = 0;

  p->gi1[0] = -1;
  p->gi1[1] = -1;
  p->gf1[0] = -1;
  p->gf1[1] = -1;
  p->gi2    = -1;
  p->gf2    = -1;

  p->si1[0] = 1.;
  p->si1[1] = 1.;
  p->sf1[0] = 1.;
  p->sf1[1] = 1.;
  p->si2    = 1.;
  p->sf2    = 1.;

  p->n      = 0;
  strcpy( p->type, "NA" );
  strcpy( p->name, "NA" );
  strcpy( p->diagrams, "NA" );
  strcpy( p->norm, "NA" );
  p->spin_project = -1;
  p->parity_project = 0;
  p->source_coords[0] = -1;
  p->source_coords[1] = -1;
  p->source_coords[2] = -1;
  p->source_coords[3] = -1;
  p->reorder = 0;
  strcpy( p->fbwd, "NA" );
  strcpy( p->group, "NA" );
  strcpy( p->irrep, "NA" );
  p->c = NULL;
  p->T = -1;
  p->d = 0;
}  // end of twopoint_function_init

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_print ( twopoint_function_type *p, char *name, FILE*ofs ) {
  fprintf(ofs, "# [twopoint_function_print] %s.name = %s\n", name, p->name);
  fprintf(ofs, "# [twopoint_function_print] %s.type = %s\n", name, p->type);
  fprintf(ofs, "# [twopoint_function_print] %s.pi1  = (%3d,%3d, %3d)\n", name, p->pi1[0], p->pi1[1], p->pi1[2] );
  fprintf(ofs, "# [twopoint_function_print] %s.pi2  = (%3d,%3d, %3d)\n", name, p->pi2[0], p->pi2[1], p->pi2[2] );
  fprintf(ofs, "# [twopoint_function_print] %s.pf1  = (%3d,%3d, %3d)\n", name, p->pf1[0], p->pf1[1], p->pf1[2] );
  fprintf(ofs, "# [twopoint_function_print] %s.pf2  = (%3d,%3d, %3d)\n", name, p->pf2[0], p->pf2[1], p->pf2[2] );

  fprintf(ofs, "# [twopoint_function_print] %s.gi1  = (%3d,%3d)\n", name, p->gi1[0], p->gi1[1] );
  fprintf(ofs, "# [twopoint_function_print] %s.gi2  =  %3d\n", name, p->gi2 );
  fprintf(ofs, "# [twopoint_function_print] %s.gf1  = (%3d,%3d)\n", name, p->gf1[0], p->gf1[1] );
  fprintf(ofs, "# [twopoint_function_print] %s.gf2  =  %3d\n", name, p->gf2 );

  fprintf(ofs, "# [twopoint_function_print] %s.si1  = ( %16.7e + I %16.7e, %16.7e + I %16.7e )\n", name, 
      creal(p->si1[0]), cimag(p->si1[0]), creal(p->si1[1]), cimag(p->si1[1]) );
  fprintf(ofs, "# [twopoint_function_print] %s.si2  =   %16.7e + I %16.7e\n", name, creal( p->si2 ), cimag ( p->si2 ));
  fprintf(ofs, "# [twopoint_function_print] %s.sf1  = ( %16.7e + I %16.7e, %16.7e + I %16.7e )\n", name, 
      creal(p->sf1[0]), cimag(p->sf1[0]), creal(p->sf1[1]), cimag(p->sf1[1]) );
  fprintf(ofs, "# [twopoint_function_print] %s.sf2  =   %16.7e + I %16.7e\n", name, creal( p->sf2 ), cimag( p->sf2 ) );

  fprintf(ofs, "# [twopoint_function_print] %s.n    =  %3d\n", name, p->n );
  fprintf(ofs, "# [twopoint_function_print] %s.spin_project    =  %3d\n", name, p->spin_project );
  fprintf(ofs, "# [twopoint_function_print] %s.parity_project  =  %3d\n", name, p->parity_project );
  fprintf(ofs, "# [twopoint_function_print] %s.diagrams        =  %s\n", name, p->diagrams );
  fprintf(ofs, "# [twopoint_function_print] %s.norm            =  %s\n", name, p->norm );
  fprintf(ofs, "# [twopoint_function_print] %s.source_coords   =  (%3d, %3d, %3d, %3d)\n", name, 
      p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );
  fprintf(ofs, "# [twopoint_function_print] %s.reorder         =  %3d\n", name, p->reorder );
  fprintf(ofs, "# [twopoint_function_print] %s.fbwd            =  %s\n", name, p->fbwd );
  fprintf(ofs, "# [twopoint_function_print] %s.T               =  %d\n", name, p->T );
  fprintf(ofs, "# [twopoint_function_print] %s.d               =  %d\n", name, p->d );
  fprintf(ofs, "# [twopoint_function_print] %s.group           =  %s\n", name, p->group );
  fprintf(ofs, "# [twopoint_function_print] %s.irrep           =  %s\n", name, p->irrep );
  if ( p->c != NULL ) {
    fprintf ( ofs, "# [twopoint_function_print] data array is set\n");
  } else {
    fprintf ( ofs, "# [twopoint_function_print] data is empty\n");
  }
  return;
}  // end of twopoint_function_print

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_copy ( twopoint_function_type *p, twopoint_function_type *r, int const copy_data ) {
  strcpy( p->name, r->name );
  strcpy( p->type, r->type );
  strcpy( p->diagrams, r->diagrams );
  strcpy( p->norm, r->norm );
  strcpy( p->fbwd, r->fbwd );
  strcpy( p->group, r->group );
  strcpy( p->irrep, r->irrep );
  memcpy( p->pi1, r->pi1, 3*sizeof(int) );
  memcpy( p->pi2, r->pi2, 3*sizeof(int) );
  memcpy( p->pf1, r->pf1, 3*sizeof(int) );
  memcpy( p->pf2, r->pf2, 3*sizeof(int) );

  memcpy( p->gi1, r->gi1, 2*sizeof(int) );
  memcpy( p->gf1, r->gf1, 2*sizeof(int) );
  p->gi2 = r->gi2;
  p->gf2 = r->gf2;

  memcpy( p->si1, r->si1, 2*sizeof(double _Complex) );
  memcpy( p->sf1, r->sf1, 2*sizeof(double _Complex) );
  p->si2 = r->si2;
  p->sf2 = r->sf2;

  p->n              = r->n;
  p->spin_project   = r->spin_project;
  p->parity_project = r->parity_project;
  memcpy( p->source_coords, r->source_coords, 4*sizeof(int) );
  p->reorder        = r->reorder;
  p->T              = r->T;
  p->d              = r->d;
  if ( copy_data ) {
    if ( r->c != NULL ) {
      if ( twopoint_function_allocate ( p ) == NULL ) {
        fprintf ( stderr, "[twopoint_function_copy] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
      } else {
        memcpy ( p->c[0][0][0], r->c[0][0][0], p->n * p->T * p->d * p->d * sizeof( double _Complex ) );
      }
    } else {
      fprintf ( stderr, "[twopoint_function_copy] Warning, source data array is NULL, cannot copy data %s %d\n", __FILE__, __LINE__ );
    }  /* end of if copy_data */
  }
  return;
}  // end of twopoint_function_copy

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_show_data ( twopoint_function_type *p, FILE*ofs ) {
  if ( ofs == NULL ) ofs = stdout;
  if ( p->c == NULL ) {
    fprintf ( stderr, "[twopoint_function_show_data] Error, data not set\n");
    return;
  }
  for ( int i = 0; i < p->n; i++ ) {
    fprintf ( ofs, "# [twopoint_function_show_data] %s data set %2d\n", p->name, i ); 
    for ( int it = 0; it < p->T; it++ ) {
      for ( int k1 = 0; k1 < p->d; k1++ ) {
        for ( int k2 = 0; k2 < p->d; k2++ ) {
          double _Complex const z = p->c[i][it][k1][k2];
          fprintf ( ofs, "    %3d %2d %2d %25.16e %25.16e\n", it, k1, k2, creal(z), cimag(z) );
        }}  // end of loop on components k1, k2
    }  // end of loop on timeslices
  }  // end of loop on diagrams
}  // end of twopoint_function_show_data

/********************************************************************************
 *
 ********************************************************************************/
void * twopoint_function_allocate ( twopoint_function_type *p ) {
  if ( p->c != NULL ) {
    fprintf ( stderr, "[twopoint_function_allocate] Error, data pointer not NULL\n");
    return ( NULL );
  }
  p->c = init_4level_ztable ( p->n, p->T, p->d, p->d );
  if ( p->c == NULL ) {
    fprintf ( stderr, "[twopoint_function_allocate] Error, data pointer is NULL %s %d\n", __FILE__, __LINE__ );
    return ( NULL );
  }
  return ( (void*)(p->c) );
}  // end of twopoint_function_allocate

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_fini ( twopoint_function_type *p ) {
  fini_4level_ztable ( &(p->c) );
}  // end of twopoint_function_fini

/********************************************************************************
 *
 ********************************************************************************/

void twopoint_function_print_diagram_key (char*key, twopoint_function_type *p, int id ) { 

  char comma[] = ",";
  char *ptr = NULL;
  char diag_str[20];
  char diagrams[_TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH];
  char fbwd_str[20] = "";

  if ( ! ( strcmp( p->fbwd, "NA" ) == 0 ) ) {
    sprintf ( fbwd_str, "%s/", p->fbwd );
  }


  strcpy( diagrams, p->diagrams );
  if ( id >= p->n ) { strcpy ( key, "NA" ); return; }

  if ( id >= 0 ) {
    ptr = strtok( diagrams, comma );
    if ( ptr == NULL ) { strcpy ( key, "NA" ); return; }
    if ( strcmp ( ptr, "NA" ) == 0 ) { strcpy ( key, "NA" ); return; }
    // fprintf(stdout, "# [twopoint_function_print_diagram_key] %d ptr = %s\n", 0, ptr);

    for( int i = 1; i <= id && ptr != NULL; i++ ) { 
      ptr = strtok(NULL, "," );
      // fprintf(stdout, "# [twopoint_function_print_diagram_key] %d ptr = %s\n", i, ptr);
    }
    if ( ptr == NULL ) { strcpy ( key, "NA" ); return; }
    sprintf( diag_str, "%s/", ptr );

  } else {
    diag_str[0] = '\0';
  }

  if ( strcmp( p->type, "b-b") == 0 ) {

    sprintf( key, "/%s/%s%sgf1%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/gi1%.2d/t%.2dx%.2dy%.2dz%.2d", p->name, diag_str, fbwd_str,
        p->gf1[0], p->pf1[0], p->pf1[1], p->pf1[2], p->gi1[0],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );


    /* sprintf( key, "/%s/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/%spx%.2dpy%.2dpz%.2d", p->name,
       p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3],
       gamma_to_C_gamma[p->gi1[0]][0], gamma_to_C_gamma[p->gf1[0]][0], diag_str, p->pf1[0], p->pf1[1], p->pf1[2] ); */


  } else if ( strcmp( p->type, "mxb-mxb") == 0 ) {

    //sprintf( key, "/%s/%s%spi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d", p->name, diag_str, fbwd_str,
    //    p->pi2[0], p->pi2[1], p->pi2[2],
    //    p->pf1[0], p->pf1[1], p->pf1[2],
    //    p->pf2[0], p->pf2[1], p->pf2[2],
    //    p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3],
    //    p->gf1[0], p->gi1[0]);

    // /pixN-pixN/z1/fwd/gf205/pf2x-01pf2y00pf2z00/gf105/pf1x01pf1y00pf1z00/gi205/pi2x00pi2y00pi2z-01/gi105/t13x02y03z04

    sprintf( key, "/%s/%s%sgf2%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/gf1%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/gi2%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi1%.2d/t%.2dx%.2dy%.2dz%.2d", 
        p->name, diag_str, fbwd_str,
        p->gf2,
        p->pf2[0], p->pf2[1], p->pf2[2],
        p->gf1[0],
        p->pf1[0], p->pf1[1], p->pf1[2],
        p->gi2,
        p->pi2[0], p->pi2[1], p->pi2[2],
        p->gi1[0],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );


  } else if ( strcmp( p->type, "mxb-b") == 0 ) {

    sprintf( key, "/%s/%s%spi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d", p->name, diag_str, fbwd_str,
        p->pi2[0], p->pi2[1], p->pi2[2],
        p->pf1[0], p->pf1[1], p->pf1[2],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3],
        p->gf1[0], p->gi1[0]);

    sprintf( key, "/%s/%s%sgf1%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/gi2%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi1%.2d/t%.2dx%.2dy%.2dz%.2d", p->name, diag_str, fbwd_str,
        p->gf1[0],
        p->pf1[0], p->pf1[1], p->pf1[2],
        p->gi2,
        p->pi2[0], p->pi2[1], p->pi2[2],
        p->gi1[0],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );


    /* /piN-D/t54x14y01z32/pi2x01pi2y01pi2z-01/gi02/gf15/t5/px01py01pz-01 */
    /* sprintf( key, "/%s/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/%spx%.2dpy%.2dpz%.2d", p->name, 
       p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3],
       p->pi2[0], p->pi2[1], p->pi2[2],
       gamma_to_C_gamma[p->gi1[0]][0], gamma_to_C_gamma[p->gf1[0]][0],
       diag_str,
       p->pf1[0], p->pf1[1], p->pf1[2]); */

  } else if ( strcmp( p->type, "m-m") == 0 ) {

    sprintf( key, "/%s/%s%sgf2%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/gi2%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/t%.2dx%.2dy%.2dz%.2d", p->name, diag_str, fbwd_str,
        p->gf2, p->pf2[0], p->pf2[1], p->pf2[2],
        p->gi2, p->pi2[0], p->pi2[1], p->pi2[2],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );
  } else {
    strcpy ( key, "NA" );
  }
  fprintf(stdout, "# [twopoint_function_print_diagram_key] key = \"%s\"\n", key);
  return;
}  // end of twopoint_function_print_key


/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_print_correlator_key (char*key, twopoint_function_type *p ) { 

  char fbwd_str[20] = "";

  if ( strcmp( p->fbwd, "NA" ) !=  0 ) {
    sprintf ( fbwd_str, "/%s", p->fbwd );
  }


  if ( strcmp( p->type, "b-b") == 0 ) {
    sprintf( key, "/%s%s/pf1x%.2dpf1y%.2dpf1z%.2d/pi1x%.2dpi1y%.2dpi1z%.2d/gf1%.2dgf2%.2d/gi1%.2dgi2%.2d/t%.2dx%.2dy%.2dz%.2d", p->name, fbwd_str,
        p->pf1[0], p->pf1[1], p->pf1[2],
        p->pi1[0], p->pi1[1], p->pi1[2],
        p->gf1[0], p->gf1[1], p->gi1[0], p->gi1[1],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3]);
  } else if ( strcmp( p->type, "mxb-mxb") == 0 ) {
    sprintf( key, "/%s%s/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/pi1x%.2dpi1y%.2dpi1z%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gf1%.2dgf2%.2d/gi1%.2dgi2%.2d/t%.2dx%.2dy%.2dz%.2d", p->name, fbwd_str,
        p->pf1[0], p->pf1[1], p->pf1[2],
        p->pf2[0], p->pf2[1], p->pf2[2],
        p->pi1[0], p->pi1[1], p->pi1[2],
        p->pi2[0], p->pi2[1], p->pi2[2],
        p->gf1[0], p->gf1[1], p->gi1[0], p->gi1[1],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3]);
  } else if ( strcmp( p->type, "mxb-b") == 0 ) {
    sprintf( key, "/%s%s/pf1x%.2dpf1y%.2dpf1z%.2d/pi1x%.2dpi1y%.2dpi1z%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gf1%.2dgf2%.2d/gi1%.2dgi2%.2d/t%.2dx%.2dy%.2dz%.2d", p->name, fbwd_str,
        p->pf1[0], p->pf1[1], p->pf1[2],
        p->pi1[0], p->pi1[1], p->pi1[2],
        p->pi2[0], p->pi2[1], p->pi2[2],
        p->gf1[0], p->gf1[1], p->gi1[0], p->gi1[1],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3]);
  } else {
    strcpy ( key, "NA" );
  }
  if ( p->parity_project == 1 ) {
    strcat( key, "/parity+1" );
  } else if ( p->parity_project == -1 ) {
    strcat( key, "/parity-1" );
  }
  fprintf(stdout, "# [twopoint_function_print_correlator_key] key = \"%s\"\n", key);
  return;
}  /* end of twopoint_function_print_correlator_key */

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 *
 ********************************************************************************/
double _Complex twopoint_function_get_correlator_phase ( twopoint_function_type *p ) {

  double _Complex zsign = 0.;

  if ( strcmp( p->type , "b-b" ) == 0 ) {

    zsign = (double _Complex) ( sigma_Cgamma_adj_g0_dagger[p->gi1[0]] * sigma_gamma_adj_g0_dagger[p->gi1[1]] );

  } else if ( strcmp( p->type , "mxb-b" ) == 0 ) {

    zsign = (double _Complex) ( sigma_Cgamma_adj_g0_dagger[p->gi1[0]] * sigma_gamma_adj_g0_dagger[p->gi1[1]] * sigma_gamma_adj_g0_dagger[p->gi2] );
    zsign *= sigma_gamma_imag[p->gi2] ? I : 1.;

  } else if ( strcmp( p->type , "mxb-mxb" ) == 0 ) {
    zsign = (double _Complex) ( sigma_Cgamma_adj_g0_dagger[p->gi1[0]] * sigma_gamma_adj_g0_dagger[p->gi1[1]] * sigma_gamma_adj_g0_dagger[p->gi2] );
    zsign *= sigma_gamma_imag[p->gi2] ? I : 1.;
    zsign *= sigma_gamma_imag[p->gf2] ? I : 1.;
  }

  if (g_verbose > 2) fprintf(stdout, "# [twopoint_function_correlator_phase] gf1 = %2d - %2d gf2 = %2d gi1 = %2d - %2d gi2 = %2d sign = %3.0f  %3.0f\n",
      p->gf1[0], p->gf1[1], p->gf2, p->gi1[0], p->gi1[1], p->gi2, creal(zsign), cimag(zsign) );

  return( zsign );

}  // end of twopoint_function_get_correlator_phase

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_correlator_phase (double _Complex * const c, twopoint_function_type *p, unsigned int const N) {

  double _Complex const zsign =  twopoint_function_get_correlator_phase ( p );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( unsigned int i = 0; i < N; i++ ) {
    c[i] *= zsign;
  }
  return;
}  // end of twopoint_function_correlator_phase

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_print_correlator_data ( double _Complex * const c, twopoint_function_type *p, FILE*ofs, int const N ) { 

  char key[400];

  if ( c == NULL || p == NULL || ofs == NULL ) {
    fprintf(stderr, "[twopoint_function_print_correlator_data] Error, incorrect arguments\n");
    return;
  }

  twopoint_function_print_correlator_key ( key, p );
  fprintf(ofs, "# %s\n", key );
  for ( int it = 0; it < N; it++ ) {
    fprintf(ofs, "  %25.16e %25.16e\n", creal(c[it]), cimag(c[it]));
  }
  return;
}  // end of twopoint_function_print_correlator_data

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_get_aff_filename_prefix (char*filename, twopoint_function_type*p) {

  //  if ( strcmp( p->type , "b-b" ) == 0 || strcmp( p->type , "mxb-b" ) == 0 ) {
  //    strcpy(filename, filename_prefix);
  //  } else if ( strcmp( p->type , "mxb-mxb" ) == 0 ) {
  //    strcpy(filename, filename_prefix2);
  //  } else {
  //    fprintf(stderr, "[twopoint_function_get_aff_filename] Error, unrecognized type %s\n", p->type);
  //  }

  if (     strcmp( p->type , "b-b"     ) == 0 
      || strcmp( p->type , "mxb-b"   ) == 0 
      || strcmp( p->type , "mxb-mxb" ) == 0 
      || strcmp( p->type , "m-m"     ) == 0 ) {
    strcpy(filename, filename_prefix);
  } else {
    fprintf(stderr, "[twopoint_function_get_aff_filename] Error, unrecognized type %s\n", p->type);
  }

  if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_get_aff_filename_prefix] prefix set to %s\n", filename );
}  /* end of twopoint_function_get_aff_filename */

/********************************************************************************
 *
 ********************************************************************************/
double twopoint_function_get_diagram_norm ( twopoint_function_type *p, int const id ) {

  char comma[] = ",";
  char *ptr = NULL;
  char norm[500];
  double r = 0;

  strcpy( norm, p->norm );
  if ( id >= p->n ) { return(0.); }

  if ( id >= 0 ) {
    ptr = strtok( norm, comma );
    if ( ptr == NULL ) { return(0.); }
    if ( strcmp ( ptr, "NA" ) == 0 ) { return(1.); }
    // fprintf(stdout, "# [twopoint_function_get_diagram_norm] %d ptr = %s\n", 0, ptr);

    for( int i = 1; i <= id && ptr != NULL; i++ ) {
      ptr = strtok(NULL, "," );
      // fprintf(stdout, "# [twopoint_function_get_diagram_norm] %d ptr = %s\n", i, ptr);
    }
    if ( ptr == NULL ) { return(0.); }
    r = atof ( ptr );

  } else {
    r = 0.;
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [twopoint_function_get_diagram_norm] diagram norm = %25.16e\n", r);

  return( r );

}  // end of twopoint_function_get_diagram_norm

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_get_diagram_name (char *diagram_name,  twopoint_function_type *p, int const id ) {

  char comma[] = ",";
  char *ptr = NULL;
  char diagrams[_TWOPOINT_FUNCTION_TYPE_MAX_STRING_LENGTH];
  strcpy ( diagram_name, "NA" );

  strcpy( diagrams, p->diagrams );
  if ( id >= p->n ) { return; }

  if ( id >= 0 ) {
    ptr = strtok( diagrams, comma );
    if ( ptr == NULL ) { return; }
    if ( strcmp ( ptr, "NA" ) == 0 ) { return; }
    // fprintf(stdout, "# [twopoint_function_get_diagram_name] %d ptr = %s\n", 0, ptr);

    for( int i = 1; i <= id && ptr != NULL; i++ ) {
      ptr = strtok(NULL, "," );
      // fprintf(stdout, "# [twopoint_function_get_diagram_name] %d ptr = %s\n", i, ptr);
    }
    if ( ptr == NULL ) { return; }
    strcpy ( diagram_name, ptr );
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [twopoint_function_get_diagram_name] diagram name = %s\n", diagram_name);

  return;

}  // end of twopoint_function_get_diagram_name

/********************************************************************************/
/********************************************************************************/


/********************************************************************************
 * read diagrams for a twopoint
 ********************************************************************************/
int twopoint_function_accumulate_diagrams ( double _Complex *** const diagram, twopoint_function_type *p, int const N, struct AffReader_s *affr ) {

  char key[500];
  int exitstatus;

  double _Complex *** buffer = init_3level_ztable ( N, 4, 4 );
  if ( buffer == NULL ) {
    fprintf(stderr, "[twopoint_function_accumulate_diagrams] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

#ifdef HAVE_LHPC_AFF
  struct AffNode_s *affn = NULL, *affdir = NULL;

  if( (affn = aff_reader_root( affr )) == NULL ) {
    fprintf(stderr, "[twopoint_function_accumulate_diagrams] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
    return(103);
  }

#endif

  /******************************************************
   * loop on diagrams within 2-point function
   ******************************************************/
  for ( int idiag = 0; idiag < p->n; idiag++ )
  {

    // fprintf(stdout, "# [twopoint_function_accumulate_diagrams] diagrams %d = %s\n", idiag, g_twopoint_function_list[i2pt].diagrams );

    twopoint_function_print_diagram_key ( key, p, idiag );

#ifdef HAVE_LHPC_AFF
    affdir = aff_reader_chpath (affr, affn, key );
    uint32_t uitems = 16 * N;
    exitstatus = aff_node_get_complex (affr, affdir, buffer[0][0], uitems );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[twopoint_function_accumulate_diagrams] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(105);
    }
#endif

    double const norm = twopoint_function_get_diagram_norm ( p, idiag );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( int i = 0; i < N; i++ ) {
      zm4x4_eq_zm4x4_pl_zm4x4_ti_re ( diagram[i], diagram[i], buffer[i], norm );
    }

  }  // end of loop on diagrams

  fini_3level_ztable ( &buffer );

  // TEST
  if ( g_verbose > 5 ) {
    char name[10] = "c_in";
    for ( int it = 0; it < N; it++ ) {
      fprintf(stdout, "# [twopoint_function_accumulate_diagrams] initial correlator t = %2d\n", it );
      zm4x4_printf ( diagram[it], name, stdout );
    }
  }

  return(0);

}  // end of twopoint_function_accumulate_diagrams

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 * read diagrams for a twopoint
 ********************************************************************************/
int twopoint_function_fill_data ( twopoint_function_type *p, char * datafile_prefix ) {

#ifdef HAVE_LHPC_AFF
  int const nT = p->T;  // timeslices
  int const d  = p->d;  // spin dimension
  int const nD = p->n;  // number of diagrams / data sets
  /* fix: does this need to be generalized ? */
  /* char const filename_prefix[] = "piN_piN_diagrams"; */

  if ( p->c == NULL ) {
    fprintf ( stdout, "# [twopoint_function_fill_data] Warning, data array not intialized; allocing p->c %s %d\n", __FILE__, __LINE__ );
 
    if ( twopoint_function_allocate ( p ) == NULL ) {
      fprintf ( stderr, "[twopoint_function_fill_data] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
      return(1);
    }
  }

  /* baryon correlators, so d = 4 hard-coded here  */
  if ( d != 4 ) {
    fprintf ( stderr, "[twopoint_function_fill_data] Error, spinor dimension must be 4 %s %d\n", __FILE__, __LINE__ );
    return(3);
  }

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  struct AffNode_s *affn = NULL;
  struct AffNode_s *affdir = NULL;
#else
  fprintf ( stderr, "[twopoint_function_fill_data] Error, HAVE_LHPC_AFF not defined %s %d\n", __FILE__, __LINE__ );
  return(1);
#endif

  char key[500];
  char key_suffix[400];
  char diagram_tag[20] = "NA";
  char filename[200];
  int exitstatus;

  /******************************************************
   * total momentum
   ******************************************************/
  int const Ptot[3] = { p->pf1[0] + p->pf2[0], p->pf1[1] + p->pf2[1], p->pf1[2] + p->pf2[2] } ;

  /******************************************************
   * reference frame momentum for current total momentum
   ******************************************************/
  int Pref[3];
  exitstatus = get_reference_rotation ( Pref, NULL, Ptot );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopoint_function_fill_data] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  } 


  /******************************************************
   * for the filename we need the reference momentum
   ******************************************************/

  /* write "last part" of key name into key_suffix;
   * including all momenta and vertex gamma ids */
  exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, p );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopoint_function_fill_data] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  /******************************************************
   * loop on diagrams / data sets within 2-point function
   ******************************************************/
  for ( int i = 0; i < nD; i++ ) {

    double _Complex *** const diagram = p->c[i];

    char diagram_tag_tmp[500];
    twopoint_function_get_diagram_name ( diagram_tag_tmp, p, i );

    if ( /* here we need a new reader */
         ( affr == NULL ) || /* reader not set */
         ( strcmp ( diagram_tag, "NA" ) == 0 ) || /* diagram tag has not been set */
         ( diagram_tag[0] != diagram_tag_tmp[0] ) /* diagram tag has changed */
       ) {

       if ( affr != NULL ) {
        aff_reader_close (affr);
        affr = NULL;
       }
       if ( strcmp ( diagram_tag, diagram_tag_tmp ) != 0 ) strcpy ( diagram_tag, diagram_tag_tmp );

      /******************************************************
       * AFF reader
       ******************************************************/
      sprintf(filename, "%s.%c.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", datafile_prefix, diagram_tag[0],
          Pref[0], Pref[1], Pref[2], Nconf,
          p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );
 
      affr = aff_reader  (filename);
      if ( const char * aff_status_str =  aff_reader_errstr(affr) ) {
        fprintf(stderr, "[twopoint_function_fill_data] Error from aff_reader for filename %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        EXIT(4);
      } else {
        if ( g_verbose > 1 ) fprintf(stdout, "# [twopoint_function_fill_data] reading data from file %s %s %d\n", filename, __FILE__, __LINE__);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[twopoint_function_fill_data] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }
    } else {
      if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_fill_data] using existing reader %s %d\n", __FILE__, __LINE__);
    }  /* end of need new reader */

    /* copy temporary diagram tag to diagram tag */
    strcpy ( diagram_tag, diagram_tag_tmp );

    /* current filename pattern uses lower-case tag, so this 
     * toupper is not needed now */
    /* strcpy ( filename_tag, diagram_tag ); */
    /* filename_tag[0] = toupper ( diagram_tag[0] ); */

    sprintf( key, "/%s/%s/%s/%s", p->name, diagram_tag, p->fbwd, key_suffix );
    if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_fill_data] key = %s %s %d\n", key, __FILE__, __LINE__ );


    affdir = aff_reader_chpath (affr, affn, key );
    uint32_t uitems = d *d * nT;
    exitstatus = aff_node_get_complex (affr, affdir, diagram[0][0], uitems );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[twopoint_function_fill_data] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(105);
    }

    if ( strcmp ( p->norm , "NA" ) != 0 ) {
      double const norm = twopoint_function_get_diagram_norm ( p, i );
      if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_fill_data] using diagram norm %2d %16.7e %s %d\n", i, norm, __FILE__, __LINE__ );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for ( int t = 0; t < nT; t++ ) {
        zm4x4_ti_eq_re ( diagram[t], norm );
      }

    }  // end of if norm not NA
#if 0
#endif  /* of if 0 */

  }  // end of loop on diagrams / data sets

  // close the AFF reader
  if ( affr != NULL ) aff_reader_close ( affr );

  // TEST
  if ( g_verbose > 5 ) {
    twopoint_function_show_data ( p, stdout );
  }

  return(0);
#else
  fprintf ( stdout, "# [twopoint_function_fill_data] non-aff version not yet implemented\n" );
  return(1);
#endif
}  // end of twopoint_function_fill_data

/********************************************************************************/
/********************************************************************************/
#if ! ( ( defined HAVE_LHPC_AFF ) || ( defined HAVE_HDF5 ) )
/********************************************************************************
 * write diagrams in a twopoint
 *
 * ASCII version
 ********************************************************************************/
int twopoint_function_write_data ( twopoint_function_type *p ) {

  int const nT     = p->T;         // timeslices
  int const ncomp  = p->d * p->d;  // number of components per timeslice of the data set; usually ( spin dimension )^2
  int const nD     = p->n;         // number of data sets, usually number of diagrams

  if ( nT <= 0 || ncomp <= 0 || nD <= 0 ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error, wrong number of timeslices, components or data sets %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( p->c == NULL ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error, no data found %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

  char key[500];
  char key_suffix[400];
  int exitstatus;

  int const Ptot[3] = { p->pf1[0] + p->pf2[0], p->pf1[1] + p->pf2[1], p->pf1[2] + p->pf2[2] };

  char filename[200];

  /******************************************************
   * set FILE pointer
   ******************************************************/
  sprintf ( filename, "%s.%s.PX%dPY%dPZ%d.%s.%.4d.t%dx%dy%dz%d.dat", p->name, p->group, Ptot[0], Ptot[1], Ptot[2], p->irrep, Nconf,
      p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );
  if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_write_data] filename = %s %s %d\n", filename, __FILE__, __LINE__ );

  FILE * ofs = fopen ( filename, "a" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[twopoint_function_write_data] Error from fopen %s %d\n", __FILE__, __LINE__);
    return(4);
  }

  /******************************************************
   * write "last part" of key name into key_suffix;
   * including all momenta and vertex gamma ids
   ******************************************************/
  exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, p );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  /******************************************************
   * loop on diagrams / data sets within 2-point function
   ******************************************************/
  for ( int i = 0; i < nD; i++ ) {

    double _Complex *** const dataset = p->c[i];

    char dataset_tag[500];
    twopoint_function_get_diagram_name ( dataset_tag, p, i );

    sprintf( key, "/%s/%s%s/%s", p->name, p->fbwd, key_suffix, dataset_tag );
    fprintf ( ofs, "# %s\n", key );

    size_t uitems = ncomp * nT;
    for ( int it = 0; it < ncomp * nT; it++ ) {
      fprintf ( ofs, "  %25.16e %25.16e\n", creal ( dataset[0][0][it] ), cimag ( dataset[0][0][it] ) );
    }
  }  // end of loop on diagrams / data sets

  fclose ( ofs );

  return(0);

}  // end of twopoint_function_write_data

/********************************************************************************/
/********************************************************************************/

#elif ( ( defined HAVE_LHPC_AFF )  && ! ( defined HAVE_HDF5 ) )

/********************************************************************************
 * write diagrams in a twopoint
 * AFF version
 ********************************************************************************/
int twopoint_function_write_data ( twopoint_function_type *p ) {

  int const nT     = p->T;         // timeslices
  int const ncomp  = p->d * p->d;  // number of components per timeslice of the data set; usually ( spin dimension )^2
  int const nD     = p->n;         // number of data sets, usually number of diagrams

  if ( nT <= 0 || ncomp <= 0 || nD <= 0 ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error, wrong number of timeslices, components or data sets %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( p->c == NULL ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error, no data found %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

  char key[500];
  char key_suffix[400];
  int exitstatus;

  int const Ptot[3] = { p->pf1[0] + p->pf2[0], p->pf1[1] + p->pf2[1], p->pf1[2] + p->pf2[2] } ;

  char filename[200];

  /******************************************************
   * open AFF writer
   ******************************************************/
  sprintf ( filename, "%s.%s.PX%dPY%dPZ%d.%s.%.4d.t%dx%dy%dz%d.aff", p->name, p->group, Ptot[0], Ptot[1], Ptot[2], p->irrep, Nconf,
      p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );
  if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_write_data] filename = %s %s %d\n", filename, __FILE__, __LINE__ );

  struct AffWriter_s * affw = aff_writer(filename);
  if ( const char * aff_status_str = aff_writer_errstr ( affw ) ) {
    fprintf(stderr, "[twopoint_function_write_data] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    return(4);
  }

  struct AffNode_s * affn = aff_writer_root ( affw );
  if( affn == NULL ) {
    fprintf(stderr, "[twopoint_function_write_data] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
    return(4);
  }

  /******************************************************
   * write "last part" of key name into key_suffix;
   * including all momenta and vertex gamma ids
   ******************************************************/
  exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, p );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  /******************************************************
   * loop on diagrams / data sets within 2-point function
   ******************************************************/
  for ( int i = 0; i < nD; i++ ) {

    double _Complex *** const dataset = p->c[i];

    char dataset_tag[500];
    twopoint_function_get_diagram_name ( dataset_tag, p, i );

    sprintf( key, "/%s/%s%s/%s", p->name, p->fbwd, key_suffix, dataset_tag );
    fprintf ( stdout, "# [twopoint_function_write_data] key = %s\n", key );

    struct AffNode_s * affdir = aff_writer_mkpath ( affw, affn, key );
    if ( affdir == NULL ) {
      fprintf ( stderr, "[twopoint_function_write_data] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__ );
      return(2);
    }

    uint32_t uitems = ncomp * nT;
    exitstatus = aff_node_put_complex ( affw, affdir, dataset[0][0], uitems );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[twopoint_function_write_data] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(105);
    }

  }  // end of loop on diagrams / data sets

  if ( const char * aff_status_str = aff_writer_close ( affw ) ) {
    fprintf(stderr, "[twopoint_function_write_data] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    return(4);
  }

  return(0);

}  // end of twopoint_function_write_data

#elif ( defined HAVE_HDF5 ) 

#define MAX_SUBGROUP_NUMBER 20

/********************************************************************************/
/********************************************************************************/

hid_t twopoint_function_h5_co_group ( hid_t * const grp_list, int * const grp_list_nmem, char * const grp_name, char * const grp_sep, hid_t const loc_id0 ) {

  hid_t lcpl_id       = H5P_DEFAULT;
  hid_t gcpl_id       = H5P_DEFAULT;
  hid_t gapl_id       = H5P_DEFAULT;

  /******************************************************
   * create the target (sub-)group and all
   * groups in hierarchy above if they don't exist
   ******************************************************/
  int nmem = *grp_list_nmem;
  char grp_name_tmp[400];
  char * grp_ptr = NULL;
  hid_t loc_id = ( nmem == 0 ) ? loc_id0 : grp_list[nmem-1]; 

  strcpy ( grp_name_tmp, grp_name );
  if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_h5_co_group] full grp_name_tmp = %s\n", grp_name_tmp );
  grp_ptr = strtok ( grp_name_tmp, grp_sep );

  while ( grp_ptr != NULL ) {
    hid_t grp;
    fprintf ( stdout, "# [twopoint_function_h5_co_group] grp_ptr = %s\n", grp_ptr );

    grp = H5Gopen2( loc_id, grp_ptr, gapl_id );
    if ( grp < 0 ) {
      fprintf ( stderr, "[twopoint_function_h5_co_group] Error from H5Gopen2 for group %s, status was %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
      grp = H5Gcreate2 (       loc_id,         grp_ptr,       lcpl_id,       gcpl_id,       gapl_id );
      if ( grp < 0 ) {
        fprintf ( stderr, "[twopoint_function_h5_co_group] Error from H5Gcreate2 for group %s, status was %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
        return ( -2 );
      } else {
        fprintf ( stdout, "# [twopoint_function_h5_co_group] created group %s %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
      }
    } else {
      fprintf ( stdout, "# [twopoint_function_h5_co_group] opened group %s %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
    }
    grp_ptr = strtok(NULL, grp_sep );

    grp_list[nmem] = grp;
    loc_id = grp;
    nmem++;
    if ( nmem == MAX_SUBGROUP_NUMBER ) {
      fprintf ( stderr, "[twopoint_function_h5_co_group] Error, grp_list_nmem has reached MAX_SUBGROUP_NUMBER %s %d\n", __FILE__, __LINE__ );
      return ( -1 );
    }
  }  /* end of loop on sub-groups */

  /******************************************************
   * update grp_list_nmem
   ******************************************************/
  *grp_list_nmem = nmem;
  if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_h5_co_group] new grp list nmem = %d %s %d\n", *grp_list_nmem, __FILE__, __LINE__ );

  return ( loc_id );

}  /* twopoint_function_h5_co_group */

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 * write diagrams in a twopoint
 * HDF5 version
 ********************************************************************************/
int twopoint_function_write_data ( twopoint_function_type *p ) {

  int const nT     = p->T;         // timeslices
  int const ncomp  = p->d * p->d;  // spin dimension
  int const nD     = p->n;         // number of diagrams / data sets

  if ( p->c == NULL ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  char key_suffix[400];
  int exitstatus;

  int const Ptot[3] = { p->pf1[0] + p->pf2[0], p->pf1[1] + p->pf2[1], p->pf1[2] + p->pf2[2] } ;

  char filename[200];

  /******************************************************
   * HDF5 writer
   ******************************************************/

  hid_t   file_id;
  herr_t  status;

  sprintf ( filename, "%s.%s.PX%dPY%dPZ%d.%s.%.4d.t%dx%dy%dz%d.h5", p->name, p->group, Ptot[0], Ptot[1], Ptot[2], p->irrep, Nconf,
      p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );

  /******************************************************
   * create or open file
   ******************************************************/
  struct stat fileStat;
  if ( stat( filename, &fileStat) < 0 ) {
    /* creat a new file */

    fprintf ( stdout, "# [twopoint_function_write_data] create new file\n" );
 
    unsigned flags = H5F_ACC_TRUNC; /* IN: File access flags. Allowable values are:
                                       H5F_ACC_TRUNC --- Truncate file, if it already exists, erasing all data previously stored in the file. 
                                       H5F_ACC_EXCL  --- Fail if file already exists. 

                                       H5F_ACC_TRUNC and H5F_ACC_EXCL are mutually exclusive; use exactly one.
                                       An additional flag, H5F_ACC_DEBUG, prints debug information. 
                                       This flag can be combined with one of the above values using the bit-wise OR operator (`|'), 
                                       but it is used only by HDF5 Library developers; it is neither tested nor supported for use in applications.  */
    hid_t fcpl_id = H5P_DEFAULT; /* IN: File creation property list identifier, used when modifying default file meta-data. 
                                    Use H5P_DEFAULT to specify default file creation properties. */
    hid_t fapl_id = H5P_DEFAULT; /* IN: File access property list identifier. If parallel file access is desired, 
                                    this is a collective call according to the communicator stored in the fapl_id. 
                                    Use H5P_DEFAULT for default file access properties. */

    //  hid_t H5Fcreate ( const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id )     
    file_id = H5Fcreate (         filename,          flags,       fcpl_id,       fapl_id );

  } else {
    /* open an existing file. */
    fprintf ( stdout, "# [twopoint_function_write_data] open existing file\n" );

    unsigned flags = H5F_ACC_RDWR;  /* IN: File access flags. Allowable values are:
                                       H5F_ACC_RDWR   --- Allow read and write access to file. 
                                       H5F_ACC_RDONLY --- Allow read-only access to file. 

                                       H5F_ACC_RDWR and H5F_ACC_RDONLY are mutually exclusive; use exactly one.
                                       An additional flag, H5F_ACC_DEBUG, prints debug information.
                                       This flag can be combined with one of the above values using the bit-wise OR operator (`|'),
                                       but it is used only by HDF5 Library developers; it is neither tested nor supported for use in applications. */
    hid_t fapl_id = H5P_DEFAULT;
    //  hid_t H5Fopen ( const char *name, unsigned flags, hid_t fapl_id ) 
    file_id = H5Fopen (         filename,         flags,        fapl_id );
  }
  if ( g_verbose > 0 ) fprintf ( stdout, "# [twopoint_function_write_data] file_id = %ld\n", file_id );


  /******************************************************
   * H5 data space and data type
   ******************************************************/

  hid_t dtype_id = H5Tcopy( H5T_NATIVE_DOUBLE );
  status = H5Tset_order ( dtype_id, H5T_ORDER_LE );
  // big_endian() ?  H5T_IEEE_F64BE : H5T_IEEE_F64LE;

  hsize_t dims[1];
  dims[0] = nT * ncomp * 2;  // number of double elements

  //             int rank                             IN: Number of dimensions of dataspace.
  //             const hsize_t * current_dims         IN: Array specifying the size of each dimension.
  //             const hsize_t * maximum_dims         IN: Array specifying the maximum size of each dimension.
  //             hid_t H5Screate_simple( int rank, const hsize_t * current_dims, const hsize_t * maximum_dims ) 
  hid_t space_id = H5Screate_simple(        1,                         dims,                          NULL);

  /******************************************************
   * some default settings for H5Dwrite
   ******************************************************/
  hid_t mem_type_id   = H5T_NATIVE_DOUBLE;
  hid_t mem_space_id  = H5S_ALL;
  hid_t file_space_id = H5S_ALL;
  hid_t xfer_plit_id  = H5P_DEFAULT;
  hid_t lcpl_id       = H5P_DEFAULT;
  hid_t dcpl_id       = H5P_DEFAULT;
  hid_t dapl_id       = H5P_DEFAULT;
  hid_t gcpl_id       = H5P_DEFAULT;
  hid_t gapl_id       = H5P_DEFAULT;
  /* size_t size_hint    = 0; */


  /******************************************************
   * get the suffix for the full group name
   *   made of vertex gamma ids and momenta
   ******************************************************/
  exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, p );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  /******************************************************
   * create the target (sub-)group and all
   * groups in hierarchy above if they don't exist
   ******************************************************/
  hid_t grp_list[MAX_SUBGROUP_NUMBER];
  hid_t loc_id;
  int grp_list_nmem = 0;
  char grp_name[500];
  char grp_sep[] = "/";
  sprintf( grp_name, "/%s/%s%s", p->name, p->fbwd, key_suffix );
  fprintf ( stdout, "# [twopoint_function_write_data] full grp_name = %s\n", grp_name );

  loc_id = twopoint_function_h5_co_group ( grp_list, &grp_list_nmem, grp_name, grp_sep, file_id );
  if ( loc_id < 0 ) {
    fprintf ( stderr, "[twopoint_function_write_data] Error from twopoint_function_h5_co_group, returned loc_id is %d %s %d\n", loc_id, __FILE__, __LINE__ );
    return ( 1 );
  }

  //    hid_t loc_id 	IN: File or group identifier.
  //    const char *name 	IN: Absolute or relative name of the o new group.
  //    size_t size_hint     	IN: Optional parameter indicating the number of bytes to reserve for the names 
  //                              that will appear in the group. A conservative estimate could result in multiple system-level I/O requests to read the group name heap;
  //                              a liberal estimate could result in a single large I/O request even when the group has just a few names. 
  //                              HDF5 stores each name with a null terminator.

  //    hid_t H5Gcreate( hid_t loc_id, const char *name, size_t size_hint )

  //    hid_t loc_id 	IN: File or group identifier
  //    const char *name     	IN: Absolute or relative name of the link to the new group
  //    hid_t lcpl_id 	IN: Link creation property list identifier
  //    hid_t gcpl_id 	IN: Group creation property list identifier
  //    hid_t gapl_id 	IN: Group access property list identifier
  //                        (No group access properties have been implemented at this time; use H5P_DEFAULT.) 

  //    hid_t H5Gcreate2 (  hid_t loc_id, const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id )


  /******************************************************
   * loop on data sets within 2-point function
   ******************************************************/
  for ( int i = 0; i < nD; i++ ) 
  {

    /******************************************************
     * construct the full key
     ******************************************************/
    char dataset_name[500];
    twopoint_function_get_diagram_name ( dataset_name, p, i );
    if ( strcmp( dataset_name, "NA" ) == 0 ) {
      fprintf ( stderr, "[twopoint_function_write_data] Warning, could not find data set number %d %s %d\n", i, __FILE__, __LINE__ );
      continue;
    } else {
      if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_write_data] dataset name = %s %s %d\n", dataset_name, __FILE__, __LINE__ );
    }

    /******************************************************
     * create/open data set group
     ******************************************************/
    hid_t dataset_grp_list[MAX_SUBGROUP_NUMBER];
    int dataset_grp_nmem = 0;
    loc_id = ( grp_list_nmem == 0 ) ? file_id : grp_list[grp_list_nmem - 1 ];

    loc_id = twopoint_function_h5_co_group ( dataset_grp_list, &dataset_grp_nmem, dataset_name, grp_sep, loc_id );
    if ( loc_id < 0 ) {
      fprintf ( stderr, "[twopoint_function_write_data] Error from twopoint_function_h5_co_group, returned loc_id is %d %s %d\n", loc_id, __FILE__, __LINE__ );
      return ( 1 );
    }

    /* pointer to data */
    double _Complex *** const dataset = p->c[i];
    if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_write_data] dataset %2d loc_id = %ld %s %d\n", i, loc_id , __FILE__, __LINE__ );

    /******************************************************
     * create a data set
     ******************************************************/

    //             hid_t loc_id 	IN: Location identifier
    //             const char *name    	IN: Dataset name
    //             hid_t dtype_id 	IN: Datatype identifier
    //             hid_t space_id 	IN: Dataspace identifier
    //             hid_t lcpl_id 	IN: Link creation property list
    //             hid_t dcpl_id 	IN: Dataset creation property list
    //             hid_t dapl_id 	IN: Dataset access property list
    //             hid_t H5Dcreate2 ( hid_t loc_id, const char *name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id )
    // hid_t dataset_id   = H5Dcreate2 ( file_id,      key,                    dtype_id,       space_id,   H5P_DEFAULT,   H5P_DEFAULT,   H5P_DEFAULT   );

    //           hid_t H5Dcreate ( hid_t loc_id, const char *name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id ) 
    hid_t dataset_id = H5Dcreate (       loc_id,           "data",       dtype_id,       space_id,       lcpl_id,       dcpl_id,       dapl_id );


    /******************************************************
     * write the current data set
     ******************************************************/

    //       hid_t dataset_id           IN: Identifier of the dataset to write to.
    //       hid_t mem_type_id          IN: Identifier of the memory datatype.
    //       hid_t mem_space_id         IN: Identifier of the memory dataspace.
    //       hid_t file_space_id        IN: Identifier of the dataset's dataspace in the file.
    //       hid_t xfer_plist_id        IN: Identifier of a transfer property list for this I/O operation.
    //       const void * buf           IN: Buffer with data to be written to the file.
    //herr_t H5Dwrite ( hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id, const void * buf )
    status = H5Dwrite (       dataset_id,       mem_type_id,       mem_space_id,       file_space_id,        xfer_plit_id,    dataset[0][0] );

    if( status < 0 ) {
      fprintf(stderr, "[twopoint_function_write_data] Error from H5Dwrite, status was %d %s %d\n", status, __FILE__, __LINE__);
      return(105);
    }

    /******************************************************
     * close the current data set
     ******************************************************/
    status = H5Dclose ( dataset_id );
    if( status < 0 ) {
      fprintf(stderr, "[twopoint_function_write_data] Error from H5Dclose, status was %d %s %d\n", status, __FILE__, __LINE__);
      return(105);
    }

    /******************************************************
     * close all (sub-)groups in reverse order
     ******************************************************/
    for ( int ig = dataset_grp_nmem - 1; ig >= 0; ig-- ) {
      status = H5Gclose ( dataset_grp_list[ig] );
      if( status < 0 ) {
        fprintf(stderr, "[twopoint_function_write_data] Error from H5Gclose, status was %d %s %d\n", status, __FILE__, __LINE__);
        return(105);
      } else {
        fprintf(stdout, "# [twopoint_function_write_data] closed group %ld %s %d\n", grp_list[ig], __FILE__, __LINE__);
      }
    }
  }  // end of loop on diagrams / data sets

  /******************************************************
   * close the data space
   ******************************************************/
  status = H5Sclose ( space_id );
  if( status < 0 ) {
    fprintf(stderr, "[twopoint_function_write_data] Error from H5Sclose, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(105);
  }

  /******************************************************
   * close all (sub-)groups in reverse order
   ******************************************************/
  for ( int i = grp_list_nmem - 1; i>= 0; i-- ) {
    status = H5Gclose ( grp_list[i] );
    if( status < 0 ) {
      fprintf(stderr, "[twopoint_function_write_data] Error from H5Gclose, status was %d %s %d\n", status, __FILE__, __LINE__);
      return(105);
    } else {
      fprintf(stdout, "# [twopoint_function_write_data] closed group %ld %s %d\n", grp_list[i], __FILE__, __LINE__);
    }
  }

  /******************************************************
   * close the data type
   ******************************************************/
  status = H5Tclose ( dtype_id );
  if( status < 0 ) {
    fprintf(stderr, "[twopoint_function_write_data] Error from H5Tclose, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(105);
  }

  /******************************************************
   * close the file
   ******************************************************/
  status = H5Fclose ( file_id );

  if( status < 0 ) {
    fprintf(stderr, "[twopoint_function_write_data] Error from H5Fclose, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(105);
  }

  return(0);

}  // end of twopoint_function_write_data

#endif


/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 * read diagrams for a twopoint
 ********************************************************************************/
int twopoint_function_data_location_identifier ( char * udli, twopoint_function_type *p, char * const datafile_prefix, int const ids , char * const sep) {

  char key[500];
  char key_suffix[400];
  char diagram_tag[500] = "NA";
  char filename[200];
  int exitstatus;

  /******************************************************
   * total momentum
   ******************************************************/
  int const Ptot[3] = { p->pf1[0] + p->pf2[0], p->pf1[1] + p->pf2[1], p->pf1[2] + p->pf2[2] } ;

  /******************************************************
   * reference frame momentum for current total momentum
   ******************************************************/
  int Pref[3];
  exitstatus = get_reference_rotation ( Pref, NULL, Ptot );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopoint_function_data_location_identifier] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  } 

  /* write "last part" of key name into key_suffix;
   * including all momenta and vertex gamma ids */
  exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, p );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopoint_function_data_location_identifier] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  /******************************************************
   * diagram tag
   ******************************************************/
  twopoint_function_get_diagram_name ( diagram_tag, p, ids );

  /******************************************************
   * data filename
   ******************************************************/
  sprintf(filename, "%s.%c.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", datafile_prefix, diagram_tag[0],
      Pref[0], Pref[1], Pref[2], Nconf,
      p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );
 
  /******************************************************
   * key name
   ******************************************************/
  sprintf( key, "/%s/%s/%s/%s", p->name, diagram_tag, p->fbwd, key_suffix );
  if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_data_location_identifier] key = %s %s %d\n", key, __FILE__, __LINE__ );

  /******************************************************
   * udli
   ******************************************************/
  sprintf ( udli, "%s%s%s", filename, sep, key );

  if ( g_verbose > 4 ) fprintf ( stdout, "# [twopoint_function_data_location_identifier] udli = %s\n", udli );

  return(0);
}  // end of twopoint_function_data_location_identifier

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 * read diagrams for a twopoint
 ********************************************************************************/
int twopoint_function_fill_data_from_udli ( twopoint_function_type *p, char * udli, int const io_proc ) {

#ifdef HAVE_LHPC_AFF
  int const nT = p->T;  // timeslices
  int const d  = p->d;  // spin dimension
  int const nD = p->n;  // number of diagrams / data sets
  char sep[] = "#";
  /* fix: does this need to be generalized ? */
  /* char const filename_prefix[] = "piN_piN_diagrams"; */

  static struct AffReader_s *affr = NULL;
  static struct AffNode_s *affn = NULL;
  struct AffNode_s *affdir = NULL;

  char key[500];
  static char filename[500] = "NA";
  char filename_tmp[500];
  int exitstatus;


  /******************************************************
   * check for NULL p or udli
   * if so, just close the AFF reader and return
   ******************************************************/
  if ( p == NULL || udli == NULL ) {
    if ( affr != NULL ) aff_reader_close ( affr );
    affr = NULL;
    return( 0 );
  }

  /******************************************************
   * check for non-NULL data array p->c
   * if NULL, allocate it according to data size
   * given in p
   ******************************************************/
  if ( p->c == NULL ) {
    if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_fill_data_from_udli] Warning, data array not intialized; allocing p->c %s %d\n", __FILE__, __LINE__ );
 
    if ( twopoint_function_allocate ( p ) == NULL ) {
      fprintf ( stderr, "[twopoint_function_fill_data_from_udli] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
      return(1);
    }
  }

  /******************************************************
   * only one data set is read at a time
   ******************************************************/
  if ( nD != 1 ) {
    fprintf ( stderr, "[twopoint_function_fill_data_from_udli] Error, read only one data set %s %d\n", __FILE__, __LINE__ );
    return(3);
  }

  if ( io_proc == 2 ) {

    double ratime = _GET_TIME;
    char name[500];
    char *ptr = NULL;

    /******************************************************
     * get filename and AFF key from udli
     ******************************************************/
    if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_fill_data_from_udli] udli = %s\n", udli );

    strcpy ( name, udli );

    ptr = strtok( name, sep );
    if ( ptr == NULL ) { 
      fprintf ( stderr, "[twopoint_function_fill_data_from_udli] Error, cannot get filename from udli %s %d\n", __FILE__, __LINE__ );
      return(5); 
    } else {
      strcpy ( filename_tmp, ptr );
      ptr = strtok ( NULL, sep );
      if ( ptr == NULL )  { 
        fprintf ( stderr, "[twopoint_function_fill_data_from_udli] Error, cannot get key from udli %s %d\n", __FILE__, __LINE__ );
        return(6); 
      } else {
        strcpy ( key, ptr );
      }
    }

    if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_fill_data_from_udli] file = %s, key = %s\n", filename_tmp, key );


    /******************************************************
     * check for new filename
     * if new filename, open new reader
     * also if AFF reader is still NULL,
     * initialize a new reader
     ******************************************************/
    if ( ( strcmp( filename, filename_tmp ) != 0 ) || ( affr == NULL ) ) {
      strcpy ( filename, filename_tmp );
      if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_fill_data_from_udli] new reader for file %s and reader\n", filename );

      if ( affr != NULL ) {
        aff_reader_close (affr);
        affr = NULL;
      }

      /******************************************************
       * AFF reader
       ******************************************************/
      affr = aff_reader  (filename);
      if ( const char * aff_status_str =  aff_reader_errstr(affr) ) {
        fprintf(stderr, "[twopoint_function_fill_data_from_udli] Error from aff_reader for filename %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        return(4);
      }

      /******************************************************
       * AFF root node
       ******************************************************/
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[twopoint_function_fill_data_from_udli] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

    } else {
      if ( g_verbose > 2 ) fprintf ( stdout, "# [twopoint_function_fill_data_from_udli] keeping previous filename %s and reader for udli %s\n", filename, udli );
    }

    /******************************************************
     * read data set
     ******************************************************/
    if ( ( affdir = aff_reader_chpath (affr, affn, key ) ) == NULL ) {
      fprintf(stderr, "[twopoint_function_fill_data_from_udli] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
      return(106);
    }
    uint32_t uitems = d *d * nT;
    exitstatus = aff_node_get_complex (affr, affdir, p->c[0][0][0], uitems );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[twopoint_function_fill_data_from_udli] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(105);
    }

    double retime = _GET_TIME;
    if ( g_verbose > 0 ) fprintf ( stdout, "# [twopoint_function_fill_data_from_udli] time for reading AFF data set = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__ );

    // TEST
    if ( g_verbose > 5 ) {
      twopoint_function_show_data ( p, stdout );
    }
#if 0
#endif  /* of if 0 */
  }  /* end of if io_proc == 2 */

  return(0);
#else
  fprintf ( stdout, "# [twopoint_function_fill_data_from_udli] non-aff version not yet implemented\n" );
  return(1);
#endif
}  // end of twopoint_function_fill_data_from_udli

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 * normalize diagrams
 ********************************************************************************/
int twopoint_function_apply_diagram_norm ( twopoint_function_type *p ) {

  int const nT = p->T;  // timeslices
  int const d  = p->d;  // spin dimension
  int const nD = p->n;  // number of diagrams / data sets

  double ratime = _GET_TIME;

  /******************************************************
   * for now, check, that d is 4
   * this should be generalized
   ******************************************************/
  if ( d != 4 ) {
    fprintf ( stderr, "[twopoint_function_apply_diagram_norm] Error, spinor dimension must be 4 %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  if ( p->c == NULL ) {
    fprintf ( stderr, "[twopoint_function_apply_diagram_norm] Error, data array not initialized %s %d\n", __FILE__, __LINE__ );
    return(2);
  }

  if ( strcmp ( p->norm , "NA" ) == 0 ) {
    if( g_verbose > 4 ) fprintf ( stderr, "[twopoint_function_apply_diagram_norm] Warning, norm is not set %s %d\n", __FILE__, __LINE__ );
    return(0);
  }

  for ( int id = 0; id < nD; id++ ) {

    double _Complex *** const diagram = p->c[id];

    double const norm = twopoint_function_get_diagram_norm ( p, id );
    if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_apply_diagram_norm] using diagram norm %2d %16.7e %s %d\n", id, norm, __FILE__, __LINE__ );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for ( int t = 0; t < nT; t++ ) {
      zm4x4_ti_eq_re ( diagram[t], norm );
    }

  }  // end of loop on data sets

  double retime = _GET_TIME;
  if ( g_verbose > 0 ) fprintf ( stdout, "# [twopoint_function_apply_diagram_norm] time for apply norm = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__ );
  return ( 0 );
}  /* end of twopoint_function_apply_diagram_norm */

/********************************************************************************/
/********************************************************************************/

/********************************************************************************
 * accumulate given p->cs in z
 ********************************************************************************/
int twopoint_function_accum_diagrams ( double _Complex *** const z, twopoint_function_type *p ) {

  if ( z == NULL || p->c == NULL || p->n <= 0 ) {
    fprintf ( stderr, "[twopoint_function_accum_diagrams] Error, nothing to copy from / to %s %d\n", __FILE__, __LINE__ );
    return ( 1 );
  }

  int const nD = p->n;

  /******************************************************
   * IF z is NOT p->c[0], then
   * put data set p->c[0] into z
   ******************************************************/
  if ( p->c[0] != z ) {
    memcpy ( z[0][0], p->c[0][0][0], p->T * p->d * p->d * sizeof(double _Complex ) );
  }

  /******************************************************
   * add data sets 1,...,p->n-1 to data set 0,
   * which is already in z
   ******************************************************/
  for ( int i = 1; i < nD; i++ ) {
    contract_diagram_zm4x4_field_pl_eq_zm4x4_field ( z, p->c[i], p->T );
  }

  return( 0 );
}  /* end of twopoint_function_accum_diagrams */

/********************************************************************************/
/********************************************************************************/


#ifdef HAVE_HDF5

/********************************************************************************
 * read projected correlators from HDF5 file
 ********************************************************************************/
int twopoint_function_correlator_from_h5_file ( twopoint_function_type * const p, int const io_proc ) {

  int exitstatus;
  char key_suffix[500];

  /******************************************************
   * io_proc 0 and 1 return
   ******************************************************/
  if ( io_proc != 2 ) return ( 0 );

  if( p->c == NULL ) {
    fprintf ( stdout, "# [twopoint_function_correlator_from_h5_file] Warning, allocate data array %s %d\n", __FILE__, __LINE__ );

    if ( twopoint_function_allocate ( p ) == NULL ) {
      fprintf ( stderr, "[twopoint_function_correlator_from_h5_file] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
      return ( 1 );
    }
  }

  /******************************************************
   * get the suffix for the full group name
   *   made of vertex gamma ids and momenta
   ******************************************************/
  exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, p );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[twopoint_function_correlator_from_h5_file] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  int const Ptot[3] = { p->pf1[0] + p->pf2[0], p->pf1[1] + p->pf2[1], p->pf1[2] + p->pf2[2] } ;

  char filename[500];

   sprintf ( filename, "%s.%s.PX%dPY%dPZ%d.%s.%.4d.t%dx%dy%dz%d.h5", p->name, p->group, Ptot[0], Ptot[1], Ptot[2], p->irrep, Nconf,
       p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );

  double * zbuffer = NULL;

  /***************************************************************************
   * time measurement
   ***************************************************************************/
  struct timeval ta, tb;
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***************************************************************************
   * open file
   ***************************************************************************/

  hid_t   file_id = -1;
  herr_t  status;

  struct stat fileStat;
  if ( stat( filename, &fileStat) < 0 ) {
    fprintf ( stderr, "[twopoint_function_correlator_from_h5_file] Error, file %s does not exist %s %d\n", filename, __FILE__, __LINE__ );
    return ( 1 );
  } else {
    /* open an existing file. */
    if ( g_verbose > 1 ) fprintf ( stdout, "# [twopoint_function_correlator_from_h5_file] open existing file\n" );
  
    unsigned flags = H5F_ACC_RDONLY;  /* IN: File access flags. Allowable values are:
                                         H5F_ACC_RDWR   --- Allow read and write access to file.
                                         H5F_ACC_RDONLY --- Allow read-only access to file.
  
                                         H5F_ACC_RDWR and H5F_ACC_RDONLY are mutually exclusive; use exactly one.
                                         An additional flag, H5F_ACC_DEBUG, prints debug information.
                                         This flag can be combined with one of the above values using the bit-wise OR operator (`|'),
                                         but it is used only by HDF5 Library developers; it is neither tested nor supported for use in applications. */
    hid_t fapl_id = H5P_DEFAULT;
    /*  hid_t H5Fopen ( const char *name, unsigned flags, hid_t fapl_id ) */
    file_id = H5Fopen (         filename,         flags,        fapl_id );

    if ( file_id < 0 ) {
      fprintf ( stderr, "[twopoint_function_correlator_from_h5_file] Error from H5Fopen %s %d\n", __FILE__, __LINE__ );
      return ( 2 );
    }
  }  /* end of if file exists */
  
  if ( g_verbose > 1 ) fprintf ( stdout, "# [twopoint_function_correlator_from_h5_file] file_id = %ld\n", file_id );
  
  /***************************************************************************
   * some default settings for H5Dopen2, H5Dread
   ***************************************************************************/
  hid_t dapl_id       = H5P_DEFAULT;

  hid_t mem_type_id   = H5T_NATIVE_DOUBLE;
  hid_t mem_space_id  = H5S_ALL;
  hid_t file_space_id = H5S_ALL;
  hid_t xfer_plist_id = H5P_DEFAULT;

  /***************************************************************************
   * loop on data sets / diagrams
   ***************************************************************************/
  for ( int ids = 0; ids < p->n; ids++ ) {

    char diagram_name[500];
    twopoint_function_get_diagram_name ( diagram_name, p, ids );

    char tag[500];
    sprintf( tag, "/%s/%s%s/%s/data", p->name, p->fbwd, key_suffix, diagram_name );

    /***************************************************************************
     * open H5 data set
     ***************************************************************************/
    hid_t dataset_id = H5Dopen2 ( file_id, tag, dapl_id );
    if ( dataset_id < 0 ) {
      fprintf ( stderr, "[twopoint_function_correlator_from_h5_file] Error from H5Dopen2 for tag %s %s %d\n", tag, __FILE__, __LINE__ );
      return ( 3 );
    }

    /***************************************************************************
     * read data set
     ***************************************************************************/
    status = H5Dread ( dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, (void*)(p->c[ids][0][0]) );
    if ( status < 0 ) {
      fprintf ( stderr, "[twopoint_function_correlator_from_h5_file] Error from H5Dread %s %d\n", __FILE__, __LINE__ );
      return ( 4 );
    }

    /***************************************************************************
     * close data set
     ***************************************************************************/
    status = H5Dclose ( dataset_id );
    if ( status < 0 ) {
      fprintf ( stderr, "[twopoint_function_correlator_from_h5_file] Error from H5Dclose %s %d\n", __FILE__, __LINE__ );
      return ( 5 );
    }

  }  /* end of loop on data sets / diagrams */

  /***************************************************************************
   * close the file
   ***************************************************************************/
  status = H5Fclose ( file_id );
  if( status < 0 ) {
    fprintf(stderr, "[twopoint_function_correlator_from_h5_file] Error from H5Fclose, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(6);
  } 

  /***************************************************************************
   * time measurement
   ***************************************************************************/
  gettimeofday ( &tb, (struct timezone *)NULL );
  
  show_time ( &ta, &tb, "twopoint_function_correlator_from_h5_file", "write h5", 1 );

  return(0);

}  /* end of twopoint_function_correlator_from_h5_file */

#endif  /* of if HAVE_HDF5 */

/***************************************************************************/
/***************************************************************************/

/***************************************************************************
 *
 ***************************************************************************/
void twopoint_function_check_reference_rotation ( twopoint_function_type * const tp, little_group_projector_type * const pr, double const deps ) {

  unsigned int const dim       = pr->rspin[0].dim;
  unsigned int const irrep_dim = pr->rtarget->dim;
  unsigned int const nrot      = pr->rtarget->n;
  unsigned int const nT        = tp[0].T;
  unsigned int const index_conv_dim[4] = { irrep_dim, irrep_dim, irrep_dim, irrep_dim };

  char name[400];

  /******************************************************
   * loops on irrep row index at snk and src
   ******************************************************/
  for ( int row_snk = 0; row_snk < irrep_dim; row_snk++ ) {
  for ( int row_src = 0; row_src < irrep_dim; row_src++ ) {

    /******************************************************
     * fixed reference index at sink
     ******************************************************/
    for ( int iref1 = 0; iref1 < irrep_dim; iref1++ ) {
 
      for ( int irot = 0; irot < 2*nrot; irot++ )
      {
        double _Complex ** Rspin  = ( irot < nrot ) ? pr->rspin[0].R[irot] :  pr->rspin[0].IR[irot-nrot];
        double _Complex ** Tirrep = ( irot < nrot ) ? pr->rtarget->R[irot] :  pr->rtarget->IR[irot-nrot];

        fprintf ( stdout, "# [twopoint_function_check_reference_rotation] \n# [test_ref_rotation] \n" );

        /******************************************************
         * show rotation
         ******************************************************/
        if ( g_verbose > 2 ) {
          sprintf ( name, "R%.2d", irot );
          rot_printf_matrix ( Rspin, dim, name, stdout );
          fprintf ( stdout, "# [twopoint_function_check_reference_rotation] \n" );
        }

        /******************************************************
         * loop on timeslices
         ******************************************************/
        for ( int it = 0; it < nT; it ++ ) {

          double _Complex *** R1  = init_3level_ztable ( irrep_dim, dim, dim ) ;
          double _Complex *** R2  = init_3level_ztable ( irrep_dim, dim, dim ) ;
          double _Complex ** Raux = init_2level_ztable ( dim, dim ) ;

          /******************************************************
           * R1_iref2 = tp_{iref1,iref2} x Rspin
           ******************************************************/
          for ( int iref2 = 0; iref2 < irrep_dim; iref2++ ) {
            int const coords[] = { iref1, iref2, row_snk, row_src };
            unsigned int idx = coords_to_index (  coords, index_conv_dim, 4 );
            rot_mat_ti_mat ( R1[iref2], tp[ idx ].c[0][it], Rspin, dim );
          }

          /******************************************************
           * R2_iref2 = T_{iref2,i} tp_{i,iref1,...}
           ******************************************************/
          for ( int iref2 = 0; iref2 < irrep_dim; iref2++ ) {
            for ( int i = 0; i < irrep_dim; i++ ) {
              int const coords[] = { iref1, i, row_snk, row_src };
              unsigned int idx = coords_to_index (  coords, index_conv_dim, 4 );
              /* R2 += Tirrep Cproj */
              rot_mat_pl_eq_mat_ti_co ( R2[iref2], tp[ idx ].c[0][it], Tirrep[iref2][i], dim );
            }
          }

          /******************************************************
           * check diff and norm
           ******************************************************/
          for ( int iref2 = 0; iref2 < irrep_dim; iref2++ ) {

            sprintf ( name, "Rsnk_r%.2d_t%.2d_ref%d_%d_%d_%d", irot, it, row_snk, row_src, iref1, iref2 );
            if ( g_verbose > 2 ) {
              rot_printf_matrix_comp ( R1[iref2], R2[iref2], dim, name, stdout );
            }

            double const diff = rot_mat_norm_diff ( R1[iref2], R2[iref2], dim );
            double const norm = sqrt ( rot_mat_norm2 ( R1[iref2], dim ) );

            if ( ( norm > deps && diff/norm < deps ) || ( norm <= deps && diff <= deps ) ) {
              fprintf ( stdout, "# [twopoint_function_check_reference_rotation] src %s diff norm %e    %e  okay\n\n", name, diff, norm );
            } else {
              fprintf ( stdout, "# [twopoint_function_check_reference_rotation] src %s diff norm %e    %e  ERROR\n\n", name, diff, norm );
            }
          }

          fini_3level_ztable ( &R1 );
          fini_3level_ztable ( &R2 );
          fini_2level_ztable ( &Raux );

        }  /* end of loop on time slices */

      }  /* end of loop on rotations */

    }  /* end of loop on iref1 */

    /******************************************************
     * fixed reference index at src
     ******************************************************/

    for ( int iref2 = 0; iref2 < irrep_dim; iref2++ ) {
 
      for ( int irot = 0; irot < 2*nrot; irot++ )
      {

        double _Complex ** Rspin  = ( irot < nrot ) ? pr->rspin[0].R[irot] :  pr->rspin[0].IR[irot-nrot];
        double _Complex ** Tirrep = ( irot < nrot ) ? pr->rtarget->R[irot] :  pr->rtarget->IR[irot-nrot];

        fprintf ( stdout, "# [twopoint_function_check_reference_rotation] \n# [twopoint_function_check_reference_rotation] \n" );

        /******************************************************
         * show rotation matrix
         ******************************************************/
        if ( g_verbose > 2 ) {
          sprintf ( name, "R%.2d", irot );
          rot_printf_matrix ( Rspin, dim, name, stdout );
          fprintf ( stdout, "# [twopoint_function_check_reference_rotation] \n" );
        }

        /******************************************************
         * loop on timeslices
         ******************************************************/
        for ( int it = 0; it < nT; it ++ ) {

          double _Complex *** R1 = init_3level_ztable ( irrep_dim, dim, dim ) ;
          double _Complex *** R2 = init_3level_ztable ( irrep_dim, dim, dim ) ;
          double _Complex ** Raux = init_2level_ztable ( dim, dim ) ;

          /******************************************************
           * R1_{iref1} = Rspin^+ tp_{iref1,iref2,...}
           ******************************************************/
          for ( int iref1 = 0; iref1 < irrep_dim; iref1++ ) {
            int const coords[] = { iref1, iref2, row_snk, row_src };
            unsigned int const idx = coords_to_index ( coords, index_conv_dim, 4 );
            rot_mat_adj_ti_mat ( R1[iref1], Rspin, tp[ idx ].c[0][it], dim );
          }

          /******************************************************
           * R2_{iref1} = T_{iref1,i}^*  tp_{i,iref2,...}
           ******************************************************/
          for ( int iref1 = 0; iref1 < irrep_dim; iref1++ ) {
            for ( int i = 0; i < irrep_dim; i++ ) {

              int const coords[] = { i, iref2, row_snk, row_src };
              unsigned int const idx = coords_to_index ( coords, index_conv_dim, 4 );
              /* R2 += Tirrep Cproj */
              rot_mat_pl_eq_mat_ti_co ( R2[iref1], tp[ idx ].c[0][it], conj( Tirrep[iref1][i] ), dim );
            }
          }

          /******************************************************
           * check diff and norm
           ******************************************************/
          for ( int iref1 = 0; iref1 < irrep_dim; iref1++ ) {

            sprintf ( name, "Rsrc_r%.2d_t%.2d_ref%d_%d_%d_%d", irot, it, row_snk, row_src, iref1, iref2 );
            if ( g_verbose > 2 ) {
              rot_printf_matrix_comp ( R1[iref1], R2[iref1], dim, name, stdout );
            }

            double const diff = rot_mat_norm_diff ( R1[iref1], R2[iref1], dim );
            double const norm = sqrt ( rot_mat_norm2 ( R1[iref1], dim ) );

            if ( ( norm > deps && diff/norm < deps ) || ( norm <= deps && diff <= deps ) ) {
              fprintf ( stdout, "# [twopoint_function_check_reference_rotation] snk %s diff norm %e    %e  okay\n\n", name, diff, norm );
            } else {
              fprintf ( stdout, "# [twopoint_function_check_reference_rotation] snk %s diff norm %e    %e  ERROR\n\n", name, diff, norm );
            }
          }

          fini_3level_ztable ( &R1 );
          fini_3level_ztable ( &R2 );
          fini_2level_ztable ( &Raux );

        }  /* end of loop on time slices */
      }  /* end of loop on rotations */

    }  /* end of loop on iref2 */

  }}  /* end of loops on row_src, row_snk */

}  /* end of twopoint_function_check_reference_rotation */

/***************************************************************************/
/***************************************************************************/

/***************************************************************************
 *
 ***************************************************************************/
void twopoint_function_check_reference_rotation_vector_spinor ( twopoint_function_type * const tp, little_group_projector_type * const pr, double const deps ) {

  unsigned int const spinor_dim        = pr->rspin[0].dim;
  unsigned int const vector_dim        = pr->rspin[1].dim;
  unsigned int const irrep_dim         = pr->rtarget->dim;
  unsigned int const nrot              = pr->rtarget->n;
  unsigned int const nT                = tp[0].T;
  unsigned int const coords_dim        = 6;
  unsigned int const index_conv_dim[6] = { irrep_dim, irrep_dim, irrep_dim, irrep_dim, vector_dim, vector_dim };

  char name[400];

  for ( int row_snk = 0; row_snk < irrep_dim; row_snk++ )
  {
  for ( int row_src = 0; row_src < irrep_dim; row_src++ )
  {

    /******************************************************
     * fixed reference index at sink
     ******************************************************/

    for ( int iref1 = 0; iref1 < irrep_dim; iref1++ )
    {
 
      for ( int irot = 0; irot < 2*nrot; irot++ )
      {
        double _Complex ** Rspin  = ( irot < nrot ) ? pr->rspin[0].R[irot] :  pr->rspin[0].IR[irot-nrot];
        double _Complex ** Rvec   = ( irot < nrot ) ? pr->rspin[1].R[irot] :  pr->rspin[1].IR[irot-nrot];
        double _Complex ** Tirrep = ( irot < nrot ) ? pr->rtarget->R[irot] :  pr->rtarget->IR[irot-nrot];

        fprintf ( stdout, "# [twopoint_function_check_reference_rotation_vector_spinor] \n# [test_ref_rotation] \n" );

        /******************************************************
         * show the spinor rotation matrix
         ******************************************************/
        if ( g_verbose > 2 ) {
          sprintf ( name, "Rspin%.2d", irot );
          rot_printf_matrix ( Rspin, spinor_dim, name, stdout );
          fprintf ( stdout, "# [twopoint_function_check_reference_rotation_vector_spinor] \n\n" );

          sprintf ( name, "Rvec%.2d", irot );
          rot_printf_matrix ( Rvec, vector_dim, name, stdout );
          fprintf ( stdout, "# [twopoint_function_check_reference_rotation_vector_spinor] \n\n" );
        }

        /******************************************************
         * loop on timeslices
         ******************************************************/
        for ( int it = 0; it < nT; it++ )
        {

          double _Complex ***** R1  = init_5level_ztable ( irrep_dim, vector_dim, vector_dim, spinor_dim, spinor_dim );
          double _Complex ***** R2  = init_5level_ztable ( irrep_dim, vector_dim, vector_dim, spinor_dim, spinor_dim );
          double _Complex ** Raux = init_2level_ztable ( spinor_dim, spinor_dim ) ;

          /******************************************************
           * R1_{iref2; vi, vk} =
           *
           *   tp_{iref1,iref2,row_snk,row_src; vi,vj} Rvec_{vj,vk} Rspin
           ******************************************************/
          for ( int iref2 = 0; iref2 < irrep_dim; iref2++ ) {

            for ( int vi = 0; vi < vector_dim; vi++ ) {
            for ( int vk = 0; vk < vector_dim; vk++ ) {

              rot_mat_zero ( R1[iref2][vi][vk], spinor_dim );
   
              for ( int vj = 0; vj < vector_dim; vj++ ) {

                int const coords[] = { iref1, iref2, row_snk, row_src, vi, vj };
                unsigned int idx = coords_to_index (  coords, index_conv_dim, coords_dim );

                rot_mat_ti_mat ( Raux, tp[ idx ].c[0][it], Rspin, spinor_dim );

                double _Complex zcoeff = Rvec[vj][vk] * ( ( irot < nrot ) ? 1. : ( pr->parity[0] * pr->parity[1] ) );

                rot_mat_pl_eq_mat_ti_co ( R1[iref2][vi][vk], Raux, zcoeff, spinor_dim );

              }  /* end of loop on left vector rotation index vj */

            }}  /* end of loop on left and right vector reference index vi, vk */

          }  /* end of loop on right irrep reference index iref2 */

          /******************************************************
           * irrep rotation of same tp [ idx ] as above
           ******************************************************/
          for ( int iref2 = 0; iref2 < irrep_dim; iref2++ ) {

            for ( int vi = 0; vi < vector_dim; vi++ ) {
            for ( int vk = 0; vk < vector_dim; vk++ ) {

              rot_mat_zero ( R2[iref2][vi][vk] , spinor_dim );

              for ( int i = 0; i < irrep_dim; i++ ) {

                int const coords[] = { iref1, i, row_snk, row_src, vi, vk };
                unsigned int idx = coords_to_index (  coords, index_conv_dim, coords_dim );

                /* R2 += Tirrep Cproj */
                rot_mat_pl_eq_mat_ti_co ( R2[iref2][vi][vk], tp[ idx ].c[0][it], Tirrep[iref2][i], spinor_dim );

              }  /* end of loop on irrep rotation index i */

            }}  /* end of loop on left and right vector reference index vi, vk */

          }  /* end of right irrep reference index iref2 */

          /******************************************************
           * check diff and norm
           ******************************************************/
          for ( int iref2 = 0; iref2 < irrep_dim; iref2++ ) {

            for ( int vi = 0; vi < vector_dim; vi++ ) {
            for ( int vk = 0; vk < vector_dim; vk++ ) {

              sprintf ( name, "Rsnk_r%.2d_t%.2d_gref%d_%d_%d_%d_vref%d_%d", irot, it, row_snk, row_src, iref1, iref2, vi, vk );
              if ( g_verbose > 2 ) {
                rot_printf_matrix_comp ( R1[iref2][vi][vk], R2[iref2][vi][vk], spinor_dim, name, stdout );
              }

              double const diff = rot_mat_norm_diff ( R1[iref2][vi][vk], R2[iref2][vi][vk], spinor_dim );
              double const norm = sqrt ( rot_mat_norm2 ( R1[iref2][vi][vk], spinor_dim ) );

              if ( ( norm > deps && diff/norm < deps ) || ( norm <= deps && diff <= deps ) ) {
                fprintf ( stdout, "# [twopoint_function_check_reference_rotation] src %s diff norm %e    %e  okay\n\n", name, diff, norm );
              } else {
                fprintf ( stdout, "# [twopoint_function_check_reference_rotation] src %s diff norm %e    %e  ERROR\n\n", name, diff, norm );
              }

            }}  /* end of loop on left and right vector reference index vi, vk */

          }  /* end of loop on right irrep reference index iref2 */

          /* deallocate */
          fini_5level_ztable ( &R1 );
          fini_5level_ztable ( &R2 );
          fini_2level_ztable ( &Raux );

        }  /* end of loop on time slices */

      }  /* end of loop on rotations */

    }  /* end of loop on iref1 */


    /******************************************************
     * fixed reference index at src
     ******************************************************/

    for ( int iref2 = 0; iref2 < irrep_dim; iref2++ ) {
 
      for ( int irot = 0; irot < 2*nrot; irot++ )
      {

        double _Complex ** Rspin  = ( irot < nrot ) ? pr->rspin[0].R[irot] :  pr->rspin[0].IR[irot-nrot];
        double _Complex ** Rvec   = ( irot < nrot ) ? pr->rspin[1].R[irot] :  pr->rspin[1].IR[irot-nrot];
        double _Complex ** Tirrep = ( irot < nrot ) ? pr->rtarget->R[irot] :  pr->rtarget->IR[irot-nrot];

        fprintf ( stdout, "# [twopoint_function_check_reference_rotation] \n# [twopoint_function_check_reference_rotation] \n" );

        /******************************************************
         * show the spinor rotation matrix
         ******************************************************/
        if ( g_verbose > 2 ) {
          sprintf ( name, "Rspin%.2d", irot );
          rot_printf_matrix ( Rspin, spinor_dim, name, stdout );
          fprintf ( stdout, "# [twopoint_function_check_reference_rotation_vector_spinor] \n\n" );

          sprintf ( name, "Rvec%.2d", irot );
          rot_printf_matrix ( Rvec, vector_dim, name, stdout );
          fprintf ( stdout, "# [twopoint_function_check_reference_rotation_vector_spinor] \n\n" );
        }

        /******************************************************
         * loop on timeslices
         ******************************************************/
        for ( int it = 0; it < nT; it++ )
        {

          double _Complex ***** R1 = init_5level_ztable ( irrep_dim, vector_dim, vector_dim, spinor_dim, spinor_dim );
          double _Complex ***** R2 = init_5level_ztable ( irrep_dim, vector_dim, vector_dim, spinor_dim, spinor_dim );
          double _Complex ** Raux = init_2level_ztable ( spinor_dim, spinor_dim );

          /******************************************************
           * R1_{iref1; vi, vk} =
           *
           *   Rvec^-1_{vi,vj} Rspin^+ tp_{iref1,iref2,row_snk,row_src; vj,vk}
           * = Rvec_{vj,vi} Rspin^+ tp_{iref1,iref2,row_snk,row_src; vj,vk}
           * for Rvec given in real vector representation in cartesian basis
           ******************************************************/
          for ( int iref1 = 0; iref1 < irrep_dim; iref1++ ) {

            for ( int vi = 0; vi < vector_dim; vi++ ) {
            for ( int vk = 0; vk < vector_dim; vk++ ) {

              rot_mat_zero ( R1[iref1][vi][vk], spinor_dim );

              for ( int vj = 0; vj < vector_dim; vj++ ) {

                int const coords[] = { iref1, iref2, row_snk, row_src, vj, vk };
                unsigned int const idx = coords_to_index ( coords, index_conv_dim, coords_dim );
                rot_mat_adj_ti_mat ( Raux, Rspin, tp[ idx ].c[0][it], spinor_dim );

                double _Complex zcoeff = Rvec[vj][vi] * ( ( irot < nrot ) ? 1. : ( pr->parity[0] * pr->parity[1] ) );

                rot_mat_pl_eq_mat_ti_co ( R1[iref1][vi][vk], Raux, zcoeff, spinor_dim );

              }  /* end of loop on vector summation index vj */

            }}  /* end of loop on vector reference indices vi, vk */

          }  /* end of loop on irrep iref1 */

          /******************************************************
           * irrep rotation of same tp [ idx ] as above
           ******************************************************/
          for ( int iref1 = 0; iref1 < irrep_dim; iref1++ ) {

            for ( int vi = 0; vi < vector_dim; vi++ ) {
            for ( int vk = 0; vk < vector_dim; vk++ ) {

              rot_mat_zero ( R2[iref1][vi][vk] , spinor_dim );

              for ( int i = 0; i < irrep_dim; i++ ) {

                int const coords[] = { i, iref2, row_snk, row_src , vi, vk };
                unsigned int const idx = coords_to_index ( coords, index_conv_dim, coords_dim );
                /* R2 += Tirrep Cproj */
                rot_mat_pl_eq_mat_ti_co ( R2[iref1][vi][vk], tp[ idx ].c[0][it], conj( Tirrep[iref1][i] ), spinor_dim );
              }
            }}  /* end of loop on vector reference indices vi, vk */
          }  /* end of loop on irrep ref index iref1 */

          /******************************************************
           * check diff and norm
           ******************************************************/
          for ( int iref1 = 0; iref1 < irrep_dim; iref1++ ) {

            for ( int vi = 0; vi < vector_dim; vi++ ) {
            for ( int vk = 0; vk < vector_dim; vk++ ) {

              sprintf ( name, "Rsrc_r%.2d_t%.2d_gref%d_%d_%d_%d_vref%d_%d", irot, it, row_snk, row_src, iref1, iref2, vi, vk );
              if ( g_verbose > 2 ) {
                rot_printf_matrix_comp ( R1[iref1][vi][vk], R2[iref1][vi][vk], spinor_dim, name, stdout );
              }

              double const diff = rot_mat_norm_diff ( R1[iref1][vi][vk], R2[iref1][vi][vk], spinor_dim );
              double const norm = sqrt ( rot_mat_norm2 ( R1[iref1][vi][vk], spinor_dim ) );

              if ( ( norm > deps && diff/norm < deps ) || ( norm <= deps && diff <= deps ) ) {
                fprintf ( stdout, "# [twopoint_function_check_reference_rotation] snk %s diff norm %e    %e  okay\n\n", name, diff, norm );
              } else {
                fprintf ( stdout, "# [twopoint_function_check_reference_rotation] snk %s diff norm %e    %e  ERROR\n\n", name, diff, norm );
              }

            }}
          }

          fini_5level_ztable ( &R1 );
          fini_5level_ztable ( &R2 );
          fini_2level_ztable ( &Raux );

        }  /* end of loop on time slices */
      }  /* end of loop on rotations */

    }  /* end of loop on iref2 */
#if 0
#endif  /* of if 0 */

  }}  /* end of loops on row_src, row_snk */

}  /* end of twopoint_function_check_reference_rotation */

}  // end of namespace cvc
