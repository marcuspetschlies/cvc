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
#include "global.h"
#include "default_input_values.h"
#include "io_utils.h"
#include "zm4x4.h"
#include "table_init_z.h"
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
  fprintf(ofs, "# [print_twopoint_function] %s.name = %s\n", name, p->name);
  fprintf(ofs, "# [print_twopoint_function] %s.type = %s\n", name, p->type);
  fprintf(ofs, "# [print_twopoint_function] %s.pi1  = (%3d,%3d, %3d)\n", name, p->pi1[0], p->pi1[1], p->pi1[2] );
  fprintf(ofs, "# [print_twopoint_function] %s.pi2  = (%3d,%3d, %3d)\n", name, p->pi2[0], p->pi2[1], p->pi2[2] );
  fprintf(ofs, "# [print_twopoint_function] %s.pf1  = (%3d,%3d, %3d)\n", name, p->pf1[0], p->pf1[1], p->pf1[2] );
  fprintf(ofs, "# [print_twopoint_function] %s.pf2  = (%3d,%3d, %3d)\n", name, p->pf2[0], p->pf2[1], p->pf2[2] );
  fprintf(ofs, "# [print_twopoint_function] %s.gi1  = (%3d,%3d)\n", name, p->gi1[0], p->gi1[1] );
  fprintf(ofs, "# [print_twopoint_function] %s.gi2  =  %3d\n", name, p->gi2 );
  fprintf(ofs, "# [print_twopoint_function] %s.gf1  = (%3d,%3d)\n", name, p->gf1[0], p->gf1[1] );
  fprintf(ofs, "# [print_twopoint_function] %s.gf2  =  %3d\n", name, p->gf2 );
  fprintf(ofs, "# [print_twopoint_function] %s.n    =  %3d\n", name, p->n );
  fprintf(ofs, "# [print_twopoint_function] %s.spin_project    =  %3d\n", name, p->spin_project );
  fprintf(ofs, "# [print_twopoint_function] %s.parity_project  =  %3d\n", name, p->parity_project );
  fprintf(ofs, "# [print_twopoint_function] %s.diagrams        =  %s\n", name, p->diagrams );
  fprintf(ofs, "# [print_twopoint_function] %s.norm            =  %s\n", name, p->norm );
  fprintf(ofs, "# [print_twopoint_function] %s.source_coords   =  (%3d, %3d, %3d, %3d)\n", name, 
      p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3] );
  fprintf(ofs, "# [print_twopoint_function] %s.reorder         =  %3d\n", name, p->reorder );
  fprintf(ofs, "# [print_twopoint_function] %s.fbwd            =  %s\n", name, p->fbwd );
  fprintf(ofs, "# [print_twopoint_function] %s.T               =  %d\n", name, p->T );
  fprintf(ofs, "# [print_twopoint_function] %s.d               =  %d\n", name, p->d );
  fprintf(ofs, "# [print_twopoint_function] %s.group           =  %s\n", name, p->group );
  fprintf(ofs, "# [print_twopoint_function] %s.irrep           =  %s\n", name, p->irrep );
  if ( p->c != NULL ) {
    fprintf ( ofs, "# [print_twopoint_function] data array is set\n");
  } else {
    fprintf ( ofs, "# [print_twopoint_function] data is empty\n");
  }
  return;
}  // end of twopoint_function_print

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_copy ( twopoint_function_type *p, twopoint_function_type *r ) {
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
  p->n = r->n;
  p->spin_project = r->spin_project;
  p->parity_project = r->parity_project;
  memcpy( p->source_coords, r->source_coords, 4*sizeof(int) );
  p->reorder = r->reorder;
  p->T       = r->T;
  p->d       = r->d;
  if ( r->c != NULL ) {
    if ( twopoint_function_allocate ( p ) != NULL ) {
      fprintf ( stderr, "[twopoint_function_copy] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
    } else {
      memcpy ( p->c[0][0][0], r->c[0][0][0], p->n * p->T * p->d * p->d * sizeof( double _Complex ) );
    }
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
  char diagrams[400];
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
  char diagrams[500];
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

    char diagram_tag_tmp[20];
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
          Ptot[0], Ptot[1], Ptot[2], Nconf,
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
      if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_fill_data] using existing reader\n");
    }  /* end of need new reader */

    /* current filename pattern uses lower-case tag, so this 
     * toupper is not needed now */
    /* strcpy ( filename_tag, diagram_tag ); */
    /* filename_tag[0] = toupper ( diagram_tag[0] ); */

    sprintf( key, "/%s/%s/%s/%s", p->name, diagram_tag, p->fbwd, key_suffix );
    if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_fill_data] key = %s\n", key );


    affdir = aff_reader_chpath (affr, affn, key );
    uint32_t uitems = d *d * nT;
    exitstatus = aff_node_get_complex (affr, affdir, diagram[0][0], uitems );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[twopoint_function_fill_data] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(105);
    }

    if ( strcmp ( p->norm , "NA" ) != 0 ) {
      double const norm = twopoint_function_get_diagram_norm ( p, i );
      if ( g_verbose > 3 ) fprintf ( stdout, "# [twopoint_function_fill_data] using diagram norm %2d %16.7e\n", i, norm );

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

}  // end of twopoint_function_fill_data

/********************************************************************************/
/********************************************************************************/
#if ! ( ( defined HAVE_LHPC_AFF ) || ( defined HAVE_HDF5 ) )
/********************************************************************************
 * write diagrams in a twopoint
 ********************************************************************************/
int twopoint_function_write_data ( twopoint_function_type *p ) {

  int const nT     = p->T;         // timeslices
  int const ncomp  = p->d * p->d;  // number of components per timeslice of the data set; usually ( spin dimension )^2
  int const nD     = p->n;         // number of data sets, usually number of diagrams

#ifndef HAVE_LHPC_AFF
  fprintf ( stderr, "[twopoint_function_write_data] Error, only AFF writing implemented %s %d\n", __FILE__, __LINE__ );
  return(3);
#endif

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
   * open AFF writer
   ******************************************************/
  sprintf ( filename, "%s.%s.PX%dPY%dPZ%d.%s.%.4d.t%dx%dy%dz%d", p->name, p->group, Ptot[0], Ptot[1], Ptot[2], p->irrep, Nconf,
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

    char dataset_tag[10];
    twopoint_function_get_diagram_name ( dataset_tag, p, i );

    sprintf( key, "/%s/%s/%s/%s", p->name, dataset_tag, p->fbwd, key_suffix );
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
  sprintf ( filename, "%s.%s.PX%dPY%dPZ%d.%s.%.4d.t%dx%dy%dz%d", p->name, p->group, Ptot[0], Ptot[1], Ptot[2], p->irrep, Nconf,
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

    char dataset_tag[10];
    twopoint_function_get_diagram_name ( dataset_tag, p, i );

    sprintf( key, "/%s/%s/%s/%s", p->name, dataset_tag, p->fbwd, key_suffix );
    fprintf ( stdout, "# [twopoint_function_write_data] key = %s\n", key );

    struct AffNode_s * affdir = aff_writer_mkpath ( affw, affn, key );
    if ( affdir == NULL ) {
      fprintf ( stderr, "[twopoint_function_write_data] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__ );
      return(2);
    }

    uint32_t uitems = ncomp * nT;
    exitstatus = aff_node_put_complex ( affw, affdir, dataset[i][0], uitems );
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
   * HDF5 reader
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
  int grp_list_nmem = 0;
  char grp_name[400], grp_name_tmp[400];
  char * grp_ptr = NULL;
  char grp_sep[] = "/";
  sprintf( grp_name, "/%s/%s%s", p->name, p->fbwd, key_suffix );
  strcpy ( grp_name_tmp, grp_name );
  fprintf ( stdout, "# [twopoint_function_write_data] full grp_name = %s\n", grp_name );
  grp_ptr = strtok ( grp_name_tmp, grp_sep );

  while ( grp_ptr != NULL ) {
    hid_t grp;
    hid_t loc_id = ( grp_list_nmem == 0 ) ? file_id : grp_list[grp_list_nmem-1];
    fprintf ( stdout, "# [twopoint_function_write_data] grp_ptr = %s\n", grp_ptr );

    grp = H5Gopen2( loc_id, grp_ptr, gapl_id );
    if ( grp < 0 ) {
      fprintf ( stderr, "[twopoint_function_write_data] Error from H5Gopen2 for group %s, status was %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
      grp = H5Gcreate2 (       loc_id,         grp_ptr,       lcpl_id,       gcpl_id,       gapl_id );
      if ( grp < 0 ) {
        fprintf ( stderr, "[twopoint_function_write_data] Error from H5Gcreate2 for group %s, status was %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
        return ( 110 );
      } else {
        fprintf ( stdout, "# [twopoint_function_write_data] created group %s %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
      }
    } else {
      fprintf ( stdout, "# [twopoint_function_write_data] opened group %s %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
    }
    grp_ptr = strtok(NULL, grp_sep );

    grp_list[grp_list_nmem] = grp;
    grp_list_nmem++;
  }  /* end of loop on sub-groups */

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

    double _Complex *** const dataset = p->c[i];
    hid_t loc_id = ( grp_list_nmem == 0 ) ? file_id : grp_list[grp_list_nmem - 1 ];
    fprintf ( stdout, "# [twopoint_function_write_data] data set %2d loc_id = %ld %s %d\n", i, loc_id , __FILE__, __LINE__ );

    /******************************************************
     * construct the full key
     ******************************************************/
    char name[10];
    twopoint_function_get_diagram_name ( name, p, i );

    fprintf ( stdout, "# [twopoint_function_write_data] name = %s %s %d\n", name, __FILE__, __LINE__ );

    /******************************************************
     * create a data set
     ******************************************************/

    //             hid_t loc_id 	IN: Location identifier
    //             const char *name     	IN: Dataset name
    //             hid_t dtype_id 	IN: Datatype identifier
    //             hid_t space_id 	IN: Dataspace identifier
    //             hid_t lcpl_id 	IN: Link creation property list
    //             hid_t dcpl_id 	IN: Dataset creation property list
    //             hid_t dapl_id 	IN: Dataset access property list
    //             hid_t H5Dcreate2 ( hid_t loc_id, const char *name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id )
    // hid_t dataset_id   = H5Dcreate2 ( file_id,      key,                    dtype_id,       space_id,   H5P_DEFAULT,   H5P_DEFAULT,   H5P_DEFAULT   );

    //           hid_t H5Dcreate ( hid_t loc_id, const char *name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id ) 
    hid_t dataset_id = H5Dcreate (       loc_id,             name,       dtype_id,       space_id,       lcpl_id,       dcpl_id,       dapl_id );


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

}  // end of namespace cvc
