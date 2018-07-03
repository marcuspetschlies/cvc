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
#include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "types.h"
#include "global.h"
#include "default_input_values.h"
#include "zm4x4.h"
#include "table_init_z.h"
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
}  /* end of twopoint_function_init */

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
  return;
}  /* end of twopoint_function_print */

/********************************************************************************
 *
 ********************************************************************************/
void twopoint_function_copy ( twopoint_function_type *p, twopoint_function_type *r ) {
  strcpy( p->name, r->name );
  strcpy( p->type, r->type );
  strcpy( p->diagrams, r->diagrams );
  strcpy( p->norm, r->norm );
  strcpy( p->fbwd, r->fbwd );
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

  return;
}  /* end of twopoint_function_print */



/********************************************************************************
 *
 ********************************************************************************/

void twopoint_function_print_diagram_key (char*key, twopoint_function_type *p, int id ) { 

  char comma[] = ",";
  char *ptr = NULL;
  char diag_str[20];
  char diagrams[400];
  char fbwd_str[10] = "";

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
    sprintf( key, "/%s/%s%spi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d", p->name, diag_str, fbwd_str,
        p->pi2[0], p->pi2[1], p->pi2[2],
        p->pf1[0], p->pf1[1], p->pf1[2],
        p->pf2[0], p->pf2[1], p->pf2[2],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3],
        p->gf1[0], p->gi1[0]);
  } else if ( strcmp( p->type, "mxb-b") == 0 ) {

    sprintf( key, "/%s/%s%spi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d", p->name, diag_str, fbwd_str,
        p->pi2[0], p->pi2[1], p->pi2[2],
        p->pf1[0], p->pf1[1], p->pf1[2],
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3],
        p->gf1[0], p->gi1[0]);

    /* /piN-D/t54x14y01z32/pi2x01pi2y01pi2z-01/gi02/gf15/t5/px01py01pz-01 */
    /* sprintf( key, "/%s/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/%spx%.2dpy%.2dpz%.2d", p->name, 
        p->source_coords[0], p->source_coords[1], p->source_coords[2], p->source_coords[3],
        p->pi2[0], p->pi2[1], p->pi2[2],
        gamma_to_C_gamma[p->gi1[0]][0], gamma_to_C_gamma[p->gf1[0]][0],
        diag_str,
        p->pf1[0], p->pf1[1], p->pf1[2]); */

  } else {
    strcpy ( key, "NA" );
  }
  fprintf(stdout, "# [twopoint_function_print_diagram_key] key = \"%s\"\n", key);
  return;
}  /* end of twopoint_function_print_key */


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

  if ( strcmp( p->type , "b-b" ) == 0 || strcmp( p->type , "mxb-b" ) == 0 ) {
    strcpy(filename, filename_prefix);
  } else if ( strcmp( p->type , "mxb-mxb" ) == 0 ) {
    strcpy(filename, filename_prefix2);
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
    if ( strcmp ( ptr, "NA" ) == 0 ) { return(0.); }
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
    for ( int it = 0; it < N; it++ ) {
      fprintf(stdout, "# [twopoint_function_accumulate_diagrams] initial correlator t = %2d\n", it );
      zm4x4_printf ( diagram[it], "c_in", stdout );
    }
  }

  return(0);

}  // end of twopoint_function_accumulate_diagrams

}  // end of namespace cvc
