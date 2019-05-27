#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "global.h"
#include "matrix_init.h"
#include "gamma.h"
#include "aff_key_conversion.h"

namespace cvc {

const int C_gamma_to_gamma[16][2] = {
    {2,-1},
    {9,1},
    {0,1},
    {7,-1},
    {11,1},
    {14,1},
    {8,-1},
    {3,1},
    {6,1},
    {1,-1},
    {13,1},
    {4,-1},
    {15,-1},
    {10,-1},
    {5,-1},
    {12,1} };

/*************************************************************************************
 *
 *************************************************************************************/
void aff_key_conversion (char*key, char * const tag, int const i_sample, int const pi2[3], int const pf1[3], int const pf2[3], int const source_coords[4], int const gamma_id, int const C_gamma_id,  int const i_spin ) {

  int C_gid = ( C_gamma_id >= 0 ) ? C_gamma_id : ( gamma_id >= 0 ? C_gamma_to_gamma[gamma_id][0] : -1 );
  if ( C_gid == -1 ) {
    fprintf ( stderr, "[aff_key_conversion] Error, could not make sense out of input gamma_id = %d and C_gamma_id = %d\n", gamma_id, C_gamma_id );
    sprintf ( key, "NA");
    return;
  }

  if ( strcmp( tag, "w_1_xi") == 0 ) {
    /* /w_1_xi/sample03/pf2x00pf2y00pf2z-01/t09x06y07z00 */

    sprintf( key, "/v3/t%.2dx%.2dy%.2dz%.2d/xi-g%.2d-u/sample%.2d/px%.2dpy%.2dpz%.2d",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        C_gid, i_sample,
        pf2[0], pf2[1], pf2[2]);

  } else if ( strcmp( tag, "w_1_phi") == 0 ) {
    /* /w_1_phi/sample02/pi2x00pi2y00pi2z01/pf1x00pf1y00pf1z01/t09x06y07z00/g06 */

    sprintf(key, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-u-ud/sample%.2d/px%.2dpy%.2dpz%.2d",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        pi2[0], pi2[1], pi2[2], C_gid, i_sample,
        pf1[0], pf1[1], pf1[2]);

  } else if ( strcmp( tag, "w_3_phi") == 0 ) {
    /* /w_3_phi/sample03/pi2x00pi2y00pi2z-01/pf1x00pf1y00pf1z-01/t01x02y03z04/g04 */

    sprintf(key, "/v2/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-ud-u/sample%.2d/px%.2dpy%.2dpz%.2d",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        pi2[0], pi2[1], pi2[2], C_gid, i_sample,
        pf1[0], pf1[1], pf1[2]);

  } else if ( strcmp( tag, "b_1_xi") == 0 ) {
    /* /b_1_xi/sample03/pi2x00pi2y00pi2z-01/pf2x00pf2y00pf2z-01/t09x06y07z00 */

    sprintf(key, "/v3/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/xi-g%.2d-ud/sample%.2d/px%.2dpy%.2dpz%.2d",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        pi2[0], pi2[1], pi2[2], C_gid, i_sample,
        pf2[0], pf2[1], pf2[2] );

  } else if ( strcmp( tag, "b_1_phi") == 0 ) {
    /* /b_1_phi/sample03/pf1x00pf1y00pf1z01/t09x06y07z00/g06 */

    sprintf(key, "/v2/t%.2dx%.2dy%.2dz%.2d/phi-g%.2d-u-u/sample%.2d/px%.2dpy%.2dpz%.2d",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        C_gid, i_sample,
        pf1[0], pf1[1], pf1[2]);

  } else if ( strcmp( tag, "z_3_phi") == 0 ) {
    /* /z_3_phi/sample00/pi2x00pi2y00pi2z-01/pf1x00pf1y00pf1z-01/t09x06y07z00/g06 */

    sprintf(key, "/v2-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d-u/sample%.2d/d%d/px%.2dpy%.2dpz%.2d",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        pi2[0], pi2[1], pi2[2], C_gid, i_sample, i_spin,
        pf1[0], pf1[1], pf1[2]);

  } else if ( strcmp( tag, "z_1_phi") == 0 ) {
    /* /z_1_phi/sample00/pi2x00pi2y00pi2z-01/pf1x00pf1y00pf1z-01/t01x02y03z04/g06 */

    sprintf(key, "/v4-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-d-u/sample%.2d/d%d/px%.2dpy%.2dpz%.2d", 
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        pi2[0], pi2[1], pi2[2], C_gid, i_sample, i_spin, pf1[0], pf1[1], pf1[2]);

  } else if ( strcmp( tag, "z_1_xi") == 0 ) {
    /* /z_1_xi/sample00/pf2x00pf2y00pf2z-01/t09x06y07z00 */

    sprintf(key, "/v3-oet/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/phi-g%.2d-u/sample%.2d/d%d/px%.2dpy%.2dpz%.2d",
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        pi2[0], pi2[1], pi2[2], C_gid, i_sample, i_spin,
        pf2[0], pf2[1], pf2[2] );
  }

}  // end of key_conversion

/*************************************************************************************/
/*************************************************************************************/

/*************************************************************************************
 *
 *************************************************************************************/
int v2_key_index_conversion ( double _Complex *buffer, int perm[4], int N, int LL[4] ) {

  const unsigned int LLvol = LL[0] * LL[1] * LL[2] * LL[3];
  const unsigned int items = (unsigned int)N * LLvol;
  const size_t bytes = items * sizeof( double _Complex );
  int ii[4];

  double _Complex *buffer_aux = (double _Complex*)malloc( bytes );
  if( buffer_aux == NULL ) {
    fprintf(stderr, "[] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  } 
  memcpy( buffer_aux, buffer, bytes );

  for ( int it = 0; it < N; it++ ) {

    for ( int alpha = 0; alpha < LL[0]; alpha++ ) {
    for ( int beta  = 0; beta  < LL[1]; beta++  ) {
    for ( int gamma = 0; gamma < LL[2]; gamma++ ) {

      for ( int n = 0; n < LL[3]; n++ ) {

        ii[perm[0]] = alpha;
        ii[perm[1]] = beta;
        ii[perm[2]] = gamma;
        ii[perm[3]] = n;
        
        unsigned int idx     = it * LLvol + LL[3]       * ( LL[2]       * ( LL[1]       * alpha + beta  ) + gamma  ) + n;
        unsigned int idxperm = it * LLvol + LL[perm[3]] * ( LL[perm[2]] * ( LL[perm[1]] * ii[0] + ii[1] ) + ii[2]  ) + ii[3];

        buffer[idx] = buffer_aux[idxperm];
      }
    }}}

  }
  free( buffer_aux);
  return(0);
}  // end of v2_index_conversion

/*************************************************************************************/
/*************************************************************************************/

/*************************************************************************************
 *
 *************************************************************************************/
int vn_oet_read_key ( double _Complex *key_buffer, char*tag, int const i_sample, int const pi2[3], int const pf1[3], int const pf2[3], int const source_coords[4], int const gamma_id, int const C_gamma_id, struct AffReader_s *affr ) {

#ifdef HAVE_LHPC_AFF
  int exitstatus, perm[4];
  char key[200];
  double _Complex **buffer_aux = NULL, ***buffer = NULL;
  struct AffNode_s *affn = NULL, *affdir = NULL;
  unsigned int LLvol;
  int LL[4];

  if ( strcmp(tag, "z_1_phi") == 0 ) {

    perm[0] = 0; perm[1] = 2; perm[2] = 1; perm[3] = 3;
    LL[0]   = 4; LL[1]   = 4; LL[2]   = 4; LL[3]   = 3;

  } else if ( strcmp(tag, "z_3_phi") == 0 ) {

    perm[0] = 1; perm[1] = 2; perm[2] = 0; perm[3] = 3;
    LL[0]   = 4; LL[1]   = 4; LL[2]   = 4; LL[3]   = 3;

  } else if ( strcmp(tag, "z_1_xi") == 0 ) {

    perm[0] = 0; perm[1] = 1; perm[2] = 2; perm[3] = 3;
    LL[0]   = 4; LL[1]   = 3; LL[2]   = 1; LL[3]   = 1;

  }

  LLvol = LL[0] * LL[1] * LL[2] * LL[3];

  exitstatus = init_2level_zbuffer ( &buffer_aux, T_global, LLvol);
  if( exitstatus != 0 ) {            
    fprintf(stderr, "[v2_oet_read_key] Error from init_2level_zbuffer, status was %d\n", exitstatus);
    return(1);                                                                 
  }         

  exitstatus = init_3level_zbuffer ( &buffer, T_global, 4, LLvol);
  if( exitstatus != 0 ) {            
    fprintf(stderr, "[v2_oet_read_key] Error from init_3level_zbuffer, status was %d\n", exitstatus);
    return(1);                                                                 
  }

  if ( buffer_aux == NULL ) {
    fprintf(stderr, "Error from malloc %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  if( (affn = aff_reader_root( affr )) == NULL ) {
    fprintf(stderr, "[v2_oet_read_key] Error, aff writer is not initialized\n");
    return(1);
  }

  for ( int ispin = 0; ispin < 4; ispin++ ) {
    aff_key_conversion ( key, tag, i_sample, pi2, pf1, pf2, source_coords, gamma_id, C_gamma_id, ispin );
    fprintf(stdout, "# [v2_oet_read_key] key = \"%s\"\n", key);


    affdir = aff_reader_chpath (affr, affn, key );
    exitstatus = aff_node_get_complex (affr, affdir, buffer_aux[0], T_global*LLvol);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[v2_oet_read_key] Error from aff_node_get_complex, status was %d\n", exitstatus);
      return(105);
    }

    exitstatus = v2_key_index_conversion ( buffer_aux[0], perm, T_global, LL );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[v2_oet_read_key] Error from v2_key_index_conversion, status was %d\n", exitstatus);
      return(105);
    }

    for ( int it = 0; it < T_global; it++ ) {
      memcpy ( buffer[it][ispin], buffer_aux[it], LLvol*sizeof(double _Complex ) );
    }
  }  /* end of loop on oet spin index */
  
  memcpy ( key_buffer, buffer[0][0], 4*LLvol*T_global*sizeof(double _Complex ) );
  fini_2level_zbuffer ( &buffer_aux );
  fini_3level_zbuffer ( &buffer );
  return(0);
#else
  fprintf( stderr, "[vn_oet_read_key] Error, non-aff version not yet implemented\n" );
  return ( 1 );
#endif
}  /* end vn_oet_read_key */

/**************************************************************************************/
/**************************************************************************************/

/**************************************************************************************
 *
 **************************************************************************************/
void aff_key_conversion_diagram (  char*key, char * const tag, int const pi1[3], int const pi2[3], int const pf1[3], int const pf2[3], 
    int const gi1, int const gi2, int const gf1, int const gf2, int const source_coords[4], char * const diag_name, int const diag_id, int const gx1_mult_C ) {

  char diag_str[100];

  int const C_gi1 = gx1_mult_C ? C_gamma_to_gamma[gi1][0] : gi1;
  int const C_gf1 = gx1_mult_C ? C_gamma_to_gamma[gf1][0] : gf1;

  if ( diag_name == NULL && diag_id < 0 ) {
    diag_str[0] = '\0';
  } else if ( diag_name == NULL && diag_id >= 0 ) {
    sprintf(diag_str, "diag%d/", diag_id);
  } else if (diag_id >= 0 ) {
    sprintf(diag_str, "%s%d/", diag_name, diag_id);
  } else {
    fprintf(stderr, "[aff_key_conversion_diagram] Error, have diagram name %s but no id\n", diag_name);
    strcpy( key, "NA" );
    return;
  }

  if ( strcmp(tag, "N-N" ) == 0 || strcmp(tag, "D-D" ) == 0 ) {

    sprintf( key, "/%s/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/%spx%.2dpy%.2dpz%.2d", tag, 
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        C_gi1, C_gf1, diag_str, pf1[0], pf1[1], pf1[2]);

  } else if ( strcmp(tag, "piN-D" ) == 0 ) {

    sprintf( key, "/%s/t%.2dx%.2dy%.2dz%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi%.2d/gf%.2d/%spx%.2dpy%.2dpz%.2d", tag, 
        source_coords[0], source_coords[1], source_coords[2], source_coords[3],
        pi2[0], pi2[1], pi2[2], C_gi1, C_gf1, diag_str, pf1[0], pf1[1], pf1[2]);

  } else if ( strcmp(tag, "m-m" ) == 0 ) {

    sprintf( key, "/%s/t%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/sample%.2d/gi%.2d/gf%.2d/px%.2dpy%.2dpz%.2d", tag,
        source_coords[0],
        pi2[0], pi2[1], pi2[2],
        diag_id, gi2, gf2, pf2[0], pf2[1], pf2[2] );

  } else {
    fprintf(stderr, "[aff_key_conversion_diagram] Error, unrecognized tag %s\n", tag);
    strcpy( key, "NA" );
    return;
  }

  return;
}  // end of aff_key_conversion_diagram

/**************************************************************************************/
/**************************************************************************************/

/**************************************************************************************
 *
 **************************************************************************************/
void gamma_name_to_gamma_signed_id (int *id, double*sign, char *name ) {

  char *dot = ".";
  char *ptr;
  char gname[200];
  strcpy(gname, name);
  gamma_matrix_type g_work1, g_work2, g_accum;

  init_gamma_matrix();

  gamma_matrix_set ( &g_accum, 4, 1. );
  gamma_matrix_init ( &g_work1);
  gamma_matrix_init ( &g_work2);

  gamma_matrix_fill ( &g_accum );
  gamma_matrix_printf (&g_accum, "g_accum", stdout);

  if (g_verbose > 2 ) fprintf(stdout, "# [gamma_name_to_gamma_signed_id] name = %s\n", gname );
  ptr = strtok( gname, dot);
  if ( ptr == NULL ) {
    *id = -1;
    return;
  }
  if (g_verbose > 3) fprintf(stdout, "# [gamma_name_to_gamma_signed_id] inital ptr = %s\n", ptr);

  while( ptr != NULL) {
    if ( ptr[0] == 'C' ) {
      if (g_verbose > 3 ) fprintf(stdout, "# [gamma_name_to_gamma_signed_id] ptr = %s using C = g0.g2\n", ptr);
      g_work1.id = 0; g_work1.s = 1.; gamma_matrix_fill ( &g_work1 );
      g_work2.id = 2; g_work1.s = 1.; gamma_matrix_fill ( &g_work2 );
      gamma_matrix_mult ( &g_accum, &g_accum, &g_work1 );
      gamma_matrix_mult ( &g_accum, &g_accum, &g_work2 );
      // gamma_matrix_printf (&g_accum, "g_accum", stdout);
    } else if ( ptr[0] == 'g') {
      int ig = atoi( &(ptr[1]));
      g_work1.id = ig; g_work1.s = 1.; gamma_matrix_fill ( &g_work1 );
      gamma_matrix_mult ( &g_accum, &g_accum, &g_work1 );
      // gamma_matrix_printf (&g_accum, "g_accum", stdout);
      if (g_verbose > 3 ) fprintf(stdout, "# [gamma_name_to_gamma_signed_id] ptr = %s g%.2d\n", ptr, ig);
    }
    ptr = strtok(NULL, dot);
  }
  *id = g_accum.id;
  *sign = g_accum.s;
  if ( g_verbose > 3 ) {
    fprintf(stdout, "# [gamma_name_to_gamma_signed_id] accumulated gamma matrix:\n");
    gamma_matrix_printf (&g_accum, name, stdout);
  }
  return;
}  // end of gamma_name_to_gamma_id


}  // end of namespace cvc
