#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "cvc_complex.h"
#include "iblas.h"
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "vdag_w_utils.h"
// #include "contract_cvc_tensor_eo_lm_factors.h"


namespace cvc {

/***********************************************************
 * contract_cvc_tensor_eo_lm_factors.cpp
 *
 * Mi 21. Mär 07:24:29 CET 2018
 *
 ***********************************************************/
int contract_cvc_tensor_eo_lm_factors (
    double ** const eo_evecs_field, unsigned int const nev, 
    double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2],
    struct AffWriter_s * affw, char * const tag, 
    const int (*momentum_list)[3], unsigned int const momentum_number,
    unsigned int const io_proc, 
    unsigned int const block_length 
) {

  unsigned int const Vhalf = VOLUME / 2;
  unsigned int const VOL3half = ( LX * LY * LZ ) / 2;
  size_t const sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);

  int exitstatus;

  char aff_tag[500];

  unsigned int const block_number = nev / block_length;
  if (io_proc == 2 && g_verbose > 3 ) {
    fprintf(stdout, "# [contract_cvc_tensor_eo_lm_factors] number of blocks = %u\n", block_number );
  }
  if ( nev - block_number * block_length != 0 ) {
    if ( io_proc == 2 ) fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error, nev not divisible by block_length\n");
    return(1);
  }

  /***********************************************************
   * auxilliary eo spinor fields with halo
   ***********************************************************/
  double ** eo_spinor_work = init_2level_dtable ( 4, _GSI( (VOLUME+RAND)/2 )  );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * set V
   ***********************************************************/
  double ** const v = eo_evecs_field;

  /***********************************************************
   * XV
   ***********************************************************/
  double ** xv = init_2level_dtable ( nev, _GSI( Vhalf )  );
  if ( xv == NULL) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  double *** contr_x = init_3level_dtable (  nev, block_length, 2*VOL3half );
  if ( contr_x == NULL ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  double _Complex *** contr_p = init_3level_ztable ( momentum_number, nev, block_length );
  if ( contr_p == NULL )  {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * NOTE: we save 2 FULL fields of size nev x 12 Vhalf complex
   *   (1) v as given by argument list
   *   (2) xv as calculated below
   ***********************************************************/

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set xv
   *
   * Xbar using Mbar_{ee}^{-1}, i.e. mzzinv[1][0] and 
   ***********************************************************/
  for ( unsigned int iev = 0; iev < nev; iev++ ) {
    memcpy( eo_spinor_work[0], v[iev], sizeof_eo_spinor_field );
    X_clover_eo ( xv[iev], eo_spinor_work[0], gauge_field, mzzinv[1][0] );
  }  // end of loop on eigenvectors

  /***********************************************************
   * auxilliary block field
   ***********************************************************/
  double *** eo_block_field = init_3level_dtable ( 4, block_length, _GSI(Vhalf) );
  if ( eo_block_field == NULL ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * W block field
   ***********************************************************/
  double ** w = init_2level_dtable ( block_length, _GSI(Vhalf) );
  if ( w == NULL ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * XW block field
   ***********************************************************/
  double ** xw = init_2level_dtable ( block_length, _GSI(Vhalf) );
  if ( xw == NULL ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(2);
  }


  /***********************************************************
   * loop on evec blocks
   ***********************************************************/
  for ( unsigned int iblock = 0; iblock < block_number; iblock++ ) {

    /***********************************************************
     * V^+ V block-wise
     ***********************************************************/

    /***********************************************************
     * loop on timeslices
     ***********************************************************/
    for ( int it = 0; it < T; it++ ) {

      /***********************************************************
       * initialize contraction fields to zero
       ***********************************************************/
      memset ( contr_p[0][0], 0, momentum_number * nev * block_length * sizeof(double _Complex ) );
      memset ( contr_x[0][0], 0, nev * block_length * 2*VOL3half * sizeof(double) );


      /***********************************************************
       * spin-color reduction
       ***********************************************************/
      exitstatus = vdag_w_spin_color_reduction ( contr_x, v, &(v[iblock*block_length]), nev, block_length, it );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }

      /***********************************************************
       * momentum projection
       ***********************************************************/
      exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 1, 0 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }

      /***********************************************************
       * write to file
       ***********************************************************/
      sprintf ( aff_tag, "%s/vv/t%.2d/b%.2d", tag, it+g_proc_coords[0]*T, iblock );
      if ( io_proc == 2 ) fprintf ( stdout, "# [contract_cvc_tensor_eo_lm_factors] current aff tag = %s\n", aff_tag );

      exitstatus = vdag_w_write_to_aff_file ( contr_p, nev, block_length, affw, aff_tag, momentum_list, momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }

    }  // end of loop on timeslices

    /***********************************************************/
    /***********************************************************/

#if 0
    /***********************************************************
     * calculate w, xw for the current block,
     * w = Cbar ( v, xv )
     * xw = X w
     ***********************************************************/
    for ( unsigned int iev = 0; iev < block_length; iev++ ) {
      memcpy( w[iev], v[iblock*block_length+iev], sizeof_eo_spinor_field );
      memcpy( eo_spinor_work[0],  xv[iblock*block_length+iev], sizeof_eo_spinor_field );
      C_clover_from_Xeo ( w[iev], eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1] );

      memcpy( eo_spinor_work[0],  w[iev], sizeof_eo_spinor_field );
      X_clover_eo ( xw[iev], eo_spinor_work[0], gauge_field, mzzinv[0][0] );
    }

    /***********************************************************
     * loop vector index mu
     ***********************************************************/
    for ( int mu = 0; mu < 4; mu++ ) {

      /***********************************************************
       * apply  g5 Gmu W
       ***********************************************************/
      for ( unsigned int iev = 0; iev < block_length; iev++ ) {
        memcpy ( eo_spinor_work[0], w[iev], sizeof_eo_spinor_field );
        /* Gmufwdr */
        apply_cvc_vertex_eo( eo_block_field[0][iev], eo_spinor_work[0], mu, 0, gauge_field, 0 );
        /* Gmubwdr */
        apply_cvc_vertex_eo( eo_block_field[1][iev], eo_spinor_work[0], mu, 1, gauge_field, 0 );
      }
      g5_phi ( eo_block_field[0][0], 2 * Vhalf * block_length );


      /***********************************************************
       * apply  g5 Gmu XW
       ***********************************************************/
      for ( unsigned int iev = 0; iev < block_length; iev++ ) {
        memcpy ( eo_spinor_work[0], xw[iev], sizeof_eo_spinor_field );
        /* Gmufwdr */
        apply_cvc_vertex_eo( eo_block_field[2][iev], eo_spinor_work[0], mu, 0, gauge_field, 1 );
        /* Gmubwdr */
        apply_cvc_vertex_eo( eo_block_field[3][iev], eo_spinor_work[0], mu, 1, gauge_field, 1 );
      }
      g5_phi ( eo_block_field[2][0], 2 * Vhalf * block_length );

      /***********************************************************
       * loop on timeslices
       ***********************************************************/
      for ( int it = 0; it < T; it++ ) {

        /***********************************************************
         * XV^+ x Gmufwd W
         ***********************************************************/

        /***********************************************************
         * spin-color reduction
         ***********************************************************/
        exitstatus = vdag_w_spin_color_reduction ( contr_x, xv, eo_block_field[0], nev, block_length, it );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * momentum projection
         ***********************************************************/
        exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 0, mu );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * V^+ x Gmubwd XW
         ***********************************************************/

        /***********************************************************
         * spin-color reduction
         ***********************************************************/
        exitstatus = vdag_w_spin_color_reduction ( contr_x, v, eo_block_field[3], nev, block_length, it+1 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * momentum projection
         ***********************************************************/
        exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 1, mu );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * XV^+ x Gmubwd W
         ***********************************************************/

        /***********************************************************
         * spin-color reduction
         ***********************************************************/
        exitstatus = vdag_w_spin_color_reduction ( contr_x, xv, eo_block_field[2], nev, block_length, it+1 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * momentum projection
         ***********************************************************/
        exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 0, mu );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * V^+ x Gmufwd XW
         ***********************************************************/

        /***********************************************************
         * spin-color reduction
         ***********************************************************/
        exitstatus = vdag_w_spin_color_reduction ( contr_x, xv, eo_block_field[1], nev, block_length, it );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * momentum projection
         ***********************************************************/
        exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 1, mu );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * write to file
         ***********************************************************/
        sprintf ( aff_tag, "%s/t%.2d/mu%d/b%.2d", tag, it+g_proc_coords[0]*T, mu, iblock );
        exitstatus = vdag_w_write_to_aff_file ( contr_p, nev, block_length, affw, aff_tag, momentum_list, momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

      }  // end of loop on timeslices

    }  // end of loop on vector index mu

#endif  // of if 0

  }  // end of loop on evec blocks

#if 0
  /***********************************************************
   * calculate all Wtildes, save them in place of XV in field
   * xv
   ***********************************************************/

  for ( unsigned int iev = 0; iev < nev; iev++ ) {
    // eo_spinor_work <- xv
    memcpy( eo_spinor_work[0],  xv[iev], sizeof_eo_spinor_field );
    // xv <- v
    memcpy( xv[iev], v[iev], sizeof_eo_spinor_field );
    // w <- C from Xeo ( v, xv )
    C_clover_from_Xeo ( xv[iev], eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1] );
  }

  /***********************************************************
   * loop on evec blocks
   ***********************************************************/
  for ( unsigned int iblock = 0; iblock < block_number; iblock++ ) {

    /***********************************************************
     * W^+ W block-wise
     ***********************************************************/

    /***********************************************************
     * loop on timeslices
     ***********************************************************/
    for ( int it = 0; it < T; it++ ) {

      /***********************************************************
       * initialize contraction fields to zero
       ***********************************************************/
      memset ( contr_p[0][0], 0, momentum_number * nev * block_length * sizeof(double _Complex ) );
      memset ( contr_x[0][0], 0, nev * block_length * 2*VOL3half * sizeof(double) );

      /***********************************************************
       * spin-color reduction
       ***********************************************************/
      exitstatus = vdag_w_spin_color_reduction ( contr_x, xv, &(xv[iblock*block_length]), nev, block_length, it );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }

      /***********************************************************
       * momentum projection
       ***********************************************************/
      exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 1, 0 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }

      /***********************************************************
       * write to file
       ***********************************************************/
      sprintf ( aff_tag, "%s/ww/t%.2d/b%.2d", tag, it+g_proc_coords[0]*T, iblock );
      exitstatus = vdag_w_write_to_aff_file ( contr_p, nev, block_length, affw, aff_tag, momentum_list, momentum_number, io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(4);
      }
    }  // end of loop on timeslices
  
  }  // end of loop on evecs blocks

#endif  // of if 0

  fini_2level_dtable ( &eo_spinor_work );
  fini_2level_dtable ( &xv );

  fini_3level_dtable ( &eo_block_field );
  fini_2level_dtable ( &w  );
  fini_2level_dtable ( &xw );

  fini_3level_dtable ( &contr_x );
  fini_3level_ztable ( &contr_p );

#if 0
#endif  // of if 0
  return(0);
}  // end of contract_cvc_tensor_eo_lm_factors

}
