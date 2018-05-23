/***********************************************************
 * contract_cvc_tensor_eo_lm_mee_ct.cpp
 *
 * Mi 21. Mär 07:21:36 CET 2018
 ***********************************************************/

/***********************************************************
 * contractions for Mee part of eo-precon lm cvc - cvc tensor
 ***********************************************************/
int const contract_cvc_tensor_eo_lm_mee_ct (
    double ** const eo_evecs_field, unsigned int const nev,
    double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2],
    struct AffWriter_s * affw, char * const tag, 
    int (* const momentum_list)[3], unsigned int const momentum_number,
    unsigned int const io_proc
) {

  const unsigned int Vhalf                      = VOLUME / 2;
  const size_t sizeof_eo_spinor_field           = _GSI( Vhalf    ) * sizeof(double);

  int exitstatus;
#ifdef HAVE_LHPC_AFF
  struct AffNode_s *affn = NULL, *affdir=NULL;
#endif

  if ( g_ts_id == 0 && g_tr_id == 0 ) {
#ifdef HAVE_LHPC_AFF
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_cvc_tensor_eo_lm_mee_ct] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
#endif
  }

  // allocate auxilliary spinor fields and spinor work (with halo)
  double ** eo_spinor_field = init_2level_dtable ( 5, _GSI(Vhalf) );
  if ( eo_spinor_field == NULL ) {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee_ct] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double ** eo_spinor_work = init_2level_dtable ( 1, _GSI( (VOLUME+RAND)/2 ) );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee_ct] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double * const v    = eo_spinor_field[ 0];
  double * const w    = eo_spinor_field[ 1];
  double * const xv   = eo_spinor_field[ 2];
  double * const xw   = eo_spinor_field[ 3];
  double * const gmuf = eo_spinor_field[ 4];
  
  /***********************************************************
   * gather scalar products for all eigenvectors
   ***********************************************************/
  double _Complex *** ct = init_5level_ztable (T, 3, momentum_number, 4, nev )
  if ( ct == NULL )  {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee_ct] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * loop on eigenvectors
   ***********************************************************/
  for ( unsigned int inev = 0; inev < nev; inev++ ) {
    // set V
    memcpy ( v, eo_evecs_field[inev], sizeof_eo_spinor_field );

    // set Xbar V
    memcpy ( eo_spinor_work[0], v, sizeof_eo_spinor_field );
    X_clover_eo ( xv, eo_spinor_work[0], gauge_field, mzzinv[1][0]);

    // set W
    //   NOTE: W, NOT Wtilde
    memcpy ( w, v,  sizeof_eo_spinor_field );
    memcpy ( eo_spinor_work[0],  xv, sizeof_eo_spinor_field );
    C_clover_from_Xeo ( w, eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1]);

    // calculate w = Cbar v
    C_clover_oo ( w, v, gauge_field, eo_spinor_work[0], mzz[1][1], mzzinv[1][0] );

    // set X W
    memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
    X_clover_eo ( xw, eo_spinor_work[0], gauge_field, mzzinv[0][0]);


    /***********************************************************
     * loop on directions
     ***********************************************************/
    for ( int imu = 0; imu < 4; imu++ ) {

      double _Complex * p = init_1level_ztable ( T );


      /***********************************************************
       * (2) ( XV )^+ ( g5 Gmufwd W ) delta_x,y
       * on even sublattice
       ***********************************************************/
      memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 0 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, xv, gmuf, 0 );
      for ( int it = 0; it < T; it++ ) {
        for ( int imom = 0; imom < momentum_number; imom++ ) {
          ct[it][1][imom][imu][inev] += p[it];
        }
      }

      /***********************************************************
       * (4) ( XV )_x^+ ( g5 Gmubwd W )_x delta_x-mu,y 
       * on even sublattice
       * x - y = + mu
       ***********************************************************/
      memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 1, gauge_field, 0 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, xv, gmuf, 0 );
      for ( int imom = 0; imom < momentum_number; imom++ ) {
        double const q[4] = { 0, 
          2.*M_PI*momentum_list[imom][0] / (double)LX_global,
          2.*M_PI*momentum_list[imom][1] / (double)LY_global,
          2.*M_PI*momentum_list[imom][2] / (double)LZ_global };

        double _Complex const ephase = cexp ( +q[mu] * I );
        int const dt  = ( imu == 0 );
        int const idt = dt + 1;
        
        for ( int it = 0; it < T; it++ ) {
          ct[it][idt][imom][imu][inev] += p[it] * ephase;
        }
      }

      /***********************************************************
       * (5) -V_y^+ ( g5 Gmufwd XW )_y delta_y+mu,x 
       * on odd sublattice
       * x - y = +mu
       ***********************************************************/
      memcpy ( eo_spinor_work[0], xw, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 1 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, xv, gmuf, 1 );
      for ( int imom = 0; imom < momentum_number; imom++ ) {
        double const q[4] = { 0, 
          2.*M_PI*momentum_list[imom][0] / (double)LX_global,
          2.*M_PI*momentum_list[imom][1] / (double)LY_global,
          2.*M_PI*momentum_list[imom][2] / (double)LZ_global };

        double _Complex const ephase = cexp ( +q[mu] * I );
        int const dt  = ( imu == 0 );
        int const idt = dt + 1;
        
        for ( int it = 0; it < T; it++ ) {
          ct[it][idt][imom][imu][inev] += -p[it] * ephase;
        }
      }


      /***********************************************************
       * (7) -(V^+ g5 Gmubwd )_y XW_x delta_yx 
       *   = + ( g5 Gmufwd V )_y^+ XW_x delta_yx
       * on even sublattice
       * x - y = 0
       ***********************************************************/
      memcpy ( eo_spinor_work[0], v, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 0 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, gmuf, xw, 0 );
      for ( int it = 0; it < T; it++ ) {
        for ( int imom = 0; imom < momentum_number; imom++ ) {
          ct[it][1][imom][imu][inev] += p[it];
        }
      }

      /***********************************************************
       * (8) + (V^+ g5 Gmubwd )_y X_yx  W_x 
       *   = - ( g5 Gmufwd V )^+_y X_yx W_x
       ***********************************************************/
      // Mbar_ee^-1 g5 Gmufwd V
      M_clover_zz_inv_matrix ( gmuf, gmuf, mzzinv[1][0] );


      for ( int imom = 0; imom < momentum_number; imom++ ) {

          ct[it][imom][imu][inev] += p[it];
      }



      // ( g5 Gmufwd XV )^+ W
      memcpy ( eo_spinor_work[0], xv, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 1 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, gmuf, w, 1 );
      for ( int it = 0; it < T; it++ ) ct[i][imu][inev] -= p[it];



      // ( V )^+ ( g5 Gmufwd XW )
      memcpy ( eo_spinor_work[0], xw, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 1 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, v, gmuf, 1 );
      for ( int it = 0; it < T; it++ ) ct[i][imu][inev] -= p[it];

      fini_1level_ztable ( &p );

    }  // end of loop on mu

  }  // end of loop on nev

  fini_2level_dtable ( &eo_spinor_field );
  fini_2level_dtable ( &eo_spinor_work );


  // process with time ray id 0 and time slice id 0
  if ( io_proc >= 1 ) {

    for ( int it = 0; it < T; it++ ) {

      for ( int imu = 0; imu < 4; imu++ ) {

        char aff_tag[200];
        sprintf ( aff_tag, "/%s/mu%d/t%.2d", tag, imu, (it+g_proc_coords[0]*T) );

        affdir = aff_writer_mkpath( affw, affn, aff_tag );

        exitstatus = aff_node_put_complex ( affw, affdir, ct[it][imu], nev );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[contract_cvc_tensor_eo_lm_mee_ct] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(15);
        }
          
      }  // end of loop on directions mu

    }  // end of loop on timeslices

  }  // end of if io_proc >= 1

  fini_3level_ztable ( &ct );

  return(0);
 
}  // end of contract_cvc_tensor_eo_lm_mee_ct
