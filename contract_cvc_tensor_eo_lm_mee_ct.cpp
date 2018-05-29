/***********************************************************
 * contract_cvc_tensor_eo_lm_mee_ct.cpp
 *
 * Mi 21. Mär 07:21:36 CET 2018
 ***********************************************************/

/***********************************************************
 * contractions for Mee part of eo-precon lm cvc - cvc tensor
 ***********************************************************/
int contract_cvc_tensor_eo_lm_mee_ct (
    double ** const eo_evecs_field, unsigned int const nev,
    double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2],
    struct AffWriter_s * affw, char * const tag, 
    int (* const momentum_list)[3], unsigned int const momentum_number,
    unsigned int const io_proc
) {

  unsigned int const Vhalf            = VOLUME / 2;
  size_t const sizeof_eo_spinor_field = _GSI( Vhalf    ) * sizeof(double);
  double const TWO_MPI_OVER_L[3]      = { 2. * M_PI / LX_global, 2. * M_PI / LY_global, 2. * M_PI / LZ_global };

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
  double ** eo_spinor_field = init_2level_dtable ( 21, _GSI(Vhalf) );
  if ( eo_spinor_field == NULL ) {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee_ct] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double ** eo_spinor_work = init_2level_dtable ( 2, _GSI( (VOLUME+RAND)/2 ) );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee_ct] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double * const v    = eo_spinor_field[ 0];
  double * const w    = eo_spinor_field[ 1];
  double * const xv   = eo_spinor_field[ 2];
  double * const xw   = eo_spinor_field[ 3];
  double * const gmufv[4] = { eo_spinor_field[ 4], eo_spinor_field[ 5], eo_spinor_field[ 6], eo_spinor_field[ 7] };
  double * const gmubv[4] = { eo_spinor_field[ 8], eo_spinor_field[ 9], eo_spinor_field[10], eo_spinor_field[11] };
  double * const gmufw[4] = { eo_spinor_field[12], eo_spinor_field[13], eo_spinor_field[14], eo_spinor_field[15] };
  double * const gmubw[4] = { eo_spinor_field[16], eo_spinor_field[17], eo_spinor_field[18], eo_spinor_field[19] };
  double * const aux  = eo_spinor_field[20];
  
  /***********************************************************
   * gather scalar products for all eigenvectors
   ***********************************************************/
  double _Complex ***** ct = init_5level_ztable (T_global, 5, momentum_number, 4, nev );
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
     * calculate shifted fields
     ***********************************************************/
    for ( int imu = 0; imu < 4; imu++ ) {
      // Gmufwd V, even target field
      memcpy ( eo_spinor_work[0], v, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmufv[imu], eo_spinor_work[0], imu, 0, gauge_field, 0 );
      
      // Gmubwd V, even target field
      memcpy ( eo_spinor_work[0], v, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmubv[imu], eo_spinor_work[0], imu, 1, gauge_field, 0 );
      
      // Gmufwd W, even target field
      memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmufw[imu], eo_spinor_work[0], imu, 0, gauge_field, 0 );
      
      // Gmubwd W, even target field
      memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmufw[imu], eo_spinor_work[0], imu, 1, gauge_field, 0 );
    }  // end of loop on directions

    /***********************************************************
     * loop on directions
     ***********************************************************/
    for ( int imu = 0; imu < 4; imu++ ) {

      double _Complex * p = init_1level_ztable ( T );
      double _Complex * p2 = init_1level_ztable ( T );

      /***********************************************************
       * (1) 
       ***********************************************************/
      M_clover_zz_inv_matrix ( aux, gmufw[imu], mzzinv[0][0] );
      g5_phi ( aux, Vhalf );

      for ( int ilambda = 0; ilambda < 4; ilambda++ ) {
        eo_spinor_spatial_scalar_product_co( p,  gmufv[ilambda], aux, 0 );
        eo_spinor_spatial_scalar_product_co( p2, gmubv[ilambda], aux, 0 );

        int const dtf = +( ilambda == 0 );
        int const idtf = dtf + 2;
        int const dtb = -( ilambda == 0 );
        int const idtb = dtb + 2;

        for ( int imom = 0; imom < momentum_number; imom++ ) {
          double const q[4] = { 0,
            TWO_MPI_OVER_L[0] * momentum_list[imom][0],
            TWO_MPI_OVER_L[1] * momentum_list[imom][1],
            TWO_MPI_OVER_L[2] * momentum_list[imom][2] };

          // lambda fwd
          double _Complex ephase = cexp ( +q[ilambda]*I );
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0]*T + ( ilambda == 0 ) + T_global ) % T_global;
            ct[x0][idtf][imom][imu][inev] -= p[it] * ephase;
          }

          // lambda bwd
          ephase = cexp ( -q[ilambda]*I );
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0]*T - ( ilambda == 0 ) + T_global ) % T_global;
            ct[x0][idtb][imom][imu][inev] += p2[it] * ephase;
          }
        }
      }

      /***********************************************************
       * (2) 
       ***********************************************************/
      memcpy ( aux, gmufw[imu], sizeof_eo_spinor_field );
      g5_phi ( aux, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, xv, aux, 0 );
      int dt = 0;
      int idt = dt + 2;
      for ( int it = 0; it < T; it++ ) {
        int const x0 = ( it + g_proc_coords[0] * T);
        for ( int imom = 0; imom < momentum_number; imom++ ) {
          ct[x0][idt][imom][imu][inev] += p[it];
        }
      }

      /***********************************************************
       * (3)
       ***********************************************************/
      M_clover_zz_inv_matrix ( aux, gmubw[imu], mzzinv[0][0] );
      g5_phi ( aux, Vhalf );

      for ( int ilambda = 0; ilambda < 4; ilambda++ ) {
        eo_spinor_spatial_scalar_product_co( p,  gmufv[ilambda], aux, 0 );
        eo_spinor_spatial_scalar_product_co( p2, gmubv[ilambda], aux, 0 );

        int const dtf = +( ilambda == 0 ) + ( imu == 0 );
        int const idtf = dtf + 2;
        int const dtb = -( ilambda == 0 ) + ( imu == 0 );
        int const idtb = dtb + 2;

        for ( int imom = 0; imom < momentum_number; imom++ ) {
          double const q[4] = { 0,
            TWO_MPI_OVER_L[0] * momentum_list[imom][0],
            TWO_MPI_OVER_L[1] * momentum_list[imom][1],
            TWO_MPI_OVER_L[2] * momentum_list[imom][2] };

          // lambda fwd
          double _Complex ephase = cexp ( (+q[ilambda] + q[imu] ) * I );
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0]*T + ( ilambda == 0 ) + T_global ) % T_global;
            ct[x0][idtf][imom][imu][inev] -= p[it] * ephase;
          }

          // lambda bwd
          ephase = cexp ( ( -q[ilambda] + q[imu] ) * I );
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0]*T - ( ilambda == 0 ) + T_global ) % T_global;
            ct[x0][idtb][imom][imu][inev] += p2[it] * ephase;
          }
        }
      }

      /***********************************************************
       * (4)
       ***********************************************************/
      memcpy ( aux, gmubw[imu], sizeof_eo_spinor_field );
      g5_phi ( aux, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, xv, aux, 0 );
     
      dt  = +( imu == 0 );
      idt = dt + 2;

      for ( int imom = 0; imom < momentum_number; imom++ ) {
        double const q[4] = { 0, 
          TWO_MPI_OVER_L[0] * momentum_list[imom][0],
          TWO_MPI_OVER_L[1] * momentum_list[imom][1],
          TWO_MPI_OVER_L[2] * momentum_list[imom][2] };

        double _Complex const ephase = cexp ( +q[imu] * I );
        for ( int it = 0; it < T; it++ ) {
          int const x0 = ( it + g_proc_coords[0]*T );
          ct[x0][idt][imom][imu][inev] += p[it] * ephase;
        }
      }

      /***********************************************************
       * (5) 
       ***********************************************************/
      memcpy ( aux, gmubv[imu], sizeof_eo_spinor_field );
      g5_phi ( aux, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, aux, xw, 0 );
      dt  = ( imu == 0 );
      idt = dt + 2;
      for ( int imom = 0; imom < momentum_number; imom++ ) {
        double const q[4] = { 0, 
          TWO_MPI_OVER_L[0] * momentum_list[imom][0],
          TWO_MPI_OVER_L[1] * momentum_list[imom][1],
          TWO_MPI_OVER_L[2] * momentum_list[imom][2] };

        double _Complex const ephase = cexp ( +q[imu] * I );
        for ( int it = 0; it < T; it++ ) {
          int const x0 = ( it + g_proc_coords[0]*T );
          ct[x0][idt][imom][imu][inev] += p[it] * ephase;
        }
      }  // end of loop on momenta

      /***********************************************************
       * (6)
       ***********************************************************/
      M_clover_zz_inv_matrix ( aux, gmubv[imu], mzzinv[1][0] );
      g5_phi ( aux, Vhalf );

      for ( int ilambda = 0; ilambda < 4; ilambda++ ) {
        eo_spinor_spatial_scalar_product_co( p,  gmufw[ilambda], aux, 0 );
        eo_spinor_spatial_scalar_product_co( p2, gmubw[ilambda], aux, 0 );

        int const dtf = +( ilambda == 0 ) + ( imu == 0 );
        int const idtf = dtf + 2;
        int const dtb = -( ilambda == 0 ) + ( imu == 0 );
        int const idtb = dtb + 2;

        for ( int imom = 0; imom < momentum_number; imom++ ) {
          double const q[4] = { 0,
            TWO_MPI_OVER_L[0] * momentum_list[imom][0],
            TWO_MPI_OVER_L[1] * momentum_list[imom][1],
            TWO_MPI_OVER_L[2] * momentum_list[imom][2] };

          // lambda fwd
          double _Complex ephase = cexp ( (+q[ilambda] + q[imu] ) * I );
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0]*T + ( ilambda == 0 ) + T_global ) % T_global;
            ct[x0][idtf][imom][imu][inev] -= p[it] * ephase;
          }

          // lambda bwd
          ephase = cexp ( ( -q[ilambda] + q[imu] ) * I );
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0]*T - ( ilambda == 0 ) + T_global ) % T_global;
            ct[x0][idtb][imom][imu][inev] += p2[it] * ephase;
          }
        }
      }
  
      /***********************************************************
       * (7) 
       ***********************************************************/
      memcpy ( aux, gmufv[imu], sizeof_eo_spinor_field );
      g5_phi ( aux, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, aux, xw, 0 );
      dt  = 0;
      idt = dt + 2;
      for ( int it = 0; it < T; it++ ) {
        for ( int imom = 0; imom < momentum_number; imom++ ) {
          int const x0 = ( it + g_proc_coords[0]*T );
          ct[x0][idt][imom][imu][inev] += p[it];
        }
      }

      /***********************************************************
       * (8)
       ***********************************************************/
      // Mbar_ee^-1 g5 Gmufwd V
      M_clover_zz_inv_matrix ( aux, gmufv[imu], mzzinv[1][0] );
      g5_phi ( aux, Vhalf );

      for ( int ilambda = 0; ilambda < 4; ilambda++ ) {
        eo_spinor_spatial_scalar_product_co( p,  aux, gmufw[ilambda], 0 );
        eo_spinor_spatial_scalar_product_co( p2, aux, gmubw[ilambda], 0 );

        int const dtf  = +( ilambda ==  0 );
        int const idtf = dtf + 2;
        int const dtb  = -( ilambda ==  0 );
        int const idtb = dtb + 2;

        for ( int imom = 0; imom < momentum_number; imom++ ) {
          double const q[4] = { 0,
            TWO_MPI_OVER_L[0] * momentum_list[imom][0],
            TWO_MPI_OVER_L[1] * momentum_list[imom][1],
            TWO_MPI_OVER_L[2] * momentum_list[imom][2] };

          double _Complex ephase = cexp ( +q[ilambda] * I );
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0] *T + ( ilambda == 0 ) + T_global ) % T_global;
            ct[x0][idtf][imom][imu][inev] -= p[it] * ephase;
          }
          ephase = cexp ( -q[ilambda] * I );
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0] *T - ( ilambda == 0 ) + T_global ) % T_global;
            ct[x0][idtb][imom][imu][inev] += p2[it] * ephase;
          }
        }
      }

      fini_1level_ztable ( &p );
      fini_1level_ztable ( &p2 );

    }  // end of loop on mu

  }  // end of loop on nev

  fini_2level_dtable ( &eo_spinor_field );
  fini_2level_dtable ( &eo_spinor_work );


  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * process with time ray id 0 and time slice id 0
   ***********************************************************/
  if ( io_proc >= 1 ) {

#ifdef HAVE_MPI
    int const items = T_global * 5 * momentum_number * 4 * nev;
    size_t const bytes = items * sizeof ( double _Complex );
    
    double _Complex * buffer = init_1level_ztable ( items );
    if ( buffer  == NULL )  {
      fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee_ct] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__);
      return(2);
    }

    memcpy ( buffer , ct[0][0][0][0], bytes );

    exitstatus = MPI_Allreduce( buffer, ct[0][0][0][0], 2*items, MPI_DOUBLE, MPI_SUM, g_tr_comm );
    if ( exitstatus != MPI_SUCCESS ) {
      fprintf ( stderr, "[] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(1);
    }

    fini_1level_ztable ( &buffer );
#endif

  }  // end of if io_proc >= 1 

  /***********************************************************
   * process io_proc = 2 write to file
   ***********************************************************/
  if ( io_proc == 2 ) {

    for ( int it = 0; it < T_global; it++ ) {

      for ( int idt = 0; idt < 5; idt++ ) {
        int const dt = idt - 2;

        for ( int imom = 0; imom < momentum_number; imom++ ) {

          for ( int imu = 0; imu < 4; imu++ ) {

            char aff_tag[200];
            sprintf ( aff_tag, "/%s/mu%d/px%.2dpy%.2dpz%.2d/t%.2d/dt%.2d", tag, imu,
               momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], it, dt );

            affdir = aff_writer_mkpath( affw, affn, aff_tag );

            exitstatus = aff_node_put_complex ( affw, affdir, ct[it][idt][imom][imu], nev );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[contract_cvc_tensor_eo_lm_mee_ct] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(15);
            }
            
          }  // end of loop on directions mu

        }  // end of loop on momenta

      }  // end of loop on dt

    }  // end of loop on timeslices

  }  // end of if io_proc == 2

  fini_5level_ztable ( &ct );

  return(0);
 
}  // end of contract_cvc_tensor_eo_lm_mee_ct
