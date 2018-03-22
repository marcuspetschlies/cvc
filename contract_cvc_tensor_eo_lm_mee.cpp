/***********************************************************
 * contract_cvc_tensor_eo_lm_mee.cpp
 *
 * Mi 21. Mär 07:21:36 CET 2018
 ***********************************************************/

/***********************************************************
 * contractions for Mee part of eo-precon lm cvc - cvc tensor
 ***********************************************************/
int const contract_cvc_tensor_eo_lm_mee (
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
      fprintf(stderr, "[contract_cvc_tensor_eo_lm_mee] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
#endif
  }

  // allocate auxilliary spinor fields and spinor work (with halo)
  double ** eo_spinor_field = init_2level_dtable ( 18, _GSI(Vhalf) );
  if ( eo_spinor_field == NULL ) {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double ** eo_spinor_work = init_2level_dtable ( 4, _GSI( (VOLUME+RAND)/2 ) );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double * const v = eo_spinor_field[ 0];
  double * const w = eo_spinor_field[ 1];
  
  // Gamma_mu^F v
  double * const gmufv[4] = { eo_spinor_field[ 2], eo_spinor_field[ 3], eo_spinor_field[ 4], eo_spinor_field[ 5] };
  // Gamma_mu^B v
  double * const gmubv[4] = { eo_spinor_field[ 6], eo_spinor_field[ 7], eo_spinor_field[ 8], eo_spinor_field[ 9] };
  // Gamma_mu^F w
  double * const gmufw[4] = { eo_spinor_field[10], eo_spinor_field[11], eo_spinor_field[12], eo_spinor_field[13] };
  // Gamma_mu^B w
  double * const gmubw[4] = { eo_spinor_field[14], eo_spinor_field[15], eo_spinor_field[16], eo_spinor_field[17] };

  /***********************************************************
   * gather scalar products for all eigenvectors
   ***********************************************************/
  double _Complex ***** meesp = NULL;
  if (g_ts_id == 0 ) {
    //                           mu x nu x timeslice x eigenvector x 4 scalar products
    meesp = init_5level_ztable ( 4,   4,   T,          nev,          4 );
    if ( meesp == NULL )  {
      fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
      return(2);
    }
  }


  /***********************************************************
   * loop on eigenvectors
   ***********************************************************/
  for ( unsigned int inev = 0; inev < nev; inev++ ) {
    // set V
    memcpy ( v, eo_evecs_field[inev], sizeof_eo_spinor_field );

    // calculate w = Cbar v
    C_clover_oo ( w, v, gauge_field, eo_spinor_work[0], mzz[1][1], mzzinv[1][0] );


    /***********************************************************
     * loop on directions
     ***********************************************************/
    for ( int imu = 0; imu < 4; imu++ ) {

      memcpy ( eo_spinor_work[0], v, sizeof_eo_spinor_field );
      /* Gmufwdr V */
      apply_cvc_vertex_eo( gmufv[imu], eo_spinor_work[0], imu, 0, gauge_field, 0 );
      /* Gmubwdr V */
      apply_cvc_vertex_eo( gmubv[imu], eo_spinor_work[0], imu, 1, gauge_field, 0 );

      memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
      /* Gmufwdr W */
      apply_cvc_vertex_eo( gmufw[imu], eo_spinor_work[0], imu, 0, gauge_field, 0 );
      /* Gmubwdr W */
      apply_cvc_vertex_eo( gmubw[imu], eo_spinor_work[0], imu, 1, gauge_field, 0 );
    }

    g5_phi ( eo_spinor_field[2], 8 * Vhalf );

    for ( int imu = 0; imu < 4; imu++ ) {
      M_clover_zz_inv_matrix ( gmufw[imu], gmufw[imu], mzzinv[0][0] );
      M_clover_zz_inv_matrix ( gmubw[imu], gmubw[imu], mzzinv[0][0] );
    }

    double _Complex ** p = init_2level_ztable ( 2, T );
    if ( p == NULL ) {
      fprintf( stderr, "[contract_cvc_tensor_eo_lm_mee] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    /***********************************************************
     * scalar products
     ***********************************************************/
    for ( unsigned int imu = 0; imu < 4; imu++ ) {
      for ( unsigned int inu = 0; inu < 4; inu++ ) {

        // initialize all p's to zero
        memset ( p[0], 0, 2*T*sizeof( double _Complex ) );

        // the 8 scalar products

        eo_spinor_spatial_scalar_product_co( p[0], gmubv[imu], gmufw[inu], 0 );
        eo_spinor_spatial_scalar_product_co( p[1], gmufv[inu], gmubw[imu], 0 );
        if ( g_ts_id == 0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int it = 0; it < T; it++ ) meesp[imu][inu][it][inev][0] = p[0][it] + p[1][it];
        }

        eo_spinor_spatial_scalar_product_co( p[0], gmufv[imu], gmufw[inu], 0 );
        eo_spinor_spatial_scalar_product_co( p[1], gmufv[inu], gmufw[imu], 0 );
        if ( g_ts_id == 0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int it = 0; it < T; it++ ) meesp[imu][inu][it][inev][1] = p[0][it] + p[1][it];
        }

        eo_spinor_spatial_scalar_product_co( p[0], gmubv[inu], gmufw[imu], 0 );
        eo_spinor_spatial_scalar_product_co( p[1], gmufv[imu], gmubw[inu], 0 );
        if ( g_ts_id == 0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int it = 0; it < T; it++ ) meesp[imu][inu][it][inev][2] = p[0][it] + p[1][it];
        }

        eo_spinor_spatial_scalar_product_co( p[0], gmubv[inu], gmubw[imu], 0 );
        eo_spinor_spatial_scalar_product_co( p[1], gmubv[imu], gmubw[inu], 0 );
        if ( g_ts_id == 0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int it = 0; it < T; it++ ) meesp[imu][inu][it][inev][3] = p[0][it] + p[1][it];
        }
      }
    }

    fini_2level_ztable ( &p );

  }  // end of loop on nev

  fini_2level_dtable ( &eo_spinor_field );
  fini_2level_dtable ( &eo_spinor_work );


  /***********************************************************
   * collect, multiply with momentum, write
   ***********************************************************/
  if ( g_ts_id == 0 ) {
    double _Complex *** meesp_buffer = NULL;
    //if ( g_tr_id == 0 ) {
      meesp_buffer = init_3level_ztable ( T_global, nev, 4 );
      if ( meesp_buffer == NULL ) {
        fprintf ( stderr, "[contract_cvc_tensor_eo_lm_mee] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
        return(6);
      }
    //}

    for ( unsigned int imu = 0; imu < 4; imu++ ) {
    for ( unsigned int inu = 0; inu < 4; inu++ ) {

#ifdef HAVE_MPI
#  if ( defined PARALLELTX ) || ( defined PARALLELTXY ) || ( defined PARALLELTXYZ ) 
      // gather at g_tr_id = 0 and g_ts_id == 0
      // fprintf ( stdout, "# [contract_cvc_tensor_eo_lm_mee] proc %d does send/recv %s %d\n", g_cart_id, __FILE__, __LINE__ );
      exitstatus = MPI_Gather( meesp[imu][inu][0][0], 2*T*nev*4, MPI_DOUBLE, meesp_buffer[0][0], 2*T*nev*4, MPI_DOUBLE, 0, g_tr_comm );
      // exitstatus = MPI_Allgather( meesp[imu][inu][0][0], 2*T*nev*4, MPI_DOUBLE, meesp_buffer[0][0], 2*T*nev*4, MPI_DOUBLE, g_tr_comm );

#  else
      exitstatus = MPI_Gather( meesp[imu][inu][0][0], 2*T*nev*4, MPI_DOUBLE, meesp_buffer[0][0], 2*T*nev*4, MPI_DOUBLE, 0, g_cart_grid );
#  endif
      if ( exitstatus != MPI_SUCCESS ) {
        fprintf ( stderr, "[contract_cvc_tensor_eo_lm_mee] Error from MPI_Gather %s %d\n", __FILE__, __LINE__ );
        return(7);
      } else {
        fprintf ( stdout, "# [contract_cvc_tensor_eo_lm_mee] MPI_Gather was successful %s %d\n", __FILE__, __LINE__ );
      }
#else
      memcpy ( meesp_buffer[0][0], meesp[imu][inu][0][0], 4 * T_global * nev * sizeof(double _Complex) );
#endif

      // process with time ray id 0 and time slice id 0
      if ( g_tr_id == 0 ) {

        int const dt[4] = { (imu == 0), 0, -(inu == 0), (imu == 0) - (inu == 0 ) };
        if ( g_verbose > 3 ) fprintf ( stdout, "# [contract_cvc_tensor_eo_lm_mee] dt values %3d %3d %3d %3d\n", dt[0], dt[1], dt[2], dt[3] );

        // loop on momenta
        for ( unsigned int imom = 0; imom < momentum_number; imom++ ) {

          double const mom[4] = {
            0, 
            2. * M_PI * momentum_list[imom][0] / LX_global,
            2. * M_PI * momentum_list[imom][1] / LY_global,
            2. * M_PI * momentum_list[imom][2] / LZ_global };
          double _Complex const phase[4] = {
            cos ( mom[imu] ) + I * sin( mom[imu] ),
            1.,
            cos ( mom[inu] ) - I * sin( mom[inu] ),
            cos ( mom[imu] - mom[inu] ) + I * sin( mom[imu] - mom[inu] ) };

          double _Complex *** meesp_out = init_3level_ztable ( T_global, 3, nev );
          if ( meesp_out == NULL ) {
            fprintf ( stderr, "[contract_cvc_tensor_eo_lm_mee] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
            return(8);
          }

          // loop on timeslices
          for ( int it = 0; it < T_global; it++ ) {
 
            int const iit = (it + ( inu==0) ) % T_global;
            int const tval[4] = { it, it, iit, iit };
            if ( g_verbose > 4 ) fprintf ( stdout, "# [contract_cvc_tensor_eo_lm_mee] t values %3d %3d %3d %3d\n", tval[0], tval[1], tval[2], tval[3] );
        
            // loop on eigenvectors
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
            for ( unsigned int iev = 0; iev < nev; iev++ ) {

              // combine the 4 scalar products with corresponding phase; sort according to dt
              for ( unsigned int isp = 0; isp < 4; isp++ )
              {
                meesp_out[it][dt[isp]+1][iev] += meesp_buffer[tval[isp]][iev][isp] * phase[isp];
              }

            }  // end of loop on eigenvectors
             
            // loop on dt
            for ( int idt = 0; idt < 3; idt++ ) {
              char aff_tag[200];
              sprintf ( aff_tag, "/%s/mu%d/nu%d/px%.2dpy%.2dpz%.2d/t%.2d/dt%d",
                  tag, imu, inu, momentum_list[imom][0], momentum_list[imom][1], momentum_list[imom][2], it, idt-1);

              affdir = aff_writer_mkpath( affw, affn, aff_tag );

              exitstatus = aff_node_put_complex ( affw, affdir, meesp_out[it][idt], nev );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[contract_cvc_tensor_eo_lm_mee] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                return(15);
              }
          
            }  // end of loop on dt

          }  // end of loop on global timeslices

        
          fini_3level_ztable ( &meesp_out );
        }  // end of loop on 3-momenta


      }  // end of if g_tr_id == 0
#if 0
#endif  // of if 0
  
    }}  // end of loop on nu, mu

    // if ( g_tr_id == 0 ) {
      fini_3level_ztable ( &meesp_buffer );
    // }

  }  // end of if g_ts_id == 0

  if (g_ts_id == 0 ) fini_5level_ztable ( &meesp );

  return(0);
 
}  // end of contract_cvc_tensor_eo_lm_mee
