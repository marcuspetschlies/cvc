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
  double * const gfv[4] = { eo_spinor_field[ 2], eo_spinor_field[ 3], eo_spinor_field[ 4], eo_spinor_field[ 5] };
  // Gamma_mu^B v
  double * const gbv[4] = { eo_spinor_field[ 6], eo_spinor_field[ 7], eo_spinor_field[ 8], eo_spinor_field[ 9] };
  // Gamma_mu^F w
  double * const gfw[4] = { eo_spinor_field[10], eo_spinor_field[11], eo_spinor_field[12], eo_spinor_field[13] };
  // Gamma_mu^B w
  double * const gbw[4] = { eo_spinor_field[14], eo_spinor_field[15], eo_spinor_field[16], eo_spinor_field[17] };

  /***********************************************************
   * gather scalar products for all eigenvectors
   ***********************************************************/
  double _Complex ***** meesp = NULL;
  
  /***********************************************************
   * NOTE: this is a large array for typical lattice size
   * and O(10^3) low modes, but ONLY 1 allocation per
   * timeslice
   ***********************************************************/
  if (g_ts_id == 0 ) {
    //                           mu x nu x timeslice x eigenvector x 4 scalar products
    meesp = init_5level_ztable ( 4,   4,   T_global,   nev,          4 );
    if ( meesp == NULL )  {
      fprintf(stderr, "# [contract_cvc_tensor_eo_lm_mee] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
      return(2);
    }
  }


  /***********************************************************
   * loop on eigenvectors
   ***********************************************************/
  for ( unsigned int inev = 0; inev < nev; inev++ )
  // for ( unsigned int inev = 0; inev <= 0; inev++ )
  {
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
      apply_cvc_vertex_eo( gfv[imu], eo_spinor_work[0], imu, 0, gauge_field, 0 );
      /* Gmubwdr V */
      apply_cvc_vertex_eo( gbv[imu], eo_spinor_work[0], imu, 1, gauge_field, 0 );

      memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
      /* Gmufwdr W */
      apply_cvc_vertex_eo( gfw[imu], eo_spinor_work[0], imu, 0, gauge_field, 0 );
      /* Gmubwdr W */
      apply_cvc_vertex_eo( gbw[imu], eo_spinor_work[0], imu, 1, gauge_field, 0 );
    }

    // multiply gfv and gbv fields by g5
    g5_phi ( eo_spinor_field[2], 8 * Vhalf );

    // multiply gfw and gbw fields by M_ee^-1
    for ( int imu = 0; imu < 4; imu++ ) {
      M_clover_zz_inv_matrix ( gfw[imu], gfw[imu], mzzinv[0][0] );
      M_clover_zz_inv_matrix ( gbw[imu], gbw[imu], mzzinv[0][0] );
    }

    double _Complex ** p = init_2level_ztable ( 2, T );
    if ( p == NULL ) {
      fprintf( stderr, "[contract_cvc_tensor_eo_lm_mee] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    /***********************************************************
     * scalar products
     ***********************************************************/
    for ( unsigned int imu = 0; imu < 4; imu++ ) 
    // for ( unsigned int imu = 0; imu <= 0; imu++ ) 
    {
      for ( unsigned int inu = 0; inu < 4; inu++ )
      // for ( unsigned int inu = 0; inu <= 0; inu++ )
      {


        // the 8 scalar products


        memset ( p[0], 0, 2*T*sizeof( double _Complex ) );
        // (1)
        eo_spinor_spatial_scalar_product_co( p[0], gbv[imu], gfw[inu], 0 );

        // (2)
        eo_spinor_spatial_scalar_product_co( p[1], gfv[inu], gbw[imu], 0 );

#if 0
#endif  // of if 0

        if ( g_ts_id == 0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0]*T - ( imu == 0 ) + T_global ) % T_global;
            meesp[imu][inu][x0][inev][0] = p[0][it] + p[1][it];
          }
        }


        memset ( p[0], 0, 2*T*sizeof( double _Complex ) );
        // (3)
        eo_spinor_spatial_scalar_product_co( p[0], gbv[inu], gfw[imu], 0 );


        // (4)  
        eo_spinor_spatial_scalar_product_co( p[1], gfv[imu], gbw[inu], 0 );
#if 0
#endif  // of if 0

        if ( g_ts_id == 0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int it = 0; it < T; it++ ) {
            int const x0 = it + g_proc_coords[0] * T;
            meesp[imu][inu][x0][inev][1] = p[0][it] + p[1][it];
          }
        }



        memset ( p[0], 0, 2*T*sizeof( double _Complex ) );
        // (5)
        eo_spinor_spatial_scalar_product_co( p[0], gfv[imu], gfw[inu], 0 );

        //for ( int ix = 0; ix < Vhalf; ix++) {
        //  for ( int isc = 0; isc < 12; isc++ ) {
        //    fprintf ( stdout, "# [contract_cvc_tensor_eo_lm_mee] nev %3d mu %d nu %d x %4d sc %2d  gfv %25.16e %25.16e  gfw %25.16e %25.16e\n",
        //        inev, imu, inu, ix, isc, gfv[imu][_GSI(ix)+2*isc], gfv[imu][_GSI(ix)+2*isc+1], gfw[inu][_GSI(ix)+2*isc], gfw[inu][_GSI(ix)+2*isc+1] );
        //  }
        //}


        // (6)
        eo_spinor_spatial_scalar_product_co( p[1], gfv[inu], gfw[imu], 0 );

        if ( g_ts_id == 0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int it = 0; it < T; it++ ) {
            int const x0 = it + g_proc_coords[0] * T;
            meesp[imu][inu][x0][inev][2] = p[0][it] + p[1][it];
          }
        }
#if 0
#endif  // of if 0



        memset ( p[0], 0, 2*T*sizeof( double _Complex ) );
        // (7)
        eo_spinor_spatial_scalar_product_co( p[0], gbv[inu], gbw[imu], 0 );


        // (8)
        eo_spinor_spatial_scalar_product_co( p[1], gbv[imu], gbw[inu], 0 );


        if ( g_ts_id == 0 ) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int it = 0; it < T; it++ ) {
            int const x0 = ( it + g_proc_coords[0]*T - ( imu == 0 ) + T_global ) % T_global;
            meesp[imu][inu][x0][inev][3] = p[0][it] + p[1][it];
          }
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
      // fprintf ( stdout, "# [contract_cvc_tensor_eo_lm_mee] proc %d does send/recv %s %d\n", g_cart_id, __FILE__, __LINE__ );
      
      exitstatus = MPI_Allreduce ( meesp[imu][inu][0][0], meesp_buffer[0][0], 2 * T_global * nev * 4, MPI_DOUBLE, MPI_SUM,  g_tr_comm );
#  else
      exitstatus = MPI_Allreduce ( meesp[imu][inu][0][0], meesp_buffer[0][0], 2 * T_global * nev * 4, MPI_DOUBLE, MPI_SUM,  g_cart_grid );
#  endif
      if ( exitstatus != MPI_SUCCESS ) {
        fprintf ( stderr, "[contract_cvc_tensor_eo_lm_mee] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
        return(7);
      }
#else
      memcpy ( meesp_buffer[0][0], meesp[imu][inu][0][0], 4 * T_global * nev * sizeof(double _Complex) );
#endif

      /***********************************************************
       * process with time ray id 0 and time slice id 0
       ***********************************************************/
      if ( g_tr_id == 0 ) {

        int const dt[4] = { 
            -( imu == 0 ),
            +( inu == 0 ),
            0, 
            -(imu == 0) + (inu == 0 ) 
        };
        if ( g_verbose > 3 ) fprintf ( stdout, "# [contract_cvc_tensor_eo_lm_mee] dt values %3d %3d %3d %3d\n", dt[0], dt[1], dt[2], dt[3] );

        /***********************************************************
         * loop on momenta
         ***********************************************************/
        for ( unsigned int imom = 0; imom < momentum_number; imom++ ) {

          double const mom[4] = {
            0, 
            2. * M_PI * momentum_list[imom][0] / LX_global,
            2. * M_PI * momentum_list[imom][1] / LY_global,
            2. * M_PI * momentum_list[imom][2] / LZ_global };
          double _Complex const phase[4] = {
            cexp ( -mom[imu] * I ),
            cexp ( +mom[inu] * I ),
            1.,
            cexp ( ( -mom[imu] + mom[inu] ) * I ) };

          double _Complex *** meesp_out = init_3level_ztable ( T_global, 3, nev );
          if ( meesp_out == NULL ) {
            fprintf ( stderr, "[contract_cvc_tensor_eo_lm_mee] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
            return(8);
          }

          /***********************************************************
           * loop on timeslices
           ***********************************************************/
          for ( int it = 0; it < T_global; it++ ) {
 
            /***********************************************************
             * loop on eigenvectors
             ***********************************************************/
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
            for ( unsigned int iev = 0; iev < nev; iev++ ) {

              /***********************************************************
               * combine the 4 scalar products with corresponding phase;
               * sort according to dt
               ***********************************************************/
              for ( unsigned int isp = 0; isp < 4; isp++ )
              // for ( unsigned int isp = 2; isp <= 2; isp++ )
              {
                meesp_out[it][dt[isp]+1][iev] += meesp_buffer[it][iev][isp] * phase[isp];
              }

            }  // end of loop on eigenvectors

            /***********************************************************
             * loop on dt
             ***********************************************************/
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
