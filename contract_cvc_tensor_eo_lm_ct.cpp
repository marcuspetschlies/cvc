/***********************************************************
 * contract_cvc_tensor_eo_lm_ct.cpp
 *
 * Mi 21. Mär 07:21:36 CET 2018
 ***********************************************************/

/***********************************************************
 * contractions for Mee part of eo-precon lm cvc - cvc tensor
 ***********************************************************/
int const contract_cvc_tensor_eo_lm_ct (
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
      fprintf(stderr, "[contract_cvc_tensor_eo_lm_ct] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
#endif
  }

  // allocate auxilliary spinor fields and spinor work (with halo)
  double ** eo_spinor_field = init_2level_dtable ( 5, _GSI(Vhalf) );
  if ( eo_spinor_field == NULL ) {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_ct] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double ** eo_spinor_work = init_2level_dtable ( 2, _GSI( (VOLUME+RAND)/2 ) );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_ct] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
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
  double _Complex *** ct = init_3level_ztable (T, 4, nev );
  if ( ct == NULL )  {
    fprintf(stderr, "# [contract_cvc_tensor_eo_lm_ct] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
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

      // ( g5 Gmufwdr V )^+ XW
      // fwd , even target field 0, 0
      memcpy ( eo_spinor_work[0], v, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 0 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, gmuf, xw, 0 );
      for ( int it = 0; it < T; it++ ) ct[it][imu][inev] -= p[it];

      // ( g5 Gmufwd XV )^+ W
      // fwd, odd target field 0, 1
      memcpy ( eo_spinor_work[0], xv, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 1 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, gmuf, w, 1 );
      for ( int it = 0; it < T; it++ ) ct[it][imu][inev] -= p[it];


      // ( XV )^+ ( g5 Gmufwd W ) on even sublattice
      // fwd, even target field 0, 0
      memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 0 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, xv, gmuf, 0 );
      for ( int it = 0; it < T; it++ ) ct[it][imu][inev] -= p[it];

      // ( V )^+ ( g5 Gmufwd XW )
      // fwd, odd target field 0, 1
      memcpy ( eo_spinor_work[0], xw, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( gmuf, eo_spinor_work[0], imu, 0, gauge_field, 1 );
      g5_phi ( gmuf, Vhalf );
      eo_spinor_spatial_scalar_product_co( p, v, gmuf, 1 );
      for ( int it = 0; it < T; it++ ) ct[it][imu][inev] -= p[it];

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
          fprintf ( stderr, "[contract_cvc_tensor_eo_lm_ct] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(15);
        }
          
      }  // end of loop on directions mu

    }  // end of loop on timeslices

  }  // end of if io_proc >= 1

  fini_3level_ztable ( &ct );

  return(0);
 
}  // end of contract_cvc_tensor_eo_lm_ct
