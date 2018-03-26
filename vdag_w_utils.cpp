/***********************************************************
 * vdag_w_utils.cpp
 *
 * Mi 21. Mär 07:27:42 CET 2018
 ***********************************************************/

/***********************************************************
 *
 * dimV must be integer multiple of dimW
 ***********************************************************/
int vdag_w_spin_color_reduction ( double ***contr, double ** const V, double ** const W, unsigned int const dimV, unsigned int const dimW, int const t ) {

  unsigned int const VOL3half = LX*LY*LZ/2;
  size_t const sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);

  double ratime, retime;

  ratime = _GET_TIME;

  double ** v_ts = init_2level_dtable ( dimV, _GSI( VOL3half) );
  if ( v_ts == NULL ) {
    fprintf( stderr, "[vdag_w_spin_color_reduction] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  double ** w_ts = init_2level_dtable ( dimW, _GSI( VOL3half) );
  if ( w_ts == NULL ) {
    fprintf( stderr, "[vdag_w_spin_color_reduction] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * copy the timeslices t % T
   ***********************************************************/
  unsigned int const offset = ( t % T) * _GSI( VOL3half );
  for ( unsigned int i = 0; i < dimV; i++ ) {
    memcpy( v_ts[i], V[i] + offset, sizeof_eo_spinor_field_timeslice );
  }

  for ( unsigned int i = 0; i < dimW; i++ ) {
    memcpy( w_ts[i], W[i] + offset, sizeof_eo_spinor_field_timeslice );
  }

#ifdef HAVE_MPI
  /***********************************************************
   * exchange here in x-space or later in p-space?
   * which is more efficient?
   ***********************************************************/

  /***********************************************************
   * if t = T, exchange
   * receive from g_nb_t_up
   * send    to   g_nb_t_dn
   ***********************************************************/
  if ( t == T ) {
    int cntr = 0;
    MPI_Status mstatus[2];
    MPI_Request request[2];
    unsigned int items = dimV * _GSI( VOL3half );

    /***********************************************************
     * exchange v_ts
     ***********************************************************/
    double * buffer = init_1level_dtable ( items ); 
    if ( buffer == NULL ) { 
      fprintf(stderr, "[vdag_w_spin_color_reduction] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
      return(2);
    }
    memcpy ( buffer, v_ts[0], dimV * sizeof_eo_spinor_field_timeslice );
 
    MPI_Isend( buffer, items, MPI_DOUBLE, g_nb_t_dn, 101, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Irecv( v_ts[0], items, MPI_DOUBLE, g_nb_t_up, 101, g_cart_grid, &request[cntr]);
    cntr++;
 
    MPI_Waitall( cntr, request, mstatus );
    fini_1level_dtable ( &buffer );

    /***********************************************************
     * exchange w_ts
     ***********************************************************/
    items = dimW * _GSI( VOL3half );
    buffer = init_1level_dtable ( dimW * _GSI(VOL3half) ); 
    if ( buffer == NULL ) { 
      fprintf(stderr, "[vdag_w_spin_color_reduction] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
      return(3);
    }

    memcpy ( buffer, w_ts[0], dimW * sizeof_eo_spinor_field_timeslice );
    cntr = 0;

    MPI_Isend( buffer, items, MPI_DOUBLE, g_nb_t_dn, 102, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Irecv( w_ts[0], items, MPI_DOUBLE, g_nb_t_up, 102, g_cart_grid, &request[cntr]);
    cntr++;

    MPI_Waitall( cntr, request, mstatus );
    fini_1level_dtable ( &buffer );

  }  /* end of if t == T */
#endif  // of ifdef HAVE_MPI

  for ( unsigned int iv = 0; iv < dimV; iv++ ) {
  for ( unsigned int iw = 0; iw < dimW; iw++ ) {
    co_field_eq_fv_dag_ti_fv ( contr[iv][iw], v_ts[iv], w_ts[iw], VOL3half );
  }}

  fini_2level_dtable ( &v_ts );
  fini_2level_dtable ( &w_ts );

  retime = _GET_TIME;
  if ( g_cart_id == 0 && g_verbose > 0) {
    fprintf(stdout, "# [vdag_w_spin_color_reduction] time for vdag_w_spin_color_reduction = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }
  return(0);
}  // end of vdag_w_spin_color_reduction

/***********************************************************/
/***********************************************************/

/***********************************************************
 * momentum projection
 ***********************************************************/
int vdag_w_momentum_projection ( 
    double _Complex ***contr_p, 
    double *** const contr_x, int const dimV, int const dimW, 
    int (* const momentum_list)[3], int const momentum_number, 
    int const t, 
    int const ieo, 
    int const mu 
) {

  int exitstatus;
  double momentum_shift[3] = {0.,0.,0.};

  if ( mu > 0 ) {
    momentum_shift[mu-1] = -1.;
  }

  if ( (exitstatus = momentum_projection_eo_timeslice ( contr_x[0][0], (double*)(contr_p[0][0]), dimV*dimW, momentum_number, momentum_list, t, ieo, momentum_shift, 1 )) != 0 ) {
    fprintf(stderr, "[vdag_w_momentum_projection] Error from momentum_projection_eo_timeslice, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  return(0);
}  // end of vdag_w_momentum_projection

/***********************************************************/
/***********************************************************/

/***********************************************************
 * write to AFF file
 ***********************************************************/
int vdag_w_write_to_aff_file ( 
    double _Complex *** const contr_tp, unsigned int const nv, unsigned int const nw, 
    struct AffWriter_s*affw, 
    char * const tag, 
    int (* const momentum_list)[3], unsigned int const momentum_number, 
    int const io_proc 
) {

#ifdef HAVE_LHPC_AFF
  uint32_t const items = nv * nw;
#else
  unsigned int const items = nv * nw;
#endif

  int exitstatus;
  double ratime, retime;
  char buffer_path[600];

  ratime = _GET_TIME;
#if 0
  if ( io_proc >= 1 ) {

#ifdef HAVE_LHPC_AFF

    struct AffNode_s *affn = aff_writer_root( affw );
    if( affn == NULL ) {
      fprintf(stderr, "[vdag_w_write_to_aff_file] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    //exitstatus = aff_name_check3 ( tag );
    //if ( g_verbose > 4 ) fprintf(stdout, "# [vdag_w_write_to_aff_file] aff_name_check status %d on tag %s %s %d\n", exitstatus , tag, __FILE__, __LINE__);
    // if ( g_verbose > 4 ) fprintf(stdout, "# [vdag_w_write_to_aff_file] current mkdir tag = %s %s %d\n", tag, __FILE__, __LINE__);

    // struct AffNode_s * affdir = aff_writer_mkdir ( affw, affn, tag );
    struct AffNode_s * affdir = aff_writer_mkpath ( affw, affn, tag );
    const char * aff_errstr = aff_writer_errstr ( affw );
    if ( aff_errstr != NULL ) {
      fprintf(stderr, "[vdag_w_write_to_aff_file] Error from aff_reader_chpath for key prefix \"%s\", status was %s %s %d\n", tag, aff_errstr, __FILE__, __LINE__ );
      return(2);
    }

    for( unsigned int i = 0; i < momentum_number; i++ ) {

      sprintf(buffer_path, "px%.2dpy%.2dpz%.2d", momentum_list[i][0], momentum_list[i][1], momentum_list[i][2] );
      // fprintf(stdout, "# [vdag_w_write_to_aff_file] current aff path = %s\n", buffer_path);

      struct AffNode_s * affpath = aff_writer_mkpath ( affw, affdir, buffer_path);

      exitstatus = aff_node_put_complex ( affw, affpath, contr_tp[i][0], items );
      if(exitstatus != 0) {
        fprintf(stderr, "[vdag_w_write_to_aff_file] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }
    }  // end of loop on momenta

#else

    for( unsigned int i = 0; i < momentum_number; i++ ) {

      sprintf( buffer_path, "%s_px%.2dpy%.2dpz%.2d", tag, momentum_list[i][0], momentum_list[i][1], momentum_list[i][2] );
      FILE *ofs = fopen ( buffer_path, "wb" );
      if ( ofs == NULL ) {
        fprintf(stderr, "[vdag_w_write_to_aff_file] Error from fopen %s %d\n", __FILE__, __LINE__);
        return(4);
      }

      if ( fwrite ( contr_tp[i][0], sizeof(double _Complex), items, ofs ) != items ) {
        fprintf ( stdout, "[vdag_w_write_to_aff_file] Error from fwrite %s %d\n", __FILE__, __LINE__ );
        return(5);
      }

      fclose ( ofs );

    }  // end of loop on momenta
#endif
  }  /* if io_proc >= 1 */


#ifdef HAVE_MPI
  if ( MPI_Barrier( g_cart_grid ) != MPI_SUCCESS ) {
    fprintf ( stderr, "# [vdag_w_write_to_aff_file] Error from MPI_Barrier %s %d\n", __FILE__, __LINE__);
    return(1);
  }
#endif
#endif  // of if 0
  retime = _GET_TIME;

  if( io_proc == 2 && g_verbose > 0) {
    // fprintf(stdout, "# [vdag_w_write_to_aff_file] time for saving momentum space results = %e seconds\n", retime-ratime);
    fprintf(stdout, "# [vdag_w_write_to_aff_file] time for saving momentum space results = %e seconds\n", retime );
    fprintf(stdout, "# [vdag_w_write_to_aff_file] time for saving momentum space results = %e seconds\n", ratime );
    fflush ( stdout );
  }

  return(0);
}  // end of vdag_w_write_to_aff_file
