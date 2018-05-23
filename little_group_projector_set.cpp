/***********************************************************
 * little_group_projector_set.cpp
 *
 * Do 19. Apr 10:12:19 CEST 2018
 *
 ***********************************************************/

int little_group_projector_set (
  /***********************************************************/
    little_group_projector_type * const p,
    little_group_type * const lg, 
    const char * irrep , const int row_target, const int interpolator_num,
    const int * interpolator_J2_list,
    const int ** interpolator_momentum_list,
    const int * interpolator_bispinor_list,
    const int * interpolator_parity_list,
    const int * interpolator_cartesian_list,
    const int ref_row_target,
    const int * ref_row_spin,
    const char * name,
    int const refframerot
  /***********************************************************/
) {

  int exitstatus;

  /***********************************************************
   * set number of interpolators
   ***********************************************************/
  if ( interpolator_num > 0 ) {
    p->n = interpolator_num;
  } else if ( p->n <= 0 ) {
    fprintf( stderr, "[little_group_projector_set] Error, number of interpolators is zero\n");
    return(1);
  }
  fprintf( stdout, "# [little_group_projector_set] p-> = %d\n", p->n );

  /***********************************************************
   * init, allocate and fill
   * target cubic group irrep rotation table
   ***********************************************************/
  if ( ( p->rtarget= (rot_mat_table_type *)malloc (  sizeof( rot_mat_table_type ) ) ) == NULL ) {
    fprintf( stderr, "[little_group_projector_set] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  init_rot_mat_table ( p->rtarget );
#if defined CUBIC_GROUP_DOUBLE_COVER
  exitstatus = set_rot_mat_table_cubic_group_double_cover ( p->rtarget, lg->name, irrep );
#elif defined CUBIC_GROUP_SINGLE_COVER
  exitstatus = set_rot_mat_table_cubic_group_single_cover ( p->rtarget, lg->name, irrep );
#endif
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[little_group_projector_set] Error from set_rot_mat_table_cubic_group %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  /***********************************************************
   * set row of target irrep, if provided
   ***********************************************************/
  if ( row_target >= 0 ) {
    p->row_target = row_target;
  }

  /***********************************************************
   * set reference row of target irrep, if provided
   * (select irrep matrix column in projector,
   *  we call it row as well)
   ***********************************************************/
  if ( ref_row_target >= 0 ) {
    p->ref_row_target = ref_row_target;
  }

  /***********************************************************
   * reference row for each spin vector, if provided
   ***********************************************************/
  if ( ref_row_spin != NULL ) {
    p->ref_row_spin = init_1level_itable ( p->n );
    if ( p->ref_row_spin == NULL ) {
      fprintf(stderr, "[little_group_projector_set] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
      return(1);
    }
    memcpy ( p->ref_row_spin, ref_row_spin, p->n*sizeof(int) );
  }

  /***********************************************************
   * set intrinsic parity of operator, if provided
   *
   * if not provided, default is +1
   ***********************************************************/
  p->parity = init_1level_itable ( p->n );
  if ( p->parity == NULL ) {
    fprintf(stderr, "[little_group_projector_set] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  if ( interpolator_parity_list != NULL ) {
    memcpy ( p->parity, interpolator_parity_list, p->n*sizeof(int) );
  } else {
    for ( int i = 0; i < p->n; i++ ) p->parity[i] = 1;
  }

  /***********************************************************
   * set the correlator name, if provided;
   * default "NA" should already be set
   ***********************************************************/
  if ( name != NULL ) {
    strcpy ( p->name , name );
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * prepare reference frame rotation
   ***********************************************************/

  double _Complex **refframerot_p = NULL;
  double _Complex ***refframerot_spin = NULL;
#if defined CUBIC_GROUP_DOUBLE_COVER
  int const use_refframerot = ( p->refframerot > -1 ) && ( p->refframerot < 48 );
#elif defined CUBIC_GROUP_SINGLE_COVER
  int const use_refframerot = ( p->refframerot > -1 ) && ( p->refframerot < 24 );
#endif

  if ( use_refframerot ) {
    // set the reference frame rotation matrix
    // for the 3-momentum vector p;
    // spin 1 in cartesian basis
    refframerot_p = rot_init_rotation_matrix ( 3 );
    if ( refframerot_p == NULL ) return(NULL);

#if defined CUBIC_GROUP_DOUBLE_COVER
    rot_mat_spin1_cartesian ( refframerot_p, cubic_group_double_cover_rotations[p->refframerot].n, cubic_group_double_cover_rotations[p->refframerot].w );
#elif defined CUBIC_GROUP_SINGLE_COVER
    rot_rotation_matrix_spherical_basis_Wigner_D ( refframerot_p, 2, cubic_group_rotations_v2[p->refframerot].a );
    rot_spherical2cartesian_3x3 ( refframerot_p, refframerot_p );
#endif
    if ( ! ( rot_mat_check_is_real_int ( refframerot_p) ) ) {
      fprintf(stderr, "[little_group_projector_set] Error rot_mat_check_is_real_int refframerot_p %s %d\n", __FILE__, __LINE__);
      return(72);
    }

    // set the reference frame rotation matrix
    // for the spin-J vectors
    refframerot_spin = (double _Complex ***) malloc ( p->n * sizeof(double _Complex **) );
    if ( refframerot_spin == NULL ) return( NULL);

    for ( int i = 0; i < p->n; i++ ) {
      refframerot_spin[i] = rot_init_rotation_matrix ( spin_dimensions[i] );
      if ( refframerot_spin[i] == NULL ) return(NULL);

#if defined CUBIC_GROUP_DOUBLE_COVER
      rot_rotation_matrix_spherical_basis ( refframerot_spin[i], spin_dimensions[i]-1, cubic_group_double_cover_rotations[p->refframerot].n, cubic_group_double_cover_rotations[p->refframerot].w );
#elif defined CUBIC_GROUP_SINGLE_COVER
      rot_rotation_matrix_spherical_basis_Wigner_D ( refframerot_spin[i], spin_dimensions[i]-1, cubic_group_rotations_v2[p->refframerot].a );
#endif
      if ( interpolator_cartesian_list[i] && ( interpolator_J2_list[i] == 2 ) ) {
        rot_spherical2cartesian_3x3 ( refframerot_spin[i], refframerot_spin[i] );
        if ( ! ( rot_mat_check_is_real_int ( refframerot_spin[i] ) ) ) {
          fprintf(stderr, "[little_group_projector_set] Error rot_mat_check_is_real_int refframerot_spin %2d %s %d\n", i,  __FILE__, __LINE__);
          return(71);
        }
      }
    }  // end of loop on interpolators

  }  // end of if refframerot

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set total momentum vector
   *
   * it is given by the little group's lg->d under consideration
   ***********************************************************/
  p->P[0] = lg->d[0];
  p->P[1] = lg->d[1];
  p->P[2] = lg->d[2];
  if ( use_refframerot ) rot_point ( p->P, p->P, refframerot_p );


  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set rotation matrix table for 3-momentum vector p
   * for each interpolator
   ***********************************************************/
  rot_mat_table_type rp;
  init_rot_mat_table ( &rp );

#if defined CUBIC_GROUP_DOUBLE_COVER
  exitstatus = set_rot_mat_table_spin ( &rp, 2, 0 );
#elif defined CUBIC_GROUP_SINGLE_COVER
  exitstatus = set_rot_mat_table_spin_single_cover ( &rp, 2, 1, 1 );
#endif

  if ( exitstatus != 0 ) {
    fprintf(stderr, "[little_group_projector_set] Error from set_rot_mat_table_spin %s %d\n", __FILE__, __LINE__);
    return(3);
  }

  if ( ( p->rp = (rot_mat_table_type*)malloc ( sizeof( rot_mat_table_type ) ) ) == NULL ) {
    fprintf(stderr, "[little_group_projector_set] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(2);
  }
  init_rot_mat_table ( p->rp );
  if(  alloc_rot_mat_table ( p->rp, rp.group, rp.irrep, rp.dim, p->rtarget->n ) != 0 ) {
    fprintf(stderr, "[little_group_projector_set] Error from alloc_rot_mat_table %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * transform p-3-vector rotations to cartesian basis
   ***********************************************************/
  for ( int irot = 0; irot < p->rtarget->n; irot++ ) {
    int rid  = p->rtarget->rid[irot];
    int rmid = p->rtarget->rmid[irot];
    (p->rp)->rid[irot]  = rid;
    (p->rp)->rmid[irot] = rmid;
    rot_spherical2cartesian_3x3 ( (p->rp)->R[irot],  rp.R[rid]  );
    rot_spherical2cartesian_3x3 ( (p->rp)->IR[irot], rp.IR[rmid] );

    if ( ! ( rot_mat_check_is_real_int ( (p->rp)->R[irot], 3 ) && rot_mat_check_is_real_int ( (p->rp)->IR[irot], 3 ) ) ) {
      fprintf(stderr, "[little_group_projector_set] Error rot_mat_check_is_real_int rot %d / %d %s %d\n", rid, rmid, __FILE__, __LINE__);
      return(7);
    }
  STOPPED HERE
    if ( use_refframerot ) {
      rot_mat_ti_mat (double _Complex **C, double _Complex **A, double _Complex **B, int N);

      rot_mat_ti_mat_adj (double _Complex **C, double _Complex **A, double _Complex **B, int N);

    }

  }  /* end of loop on rotation group elements in target irrep */
  fini_rot_mat_table ( &rp );

  /***********************************************************
   * set rotation matrix table for each interpolator
   ***********************************************************/
  if ( ( p->rspin = (rot_mat_table_type*)malloc ( p->n * sizeof( rot_mat_table_type ) ) ) == NULL ) {
    fprintf(stderr, "[little_group_projector_set] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(4);
  }

  for ( int i = 0; i < p->n; i++ ) {
    rot_mat_table_type rspin;
    init_rot_mat_table ( &rspin );
#if defined CUBIC_GROUP_DOUBLE_COVER
    exitstatus = set_rot_mat_table_spin ( &rspin, interpolator_J2_list[i], interpolator_bispinor_list[i] );
#elif defined CUBIC_GROUP_SINGLE_COVER
    exitstatus = set_rot_mat_table_spin_single_cover (&rspin, interpolator_J2_list[i], 1, 1 );
#endif
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[little_group_projector_set] Error from set_rot_mat_table_spin %s %d\n", __FILE__, __LINE__);
      return(5);
    }

    init_rot_mat_table ( &(p->rspin[i]) );
    if(  alloc_rot_mat_table ( &(p->rspin[i]), rspin.group, rspin.irrep, rspin.dim, p->rtarget->n ) != 0 ) {
      fprintf(stderr, "[little_group_projector_set] Error from alloc_rot_mat_table %s %d\n", __FILE__, __LINE__);
      return(2);
    }
    for ( int irot = 0; irot < p->rtarget->n; irot++ ) {
      int rid  = p->rtarget->rid[irot];
      int rmid = p->rtarget->rmid[irot];
      p->rspin[i].rid[irot]  = rid;
      p->rspin[i].rmid[irot] = rmid;

      /***********************************************************
       * check, whether we have spin 1 and want to use cartesian
       * basis instead of spherical
       ***********************************************************/
      if ( ( interpolator_cartesian_list == NULL ) || ( interpolator_J2_list[i] != 2 )  ) {
        rot_mat_assign ( p->rspin[i].R[irot],  rspin.R[rid],   rspin.dim );
        rot_mat_assign ( p->rspin[i].IR[irot], rspin.IR[rmid], rspin.dim );
      } else {
        if ( interpolator_cartesian_list[i] && ( interpolator_J2_list[i] == 2 ) ) {
          rot_spherical2cartesian_3x3 ( (p->rspin[i]).R[irot],  rspin.R[rid]  );
          rot_spherical2cartesian_3x3 ( (p->rspin[i]).IR[irot], rspin.IR[rmid] );
          if ( ! ( rot_mat_check_is_real_int ( (p->rspin[i]).R[irot], 3 ) && rot_mat_check_is_real_int ( (p->rspin[i]).IR[irot], 3 ) ) ) {
            fprintf(stderr, "[little_group_projector_set] Error rot_mat_check_is_real_int rot %d / %d %s %d\n", rid, rmid, __FILE__, __LINE__);
            return(7);
          }

        } else {
          rot_mat_assign ( p->rspin[i].R[irot],  rspin.R[rid],   rspin.dim );
          rot_mat_assign ( p->rspin[i].IR[irot], rspin.IR[rmid], rspin.dim );
        }
      }
    }  /* end of loop on p->rtarget->n rotations */

    fini_rot_mat_table ( &rspin );
  }  /* end of loop on p->n interpolators */

  /***********************************************************
   * allocate and set 3-momentum for each interpolator
   ***********************************************************/
  p->p = init_2level_itable ( p->n, 3 );
  if ( p->p == NULL ) {
    fprintf(stderr, "[little_group_projector_set] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    return(5);
  }
  if ( interpolator_momentum_list != NULL ) {
    memcpy ( p->p[0], interpolator_momentum_list[0], 3*p->n * sizeof(int) );
  }
  if ( use_refframerot ) {
    for ( int i = 0; i < p->n; i++ ) rot_point ( p->p[i], p->p[i], refframerot_p );
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free refframerot rotation matrices
   ***********************************************************/
  if ( refframerot_p != NULL ) rot_fini_rotation_matrix ( &refframerot_p );
  if ( refframerot_spin != NULL ) {
    for ( int i = 0; i < p->n; i++ ) rot_fini_rotation_matrix ( &(refframerot_spin[i]) );
    free ( refframerot_spin );
  }



  return(0);
}  // end of little_group_projector_set


/***********************************************************/
/***********************************************************/
