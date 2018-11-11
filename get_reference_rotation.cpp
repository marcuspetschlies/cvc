#define _SQR(_a) ((_a)*(_a))

#define _INT3_EQ_INT3(_p,_q) ( ( (_p)[0] == (_q)[0] ) && ( (_p)[1] == (_q)[1] ) && ( (_p)[2] == (_q)[2] ) )

inline int comp_int3_abs ( int const a[3] , int const b[3] ) {
  int a2[3] = { iabs(a[0]), iabs(a[1]), iabs(a[2]) };
  int b2[3] = { iabs(b[0]), iabs(b[1]), iabs(b[2]) };

  if ( a2[1] < a2[0] ) {
    int c = a2[0];
    a2[0] = a2[1];
    a2[1] = c;
  }
  if ( a2[2] < a2[1] ) {
    int c = a2[1];
    a2[1] = a2[2];
    a2[2] = c;
  }
  if ( a2[1] < a2[0] ) {
    int c = a2[0];
    a2[0] = a2[1];
    a2[1] = c;
  }

  if ( b2[1] < b2[0] ) {
    int c = b2[0];
    b2[0] = b2[1];
    b2[1] = c;
  }
  if ( b2[2] < b2[1] ) {
    int c = b2[1];
    b2[1] = b2[2];
    b2[2] = c;
  }
  if ( b2[1] < b2[0] ) {
    int c = b2[0];
    b2[0] = b2[1];
    b2[1] = c;
  }

  return ( _INT3_EQ_INT3 ( a2, b2 ) );

}  /* end of comp_int3_abs */

/***********************************************************
 * return the first valid reference rotation found in the
 * list of all rotations
 ***********************************************************/

int get_reference_rotation ( int pref[3], int *Rref, int const p[3] ) {

  /***********************************************************
   * momentum lists, up to d^+ d = 3
   ***********************************************************/

  int const pzero[3] = {0,0,0};

  int const momentum_ref[3][3] = { {0,0,1}, {1,1,0}, {1,1,1} };

  /*
  int const momentum_num[3] = { 6, 12, 8};
  int const momentum_class_c4v[ 6][3] = { {0,0,1}, {0,0,-1}, {0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0} };
  int const momentum_class_c2v[12][3] = { {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1} };
  int const momentum_class_c3v[ 8][3] = { {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}, {-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1} };
  */

#ifdef CUBIC_GROUP_DOUBLE_COVER
  char const  *momentum_class_name[3] = { "2C4v", "2C2v", "2C3v" };
#elif defined CUBIC_GROUP_SINGLE_COVER 
  char const  *momentum_class_name[3] = {  "C4v",  "C2v",  "C3v" };
#endif

  /***********************************************************
   * return -1 for momentum p = 0,0,0 and for any of the
   * reference momenta itself
   ***********************************************************/
  if ( _INT3_EQ_INT3(p, pzero) || 
       _INT3_EQ_INT3(p, momentum_ref[0]) || 
       _INT3_EQ_INT3(p, momentum_ref[1]) || 
       _INT3_EQ_INT3(p, momentum_ref[2])  ) {

    if ( g_verbose > 1 ) fprintf ( stdout, "# [get_reference_rotation] zero momentum, ref -1\n" );
    pref[0] = p[0];
    pref[1] = p[1];
    pref[2] = p[2];
    *Rref = -1;
    return ( 0 );
  }

  /***********************************************************
   * loop on momentum classes
   ***********************************************************/
  for ( int ic = 0; ic < 3; ic++ ) 
  {

    if ( comp_int3_abs ( p , momentum_ref[0] ) {

      if ( g_verbose > 1 ) fprintf ( stdout, "# [get_reference_rotation] p = %3d %3d %3d has pref %3d %3d %3d\n",
          p[0], p[1], p[2], momentum_ref[ic][0], momentum_ref[ic][1], momentum_ref[ic][2] );

      /***********************************************************
       * loop on rotations
       ***********************************************************/

#ifdef CUBIC_GROUP_DOUBLE_COVER
      for(int irot=0; irot < 48; irot++ ) {

        double _Complex ** S = rot_init_rotation_matrix (3);
        double _Complex ** R = rot_init_rotation_matrix (3);

        rot_rotation_matrix_spherical_basis ( S, 2, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

        rot_spherical2cartesian_3x3 ( R, S );

        if ( ! rot_mat_check_is_real_int ( R, 3 ) ) {
          fprintf(stderr, "[reference_rotations] rot_mat_check_is_real_int false for matrix R rot %2d\n", irot);
          return ( 1 );
        }

        int prot[3] = {0,0,0};
        rot_point ( prot, (int*)momentum_ref[ic], R );

        double norm = sqrt(
          _SQR(prot[0] - momentum_class[imom][0]) + 
          _SQR(prot[1] - momentum_class[imom][1]) + 
          _SQR(prot[2] - momentum_class[imom][2]) );

      
        if ( norm == 0 ) {
          rot_count++;
          fprintf( stdout, "# [reference_rotations] rot %2d n = ( %d,  %d,  %d) w = %16.6e pi\n", irot+1,
              cubic_group_double_cover_rotations[irot].n[0], cubic_group_double_cover_rotations[irot].n[1], cubic_group_double_cover_rotations[irot].n[2],
              cubic_group_double_cover_rotations[irot].w / M_PI );

          char name[40];
          sprintf(name, "Rref[[%d,%2d,%2u]]", ic, imom, rot_count);
          rot_printf_rint_matrix (R, 3, name, stdout );
        }

        rot_fini_rotation_matrix ( &S );
        rot_fini_rotation_matrix ( &R );

      }  // end of loop on rotations
#endif  /* of ifdef CUBIC_GROUP_DOUBLE_COVER */

    }  /* end of if p == pref */

    break;
#if 0
#endif  // of if 0

#if 0
      for(int irot=0; irot < 24; irot++ ) {

        double _Complex ** S = rot_init_rotation_matrix (3);
        double _Complex ** R = rot_init_rotation_matrix (3);

        rot_rotation_matrix_spherical_basis_Wigner_D ( S, 2, cubic_group_rotations_v2[irot].a );

        rot_spherical2cartesian_3x3 ( R, S );

        if ( ! rot_mat_check_is_real_int ( R, 3 ) ) {
          fprintf(stderr, "[reference_rotations] rot_mat_check_is_real_int false for matrix R rot %2d\n", irot);
          EXIT(6);
        }

        int prot[3] = {0,0,0};
        rot_point ( prot, (int*)momentum_ref[ic], R );

        double norm = sqrt(
          _SQR(prot[0] - momentum_class[imom][0]) + 
          _SQR(prot[1] - momentum_class[imom][1]) + 
          _SQR(prot[2] - momentum_class[imom][2]) );

      
        if ( norm == 0 ) {
          rot_count++;
          fprintf( stdout, "# [reference_rotations] rot %2d %5s a = %16.7e pi %16.7e pi %16.7e pi\n", irot+1,
              cubic_group_rotations_v2[irot].name,
              cubic_group_rotations_v2[irot].a[0] / M_PI,
              cubic_group_rotations_v2[irot].a[1] / M_PI,
              cubic_group_rotations_v2[irot].a[2] / M_PI );

          char name[40];
          sprintf(name, "Rref[[%d,%2d,%2u]]", ic, imom, rot_count);
          rot_printf_rint_matrix (R, 3, name, stdout );
        }

        rot_fini_rotation_matrix ( &S );
        rot_fini_rotation_matrix ( &R );

      }  // end of loop on rotations

#endif  // of if 0

  }  /* end of loop on momentum classes */

  return(0);
}  /* end of get_reference_rotation */
