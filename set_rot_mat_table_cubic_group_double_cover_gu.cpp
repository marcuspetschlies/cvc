/***********************************************************
 * set_rot_mat_table_cubic_group_double_cover_gu.cpp
 *
 * Do 19. Apr 08:44:50 CEST 2018
 *
 * Our standard reference JHEP08(2008)024 gives representation
 * matrices for A1, A2, E, T1, T2, G1, G2, H of 2O
 * i.e. no g/u parity projection
 *
 * Here we complete to A1g, A1u, ... for 2Oh by using
 *
 * T^{Xg]( R ) = +T^{Xg}( IR )
 * T^{Xu]( R ) = -T^{Xu}( IR )
 *
 * for all irreps X
 *
 ***********************************************************/

/***********************************************************
 * irrep matrices for double cover
 ***********************************************************/
int set_rot_mat_table_cubic_group_double_cover ( rot_mat_table_type *t, const char *group, const char*irrep ) {

  const double ONE_HALF   = 0.5;
  const double SQRT3_HALF = 0.5 * sqrt(3.);

  int exitstatus;

  /***********************************************************
   * LG 2Oh
   ***********************************************************/
  if ( strcmp ( group, "2Oh" ) == 0 ) {
    int nrot = 48;
    
    /***********************************************************
     * LG 2Oh irrep A1
     ***********************************************************/
    if ( strcmp ( irrep, "A1g" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        t->rmid[i] = i;
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /***********************************************************
     * LG 2Oh irrep A1u
     ***********************************************************/
    } else if ( strcmp ( irrep, "A1u" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        t->rmid[i] = i;
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

    /***********************************************************
     * LG 2Oh irrep A2g
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2g" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) { 
        t->rid[i]      = i; 
        t->rmid[i]     = i; 
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

      /* 6C8', 6C8 */
      for ( int i = 7; i <= 18; i++ ) {
        t->R[i][0][0]  = -1.;
        t->IR[i][0][0] = -1.;
      }
     
      /* 12C4' */
      for ( int i = 35; i <= 46; i++ ) {
        t->R[i][0][0]  = -1.;
        t->IR[i][0][0] = -1.;
      }

    /***********************************************************
     * LG 2Oh irrep A2u
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2u" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) { 
        t->rid[i]      = i; 
        t->rmid[i]     = i; 
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

      /* 6C8', 6C8 */
      for ( int i = 7; i <= 18; i++ ) {
        t->R[i][0][0]  = -1.;
        t->IR[i][0][0] =  1.;
      }
     
      /* 12C4' */
      for ( int i = 35; i <= 46; i++ ) {
        t->R[i][0][0]  = -1.;
        t->IR[i][0][0] =  1.;
      }


    /***********************************************************
     * LG 2Oh irrep E
     ***********************************************************/
    } else if ( ( strcmp ( irrep, "Eg" ) == 0 ) || ( strcmp ( irrep, "Eu" ) == 0 ) ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) { t->rid[i] = i; }

      /* R1, R48 */
      t->R[ 0][0][0] = 1.; t->R[ 0][1][1] = 1.;
      t->R[47][0][0] = 1.; t->R[47][1][1] = 1.;
      /* R2, R5 */
      t->R[ 1][0][0] = 1.; t->R[ 1][1][1] = 1.;
      t->R[ 4][0][0] = 1.; t->R[ 4][1][1] = 1.;
      /* R3, R6 */
      t->R[ 2][0][0] = 1.; t->R[ 2][1][1] = 1.;
      t->R[ 5][0][0] = 1.; t->R[ 5][1][1] = 1.;
      /* R4, R7 */
      t->R[ 3][0][0] = 1.; t->R[ 3][1][1] = 1.;
      t->R[ 6][0][0] = 1.; t->R[ 6][1][1] = 1.;

      /* R13, R16 sigma_3 */
      t->R[12][0][0] = 1.; t->R[12][1][1] = -1.;
      t->R[15][0][0] = 1.; t->R[15][1][1] = -1.;
      /* R10, R19 sigma_3 */
      t->R[ 9][0][0] = 1.; t->R[ 9][1][1] = -1.;
      t->R[18][0][0] = 1.; t->R[18][1][1] = -1.;
      /* R38, R44 sigma_3 */
      t->R[37][0][0] = 1.; t->R[37][1][1] = -1.;
      t->R[43][0][0] = 1.; t->R[43][1][1] = -1.;
      /* R39, R45 sigma_3 */
      t->R[38][0][0] = 1.; t->R[38][1][1] = -1.;
      t->R[44][0][0] = 1.; t->R[44][1][1] = -1.;
      
      /* -1/2 1 + i sqrt(3)/2 sigma_2 */

      /* R24, R28 */
      t->R[23][0][0] = -ONE_HALF;   t->R[23][1][1] = -ONE_HALF;
      t->R[23][0][1] =  SQRT3_HALF; t->R[23][1][0] = -SQRT3_HALF;
      t->R[27][0][0] = -ONE_HALF;   t->R[27][1][1] = -ONE_HALF;
      t->R[27][0][1] =  SQRT3_HALF; t->R[27][1][0] = -SQRT3_HALF;
      /* R21, R33 */
      t->R[20][0][0] = -ONE_HALF;   t->R[20][1][1] = -ONE_HALF;
      t->R[20][0][1] =  SQRT3_HALF; t->R[20][1][0] = -SQRT3_HALF;
      t->R[32][0][0] = -ONE_HALF;   t->R[32][1][1] = -ONE_HALF;
      t->R[32][0][1] =  SQRT3_HALF; t->R[32][1][0] = -SQRT3_HALF;
      /* R26, R30 */
      t->R[25][0][0] = -ONE_HALF;   t->R[25][1][1] = -ONE_HALF;
      t->R[25][0][1] =  SQRT3_HALF; t->R[25][1][0] = -SQRT3_HALF;
      t->R[29][0][0] = -ONE_HALF;   t->R[29][1][1] = -ONE_HALF;
      t->R[29][0][1] =  SQRT3_HALF; t->R[29][1][0] = -SQRT3_HALF;
      /* R23, R35 */
      t->R[22][0][0] = -ONE_HALF;   t->R[22][1][1] = -ONE_HALF;
      t->R[22][0][1] =  SQRT3_HALF; t->R[22][1][0] = -SQRT3_HALF;
      t->R[34][0][0] = -ONE_HALF;   t->R[34][1][1] = -ONE_HALF;
      t->R[34][0][1] =  SQRT3_HALF; t->R[34][1][0] = -SQRT3_HALF;
   
      /* -1/2 1 - i sqrt(3)/2 sigma_2 */

      /* R20, R32 */
      t->R[19][0][0] = -ONE_HALF;   t->R[19][1][1] = -ONE_HALF;
      t->R[19][0][1] = -SQRT3_HALF; t->R[19][1][0] =  SQRT3_HALF;
      t->R[31][0][0] = -ONE_HALF;   t->R[31][1][1] = -ONE_HALF;
      t->R[31][0][1] = -SQRT3_HALF; t->R[31][1][0] =  SQRT3_HALF;
      /* R25, R29 */
      t->R[24][0][0] = -ONE_HALF;   t->R[24][1][1] = -ONE_HALF;
      t->R[24][0][1] = -SQRT3_HALF; t->R[24][1][0] =  SQRT3_HALF;
      t->R[28][0][0] = -ONE_HALF;   t->R[28][1][1] = -ONE_HALF;
      t->R[28][0][1] = -SQRT3_HALF; t->R[28][1][0] =  SQRT3_HALF;
      /* R22, R34 */
      t->R[21][0][0] = -ONE_HALF;   t->R[21][1][1] = -ONE_HALF;
      t->R[21][0][1] = -SQRT3_HALF; t->R[21][1][0] =  SQRT3_HALF;
      t->R[33][0][0] = -ONE_HALF;   t->R[33][1][1] = -ONE_HALF;
      t->R[33][0][1] = -SQRT3_HALF; t->R[33][1][0] =  SQRT3_HALF;
      /* R27, R31 */
      t->R[26][0][0] = -ONE_HALF;   t->R[26][1][1] = -ONE_HALF;
      t->R[26][0][1] = -SQRT3_HALF; t->R[26][1][0] =  SQRT3_HALF;
      t->R[30][0][0] = -ONE_HALF;   t->R[30][1][1] = -ONE_HALF;
      t->R[30][0][1] = -SQRT3_HALF; t->R[30][1][0] =  SQRT3_HALF;

      /* -cos(pi/3) sigma_3 - sin(pi/3) sigma_1 */

      /* R11, R14 */
      t->R[10][0][0] = -ONE_HALF;   t->R[10][1][1] =  ONE_HALF;
      t->R[10][0][1] = -SQRT3_HALF; t->R[10][1][0] = -SQRT3_HALF;
      t->R[13][0][0] = -ONE_HALF;   t->R[13][1][1] =  ONE_HALF;
      t->R[13][0][1] = -SQRT3_HALF; t->R[13][1][0] = -SQRT3_HALF;
      /* R8, R17 */
      t->R[ 7][0][0] = -ONE_HALF;   t->R[ 7][1][1] =  ONE_HALF;
      t->R[ 7][0][1] = -SQRT3_HALF; t->R[ 7][1][0] = -SQRT3_HALF;
      t->R[16][0][0] = -ONE_HALF;   t->R[16][1][1] =  ONE_HALF;
      t->R[16][0][1] = -SQRT3_HALF; t->R[16][1][0] = -SQRT3_HALF;
      /* R36, R42 */
      t->R[35][0][0] = -ONE_HALF;   t->R[35][1][1] =  ONE_HALF;
      t->R[35][0][1] = -SQRT3_HALF; t->R[35][1][0] = -SQRT3_HALF;
      t->R[41][0][0] = -ONE_HALF;   t->R[41][1][1] =  ONE_HALF;
      t->R[41][0][1] = -SQRT3_HALF; t->R[41][1][0] = -SQRT3_HALF;
      /* R37, R43 */
      t->R[36][0][0] = -ONE_HALF;   t->R[36][1][1] =  ONE_HALF;
      t->R[36][0][1] = -SQRT3_HALF; t->R[36][1][0] = -SQRT3_HALF;
      t->R[42][0][0] = -ONE_HALF;   t->R[42][1][1] =  ONE_HALF;
      t->R[42][0][1] = -SQRT3_HALF; t->R[42][1][0] = -SQRT3_HALF;

      /* -cos(pi/3) sigma_3 + sin(pi/3) sigma_1 */

      /* R12, R15 */
      t->R[11][0][0] = -ONE_HALF;   t->R[11][1][1] =  ONE_HALF;
      t->R[11][0][1] =  SQRT3_HALF; t->R[11][1][0] =  SQRT3_HALF;
      t->R[14][0][0] = -ONE_HALF;   t->R[14][1][1] =  ONE_HALF;
      t->R[14][0][1] =  SQRT3_HALF; t->R[14][1][0] =  SQRT3_HALF;
      /* R9, R18 */
      t->R[ 8][0][0] = -ONE_HALF;   t->R[ 8][1][1] =  ONE_HALF;
      t->R[ 8][0][1] =  SQRT3_HALF; t->R[ 8][1][0] =  SQRT3_HALF;
      t->R[17][0][0] = -ONE_HALF;   t->R[17][1][1] =  ONE_HALF;
      t->R[17][0][1] =  SQRT3_HALF; t->R[17][1][0] =  SQRT3_HALF;
      /* R40, R46 */
      t->R[39][0][0] = -ONE_HALF;   t->R[39][1][1] =  ONE_HALF;
      t->R[39][0][1] =  SQRT3_HALF; t->R[39][1][0] =  SQRT3_HALF;
      t->R[45][0][0] = -ONE_HALF;   t->R[45][1][1] =  ONE_HALF;
      t->R[45][0][1] =  SQRT3_HALF; t->R[45][1][0] =  SQRT3_HALF;
      /* R41, R47 */
      t->R[40][0][0] = -ONE_HALF;   t->R[40][1][1] =  ONE_HALF;
      t->R[40][0][1] =  SQRT3_HALF; t->R[40][1][0] =  SQRT3_HALF;
      t->R[46][0][0] = -ONE_HALF;   t->R[46][1][1] =  ONE_HALF;
      t->R[46][0][1] =  SQRT3_HALF; t->R[46][1][0] =  SQRT3_HALF;

      memcpy ( t->rmid, t->rid, nrot * sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], nrot * 4 * sizeof(double _Complex ) );

      /***********************************************************
       * multiply minus sign to IR irrep matrices
       ***********************************************************/
      if ( strcmp ( irrep, "Eu" ) == 0 ) {
        for ( int irot = 0; irot < nrot; irot++ ) rot_mat_ti_eq_re ( t->IR[irot], -1., 2 );
      }

    /***********************************************************
     * LG 2Oh irrep T1g , T1u
     ***********************************************************/
    } else if ( ( strcmp ( irrep, "T1g" ) == 0  ) || ( strcmp ( irrep, "T1u" ) == 0 ) ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 3, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }
      
      memcpy ( t->rmid, t->rid, nrot * sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], nrot*9*sizeof(double _Complex ) );

      /***********************************************************
       * multiply minus sign to IR irrep matrices
       ***********************************************************/
      if ( strcmp ( irrep, "T1u" ) == 0 ) {
        for ( int irot = 0; irot < nrot; irot++ ) rot_mat_ti_eq_re ( t->IR[irot], -1., 3 );
      }

    /***********************************************************
     * LG 2Oh irrep T2g, T2u
     ***********************************************************/
    } else if ( ( strcmp ( irrep, "T2g" ) == 0 ) || ( strcmp ( irrep, "T2u" ) == 0 ) ) { 

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 3, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 2, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }
  
      /* 6C8, 6C8' additional minus sign, R8 to R19 */
      for ( int i =  7; i <= 18; i++ ) { rot_mat_ti_eq_re ( t->R[i], -1., 3 ); }

      /* 12C4' additional minus sign, R36 to R47 */
      for ( int i = 35; i <= 46; i++ ) { rot_mat_ti_eq_re ( t->R[i], -1., 3 ); }
          
      memcpy ( t->rmid, t->rid, nrot * sizeof( int ) );
      memcpy ( t->IR[0][0], t->R[0][0], nrot * 9 * sizeof(double _Complex ) );

      /***********************************************************
       * multiply minus sign to IR irrep matrices
       ***********************************************************/
      if ( strcmp ( irrep, "T2u" ) == 0 ) {
        for ( int irot = 0; irot < nrot; irot++ ) rot_mat_ti_eq_re ( t->IR[irot], -1., 3 );
      }

    /***********************************************************
     * LG 2Oh irrep G1g, G1u
     ***********************************************************/
    } else if ( ( strcmp ( irrep, "G1g" ) == 0 ) || ( strcmp ( irrep, "G1u" ) == 0 ) ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 1, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }

      memcpy ( t->rmid, t->rid, nrot * sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], nrot * 4 * sizeof(double _Complex ) );

      /***********************************************************
       * multiply minus sign to IR irrep matrices
       ***********************************************************/
      if ( strcmp ( irrep, "G1u" ) == 0 ) {
        for ( int irot = 0; irot < nrot; irot++ ) rot_mat_ti_eq_re ( t->IR[irot], -1., 2 );
      }

    /***********************************************************
     * LG 2Oh irrep G2g, G2u
     ***********************************************************/
    } else if ( ( strcmp ( irrep, "G2g" ) == 0 ) || ( strcmp ( irrep, "G2u" ) == 0 ) ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 1, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }

      /* 6C8, 6C8' additional sign */
      for ( int i =  7; i <= 18; i++ ) { rot_mat_ti_eq_re ( t->R[i], -1., 2 ); }

      /* 12C4' additional sign */
      for ( int i = 35; i <= 46; i++ ) { rot_mat_ti_eq_re ( t->R[i], -1., 2 ); }


      memcpy ( t->rmid, t->rid, nrot * sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], nrot * 4 * sizeof(double _Complex ) );

      /***********************************************************
       * multiply minus sign to IR irrep matrices
       ***********************************************************/
      if ( strcmp ( irrep, "G2u" ) == 0 ) {
        for ( int irot = 0; irot < nrot; irot++ ) rot_mat_ti_eq_re ( t->IR[irot], -1., 2 );
      }

    /***********************************************************
     * LG 2Oh irrep H
     ***********************************************************/
    } else if ( ( strcmp ( irrep, "Hg" ) == 0 ) || ( strcmp ( irrep, "Hu" ) == 0 ) ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 4, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        rot_rotation_matrix_spherical_basis ( t->R[i], 3, cubic_group_double_cover_rotations[i].n, cubic_group_double_cover_rotations[i].w );
      }

      memcpy ( t->rmid, t->rid, nrot * sizeof(int) );
      memcpy ( t->IR[0][0], t->R[0][0], nrot * 16 * sizeof(double _Complex ) );

      /***********************************************************
       * multiply minus sign to IR irrep matrices
       ***********************************************************/
      if ( strcmp ( irrep, "Hu" ) == 0 ) {
        for ( int irot = 0; irot < nrot; irot++ ) rot_mat_ti_eq_re ( t->IR[irot], -1., 4 );
      }

    } else {
      fprintf(stderr, "[set_rot_mat_table_cubic_double_cover] unknown irrep name %s\n", irrep );
      return(1);
    }

  /***********************************************************
   * LG 2C4v
   ***********************************************************/
  } else if ( strcmp ( group, "2C4v" ) == 0 ) {

    const int nrot = 8;
    int rid[nrot]  = {  0,  3,  6,  9, 12, 15, 18, 47 };
    int rmid[nrot] = {  1,  2,  4,  5, 37, 38, 43, 44 };

    /***********************************************************
     * LG 2C4v irrep A1
     ***********************************************************/
    if ( strcmp ( irrep, "A1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot*sizeof(int) );
      memcpy ( t->rmid, rmid, nrot*sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /***********************************************************
     * LG 2C4v irrep A2
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

    /***********************************************************
     * LG 2C4v irrep B1
     ***********************************************************/
    } else if ( strcmp ( irrep, "B1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

      /* 2C8' */
      t->R[3][0][0]  = -1.;
      t->R[4][0][0]  = -1.;
      /* 2C8 */
      t->R[5][0][0]  = -1.;
      t->R[6][0][0]  = -1.;

      /* 4IC4' */
      t->IR[4][0][0] = -1.;
      t->IR[5][0][0] = -1.;
      t->IR[6][0][0] = -1.;
      t->IR[7][0][0] = -1.;

    /***********************************************************
     * LG 2C4v irrep B2
     ***********************************************************/
    } else if ( strcmp ( irrep, "B2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

      /* 2C8' */
      t->R[3][0][0]  = -1.;
      t->R[4][0][0]  = -1.;
      /* 2C8 */
      t->R[5][0][0]  = -1.;
      t->R[6][0][0]  = -1.;

      /* 4IC4' */
      t->IR[0][0][0] = -1.;
      t->IR[1][0][0] = -1.;
      t->IR[2][0][0] = -1.;
      t->IR[3][0][0] = -1.;

    /***********************************************************
     * LG 2C4v irrep E
     ***********************************************************/
    } else if ( strcmp ( irrep, "E" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      /* I R1, J R48 1 */
      t->R[0][0][0]  =  1; t->R[0][1][1]  =  1;
      t->R[7][0][0]  =  1; t->R[7][1][1]  =  1;

      /* 2C4 R4,R7 -1 */
      t->R[1][0][0]  = -1; t->R[1][1][1]  = -1;
      t->R[2][0][0]  = -1; t->R[2][1][1]  = -1;

      /* 2C8' R10,R13 */
      t->R[3][0][0]  =  I; t->R[3][1][1]  = -I;
      t->R[4][0][0]  = -I; t->R[4][1][1]  =  I;

      /* 2C8 R16,R19 */
      t->R[5][0][0]  = -I; t->R[5][1][1]  =  I;
      t->R[6][0][0]  =  I; t->R[6][1][1]  = -I;

      /* 4IC4 IR2,IR3,IR5,IR6 */
      t->IR[0][0][1] =  1; t->IR[0][1][0]  =  1;
      t->IR[1][0][1] = -1; t->IR[1][1][0]  = -1;
      t->IR[2][0][1] =  1; t->IR[2][1][0]  =  1;
      t->IR[3][0][1] = -1; t->IR[3][1][0]  = -1;

      /* 4IC4' IR38,IR39,IR44,IR45 */
      t->IR[4][0][1] =  I; t->IR[4][1][0]  = -I;
      t->IR[5][0][1] = -I; t->IR[5][1][0]  =  I;
      t->IR[6][0][1] =  I; t->IR[6][1][0]  = -I;
      t->IR[7][0][1] = -I; t->IR[7][1][0]  =  I;

    /***********************************************************
     * LG 2C4v irrep G1
     ***********************************************************/
    } else if ( strcmp ( irrep, "G1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        int k = t->rid[i];
        rot_rotation_matrix_spherical_basis ( t->R[i], 1, cubic_group_double_cover_rotations[k].n, cubic_group_double_cover_rotations[k].w );
        k = t->rmid[i];
        rot_rotation_matrix_spherical_basis ( t->IR[i], 1, cubic_group_double_cover_rotations[k].n, cubic_group_double_cover_rotations[k].w );
        rot_mat_ti_eq_re ( t->IR[i], -1., 2);
      }

    /***********************************************************
     * LG 2C4v irrep G2
     ***********************************************************/
    } else if ( strcmp ( irrep, "G2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        int k = t->rid[i];
        rot_rotation_matrix_spherical_basis ( t->R[i], 1, cubic_group_double_cover_rotations[k].n, cubic_group_double_cover_rotations[k].w );
        k = t->rmid[i];
        rot_rotation_matrix_spherical_basis ( t->IR[i], 1, cubic_group_double_cover_rotations[k].n, cubic_group_double_cover_rotations[k].w );
      }

      /* 2C8' */
      rot_mat_ti_eq_re ( t->R[3], -1., 2);
      rot_mat_ti_eq_re ( t->R[4], -1., 2);
      /* 2C8 */
      rot_mat_ti_eq_re ( t->R[5], -1., 2);
      rot_mat_ti_eq_re ( t->R[6], -1., 2);

      /* 4IC4' */
      rot_mat_ti_eq_re ( t->IR[4], -1., 2);
      rot_mat_ti_eq_re ( t->IR[5], -1., 2);
      rot_mat_ti_eq_re ( t->IR[6], -1., 2);
      rot_mat_ti_eq_re ( t->IR[7], -1., 2);

    }

  /***********************************************************
   * LG 2C2v
   ***********************************************************/
  } else if ( strcmp ( group, "2C2v" ) == 0 ) {

    const int nrot = 4;
    int rid[nrot]  = {  0,  37, 43, 47 };
    int rmid[nrot] = {  3,   6, 38, 44 };

    /***********************************************************
     * LG 2C2v irrep A1
     ***********************************************************/
    if ( strcmp ( irrep, "A1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /***********************************************************
     * LG 2C2v irrep A2
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

    /***********************************************************
     * LG 2C2v irrep B1
     ***********************************************************/
    } else if ( strcmp ( irrep, "B1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

        t->R[0][0][0]  =  1.;
        t->R[1][0][0]  = -1.;
        t->R[2][0][0]  = -1.;
        t->R[3][0][0]  =  1.;

        t->IR[0][0][0] = -1.;
        t->IR[1][0][0] = -1.;
        t->IR[2][0][0] = +1.;
        t->IR[3][0][0] = +1.;

    /***********************************************************
     * LG 2C2v irrep B2
     ***********************************************************/
    } else if ( strcmp ( irrep, "B2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

        t->R[0][0][0]  =  1.;
        t->R[1][0][0]  = -1.;
        t->R[2][0][0]  = -1.;
        t->R[3][0][0]  =  1.;

        t->IR[0][0][0] = +1.;
        t->IR[1][0][0] = +1.;
        t->IR[2][0][0] = -1.;
        t->IR[3][0][0] = -1.;
    
    /***********************************************************
     * LG 2C2v irrep G1
     ***********************************************************/
    } else if ( strcmp ( irrep, "G1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        int k = t->rid[i];
        rot_rotation_matrix_spherical_basis ( t->R[i], 1, cubic_group_double_cover_rotations[k].n, cubic_group_double_cover_rotations[k].w );
        k = t->rmid[i];
        rot_rotation_matrix_spherical_basis ( t->IR[i], 1, cubic_group_double_cover_rotations[k].n, cubic_group_double_cover_rotations[k].w );
        rot_mat_ti_eq_re ( t->IR[i], -1., 2);
      }
    }

  /***********************************************************
   * LG 2C3v
   ***********************************************************/
  } else if ( strcmp ( group, "2C3v" ) == 0 ) {
    const int nrot = 6;
    int rid[nrot]  = {  0, 19, 23, 27, 31, 47 };
    int rmid[nrot] = { 36, 44, 46, 38, 40, 42 };

    /***********************************************************
     * LG 2C3v irrep A1
     ***********************************************************/
    if ( strcmp ( irrep, "A1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /***********************************************************
     * LG 2C3v irrep A2
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

    /***********************************************************
     * LG 2C3v irrep K1
     ***********************************************************/
    } else if ( strcmp ( irrep, "K1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      /* I */
      t->R[0][0][0]  =  1.;
      /* 2C6 */
      t->R[1][0][0]  = -1.;
      t->R[2][0][0]  = -1.;
      /* 2C3 */
      t->R[3][0][0]  =  1.;
      t->R[4][0][0]  =  1.;
      /* J */
      t->R[5][0][0]  = -1.;

      /* 3IC4 */
      t->IR[0][0][0] =  I;
      t->IR[1][0][0] =  I;
      t->IR[2][0][0] =  I;
      /* 3IC4' */
      t->IR[3][0][0] = -I;
      t->IR[4][0][0] = -I;
      t->IR[5][0][0] = -I;

    /***********************************************************
     * LG 2C3v irrep K2
     ***********************************************************/
    } else if ( strcmp ( irrep, "K2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      /* I */
      t->R[0][0][0]  =  1.;
      /* 2C6 */
      t->R[1][0][0]  = -1.;
      t->R[2][0][0]  = -1.;
      /* 2C3 */
      t->R[3][0][0]  =  1.;
      t->R[4][0][0]  =  1.;
      /* J */
      t->R[5][0][0]  = -1.;

      /* 3IC4 */
      t->IR[0][0][0] = -I;
      t->IR[1][0][0] = -I;
      t->IR[2][0][0] = -I;
      /* 3IC4' */
      t->IR[3][0][0] =  I;
      t->IR[4][0][0] =  I;
      t->IR[5][0][0] =  I;

    /***********************************************************
     * LG 2C3v irrep E
     ***********************************************************/
    } else if ( strcmp ( irrep, "E" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      /* I R1, J R48 */
      t->R[0][0][0]  =  1.; t->R[0][1][1]  =  1.;
      t->R[5][0][0]  =  1.; t->R[5][1][1]  =  1.;

      /* 2C6 R20,R24 */
      t->R[1][0][0]  = -0.5;
      t->R[1][1][1]  = -0.5;
      t->R[1][0][1]  = -SQRT3_HALF;
      t->R[1][1][0]  =  SQRT3_HALF;

      t->R[2][0][0]  = -0.5;
      t->R[2][1][1]  = -0.5;
      t->R[2][0][1]  =  SQRT3_HALF;
      t->R[2][1][0]  = -SQRT3_HALF;

      /* 2C3 R28,R32 */
      t->R[3][0][0]  = -0.5;
      t->R[3][1][1]  = -0.5;
      t->R[3][0][1]  =  SQRT3_HALF;
      t->R[3][1][0]  = -SQRT3_HALF;

      t->R[4][0][0]  = -0.5;
      t->R[4][1][1]  = -0.5;
      t->R[4][0][1]  = -SQRT3_HALF;
      t->R[4][1][0]  =  SQRT3_HALF;

      /* 3IC4 IR37,IR45,IR47 */
      t->IR[0][0][0]  =  0.5;
      t->IR[0][1][1]  = -0.5;
      t->IR[0][0][1]  =  SQRT3_HALF;
      t->IR[0][1][0]  =  SQRT3_HALF;

      t->IR[1][0][0]  = -1.;
      t->IR[1][1][1]  =  1.;

      t->IR[2][0][0]  =  0.5;
      t->IR[2][1][1]  = -0.5;
      t->IR[2][0][1]  = -SQRT3_HALF;
      t->IR[2][1][0]  = -SQRT3_HALF;

      t->IR[3][0][0]  = -1.;
      t->IR[3][1][1]  =  1.;

      t->IR[4][0][0]  =  0.5;
      t->IR[4][1][1]  = -0.5;
      t->IR[4][0][1]  = -SQRT3_HALF;
      t->IR[4][1][0]  = -SQRT3_HALF;

      t->IR[5][0][0]  =  0.5;
      t->IR[5][1][1]  = -0.5;
      t->IR[5][0][1]  =  SQRT3_HALF;
      t->IR[5][1][0]  =  SQRT3_HALF;

    /***********************************************************
     * LG 2C3v irrep G1
     ***********************************************************/
    } else if ( strcmp ( irrep, "G1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_double_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        int k = t->rid[i];
        rot_rotation_matrix_spherical_basis ( t->R[i], 1, cubic_group_double_cover_rotations[k].n, cubic_group_double_cover_rotations[k].w );
        k = t->rmid[i];
        rot_rotation_matrix_spherical_basis ( t->IR[i], 1, cubic_group_double_cover_rotations[k].n, cubic_group_double_cover_rotations[k].w );
        rot_mat_ti_eq_re ( t->IR[i], -1., 2);
      }

    }
  } else {
    fprintf(stderr, "[set_rot_mat_table_cubic_double_cover] unknown group name %s\n", group );
    return(1);
  }
 
  return(0);
}  // end of set_rot_mat_table_cubic_group_double_cover
