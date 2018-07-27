/***********************************************************/
/***********************************************************/

/***********************************************************
 * irrep matrices for single cover
 ***********************************************************/
int set_rot_mat_table_cubic_group_single_cover ( rot_mat_table_type *t, const char *group, const char*irrep ) {

  const double SQRT3_HALF = 0.5 * sqrt(3.);

  int exitstatus;

  /***********************************************************
   * LG Oh
   ***********************************************************/
  if ( strcmp ( group, "Oh" ) == 0 ) {
    int nrot = 24;
    
    /***********************************************************
     * LG Oh irrep A1
     ***********************************************************/
    if ( strcmp ( irrep, "A1g" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        t->rmid[i] = i;
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_A1g_v2.cpp"

    /***********************************************************
     * LG Oh irrep A1u
     ***********************************************************/
    } else if ( strcmp ( irrep, "A1u" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i] = i;
        t->rmid[i] = i;
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_A1u_v2.cpp"

    /***********************************************************
     * LG Oh irrep A2g
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2g" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) { 
        t->rid[i]      = i; 
        t->rmid[i]     = i; 
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_A2g_v2.cpp"

    /***********************************************************
     * LG Oh irrep A2u
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2u" ) == 0 ) {
      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) { 
        t->rid[i]      = i; 
        t->rmid[i]     = i; 
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_A2u_v2.cpp"

    /***********************************************************
     * LG Oh irrep Eg
     ***********************************************************/
    } else if ( strcmp ( irrep, "Eg" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) { 
        t->rid[i]  = i; 
        t->rmid[i] = i; 
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_Eg_v2.cpp"

    /***********************************************************
     * LG Oh irrep Eu
     ***********************************************************/
    } else if ( strcmp ( irrep, "Eu" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) { 
        t->rid[i]  = i; 
        t->rmid[i] = i; 
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_Eu_v2.cpp"

    /***********************************************************
     * LG Oh irrep T1g
     ***********************************************************/
    } else if ( strcmp ( irrep, "T1g" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 3, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i]  = i;
        t->rmid[i] = i;
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_T1g_v2.cpp"

    /***********************************************************
     * LG Oh irrep T1u
     ***********************************************************/
    } else if ( strcmp ( irrep, "T1u" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 3, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i]  = i;
        t->rmid[i] = i;
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_T1u_v2.cpp"
      
    /***********************************************************
     * LG Oh irrep T2g
     ***********************************************************/
    } else if ( strcmp ( irrep, "T2g" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 3, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i]  = i;
        t->rmid[i] = i;
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_T2g_v2.cpp"

    /***********************************************************
     * LG Oh irrep T2u
     ***********************************************************/
    } else if ( strcmp ( irrep, "T2u" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 3, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      for ( int i = 0; i < nrot; i++ ) {
        t->rid[i]  = i;
        t->rmid[i] = i;
      }
      double _Complex ***R = t->R;
      double _Complex ***IR = t->IR;
#include "./irrep_matrices/set_rot_mat_table_cubic_group_single_cover_Oh_T2u_v2.cpp"

    } else {
      fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] unknown irrep name %s\n", irrep );
      return(1);
    }

  /***********************************************************
   * LG C4v
   ***********************************************************/
  } else if ( strcmp ( group, "C4v" ) == 0 ) {

    int const nrot = 4;
    int rid[nrot]  = {  0, 14, 13, 23 };
    int rmid[nrot] = { 21, 22, 17, 18 };

    /***********************************************************
     * LG C4v irrep A1
     ***********************************************************/
    if ( strcmp ( irrep, "A1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot*sizeof(int) );
      memcpy ( t->rmid, rmid, nrot*sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /***********************************************************
     * LG C4v irrep A2
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

    /***********************************************************
     * LG C4v irrep B1
     ***********************************************************/
    } else if ( strcmp ( irrep, "B1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      t->R[0][0][0]  =  1.;
      t->R[1][0][0]  = -1.;
      t->R[2][0][0]  = -1.;
      t->R[3][0][0]  =  1.;

      t->IR[0][0][0] =  1.;
      t->IR[1][0][0] =  1.;
      t->IR[2][0][0] = -1.;
      t->IR[3][0][0] = -1.;

    /***********************************************************
     * LG C4v irrep B2
     ***********************************************************/
    } else if ( strcmp ( irrep, "B2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

      t->R[0][0][0]  =  1.;
      t->R[1][0][0]  = -1.;
      t->R[2][0][0]  = -1.;
      t->R[3][0][0]  =  1.;

      t->IR[0][0][0] = -1.;
      t->IR[1][0][0] = -1.;
      t->IR[2][0][0] =  1.;
      t->IR[3][0][0] =  1.;

    /***********************************************************
     * LG C4v irrep E
     ***********************************************************/
    } else if ( strcmp ( irrep, "E" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }

      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      /* I R1 */
      t->R[0][0][0]  =  1; t->R[0][1][1]  =  1;

      t->R[1][0][0]  = -I; t->R[1][1][1]  =  I;

      t->R[2][0][0]  =  I; t->R[2][1][1]  = -I;

      t->R[3][0][0]  = -1; t->R[3][1][1]  = -1;

      t->IR[0][0][1] = -1; t->IR[0][1][0] = -1;

      t->IR[1][0][1] =  1; t->IR[1][1][0] =  1;

      t->IR[2][0][1] =  I; t->IR[2][1][0] = -I;

      t->IR[3][0][1] = -I; t->IR[3][1][0] =  I;

    } else {
      fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] unknown irrep name %s\n", irrep );
      return(1);
    }

  /***********************************************************
   * LG C2v
   ***********************************************************/
  } else if ( strcmp ( group, "C2v" ) == 0 ) {

    int const nrot = 2;
    int rid[nrot]  = {  0, 17 };
    int rmid[nrot] = { 23, 18 };

    /***********************************************************
     * LG C2v irrep A1
     ***********************************************************/
    if ( strcmp ( irrep, "A1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /***********************************************************
     * LG C2v irrep A2
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

    /***********************************************************
     * LG C2v irrep B1
     ***********************************************************/
    } else if ( strcmp ( irrep, "B1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

        t->R[0][0][0]  =  1.;
        t->R[1][0][0]  = -1.;

        t->IR[0][0][0] = -1.;
        t->IR[1][0][0] =  1.;

    /***********************************************************
     * LG C2v irrep B2
     ***********************************************************/
    } else if ( strcmp ( irrep, "B2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

        t->R[0][0][0]  =  1.;
        t->R[1][0][0]  = -1.;

        t->IR[0][0][0] =  1.;
        t->IR[1][0][0] = -1.;

    } else {
      fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] unknown irrep name %s\n", irrep );
      return(1);
    }

  /***********************************************************
   * LG C3v
   ***********************************************************/
  } else if ( strcmp ( group, "C3v" ) == 0 ) {
    int const nrot = 3;
    int rid[nrot]  = {  0,  2,  1 };
    int rmid[nrot] = { 16, 20, 18 };

    /***********************************************************
     * LG C3v irrep A1
     ***********************************************************/
    if ( strcmp ( irrep, "A1" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot ) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  = 1.;
        t->IR[i][0][0] = 1.;
      }

    /***********************************************************
     * LG C3v irrep A2
     ***********************************************************/
    } else if ( strcmp ( irrep, "A2" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 1, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      for ( int i = 0; i < nrot; i++ ) {
        t->R[i][0][0]  =  1.;
        t->IR[i][0][0] = -1.;
      }

    /***********************************************************
     * LG C3v irrep E
     ***********************************************************/
    } else if ( strcmp ( irrep, "E" ) == 0 ) {

      if ( ( exitstatus = alloc_rot_mat_table ( t, group, irrep, 2, nrot) ) != 0 ) {
        fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] Error from alloc_rot_mat_table, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(1);
      }
      memcpy ( t->rid,  rid,  nrot * sizeof(int) );
      memcpy ( t->rmid, rmid, nrot * sizeof(int) );

      t->R[0][0][0]  =  1.; 
      t->R[0][1][1]  =  1.;

      t->R[1][0][0]  = -0.5 - SQRT3_HALF * I;
      t->R[1][1][1]  = -0.5 + SQRT3_HALF * I;

      t->R[2][0][0]  = -0.5 + SQRT3_HALF * I;
      t->R[2][1][1]  = -0.5 - SQRT3_HALF * I;

      t->IR[0][0][1] = -1.;
      t->IR[0][1][0] = -1.;

      t->IR[1][0][1] =  0.5 - SQRT3_HALF * I;
      t->IR[1][1][0] =  0.5 + SQRT3_HALF * I;

      t->IR[2][0][1] =  0.5 + SQRT3_HALF * I;
      t->IR[2][1][0] =  0.5 - SQRT3_HALF * I;

    } else {
      fprintf(stderr, "[set_rot_mat_table_cubic_group_single_cover] unknown irrep name %s\n", irrep );
      return(1);
    }

  } else {
    fprintf(stderr, "[set_rot_mat_table_cubic_double_cover] unknown group name %s\n", group );
    return(1);
  }
 
  return(0);
}  // end of set_rot_mat_table_cubic_group_single_cover
