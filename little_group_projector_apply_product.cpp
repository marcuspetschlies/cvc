/***********************************************************
 * application to projeciton to tensor operator
 * product
 ***********************************************************/

void product_vector_printf ( double _Complex *v, int*dim, int n, char*name, FILE*ofs ) {

  const double eps = 9.e-15;
  int pdim = 1;
  for ( int i = 0; i < n; i++ ) pdim*=dim[i];
  int * coords = init_1level_itable ( n );

  fprintf( ofs, "# [product_vector_printf] %s\n", name);
  fprintf( ofs, "   %s <- array( dim=c( %d", name, dim[0]);
  for ( int i = 1; i < n; i++ ) fprintf( ofs, ", %d ", dim[i] );
  fprintf( ofs, ") )\n");

  for ( int idx = 0; idx < pdim; idx++ ) {
    product_vector_index2coords ( idx, coords, dim, n );
    fprintf( ofs, "   %s[ %d", name, coords[0]+1);
    for ( int i = 1; i < n; i++ ) fprintf( ofs, ", %2d ", coords[i]+1 );
    fprintf( ofs, "] <- %25.16e + %25.16e*1.i\n", dgeps( creal(v[idx]), eps), dgeps( cimag(v[idx]),eps ) );
  }

  fini_1level_itable ( &coords );
  return;
}  // end of function product_vector_printf


/***********************************************************/
/***********************************************************/

void product_vector_project_accum ( double _Complex * const v, rot_mat_table_type * const r, int const rid, int const rmid, double _Complex  * const v0, double _Complex const c1, double _Complex const c2, int * const dim , int const n ) {
  
  int pdim =1;

  for ( int i = 0; i < n; i++ ) pdim *= dim[i];

  int ** coords = init_2level_itable ( pdim, n );
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );

  for ( int idx = 0; idx < pdim; idx++ ) {

    double _Complex res = 0.;

    for ( int kdx = 0; kdx < pdim; kdx++ ) {
      double _Complex a = 1.;
      if ( rid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].R[rid][coords[idx][l]][coords[kdx][l]];
      } else if ( rmid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].IR[rmid][coords[idx][l]][coords[kdx][l]];
      } else { a = 0.; }

      res += a * v0[kdx];
    }
    v[idx] = c2 * v[idx] + c1 * res;
  }

  fini_2level_itable ( &coords );
  return;
}  // end of product_vector_project_accum


/***********************************************************/
/***********************************************************/

void product_mat_pl_eq_mat_ti_co ( double _Complex ** const R, rot_mat_table_type * const r, int const rid, int const rmid, double _Complex const c, int * const dim, int const n ) {
  
  int pdim =1;

  for ( int i = 0; i < n; i++ ) pdim *= dim[i];

  int ** coords = init_2level_itable ( pdim, n );
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );

  for ( int idx  = 0; idx < pdim; idx++ ) {
    for ( int kdx  = 0; kdx < pdim; kdx++ ) {
      double _Complex a = 1.;
      if ( rid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].R[rid][coords[idx][l]][coords[kdx][l]];
        /* fprintf ( stdout, "# [product_mat_pl_eq_mat_ti_co] idx %3d kdx %3d a %25.16e %25.16e\n", idx, kdx, creal(a), cimag(a) ); */
      } else if ( rmid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].IR[rmid][coords[idx][l]][coords[kdx][l]];
      } else { a = 0.;}
      R[idx][kdx] += c * a;
    }
  }
  
  fini_2level_itable ( &coords );
  return;
}  // end of product_mat_pl_eq_mat_ti_co

/***********************************************************/
/***********************************************************/

void rot_mat_table_eq_product_mat_table ( rot_mat_table_type * const r, rot_mat_table_type * const s, int const n ) {
  
  int dim[n];
  int pdim =1;

  for ( int i = 0; i < n; i++ ) {
    dim[i] = s[i].dim;
    pdim *= dim[i];

  }
  int nrot = s[0].n;

  init_rot_mat_table ( r );

  alloc_rot_mat_table ( r, "NA", "NA", pdim, nrot );

  int ** coords = init_2level_itable ( pdim, n );
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );

  for ( int irot = 0; irot < nrot; irot++ ) {

    for ( int idx  = 0; idx < pdim; idx++ ) {
      for ( int kdx  = 0; kdx < pdim; kdx++ ) {
        double _Complex a = 1.;
        for ( int l = 0; l < n; l++ ) a *= s[l].R[irot][coords[idx][l]][coords[kdx][l]];
        r->R[irot][idx][kdx] = a;
      }
    }

    for ( int idx  = 0; idx < pdim; idx++ ) {
      for ( int kdx  = 0; kdx < pdim; kdx++ ) {
        double _Complex a = 1.;
        for ( int l = 0; l < n; l++ ) a *= s[l].IR[irot][coords[idx][l]][coords[kdx][l]];
        r->IR[irot][idx][kdx] = a;
      }
    }
  }
  
  fini_2level_itable ( &coords );
  return;
}  // end of product_mat_pl_eq_mat_ti_co

/***********************************************************/
/***********************************************************/

/***********************************************************
 * mixed product
 * R is rotation matrix, dim = pdim
 * r is list of spin matrices, dim = dim[i]
 * s is rotation matrix, dim = pdim
 ***********************************************************/
void rot_mat_eq_product_mat_ti_rot_mat ( double _Complex ** const R, rot_mat_table_type * const r, int const rid, int const rmid, double _Complex ** const S, int const n ) {

  int pdim =1;

  int * dim = init_1level_itable ( n );
  for ( int i = 0; i < n; i++ ) {
    dim[i] = r[i].dim;
    pdim *= r[i].dim;
  }

  int ** coords = init_2level_itable ( pdim, n );
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );
  memset ( R[0], 0, pdim*pdim*sizeof(double _Complex) );

  for ( int idx  = 0; idx < pdim; idx++ ) {
    for ( int kdx  = 0; kdx < pdim; kdx++ ) {
      double _Complex a = 1.;
      if ( rid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].R[rid][coords[idx][l]][coords[kdx][l]];
      } else if ( rmid > -1 ) {
        for ( int l = 0; l < n; l++ ) a *= r[l].IR[rmid][coords[idx][l]][coords[kdx][l]];
      } else { a = 0.;}

      for ( int ldx  = 0; ldx < pdim; ldx++ ) {
        R[idx][ldx] += a * S[kdx][ldx];
      }

    }
  }
  
  fini_2level_itable ( &coords );
  fini_1level_itable ( &dim );
  return;
}  // end of product_mat_ti_mat


/***********************************************************/
/***********************************************************/

/***********************************************************
 * print a direct product matrix
 ***********************************************************/
int product_mat_printf ( double _Complex ** const R, int * const dim, int const n, const char *name, FILE  * const ofs ) {

  const double eps = 9.e-15;
  int pdim =1;
  for ( int i = 0; i < n; i++ ) pdim *= dim[i];

  int ** coords = init_2level_itable ( pdim, n );
  if ( coords == NULL ) {
    fprintf (stderr, "[product_mat_printf] Error from init_2level_itable\n");
    return(1);
  }
  for ( int i=0; i < pdim; i++ ) product_vector_index2coords ( i, coords[i], dim, n );

  fprintf ( ofs, "# [product_mat_printf] %s\n", name);
  fprintf ( ofs, "%s <- array( dim=c( %d", name, dim[0] );
  for ( int i = 1; i < n; i++ ) fprintf( ofs, ", %d ", dim[i] );
  for ( int i = 0; i < n; i++ ) fprintf( ofs, ", %d ", dim[i] );
  fprintf ( ofs, "))\n" );

  for ( int idx  = 0; idx < pdim; idx++ ) {

    for ( int kdx  = 0; kdx < pdim; kdx++ ) {

      fprintf ( ofs, "%s[%2d", name, coords[idx][0]+1 );
      for ( int i = 1; i < n; i++ ) fprintf( ofs, ", %2d ", coords[idx][i]+1 );
      for ( int i = 0; i < n; i++ ) fprintf( ofs, ", %2d ", coords[kdx][i]+1 );
      fprintf ( ofs, "] <- %25.16e + %25.16e*1.i\n", 
          dgeps ( creal( R[idx][kdx] ), eps), dgeps ( cimag( R[idx][kdx] ), eps) );
    }
  }

  fini_2level_itable ( &coords );
  return(0);
}  // end of product_mat_printf

/***********************************************************/
/***********************************************************/

/***********************************************************
 * apply the projector to a product state
 ***********************************************************/
little_group_projector_applicator_type * little_group_projector_apply_product ( little_group_projector_type * const p , FILE * const ofs) {

  int exitstatus;
  char name[20];
  int frame_is_cmf = ( p->P[0] == 0 && p->P[1] == 0 && p->P[2] == 0 );
  int pdim = 1;


  /***********************************************************
   * allocate spin vectors, to which spin rotations are applied
   ***********************************************************/
  int * spin_dimensions = init_1level_itable ( p->n );
  if ( spin_dimensions == NULL ) {
    fprintf ( stderr, "# [little_group_projector_apply_product] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  for ( int i = 0; i < p->n; i++ ) {
    spin_dimensions[i] = p->rspin[i].dim;
    pdim *= p->rspin[i].dim;
  }
  fprintf ( stdout, "# [little_group_projector_apply_product] spinor product dimension = %d\n", pdim );

  double _Complex * sv0 = init_1level_ztable( pdim );
  if ( sv0 == NULL ) {
    fprintf ( stderr, "# [little_group_projector_apply_product] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize spin vectors according to ref_row_spin
   * only for 2 interpolators
   ***********************************************************/
  if ( ( p->n == 2  ) && ( p->ref_row_spin[0] < 0 ) && ( p->ref_row_spin[1] <= 0 ) ) {
    int const J2_1 = rot_mat_table_get_spin2 ( &(p->rspin[0]) );
    int const J2_2 = rot_mat_table_get_spin2 ( &(p->rspin[1]) );

    int const J2_3 = -p->ref_row_spin[0];
    int const M2_3 = J2_3 + 2*p->ref_row_spin[1];

    const int bispinor[2] = {
      rot_mat_table_get_bispinor ( &(p->rspin[0]) ),
      rot_mat_table_get_bispinor ( &(p->rspin[1]) ) };

    /***********************************************************
     * use Clebsch-Gordan coefficients
     ***********************************************************/
    for ( int i1 = 0; i1 <= J2_1; i1++ ) {
      int const M2_1 = J2_1 - 2*i1;
    for ( int i2 = 0; i2 <= J2_2; i2++ ) {
      int const M2_2 = J2_2 - 2*i2;
      fprintf ( ofs, "# [little_group_projector_apply_product] J2_1 = %2d M2_1 = %2d   J2_2 = %2d M2_2 = %2d   J2_3 = %2d M2_3 = %2d  bispinor %d %d\n", J2_1, M2_1, J2_2, M2_2, J2_3, M2_3,
         bispinor[0], bispinor[1] );
      for ( int j1 = 0; j1 <= bispinor[0]; j1++ ) {
        for ( int j2 = 0; j2 <= bispinor[1]; j2++ ) {
          const int coords[2] = {i1+j1*(J2_1+1), i2+j2*(J2_2+1)};
          sv0[ product_vector_coords2index ( coords, spin_dimensions, 2 ) ] = clebsch_gordan_coeff ( J2_3, M2_3, J2_1, M2_1, J2_2, M2_2 );
        }
      }
    }}
  } else {
    product_vector_set_element ( sv0, 1.0, p->ref_row_spin, spin_dimensions, p->n );
  }

  product_vector_printf ( sv0, spin_dimensions, p->n,  "v0", ofs );

  /***********************************************************
   * set up subduction matrix disregarding momenta
   ***********************************************************/
  rot_mat_table_type RR;
  init_rot_mat_table ( &RR );
  exitstatus = alloc_rot_mat_table ( &RR, "NA", "NA", pdim, p->rtarget->dim );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[little_group_projector_apply_product] Error from alloc_rot_mat_table, status was %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
    return(1);
  }

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * loop on rows of target irrep
   ***********************************************************/
  for ( int row_target = 0; row_target < p->rtarget->dim; row_target++ ) {


    /***********************************************************
     * allocate sv1
     ***********************************************************/
    double _Complex ** sv1 = init_2level_ztable ( p->rtarget->dim, pdim );
    if ( sv1 == NULL ) {
      fprintf ( stderr, "[little_group_projector_apply_product] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
      return(1);
    }


    /***********************************************************/
    /***********************************************************/

    double _Complex *Rsv = init_1level_ztable ( pdim );

    /***********************************************************
     * TEST
     ***********************************************************/
    double _Complex **R = rot_init_rotation_matrix ( pdim );
    /***********************************************************
     * END OF TEST
     ***********************************************************/

    /***********************************************************
     * loop on rotation group elements R
     ***********************************************************/
    for ( int irot = 0; irot < p->rtarget->n; irot++ ) {

      fprintf ( stdout, "# [little_group_projector_apply_product] lg %s irrep %s irot %2d rid %2d\n", p->rtarget->group, p->rtarget->irrep, irot,
          p->rtarget->rid[irot] );

      /* This is choice according to my notes */
      double _Complex z_irrep_matrix_coeff = conj ( p->rtarget->R[irot][row_target][p->ref_row_target] );


      /***********************************************************
       * add spin rotation x sv0 as product
       ***********************************************************/
      product_vector_project_accum ( Rsv, p->rspin, irot, -1, sv0, z_irrep_matrix_coeff, 1., spin_dimensions, p->n );

      /***********************************************************
       * TEST
       ***********************************************************/
      product_mat_pl_eq_mat_ti_co ( R, p->rspin, irot, -1, z_irrep_matrix_coeff, spin_dimensions, p->n );
      /***********************************************************
       * END OF TEST
       ***********************************************************/

    }  /* end of loop on rotations R */

    /***********************************************************
     * TEST
     ***********************************************************/
    sprintf ( name, "vsub[[%d]]", row_target+1 );
    product_vector_printf ( Rsv, spin_dimensions, p->n, name,  ofs  );
    /***********************************************************
     * END OF TEST
     ***********************************************************/


    /***********************************************************
     * TEST
     ***********************************************************/
    sprintf ( name, "Rsub[[%d]]", row_target+1 );
    product_mat_printf ( R, spin_dimensions, p->n, name, ofs );

    rot_mat_assign ( RR.R[row_target], R, pdim );
    rot_fini_rotation_matrix ( &R );

    /***********************************************************
     * END OF TEST
     ***********************************************************/

    /***********************************************************
     * not center of mass frame, include IR rotations
     ***********************************************************/
    // if ( !frame_is_cmf )  { 
      if ( g_verbose > 2 ) fprintf( stdout, "# [little_group_projector_apply_product] including IR rotations\n");

      double _Complex *IRsv = init_1level_ztable( pdim );

      /***********************************************************
       * TEST
       ***********************************************************/
      double _Complex **IR = rot_init_rotation_matrix ( pdim );
      /***********************************************************
       * END OF TEST
       ***********************************************************/


      /***********************************************************
       * loop on rotation group elements IR
       ***********************************************************/
      for ( int irot = 0; irot < p->rtarget->n; irot++ ) {

        fprintf ( stdout, "# [little_group_projector_apply_product] lg %s irrep %s irot %2d rmid %2d\n",
            p->rtarget->group, p->rtarget->irrep, irot, p->rtarget->rmid[irot] );

        /* This is choice according to my notes */
        double _Complex z_irrep_matrix_coeff = conj ( p->rtarget->IR[irot][row_target][p->ref_row_target] );

        /* TEST */
        /* fprintf(stdout, "# [little_group_projector_apply_product] T Gamma (IR) coeff rot %2d = %25.16e %25.16e\n", rmid, creal(z_irrep_matrix_coeff), cimag(z_irrep_matrix_coeff) ); */

        /***********************************************************
         * add rotation-reflection applied to sv0 as product
         ***********************************************************/
        //STOPPED HERE
        //  include intrinsic parity
        product_vector_project_accum ( IRsv, p->rspin, -1, irot, sv0, z_irrep_matrix_coeff, 1., spin_dimensions, p->n );

        /***********************************************************
         * TEST
         ***********************************************************/
        product_mat_pl_eq_mat_ti_co ( IR, p->rspin, -1, irot, z_irrep_matrix_coeff, spin_dimensions, p->n );
        /***********************************************************
         * END OF TEST
         ***********************************************************/

      }  /* end of loop on rotations IR */

      /***********************************************************
       * TEST
       ***********************************************************/
      sprintf ( name, "Ivsub[[%d]]", row_target+1 );
      product_vector_printf ( IRsv, spin_dimensions, p->n, name,  ofs  );
      /***********************************************************
       * END OF TEST
       ***********************************************************/

      /***********************************************************
       * add IRsv to Rsv, normalize
       ***********************************************************/

      rot_vec_pl_eq_vec_ti_co ( Rsv, IRsv, 1.0, pdim );


      /***********************************************************
       * TEST
       ***********************************************************/
      sprintf ( name, "IRsub[[%d]]", row_target+1 );
      product_mat_printf ( IR, spin_dimensions, p->n, name, ofs );

      rot_mat_pl_eq_mat_ti_co ( RR.R[row_target], IR, 1.0,  pdim );

      rot_fini_rotation_matrix ( &IR );
      /***********************************************************
       * END OF TEST
       ***********************************************************/

      fini_1level_ztable( &IRsv );
    // }  /* end of if not center of mass frame */

    /***********************************************************
     * normalize Rsv+IRsv, show Cvsub
     ***********************************************************/
    rot_vec_normalize ( Rsv, pdim );
    sprintf ( name, "Cvsub[[%d]]", row_target+1 );
    product_vector_printf ( Rsv, spin_dimensions, p->n, name, ofs );

    /***********************************************************/
    /***********************************************************/

    /***********************************************************
     * TEST
     ***********************************************************/

    rot_mat_ti_eq_re ( RR.R[row_target], p->rtarget->dim /( 2. * p->rtarget->n ), pdim);
    sprintf ( name, "RRsub[[%d]]", row_target+1 );
    product_mat_printf ( RR.R[row_target], spin_dimensions, p->n, name, ofs );


    /***********************************************************
     * END OF TEST
     ***********************************************************/

    fini_1level_ztable( &Rsv );

  }  // end of loop on row_target

  /***********************************************************/
  /***********************************************************/


  /***********************************************************
   * check rotation properties of RR, deallocate RR
   ***********************************************************/
  exitstatus = rot_mat_table_rotate_multiplett_product ( &RR, p->rspin, p->rtarget, p->n, !frame_is_cmf, ofs );
  if ( exitstatus != 0 ) {
    fprintf( stderr, "[little_group_projector_apply_product] Error from rot_mat_table_rotate_multiplett_product, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  fini_rot_mat_table ( &RR );


  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * deallocate sv1
   ***********************************************************/
  fini_2level_ztable ( &sv1 );

  fini_1level_itable ( &spin_dimensions );
  fini_1level_ztable( &sv0 );
  return(0);

}  /* end of little_group_projector_apply_product */

/***********************************************************/
/***********************************************************/


/***********************************************************
 * rotate a projection matrix multiplett for product matrix
 ***********************************************************/
int rot_mat_table_rotate_multiplett_product ( 
    rot_mat_table_type *rtab,
    rot_mat_table_type *rapply,
    rot_mat_table_type *rtarget,
    int n, int with_IR, FILE*ofs
) {

  const double eps = 2.e-14;
  int pdim = 1;
  for ( int i = 0; i < n; i++ ) pdim *= rapply[i].dim;

  if ( rtab->dim != pdim ) {
    fprintf(stderr, "[rot_mat_table_rotate_multiplett_product] Error, incompatible dimensions\n");
    return(1);
  }

  if ( rtab->n != rtarget->dim ) {
    fprintf(stderr, "[rot_mat_table_rotate_multiplett_product] Error, incompatible number of rotations in rtab and matrix dimension in rtarget\n");
    return(2);
  }
  
  /***********************************************************/
  /***********************************************************/

  for ( int i = 0; i < n; i++ ) 
    fprintf ( ofs, "# [rot_mat_table_rotate_multiplett_product] using rapply(%d) %s / %s for rtarget %s / %s\n", i, rapply[i].group, rapply[i].irrep, rtarget->group, rtarget->irrep );
  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * loop on elements in rtab
   ***********************************************************/
  for ( int ia = 0; ia < rtab->n; ia++ ) {

    /***********************************************************
     * loop on rotations = elements of rtarget
     *
     * R rotations
     ***********************************************************/
    for ( int irot = 0; irot < rtarget->n; irot++ ) {

      double _Complex **R2 = rot_init_rotation_matrix ( pdim );
      double _Complex **R3 = rot_init_rotation_matrix ( pdim );

      rot_mat_eq_product_mat_ti_rot_mat ( R2, rapply, irot, -1, rtab->R[ia], n );

      for ( int k = 0; k < rtarget->dim; k++ ) {
        rot_mat_pl_eq_mat_ti_co ( R3, rtab->R[k], rtarget->R[irot][k][ia], rtab->dim );
      }

      char name[100];
      sprintf( name, "R2[[%2d]]", rtarget->rid[irot] );
      rot_printf_matrix ( R2, rtab->dim, name, ofs );
      fprintf(ofs, "\n");
      sprintf( name, "R3[[%2d]]", rtarget->rid[irot] );
      rot_printf_matrix ( R3, rtab->dim, name, ofs );
      double norm = rot_mat_norm_diff ( R2, R3, rtab->dim );
      double norm2 = sqrt( rot_mat_norm2 ( R2, rtab->dim ) );

      fprintf(ofs, "# [rot_mat_table_rotate_multiplett_product] irot %2d rid  %2d norm diff = %16.7e / %16.7e   %d\n\n",
          irot, rtarget->rid[irot], norm, norm2, fabs(norm)<eps );

      rot_fini_rotation_matrix ( &R2 );
      rot_fini_rotation_matrix ( &R3 );
    }

    /***********************************************************
     * IR rotations
     ***********************************************************/
    if ( with_IR ) {
      for ( int irot = 0; irot < rtarget->n; irot++ ) {

        double _Complex **R2 = rot_init_rotation_matrix ( rtab->dim );
        double _Complex **R3 = rot_init_rotation_matrix ( rtab->dim );

        rot_mat_eq_product_mat_ti_rot_mat ( R2, rapply, -1, irot, rtab->R[ia], n );

        for ( int k = 0; k < rtarget->dim; k++ ) {
          rot_mat_pl_eq_mat_ti_co ( R3, rtab->R[k], rtarget->IR[irot][k][ia], rtab->dim );
        }

        char name[100];
        sprintf( name, "IR2[[%2d]]", rtarget->rmid[irot] );
        rot_printf_matrix ( R2, rtab->dim, name, ofs );
        fprintf(ofs, "\n");
        sprintf( name, "IR3[[%2d]]", rtarget->rmid[irot] );
        rot_printf_matrix ( R3, rtab->dim, name, ofs );
        double norm = rot_mat_norm_diff ( R2, R3, rtab->dim );
        double norm2 = sqrt( rot_mat_norm2 ( R2, rtab->dim ) );
        fprintf(ofs, "# [rot_mat_table_rotate_multiplett_product] irot %2d rmid %2d norm diff = %16.7e / %16.7e   %d\n\n",
            irot, rtarget->rmid[irot], norm, norm2, fabs(norm)<eps );

        rot_fini_rotation_matrix ( &R2 );
        rot_fini_rotation_matrix ( &R3 );
      }
    }

  }  // end of loop on elements of rtab

  return(0);
}  // end of rot_mat_table_rotate_multiplett_product

/***********************************************************/
/***********************************************************/
