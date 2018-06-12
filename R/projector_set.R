# ***********************************************************
# * projector_set.R
# *
# ***********************************************************

projector_set <- function(
    # ***********************************************************
    # * these are mandatory
    # ***********************************************************
    lg, 
    irrep,
    interpolator_num,
    interpolator_J2_list,
    # ***********************************************************
    # * these need not be set now, may control behavior later
    # ***********************************************************
    refframerot                 = NULL,
    row_target                  = NULL,
    interpolator_momentum_list  = NULL,
    interpolator_bispinor_list  = NULL,
    interpolator_parity_list    = NULL,
    interpolator_cartesian_list = NULL,
    ref_row_target              = NULL,
    ref_row_spin                = NULL,
    name                        = NULL
) {

  p <- list()

  if ( missing (interpolator_num) ) stop("need number of interpolating fields")
  p$n <- interpolator_num

  if ( missing ( lg ) || missing (irrep) ) stop("need little group, irrep and row")

  # ***********************************************************
  # * init fill target cubic group irrep rotation table
  # ***********************************************************
  p$rtarget <- set_rot_mat_table_cubic_group_double_cover( lg$group, irrep )
   
  # ***********************************************************
  # * set row of target irrep
  # ***********************************************************
  if ( !is.null ( row_target ) ) p$row_target <- row_target

  # ***********************************************************
  # * set reference row of target irrep, if provided
  # * (select irrep matrix column in projector,
  # *  we call it row as well)
  # ***********************************************************
  if ( !is.null ( ref_row_target ) ) p$ref_row_target <- ref_row_target

  # ***********************************************************
  # * reference row for each spin vector, if provided
  # ***********************************************************
  if ( !is.null ( ref_row_spin )  ) p$ref_row_spin <- ref_row_spin

  # ***********************************************************
  # * set intrinsic parity of operator, if provided
  # *
  # * if not provided, default is +1
  # ***********************************************************
  if ( !is.null ( interpolator_parity_list ) ) p$parity <- interpolator_parity_list
  else p$parity <- rep ( 1, times = p$n )

  # ***********************************************************
  # * set bispinor list
  # ***********************************************************
  if ( !is.null ( interpolator_bispinor_list ) ) p$interpolator_bispinor_list <- interpolator_bispinor_list
  else p$interpolator_bispinor_list <- rep( FALSE, times=p$n )

  # ***********************************************************
  # * set cartesian list
  # ***********************************************************
  if ( !is.null ( interpolator_cartesian_list ) ) p$interpolator_cartesian_list <- interpolator_cartesian_list
  else p$interpolator_cartesian_list <- rep( FALSE, times=p$n )

  # ***********************************************************
  # * set the correlator name, if provided;
  # * default "NA" should already be set
  # ***********************************************************
  if ( !is.null ( p$name ) ) p$name <- name

  # ***********************************************************
  # * prepare reference frame rotation
  # ***********************************************************

  if ( !is.null ( refframerot )) {
    rid <- refframerot
    p$refframerot <- list()
    p$refframerot$rid <- rid

    # set the reference frame rotation matrix
    # for the 3-momentum vector p;
    # spin 1 in cartesian basis
    p$refframerot$prot <- rot_mat_spin1_cartesian ( cubic_group_double_cover_rotations[[rid]]$n, cubic_group_double_cover_rotations[[rid]]$omega )

    if ( ! rot_mat_check_is_real_int ( p$refframerot$prot ) ) stop ( "rot_mat_check_is_real_int failed" )

    # set the reference frame rotation matrix
    # for the spin-J vectors
    p$refframerot$srot <- vector ( mode = "list" ) 

    for ( i in 1:p$n ) {

      if ( p$interpolator_bispinor_list[i] ) {
        p$refframerot$srot[[i]] <- rot_bispinor_rotation_matrix_spherical_basis ( cubic_group_double_cover_rotations[[rid]]$n, cubic_group_double_cover_rotations[[rid]]$w )
      } else {
        p$refframerot$srot[[i]] <- rot_rotation_matrix_spherical_basis ( interpolator_J2_list[i], cubic_group_double_cover_rotations[[rid]]$n, cubic_group_double_cover_rotations[[rid]]$w )
      }

      if ( p$interpolator_cartesian_list[i] && ( interpolator_J2_list[i] == 2 ) ) {
        p$refframerot$srot[[i]] <- rot_spherical2cartesian_3x3 ( p$refframerot$srot[[i]] );

        if ( ! rot_mat_check_is_real_int ( p$refframerot$srot[[i]] ) ) stop ( "rot_mat_check_is_real_int failed" )
      }
    }  # end of loop on interpolators

  }  # end of if refframerot

  # ***********************************************************
  # ***********************************************************

  # ***********************************************************
  # * set total momentum vector
  # *
  # * it is given by the little group's lg$d under consideration
  # ***********************************************************
  
  if ( !is.null ( p$refframerot ) ) {
    round ( p$refframerot$prot %*% lg$d )
  } else {
    p$P <- lg$d
  }

  # ***********************************************************
  # ***********************************************************

  # ***********************************************************
  # * set rotation matrix table for 3-momentum vector p
  # * for each interpolator
  # *
  # *
  # * transform p-3-vector rotations to cartesian basis
  # ***********************************************************
  p$rp <- list()
  p$rp$rid  <- p$rtarget$rid
  p$rp$rmid <- p$rtarget$rmid
  p$rp$R    <- vector(mode="list")
  p$rp$IR   <- vector(mode="list")

  for ( i in 1:length(p$rp$rid) ) {
    rid          <- p$rp$rid[i];
    p$rp$R[[i]]  <- rot_rotation_matrix_spherical_basis ( 2, cubic_group_double_cover_rotations[[rid]]$n, cubic_group_double_cover_rotations[[rid]]$w )
    p$rp$R[[i]]  <- rot_spherical2cartesian_3x3 ( p$rp$R[[i]]  );
    p$rp$R[[i]]  <- rot_mat_check_is_real_int ( p$rp$R[[i]] ) 

    rmid <- p$rp$rmid[i];
    p$rp$IR[[i]] <- rot_inversion_matrix_spherical_basis ( 2, FALSE  ) %*% rot_rotation_matrix_spherical_basis ( 2, cubic_group_double_cover_rotations[[rmid]]$n, cubic_group_double_cover_rotations[[rmid]]$w )
    p$rp$IR[[i]] <- rot_spherical2cartesian_3x3 ( p$rp$IR[[i]] );
    p$rp$IR[[i]] <- rot_mat_check_is_real_int ( p$rp$IR[[i]] )

    if ( !is.null ( p$refframerot ) ) {

      p$rp$R[[i]]  <- p$refframerot$prot %*% p$rp$R[[i]]  %*% t( p$refframerot$prot )
      p$rp$IR[[i]] <- p$refframerot$prot %*% p$rp$IR[[i]] %*% t( p$refframerot$prot )

    }  # end of use_refframerot

  }  # end of loop on rotation group elements in target irrep

  # ***********************************************************
  # * set rotation matrix table for each interpolator
  # ***********************************************************

  p$rspin <- vector ( mode ="list" )

  for (  i in 1:p$n ) {

    p$rspin[[i]] <- list()
    p$rspin[[i]]$rid  <- p$rtarget$rid
    p$rspin[[i]]$rmid <- p$rtarget$rmid
    p$rspin[[i]]$R    <- vector(mode="list")
    p$rspin[[i]]$IR   <- vector(mode="list")
    nrot <- length( p$rtarget$rid )


    for ( k in 1:length( p$rspin[[i]]$rid) ) {
      rid  <- p$rspin[[i]]$rid[k]
      rmid <- p$rspin[[i]]$rmid[k]
      
      p$rspin[[i]]$R[[k]] <- rot_rotation_matrix_spherical_basis ( interpolator_J2_list[i], cubic_group_double_cover_rotations[[rid]]$n, cubic_group_double_cover_rotations[[rid]]$w )

      p$rspin[[i]]$IR[[k]] <- rot_inversion_matrix_spherical_basis ( interpolator_J2_list[i], p$interpolator_bispinor_list[i] ) %*% rot_rotation_matrix_spherical_basis ( interpolator_J2_list[i], cubic_group_double_cover_rotations[[rmid]]$n, cubic_group_double_cover_rotations[[rmid]]$w )

      # ***********************************************************
      # * check, whether we have spin 1 and want to use cartesian
      # * basis instead of spherical
      # ***********************************************************
      if ( !is.null ( interpolator_cartesian_list ) ) {
        if ( interpolator_cartesian_list[i] && ( interpolator_J2_list[i] == 2 ) ) {
          p$rspin[[i]]$R[[k]]  <- rot_spherical2cartesian_3x3 ( p$rspin[[i]]$R[[k]]  )
          p$rspin[[i]]$IR[[k]] <- rot_spherical2cartesian_3x3 ( p$rspin[[i]]$IR[[k]] )
        }
      }  # end of check on Cartesian basis

      # ***********************************************************
      # ***********************************************************

      # ***********************************************************
      # * reference frame rotation
      # ***********************************************************
      if ( !is.null ( p$refframerot ) ) {

        p$rspin[[i]]$R[[k]] <- p$refframerot$srot[[i]] %*% p$rspin[[i]]$R[[k]] %*% Conj ( t ( p$refframerot$srot[[i]] ) )

        p$rspin[[i]]$IR[[k]] <- p$refframerot$srot[[i]] %*% p$rspin[[i]]$IR[[k]] %*% Conj ( t ( p$refframerot$srot[[i]] ) )

      }

    }  # * end of loop on p->rtarget->n rotations 

  }  # * end of loop on p->n interpolators

  # ***********************************************************
  # * allocate and set 3-momentum for each interpolator
  # ***********************************************************

  if ( !is.null ( interpolator_momentum_list ) ) {
    p$p <- interpolator_momentum_list

    if ( !is.null ( p$refframerot )) {
      for ( i in 1:length( p$p ) )  p$p[[i]] <- round ( p$refframerot$prot %*% p$p[[i]] )
    }
  }

  # ***********************************************************
  # ***********************************************************

  # ***********************************************************
  # * return projector
  # ***********************************************************

  return( p );
}  # end of projector_set


# ***********************************************************
# ***********************************************************
