# ******************************************************************
# * little groups with their irreps
# ******************************************************************
source("little_groups_2Oh.R")
source("R_gamma.R")
source("rotations.R")
source("set_cubic_group_double_cover_elements.R")
source("set_rot_mat_table_cubic_group_double_cover_gu.R")
source("projector_set.R")

# ******************************************************************
# * matrix norm, sqrt( trace ( A^+ A ) )
# ******************************************************************
mat_norm <- function(d) {
  return( abs( sqrt( sum( diag( Conj(t(d)) %*% d ) ) ) ) )
}

# ******************************************************************
# * norm of A - B
# ******************************************************************
mat_norm_diff <- function(A,B) {
  return( mat_norm( A - B ))
}

# ******************************************************************
# *
# ******************************************************************
test_projector <- function ( J = 0) {

  eps <- 1.e-12

  interpolator_num     <- 1
  interpolator_J2_list <- c( 2*J )
  
  for ( igr in 1:length(little_groups_2Oh ) )
  {
    gr <- little_groups_2Oh[[igr]]

    for ( ir in 1:length( gr$irreps ) )
    {
      irrep     <- gr$irreps[[ir]]
      irrep_dim <- gr$irreps_dim[ir]

      cat( "# [test_projector] little group ", gr$group, " irrep ", irrep, "\n" )
      pr        <- projector_set ( lg=gr, irrep = irrep, interpolator_num = interpolator_num, interpolator_J2_list = interpolator_J2_list )

      # * number of group elements in current irrep = 2 x nrot = # Rs + # IRs
      nrot      <- length ( pr$rtarget$rid )

      # * P = projector, holds subduction coefficients
      P         <- vector( mode = "list" )

      for ( ref_row in 1:irrep_dim ) {

        # ******************************************************************
        # * here we calculate the subduction coefficients for all rows
        # *   of the present irrep
        # ******************************************************************
        for ( row in 1:irrep_dim ) {

          P[[row]] <- array ( 0, dim=dim( pr$rspin[[1]]$R[[1]] ) )

          # * the group projection formula
          for ( irot in 1:nrot ) {

            P[[row]] <- P[[row]] + Conj(pr$rtarget$R[[irot]][row,ref_row]) * pr$rspin[[1]]$R[[irot]] + Conj(pr$rtarget$IR[[irot]][row,ref_row]) * pr$rspin[[1]]$IR[[irot]]

          }

          # * normalize P <- P * dim of irrep / total number of rotations (Rs and IRs in our nomenclature)
          P[[row]] <- P[[row]] * irrep_dim / ( 2 * nrot )

        }  # end or loop on row

        # ******************************************************************
        # * here we to the dipi-tipi test for Rs
        # ******************************************************************

        norm_diff <- array ( dim = c( nrot, irrep_dim ,2 ) )

        for ( irot in 1:nrot ) {

          A <- vector ( mode = "list" )

          for ( l in 1:irrep_dim ) {

            A[[l]] <- array ( 0, dim = dim( P[[1]] ) )

            for ( k in 1:irrep_dim ) {

              A[[l]] <- A[[l]] + pr$rtarget$R[[irot]][k,l] * P[[k]]

            }
          }

          B <- vector ( mode = "list" )

          for ( l in 1:irrep_dim ) {
            B[[l]] <- pr$rspin[[1]]$R[[irot]] %*% P[[l]]
          }

          for ( l in 1:irrep_dim ) {
            # cat ( " row ", l, " ref row ", ref_row, " rot ", irot, " norm diff ", mat_norm_diff (A[[l]],  B[[l]] ), " / ", mat_norm(A[[l]]),  "\n" )
            norm_diff[irot,l,] <- c( mat_norm_diff (A[[l]],  B[[l]] ), mat_norm(A[[l]])  )
          }

        }

        if ( any ( norm_diff[,,1] > eps ) )  {
          cat(" R-test failed for group ", gr$group, " irrep ", irrep, " ref row ", ref_row , "\n" )
          return( norm_diff )
        } else {
          cat(" R-test successfull for group ", gr$group, " irrep ", irrep, " ref row ", ref_row , "\n" )
        }


        # ******************************************************************
        # * and for IRs 
        # ******************************************************************

        norm_diff <- array ( dim = c( nrot, irrep_dim ,2 ) )

        for ( irot in 1:nrot ) {

          A <- vector ( mode = "list" )#

          for ( l in 1:irrep_dim ) {

            A[[l]] <- array ( 0, dim = dim( P[[1]] ) )

            for ( k in 1:irrep_dim ) {

              A[[l]] <- A[[l]] + pr$rtarget$IR[[irot]][k,l] * P[[k]]

            }
          }

          B <- vector ( mode = "list" )

          for ( l in 1:irrep_dim ) {
            B[[l]] <- pr$rspin[[1]]$IR[[irot]] %*% P[[l]]
          }

          for ( l in 1:irrep_dim ) {
            # cat ( " row ", l, " ref row ", ref_row, " rot ", irot, " norm diff ", mat_norm_diff (A[[l]],  B[[l]] ), " / ", mat_norm(A[[l]]),  "\n" )
            norm_diff[irot,l,] <- c( mat_norm_diff (A[[l]],  B[[l]] ), mat_norm(A[[l]])  )
          }

        }

        if ( any ( norm_diff[,,1] > eps ) )  {
          cat("IR-test failed for group ", gr$group, " irrep ", irrep, " ref row ", ref_row , "\n" )
          return( norm_diff )
        } else {
          cat("IR-test successfull for group ", gr$group, " irrep ", irrep, " ref row ", ref_row , "\n" )
        }

      }  # end of loop on ref_row

    }  # end of loop on irreps in lg
  }  # end of loop on lg

}  # end of test_projector
