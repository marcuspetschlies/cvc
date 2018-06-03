
# ******************************************************************
# * little groups with their irreps
# ******************************************************************
group_irrep_list <- vector ( mode = "list" )

group_irrep_list[["2Oh"]]  <- c( "A1g", "A1u", "A2g", "A2u", "Eg", "Eu", "T1g", "T1u", "T2g", "T2u", "G1g", "G1u", "G2g", "G2u", "Hg", "Hu" )

group_irrep_list[["2C4v"]] <- c( "A1", "A2", "B1", "B2", "E", "G1", "G2" )

group_irrep_list[["2C2v"]] <- c( "A1", "A2", "B1", "B2", "G1" )

group_irrep_list[["2C3v"]] <- c( "A1", "A2", "K1", "K2", "E", "G1" )


# ******************************************************************
# * matrix A unitary ?
# ******************************************************************
is.unitary <- function( A, eps=1.e-12 ) {

  r <- max( abs( Conj(t(A)) %*% A  - diag( rep(1, times=dim(A)[1] ) )  ))

  return ( r < eps )
}

# ******************************************************************
# * matrix A unimodular in absolute value ?
# ******************************************************************
is.unimodularabs <- function(A, eps=1.e-12 ) {
 
  d <- abs( abs( prod(eigen(A, only.values=TRUE)$values) ) - 1 )

  return( d < eps )
}

# ******************************************************************
# * matrix A special-unitary ?
# ******************************************************************
is.special_unitary <- function(A, eps=1.e-12 ) {
 
  d <- abs( prod(eigen(A, only.values=TRUE)$values) - 1 )

  return( is.unitary (A) && (d < eps ) )
}

# ******************************************************************
# *
# ******************************************************************
test_irrep <- function ( test_unitary = TRUE, test_orthogonal = TRUE) {

  eps <- 1.e-12

  res <- vector( mode = "list" )

  for ( group in c( "2Oh", "2C4v", "2C2v", "2C3v" ) ) 
  {

    t <- vector ( mode = "list" )

    for ( irrep in group_irrep_list[[group]] ) {

      t[[irrep]] <- set_rot_mat_table_cubic_group_double_cover( group = group, irrep = irrep )

      # * test unitarity of representation
      if ( test_unitary ) {
        for ( i in 1 : length(t[[irrep]]$R)  ) cat( "# [test_irrep]  group , ", group, " irrep ", irrep, "  R[", i, "] is unitary ", is.unitary( t[[irrep]]$R[[i]] ),  "\n" )
        for ( i in 1 : length(t[[irrep]]$IR) ) cat( "# [test_irrep]  group , ", group, " irrep ", irrep, " IR[", i, "] is unitary ", is.unitary( t[[irrep]]$R[[i]] ),  "\n" )
      }
    }

    # * test orthogonality or representations within little group
    if ( test_orthogonal ) {
      filename <- paste( "test_irrep.", group, ".orth", sep="" )
      cat( "# [test_irrep] ", date(), "\n", file=filename, append=FALSE )
      for ( irrep1 in group_irrep_list[[group]] ) {
      for ( irrep2 in group_irrep_list[[group]] ) {
 
        n <- length(t[[irrep1]]$R)
        d1 <- dim( t[[irrep1]]$R[[1]] )[1]
        d2 <- dim( t[[irrep2]]$R[[1]] )[1]

        A <- array( 0, dim = c( dim(t[[irrep1]]$R[[1]]) , dim(t[[irrep2]]$R[[1]]) ))

        for ( i in 1 : n ) {
          A <- A + outer ( X=t[[irrep1]]$R[[i]],  Y=Conj( t[[irrep2]]$R[[i]] ),  FUN="*" )
          A <- A + outer ( X=t[[irrep1]]$IR[[i]], Y=Conj( t[[irrep2]]$IR[[i]] ), FUN="*" )
        }

        # normalize times irrep-dimension / number of group elements
        A <- A * d1 / (2*n)

        for ( i1 in 1:d1 ) {
        for ( i2 in 1:d1 ) {
          for ( k1 in 1:d2 ) {
          for ( k2 in 1:d2 ) {
            w1 <- abs(A[i1,i2,k1,k2])>eps
            w2 <- ( i1==k1 && i2==k2 && irrep1 == irrep2)
            cat( "group ", group, " irrep ", irrep1, " - ", irrep2, "   ", i1, "  ", i2, "  ", k1, "  ", k2, "  ", w1, "  ", 
                w2, "  ", w1 == w2, "\n", file=filename, append=TRUE )
          }}
        }}
      }}
    }  # end of if test orth

    res[[group]] <- t
  }
  return(res)
}
