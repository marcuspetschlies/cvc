# ******************************************************************
# * little groups with their irreps
# ******************************************************************
source("little_groups_2Oh.R")

# ******************************************************************
# *
# ******************************************************************
test_projector <- function () {

  
  for ( ig in 1:length( little_groups_2Oh ) )
  {

    for ( ir in 1:length( little_groups_2Oh[[ig]]$irreps )
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
