gamma_basis <- vector(mode="list")
gamma_basis[["tmlqcd"]] <-vector(mode="list")


gamma_basis[["tmlqcd"]][["t"]] <- array(0, dim=c(4,4))
gamma_basis[["tmlqcd"]][["t"]][1,3] = -1
gamma_basis[["tmlqcd"]][["t"]][2,4] = -1
gamma_basis[["tmlqcd"]][["t"]][3,1] = -1
gamma_basis[["tmlqcd"]][["t"]][4,2] = -1

gamma_basis[["tmlqcd"]][["x"]] <- array(0, dim=c(4,4))
gamma_basis[["tmlqcd"]][["x"]][1,4] = -1.i
gamma_basis[["tmlqcd"]][["x"]][2,3] = -1.i
gamma_basis[["tmlqcd"]][["x"]][3,2] = +1.i
gamma_basis[["tmlqcd"]][["x"]][4,1] = +1.i

gamma_basis[["tmlqcd"]][["y"]] <- array(0, dim=c(4,4))
gamma_basis[["tmlqcd"]][["y"]][1,4] = -1
gamma_basis[["tmlqcd"]][["y"]][2,3] = +1
gamma_basis[["tmlqcd"]][["y"]][3,2] = +1
gamma_basis[["tmlqcd"]][["y"]][4,1] = -1

gamma_basis[["tmlqcd"]][["z"]] <- array(0, dim=c(4,4))
gamma_basis[["tmlqcd"]][["z"]][1,3] = -1.i
gamma_basis[["tmlqcd"]][["z"]][2,4] = +1.i
gamma_basis[["tmlqcd"]][["z"]][3,1] = +1.i
gamma_basis[["tmlqcd"]][["z"]][4,2] = -1.i

gamma_basis[["tmlqcd"]][["id"]] <- array(0, dim=c(4,4))
gamma_basis[["tmlqcd"]][["id"]][1,1] = 1
gamma_basis[["tmlqcd"]][["id"]][2,2] = 1
gamma_basis[["tmlqcd"]][["id"]][3,3] = 1
gamma_basis[["tmlqcd"]][["id"]][4,4] = 1

gamma_basis[["tmlqcd"]][["5"]] <- array(0, dim=c(4,4))
gamma_basis[["tmlqcd"]][["5"]][1,1] = +1
gamma_basis[["tmlqcd"]][["5"]][2,2] = +1
gamma_basis[["tmlqcd"]][["5"]][3,3] = -1
gamma_basis[["tmlqcd"]][["5"]][4,4] = -1

set_gamma_basis_all <- function ( g ) {

  r <- list()

  r[["0"]] <- g[["t"]]
  r[["1"]] <- g[["x"]]
  r[["2"]] <- g[["y"]]
  r[["3"]] <- g[["z"]]

  r[["4"]] <- g[["id"]]

  r[["5"]] <- g[["5"]]

  r[["6"]] <- g[["t"]] %*% g[["5"]]
  r[["7"]] <- g[["x"]] %*% g[["5"]]
  r[["8"]] <- g[["y"]] %*% g[["5"]]
  r[["9"]] <- g[["z"]] %*% g[["5"]]

  r[["10"]] <- g[["t"]] %*% g[["x"]]
  r[["11"]] <- g[["t"]] %*% g[["y"]]
  r[["12"]] <- g[["t"]] %*% g[["z"]]

  r[["13"]] <- g[["x"]] %*% g[["y"]]
  r[["14"]] <- g[["x"]] %*% g[["z"]]
  r[["15"]] <- g[["y"]] %*% g[["z"]]

  return( invisible(r) )
}  # end of set_gamma_basis_all

set_Cgamma_basis_matching <- function(g) {

  eps <- 1.e-12

  b <- set_gamma_basis_all ( g )

  C <- g[["t"]] %*% g[["y"]]

  Cg <- list()
  Cg[["C"]] <- C
  Cg[["Cg5"]] <- C %*% g[["5"]]
  Cg[["Cg0"]] <- C %*% g[["t"]]
  Cg[["Cg5g0"]] <- C %*% g[["5"]] %*% g[["t"]]

  Cg[["Cgx"]] <- C %*% g[["x"]]
  Cg[["Cgy"]] <- C %*% g[["y"]]
  Cg[["Cgz"]] <- C %*% g[["z"]]

  Cg[["Cgxg5"]] <- C %*% g[["x"]] %*% g[["5"]]
  Cg[["Cgyg5"]] <- C %*% g[["y"]] %*% g[["5"]]
  Cg[["Cgzg5"]] <- C %*% g[["z"]] %*% g[["5"]]

  Cg[["Cgxg0"]] <- C %*% g[["x"]] %*% g[["t"]]
  Cg[["Cgyg0"]] <- C %*% g[["y"]] %*% g[["t"]]
  Cg[["Cgzg0"]] <- C %*% g[["z"]] %*% g[["t"]]

  Cg[["Cgxg5g0"]] <- C %*% g[["x"]] %*% g[["5"]] %*% g[["t"]]
  Cg[["Cgyg5g0"]] <- C %*% g[["y"]] %*% g[["5"]] %*% g[["t"]]
  Cg[["Cgzg5g0"]] <- C %*% g[["z"]] %*% g[["5"]] %*% g[["t"]]

  ##  return(Cg)
  z <- array ( dim=c(16, 16) )

  # cat ( "  double _Complex const gamma_basis_matching_coeff[16] = {\n" )
  cat ( "  double const Cgamma_basis_matching_coeff[16] = {\n" )

  for ( i in 1:16 ) {
    
    for ( k in 1:16 ) {
      z[i,k] <- sum( diag ( t(Conj(Cg[[k]])) %*% b[[i]] ) ) / 4
    }

    idx <- which( abs(z[i,]) > eps )
    if ( length(idx) > 1 || length(idx) == 0 ) stop("no / too many matches for i = ", i )

    cat ( formatC( Re(z[i,idx]), width=8, digits=2, format="f" )
          # " + ",
          # formatC( Im(z[i,idx]), width=6, digits=2, format="f" ), "*I" 
        )

    if ( i < 16 ) {
      cat ( ",  /* " )
    } else {
      cat ( "   /* " )
    }
    cat( formatC( i-1, width=2, digits=2, format="d" ), " =  ",
         formatC( names(Cg[idx]), width=-10, format="s" ), " */\n", sep="" )
  }

  cat( "  };\n" )

  return ( invisible ( z ) )
}  # end of set_Cgamma_basis_matching 
