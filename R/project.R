

project <- function ( R, T, alpha, beta ) {

  res <- array (0,  dim = dim(R[[1]]))

  for (i in 1:length(T) ) {
    # res <- res + T[[i]][alpha,beta] * R[[i]]
    res <- res + Conj( T[[i]][alpha,beta] ) * R[[i]]
  }
  return(res)
}

project_2c4v <- function( tR, tIR, sR, sIR, alpha, beta ) {

  rid  <- c(1,4,7,10,13,16,19,48)

  rmid <- c(2,3,5,6,38,39,44,45)

  resR <- array (0, dim=dim(sR[[1]] ))
  resIR <- array (0, dim=dim(sR[[1]] ))

  for ( i in 1:length(rid) ) {
    # resR <- resR + Conj( tR[[rid[i]]][alpha,beta] ) * sR[[rid[i]]]
    resR <- resR +  tR[[rid[i]]][alpha,beta]  * sR[[rid[i]]]
  }

  for ( i in 1:length(rmid) ) {
    # resIR <- resIR + Conj( tIR[[rmid[i]]][alpha,beta] ) * sIR[[rmid[i]]]
    resIR <- resIR +  tIR[[rmid[i]]][alpha,beta]  * sIR[[rmid[i]]]
  }

  return(list(R=resR, IR=resIR))
}

check_lg <- function( d, R) {

  for ( i in 1:length(R) ) {
    if ( sum( abs( R[[i]] %*% d - d )^2 ) < 1.e-14 ) cat ( "# d = ", d, " has lg element ", i, "\n" )
  }

  for ( i in 1:length(R) ) {
    if ( sum( abs( R[[i]] %*% d + d )^2 ) < 1.e-14 ) cat ( "# d = ", d, " has lg I-element ", i, "\n" )
  }

}

prod_project_cmf <- function( R1, R2, T, alpha, beta) {


  d1 <- dim(R1[[1]])[1]
  d2 <- dim(R2[[1]])[1]

  res <- array( 0, dim = c(d1,d2,d1,d2) )

  for ( i in 1:length(T) ) {

    A <-  array( 0, dim = c(d1,d2, d1,d2) )

    for ( j1 in 1:d1 ) {
    for ( j2 in 1:d1 ) {

      for ( k1 in 1:d2 ) {
      for ( k2 in 1:d2 ) {
        A[j1,k1,j2,k2] = R1[[i]][j1,j2] * R2[[i]][k1,k2]
      }}
    }}
    res <- res + Conj ( T[[i]][alpha,beta] ) * A
  }

  return(res)
}
