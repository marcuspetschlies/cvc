perm4 <- function() {
  cat("# ", date(), "\n", sep="", file="perm_tab4.in", append=FALSE)
  i <- 0
  for ( i1 in 1:4 ) {
  for ( i2 in 1:4 ) {
  for ( i3 in 1:4 ) {
  for ( i4 in 1:4 ) {
    a <- array(0, dim=c(4,4))
    a[1,i1] <- 1
    a[2,i2] <- 1
    a[3,i3] <- 1
    a[4,i4] <- 1
    d <- det(a)
    if ( d == 0 ) next
    cat("perm_tab4[",4*i, "]=",i1-1,"\n", sep="", file="perm_tab4.in", append=TRUE)
    cat("perm_tab4[",4*i+1, "]=",i2-1,"\n", sep="", file="perm_tab4.in", append=TRUE)
    cat("perm_tab4[",4*i+2, "]=",i3-1,"\n", sep="", file="perm_tab4.in", append=TRUE)
    cat("perm_tab4[",4*i+3, "]=",i4-1,"\n", sep="", file="perm_tab4.in", append=TRUE)
    cat("perm_tab4_sign[",i,"]=",d,"\n", sep="", file="perm_tab4.in", append=TRUE)
    cat("#\n", file="perm_tab4.in", append=TRUE)
    i <- i+1
  }}}}
}
