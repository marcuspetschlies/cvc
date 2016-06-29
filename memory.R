memory <- function(LL,p,n) {

  L <- LL / p
  nproc <- prod(p)

  VOLUME <- prod(L)

  RAND <- 2*( L[1]*L[2]*L[3] + L[2]*L[3]*L[4] + L[3]*L[4]*L[1] + L[4]*L[1]*L[2])

  EDGES <- 4 * ( L[3]*L[4] + L[2]*L[4] + L[1]*L[4] + L[2]*L[3] + L[1]*L[3] + L[1]*L[2] )

  cat("# nproc          = ", nproc, "\n")
  cat("# VOLUME         = ", VOLUME, "\n")
  cat("# VOLUME+RAND    = ", VOLUME+RAND, "\n")
  cat("# VOLUMEPLUSRAND = ", VOLUME+RAND+EDGES, "\n")

  m_g <- (VOLUME+RAND+EDGES) * 72 * 8 / 1024^3 * nproc

  m_eo <- (VOLUME+RAND) * 12 * 8 /1024^3 * nproc * (2*n+3)

  cat("# mem gauge = ", m_g, " GB\n")
  cat("# mem eo(", n, ")    = ", m_eo, " GB\n")

}
