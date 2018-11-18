require("hadron")

no_testsites <- 1000
no_testvals <- 50000

rand_file <- file("random.dat", "rb")
ord_file <- file("ordered.dat", "rb")

rand_dat <- readBin(con = rand_file,
                    what = "double",
                    n = no_testsites*no_testvals)

ord_dat <- readBin(con = ord_file,
                   what = "double",
                   n = no_testsites*no_testvals)
                     
close(rand_file)
close(ord_file)

ord_mtx <- matrix(ord_dat,
                  nrow=no_testvals,
                  ncol=no_testsites,
                  byrow=TRUE)

rand_mtx <- matrix(rand_dat,
                   nrow=no_testvals,
                   ncol=no_testsites,
                   byrow=TRUE)

rand_cor <- cor(rand_mtx)
ord_cor <- cor(ord_mtx)

print(max(rand_cor[upper.tri(rand_cor, diag=FALSE)]))
print(max(ord_cor[upper.tri(ord_cor, diag=FALSE)]))

rand_uwerr <- apply(X=rand_mtx,
                    FUN=function(x) { uwerrprimary(data=x)$tauint },
                    MARGIN=2)

ord_uwerr <- apply(X=ord_mtx,
                   FUN=function(x) { uwerrprimary(data=x)$tauint },
                   MARGIN=2)

print(c(min(rand_uwerr), max(rand_uwerr), var(rand_uwerr)))
print(c(min(ord_uwerr), max(ord_uwerr), var(ord_uwerr)))

