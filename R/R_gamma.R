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
