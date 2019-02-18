/******************************************
 * uwerr functions
 *
 * PURPOSE:
 * DONE:
 * TODO:
 * - error on Gamma_F(t) and tau_int(t)
 * CHANGES:
 ******************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "global.h"
#ifdef HAVE_LIBGSL
#  include "igsl.h"
#endif
#include "uwerr.h"
#include "incomp_gamma.h"

int uwerr_verbose;

/*******************************************************
 * uwerr_init
 *
 * - initialize uwerr struct
 *******************************************************/
int uwerr_init( uwerr * const u) {
  int i;
  u->value   = 0.;
  u->dvalue  = 0.;
  u->ddvalue = 0.;
  u->tauint  = 0.;
  u->dtauint = 0.;
  u->Wopt    = 0;
  u->tau = NULL;
  u->gamma = NULL;
  u->nalpha = 0;
  u->nreplica = 0;
  for(i=0; i<MAX_NO_REPLICA; i++) u->n_r[i] = 0;
  u->Wmax = 0;
  u->npara = 0;
  u->para = NULL;
  u->obsname[0] = '\0';
  u->write_flag = 0;
  u->func = NULL;
  u->dfunc = NULL;
  u->s_tau = 0.;
  u->ipo = 0;
  u->Qval = 0.;
  u->p_r  = NULL;
  u->bins = NULL;
  u->binbd = NULL;
  u->p_r_mean = 0.;
  u->p_r_var  = 0.;
  return(0);
}  /* end of uwerr_init  */

/*******************************************************
 * uwerr_free
 *
 * - free uwerr data fields
 *******************************************************/
int uwerr_free ( uwerr * const u ) {
  if(u->tau   !=NULL) { free(u->tau);   u->tau   = NULL; }
  if(u->gamma !=NULL) { free(u->gamma); u->gamma = NULL; }
  if(u->p_r   !=NULL) { free(u->p_r);   u->p_r   = NULL; }
  if(u->bins  !=NULL) { free(u->bins);  u->bins  = NULL; }
  if(u->binbd !=NULL) { free(u->binbd); u->binbd = NULL; }
  return(0);
}  /* end of uwerr_free */

/*******************************************************
 * uwerr_calloc
 *
 * - allocate mem for UWerr Gamma, tau_int, histogram
 *******************************************************/
int uwerr_calloc ( uwerr * const u ) {
  int k;
  if (u->Wmax == 0) {
    fprintf(stderr, "[uwerr_calloc] Error, Wmax=%lu; obsname=%s\n", u->Wmax, u->obsname);
    return(1);
  } else if(u->gamma != NULL) {
    fprintf(stderr, "[uwerr_calloc] Error, gamma not NULL; obsname=%s\n", u->obsname);
    return(1);
  } else {
    u->gamma  = (double*)calloc(u->Wmax, sizeof(double));
  }
  if(u->Wmax == 0) {
    fprintf(stderr, "[uwerr_calloc] Error, Wmax=0\n");
    return(2);
  } else if (u->tau != NULL) {
    fprintf(stderr, "[uwerr_calloc] Error, tau not NULL\n");
    return(2);
  } else {
    u->tau  = (double*)calloc(u->Wmax, sizeof(double));
  }
  if(u->nreplica>1) {
    if(u->p_r==NULL) {
      u->p_r = (double*)calloc(u->nreplica, sizeof(double));
    } else {
      fprintf(stderr, "[uwerr_calloc] Error, p_r not NULL\n");
      return(3);
    }
  } else {
    // TEST
    //fprintf(stdout, "[uwerr] Warning, nreplica<=1\n");
  }
  k = _num_bin(u->nreplica);
  if(k>3) {
    if(u->bins==NULL && u->binbd==NULL) {
      u->bins  = (double*)calloc(k, sizeof(double));
      u->binbd = (double*)calloc(k+1, sizeof(double));
    } else {
      fprintf(stderr, "[uwerr_calloc] Error, bins/binbd not NULL\n");
      return(4);
    }
  } else {
    // TEST
    //fprintf(stdout, "[uwerr] Warning, number of bins <= 3, no allocation\n");
  }
  return(0);
}  /* end of uwerr_calloc */

/*******************************************************
 * uwerr_printf
 *
 * - print uwerr data to files
 *******************************************************/
int uwerr_printf ( uwerr const u ) {

  size_t W, k, n;
  char format[400], filename[400];
  FILE *ofs=NULL;

  if(u.write_flag == 0) {
    fprintf(stderr, "[uwerr_printf] Error, write_flag is 0\n");
    return(1);
  }

  /* (1) value, dvalue, ... */  
  sprintf(filename, "%s_uwerr", u.obsname);
  if(u.write_flag==1) {
    ofs = fopen(filename, "w");
  } else {
    ofs = fopen(filename, "a");
  }
  if(ofs==NULL) {
    fprintf(stderr, "[uwerr_printf] Could not open file %s\n", filename);
    return(2);
  }
  strcpy(format, "%d%25.16e%25.16e%25.16e%25.16e%25.16e%25.16e%4d\n");
  fprintf(ofs, format, u.ipo, u.value, u.dvalue, u.ddvalue, u.tauint, u.dtauint, u.Qval, u.Wopt);
  if (fclose(ofs)!=0) {
    fprintf(stderr, "[uwerr_printf] Could not close file %s\n", filename);
    return(3);
  }

  /* (2) Gamma_F */
  sprintf(filename, "%s_uwerr_gamma", u.obsname);
  if(u.write_flag==1) {
    ofs = fopen(filename, "w");
  } else {
    ofs = fopen(filename, "a");
  }
  if(ofs==NULL) {
    fprintf(stderr, "[uwerr_printf] Could not open file %s\n", filename);
    return(4);
  }
  strcpy(format, "%lu%25.16e\n");
  fprintf(ofs, "# obsname = %s \t ipo = %lu\n", u.obsname, u.ipo);
  for(W=0; W <= u.Wopt; W++) {
    fprintf(ofs, format, W, (u.gamma)[W]);
  }
  if (fclose(ofs)!=0) {
    fprintf(stderr, "[uwerr_printf] Could not close file %s\n", filename);
    return(5);
  }

  /* (3) tau_int */
  sprintf(filename, "%s_uwerr_tauint", u.obsname);
  if(u.write_flag==1) {
    ofs = fopen(filename, "w");
  } else {
    ofs = fopen(filename, "a");
  }
  if(ofs==NULL) {
    fprintf(stderr, "[uwerr_printf] Could not open file %s\n", filename);
    return(6);
  }
  fprintf(ofs, "# obsname = %s \t ipo = %lu\n", u.obsname, u.ipo);
  for(W=0; W <= u.Wopt; W++) {
    fprintf(ofs, format, W, (u.tau)[W]);
  }
  if (fclose(ofs)!=0) {
    fprintf(stderr, "[uwerr_printf] Could not close file %s\n", filename);
    return(7);
  }

  /* (4) histogram */
  if(u.nreplica>1) {
    k = _num_bin(u.nreplica);
    sprintf(filename, "%s_uwerr_hist", u.obsname);
    if(u.write_flag==1) {
      ofs = fopen(filename, "w");
    } else {
      ofs = fopen(filename, "a");
    }
    if(ofs==NULL) {
      fprintf(stderr, "[uwerr_printf] Error, could not open file %s for writing\n", filename);
      return(8);
    }
    fprintf(ofs, "# p_r_mean = %25.16e\n# p_r_var  = %25.16e\n", u.p_r_mean, u.p_r_var);
    fprintf(ofs, "# n p_r\n");
    for(n=0; n<u.nreplica; n++) fprintf(ofs, "%4lu%25.16e\n", n, u.p_r[n]);
    if(k<=3) {
      fprintf(ofs, "# k = %lu is too small\n", k);
    } else {
      fprintf(ofs, "# number of entries: nreplica = %lu\n"\
                   "# number of classes: k = %lu\n", u.nreplica, k);
      strcpy(format, "%25.16e\t%25.16e%25.16e\n");
      for(n=0; n<k; n++) fprintf(ofs, format, u.binbd[n], u.binbd[n+1], u.bins[n]);
    }
    if( fclose(ofs) != 0 ) {
      fprintf(stderr, "[uwerr_printf] Error, could not close file %s\n", filename);
      return(9);
    }
  }

  return(0);
}  /* end of uwerr_printf */

/*******************************************************
 * uwerr_analysis
 *
 *******************************************************/
#define _UWERR_FREE_EXIT(_exit_status) { \
  if(F_b     != NULL) free(F_b);         \
  if(a_bb    != NULL) free(a_bb);        \
  if(a_b     != NULL) {                  \
    if(a_b[0]!= NULL) free(a_b[0]);      \
    free(a_b);                           \
  }                                      \
  if(a_proj  != NULL) free(a_proj);      \
  if(f_alpha != NULL) free(f_alpha);     \
  if(h_alpha != NULL) free(h_alpha);     \
  if(m_alpha != NULL) free(m_alpha);     \
  return(_exit_status);                  \
}

int uwerr_analysis(double * const data, uwerr * const u) {

  size_t nalpha   = u->nalpha;
  size_t nreplica = u->nreplica;
  size_t ipo      = u->ipo;
  double s_tau    = u->s_tau;

  size_t i, n;
  size_t ndata, Wmax, W, k;
  double **a_b=NULL, *a_bb=NULL, *a_proj=NULL, a_bb_proj;
  double *F_b=NULL, F_bb, F_bw;
  double *Gamma_F=NULL, C_F, C_Fopt, v_Fbb, tau, *tau_int=NULL;
  double *f_alpha=NULL, *h_alpha=NULL, *m_alpha=NULL, *data_ptr=NULL, func_res;
  double chisqr, delta, lobd;
  int status;

  // TEST
/*
  printf("[uwerr] The following arguments have been read:\n");
  printf("[uwerr] nalpha   = %lu\n", nalpha);
  printf("[uwerr] nreplica = %lu\n", nreplica);
  for(i=0; i<nreplica; i++) {
    printf("[uwerr] n_r(%lu)  = %lu\n", i, u->n_r[i]);
  }
  printf("[uwerr] npara    = %lu\n", u->npara);
  printf("[uwerr] ipo      = %lu\n", ipo);
  printf("[uwerr] s_tau    = %e\n", s_tau);
  printf("[uwerr] obsname  = %s\n", u->obsname);
  printf("[uwerr] write_flag = %d\n", u->write_flag);
*/
  /*************************************************************
   * check if combination of values in ipo an func are allowed *
   *************************************************************/
  if ( ipo==0 && u->func==NULL) {
    fprintf(stderr, "[uwerr] illegal values of func and ipo, return\n");
    return(1);
  }

  /* ndata - total number of rows in data */
  for( i=1, ndata = u->n_r[0]; i<nreplica; ndata += u->n_r[i++] );
  /* Wmax - longest possible summation index + 1 */
  MIN_UINT(u->n_r, nreplica, &Wmax);

  // TEST
  //fprintf(stdout, "[uwerr] ndata=%lu; Wmax=%lu, obsname=%s\n", ndata, Wmax, u->obsname);

  u->Wmax = Wmax;

  // TEST
  //fprintf(stdout, "[uwerr] Wmax=%lu, obsname=%s\n", u->Wmax, u->obsname);
  
  if( (status=uwerr_calloc(u)) != 0 ) {
    fprintf(stderr, "[uwerr] Error, allocation of u returned status %d\n", status);
    return(218);
  }

  /*******************
   * allocate memory *
   *******************/
  if((F_b=(double*)calloc(nreplica, sizeof(double)))==NULL) return(200);
  if (u->func!=NULL) {  // only necessary in case of derived quantity
    if((a_bb=(double*)calloc(nalpha,   sizeof(double)))==NULL) return(201);
    if((a_b=(double**)calloc(nreplica, sizeof(double*)))==NULL) return(202);
    if((a_b[0]=(double*)calloc(nreplica*nalpha, sizeof(double)))==NULL) return(203);
    for(n=1; n<nreplica; n++) { a_b[n] = a_b[n-1] + nalpha; }
  }

  /*********************************************************************
   * calculate estimators for primary observable/derived quantity      *
   *********************************************************************/
  if(u->func==NULL) {
    data_ptr = data+ipo-1; 
    for(n=0; n<nreplica; n++) {
      ARITHMEAN(data_ptr, nalpha, u->n_r[n], F_b+n);  /* arithmetic mean for replica */
      data_ptr += u->n_r[n] * nalpha;  /* pointer set to beginning of next replica */

      // TEST
      //fprintf(stdout, "[uwerr] F_b[%lu] = %25.16e\n", n, *(F_b+n));
    }
    ARITHMEAN(data+ipo-1, nalpha, ndata, &F_bb);  /* mean including all data for ipo */

    // TEST
    //fprintf(stdout, "[uwerr] F_bb = %25.16e\n", F_bb);
  }
  else if (u->func!=NULL) {          /* estimators for derived quantity */
    /* calculate means per replica and total mean */
    for(i=0; i<nalpha; i++) {
      data_ptr = data + i;
      for(n=0; n<nreplica; n++) {
	ARITHMEAN(data_ptr, nalpha, u->n_r[n], a_b[n]+i);
	data_ptr += u->n_r[n] * nalpha;
      }
      ARITHMEAN(data+i, nalpha, ndata, a_bb+i);
    }
    /* calculate estimators per replica for derived quatity */
    for(n=0; n<nreplica; n++) {
      u->func((void*)a_b[n], u->para, F_b+n);         /* est. for means per replicum */
    }
    u->func((void*)a_bb, u->para, &F_bb);                /* est. for total mean */
  }

  /* weighed mean of F_b's with weights n_r if nreplica > 1 */
  if(nreplica > 1) {
    WEIGHEDMEAN(F_b, 1, nreplica, &F_bw, u->n_r);

    // TEST 
    //fprintf(stdout, "[uwerr] F_bw = %25.16e\n", F_bw);
  }

  // TEST
  if (u->func!=NULL) {
    //for(i=0; i<nalpha; i++) fprintf(stdout, "# a_bb[%d] = %25.16e\n", i, a_bb[i]); 
    //for(i=0; i<nreplica; i++) fprintf(stdout, "# F_b[%d] = %25.16e\n", i, F_b[i]);
  }

  /***********************************************
   * calculate projection of data and mean value *
   ***********************************************/
  if( (a_proj  = (double *)calloc(ndata,  sizeof(double)))==NULL) return(211);
  if(u->func==NULL) { 
    /* data is projectet to itself in case of prim. obs. */
    COPY_STRIDED(a_proj, data+ipo-1, nalpha, ndata);
    a_bb_proj = F_bb;          /* projected mean is total mean */
  }
  else if (u->func!=NULL) {
    if((f_alpha = (double *)calloc(nalpha, sizeof(double)))==NULL) return(212);
    if((h_alpha = (double *)calloc(nalpha, sizeof(double)))==NULL) return(213);
    if((m_alpha = (double *)calloc(nalpha, sizeof(double)))==NULL) return(214);
    
    /* calculate derivatives of func with respect to A_alpha */
    if(u->dfunc != NULL ) {
      u->dfunc((void*)a_bb, u->para, f_alpha);
    } else {
      for(i=0; i<nalpha; i++) {
        SET_TO(h_alpha, nalpha, 0.0);
        STDDEV(data+i, nalpha, ndata, h_alpha+i);
        h_alpha[i] *= sqrt((double)(ndata-1)/(double)ndata);

        // TEST
        //fprintf(stdout, "[uwerr] halpha[%lu] = %25.16e\n", i, h_alpha[i]);
        if(*(h_alpha+i)==0.0) {
          fprintf(stderr, "[uwerr] Warning: no fluctuation in primary observable %lu\n", i);
          *(f_alpha + i) = 0.0;
        } else {
          ADD_ASSIGN(m_alpha, a_bb, h_alpha, nalpha);
          u->func((void*)m_alpha, u->para, &func_res);
          *(f_alpha+i) = func_res;
          SUB_ASSIGN(m_alpha, a_bb, h_alpha, nalpha);
          u->func((void*)m_alpha, u->para, &func_res);
          *(f_alpha+i) -= func_res;
          *(f_alpha+i) = *(f_alpha+i) / (2.0 * *(h_alpha+i));
        }
      }
    }
    SET_TO(a_proj, ndata, 0.0);
    a_bb_proj = 0.0;
    for(n=0; n<ndata; n++) { 
      for(i=0; i<nalpha; i++) {
	a_proj[n] = a_proj[n] + data[i+n*nalpha] * f_alpha[i];
      }
    }
    for(i=0; i<nalpha; i++) {
      a_bb_proj = a_bb_proj + a_bb[i] * f_alpha[i];
    }

    // TEST
    //for(i=0; i<nalpha; i++) fprintf(stdout, "# [uwerr] falpha[%d] = %25.16e\n", i, f_alpha[i]);
    
    free(m_alpha); m_alpha = NULL;
    free(f_alpha); f_alpha = NULL;
    free(h_alpha); h_alpha = NULL;
    free(a_b[0]);  a_b[0]  = NULL;
    free(a_b);     a_b     = NULL;
    free(a_bb);    a_bb    = NULL;

    // TEST
    //for(n=0; n<ndata; n++) fprintf(stdout, "[uwerr] delpro[%lu] = %25.16e\n", n, a_proj[n]-a_bb_proj); 
    //for(n=0; n<ndata; n++) fprintf(stdout, "[uwerr] a_proj[%lu] = %25.16e\n", n, a_proj[n]); 
    //fprintf(stdout, "[uwerr] a_bb_proj = %25.16e\n", a_bb_proj);
  }

  /**********************************************************************
   * calculate error, error of the error; automatic windowing condition *
   **********************************************************************/

  /* (1) Gamma_F(t), t=0,...,Wmax */
  Gamma_F = u->gamma;
  tau_int = u->tau;
  SET_TO(Gamma_F, Wmax, 0.0);
  SET_TO(tau_int, Wmax, 0.0);
  VARIANCE_FIXED_MEAN(a_proj, a_bb_proj, 1, ndata, &v_Fbb);
  C_F        = v_Fbb;
  Gamma_F[0] = v_Fbb;

  // TEST
  //fprintf(stdout, "[uwerr] Gamma_F[0] = %25.16e\n", Gamma_F[0]);
  if (Gamma_F[0]==0.0) {
    fprintf(stderr, "[uwerr] ERROR, no fluctuations; return\n");
    uwerr_free(u);
    _UWERR_FREE_EXIT(-5)
  }
  tau_int[0] = 0.5; 
  for(W=1; W<Wmax-1; W++) {
    /* calculate Gamma_F(W) */
    data_ptr = a_proj;
    for(n=0; n<nreplica; n++) {
      for(i=0; i<(u->n_r[n]-W); i++) {
        Gamma_F[W] += (data_ptr[i] - a_bb_proj) * (data_ptr[i+W] - a_bb_proj);
      }
      data_ptr = data_ptr + u->n_r[n];
    }
    Gamma_F[W] /= (double)(ndata-nreplica*W);
    C_F = C_F + 2.0 * Gamma_F[W];
    tau_int[0] = C_F / (2.0*v_Fbb);
    if(tau_int[0] < 0.5) {
      if(uwerr_verbose) fprintf(stderr, "[uwerr] Warning: tau_int < 0.5; tau set to %e\n", TINY);
      tau = TINY;
    } else {
      tau = s_tau / log( ( tau_int[0] + 0.5 ) / ( tau_int[0] - 0.5 ) );
    }
    if( exp(-(double)W / tau) - tau / sqrt((double)(W*ndata)) < 0.0 ) {
      u->Wopt = W;

      // TEST
      // fprintf(stdout, "[uwerr] Wopt = %lu\n", u->Wopt);

      break;
    }

    // TEST
    // fprintf(stdout, "[uwerr] W=%lu\tGamma_F=%25.16e\ttau=%25.16e\n", W, Gamma_F[W], tau);
  }

  if(W==Wmax-1) {
    fprintf(stderr, "[uwerr] windowing condition failed after W = %lu\n", W);
    _UWERR_FREE_EXIT(-6);
  }
  else {
    SUM(Gamma_F+1, 1, u->Wopt, &C_Fopt);
    C_Fopt = 2.0 * C_Fopt + Gamma_F[0];

    // TEST
    //fprintf(stdout, "[uwerr] before: C_Fopt = %25.16e\n", C_Fopt);

    for(W=0; W <= u->Wopt; W++) {
      Gamma_F[W] = Gamma_F[W] + C_Fopt/((double)ndata);
    }
    SUM(Gamma_F+1, 1, u->Wopt, &C_Fopt);
    C_Fopt = 2.0 * C_Fopt + Gamma_F[0];

    // TEST
    //fprintf(stdout, "[uwerr] after:  C_Fopt = %25.16e\n", C_Fopt);
    v_Fbb = Gamma_F[0];
    tau_int[0] = 0.5*v_Fbb;
    for(W=1; W <= u->Wopt; W++) tau_int[W] = tau_int[W-1] + Gamma_F[W];
    for(W=0; W <= u->Wopt; W++) tau_int[W] /= v_Fbb;
  }

  /***********************************
   * bias cancellation of mean value *
   ***********************************/
  // TEST
  //fprintf(stdout, "[uwerr] before:  F_bb = %25.16e\n", F_bb);

  if(nreplica > 1 ) {
    F_bb = ( (double)nreplica * F_bb - F_bw ) / ((double)(nreplica-1));
  }

  // TEST
  //fprintf(stdout, "[uwerr] after:   F_bb = %25.16e\n", F_bb);

  /**************************
   * calculation of results *
   **************************/
  u->value    = F_bb;
  u->dvalue   = sqrt(C_Fopt/((double)ndata));
  u->ddvalue  = u->dvalue * sqrt(((double)(u->Wopt) + 0.5) / (double)ndata);
  u->tauint  = C_Fopt / (2.0 * v_Fbb);
  u->dtauint = sqrt( 4. * ((double)(u->Wopt) + 0.5 - u->tauint) / (double)ndata ) * u->tauint;

  /*******************************************
   * consistency checks in case nreplica > 0 *
   *******************************************/
  if(nreplica>1) {
  /* (1) calculate Q-value <---> determine goodness of the fit F_b(n) = F_bw = const. */
    chisqr = 0.0;
    for(n=0; n<nreplica; n++) {
      chisqr = chisqr + _SQR( F_b[n] - F_bw ) / ( C_Fopt / (double)(u->n_r[n]) );
    }

    // TEST
    //fprintf(stdout, "[uwerr] chisqr = %18.16e\n", chisqr);
    //fprintf(stdout, "[uwerr] n      = %lu     \n", (nreplica-1)/2);

    u->Qval = 1.0 - incomp_gamma(chisqr/2.0, ((double)nreplica-1.)/2.);
  
  /* (2) inspection of p_r's defined below in a histogramm */
    for(n=0; n<nreplica; n++) {
      u->p_r[n] = (F_b[n] - F_bw) / \
	(u->dvalue * sqrt( ( (double)ndata/(double)(u->n_r[n])) - 1.0 ) );
    }
    ARITHMEAN(u->p_r, 1, nreplica, &(u->p_r_mean));
    VARIANCE(u->p_r, 1, nreplica, &(u->p_r_var));
    k = _num_bin(nreplica);
    if(k<=3) /* not enough classes for a meaningful histogramm */ {
      fprintf(stderr, "[uwerr] k = %lu is too small\n", k);
    } else {
      ABS_MAX_DBL(u->p_r, nreplica, &lobd); /* max{|p_r's|} */
      lobd = lobd *(1.0+TINY);
      delta = 2.0*lobd/(double)k;        /* expected distribution around mean=0 */
      lobd = -lobd;                      /* lower boundary of abscissa */
      SET_TO(u->bins,  k,   0.0);
      SET_TO(u->binbd, k+1, 0.0);
      for(n=0; n<nreplica; n++) {
        i = (int)((u->p_r[n] - lobd)/delta);
        u->bins[i] += 1.0;
      }
      for(i=0; i<k+1; i++) u->binbd[i] = lobd + (double)i * delta;
    }
  }

  /**************************
   * output                 *
   **************************/
  if(u->write_flag != 0) {
    if( (status=uwerr_printf(*u))!=0 ) {
      fprintf(stderr, "[uwerr] Error, uwerr_print returned %d\n", status);
      _UWERR_FREE_EXIT(217);
    }
  }

  /*****************************
   * free allocated disk space *
   *****************************/
  free(F_b);
  free(a_proj);

  return(0);

}  /* end of uwerr_analysis */
