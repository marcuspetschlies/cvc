/***********************************
 * uwerr.c
 *
 * PURPOSE:
 * - code to perform Gamma method by Ulli Wolff
 * DONE:
 * - Gamma_F positively checked
 * - tau positively checked
 * TODO:
 * - Wopt ??
 * - F_b in case of ipo>0 && func=NULL
 * - value, dvalue, ddvalue, tau_intbb, dtau_intbb in case ipo>0 && func==NULL
 * - h_alpha ??
 * - f_alpha ??
 * CHANGES:
 ***********************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "stats.h"
#include "dquant.h"
#include "uwerr.h"
#include "incomp_gamma.h"

int uwerr (char* append) {

  const double epsilon = 2.0e-16;

  int i, n, label;
  int ndata, Wmax, W, Wopt, k;
  double **a_b, *a_bb, *a_proj, a_bb_proj;
  double *F_b, *F_bb, *F_bw;
  double *Gamma_F, C_F, C_Fopt, v_Fbb, dv_Fbb, tau, *tau_int;
  double *f_alpha, *h_alpha, *m_alpha, *data_ptr, func_res;
  double value, dvalue, ddvalue, tau_intbb, dtau_intbb;
  double chisqr, Qval, *p_r, p_r_mean, p_r_var, delta, lobd, *bins;
  char filename[80], format[80];
  FILE *ofs;

  printf("[uwerr] The following arguments have been read:\n");
  printf("[uwerr] nalpha   = %d\n", nalpha);
  printf("[uwerr] nreplica = %d\n", nreplica);
  for(i=0; i<nreplica; i++) {
    printf("[uwerr] n_r(%2d)  = %d\n", i, n_r[i]);
  }
  printf("[uwerr] npara    = %d\n", npara);
  for(i=0; i<npara; i++) {
    printf("[uwerr] para(%2d) = %e\n", i, para[i]);
  }
  printf("[uwerr] ipo      = %d\n", ipo);
  printf("[uwerr] s_tau    = %e\n", s_tau);
  printf("[uwerr] obsname  = %s\n", obsname);
  printf("[uwerr] append   = %s\n", append);

  fprintf(stdout, "[uwerr]: Starting ...\n");

  /*************************************************************
   * check if combination of values in ipo an func are allowed *
   *************************************************************/
  label = ipo;
  if(ipo>0 && func!=NULL) {
    ipo = 0;
  }
  else if ( ipo==0 && func==NULL) {
    fprintf(stdout, "[uwerr] illegal values of func and ipo, return");
    return(1);
  }

  fprintf(stdout, "[uwerr]: checked ipo and func\n");

  /* ndata - total number of rows in data */
  for( i=1, ndata = *n_r; i<nreplica; ndata += *(n_r + i++) );
  /* Wmax - longest possible summation index + 1 */
  MIN_INT(n_r, nreplica, &Wmax);

  fprintf(stdout, "[uwerr]: have ndata and Wmax ready\n");

  /*******************
   * allocate memory *
   *******************/
  F_b     = (double *)calloc(nreplica, sizeof(double));
  F_bb    = (double *)calloc(1, sizeof(double));
  F_bw    = (double *)calloc(1, sizeof(double));
  Gamma_F = (double *)calloc(Wmax, sizeof(double));  
  tau_int = (double *)calloc(Wmax, sizeof(double));
  if (ipo==0 && func!=NULL) /* only necessary in case of derived quantity */ {
    a_b  = (double**)calloc(nreplica, sizeof(double*));
    a_bb = (double *)calloc(nalpha,   sizeof(double));
    for(n=0; n<nreplica; n++) { *(a_b+n)=(double*)calloc(nalpha, sizeof(double)); }
  }

  fprintf(stdout, "[uwerr]: allocated memory\n");

  /*********************************************************************
   * calculate estimators for primary observable/derived quantity      *
   *********************************************************************/
  if(ipo>0 && func==NULL)     /* here estimators for one of the prim. observables */ {
    data_ptr = *(data+ipo-1); /* points to column of ipo in data */
    for(n=0; n<nreplica; n++) {
      ARITHMEAN(data_ptr, *(n_r+n), F_b+n); /* arithmetic mean for replia */
      data_ptr = data_ptr + *(n_r+n);       /* pointer set to beginning of next replia */
      /* test */
      fprintf(stdout, "[uwerr] F_b(%d) = %18.16e\n", n, *(F_b+n));
    }
    ARITHMEAN(*(data+ipo-1), ndata, F_bb);  /* mean including all data for ipo */
    /* test */
    fprintf(stdout, "[uwerr] F_bn = %18.16e\n", *F_bb);
  }
  else if (ipo==0 && func!=NULL) {          /* estimators for derived quantity */
    /* calculate means per replica and total mean */
    for(i=0; i<nalpha; i++) {
      data_ptr = *(data+i);
      for(n=0; n<nreplica; n++) {
	ARITHMEAN(data_ptr, *(n_r+n), *(a_b+n)+i);
	data_ptr += *(n_r+n);
      }
      ARITHMEAN(*(data+i), ndata, a_bb+i);
    }
    /* calculate estimators per replica for derived quatity */
    for(n=0; n<nreplica; n++) {
      func(nalpha, *(a_b+n), npara, para, F_b+n);         /* est. for means per replicum */
    }
    func(nalpha, a_bb, npara, para, F_bb);                /* est. for total mean */
  }
  /* in case of more than one replica 
     calculate weighed mean of F_b's with weights n_r */
  if(nreplica > 1) {
    WEIGHEDMEAN(F_b, nreplica, F_bw, n_r);
    /* test */
    fprintf(stdout, "[uwerr] F_bw = %18.16e\n", *F_bw);
  }

  fprintf(stdout, "[uwerr]: have estimators ready\n");

  /***********************************************
   * calculate projection of data and mean value *
   ***********************************************/
  if(ipo>0 && func==NULL) {
    a_proj = *(data + ipo - 1); /* data is projectet to itself in case of prim.
				   observable */
    a_bb_proj = *F_bb;          /* projected mean is total mean */
  }
  else if (ipo==0 && func!=NULL) {
    f_alpha = (double *)calloc(nalpha, sizeof(double));
    h_alpha = (double *)calloc(nalpha, sizeof(double));
    m_alpha = (double *)calloc(ndata, sizeof(double));
    a_proj  = (double *)calloc(ndata, sizeof(double));
    
    /* calculate derivatives of func with respect to A_alpha */
    for(i=0; i<nalpha; i++) { /* loop over all prim. observables */
      SET_TO(h_alpha, nalpha, 0.0); 
      STDDEV(*(data+i), ndata, h_alpha+i);
      /* test */
      fprintf(stdout, "[uwerr] halpha = %18.16e\n", *(h_alpha+i));
      if(*(h_alpha+i)==0.0) {
	fprintf(stdout, "[uwerr] Warning: no fluctuation in primary observable %d\n", i);
	*(f_alpha + i) = 0.0;
      }
      else {
	ADD_ASSIGN(m_alpha, a_bb, h_alpha, nalpha);
	func(nalpha, m_alpha, npara, para, &func_res);
	*(f_alpha+i) = func_res;
	SUB_ASSIGN(m_alpha, a_bb, h_alpha, nalpha);
	func(nalpha, m_alpha, npara, para, &func_res);
	*(f_alpha+i) -= func_res;
	*(f_alpha+i) = *(f_alpha+i) / (2.0 * *(h_alpha+i));
      }
    }
    SET_TO(a_proj, ndata, 0.0);
    a_bb_proj = 0.0;
    for(i=0; i<nalpha; i++) {
      for(n=0; n<ndata; n++) { 
	*(a_proj + n) = *(a_proj + n) + ( *(*(data+i)+n) ) * ( *(f_alpha+i) );
      }
      a_bb_proj = a_bb_proj + *(a_bb+i) * (*(f_alpha+i));
    }
    free(m_alpha);
    free(f_alpha);
    free(h_alpha);
    for(n=0; n<nreplica; n++) { free(*(a_b+n)); }
    free(a_b);
    free(a_bb);
  }

  fprintf(stdout, "[uwerr]: have projected data ready\n");

  /**********************************************************************
   * calculate error, error of the error; automatic windowing condition *
   **********************************************************************/

  /* (1) Gamma_F(t), t=0,...,Wmax */
  SET_TO(Gamma_F, Wmax, 0.0);  
  SET_TO(tau_int, Wmax, 0.0);
  for(i=0,v_Fbb=0.0; i<ndata; i++) {
    v_Fbb = v_Fbb + SQR( (*(a_proj+i) - a_bb_proj) );
  }
  v_Fbb /= (double)ndata;
  C_F      = v_Fbb;
  *Gamma_F = v_Fbb;
  /* test */
  fprintf(stdout, "[uwerr] a_bb_proj  = %18.16e\n", a_bb_proj);
  fprintf(stdout, "[uwerr] Gamma_F(%1d) = %18.16e\n", 0, *Gamma_F);
  if (*Gamma_F==0.0) {
    fprintf(stderr, "[uwerr] ERROR, no fluctuations; return\n");
    strcpy(filename, obsname);
    strcat(filename,"_uwerr");
    ofs = fopen(filename, append);
    if ((void*)ofs==NULL) {
      fprintf(stderr, "[uwerr] Could not open file %s\n", filename);
      return(1);
    }
    fprintf(ofs, "%d\t%18.16e\t%18.16e\t%18.16e\t%18.16e\t%18.16e\t%18.16e\t"\
	    "%18.16e\t%18.16e\n", label, *F_bb, 0.0, 0.0, 0.0, \
	    0.0, -1.0, 0.0, 0.0);
    if (fclose(ofs)!=0) {
      fprintf(stderr, "[uwerr] Could not close file %s\n", filename);
      return(1);
    }
    return(-5);
  }
  *tau_int = 0.5; 
  for(W=1; W<Wmax-1; W++) {
    /* calculate Gamma_F(W) */
    data_ptr = a_proj;
    for(n=0; n<nreplica; n++) {
      for(i=0; i<(*(n_r+n)-W); i++) {
	*(Gamma_F+W) += (*(data_ptr+i) - a_bb_proj) * (*(data_ptr+i+W) - a_bb_proj);
      }
      data_ptr = data_ptr + *(n_r+n);
    }
    *(Gamma_F+W) = *(Gamma_F+W) / (double)(ndata-nreplica*W);
    /* test */
    fprintf(stdout, "[uwerr] Gamma_F(%d) = %18.16e\n", W, *(Gamma_F+W));
    C_F = C_F + 2.0 * *(Gamma_F+W);
    *tau_int = C_F / (2.0*v_Fbb);
    if(*tau_int < 0.5) {
      fprintf(stdout, "[uwerr] Warning: tau_int < 0.5; tau set to %f\n", TINY);
      tau = TINY;
    }
    else {
      tau = s_tau / log( (*tau_int+0.5) / (*tau_int-0.5) );
    }
    /* test */
    fprintf(stdout, "[uwerr] tau(%d) = %18.16e\n", W, tau);
    if( exp(-(double)W / tau) - tau / sqrt((double)(W*ndata)) < 0.0 ) {
      Wopt = W;
      /* test */
      fprintf(stdout, "[uwerr] Wopt = %d\n", Wopt);
      break;
     }
  }

  if(W==Wmax-1) {
    fprintf(stdout, "[uwerr] windowing condition failed after W = %d\n", W);
    return(1);
  }
  else {
    SUM(Gamma_F+1, Wopt, &C_Fopt);
    C_Fopt = 2.0 * C_Fopt + *Gamma_F;
    /* test */
    fprintf(stdout, "[uwerr] before: C_Fopt = %18.16e\n", C_Fopt);
    for(W=0; W<=Wopt; W++) {
      *(Gamma_F+W) = *(Gamma_F+W) + C_Fopt/((double)ndata);
    }
    SUM(Gamma_F+1, Wopt, &C_Fopt);
    C_Fopt = 2.0 * C_Fopt + *Gamma_F;
    /* test */
    fprintf(stdout, "[uwerr] after: C_Fopt = %18.16e\n", C_Fopt);
    v_Fbb = *Gamma_F;
    *tau_int = 0.5*v_Fbb;
    for(W=1; W<=Wopt; W++) *(tau_int+W) = *(tau_int+W-1) + *(Gamma_F+W);
    for(W=0; W<=Wopt; W++) *(tau_int+W) /= v_Fbb;
  }

  fprintf(stdout, "[uwerr]: perfomed automatic windowing\n");

  /***********************************
   * bias cancellation of mean value *
   ***********************************/
  if(nreplica > 1 ) {
    *F_bb = ( (double)nreplica * *F_bb - *F_bw ) / ((double)(nreplica-1));
  }

  fprintf(stdout, "[uwerr]: leading bias cancelled\n");


  /**************************
   * calculation of results *
   **************************/
  value = *F_bb;
  dvalue = sqrt(C_Fopt/((double)ndata));
  ddvalue = dvalue * sqrt((Wopt + 0.5)/ndata);
  tau_intbb = C_Fopt / (2.0 * v_Fbb);
  dtau_intbb = sqrt( 2.0 * ( 2.0*Wopt-3.0*tau_intbb + 1 + \
			     1/(4.0*tau_intbb))/((double)ndata) ) * tau_intbb;
  dv_Fbb = sqrt(2.0*(tau_intbb + 1/(4.0*tau_intbb)) / (double)ndata) * v_Fbb;

  /*******************************************
   * consistency checks in case nreplica > 0 *
   *******************************************/
  if(nreplica>1) {
  /* (1) calculate Q-value <---> determine goodness of the fit 
     F_b(n) = F_bw = const. */
    chisqr = 0.0;
    for(n=0; n<nreplica; n++) {
      chisqr = chisqr + SQR( *(F_b+n) - *F_bw ) / (C_Fopt/(double)(*(n_r+n)));
    }
    /* test */
    fprintf(stdout, "[uwerr]: chisqr = %18.16e\n", chisqr);
    fprintf(stdout, "[uwerr]: n      = %d     \n", (nreplica-1)/2);

    Qval = 1.0 - incomp_gamma(chisqr/2.0, (nreplica-1)/2);
  
  /* (2) inspection of p_r's defined below in a histogramm */
    p_r = (double *)calloc(nreplica, sizeof(double));
    for(n=0; n<nreplica; n++) {
      *(p_r+n) = (*(F_b+n) - *F_bw) / \
	(dvalue*sqrt(((double)ndata/(double)(*(n_r+n)))-1.0));
    }
    ARITHMEAN(p_r, nreplica, &p_r_mean);
    VAR(p_r, nreplica, &p_r_var);
    k = 1 + (int)rint(log((double)nreplica)/log(2.0));
    strcpy(filename, obsname);
    strcat(filename, "_uwerr_hist");
    ofs = fopen(filename, append);
    fprintf(ofs, "# mean of p_r's:\tp_r_mean = %8.6e\n" \
	    "# variance of p_r's:\tp_r_var = %8.6e\n",  \
	    p_r_mean, p_r_var);
    strcpy(format, "%%dst p_r(%2d) = %18.16e\n");
    for(n=0; n<nreplica; n++) {
      fprintf(ofs, format, n, *(p_r+n));
    }
    if(k<3) /* not enough classes for a meaningful histogramm */ {
      fprintf(ofs, "# [uwerr]: k = %d is to small\n", k);
    }
    else {
      ABS_MAX_DBL(p_r, nreplica, &lobd); /* max{|p_r's|} */
      lobd = lobd *(1.0+TINY);
      delta = 2.0*lobd/(double)k;        /* expected distribution around mean=0 */
      lobd = -lobd;                      /* lower boundary of abscissa */
      bins = (double *)calloc(k, sizeof(double)); /* contains number of entries */
      SET_TO(bins, k, 0.0);                       /* for each class */
      for(n=0; n<nreplica; n++) /* inc. bins(i) by 1, if p_r(n) is in class i */ {
	i = (int)((*(p_r+n) - lobd)/delta);
	*(bins + i) = *(bins + i) + 1.0;
      }
      fprintf(ofs, "# number of entries:\tnreplica = %d\n" \
	      "# number of classes:\tk = %d\n"             \
	      "# lower boundary:\tlobd = %8.6e\n"          \
	      "# bin width:\tdelta = %8.6e\n",             \
	      nreplica, k, lobd, delta);
      strcpy(format, "%%hst %18.16e\t%18.16e\n");
      for(i=0; i<k; i++) {
	fprintf(ofs, format, lobd+((double)i+0.5)*delta, *(bins+i));
      }
    }
    fclose(ofs);
  }

  /**************************
   * output                 *
   **************************/

  /* (1) value, dvalue, ... */  
  strcpy(filename, obsname);
  strcat(filename,"_uwerr");
  ofs = fopen(filename, append);
  if ((void*)ofs==NULL) {
    fprintf(stderr, "[uwerr] Could not open file %s\n", filename);
    return(1);
  }
  strcpy(format, "%d\t%18.16e\t%18.16e\t%18.16e\t%18.16e\t%18.16e\t%18.16e\t%18.16e\t%18.16e\n");
  fprintf(ofs, format, label, value, dvalue, ddvalue, tau_intbb, dtau_intbb, Qval, v_Fbb, dv_Fbb);
  if (fclose(ofs)!=0) {
    fprintf(stderr, "[uwerr] Could not close file %s\n", filename);
    return(1);
  }

  /* (2) Gamma_F */
  strcpy(filename, obsname);
  strcat(filename, "_uwerr_Gamma");
  ofs = fopen(filename, append);
  if ((void*)ofs==NULL) {
    fprintf(stderr, "[uwerr] Could not open file %s\n", filename);
    return(1);
  }
  strcpy(format, "%d\t%18.16e\n");
  fprintf(ofs, "# obsname = %s \t ipo = %d", obsname, ipo);
  for(W=0; W<=Wopt; W++) {
    fprintf(ofs, format, W, *(Gamma_F+W));
  }
  if (fclose(ofs)!=0) {
    fprintf(stderr, "[uwerr] Could not close file %s\n", filename);
    return(1);
  }

  /* (3) tau_int */
  strcpy(filename, obsname);
  strcat(filename, "_uwerr_tauint");
  ofs = fopen(filename, append);
  fprintf(ofs, "# obsname = %s \t ipo = %d", obsname, ipo);
  for(W=0; W<=Wopt; W++) {
    fprintf(ofs, format, W, *(tau_int+W));
  }
  fclose(ofs);

  fprintf(stdout, "[uwerr]: output written\n");

  /*****************************
   * free allocated disk space *
   *****************************/
  free(F_b);
  free(F_bb);
  free(F_bw);
  free(Gamma_F);
  free(tau_int);
  if(ipo==0 && func!=NULL) {
    free(a_proj);
  }

  return(0);

}
