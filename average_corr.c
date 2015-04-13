/*************************************************************
*average_corr.c 
*
*August, 16th, 2011
*average correlators obtained with get_corr from cvc_v_p
*for several point sources
*************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "stat5.h"

FILE *datafile, *resultfile;

#define FILESTOTAL 30
#define LINESTOTAL 24 

int main(){
	char g[3], file[200], file_prefix[10],result[10];
	int i,j, gc, T,LX,LY,LZ,gid, dummy1, dummy2,dummy3;
	double *corrt, *dummy, g_mu, g_kappa;
	int *numberq;

strcpy(file_prefix,"rho");
strcpy(result, "rho");

corrt=(double *) calloc(LINESTOTAL,sizeof(double));
  if(corrt==(double*)NULL) {
    printf("could not allocate memory for corrt\n");
    return(105);
  }
 dummy=(double *) calloc(LINESTOTAL,sizeof(double));
  if(dummy==(double*)NULL) {
    printf("could not allocate memory for dummy\n");
    return(106);
  }


clear5(LINESTOTAL,100);

for(i=0;i<FILESTOTAL;i++){

  gc=500+i*10;
  
  sprintf(file,"%s.%.4d", file_prefix, gc);
  
  datafile=fopen(file,"r");

   if (!datafile)
    {
     printf("*** Error opening %s ***\n",file);
     exit;
    }
    
   fscanf(datafile, "# %d %d %d %d %d %lf %lf\n", &gid, &T, &LX, &LY, &LZ, &g_kappa, &g_mu);
   for(j=0; j<LINESTOTAL; j++){
     fscanf(datafile, "%d %d %d %lf %lf %d\n", &dummy1, &dummy2, &dummy3, corrt+j, dummy+j, &gid);
//    printf("j=%4d,pi_q= %25.16lf\n",j,pi_q[j]);
    accum5(j+1,*(corrt+j));
   }

  fclose(datafile);
}

resultfile=fopen(result,"w");

  if (!resultfile)
    {
     printf("*** Error opening %s ***\n",result);
     exit;
    }

for(j=0; j<LINESTOTAL; j++){
   fprintf(resultfile, "%3d %3d %3d %25.16e %25.16e %.4d\n", 0, 0, j, 
	  aver5(j+1), sigma5(j+1), gc);
}

fclose(resultfile);

free(corrt);
free(dummy);

return 0;
}
