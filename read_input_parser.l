/***********************************************************************
 * read_input_parser.l
 * - parsing cvc.input to program
 * - use flex -P <prefix_> ...
 ***********************************************************************/

SPC [[:blank:]]+
CMD [:][[:space:]]+
RLN [1-9(10)(11)(12)(13)(14)(15)(16)][:]
DIGIT [[:digit:]]
ZT [0-9(10)(11)]
IDXEX ("-"{DIGIT}+)
SIGN ("+"|"-")
FLT {SIGN}?{DIGIT}*+"."{DIGIT}*(e("-"|"+")?{DIGIT}+)?
FILENAME [a-zA-Z0-9_".""-""/"][a-zA-z0-9"."_"-""/"]+
NAME [a-zA-Z0-9_]+
TYPE [0-9A-Z]+
SDIGIT {SIGN}?[[:digit:]]
STRING [a-zA-z0-9"."_"-""/"","]+

%{
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "global.h"
#include "read_input_parser.h"
#include "default_input_values.h"
#include "cvc_utils.h"

namespace cvc {

  /* Name of the parsing routine */
#define YY_DECL         int parse_config()
#define YY_NO_UNPUT

  /* declaration of input parameters */
  int i=0;
  int line_of_file=1;
  int comment_caller;
  int name_caller;
  int a,b;
  float c;
  int reread = 0;
  char name[100];
  char * type;
  double ft_rmax[4];

  int cvc_verbose = 1;
  int myverbose = 0;

}
using namespace cvc;
%}

%option never-interactive
%option debug

%x TT
%x LLX
%x LLY
%x LLZ
%x LL
%x LL5
%x NNCONF
%x KAPPA
%x MU
%x MUSIGMA
%x MUDELTA
%x MUBAR
%x M5
%x M0
%x EPSBAR
%x SOURCEID
%x SOURCEID2
%x GAUGEID
%x GAUGEID2
%x GAUGESTEP
%x NNSAVE
%x FFORMAT
%x BCT
%x BCX
%x BCY
%x BCZ
%x PREFIX
%x PREFIX2
%x PREFIX3
%x GAUGEPREFIX
%x OUTPREFIX
%x PATHPREFIX
%x RESUME
%x SUBTRACT
%x SOURCELOC
%x SSEED
%x NOISETYPE
%x SRCTYPE
%x SPREC
%x MAXITER
%x SOURCEIDSTEP
%x HPEMIN
%x HPEMAX
%x HPEORD
%x CUTDIRT
%x CUTDIRX
%x CUTDIRY
%x CUTDIRZ
%x CUTANG
%x CUTRAD
%x RMIN
%x RMAX
%x AVGTT
%x AVGLL
%x MODEL_DCOEFFRE
%x MODEL_DCOEFFIM
%x MODEL_MRHO
%x FTRMAX1
%x FTRMAX2
%x FTRMAX3
%x FTRMAX4
%x PROPNORMSQR
%x QHATSQRMIN
%x QHATSQRMAX
%x NLONG
%x NAPE
%x NJACOBI
%x NHYP
%x ALPHAAPE
%x ALPHAHYP
%x KAPPAJACOBI
%x SRCTIMESLICE
%x SEQSRCTIMESLICE
%x MMSNOMASSES
%x NOLIGHTMASSES
%x NOSTRANGEMASSES
%x NOTPROCS
%x NOXPROCS
%x NOYPROCS
%x LOCLOC
%x LOCSME
%x SMELOC
%x SMESME
%x ROTATE
%x PROPPOS
%x GPUDEVICENUMBER
%x GPUPERNODE
%x RELDELTA
%x COHSRC
%x COHSRCBASE
%x COHSRCDELTA
%x GAUGEFILEFORMAT
%x RNGFILE
%x SRCINDEX
%x PROPBCTYPE
%x PROPGAMMA
%x PROPPREC
%x WRITESRC
%x READSRC
%x NSAMPLE
%x ASOVERA
%x NUMTHREADS
%x SRCMOMX
%x SRCMOMY
%x SRCMOMZ
%x SNKMOMX
%x SNKMOMY
%x SNKMOMZ
%x SEQSRCMOMX
%x SEQSRCMOMY
%x SEQSRCMOMZ
%x CPUPREC
%x GPUPREC
%x GPUPRECSLOPPY
%x INVERTERTYPE
%x MMSID
%x CHKINVERSION
%x COMMENT
%x ERROR

%%
^T{SPC}*={SPC}*                            BEGIN(TT);
^L{SPC}*={SPC}*                            BEGIN(LL);
^L5{SPC}*={SPC}*                           BEGIN(LL5);
^LX{SPC}*={SPC}*                           BEGIN(LLX);
^LY{SPC}*={SPC}*                           BEGIN(LLY);
^LZ{SPC}*={SPC}*                           BEGIN(LLZ);
^mu{SPC}*={SPC}*                           BEGIN(MU);
^musigma{SPC}*={SPC}*                      BEGIN(MUSIGMA);
^mudelta{SPC}*={SPC}*                      BEGIN(MUDELTA);
^mubar{SPC}*={SPC}*                        BEGIN(MUBAR);
^m5{SPC}*={SPC}*                           BEGIN(M5);
^m0{SPC}*={SPC}*                           BEGIN(M0);
^epsbar{SPC}*={SPC}*                       BEGIN(EPSBAR);
^Nconf{SPC}*={SPC}*                        BEGIN(NNCONF);
^kappa{SPC}*={SPC}*                        BEGIN(KAPPA);
^sourceid{SPC}*={SPC}*                     BEGIN(SOURCEID);
^sourceid2{SPC}*={SPC}*                    BEGIN(SOURCEID2);
^gaugeid{SPC}*={SPC}*                      BEGIN(GAUGEID);
^gaugeid2{SPC}*={SPC}*                     BEGIN(GAUGEID2);
^gaugeid_step{SPC}*={SPC}*                 BEGIN(GAUGESTEP);
^Nsave{SPC}*={SPC}*                        BEGIN(NNSAVE);
^format{SPC}*={SPC}*                       BEGIN(FFORMAT);
^BCangleT{SPC}*={SPC}*                     BEGIN(BCT);
^BCangleX{SPC}*={SPC}*                     BEGIN(BCX);
^BCangleY{SPC}*={SPC}*                     BEGIN(BCY);
^BCangleZ{SPC}*={SPC}*                     BEGIN(BCZ);
^filename_prefix{SPC}*={SPC}*              BEGIN(PREFIX);
^filename_prefix2{SPC}*={SPC}*             BEGIN(PREFIX2);
^filename_prefix3{SPC}*={SPC}*             BEGIN(PREFIX3);
^gaugefilename_prefix{SPC}*={SPC}*         BEGIN(GAUGEPREFIX);
^outfile_prefix{SPC}*={SPC}*               BEGIN(OUTPREFIX);
^path_prefix{SPC}*={SPC}*                  BEGIN(PATHPREFIX);
^resume{SPC}*={SPC}*                       BEGIN(RESUME);
^subtract{SPC}*={SPC}*                     BEGIN(SUBTRACT);
^source_location{SPC}*={SPC}*              BEGIN(SOURCELOC);
^seed{SPC}*={SPC}*                         BEGIN(SSEED);
^noise_type{SPC}*={SPC}*                   BEGIN(NOISETYPE);
^source_type{SPC}*={SPC}*                  BEGIN(SRCTYPE);
^solver_precision{SPC}*={SPC}*             BEGIN(SPREC);
^reliable_delta{SPC}*={SPC}*               BEGIN(RELDELTA);
^niter_max{SPC}*={SPC}*                    BEGIN(MAXITER);
^sourceid_step{SPC}*={SPC}*                BEGIN(SOURCEIDSTEP);
^hpe_order_min{SPC}*={SPC}*                BEGIN(HPEMIN);
^hpe_order_max{SPC}*={SPC}*                BEGIN(HPEMAX);
^hpe_order{SPC}*={SPC}*                    BEGIN(HPEORD);
^cut_angle{SPC}*={SPC}*                    BEGIN(CUTANG);
^cut_radius{SPC}*={SPC}*                   BEGIN(CUTRAD);
^cut_dirT{SPC}*={SPC}*                     BEGIN(CUTDIRT);
^cut_dirX{SPC}*={SPC}*                     BEGIN(CUTDIRX);
^cut_dirY{SPC}*={SPC}*                     BEGIN(CUTDIRY);
^cut_dirZ{SPC}*={SPC}*                     BEGIN(CUTDIRZ);
^Rmin{SPC}*={SPC}*                         BEGIN(RMIN);
^Rmax{SPC}*={SPC}*                         BEGIN(RMAX);
^avgT{SPC}*={SPC}*                         BEGIN(AVGTT);
^avgL{SPC}*={SPC}*                         BEGIN(AVGLL);
^dcoeff_re{SPC}*={SPC}*                    BEGIN(MODEL_DCOEFFRE);
^dcoeff_im{SPC}*={SPC}*                    BEGIN(MODEL_DCOEFFIM);
^mrho{SPC}*={SPC}*                         BEGIN(MODEL_MRHO);
^ft_rmax1{SPC}*={SPC}*                     BEGIN(FTRMAX1);
^ft_rmax2{SPC}*={SPC}*                     BEGIN(FTRMAX2);
^ft_rmax3{SPC}*={SPC}*                     BEGIN(FTRMAX3);
^ft_rmax4{SPC}*={SPC}*                     BEGIN(FTRMAX4);
^prop_normsqr{SPC}*={SPC}*                 BEGIN(PROPNORMSQR);
^qhatsqr_min{SPC}*={SPC}*                  BEGIN(QHATSQRMIN);
^qhatsqr_max{SPC}*={SPC}*                  BEGIN(QHATSQRMAX);
^Nlong{SPC}*={SPC}*                        BEGIN(NLONG);
^N_ape{SPC}*={SPC}*                        BEGIN(NAPE);
^N_Jacobi{SPC}*={SPC}*                     BEGIN(NJACOBI);
^alpha_ape{SPC}*={SPC}*                    BEGIN(ALPHAAPE);
^N_hyp{SPC}*={SPC}*                        BEGIN(NHYP);
^alpha_hyp{SPC}*={SPC}*                    BEGIN(ALPHAHYP);
^kappa_Jacobi{SPC}*={SPC}*                 BEGIN(KAPPAJACOBI);
^source_timeslice{SPC}*={SPC}*             BEGIN(SRCTIMESLICE);
^sequential_source_timeslice{SPC}*={SPC}*  BEGIN(SEQSRCTIMESLICE);
^no_extra_masses{SPC}*={SPC}*              BEGIN(MMSNOMASSES);
^no_light_masses{SPC}*={SPC}*              BEGIN(NOLIGHTMASSES);
^no_strange_masses{SPC}*={SPC}*            BEGIN(NOSTRANGEMASSES);
^NrTProcs{SPC}*={SPC}*                     BEGIN(NOTPROCS);
^NrXProcs{SPC}*={SPC}*                     BEGIN(NOXPROCS);
^NrYProcs{SPC}*={SPC}*                     BEGIN(NOYPROCS);
^local_local{SPC}*={SPC}*                  BEGIN(LOCLOC);
^local_smeared{SPC}*={SPC}*                BEGIN(LOCSME);
^smeared_local{SPC}*={SPC}*                BEGIN(SMELOC);
^smeared_smeared{SPC}*={SPC}*              BEGIN(SMESME);
^rotate_etmc_ukqcd{SPC}*={SPC}*            BEGIN(ROTATE);
^propagator_position{SPC}*={SPC}*          BEGIN(PROPPOS);
^gpu_device_number{SPC}*={SPC}*            BEGIN(GPUDEVICENUMBER);
^gpu_per_node{SPC}*={SPC}*                 BEGIN(GPUPERNODE);
^coherent_source{SPC}*={SPC}*              BEGIN(COHSRC);
^coherent_source_base{SPC}*={SPC}*         BEGIN(COHSRCBASE);
^coherent_source_delta{SPC}*={SPC}*        BEGIN(COHSRCDELTA);
^gauge_file_format{SPC}*={SPC}*            BEGIN(GAUGEFILEFORMAT);
^rng_filename{SPC}*={SPC}*                 BEGIN(RNGFILE);
^source_indices{SPC}*={SPC}*               BEGIN(SRCINDEX);
^propagator_bc_type{SPC}*={SPC}*           BEGIN(PROPBCTYPE);
^propagator_gamma_basis{SPC}*={SPC}*       BEGIN(PROPGAMMA);
^propagator_precision{SPC}*={SPC}*         BEGIN(PROPPREC);
^write_source{SPC}*={SPC}*                 BEGIN(WRITESRC);
^read_source{SPC}*={SPC}*                  BEGIN(READSRC);
^samples{SPC}*={SPC}*                      BEGIN(NSAMPLE);
^as_over_a{SPC}*={SPC}*                    BEGIN(ASOVERA);
^num_threads{SPC}*={SPC}*                  BEGIN(NUMTHREADS);
^source_momentum_x{SPC}*={SPC}*            BEGIN(SRCMOMX);
^source_momentum_y{SPC}*={SPC}*            BEGIN(SRCMOMY);
^source_momentum_z{SPC}*={SPC}*            BEGIN(SRCMOMZ);
^sink_momentum_x{SPC}*={SPC}*              BEGIN(SNKMOMX);
^sink_momentum_y{SPC}*={SPC}*              BEGIN(SNKMOMY);
^sink_momentum_z{SPC}*={SPC}*              BEGIN(SNKMOMZ);
^seq_source_momentum_x{SPC}*={SPC}*        BEGIN(SEQSRCMOMX);
^seq_source_momentum_y{SPC}*={SPC}*        BEGIN(SEQSRCMOMY);
^seq_source_momentum_z{SPC}*={SPC}*        BEGIN(SEQSRCMOMZ);
^cpu_precision{SPC}*={SPC}*                BEGIN(CPUPREC);
^gpu_precision{SPC}*={SPC}*                BEGIN(GPUPREC);
^gpu_precision_sloppy{SPC}*={SPC}*         BEGIN(GPUPRECSLOPPY);
^inverter_type{SPC}*={SPC}*                BEGIN(INVERTERTYPE);
^mms_id{SPC}*={SPC}*                       BEGIN(MMSID);
^check_inversion{SPC}*={SPC}*              BEGIN(CHKINVERSION);


  /* start namespace inclusion here */

<TT>{DIGIT}+                  {
  T_global = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] T_global =%s\n", yytext);
}
<LL>{DIGIT}+                  {
  L = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] L =%s\n", yytext);
}
<LL5>{DIGIT}+                  {
  L5 = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] L5 =%s\n", yytext);
}
<LLX>{DIGIT}+                  {
  LX = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] LX =%s\n", yytext);
}
<LLY>{DIGIT}+                  {
  LY = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] LY =%s\n", yytext);
}
<LLZ>{DIGIT}+                  {
  LZ = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] LZ =%s\n", yytext);
}
<KAPPA>{FLT}  {
  g_kappa=atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] kappa=%s \n", yytext);
}
<MU>{FLT}  {
  g_mu=atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] mu=%s \n", yytext);
}
<M5>{FLT}  {
  g_m5 = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] m5=%s \n", yytext);
}
<MUSIGMA>{FLT}  {
  g_musigma=atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] musigma=%s \n", yytext);
}
<MUDELTA>{FLT}  {
  g_mudelta=atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] mudelta=%s \n", yytext);
}
<MUBAR>{FLT}  {
  g_mubar=atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] mubar=%s \n", yytext);
}
<M0>{FLT}  {
  g_m0 = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] m0=%s \n", yytext);
}
<EPSBAR>{FLT}  {
  g_epsbar=atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] mu=%s \n", yytext);
}
<NNSAVE>{SDIGIT}+ {
  Nsave=atoi(yytext); 
  if(myverbose!=0) printf("# [read_input_parser] Nsave= %s \n",yytext);
}
<NNCONF>{DIGIT}+ {
  Nconf=atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] Nconf=%s\n", yytext);
}
<BCT>{FLT} {
  BCangle[0]=atof(yytext);
  if(myverbose != 0) printf("# [read_input_parser] BCangle[0] for bound. in time set to %e\n", BCangle[0]);
}
<BCX>{FLT} {
  BCangle[1]=atof(yytext);
  if(myverbose != 0) printf("# [read_input_parser] BCangle[1] for bound. in x set to %e\n", BCangle[1]);
}
<BCY>{FLT} {
  BCangle[2]=atof(yytext);
  if(myverbose != 0) printf("# [read_input_parser] BCangle[2] for bound. in y set to %e\n", BCangle[2]);
}
<BCZ>{FLT} {
  BCangle[3]=atof(yytext);
  if(myverbose != 0) printf("# [read_input_parser] BCangle[3] for bound. in z set to %e\n", BCangle[3]);
}
<SOURCEID>{DIGIT}+ {
  g_sourceid = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] first source id set to %s\n", yytext);
}
<SOURCEID2>{DIGIT}+ {
  g_sourceid2 = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] second source id set to %s\n", yytext);
}
<SOURCEIDSTEP>{DIGIT}+ {
  g_sourceid_step = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] source id step set to %s\n", yytext);
}
<GAUGEID>{DIGIT}+ {
  g_gaugeid = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] first gauge id set to %s\n", yytext);
}
<GAUGEID2>{DIGIT}+ {
  g_gaugeid2 = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] second gauge id set to %s\n", yytext);
}
<GAUGESTEP>{DIGIT}+ {
  g_gauge_step = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] gauge step size set to %s\n", yytext);
}
<PREFIX>{FILENAME} {
  strcpy(filename_prefix,yytext);
  if(myverbose!=0) printf("# [read_input_parser] prop. filename prefix set to %s\n",yytext);
}
<PREFIX2>{FILENAME} {
  strcpy(filename_prefix2,yytext);
  if(myverbose!=0) printf("# [read_input_parser] prop. 2 filename prefix set to %s\n",yytext);
}
<PREFIX3>{FILENAME} {
  strcpy(filename_prefix3,yytext);
  if(myverbose!=0) printf("# [read_input_parser] prop. 3 filename prefix set to %s\n",yytext);
}
<GAUGEPREFIX>{FILENAME} {
  strcpy(gaugefilename_prefix,yytext);
  if(myverbose!=0) printf("# [read_input_parser] gauge filename prefix set to %s\n",yytext);
}
<OUTPREFIX>{FILENAME} {
  strcpy(g_outfile_prefix,yytext);
  if(myverbose!=0) printf("# [read_input_parser] out filename prefix set to %s\n",yytext);
}
<PATHPREFIX>{FILENAME} {
  strcpy(g_path_prefix,yytext);
  if(myverbose!=0) printf("# [read_input_parser] path prefix set to %s\n",yytext);
}
<RESUME>{DIGIT}+ {
  g_resume=atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] g_resume set to %s\n",yytext);
}
<SUBTRACT>{DIGIT}+ {
  g_subtract=atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] g_subtract set to %s\n",yytext);
}
<SOURCELOC>{DIGIT}+ {
  g_source_location=atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] source location set to %s\n",yytext);
}
<FFORMAT>{DIGIT}+ {
  format = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] format set to %s\n",yytext);
}  
<SSEED>{DIGIT}+ {
  g_seed = (unsigned int)atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] g_seed set to %s\n",yytext);
}  
<NOISETYPE>{NAME} {
  if(strcmp(yytext, "Gaussian")==0 ) {
    g_noise_type = 1;
  } else if(strcmp(yytext, "Z2")==0 ) {
    g_noise_type = 2;
  }
  if(myverbose!=0) printf("# [read_input_parser] noise type set to %s\n",yytext);
}  
<SRCTYPE>{FILENAME} {
  if(strcmp(yytext, "Point")==0) {
    g_source_type = 0;
  } else if (strcmp(yytext, "Volume")==0 ) {
    g_source_type = 1;
  } else if (strcmp(yytext, "Timeslice")==0 ) {
    g_source_type = 2;
  } else if (strcmp(yytext, "TimesliceOneEnd")==0 ) {
    g_source_type = 3;
  } else if (strcmp(yytext, "TimesliceOneEndColor")==0 ) {
    g_source_type = 4;
  } else if (strcmp(yytext, "SequentialPoint")==0 ) {
    g_source_type = 5;
  } else if (strcmp(yytext, "Other")==0 ) {
    g_source_type = -1;
  } 
  if(myverbose!=0) printf("# [read_input_parser] source type set to %s\n",yytext);
}
<SPREC>{FLT} {
  solver_precision = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] solver precision set to %s\n",yytext);
}
<RELDELTA>{FLT} {
  reliable_delta = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] reliable delta set to %s\n",yytext);
}
<MAXITER>{DIGIT}+ {
  niter_max = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] max. number of iterations set to %s\n",yytext);
}
<HPEMIN>{DIGIT}+ {
  hpe_order_min = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] min. HPE order set to %s\n",yytext);
}
<HPEMAX>{DIGIT}+ {
  hpe_order_max = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] max. HPE order set to %s\n",yytext);
}
<HPEORD>{DIGIT}+ {
  hpe_order = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] HPE order set to %s\n",yytext);
}
<CUTANG>{FLT} {
  g_cutangle = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] cone cut angle set to %s\n",yytext);
}
<CUTRAD>{FLT} {
  g_cutradius = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] cone cut radius set to %s\n",yytext);
}
<CUTDIRT>{DIGIT}+ {
  g_cutdir[0] = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] t-comp. of cut dir angle set to %s\n",yytext);
}
<CUTDIRX>{DIGIT}+ {
  g_cutdir[1] = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] x-comp. of cut dir angle set to %s\n",yytext);
}
<CUTDIRY>{DIGIT}+ {
  g_cutdir[2] = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] y-comp. of cut dir angle set to %s\n",yytext);
}
<CUTDIRZ>{DIGIT}+ {
  g_cutdir[3] = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] z-comp. of cut dir angle set to %s\n",yytext);
}
<RMIN>{FLT} {
  g_rmin = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] minimal distance set to %s\n",yytext);
}
<RMAX>{FLT} {
  g_rmax = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] maximal distance set to %s\n",yytext);
}
<AVGTT>{DIGIT}+                  {
  avgT = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] avgT =%s\n", yytext);
}
<AVGLL>{DIGIT}+                  {
  avgL = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] avgL =%s\n", yytext);
}
<MODEL_DCOEFFRE>{FLT} {
  model_dcoeff_re = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] real part of d-coeff. set to %s\n", yytext);
}
<MODEL_DCOEFFIM>{FLT} {
  model_dcoeff_im = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] imaginary part of d-coeff. set to %s\n", yytext);
}
<MODEL_MRHO>{FLT} {
  model_mrho = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] rho-meson mass set to %s\n", yytext);
}
<FTRMAX1>{FLT} {
  ft_rmax[0] = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] 1st rmax for FT set to %s\n", yytext);
}
<FTRMAX2>{FLT} {
  ft_rmax[1] = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] 2nd rmax for FT set to %s\n", yytext);
}
<FTRMAX3>{FLT} {
  ft_rmax[2] = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] 3rd rmax for FT set to %s\n", yytext);
}
<FTRMAX4>{FLT} {
  ft_rmax[3] = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] 4th rmax for FT set to %s\n", yytext);
}
<PROPNORMSQR>{FLT} {
  g_prop_normsqr = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] propagator norm squared set to %s\n", yytext);
}
<QHATSQRMIN>{FLT} {
  g_qhatsqr_min = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set minimal squared Qhat to %s\n", yytext);
}
<QHATSQRMAX>{FLT} {
  g_qhatsqr_max = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set maximal squared Qhat to %s\n", yytext);
}
<NLONG>{DIGIT}+ {
  Nlong = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set Nlong to %s\n", yytext);
}
<NAPE>{DIGIT}+ {
  N_ape = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set N_ape to %s\n", yytext);
}
<NHYP>{DIGIT}+ {
  N_hyp = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set N_hyp to %s\n", yytext);
}
<NJACOBI>{DIGIT}+ {
  N_Jacobi = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set N_Jacobi to %s\n", yytext);
}
<ALPHAAPE>{FLT} {
  alpha_ape = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set alpha_ape to %s\n", yytext);
}
<ALPHAHYP>{FILENAME} {
  sscanf(yytext, "%lf/%lf/%lf", alpha_hyp, alpha_hyp+1, alpha_hyp+2);
  if(myverbose!=0) printf("# [read_input_parser] set alpha_hyp to %f, %f, %f\n", alpha_hyp[0], alpha_hyp[1], alpha_hyp[2]);
  /* if(myverbose!=0) printf("# [read_input_parser] yytext is %s\n", yytext); */
}
<KAPPAJACOBI>{FLT} {
  kappa_Jacobi = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set kappa_Jacobi to %s\n", yytext);
}
<SRCTIMESLICE>{DIGIT}+ {
  g_source_timeslice = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set source timeslice to %s\n", yytext);
}
<SEQSRCTIMESLICE>{DIGIT}+ {
  g_sequential_source_timeslice = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set sequential source timeslice to %s\n", yytext);
}
<MMSNOMASSES>{DIGIT}+ {
  g_no_extra_masses = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set no. of extra masses for mms to %s\n", yytext);
}
<NOLIGHTMASSES>{DIGIT}+ {
  g_no_light_masses = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set no. of light masses for mms to %s\n", yytext);
}
<NOSTRANGEMASSES>{DIGIT}+ {
  g_no_strange_masses = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set no. of strange masses for mms to %s\n", yytext);
}
<NOTPROCS>{DIGIT}+ {
  g_nproc_t = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set g_nproc_t to %s\n", yytext);
}
<NOXPROCS>{DIGIT}+ {
  g_nproc_x = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set g_nproc_x to %s\n", yytext);
}
<NOYPROCS>{DIGIT}+ {
  g_nproc_y = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set g_nproc_y to %s\n", yytext);
}
<LOCLOC>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    g_local_local = 1;
    if(myverbose!=0) printf("# [read_input_parser] do local-local contraction\n");
  }
}
<LOCSME>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    g_local_smeared = 1;
    if(myverbose!=0) printf("# [read_input_parser] do local-smeared contraction\n");
  }
}
<SMELOC>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    g_smeared_local = 1;
    if(myverbose!=0) printf("# [read_input_parser] do smeared-local contraction\n");
  }
}
<SMESME>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    g_smeared_smeared = 1;
    if(myverbose!=0) printf("# [read_input_parser] do smeared-smeared contraction\n");
  }
}
<ROTATE>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    g_rotate_ETMC_UKQCD = 1;
    if(myverbose!=0) printf("# [read_input_parser] rotate propagators UKQCD -> ETMC\n");
  }
}
<PROPPOS>{DIGIT}+ {
  g_propagator_position = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set propagator position to %s\n", yytext);
}

<GPUDEVICENUMBER>{DIGIT}+ {
  g_gpu_device_number = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set GPU device number to %s\n", yytext);
}
<GPUPERNODE>{DIGIT}+ {
  g_gpu_per_node = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set number of GPUs  per node to %s\n", yytext);
}

<COHSRC>{DIGIT}+ {
  g_coherent_source = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set coherent source to %s\n", yytext);
}

<COHSRCBASE>{DIGIT}+ {
  g_coherent_source_base = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set coherent source base to %s\n", yytext);
}

<COHSRCDELTA>{DIGIT}+ {
  g_coherent_source_delta = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set coherent source delta to %s\n", yytext);
}
<GAUGEFILEFORMAT>{NAME} {
  if(strcmp(yytext, "ildg")==0) {
    g_gauge_file_format = 0;
  } else if(strcmp(yytext, "nersc")==0) {
    g_gauge_file_format = 1;
  }
  if(myverbose!=0) printf("# [read_input_parser] set gauge file format to %s\n", yytext);
}
<RNGFILE>{FILENAME} {
//  g_rng_filename = (char*)malloc((strlen(yytext)+1)*sizeof(char));
  strcpy(g_rng_filename, yytext);
  if(myverbose!=0) printf("# [read_input_parser] g_rng_filename set to %s\n", g_rng_filename);
}
<SRCINDEX>{DIGIT}+ {
  g_source_index[0] = atoi(yytext);
  g_source_index[1] = g_source_index[0];
  if(myverbose!=0) printf("# [read_input_parser] start index set to %d\n", g_source_index[0]);
}
<SRCINDEX>{IDXEX} {
  sscanf(yytext, "-%d", g_source_index+1);
  if(myverbose!=0) printf("# [read_input_parser] end index set to %d\n", g_source_index[1]);
}
<PROPBCTYPE>{NAME} {
  if(strcmp(yytext, "Phase")==0) {
    g_propagator_bc_type = 0;
  } else if(strcmp(yytext, "Sign")==0) {
    g_propagator_bc_type = 1;
  } else {
    g_propagator_bc_type = -1;
  }
  if(myverbose!=0) printf("# [read_input_parser] propagator boundary condition type set to %d\n", g_propagator_bc_type);
}
<WRITESRC>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    g_write_source = 1;
  } else if(strcmp(yytext,"no")==0) {
    g_write_source = 0;
  }
  if(myverbose!=0) printf("# [read_input_parser] set write source to %d\n", g_write_source);  
}
<READSRC>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    g_read_source = 1;
  } else if(strcmp(yytext,"no")==0) {
    g_read_source = 0;
  }
  if(myverbose!=0) printf("# [read_input_parser] set write source to %d\n", g_write_source);  
}
<PROPGAMMA>{NAME} {
  if(strcmp(yytext,"DeGrandRossi")==0) {
    g_propagator_gamma_basis = 1;
  } else if(strcmp(yytext,"UKQCD")==0) {
    g_propagator_gamma_basis = 2;
  }
  if(myverbose!=0) printf("# [read_input_parser] set propagator gamma basis type to %d\n", g_propagator_gamma_basis);
}
<PROPPREC>{DIGIT}+ {
  g_propagator_precision = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set propagator precision to %d\n", g_propagator_precision);
}
<NSAMPLE>{DIGIT}+ {
  g_nsample = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set number of samples to %d\n", g_nsample);
}
<ASOVERA>{FLT} {
  g_as_over_a = atof(yytext);
  if(myverbose!=0) printf("# [read_input_parser] as / a set to  %s\n",yytext);
}
<NUMTHREADS>{DIGIT}+ {
  g_num_threads = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] number of threads =%d\n", g_num_threads);
}
<SRCMOMX>{SDIGIT}+ {
  g_source_momentum[0] = atoi(yytext);
  g_source_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] x-comp. of source momentum set to %d\n", g_source_momentum[0]);
}
<SRCMOMY>{SDIGIT}+ {
  g_source_momentum[1] = atoi(yytext);
  g_source_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] y-comp. of source momentum set to %d\n", g_source_momentum[1]);
}
<SRCMOMZ>{SDIGIT}+ {
  g_source_momentum[2] = atoi(yytext);
  g_source_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] z-comp. of source momentum set to %d\n", g_source_momentum[2]);
}
<SNKMOMX>{SDIGIT}+ {
  g_sink_momentum[0] = atoi(yytext);
  g_sink_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] x-comp. of sink momentum set to %d\n", g_sink_momentum[0]);
}
<SNKMOMY>{SDIGIT}+ {
  g_sink_momentum[1] = atoi(yytext);
  g_sink_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] y-comp. of sink momentum set to %d\n", g_sink_momentum[1]);
}
<SNKMOMZ>{SDIGIT}+ {
  g_sink_momentum[2] = atoi(yytext);
  g_sink_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] z-comp. of sink momentum set to %d\n", g_sink_momentum[2]);
}
<SEQSRCMOMX>{SDIGIT}+ {
  g_seq_source_momentum[0] = atoi(yytext);
  g_seq_source_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] x-comp. of sequential source momentum set to %d\n", g_seq_source_momentum[0]);
}
<SEQSRCMOMY>{SDIGIT}+ {
  g_seq_source_momentum[1] = atoi(yytext);
  g_seq_source_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] y-comp. of sequential source momentum set to %d\n", g_seq_source_momentum[1]);
}
<SEQSRCMOMZ>{SDIGIT}+ {
  g_seq_source_momentum[2] = atoi(yytext);
  g_seq_source_momentum_set = 1;
  if(myverbose!=0) printf("# [read_input_parser] z-comp. of sequential source momentum set to %d\n", g_seq_source_momentum[2]);
}
<CPUPREC>{NAME} {
  if(strcmp(yytext, "half")==0 ) {
    g_cpu_prec = 0;
  } else if(strcmp(yytext, "single")==0 ) {
    g_cpu_prec = 1;
  } else if(strcmp(yytext, "double")==0 ) {
    g_cpu_prec = 2;
  }
  if(myverbose!=0) printf("# [read_input_parser] CPU precision (for QUDA library) set to %s\n",yytext);
}
<GPUPREC>{NAME} {
  if(strcmp(yytext, "half")==0 ) {
    g_gpu_prec = 0;
  } else if(strcmp(yytext, "single")==0 ) {
    g_gpu_prec = 1;
  } else if(strcmp(yytext, "double")==0 ) {
    g_gpu_prec = 2;
  }
  if(myverbose!=0) printf("# [read_input_parser] GPU precision (for QUDA library) set to %s\n",yytext);
}
<GPUPRECSLOPPY>{NAME} {
  if(strcmp(yytext, "half")==0 ) {
    g_gpu_prec_sloppy = 0;
  } else if(strcmp(yytext, "single")==0 ) {
    g_gpu_prec_sloppy = 1;
  } else if(strcmp(yytext, "double")==0 ) {
    g_gpu_prec_sloppy = 2;
  }
  if(myverbose!=0) printf("# [read_input_parser] GPU sloppy precision (for QUDA library) set to %s\n",yytext);
}
<INVERTERTYPE>{NAME} {
  strcpy(g_inverter_type_name, yytext);
  if(myverbose!=0) printf("# [read_input_parser] inverter type name set to %s\n",yytext);
}
<MMSID>{DIGIT}+ {
  g_mms_id = atoi(yytext);
  if(myverbose!=0) printf("# [read_input_parser] set mms id to %s\n", yytext);
}
<CHKINVERSION>{NAME} {
  if(strcmp(yytext, "yes")==0) {
    g_check_inversion = 1;
  } else if(strcmp(yytext, "no")==0)  {
    g_check_inversion = 0;
  }
  if(myverbose!=0) printf("# [read_input_parser] check inversion %d\n",g_check_inversion);
}

<*>^#   {
   comment_caller = YY_START;   
   BEGIN(COMMENT);
}
<*>{SPC}*#    {
   comment_caller = YY_START;
   BEGIN(COMMENT);
}
<COMMENT>[^\n]*             {
  BEGIN(comment_caller);
}

<*>\n                       {
  line_of_file++;
  BEGIN(0);
}

<*>. {
  BEGIN(ERROR);
}
<ERROR>[^\t\n]*             {
  printf("# [read_input_parser] Parsing error in line %d\nAborting...!\n", line_of_file);
  exit(1);
}

  /* end namespace inclusion here */

%%

/*
 *  Dummy (but not dumb) routine - well, function
 */

namespace cvc {

int yywrap()
{
  return(1);
}

/* 
 * This is the function to parse the input file.
 * default values for all paramters will be set
 * correspondig to settings in
 * default_input_values.h
 *
 * read_input expects the filename of the input file
 * as an input parameter.
 *
 * read_input returns 2 if the input file did not exist 
 */

int read_input_parser(char * conf_file){

  /********************************************
   * Setting default values!
   ********************************************/
  reread = 0;
  set_default_input_values();

  if( (g_proc_id == 0) && (cvc_verbose>0) ) {
    myverbose = 1;
  }
  if ((yyin = fopen(conf_file, "rt")) == NULL){
    return(2);
  }
  yyout = fopen("/dev/null", "w");

  parse_config();  
  
  fclose(yyout);
  fclose(yyin);
  return(0);
}


/* 
 * This is the function to parse the input file 
 * again. Only parameters are changed, that
 * are specified in the input file.
 * default values for paramters will not be set.
 *
 * reread_input expects the filename of the input file
 * as an input parameter.
 *
 * reread_input returns 2 if the input file did not exist 
 */

int reread_input_parser(char * conf_file){

  if(cvc_verbose && g_proc_id == 0) {
    myverbose = 1;
  }
  reread = 1;

  /********************************************
   * Setting default values!
   ********************************************/

  /********************************************/

  if ((yyin = fopen(conf_file, "rt")) == NULL){
    return(2);
  }
  yyout = fopen("/dev/null", "w");

  parse_config();  

  fclose(yyout);
  fclose(yyin);
  return(0);
}

}  /* of namespace cvc */
