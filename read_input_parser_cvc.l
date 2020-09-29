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
FILENAME [a-zA-Z0-9_".""\-""/"][a-zA-Z0-9"."_"\-""/"]+
CORRCONFIG [0-9","";""\-""("")"][0-9","";""\-""("")"]+
NAME [a-zA-Z0-9_]+
TYPE [0-9A-Z]+
SDIGIT {SIGN}?[[:digit:]]
STRING [a-zA-Z0-9"."_"\-""/"","+]+
EQL {SPC}*={SPC}*

%{
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "types.h"
#include "global.h"
#include "read_input_parser.h"
#include "default_input_values.h"
#include "set_default.h"
#include "cvc_utils.h"
#include "twopoint_function_utils.h"

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
  char fname[100];
  char * type;
  double ft_rmax[4];

  int cvc_verbose = 1;
  int myverbose = 0;
  int myprocid = 0;

  twopoint_function_type twopoint_function_generic, *twopoint_function_ptr=NULL;
  int twopoint_function_generic_set = 0;
  char twopoint_function_set_type[10];
}
%}

%option never-interactive

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
%x CSW
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
%x SEQPREFIX
%x SEQPREFIX2
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
%x SEQSRCLOCATIONX
%x SEQSRCLOCATIONY
%x SEQSRCLOCATIONZ
%x SEQSRCGAMMAID
%x SRCGAMMAID
%x MMSNOMASSES
%x NOLIGHTMASSES
%x NOSTRANGEMASSES
%x NOTPROCS
%x NOXPROCS
%x NOYPROCS
%x NOZPROCS
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
%x WRITEPROP
%x READPROP
%x READSEQPROP
%x WRITESEQSRC
%x WRITESEQPROP
%x NSAMPLE
%x NSAMPLEOET
%x ASOVERA
%x NUMTHREADS
%x SRCMOM
%x SRCMOMX
%x SRCMOMY
%x SRCMOMZ
%x SNKMOM
%x SNKMOMX
%x SNKMOMY
%x SNKMOMZ
%x SEQSRCMOM
%x SEQSRCMOMX
%x SEQSRCMOMY
%x SEQSRCMOMZ
%x SEQ2SRCMOM
%x SEQ2SRCMOMX
%x SEQ2SRCMOMY
%x SEQ2SRCMOMZ
%x CPUPREC
%x GPUPREC
%x GPUPRECSLOPPY
%x INVERTERTYPE
%x MMSID
%x CHKINVERSION
%x COMMENT
%x ERROR
%x FERMION_TYPE
%x SOURCECOORDS
%x SRCSNKTIMESEP
%x COHERENTSOURCENUMBER
%x VERBOSITYLEVEL
%x TOTALMOMENTUM
%x INITTWOPOINTFUNCTION
%x TWOPOINTFUNCTION
%x GAMMA_F1_NUCLEON
%x GAMMA_F1_DELTA
%x GAMMA_F2
%x GAMMA_CURRENT


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
^csw{SPC}*={SPC}*                          BEGIN(CSW);
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
^sequential_filename_prefix{SPC}*={SPC}*   BEGIN(SEQPREFIX);
^sequential_filename_prefix2{SPC}*={SPC}*  BEGIN(SEQPREFIX2);
^gaugefilename_prefix{SPC}*={SPC}*         BEGIN(GAUGEPREFIX);
^outfile_prefix{SPC}*={SPC}*               BEGIN(OUTPREFIX);
^path_prefix{SPC}*={SPC}*                  BEGIN(PATHPREFIX);
^resume{SPC}*={SPC}*                       BEGIN(RESUME);
^subtract{SPC}*={SPC}*                     BEGIN(SUBTRACT);
^source_location{SPC}*={SPC}*              BEGIN(SOURCELOC);
^source_coords{SPC}*={SPC}*                BEGIN(SOURCECOORDS);
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
^sequential_source_location_x{SPC}*={SPC}* BEGIN(SEQSRCLOCATIONX);
^sequential_source_location_y{SPC}*={SPC}* BEGIN(SEQSRCLOCATIONY);
^sequential_source_location_z{SPC}*={SPC}* BEGIN(SEQSRCLOCATIONZ);

^sequential_source_gamma_id{SPC}*={SPC}*   BEGIN(SEQSRCGAMMAID);
^source_gamma_id{SPC}*={SPC}*              BEGIN(SRCGAMMAID);


^no_extra_masses{SPC}*={SPC}*              BEGIN(MMSNOMASSES);
^no_light_masses{SPC}*={SPC}*              BEGIN(NOLIGHTMASSES);
^no_strange_masses{SPC}*={SPC}*            BEGIN(NOSTRANGEMASSES);
^NrTProcs{SPC}*={SPC}*                     BEGIN(NOTPROCS);
^NrXProcs{SPC}*={SPC}*                     BEGIN(NOXPROCS);
^NrYProcs{SPC}*={SPC}*                     BEGIN(NOYPROCS);
^NrZProcs{SPC}*={SPC}*                     BEGIN(NOZPROCS);
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

^write_propagator{SPC}*={SPC}*             BEGIN(WRITEPROP);
^read_propagator{SPC}*={SPC}*              BEGIN(READPROP);

^read_sequential_propagator{SPC}*={SPC}*   BEGIN(READSEQPROP);
^write_sequential_source{SPC}*={SPC}*      BEGIN(WRITESEQSRC);
^write_sequential_propagator{SPC}*={SPC}*  BEGIN(WRITESEQPROP);

^samples{SPC}*={SPC}*                      BEGIN(NSAMPLE);
^samples_oet{SPC}*={SPC}*                  BEGIN(NSAMPLEOET);

^as_over_a{SPC}*={SPC}*                    BEGIN(ASOVERA);
^num_threads{SPC}*={SPC}*                  BEGIN(NUMTHREADS);
^source_momentum{SPC}*={SPC}*              BEGIN(SRCMOM);
^source_momentum_x{SPC}*={SPC}*            BEGIN(SRCMOMX);
^source_momentum_y{SPC}*={SPC}*            BEGIN(SRCMOMY);
^source_momentum_z{SPC}*={SPC}*            BEGIN(SRCMOMZ);
^sink_momentum{SPC}*={SPC}*                BEGIN(SNKMOM);
^sink_momentum_x{SPC}*={SPC}*              BEGIN(SNKMOMX);
^sink_momentum_y{SPC}*={SPC}*              BEGIN(SNKMOMY);
^sink_momentum_z{SPC}*={SPC}*              BEGIN(SNKMOMZ);
^seq_source_momentum_x{SPC}*={SPC}*        BEGIN(SEQSRCMOMX);
^seq_source_momentum_y{SPC}*={SPC}*        BEGIN(SEQSRCMOMY);
^seq_source_momentum_z{SPC}*={SPC}*        BEGIN(SEQSRCMOMZ);
^seq_source_momentum{SPC}*={SPC}*          BEGIN(SEQSRCMOM);
^seq2_source_momentum_x{SPC}*={SPC}*       BEGIN(SEQ2SRCMOMX);
^seq2_source_momentum_y{SPC}*={SPC}*       BEGIN(SEQ2SRCMOMY);
^seq2_source_momentum_z{SPC}*={SPC}*       BEGIN(SEQ2SRCMOMZ);
^seq2_source_momentum{SPC}*={SPC}*         BEGIN(SEQ2SRCMOM);
^total_momentum{SPC}*={SPC}*               BEGIN(TOTALMOMENTUM);
^cpu_precision{SPC}*={SPC}*                BEGIN(CPUPREC);
^gpu_precision{SPC}*={SPC}*                BEGIN(GPUPREC);
^gpu_precision_sloppy{SPC}*={SPC}*         BEGIN(GPUPRECSLOPPY);
^inverter_type{SPC}*={SPC}*                BEGIN(INVERTERTYPE);
^mms_id{SPC}*={SPC}*                       BEGIN(MMSID);
^check_inversion{SPC}*={SPC}*              BEGIN(CHKINVERSION);

^fermion_type{SPC}*={SPC}*                 BEGIN(FERMION_TYPE);
^src_snk_time_separation{SPC}*={SPC}*      BEGIN(SRCSNKTIMESEP);

^coherent_source_number{SPC}*={SPC}*       BEGIN(COHERENTSOURCENUMBER);

^verbosity_level{SPC}*={SPC}*              BEGIN(VERBOSITYLEVEL);

^gamma_f1_nucleon{SPC}*={SPC}*             BEGIN(GAMMA_F1_NUCLEON);
^gamma_f1_delta{SPC}*={SPC}*               BEGIN(GAMMA_F1_DELTA);
^gamma_f2{SPC}*={SPC}*                     BEGIN(GAMMA_F2);
^gamma_current{SPC}*={SPC}*                BEGIN(GAMMA_CURRENT);

^BeginTwopointFunction                     BEGIN(INITTWOPOINTFUNCTION);


  /* start namespace inclusion here */

<TT>{DIGIT}+                  {
  cvc::T_global = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::T_global =%s\n", yytext);
}
<LL>{DIGIT}+                  {
  cvc::L = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::L =%s\n", yytext);
}
<LL5>{DIGIT}+                  {
  cvc::L5 = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::L5 =%s\n", yytext);
}
<LLX>{DIGIT}+                  {
  cvc::LX = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::LX =%s\n", yytext);
}
<LLY>{DIGIT}+                  {
  cvc::LY = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::LY =%s\n", yytext);
}
<LLZ>{DIGIT}+                  {
  cvc::LZ = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::LZ =%s\n", yytext);
}
<KAPPA>{FLT}  {
  cvc::g_kappa=atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] kappa=%s \n", yytext);
}
<MU>{FLT}  {
  cvc::g_mu=atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] mu=%s \n", yytext);
}
<M5>{FLT}  {
  cvc::g_m5 = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] m5=%s \n", yytext);
}
<MUSIGMA>{FLT}  {
  cvc::g_musigma=atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] musigma=%s \n", yytext);
}
<MUDELTA>{FLT}  {
  cvc::g_mudelta=atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] mudelta=%s \n", yytext);
}
<MUBAR>{FLT}  {
  cvc::g_mubar=atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] mubar=%s \n", yytext);
}
<M0>{FLT}  {
  cvc::g_m0 = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] m0=%s \n", yytext);
}
<EPSBAR>{FLT}  {
  cvc::g_epsbar=atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] mu=%s \n", yytext);
}

<CSW>{FLT}  {
  cvc::g_csw = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] csw = %s \n", yytext);
}

<NNSAVE>{SDIGIT}+ {
  cvc::Nsave=atoi(yytext); 
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::Nsave= %s \n",yytext);
}
<NNCONF>{DIGIT}+ {
  cvc::Nconf=atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::Nconf=%s\n", yytext);
}
<BCT>{FLT} {
  cvc::BCangle[0]=atof(yytext);
  if(cvc::myverbose != 0) printf("# [read_input_parser] cvc::BCangle[0] for bound. in time set to %e\n", cvc::BCangle[0]);
}
<BCX>{FLT} {
  cvc::BCangle[1]=atof(yytext);
  if(cvc::myverbose != 0) printf("# [read_input_parser] cvc::BCangle[1] for bound. in x set to %e\n", cvc::BCangle[1]);
}
<BCY>{FLT} {
  cvc::BCangle[2]=atof(yytext);
  if(cvc::myverbose != 0) printf("# [read_input_parser] cvc::BCangle[2] for bound. in y set to %e\n", cvc::BCangle[2]);
}
<BCZ>{FLT} {
  cvc::BCangle[3]=atof(yytext);
  if(cvc::myverbose != 0) printf("# [read_input_parser] cvc::BCangle[3] for bound. in z set to %e\n", cvc::BCangle[3]);
}
<SOURCEID>{DIGIT}+ {
  cvc::g_sourceid = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] first source id set to %s\n", yytext);
}
<SOURCEID2>{DIGIT}+ {
  cvc::g_sourceid2 = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] second source id set to %s\n", yytext);
}
<SOURCEIDSTEP>{DIGIT}+ {
  cvc::g_sourceid_step = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] source id step set to %s\n", yytext);
}
<GAUGEID>{DIGIT}+ {
  cvc::g_gaugeid = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] first gauge id set to %s\n", yytext);
}
<GAUGEID2>{DIGIT}+ {
  cvc::g_gaugeid2 = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] second gauge id set to %s\n", yytext);
}
<GAUGESTEP>{DIGIT}+ {
  cvc::g_gauge_step = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] gauge step size set to %s\n", yytext);
}
<PREFIX>{FILENAME} {
  strcpy(cvc::filename_prefix,yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] prop. filename prefix set to %s\n",yytext);
}
<PREFIX2>{FILENAME} {
  strcpy(cvc::filename_prefix2,yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] prop. 2 filename prefix set to %s\n",yytext);
}
<PREFIX3>{FILENAME} {
  strcpy(cvc::filename_prefix3,yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] prop. 3 filename prefix set to %s\n",yytext);
}

<SEQPREFIX>{FILENAME} {
  strcpy(cvc::g_sequential_filename_prefix,yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] sequential prop. filename prefix set to %s\n",yytext);
}

<SEQPREFIX2>{FILENAME} {
  strcpy(cvc::g_sequential_filename_prefix2,yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] sequential prop. filename prefix 2 set to %s\n",yytext);
}

<GAUGEPREFIX>{FILENAME} {
  strcpy(cvc::gaugefilename_prefix,yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] gauge filename prefix set to %s\n",yytext);
}
<OUTPREFIX>{FILENAME} {
  strcpy(cvc::g_outfile_prefix,yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] out filename prefix set to %s\n",yytext);
}
<PATHPREFIX>{FILENAME} {
  strcpy(cvc::g_path_prefix,yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] path prefix set to %s\n",yytext);
}
<RESUME>{DIGIT}+ {
  cvc::g_resume=atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::g_resume set to %s\n",yytext);
}
<SUBTRACT>{DIGIT}+ {
  cvc::g_subtract=atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::g_subtract set to %s\n",yytext);
}
<SOURCELOC>{DIGIT}+ {
  cvc::g_source_location=atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] source location set to %s\n",yytext);
}
<FFORMAT>{DIGIT}+ {
  cvc::format = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::format set to %s\n",yytext);
}  
<SSEED>{DIGIT}+ {
  cvc::g_seed = (unsigned int)atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::g_seed set to %s\n",yytext);
}  
<NOISETYPE>{NAME} {
  if(strcmp(yytext, "Gaussian")==0 ) {
    cvc::g_noise_type = 1;
  } else if(strcmp(yytext, "Z2")==0 ) {
    cvc::g_noise_type = 2;
  } else if(strcmp(yytext, "Z3")==0 ) {
    cvc::g_noise_type = 3;
  } else if(strcmp(yytext, "B2")==0 ) {
    cvc::g_noise_type = 4;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] noise type set to %s\n",yytext);
}  
<SRCTYPE>{FILENAME} {
  if(strcmp(yytext, "Point")==0) {
    cvc::g_source_type = 0;
  } else if (strcmp(yytext, "Volume")==0 ) {
    cvc::g_source_type = 1;
  } else if (strcmp(yytext, "Timeslice")==0 ) {
    cvc::g_source_type = 2;
  } else if (strcmp(yytext, "TimesliceOneEnd")==0 ) {
    cvc::g_source_type = 3;
  } else if (strcmp(yytext, "TimesliceOneEndColor")==0 ) {
    cvc::g_source_type = 4;
  } else if (strcmp(yytext, "SequentialPoint")==0 ) {
    cvc::g_source_type = 5;
  } else if (strcmp(yytext, "Other")==0 ) {
    cvc::g_source_type = -1;
  } 
  if(cvc::myverbose!=0) printf("# [read_input_parser] source type set to %s\n",yytext);
}
<SPREC>{FLT} {
  cvc::solver_precision = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] solver precision set to %s\n",yytext);
}
<RELDELTA>{FLT} {
  cvc::reliable_delta = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] reliable delta set to %s\n",yytext);
}
<MAXITER>{DIGIT}+ {
  cvc::niter_max = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] max. number of iterations set to %s\n",yytext);
}
<HPEMIN>{DIGIT}+ {
  cvc::hpe_order_min = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] min. HPE order set to %s\n",yytext);
}
<HPEMAX>{DIGIT}+ {
  cvc::hpe_order_max = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] max. HPE order set to %s\n",yytext);
}
<HPEORD>{DIGIT}+ {
  cvc::hpe_order = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] HPE order set to %s\n",yytext);
}
<CUTANG>{FLT} {
  cvc::g_cutangle = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cone cut angle set to %s\n",yytext);
}
<CUTRAD>{FLT} {
  cvc::g_cutradius = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cone cut radius set to %s\n",yytext);
}
<CUTDIRT>{DIGIT}+ {
  cvc::g_cutdir[0] = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] t-comp. of cut dir angle set to %s\n",yytext);
}
<CUTDIRX>{DIGIT}+ {
  cvc::g_cutdir[1] = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] x-comp. of cut dir angle set to %s\n",yytext);
}
<CUTDIRY>{DIGIT}+ {
  cvc::g_cutdir[2] = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] y-comp. of cut dir angle set to %s\n",yytext);
}
<CUTDIRZ>{DIGIT}+ {
  cvc::g_cutdir[3] = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] z-comp. of cut dir angle set to %s\n",yytext);
}
<RMIN>{FLT} {
  cvc::g_rmin = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] minimal distance set to %s\n",yytext);
}
<RMAX>{FLT} {
  cvc::g_rmax = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] maximal distance set to %s\n",yytext);
}
<AVGTT>{DIGIT}+                  {
  cvc::avgT = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::avgT =%s\n", yytext);
}
<AVGLL>{DIGIT}+                  {
  cvc::avgL = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::avgL =%s\n", yytext);
}
<MODEL_DCOEFFRE>{FLT} {
  cvc::model_dcoeff_re = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] real part of d-coeff. set to %s\n", yytext);
}
<MODEL_DCOEFFIM>{FLT} {
  cvc::model_dcoeff_im = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] imaginary part of d-coeff. set to %s\n", yytext);
}
<MODEL_MRHO>{FLT} {
  cvc::model_mrho = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] rho-meson mass set to %s\n", yytext);
}
<FTRMAX1>{FLT} {
  cvc::ft_rmax[0] = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] 1st rmax for FT set to %s\n", yytext);
}
<FTRMAX2>{FLT} {
  cvc::ft_rmax[1] = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] 2nd rmax for FT set to %s\n", yytext);
}
<FTRMAX3>{FLT} {
  cvc::ft_rmax[2] = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] 3rd rmax for FT set to %s\n", yytext);
}
<FTRMAX4>{FLT} {
  cvc::ft_rmax[3] = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] 4th rmax for FT set to %s\n", yytext);
}
<PROPNORMSQR>{FLT} {
  cvc::g_prop_normsqr = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] propagator norm squared set to %s\n", yytext);
}
<QHATSQRMIN>{FLT} {
  cvc::g_qhatsqr_min = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set minimal squared Qhat to %s\n", yytext);
}
<QHATSQRMAX>{FLT} {
  cvc::g_qhatsqr_max = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set maximal squared Qhat to %s\n", yytext);
}
<NLONG>{DIGIT}+ {
  cvc::Nlong = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::Nlong to %s\n", yytext);
}
<NAPE>{DIGIT}+ {
  cvc::N_ape = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::N_ape to %s\n", yytext);
}
<NHYP>{DIGIT}+ {
  cvc::N_hyp = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::N_hyp to %s\n", yytext);
}
<NJACOBI>{DIGIT}+ {
  cvc::N_Jacobi = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::N_Jacobi to %s\n", yytext);
}
<ALPHAAPE>{FLT} {
  cvc::alpha_ape = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::alpha_ape to %s\n", yytext);
}
<ALPHAHYP>{FILENAME} {
  sscanf(yytext, "%lf/%lf/%lf", cvc::alpha_hyp, cvc::alpha_hyp+1, cvc::alpha_hyp+2);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set alpha_hyp to %f, %f, %f\n", cvc::alpha_hyp[0], cvc::alpha_hyp[1], cvc::alpha_hyp[2]);
  /* if(cvc::myverbose!=0) printf("# [read_input_parser] yytext is %s\n", yytext); */
}
<KAPPAJACOBI>{FLT} {
  cvc::kappa_Jacobi = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::kappa_Jacobi to %s\n", yytext);
}
<SRCTIMESLICE>{DIGIT}+ {
  cvc::g_source_timeslice = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set source timeslice to %s\n", yytext);
}
<SEQSRCTIMESLICE>{SDIGIT}+ {
  if( cvc::g_sequential_source_timeslice_number == 0) {
    cvc::g_sequential_source_timeslice = atoi(yytext);
  }
  cvc::g_sequential_source_timeslice_list[ cvc::g_sequential_source_timeslice_number ] = atoi(yytext);
  cvc::g_sequential_source_timeslice_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] added sequential source timeslice  %s; total number is now %d\n", yytext, cvc::g_sequential_source_timeslice_number);
}

<SEQSRCLOCATIONX>{SDIGIT}+ {
  cvc::g_sequential_source_location_x = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set sequential source location x to %s\n", yytext);
}
<SEQSRCLOCATIONY>{SDIGIT}+ {
  cvc::g_sequential_source_location_y = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set sequential source location y to %s\n", yytext);
}
<SEQSRCLOCATIONZ>{SDIGIT}+ {
  cvc::g_sequential_source_location_z = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set sequential source location z to %s\n", yytext);
}

<SEQSRCGAMMAID>{DIGIT}+ {
  cvc::g_sequential_source_gamma_id_list[cvc::g_sequential_source_gamma_id_number] = atoi(yytext);
  if( cvc::g_sequential_source_gamma_id_number == 0 ) {
    cvc::g_sequential_source_gamma_id = cvc::g_sequential_source_gamma_id_list[cvc::g_sequential_source_gamma_id_number];
  }
  cvc::g_sequential_source_gamma_id_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] set new entry in sequential source gamma id list to %s, total number is now %d\n", yytext, cvc::g_sequential_source_gamma_id_number);
}

<SRCGAMMAID>{DIGIT}+ {
  cvc::g_source_gamma_id_list[cvc::g_source_gamma_id_number] = atoi(yytext);
  if( cvc::g_source_gamma_id_number == 0 ) {
    cvc::g_source_gamma_id = cvc::g_source_gamma_id_list[cvc::g_source_gamma_id_number];
  }
  cvc::g_source_gamma_id_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] set new entry in source gamma id list to %s, total number is now %d\n", yytext, cvc::g_source_gamma_id_number);
}


<MMSNOMASSES>{DIGIT}+ {
  cvc::g_no_extra_masses = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set no. of extra masses for mms to %s\n", yytext);
}
<NOLIGHTMASSES>{DIGIT}+ {
  cvc::g_no_light_masses = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set no. of light masses for mms to %s\n", yytext);
}
<NOSTRANGEMASSES>{DIGIT}+ {
  cvc::g_no_strange_masses = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set no. of strange masses for mms to %s\n", yytext);
}
<NOTPROCS>{DIGIT}+ {
  cvc::g_nproc_t = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::g_nproc_t to %s\n", yytext);
}
<NOXPROCS>{DIGIT}+ {
  cvc::g_nproc_x = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::g_nproc_x to %s\n", yytext);
}
<NOYPROCS>{DIGIT}+ {
  cvc::g_nproc_y = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::g_nproc_y to %s\n", yytext);
}
<NOZPROCS>{DIGIT}+ {
  cvc::g_nproc_z = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set cvc::g_nproc_z to %s\n", yytext);
}

<LOCLOC>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    cvc::g_local_local = 1;
    if(cvc::myverbose!=0) printf("# [read_input_parser] do local-local contraction\n");
  }
}
<LOCSME>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    cvc::g_local_smeared = 1;
    if(cvc::myverbose!=0) printf("# [read_input_parser] do local-smeared contraction\n");
  }
}
<SMELOC>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    cvc::g_smeared_local = 1;
    if(cvc::myverbose!=0) printf("# [read_input_parser] do smeared-local contraction\n");
  }
}
<SMESME>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    cvc::g_smeared_smeared = 1;
    if(cvc::myverbose!=0) printf("# [read_input_parser] do smeared-smeared contraction\n");
  }
}
<ROTATE>{FILENAME} {
  if(strcmp(yytext, "yes")==0) {
    cvc::g_rotate_ETMC_UKQCD = 1;
    if(cvc::myverbose!=0) printf("# [read_input_parser] rotate propagators UKQCD -> ETMC\n");
  }
}
<PROPPOS>{DIGIT}+ {
  cvc::g_propagator_position = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set propagator position to %s\n", yytext);
}

<GPUDEVICENUMBER>{DIGIT}+ {
  cvc::g_gpu_device_number = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set GPU device number to %s\n", yytext);
}
<GPUPERNODE>{DIGIT}+ {
  cvc::g_gpu_per_node = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set number of GPUs  per node to %s\n", yytext);
}

<COHSRC>{DIGIT}+ {
  cvc::g_coherent_source = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set coherent source to %s\n", yytext);
}

<COHSRCBASE>{DIGIT}+ {
  cvc::g_coherent_source_base = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set coherent source base to %s\n", yytext);
}

<COHSRCDELTA>{DIGIT}+ {
  cvc::g_coherent_source_delta = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set coherent source delta to %s\n", yytext);
}
<GAUGEFILEFORMAT>{NAME} {
  if(strcmp(yytext, "ildg")==0) {
    cvc::g_gauge_file_format = 0;
  } else if(strcmp(yytext, "nersc")==0) {
    cvc::g_gauge_file_format = 1;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set gauge file cvc::format to %s\n", yytext);
}
<RNGFILE>{FILENAME} {
//  cvc::g_rng_filename = (char*)malloc((strlen(yytext)+1)*sizeof(char));
  strcpy(cvc::g_rng_filename, yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::g_rng_filename set to %s\n", cvc::g_rng_filename);
}
<SRCINDEX>{DIGIT}+ {
  cvc::g_source_index[0] = atoi(yytext);
  cvc::g_source_index[1] = cvc::g_source_index[0];
  if(cvc::myverbose!=0) printf("# [read_input_parser] start index set to %d\n", cvc::g_source_index[0]);
}
<SRCINDEX>{IDXEX} {
  sscanf(yytext, "-%d", cvc::g_source_index+1);
  if(cvc::myverbose!=0) printf("# [read_input_parser] end index set to %d\n", cvc::g_source_index[1]);
}
<PROPBCTYPE>{NAME} {
  if(strcmp(yytext, "Phase")==0) {
    cvc::g_propagator_bc_type = 0;
  } else if(strcmp(yytext, "Sign")==0) {
    cvc::g_propagator_bc_type = 1;
  } else {
    cvc::g_propagator_bc_type = -1;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] propagator boundary condition type set to %d\n", cvc::g_propagator_bc_type);
}
<WRITESRC>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    cvc::g_write_source = 1;
  } else if(strcmp(yytext,"no")==0) {
    cvc::g_write_source = 0;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set write source to %d\n", cvc::g_write_source);  
}
<READSRC>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    cvc::g_read_source = 1;
  } else if(strcmp(yytext,"no")==0) {
    cvc::g_read_source = 0;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set write source to %d\n", cvc::g_write_source);  
}
<WRITEPROP>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    cvc::g_write_propagator = 1;
  } else if(strcmp(yytext,"no")==0) {
    cvc::g_write_propagator = 0;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set write propagator to %d\n", cvc::g_write_propagator);  
}
<READPROP>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    cvc::g_read_propagator = 1;
  } else if(strcmp(yytext,"no")==0) {
    cvc::g_read_propagator = 0;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set read propagator to %d\n", cvc::g_read_propagator);  
}
<READSEQPROP>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    cvc::g_read_sequential_propagator = 1;
  } else if(strcmp(yytext,"no")==0) {
    cvc::g_read_sequential_propagator = 0;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set read sequential propagator to %d\n", cvc::g_read_sequential_propagator);  
}

<WRITESEQSRC>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    cvc::g_write_sequential_source = 1;
  } else if(strcmp(yytext,"no")==0) {
    cvc::g_write_sequential_source = 0;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set write sequential source  to %d\n", cvc::g_write_sequential_source);
}

<WRITESEQPROP>{NAME} {
  if(strcmp(yytext,"yes")==0) {
    cvc::g_write_sequential_propagator = 1;
  } else if(strcmp(yytext,"no")==0) {
    cvc::g_write_sequential_propagator = 0;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set write sequential propagator  to %d\n", cvc::g_write_sequential_propagator);
}

<PROPGAMMA>{NAME} {
  if(strcmp(yytext,"DeGrandRossi")==0) {
    cvc::g_propagator_gamma_basis = 1;
  } else if(strcmp(yytext,"UKQCD")==0) {
    cvc::g_propagator_gamma_basis = 2;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] set propagator gamma basis type to %d\n", cvc::g_propagator_gamma_basis);
}
<PROPPREC>{DIGIT}+ {
  cvc::g_propagator_precision = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set propagator precision to %d\n", cvc::g_propagator_precision);
}

<NSAMPLE>{DIGIT}+ {
  cvc::g_nsample = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set number of samples to %d\n", cvc::g_nsample);
}

<NSAMPLEOET>{DIGIT}+ {
  cvc::g_nsample_oet = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set number of oet samples to %d\n", cvc::g_nsample_oet);
}

<ASOVERA>{FLT} {
  cvc::g_as_over_a = atof(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] as / a set to  %s\n",yytext);
}
<NUMTHREADS>{DIGIT}+ {
  cvc::g_num_threads = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] number of threads =%d\n", cvc::g_num_threads);
}

<SRCMOMX>{SDIGIT}+ {
  cvc::g_source_momentum[0] = atoi(yytext);
  cvc::g_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] x-comp. of source momentum set to %d\n", cvc::g_source_momentum[0]);
}
<SRCMOMY>{SDIGIT}+ {
  cvc::g_source_momentum[1] = atoi(yytext);
  cvc::g_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] y-comp. of source momentum set to %d\n", cvc::g_source_momentum[1]);
}
<SRCMOMZ>{SDIGIT}+ {
  cvc::g_source_momentum[2] = atoi(yytext);
  cvc::g_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] z-comp. of source momentum set to %d\n", cvc::g_source_momentum[2]);
}

<SRCMOM>{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
  sscanf(yytext, "%d,%d,%d", cvc::g_source_momentum_list[cvc::g_source_momentum_number],
      cvc::g_source_momentum_list[cvc::g_source_momentum_number]+1, cvc::g_source_momentum_list[cvc::g_source_momentum_number]+2);
  cvc::g_source_momentum_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] added source momentum %s to list; total number is now %d\n", yytext, cvc::g_source_momentum_number);
}

<SNKMOMX>{SDIGIT}+ {
  cvc::g_sink_momentum[0] = atoi(yytext);
  cvc::g_sink_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] x-comp. of sink momentum set to %d\n", cvc::g_sink_momentum[0]);
}
<SNKMOMY>{SDIGIT}+ {
  cvc::g_sink_momentum[1] = atoi(yytext);
  cvc::g_sink_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] y-comp. of sink momentum set to %d\n", cvc::g_sink_momentum[1]);
}
<SNKMOMZ>{SDIGIT}+ {
  cvc::g_sink_momentum[2] = atoi(yytext);
  cvc::g_sink_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] z-comp. of sink momentum set to %d\n", cvc::g_sink_momentum[2]);
}
<SNKMOM>{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
  sscanf(yytext, "%d,%d,%d", cvc::g_sink_momentum_list[cvc::g_sink_momentum_number],
      cvc::g_sink_momentum_list[cvc::g_sink_momentum_number]+1, cvc::g_sink_momentum_list[cvc::g_sink_momentum_number]+2);
  cvc::g_sink_momentum_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] added sink momentum %s to list; total number is now %d\n", yytext, cvc::g_sink_momentum_number);
}

<SEQSRCMOMX>{SDIGIT}+ {
  cvc::g_seq_source_momentum[0] = atoi(yytext);
  cvc::g_seq_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] x-comp. of sequential source momentum set to %d\n", cvc::g_seq_source_momentum[0]);
}
<SEQSRCMOMY>{SDIGIT}+ {
  cvc::g_seq_source_momentum[1] = atoi(yytext);
  cvc::g_seq_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] y-comp. of sequential source momentum set to %d\n", cvc::g_seq_source_momentum[1]);
}
<SEQSRCMOMZ>{SDIGIT}+ {
  cvc::g_seq_source_momentum[2] = atoi(yytext);
  cvc::g_seq_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] z-comp. of sequential source momentum set to %d\n", cvc::g_seq_source_momentum[2]);
}

<SEQSRCMOM>{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
  sscanf(yytext, "%d,%d,%d", cvc::g_seq_source_momentum_list[cvc::g_seq_source_momentum_number],
      cvc::g_seq_source_momentum_list[cvc::g_seq_source_momentum_number]+1, cvc::g_seq_source_momentum_list[cvc::g_seq_source_momentum_number]+2);
  cvc::g_seq_source_momentum_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] added sequential source momentum %s to list; total number is now %d\n", yytext, cvc::g_seq_source_momentum_number);
}

<SEQ2SRCMOMX>{SDIGIT}+ {
  cvc::g_seq2_source_momentum[0] = atoi(yytext);
  cvc::g_seq2_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] x-comp. of sequential^2 source momentum set to %d\n", cvc::g_seq2_source_momentum[0]);
}
<SEQ2SRCMOMY>{SDIGIT}+ {
  cvc::g_seq2_source_momentum[1] = atoi(yytext);
  cvc::g_seq2_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] y-comp. of sequential^2 source momentum set to %d\n", cvc::g_seq2_source_momentum[1]);
}
<SEQ2SRCMOMZ>{SDIGIT}+ {
  cvc::g_seq2_source_momentum[2] = atoi(yytext);
  cvc::g_seq2_source_momentum_set = 1;
  if(cvc::myverbose!=0) printf("# [read_input_parser] z-comp. of sequential^2 source momentum set to %d\n", cvc::g_seq2_source_momentum[2]);
}

<SEQ2SRCMOM>{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
  sscanf(yytext, "%d,%d,%d", cvc::g_seq2_source_momentum_list[cvc::g_seq2_source_momentum_number],
      cvc::g_seq2_source_momentum_list[cvc::g_seq2_source_momentum_number]+1, cvc::g_seq2_source_momentum_list[cvc::g_seq2_source_momentum_number]+2);
  cvc::g_seq2_source_momentum_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] added sequential^2 source momentum %s to list; total number is now %d\n", yytext, cvc::g_seq2_source_momentum_number);
}


<TOTALMOMENTUM>{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
  sscanf(yytext, "%d,%d,%d", cvc::g_total_momentum_list[cvc::g_total_momentum_number],
      cvc::g_total_momentum_list[cvc::g_total_momentum_number]+1, cvc::g_total_momentum_list[cvc::g_total_momentum_number]+2);
  cvc::g_total_momentum_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] added total momentum %s to list; total number is now %d\n", yytext, cvc::g_total_momentum_number);
}

<CPUPREC>{NAME} {
  if(strcmp(yytext, "half")==0 ) {
    cvc::g_cpu_prec = 0;
  } else if(strcmp(yytext, "single")==0 ) {
    cvc::g_cpu_prec = 1;
  } else if(strcmp(yytext, "double")==0 ) {
    cvc::g_cpu_prec = 2;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] CPU precision (for QUDA library) set to %s\n",yytext);
}
<GPUPREC>{NAME} {
  if(strcmp(yytext, "half")==0 ) {
    cvc::g_gpu_prec = 0;
  } else if(strcmp(yytext, "single")==0 ) {
    cvc::g_gpu_prec = 1;
  } else if(strcmp(yytext, "double")==0 ) {
    cvc::g_gpu_prec = 2;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] GPU precision (for QUDA library) set to %s\n",yytext);
}
<GPUPRECSLOPPY>{NAME} {
  if(strcmp(yytext, "half")==0 ) {
    cvc::g_gpu_prec_sloppy = 0;
  } else if(strcmp(yytext, "single")==0 ) {
    cvc::g_gpu_prec_sloppy = 1;
  } else if(strcmp(yytext, "double")==0 ) {
    cvc::g_gpu_prec_sloppy = 2;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] GPU sloppy precision (for QUDA library) set to %s\n",yytext);
}
<INVERTERTYPE>{NAME} {
  strcpy(cvc::g_inverter_type_name, yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] inverter type name set to %s\n",yytext);
}
<MMSID>{DIGIT}+ {
  cvc::g_mms_id = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] set mms id to %s\n", yytext);
}
<CHKINVERSION>{NAME} {
  if(strcmp(yytext, "yes")==0) {
    cvc::g_check_inversion = 1;
  } else if(strcmp(yytext, "no")==0)  {
    cvc::g_check_inversion = 0;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] check inversion %d\n",cvc::g_check_inversion);
}

<SOURCECOORDS>{CORRCONFIG} {

  if ( cvc::g_source_location_number == MAX_SOURCE_LOCATION_NUMBER ) {
    fprintf(stderr, "[read_input_parser] Error, too many source locations\n");
    exit(1);
  } else {
    if(cvc::myverbose) fprintf(stdout, "# [read_input_parser] extracting source coords %d\n", cvc::g_source_location_number);
  }

  sscanf(yytext, "%d,%d,%d,%d",
    cvc::g_source_coords_list[cvc::g_source_location_number],
    cvc::g_source_coords_list[cvc::g_source_location_number]+1,
    cvc::g_source_coords_list[cvc::g_source_location_number]+2,
    cvc::g_source_coords_list[cvc::g_source_location_number]+3);

  cvc::g_source_location_number++;
  if(cvc::myverbose!=0) printf("# [read_input_parser] source coords no. %d set to %s\n", cvc::g_source_location_number, yytext);
}

<INITTWOPOINTFUNCTION>Generic{SPC}* {
  if(cvc::myverbose) printf("# [read_input_parser] Initialization generic twopoint_function line %d\n", cvc::line_of_file);
  cvc::twopoint_function_ptr = &(cvc::twopoint_function_generic);
  cvc::twopoint_function_init ( cvc::twopoint_function_ptr );
  strcpy(cvc::twopoint_function_set_type, "generic");
  BEGIN(TWOPOINTFUNCTION);
}

<INITTWOPOINTFUNCTION>Init{SPC}* {
  if(cvc::g_twopoint_function_number==MAX_TWOPOINT_FUNCTION_NUM) {
    fprintf(stderr, "[read_input_parser] Error, too many twopoint_function\n");
    exit(1);
  }
  if(cvc::myverbose) printf("# [read_input_parser] Initialization twopoint_function line %d\n", cvc::line_of_file);
  cvc::twopoint_function_ptr = &(cvc::g_twopoint_function_list[cvc::g_twopoint_function_number]);
  cvc::twopoint_function_init ( cvc::twopoint_function_ptr );
  strcpy(cvc::twopoint_function_set_type, "list");
  if(cvc::twopoint_function_generic_set==1) {
    cvc::twopoint_function_copy ( cvc::twopoint_function_ptr, &(cvc::twopoint_function_generic), 1 );
    if(cvc::myverbose) printf("# [read_input_parser] line %d copy from generic \n", cvc::line_of_file);
  }
  BEGIN(TWOPOINTFUNCTION);
}

<TWOPOINTFUNCTION>{
  {SPC}*pi1{EQL}{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] =  %d,%d,%d", cvc::name,
        cvc::twopoint_function_ptr->pi1,
        cvc::twopoint_function_ptr->pi1+1,
        cvc::twopoint_function_ptr->pi1+2 );
  }
  {SPC}*pi2{EQL}{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d,%d,%d", cvc::name,
        cvc::twopoint_function_ptr->pi2,
        cvc::twopoint_function_ptr->pi2+1,
        cvc::twopoint_function_ptr->pi2+2 );
  }
  {SPC}*pf1{EQL}{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d,%d,%d", cvc::name,
        cvc::twopoint_function_ptr->pf1,
        cvc::twopoint_function_ptr->pf1+1,
        cvc::twopoint_function_ptr->pf1+2 );
  }
  {SPC}*pf2{EQL}{SDIGIT}+,{SDIGIT}+,{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d,%d,%d", cvc::name,
        cvc::twopoint_function_ptr->pf2,
        cvc::twopoint_function_ptr->pf2+1,
        cvc::twopoint_function_ptr->pf2+2 );
  }

  {SPC}*gi1{EQL}{SDIGIT}+,{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d,%d", cvc::name,
        cvc::twopoint_function_ptr->gi1,
        cvc::twopoint_function_ptr->gi1+1);
  }

  {SPC}*gf1{EQL}{SDIGIT}+,{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d,%d", cvc::name,
        cvc::twopoint_function_ptr->gf1,
        cvc::twopoint_function_ptr->gf1+1);
  }

  {SPC}*source_coords{EQL}{DIGIT}+,{DIGIT}+,{DIGIT}+,{DIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9_] = %d,%d,%d,%d", cvc::name,
        &(cvc::twopoint_function_ptr->source_coords[0]),
        &(cvc::twopoint_function_ptr->source_coords[1]),
        &(cvc::twopoint_function_ptr->source_coords[2]),
        &(cvc::twopoint_function_ptr->source_coords[3]) );
  }

  {SPC}*gi2{EQL}{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d", cvc::name,
        &(cvc::twopoint_function_ptr->gi2) );
  }

  {SPC}*gf2{EQL}{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d", cvc::name,
        &(cvc::twopoint_function_ptr->gf2) );
  }

  {SPC}*n{EQL}{DIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d", cvc::name,
        &(cvc::twopoint_function_ptr->n) );
  }

  {SPC}*tag{EQL}{STRING} {
    sscanf(yytext, " %[a-zA-Z0-9] = %s", cvc::name, cvc::fname );
    strcpy( cvc::twopoint_function_ptr->name , cvc::fname );
  }

  {SPC}*type{EQL}{STRING} {
    sscanf(yytext, " %[a-zA-Z0-9] = %s", cvc::name, cvc::fname );
    strcpy(cvc::twopoint_function_ptr->type, cvc::fname );
  }

  {SPC}*spin_project{EQL}{DIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9_] = %d", cvc::name,
        &(cvc::twopoint_function_ptr->spin_project) );
  }

  {SPC}*T{EQL}{DIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9_] = %d", cvc::name,
        &(cvc::twopoint_function_ptr->T) );
  }

  {SPC}*d{EQL}{DIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9_] = %d", cvc::name,
        &(cvc::twopoint_function_ptr->d) );
  }

  {SPC}*parity_project{EQL}{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9_] = %d", cvc::name,
        &(cvc::twopoint_function_ptr->parity_project) );
  }

  {SPC}*diagrams{EQL}{STRING} {
    sscanf(yytext, " %[a-zA-Z0-9] = %s", cvc::name, cvc::twopoint_function_ptr->diagrams );
  }

  {SPC}*reorder{EQL}{SDIGIT}+ {
    sscanf(yytext, " %[a-zA-Z0-9] = %d", cvc::name,
        &(cvc::twopoint_function_ptr->reorder) );
  }

  {SPC}*norm{EQL}{STRING} {
    sscanf(yytext, " %[a-zA-Z0-9] = %s", cvc::name, cvc::twopoint_function_ptr->norm );
  }

  {SPC}*fbwd{EQL}{STRING} {
    sscanf(yytext, " %[a-zA-Z0-9] = %s", cvc::name, cvc::twopoint_function_ptr->fbwd );
  }

  {SPC}*group{EQL}{STRING} {
    sscanf(yytext, " %[a-zA-Z0-9] = %s", cvc::name, cvc::twopoint_function_ptr->group );
  }

  {SPC}*irrep{EQL}{STRING} {
    sscanf(yytext, " %[a-zA-Z0-9] = %s", cvc::name, cvc::twopoint_function_ptr->irrep );
  }

  {SPC}*sf1{EQL}{FLT},{FLT},{FLT},{FLT} {
    double dtmp[4];
    sscanf(yytext, " %[a-zA-Z0-9] = %lf,%lf,%lf,%lf", cvc::name, dtmp, dtmp+1, dtmp+2, dtmp+3 );
    
    cvc::twopoint_function_ptr->sf1[0] = dtmp[0] + dtmp[1] * I;
    cvc::twopoint_function_ptr->sf1[1] = dtmp[2] + dtmp[3] * I;
    if ( cvc::myverbose ) printf ( "# [read_input_parser] sf1 set to %16.7e I %16.7e and %16.7e I %16.7e\n",
        creal ( cvc::twopoint_function_ptr->sf1[0] ), cimag ( cvc::twopoint_function_ptr->sf1[0] ),
        creal ( cvc::twopoint_function_ptr->sf1[1] ), cimag ( cvc::twopoint_function_ptr->sf1[1] ) );
  }

  {SPC}*si1{EQL}{FLT},{FLT},{FLT},{FLT} {
    double dtmp[4];
    sscanf(yytext, " %[a-zA-Z0-9] = %lf,%lf,%lf,%lf", cvc::name, dtmp, dtmp+1, dtmp+2, dtmp+3 );
    
    cvc::twopoint_function_ptr->si1[0] = dtmp[0] + dtmp[1] * I;
    cvc::twopoint_function_ptr->si1[1] = dtmp[2] + dtmp[3] * I;
    if ( cvc::myverbose ) printf ( "# [read_input_parser] si1 set to %16.7e I %16.7e and %16.7e I %16.7e\n",
        creal ( cvc::twopoint_function_ptr->si1[0] ), cimag ( cvc::twopoint_function_ptr->si1[0] ),
        creal ( cvc::twopoint_function_ptr->si1[1] ), cimag ( cvc::twopoint_function_ptr->si1[1] ) );
  }

  {SPC}*si2{EQL}{FLT},{FLT},{FLT},{FLT} {
    double dtmp[2];
    sscanf(yytext, " %[a-zA-Z0-9] = %lf,%lf", cvc::name, dtmp, dtmp+1 );
    
    cvc::twopoint_function_ptr->si2 = dtmp[0] + dtmp[1] * I;
    if ( cvc::myverbose ) printf ( "# [read_input_parser] si2 set to %16.7e I %16.7e\n",
        creal ( cvc::twopoint_function_ptr->si2 ), cimag ( cvc::twopoint_function_ptr->si2 ) );
  }

  {SPC}*sf2{EQL}{FLT},{FLT},{FLT},{FLT} {
    double dtmp[2];
    sscanf(yytext, " %[a-zA-Z0-9] = %lf,%lf", cvc::name, dtmp, dtmp+1 );
    
    cvc::twopoint_function_ptr->sf2 = dtmp[0] + dtmp[1] * I;
    if ( cvc::myverbose ) printf ( "# [read_input_parser] sf2 set to %16.7e I %16.7e\n",
        creal ( cvc::twopoint_function_ptr->sf2 ), cimag ( cvc::twopoint_function_ptr->sf2 ) );
  }

  EndTwopointFunction{SPC}* {
    if(strcmp(cvc::twopoint_function_set_type, "list")==0 ) {
      if( cvc::myverbose ) printf("# [read_input_parser] 2-point function no. %d parsed in line %d\n\n", cvc::g_twopoint_function_number, cvc::line_of_file);
      cvc::g_twopoint_function_number++;
    } else if (strcmp(cvc::twopoint_function_set_type,"generic")==0) {
      cvc::twopoint_function_generic_set = 1;
      if( cvc::myverbose ) printf("# [read_input_parser] generic 2-point function parsed in line %d\n\n", cvc::line_of_file);
    }
    BEGIN(0);
  }
}

<GAMMA_F1_NUCLEON>{STRING} {
  if(cvc::g_gamma_f1_nucleon_number==MAX_GAMMA_NUMBER) {
    fprintf(stderr, "Error, too many f1 nucleon gamma structures\n");
    exit(1);
  }
  strcpy(cvc::g_gamma_f1_nucleon_list[cvc::g_gamma_f1_nucleon_number], yytext);
  cvc::g_gamma_f1_nucleon_number++;
}

<GAMMA_F1_DELTA>{STRING} {
  if(cvc::g_gamma_f1_delta_number==MAX_GAMMA_NUMBER) {
    fprintf(stderr, "Error, too many f1 delta gamma structures\n");
    exit(1);
  }
  strcpy(cvc::g_gamma_f1_delta_list[cvc::g_gamma_f1_delta_number], yytext );
  cvc::g_gamma_f1_delta_number++;
}
<GAMMA_F2>{STRING} {
  if(cvc::g_gamma_f2_number==MAX_GAMMA_NUMBER) {
    fprintf(stderr, "Error, too many f2 gamma structures\n");
    exit(1);
  }
  strcpy(cvc::g_gamma_f2_list[cvc::g_gamma_f2_number], yytext );
  cvc::g_gamma_f2_number++;
}
<GAMMA_CURRENT>{STRING} {
  if(cvc::g_gamma_current_number==MAX_GAMMA_NUMBER) {
    fprintf(stderr, "Error, too many current gamma structures\n");
    exit(1);
  }
  strcpy( cvc::g_gamma_current_list[cvc::g_gamma_current_number], yytext );
  cvc::g_gamma_current_number++;
}


<FERMION_TYPE>{FILENAME} {
  if(strcmp(yytext, "TwistedMass")==0) {
    cvc::g_fermion_type = _TM_FERMION;
  } else if (strcmp(yytext, "Wilson")==0 ) {
    cvc::g_fermion_type = _WILSON_FERMION;
  } else if (strcmp(yytext, "DomainWall")==0 ) {
    cvc::g_fermion_type = _DW_WILSON_FERMION;
  }
  if(cvc::myverbose!=0) printf("# [read_input_parser] fermion type set to %d\n", cvc::g_fermion_type);
}

<SRCSNKTIMESEP>{DIGIT}+ {
  cvc::g_src_snk_time_separation = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] cvc::g_src_snk_time_separation = %s\n", yytext);
}

<COHERENTSOURCENUMBER>{DIGIT}+                  {
  cvc::g_coherent_source_number = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] coherent source number set to =%d\n", cvc::g_coherent_source_number);
}

<VERBOSITYLEVEL>{DIGIT}+                  {
  cvc::g_verbose = atoi(yytext);
  if(cvc::myverbose!=0) printf("# [read_input_parser] global verbosity level set to =%d\n", cvc::g_verbose);
}


<*>^#   {
   cvc::comment_caller = YY_START;   
   BEGIN(COMMENT);
}
<*>{SPC}*#    {
   cvc::comment_caller = YY_START;
   BEGIN(COMMENT);
}
<COMMENT>[^\n]*             {
  BEGIN(cvc::comment_caller);
}

<INITTWOPOINTFUNCTION,TWOPOINTFUNCTION>{SPC}*\n   {
  cvc::line_of_file++;
}


<*>\n                       {
  cvc::line_of_file++;
  BEGIN(0);
}

<*>. {
  BEGIN(ERROR);
}
<ERROR>[^\t\n]*             {
  printf("# [read_input_parser] Parsing error in line %d\nAborting...!\n", cvc::line_of_file);
  exit(1);
}

  /* end namespace inclusion here */

%%

/*
 *  Dummy (but not dumb) routine - well, function
 */


int yywrap()
{
  return(1);
}

namespace cvc {
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

#ifdef HAVE_MPI
  MPI_Comm_rank( MPI_COMM_WORLD, &myprocid);
#endif

  /********************************************
   * Setting default values!
   ********************************************/
  reread = 0;
  set_default_input_values();

  if( (myprocid == 0) && (cvc_verbose>0) ) {
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

#ifdef HAVE_MPI
  MPI_Comm_rank( MPI_COMM_WORLD, &myprocid);
#endif

  if(  (cvc_verbose>0) && myprocid == 0) {
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
