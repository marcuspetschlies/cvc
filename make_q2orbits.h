#ifndef _MAKE_Q2ORBITS_H
#  define _MAKE_Q2ORBITS_H

# define _sqr(_x) ((_x)*(_x))
# define _qrt(_x) (_sqr(_x)*_sqr(_x))
# define _hex(_x) (_qrt(_x)*_sqr(_x))
# define _oct(_x) (_qrt(_x)*_qrt(_x))

int make_qid_lists(int *q2id, int *qhat2id, double **q2list, double **qhat2list, int *q2count, int *qhat2count);
int make_q2orbits(int **q2_id, int ***q2_list, double **q2_val, int **q2_count, int *q2_nc, double **h3_val, int h3_nc);
int make_q4orbits(int ***q4_id, double ***q4_val, int ***q4_count, int **q4_nc,
  int **q2_list, int *q2_count, int q2_nc, double **h3_val);
int make_rid_list(int **rid, double **rlist, int *rcount, double Rmin, double Rmax);
#endif
