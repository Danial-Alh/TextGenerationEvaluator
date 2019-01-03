#ifndef BLEU_CPP_H
#define BLEU_CPP_H
#include <string>
#include <vector>
#include "fraction.h"
#include "counter.h"
#include "custmap.h"

using namespace std;

class BLEU_CPP
{
public:
  ~BLEU_CPP();
  BLEU_CPP();
  BLEU_CPP(vector<vector<string>>, float[], int, int, bool, BLEU_CPP *);
  void get_score(vector<vector<string>>, double *);

private:
  vector<string> **references;
  vector<string> ***references_ngrams;
  Counter ***references_counts;
  CustomMap **reference_max_counts;
  float *weights;
  int smoothing_function;
  bool auto_reweigh;
  int max_n;
  int *ref_lens;
  int number_of_refs;
  int n_cores;

  void get_max_counts(int);
};

#endif