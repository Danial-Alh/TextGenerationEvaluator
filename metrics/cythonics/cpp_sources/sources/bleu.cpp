#include <iostream>
#include <algorithm>
#include <utility>
#include <thread>
#include <stdexcept>
#include <limits>
#include "bleu.h"
#include "counter.cpp"
#include "nltk.cpp"

using namespace std;

BLEU_CPP::BLEU_CPP()
{
}

BLEU_CPP::~BLEU_CPP()
{
    delete[] ref_lens;

    for (int i = 0; i < this->number_of_refs; i++)
        delete references[i];
    delete[] this->references;

    for (int n = 0; n < this->max_n; n++)
    {
        for (int i = 0; i < this->number_of_refs; i++)
            delete this->references_ngrams[n][i];
        delete[] this->references_ngrams[n];
    }
    delete[] this->references_ngrams;

    for (int n = 0; n < this->max_n; n++)
    {
        for (int i = 0; i < this->number_of_refs; i++)
            delete this->references_counts[n][i];
        delete[] this->references_counts[n];
    }
    delete[] this->references_counts;

    for (int n = 0; n < this->max_n; n++)
        delete this->reference_max_counts[n];
    delete[] this->reference_max_counts;
}

BLEU_CPP::BLEU_CPP(vector<vector<string>> lines_of_tokens, float weights[],
                   int max_n, int smoothing_func, bool auto_reweight, BLEU_CPP *other_instance)
{
    this->n_cores = thread::hardware_concurrency();
    this->references = new vector<string> *[lines_of_tokens.size()];
    this->references_ngrams = new vector<string> **[max_n];
    for (int i = 0; i < max_n; i++)
        this->references_ngrams[i] = new vector<string> *[lines_of_tokens.size()];
    this->references_counts = new Counter **[max_n];
    for (int i = 0; i < max_n; i++)
        this->references_counts[i] = new Counter *[lines_of_tokens.size()];
    this->reference_max_counts = new CustomMap *[max_n];

    this->ref_lens = new int[lines_of_tokens.size()];
    this->weights = weights;
    this->max_n = max_n;
    this->auto_reweigh = auto_reweigh;
    this->smoothing_function = smoothing_func;
    this->number_of_refs = (int)lines_of_tokens.size();

    cout << "bleu" << max_n << " init!" << endl;

    for (int i = 0; i < this->number_of_refs; i++)
    {
        this->references[i] = new vector<string>(lines_of_tokens[i]);
        this->ref_lens[i] = lines_of_tokens[i].size();
    }
    for (int n = 0; n < this->max_n; n++)
    {
        if (other_instance == NULL || n >= other_instance->max_n)
            for (int i = 0; i < this->number_of_refs; i++)
                this->references_ngrams[n][i] = get_ngrams(this->references[i], n + 1);
        else
            for (int i = 0; i < this->number_of_refs; i++)
                this->references_ngrams[n][i] = new vector<string>(*(other_instance->references_ngrams[n][i]));
    }
    for (int n = 0; n < this->max_n; n++)
    {
        if (other_instance == NULL || n >= other_instance->max_n)
            for (int i = 0; i < this->number_of_refs; i++)
                this->references_counts[n][i] = new Counter(this->references_ngrams[n][i]);
        else
            for (int i = 0; i < this->number_of_refs; i++)
                this->references_counts[n][i] = new Counter(*other_instance->references_counts[n][i]);
    }
    for (int n = 0; n < max_n; n++)
    {
        if (other_instance == NULL || n >= other_instance->max_n)
        {
            this->reference_max_counts[n] = new CustomMap();
            this->get_max_counts(n);
        }
        else
            this->reference_max_counts[n] = new CustomMap(*(other_instance->reference_max_counts[n]));
    }
}

void BLEU_CPP::get_max_counts(int n)
{
    vector<string> ngram_keys = vector<string>();
    for (int i = 0; i < this->number_of_refs; i++)
        for (string &ng : *this->references_ngrams[n][i])
            if (find(ngram_keys.begin(), ngram_keys.end(), ng) == ngram_keys.end())
                ngram_keys.push_back(ng);
    cout << n + 1 << "grams: " << ngram_keys.size() << endl;

    int temp_max_counts[ngram_keys.size()];
#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < (int)ngram_keys.size(); i++)
        {
            string &ng = ngram_keys[i];
            int counts[number_of_refs];
            for (int j = 0; j < number_of_refs; j++)
                counts[j] = references_counts[n][j]->get(ng);
            int max_value = *max_element(counts, counts + number_of_refs);
            temp_max_counts[i] = max_value;
        }
    }
    for (int i = 0; i < (int)ngram_keys.size(); i++)
        (*reference_max_counts[n])[ngram_keys[i]] = temp_max_counts[i];
}

void BLEU_CPP::get_score(vector<vector<string>> hypotheses, double *results)
{
    if (results == NULL)
        throw invalid_argument("results ptr points to NULL!");
    cout << "calculating bleu" << max_n << " scores!" << endl;
#pragma omp parallel
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < (int)hypotheses.size(); i++)
        {
            vector<string> *hypothesis = &(hypotheses[i]);
            results[i] = corpus_bleu(number_of_refs, max_n,
                                     references,
                                     hypothesis,
                                     reference_max_counts,
                                     ref_lens,
                                     weights,
                                     smoothing_function,
                                     auto_reweigh);
        }
    }
}