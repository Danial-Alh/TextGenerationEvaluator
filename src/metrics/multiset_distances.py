from collections import Counter
from functools import reduce

import numpy as np
from nltk.translate.bleu_score import ngrams
from torchtext.data import ReversibleField

from metrics.base_metric import BaseMetric


class metric_names:
    jaccard = "Jaccard"
    sorensen = "Sorensen"
    canberra = "Canberra"
    minkowski = "Minkowski"


class MultisetDistances(BaseMetric):

    def __init__(self, references, min_n=3, max_n=5, parser: ReversibleField = None, parse=True):
        super().__init__()
        # print('multiset distances init upto {}!'.format(max_n))
        assert max_n >= min_n
        assert min_n >= 1
        assert parser is not None

        self.max_n = max_n
        self.min_n = min_n
        self.parser = parser

        self.ref_ngram_counts = self._get_ngram_counts(references, parse=parse)
        print('LOG: MultisetDistances init done!')

    def _get_ngram_counts(self, sentences, parse):
        samples_size = len(sentences)
        if parse:
            sentences = self.parser.reverse(sentences)
        sentences = [self.parser.tokenize(r) for r in sentences]
        sentences_ngrams = \
            [
                list(map(
                    lambda x: list(ngrams(x, n + 1)) if len(x) >= n else [],
                    sentences
                ))
                for n in range(self.max_n)
            ]
        all_counters = [Counter([x for y in sentences_ngrams[n] for x in y])
                        for n in range(self.max_n)]
        for n_counter in all_counters:
            for run in n_counter.keys():
                n_counter[run] /= samples_size
        return all_counters

    def _get_union_intersection_ngrams_with_refernce_ngrams(self, sentences, parse):
        sample_ngram_counts = self._get_ngram_counts(sentences, parse)
        ngrams_intersection = [sample_ngram_counts[i] & self.ref_ngram_counts[i]
                               for i in range(self.max_n)]  # intersection:  min(c[x], d[x])
        ngrams_union = [sample_ngram_counts[i] | self.ref_ngram_counts[i]
                        for i in range(self.max_n)]  # union:  max(c[x], d[x])
        ngrams_abs_diff = [ngrams_union[i] - ngrams_intersection[i]
                           for i in range(self.max_n)]
        ngrams_added = [sample_ngram_counts[i] + self.ref_ngram_counts[i]
                        for i in range(self.max_n)]

        return ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added

    def _final_average(self, score_value):
        return np.power(reduce(lambda x, y: x * y, score_value), 1. / float(len(score_value)))

    def _jaccard(self, ngrams_intersection, ngrams_union):
        jaccard_value = [float(sum(ngrams_intersection[n].values())) / sum(ngrams_union[n].values()) for n in
                         range(self.max_n)]
        return jaccard_value

    def get_jaccard_score(self, sentences, parse):
        print('LOG: calculating MS-Jaccard!')
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = \
            self._get_union_intersection_ngrams_with_refernce_ngrams(sentences, parse)

        jaccard_value = self._jaccard(
            ngrams_intersection=ngrams_intersection, ngrams_union=ngrams_union)

        result = {n: self._final_average(jaccard_value[:n])
                  for n in range(self.min_n, self.max_n + 1)}
        print('LOG: done!')
        return result

    def _sorensen(self, ngrams_abs_diff, ngrams_added):
        sorensen_value = [float(sum(ngrams_abs_diff[n].values())) / sum(ngrams_added[n].values()) for n in
                          range(self.max_n)]
        return sorensen_value

    def get_sorensen_score(self, sentences, parse):
        print('Sorensen distances preprocess upto {}!'.format(self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = \
            self._get_union_intersection_ngrams_with_refernce_ngrams(sentences, parse)

        sorensen_value = self._sorensen(ngrams_abs_diff=ngrams_abs_diff, ngrams_added=ngrams_added)

        return {n: self._final_average(sorensen_value[:n]) for n in range(self.min_n, self.max_n + 1)}

    def _canberra(self, ngrams_abs_diff, ngrams_added):
        canberra_value = [np.sum([ngrams_abs_diff[n][key] / float(ngrams_added[n][key]) for key in ngrams_abs_diff[n]])
                          for n in range(self.max_n)]
        return canberra_value

    def get_canberra_score(self, sentences, parse):
        print('Canberra distances preprocess upto {}!'.format(self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added =\
            self._get_union_intersection_ngrams_with_refernce_ngrams(sentences)
        canberra_value = self._canberra(ngrams_abs_diff=ngrams_abs_diff, ngrams_added=ngrams_added)
        return {n: self._final_average(canberra_value[:n]) for n in range(self.min_n, self.max_n + 1)}

    def _minkowski(self, ngrams_abs_diff, p):
        minkowski_value = [np.power(np.sum(np.power(list(ngrams_abs_diff[n].values()), p)), 1. / p) for n in
                           range(self.max_n)]
        return minkowski_value

    def get_minkowski_score(self, sentences, p):
        print('Minkowski (p={}) distances preprocess upto {}!'.format(p, self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = \
            self._get_union_intersection_ngrams_with_refernce_ngrams(sentences, parse)

        minkowski_value = self._minkowski(ngrams_abs_diff=ngrams_abs_diff, p=p)

        return {n: self._final_average(minkowski_value[:n]) for n in range(self.min_n, self.max_n + 1)}

    def get_score(self, metric_name, sentences, parse, **kwargs):
        func_name = 'get_{}_score'.format(metric_name)
        func = getattr(self, func_name)
        return func(sentences, parse=parse, **kwargs)

    def get_all_score(self, sentences, max_mikowski_order=3, parse=True):
        print('multiset distances preprocess upto {}!'.format(self.max_n))
        ngrams_intersection, ngrams_union, ngrams_abs_diff, ngrams_added = \
            self._get_union_intersection_ngrams_with_refernce_ngrams(sentences, parse)

        temp_results = {}

        print('multiset distances evaluating upto {}!'.format(self.max_n))
        temp_results[metric_names.jaccard] = self._jaccard(
            ngrams_intersection=ngrams_intersection,
            ngrams_union=ngrams_union)
        temp_results[metric_names.sorensen] = self._sorensen(
            ngrams_abs_diff=ngrams_abs_diff,
            ngrams_added=ngrams_added)
        temp_results[metric_names.canberra] = self._canberra(
            ngrams_abs_diff=ngrams_abs_diff,
            ngrams_added=ngrams_added)

        for p in range(1, max_mikowski_order + 1):
            temp_results['p%d-%s' % (p, metric_names.minkowski)
                         ] = self._minkowski(ngrams_abs_diff=ngrams_abs_diff, p=p)

        result = {}
        for key in temp_results:
            for n in range(self.min_n, self.max_n + 1):
                result[key + '%d' % n] = self._final_average(temp_results[key][:n])
        return result


if __name__ == "__main__":
    # print(MultisetDistances([['saadi', 'saadi']], max_n=2).get_all_score([['saadi', 'ferdowsi']]))
    # print(MultisetDistances([['saadi', 'saadi']], max_n=2).get_all_score([['saadi', 'saadi']]))

    ref1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever',
            'heed', 'Party', 'commands']
    ref2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always',
            'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    ref3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions',
            'of', 'the', 'party']
    sen1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys',
            'the', 'commands', 'of', 'the', 'party']
    sen2 = ['he', 'read', 'the', 'book', 'because', 'he',
            'was', 'interested', 'in', 'world', 'history']

    references = [ref1, ref2, ref3]
    sentences = [sen1, sen2]

    msd = MultisetDistances(references=references)
    print(msd.get_jaccard_score(sentences=sentences))
