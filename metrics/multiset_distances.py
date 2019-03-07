from collections import Counter
from functools import reduce

import numpy as np

from utils.nltk_utils import get_ngrams


class MultisetDistances:
    def __init__(self, references, min_n=1, max_n=3, min_mikowski_order=1, max_mikowski_order=3):
        super().__init__()
        print('multiset distances init upto {}!'.format(max_n))
        self.references = references
        self.max_n = max_n
        self.min_n = min_n
        self.max_mikowski_order = max_mikowski_order
        self.min_mikowski_order = min_mikowski_order
        assert self.max_n >= self.min_n
        assert self.min_n >= 1
        assert self.min_mikowski_order >= 1
        self.ref_ngrams = self._get_ngrams(references)

    def get_cached_fields(self):
        return self.ref_ngrams,

    def _get_ngrams(self, samples):
        samples_size = len(samples)
        all_counters = [Counter([x for y in get_ngrams(samples, n + 1) for x in y])
                        for n in range(self.max_n)]
        for n_counter in all_counters:
            for k in n_counter.keys():
                n_counter[k] /= samples_size
        return all_counters

    def get_score(self, samples):
        print('multiset distances preprocess upto {}!'.format(self.max_n))
        sample_ngrams = self._get_ngrams(samples)
        ngrams_intersection = [sample_ngrams[i] & self.ref_ngrams[i]
                               for i in range(self.max_n)]  # intersection:  min(c[x], d[x])
        ngrams_union = [sample_ngrams[i] | self.ref_ngrams[i]
                        for i in range(self.max_n)]  # union:  max(c[x], d[x])
        ngrams_abs_diff = [ngrams_union[i] - ngrams_intersection[i] \
                           for i in range(self.max_n)]
        ngrams_added = [sample_ngrams[i] + self.ref_ngrams[i]
                        for i in range(self.max_n)]
        temp_results = {}

        print('multiset distances evaluating upto {}!'.format(self.max_n))
        temp_results['jaccard'] = [float(sum(ngrams_intersection[n].values())) / \
                                   sum(ngrams_union[n].values())
                                   for n in range(self.max_n)]
        temp_results['sorensen'] = [float(sum(ngrams_abs_diff[n].values())) / \
                                    sum(ngrams_added[n].values())
                                    for n in range(self.max_n)]
        temp_results['canberra'] = [np.sum([ngrams_abs_diff[n][key] / float(ngrams_added[n][key]) \
                                            for key in ngrams_abs_diff[n]])
                                    for n in range(self.max_n)]
        for p in range(1, self.max_mikowski_order + 1):
            temp_results['p%d-minkowski' % (p,)] = [np.power(np.sum(np.power(list(ngrams_abs_diff[n].values()), p))
                                                             , 1. / p)
                                                    for n in range(self.max_n)]

        result = {}
        for key in temp_results:
            for n in range(self.min_n, self.max_n + 1):
                result[key + '%d' % n] = np.power(reduce(lambda x, y: x * y, temp_results[key][:n]), 1. / n)
        return result


if __name__ == "__main__":
    print(MultisetDistances([['سعدی', 'سعدی']], max_n=2).get_score([['سعدی', 'فردوسی']]))
    print(MultisetDistances([['سعدی', 'سعدی']], max_n=2).get_score([['سعدی', 'سعدی']]))
