from collections import Counter
from functools import reduce

import numpy as np

from utils import get_ngrams


class MSJaccard:
    def __init__(self, references, max_n=3, cached_fields=None):
        super().__init__()
        print('msjaccard{} init!'.format(max_n))
        self.references = references
        self.max_n = max_n
        if cached_fields is None:
            self.ref_ngrams = self._get_ngrams(references)
        else:
            self.ref_ngrams, = cached_fields
            self.ref_ngrams = self.ref_ngrams[:self.max_n]

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

    def get_score(self, samples, cache=None, return_cache=False):
        print('evaluating ms-jaccard {}!'.format(self.max_n))
        if cache is None:
            sample_ngrams = self._get_ngrams(samples)
            ngrams_intersection = [sample_ngrams[i] & self.ref_ngrams[i]
                                   for i in range(self.max_n)]  # intersection:  min(c[x], d[x])
            ngrams_union = [sample_ngrams[i] | self.ref_ngrams[i]
                            for i in range(self.max_n)]  # union:  max(c[x], d[x])
        else:
            ngrams_intersection, ngrams_union = cache
            ngrams_intersection, ngrams_union = ngrams_intersection[:self.max_n], ngrams_union[:self.max_n]
        result = np.power(reduce(lambda x, y: x * y,
                                 [float(sum(ngrams_intersection[i].values())) / sum(ngrams_union[i].values()) for i in
                                  range(self.max_n)]),
                          1. / self.max_n)
        if return_cache:
            return result, (ngrams_intersection, ngrams_union)
        return result


if __name__ == "__main__":
    print(MSJaccard([['سعدی', 'سعدی']], 1).get_score([['سعدی', 'فردوسی']]))
    print(MSJaccard([['سعدی', 'سعدی']], 1).get_score([['سعدی', 'سعدی']]))
