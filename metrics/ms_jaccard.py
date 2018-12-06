from collections import Counter
from functools import reduce

import numpy as np

from utils import get_ngrams


class MSJaccard:
    def __init__(self, references, max_ngram=3):
        super().__init__()
        self.references = references
        self.max_ngram = max_ngram
        self.ref_ngrams = self._get_ngrams(references)

    def _get_ngrams(self, samples):
        samples_size = len(samples)
        all_counters = [Counter([x for y in get_ngrams(samples, n + 1) for x in y])
                        for n in range(self.max_ngram)]
        for n_counter in all_counters:
            for k in n_counter.keys():
                n_counter[k] /= samples_size
        return all_counters

    def jaccard(self, samples):
        sample_ngrams = self._get_ngrams(samples)
        ngrams_intersection = [sample_ngrams[i] & self.ref_ngrams[i]
                               for i in range(self.max_ngram)]  # intersection:  min(c[x], d[x])
        ngrams_union = [sample_ngrams[i] | self.ref_ngrams[i]
                        for i in range(self.max_ngram)]  # union:  max(c[x], d[x])
        return np.power(reduce(lambda x, y: x * y,
                               [float(sum(ngrams_intersection[i].values())) / sum(ngrams_union[i].values()) for i in
                                range(self.max_ngram)]),
                        1. / self.max_ngram)


if __name__ == "__main__":
    print(MSJaccard([['سعدی', 'سعدی']], 1).jaccard([['سعدی', 'فردوسی']]))
    print(MSJaccard([['سعدی', 'سعدی']], 1).jaccard([['سعدی', 'سعدی']]))
