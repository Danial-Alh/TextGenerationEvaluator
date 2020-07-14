import numpy as np
from torchtext.data import ReversibleField

from metrics.base_metric import BaseMetric


class Bleu(BaseMetric):
    def __init__(self, samples, min_n=2, max_n=5, parser: ReversibleField = None, parse=True):
        super().__init__('bleu')
        from fast_bleu import BLEU as FBLEU

        assert max_n >= min_n
        assert min_n >= 1

        if parse:
            samples = parser.reverse(samples)
        ref_tokens = [parser.tokenize(r) for r in samples]
        self.parser = parser
        w = {i: np.ones(i) / i for i in range(min_n, max_n + 1)}
        self.bleu = FBLEU(ref_tokens, w)
        print('bleu instance created!')

    def get_score(self, samples, parse=True):
        if parse:
            samples = self.parser.reverse(samples)
        samples = [self.parser.tokenize(r) for r in samples]
        scores = self.bleu.get_score(samples)
        return {run: np.mean(scores[run]) for run in scores.keys()}, scores


class SelfBleu(BaseMetric):
    def __init__(self, samples, min_n=2, max_n=5, parser: ReversibleField = None, parse=True):
        super().__init__('self-bleu')
        from fast_bleu import SelfBLEU as FSBLEU

        assert max_n >= min_n
        assert min_n >= 1

        if parse:
            samples = parser.reverse(samples)
        ref_tokens = [parser.tokenize(r) for r in samples]
        w = {i: np.ones(i) / i for i in range(min_n, max_n + 1)}
        self.selfbleu = FSBLEU(ref_tokens, w)
        print('self-bleu instance created!')

    def get_score(self):
        scores = self.selfbleu.get_score()
        return {run: np.mean(scores[run]) for run in scores.keys()}, scores
