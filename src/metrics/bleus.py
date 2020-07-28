import numpy as np
from torchtext.data import ReversibleField

from metrics.base_metric import BaseMetric


class Bleu(BaseMetric):
    def __init__(self, samples, min_n=2, max_n=5, parser: ReversibleField = None, parse=True):
        super().__init__()
        from fast_bleu import BLEU as FBLEU

        assert max_n >= min_n
        assert min_n >= 1

        if parse:
            samples = parser.reverse(samples)
        ref_tokens = [parser.tokenize(r) for r in samples]
        self.parser = parser
        w = {i: np.ones(i) / i for i in range(min_n, max_n + 1)}
        self.bleu = FBLEU(ref_tokens, w, verbose=True)
        print('LOG: BLEU init done!')

    def get_score(self, samples, parse=True):
        print('LOG: calculating BLEU!')
        if parse:
            samples = self.parser.reverse(samples)
        samples = [self.parser.tokenize(r) for r in samples]
        scores = self.bleu.get_score(samples)
        result = ({run: np.mean(scores[run]) for run in scores.keys()}, scores)
        print('LOG: done!')
        return result


class SelfBleu(BaseMetric):
    def __init__(self, samples, min_n=2, max_n=5, parser: ReversibleField = None, parse=True):
        super().__init__()
        from fast_bleu import SelfBLEU as FSBLEU

        assert max_n >= min_n
        assert min_n >= 1

        if parse:
            samples = parser.reverse(samples)
        ref_tokens = [parser.tokenize(r) for r in samples]
        w = {i: np.ones(i) / i for i in range(min_n, max_n + 1)}
        self.selfbleu = FSBLEU(ref_tokens, w, verbose=True)
        print('LOG: SelfBLEU init done!')

    def get_score(self):
        print('LOG: calculating SelfBLEU!')
        scores = self.selfbleu.get_score()
        result = ({run: np.mean(scores[run]) for run in scores.keys()}, scores)
        print('LOG: done!')
        return result


class ReverseBleu(BaseMetric):
    def __init__(self, ref_samples, hyp_samples, min_n=2, max_n=5, parser: ReversibleField = None, parse=True):
        super().__init__()
        from fast_bleu import BLEU as FBLEU

        assert max_n >= min_n
        assert min_n >= 1

        if parse:
            ref_samples = parser.reverse(ref_samples)
            hyp_samples = parser.reverse(hyp_samples)
        self.ref_tokens = [parser.tokenize(r) for r in ref_samples]
        self.hyp_tokens = [parser.tokenize(r) for r in hyp_samples]
        self.parser = parser
        w = {i: np.ones(i) / i for i in range(min_n, max_n + 1)}
        self.bleu = FBLEU(self.hyp_tokens, w, verbose=True)
        print('LOG: ReverseBLEU init done!')

    def get_score(self):
        print('LOG: calculating ReverseBLEU!')
        scores = self.bleu.get_score(self.ref_tokens)
        result = ({run: np.mean(scores[run]) for run in scores.keys()}, scores)
        print('LOG: done!')
        return result
