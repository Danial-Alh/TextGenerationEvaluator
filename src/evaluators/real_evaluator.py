import numpy as np
from torchtext.data import ReversibleField

from metrics.bert_distances import EMBD, FBD
from metrics.bleus import Bleu, SelfBleu, ReverseBleu
from metrics.multiset_distances import MultisetDistances
from previous_works.model_wrappers.base_model import BaseModel

from .base_evaluator import Evaluator


class RealWorldEvaluator(Evaluator):
    test_restore_types = ['bleu3', 'bleu4', 'bleu5', 'last_iter']

    def __init__(self, train_ds, valid_ds, test_ds,
                 parser: ReversibleField, mode, dm_name):
        super().__init__(train_ds, valid_ds, test_ds, parser, mode, dm_name)

    def init_metrics(self, mode):
        if mode == 'train':
            print(len(self.train_ds), len(self.valid_ds))
            valid_sentences = self.parser.detokenize(self.valid_ds.text)
            self.bleu = Bleu(valid_sentences, 3, 5, self.parser, parse=False)
        elif mode == 'eval':
            test_sentences = self.parser.detokenize(self.test_ds.text)
            self.bleu = Bleu(test_sentences, 2, 5, self.parser, parse=False)
            self.multiset_distances_rfull = MultisetDistances(test_sentences, min_n=2, max_n=5,
                                                              parser=self.parser, parse=False)
            self.multiset_distances_rsub = MultisetDistances(test_sentences[:int(len(self.test_ds) * self.SUBSAMPLE_FRAC)],
                                                             min_n=2, max_n=5,
                                                             parser=self.parser, parse=False)
            self.fbd = FBD(test_sentences, 'bert-base-uncased', self.BERT_PATH)
            self.embd = EMBD(test_sentences, 'bert-base-uncased', self.BERT_PATH)
        elif mode == 'gen':
            pass
        else:
            raise BaseException('invalid evaluator mode!')

    def get_initial_scores_during_training(self):
        return {
            'bleu3': {"value": 0.0, "epoch": -1},
            'bleu4': {"value": 0.0, "epoch": -1},
            'bleu5': {"value": 0.0, "epoch": -1},
            'neg_nll': {"value": -np.inf, "epoch": -1}
        }

    def get_during_training_scores(self, model: BaseModel, train_temperature):
        samples = model.generate_samples(self.during_training_n_sampling, train_temperature)
        samples = self.parser.detokenize(samples)
        new_scores = {
            'neg_nll': -model.get_nll(self.valid_ds.text, train_temperature)
        }
        for i, v in self.bleu.get_score(samples, parse=False)[0].items():
            new_scores['bleu{}'.format(i)] = v

        return new_scores

    def add_persample_metrics(self, dumping_object, model, test_temperature):
        if model.get_name().lower() == 'real':
            dummy_arr = [0.0 for _ in range(len(dumping_object['test']['text']))]
            return {'generated': {'nllq': dummy_arr}, 'test': {'nllq': dummy_arr}}
        nllqfromp = model.get_persample_nll(dumping_object['test']['tokens'], test_temperature)
        nllqfromq = model.get_persample_nll(dumping_object['generated']['tokens'], test_temperature)
        dumping_object['generated']['nllq'] = nllqfromq
        dumping_object['test']['nllq'] = nllqfromp

    def get_test_scores(self, samples):
        test_sentences = [r.sentence for r in samples['test']]
        generated_sentences = [r.sentence for r in samples['generated']]

        if self.SUBSAMPLE_FRAC == -1:
            subsamples_mask = np.arange(len(generated_sentences))
            subsampled_generated_sentences = generated_sentences
        else:
            subsamples_mask = np.random.choice(np.arange(len(generated_sentences)),
                                               int(self.SUBSAMPLE_FRAC * len(self.test_ds)),
                                               replace=False)
            subsampled_generated_sentences = np.array(generated_sentences)[subsamples_mask].tolist()

        bleu_result = self.bleu.get_score(generated_sentences, parse=False)[1]
        selfbleu_result = SelfBleu(generated_sentences, 2, 5, self.parser, parse=False)\
            .get_score()[1]
        revbleu_result = ReverseBleu(test_sentences, generated_sentences, 2, 5, self.parser, parse=False)\
            .get_score()[1]

        jaccard_result_rfull_hfull = self.multiset_distances_rfull\
            .get_score('jaccard', generated_sentences, parse=False)
        jaccard_result_rfull_hsub = self.multiset_distances_rfull\
            .get_score('jaccard', subsampled_generated_sentences, parse=False)
        jaccard_result_rsub_hfull = self.multiset_distances_rsub\
            .get_score('jaccard', generated_sentences, parse=False)
        jaccard_result_rsub_hsub = self.multiset_distances_rsub\
            .get_score('jaccard', subsampled_generated_sentences, parse=False)

        fbd_result = self.fbd.get_score(generated_sentences)
        embd_result = self.embd.get_score(generated_sentences)

        persample_scores = {}

        mean_scores = {
            'nll': np.mean([r.metrics['nllq'].value for r in samples['test']]),
            'fbd': fbd_result,
            'embd': embd_result
        }

        for i in jaccard_result_rfull_hsub.keys():
            temp_frac = int(self.SUBSAMPLE_FRAC * 100)
            mean_scores['jaccard{}_r{}_h{}'.format(i, 100, 100)] = \
                jaccard_result_rfull_hfull[i]
            mean_scores['jaccard{}_r{}_h{}'.format(i, 100, temp_frac)] =\
                jaccard_result_rfull_hsub[i]
            mean_scores['jaccard{}_r{}_h{}'.format(i, temp_frac, 100)] =\
                jaccard_result_rsub_hfull[i]
            mean_scores['jaccard{}_r{}_h{}'.format(i, temp_frac, temp_frac)] =\
                jaccard_result_rsub_hsub[i]

        for i, v in revbleu_result.items():
            mean_scores['revbleu{}'.format(i)] = {'value': np.mean(v), 'std': np.std(v)}

        for i, v in bleu_result.items():
            persample_scores['bleu{}'.format(i)] = v

        for i, v in selfbleu_result.items():
            persample_scores['selfbleu{}'.format(i)] = {'ids': subsamples_mask, 'values': v}

        for key, values in persample_scores.items():
            if isinstance(values, dict):
                mean_scores[key] = {
                    'value': np.mean(values['values']),
                    'std': np.std(values['values'])
                }
            else:
                mean_scores[key] = {
                    'value': np.mean(values),
                    'std': np.std(values)
                }
        return persample_scores, mean_scores
