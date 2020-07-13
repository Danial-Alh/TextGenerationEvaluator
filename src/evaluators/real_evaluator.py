import numpy as np
from torchtext.data import ReversibleField

from metrics.bert_distances import EMBD, FBD
from metrics.bleus import Bleu, SelfBleu
from metrics.multiset_distances import MultisetDistances
from db_management.models import ModelSamples
from previous_works.model_wrappers.base_model import BaseModel

from .base_evaluator import Evaluator


class RealWorldEvaluator(Evaluator):
    test_restore_types = ['bleu3', 'bleu4', 'bleu5', 'last_iter']

    def __init__(self, train_ds, valid_ds, test_ds, parser: ReversibleField, mode, k, temperature, dm_name):
        super().__init__(train_ds, valid_ds, test_ds, parser, mode, k, temperature, dm_name)
        self.selfbleu_n_sampling = super.selfbleu_n_s

    def init_metrics(self, mode):
        if mode == 'train':
            print(len(self.train_ds), len(self.valid_ds))
            valid_sentences = self.parser.detokenize(self.valid_ds)
            self.bleu = Bleu(valid_sentences, 3, 5, self.parser, parse=False)
        elif mode == 'eval':
            test_sentences = self.parser.detokenize(self.test_ds)
            self.bleu = Bleu(test_sentences, 2, 5, self.parser, parse=False)
            self.multiset_distances = MultisetDistances(test_sentences, min_n=2, max_n=5)
            self.fbd = FBD(test_sentences, 'bert-base-uncased', self.bert_path)
            self.embd = EMBD(test_sentences, 'bert-base-uncased', self.bert_path)
        elif mode == 'gen':
            pass
        elif mode == 'eval_precheck':
            pass
        else:
            raise BaseException('invalid evaluator mode!')

    def get_initial_scores_during_training(self):
        return {
            'bleu3': [{"value": 0.0, "epoch": -1}],
            'bleu4': [{"value": 0.0, "epoch": -1}],
            'bleu5': [{"value": 0.0, "epoch": -1}],
            '-nll': [{"value": -np.inf, "epoch": -1}]
        }

    def get_during_training_scores(self, model: BaseModel):
        samples = model.generate_samples(self.during_training_n_sampling)
        samples = self.parser.reverse(samples)
        new_scores = {
            '-nll': -model.get_nll(self.temperature)
        }
        for i, v in self.bleu.get_score(samples, parse=False)[0].items():
            new_scores['bleu{}'.format(i)] = v

        return new_scores

    def add_sumplementary_information(self, dumping_object, model, restore_type):
        if model.get_name().lower() == 'real':
            dummy_arr = [1. for _ in range(len(dumping_object['test']['text']))]
            return {'gen': {'lnq': dummy_arr}, 'test': {'lnq': dummy_arr}}
        lnqfromp = model.get_persample_ll(self.temperature, dumping_object['test']['tokens'])
        lnqfromq = model.get_persample_ll(self.temperature, dumping_object['gen']['tokens'])
        return {'gen': {'lnq': lnqfromq}, 'test': {'lnq': lnqfromp}}

    def get_test_scores(self, samples: ModelSamples):
        # generated_tokens = [r.tokens for r in samples.generated_samples]
        generated_sentences = [r.sentence for r in samples.generated_samples]

        if self.selfbleu_n_sampling == -1:
            subsampled_sentences = generated_sentences
            subsamples_mask = np.arange(len(generated_sentences))
        else:
            subsamples_mask = np.random.choice(np.arange(len(generated_sentences)),
                                               self.selfbleu_n_sampling, replace=False)
            subsampled_sentences = np.array(generated_sentences)[subsamples_mask].tolist()

        bleu_result = self.bleu.get_score(generated_sentences)[1]
        selfbleu_result = SelfBleu(subsampled_sentences, 2, 5, self.parser, parse=False)\
            .get_score()[1]
        jaccard_result = self.multiset_distances.get_score('jaccard', generated_sentences)
        fbd_result = self.fbd.get_score(generated_sentences)
        embd_result = self.embd.get_score(generated_sentences)

        persample_scores = {
            '-nll': [r.metrics['lnq'] for r in samples.test_samples],
        }
        mean_scores = {
            **{
                'fbd': fbd_result,
                'embd': embd_result
            },
            **{jaccard_result}
        }

        for i, v in bleu_result.items():
            persample_scores['bleu{}'.format(i)] = v

        for i, v in selfbleu_result.items():
            persample_scores['selfbleu{}'.format(i)] = {'ids': subsamples_mask, 'values': v}

        for key in persample_scores:
            mean_scores[key] = np.mean(persample_scores[key])
        return persample_scores, mean_scores
