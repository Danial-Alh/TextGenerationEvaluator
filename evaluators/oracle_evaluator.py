
class OracleEvaluator(Evaluator):
    test_restore_types = ['-nll_oracle', 'last_iter']

    def __init__(self, train_data, valid_data, test_data, parser, mode, k, temperature, dm_name):
        super().__init__(train_data, valid_data, test_data, parser, mode, k, temperature, dm_name)
        self.selfbleu_n_sampling = selfbleu_n_s

    def init_metrics(self, mode):
        if mode == 'train':
            self.oracle = Oracle_LSTM()
            print(len(self.train_data), len(self.valid_data))
            valid_tokens = word_base_tokenize(self.parser.id_format2line(self.valid_data))
            self.bleu5 = Bleu(valid_tokens, weights=np.ones(5) / 5.)
            self.bleu4 = Bleu(valid_tokens, weights=np.ones(4) / 4., other_instance=self.bleu5)
            self.bleu3 = Bleu(valid_tokens, weights=np.ones(3) / 3., other_instance=self.bleu5)
        elif mode == 'eval':
            test_tokens = word_base_tokenize(self.test_data)
            self.bleu5 = Bleu(test_tokens, weights=np.ones(5) / 5.)
            self.bleu4 = Bleu(test_tokens, weights=np.ones(4) / 4., other_instance=self.bleu5)
            self.bleu3 = Bleu(test_tokens, weights=np.ones(3) / 3., other_instance=self.bleu5)
            self.bleu2 = Bleu(test_tokens, weights=np.ones(2) / 2., other_instance=self.bleu5)
            self.multiset_distances = MultisetDistances(test_tokens, min_n=2, max_n=5)
        elif mode == 'gen':
            self.oracle = Oracle_LSTM()
        elif mode == 'eval_precheck':
            pass
        else:
            raise BaseException('invalid evaluator mode!')

    def get_initial_scores_during_training(self):
        return {
            'bleu3': [{"value": 0.0, "epoch": -1}],
            'bleu4': [{"value": 0.0, "epoch": -1}],
            'bleu5': [{"value": 0.0, "epoch": -1}],
            '-nll_oracle': [{"value": -np.inf, "epoch": -1}],
            '-nll': [{"value": -np.inf, "epoch": -1}]
        }

    def get_during_training_scores(self, model: BaseModel):
        new_samples = model.generate_samples(self.during_training_n_sampling)
        samples = self.parser.id_format2line(new_samples, merge=False)
        new_scores = {
            'bleu3': np.mean(self.bleu3.get_score(samples)),
            'bleu4': np.mean(self.bleu4.get_score(samples)),
            'bleu5': np.mean(self.bleu5.get_score(samples)),
            '-nll_oracle': np.mean(self.oracle.log_probability(new_samples)),
            '-nll': -model.get_nll(self.temperature)
        }
        return new_scores

    def get_sample_additional_fields(self, model: BaseModel, sample_codes, test_codes, restore_type):
        if model.get_name().lower() == 'real':
            dummy_arr = [1. for _ in range(len(test_codes))]
            return {'gen': {'lnq': dummy_arr, 'lnp': dummy_arr},
                    'test': {'lnq': dummy_arr, 'lnp': dummy_arr}}
        test_lines = self.parser.id_format2line(test_codes, merge=False)
        sample_lines = self.parser.id_format2line(sample_codes, merge=False)
        test_lines = np.array([[int(x) for x in y] for y in test_lines])
        sample_lines = np.array([[int(x) for x in y] for y in sample_lines])
        print(test_lines.shape)
        print(sample_lines.shape)

        lnqfromp = model.get_persample_ll(self.temperature, test_codes)
        lnqfromq = model.get_persample_ll(self.temperature, sample_codes)
        lnpfromp = self.oracle.log_probability(test_lines)
        lnpfromq = self.oracle.log_probability(sample_lines)
        return {'gen': {'lnq': lnqfromq, 'lnp': lnpfromq},
                'test': {'lnq': lnqfromp, 'lnp': lnpfromp}}

    def get_test_scores(self, refs_with_additional_fields, samples_with_additional_fields):
        from metrics.divergences import Bhattacharyya, Jeffreys
        sample_lines = [r['text'] for r in samples_with_additional_fields]
        sample_tokens = word_base_tokenize(sample_lines)

        if self.selfbleu_n_sampling == -1:
            subsampled_tokens = sample_tokens
            subsamples_mask = [i for i in range(len(sample_tokens))]
        else:
            subsamples_mask = np.random.choice(
                range(len(sample_tokens)), self.selfbleu_n_sampling, replace=False)
            subsampled_tokens = np.array(sample_tokens)[subsamples_mask].tolist()

        lnqfromp = np.array([r['lnq'] for r in refs_with_additional_fields])
        lnqfromq = np.array([r['lnq'] for r in samples_with_additional_fields])
        lnpfromp = np.array([r['lnp'] for r in refs_with_additional_fields])
        lnpfromq = np.array([r['lnp'] for r in samples_with_additional_fields])

        print(lnqfromp.shape)
        print(lnqfromq.shape)
        print(lnpfromp.shape)
        print(lnpfromq.shape)

        self_bleu5 = SelfBleu(subsampled_tokens, weights=np.ones(5) / 5.)

        scores_persample = {
            'lnqfromp': list(lnqfromp),
            'lnqfromq': list(lnqfromq),
            'lnpfromp': list(lnpfromp),
            'lnpfromq': list(lnpfromq),
            'bleu2': self.bleu2.get_score(sample_tokens),
            'bleu3': self.bleu3.get_score(sample_tokens),
            'bleu4': self.bleu4.get_score(sample_tokens),
            'bleu5': self.bleu5.get_score(sample_tokens),
            'self_bleu5': self_bleu5.get_score(),
            'self_bleu4': SelfBleu(subsampled_tokens, weights=np.ones(4) / 4., other_instance=self_bleu5).get_score(),
            'self_bleu3': SelfBleu(subsampled_tokens, weights=np.ones(3) / 3., other_instance=self_bleu5).get_score(),
            'self_bleu2': SelfBleu(subsampled_tokens, weights=np.ones(2) / 2., other_instance=self_bleu5).get_score(),
        }
        scores_persample['sub_bleu2'] = list(np.array(scores_persample['bleu2'])[subsamples_mask])
        scores_persample['sub_bleu3'] = list(np.array(scores_persample['bleu3'])[subsamples_mask])
        scores_persample['sub_bleu4'] = list(np.array(scores_persample['bleu4'])[subsamples_mask])
        scores_persample['sub_bleu5'] = list(np.array(scores_persample['bleu5'])[subsamples_mask])

        scores_mean = {
            'bhattacharyya': Bhattacharyya(lnpfromp, lnqfromp, lnpfromq, lnqfromq),
            'jeffreys': Jeffreys(lnpfromp, lnqfromp, lnpfromq, lnqfromq),
        }
        scores_mean = {**scores_mean, **self.multiset_distances.get_score(sample_tokens)}
        for key in scores_persample:
            scores_mean[key] = np.mean(scores_persample[key])
        return scores_persample, scores_mean
