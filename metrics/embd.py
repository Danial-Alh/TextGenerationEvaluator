import ot

from metrics.bert.extract_features import get_features


class EMBD:
    # inputs must be list of real text as str, not tokenized or ...
    def __init__(self, refrence_list_of_text, max_length=32, bert_model_dir="../data/bert_models/", lower_case=True,
                 batch_size=8):
        self.max_length = max_length
        self.bert_model_dir = bert_model_dir
        self.lower_case = lower_case
        self.batch_size = batch_size

        self.refrence_features = self._get_features(refrence_list_of_text)  # sample * feature
        assert self.refrence_features.shape[0] == len(refrence_list_of_text)

    def _get_features(self, list_of_text):
        return get_features(list_of_text=list_of_text, max_length=self.max_length, bert_moded_dir=self.bert_model_dir,
                            lower_case=self.lower_case, batch_size=self.batch_size)

    def get_score(self, list_of_text):
        features = self._get_features(list_of_text)
        M = ot.dist(self.refrence_features, features, metric="sqeuclidean")
        return ot.emd2(a=[], b=[], M=M, numItermax=10**6)
