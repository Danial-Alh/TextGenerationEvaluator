import math

import numpy as np
import ot
from scipy import linalg

from metrics.bert.extract_features import get_features


# https://arxiv.org/pdf/1706.08500.pdf
# from https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class BertDistance:
    # inputs must be list of real text as str, not tokenized or ...
    def __init__(self, refrence_list_of_text, max_length=45, bert_model_dir="../data/bert_models/", lower_case=True,
                 batch_size=8):
        self.max_length = max_length
        self.bert_model_dir = bert_model_dir
        self.lower_case = lower_case
        self.batch_size = batch_size

        self.reference_features = self._get_features(refrence_list_of_text)  # sample * feature
        assert self.reference_features.shape[0] == len(refrence_list_of_text)
        self.refrence_mu, self.refrence_sigma = self._calculate_statistics(self.reference_features)

    def _get_features(self, list_of_text):
        return get_features(list_of_text=list_of_text, max_length=self.max_length, bert_moded_dir=self.bert_model_dir,
                            lower_case=self.lower_case, batch_size=self.batch_size)

    def _calculate_statistics(self, features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def get_score(self, list_of_text):
        features = self._get_features(list_of_text)
        assert not diff(self.reference_features.shape[0], features.shape[0]), \
            "[!] Warning: different size between reference and input can be effect on the result. ref: {}, test: {}" \
                .format(self.reference_features.shape[0], features.shape[0])

        mu, sigma = self._calculate_statistics(features)
        FBD_res = math.sqrt(calculate_frechet_distance(self.refrence_mu, self.refrence_sigma, mu, sigma))

        # FBD W1_on euclidean metric between gaussian: http://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
        res = {"fbd": FBD_res}

        # Todo : use probability of sample for weighting samples
        M = ot.dist(self.reference_features, features, metric="euclidean")
        res["w1_euclidean"] = ot.emd2(a=[], b=[], M=M)

        M = ot.dist(self.reference_features, features, metric="sqeuclidean")
        res["w2_euclidean"] = math.sqrt(ot.emd2(a=[], b=[], M=M))

        M = ot.dist(self.reference_features, features, metric="cosine")
        res["w1_cosine"] = ot.emd2(a=[], b=[], M=M)

        M = np.square(ot.dist(self.reference_features, features, metric="cosine"))
        res["w2_cosine"] = math.sqrt(ot.emd2(a=[], b=[], M=M))
        return res


def diff(a, b, thresh=0.05):
    return float(abs(a - b)) / float(min(a, b)) > thresh


if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups
    import nltk


    def data(cat):
        data = fetch_20newsgroups(subset='train', categories=[cat]).data
        res = []
        for doc in data:
            res += nltk.sent_tokenize(doc)[4:]
        return res[:100]


    cats = ["rec.sport.hockey", "rec.motorcycles", "sci.space"]
    res = {}
    for cat1 in cats:
        res[cat1] = {}
        evaluator = BertDistance(data(cat1), bert_model_dir="./data/bert_models/")
        for cat2 in cats:
            res[cat1][cat2] = evaluator.get_score(data(cat2))

    metric_names = res[cats[0]][cats[1]].keys()
    print("\t".join(cats))
    for metric_name in metric_names:
        print("=" * 15 + "[" + metric_name + "]" + "=" * 15)
        for cat1 in cats:
            print("\t".join([cat1] + list(map(str, [res[cat1][cat2][metric_name] for cat2 in cats]))))
