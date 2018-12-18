import numpy as np
from scipy import linalg

from metrics.BERT.extract_features import get_features


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


class FBD:
    # inputs must be list of real text as str, not tokenized or ...
    def __init__(self, refrence_list_of_text, max_length=32, bert_model_dir="../data/bert_models/", lower_case=True,
                 batch_size=8):
        self.max_length = max_length
        self.bert_model_dir = bert_model_dir
        self.lower_case = lower_case
        self.batch_size = batch_size

        self.refrence_mu, self.refrence_sigma = self._calculate_statistics(refrence_list_of_text)

    def _get_features(self, list_of_text):
        return get_features(list_of_text=list_of_text, max_length=self.max_length, bert_moded_dir=self.bert_model_dir,
                            lower_case=self.lower_case, batch_size=self.batch_size)

    def _calculate_statistics(self, list_of_text):
        features = self._get_features(list_of_text)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def get_score(self, list_of_text):
        mu, sigma = self._calculate_statistics(list_of_text)
        return calculate_frechet_distance(self.refrence_mu, self.refrence_sigma, mu, sigma)


if __name__ == "__main__":
    from datamanagers.datamanager import DataManager

    datamanager = DataManager(batch_size=64, length=20, vocab_size=None,
                              minlength_threshold=-1,
                              dataset_path="../data/dataset/coco/train.txt",
                              dict_path="../data/dataset/coco/dict.pickle")
    tmp = datamanager.subsample_sequences(10000, True)
    reverse_func = lambda listolist: datamanager.reverse(listolist).split("\n")
    fbd = FBD(reverse_func(tmp[:5000]), bert_model_dir="../data/bert_models/uncased_L-12_H-768_A-12/")
    print(fbd.get_score(reverse_func(tmp[5000:])))
