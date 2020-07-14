# from scipy.special import logsumexp
import numpy as np
from scipy.misc import logsumexp


def logmeanexp(inp):
    # log(mean(exp(x)))
    l = inp.shape[0]
    assert inp.shape == (l,)
    return logsumexp(inp - np.log(l))


def log1pexp(inp):
    # log(1 + exp(x))
    return np.logaddexp(0., inp)


def lndiff(nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq):
    # nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq = map(np.array, [nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq])
    assert nllp_fromp.shape == nllq_fromp.shape
    assert nllp_fromq.shape == nllq_fromq.shape

    lndiff_p = nllp_fromp - nllq_fromp
    lndiff_q = nllp_fromq - nllq_fromq

    return lndiff_p, lndiff_q


def Bhattacharyya(nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq):
    lndiff_p, lndiff_q = lndiff(nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq)

    res = -0.5 * (logmeanexp(0.5 * lndiff_q) + logmeanexp(-0.5 * lndiff_p))
    # res = np.exp(0.5 * lndiff1) + np.exp(-0.5 * lndiff2)
    # res = -1. * np.log(np.mean(res))
    return float(res)


def JensenShannon(nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq):
    lndiff_p, lndiff_q = lndiff(nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq)

    # TODO: check numerical error (https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
    res = np.mean(log1pexp(lndiff_q)) + np.mean(log1pexp(-1. * lndiff_p))

    # res = np.mean(np.log(1. + np.exp(lndiff1)) + np.log(1. + np.exp(-1. * lndiff2)))
    # res = np.mean(np.log(1. + np.exp(lndiff1) + np.exp(-1. * lndiff2) + np.exp(lndiff1 - lndiff2)))

    return np.log(2.) - 0.5 * res


def Jeffreys(nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq):
    lndiff_p, lndiff_q = lndiff(nllp_fromp, nllq_fromp, nllp_fromq, nllq_fromq)
    return np.mean(lndiff_p) - np.mean(lndiff_q)
