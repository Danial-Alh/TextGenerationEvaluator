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


def lndiff(lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq):
    # lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq = map(np.array, [lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq])
    assert lnp_fromp.shape == lnq_fromp.shape
    assert lnp_fromq.shape == lnq_fromq.shape

    lndiff_p = lnp_fromp - lnq_fromp
    lndiff_q = lnp_fromq - lnq_fromq

    return lndiff_p, lndiff_q


def Bhattacharyya(lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq):
    lndiff_p, lndiff_q = lndiff(lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq)

    res = -0.5 * (logmeanexp(0.5 * lndiff_q) + logmeanexp(-0.5 * lndiff_p))
    # res = np.exp(0.5 * lndiff1) + np.exp(-0.5 * lndiff2)
    # res = -1. * np.log(np.mean(res))
    return float(res)


def JensenShannon(lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq):
    lndiff_p, lndiff_q = lndiff(lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq)

    # TODO: check numerical error (https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf)
    res = np.mean(log1pexp(lndiff_q)) + np.mean(log1pexp(-1. * lndiff_p))

    # res = np.mean(np.log(1. + np.exp(lndiff1)) + np.log(1. + np.exp(-1. * lndiff2)))
    # res = np.mean(np.log(1. + np.exp(lndiff1) + np.exp(-1. * lndiff2) + np.exp(lndiff1 - lndiff2)))

    return np.log(2.) - 0.5 * res


def Jeffreys(lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq):
    lndiff_p, lndiff_q = lndiff(lnp_fromp, lnq_fromp, lnp_fromq, lnq_fromq)
    return np.mean(lndiff_p) - np.mean(lndiff_q)
