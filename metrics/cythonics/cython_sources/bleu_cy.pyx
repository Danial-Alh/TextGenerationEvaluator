# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from cython_sources.bleu_cpp cimport BLEU_CPP
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef class Bleu:
    cdef BLEU_CPP* bleu_instance
    cdef float* temp_weights

    def __cinit__(self, references, weights, int smoothing_function=1, bool auto_reweigh=False,
                  Bleu other_instance=None):
        cdef vector[vector[string]] temp_refs = vector[vector[string]]()
        self.temp_weights = <float*> PyMem_Malloc(len(weights) * sizeof(float))
        for ref in references:
            temp_refs.push_back(vector[string]())
            for token in ref:
                temp_refs.back().push_back(token.encode())
        for i, w in enumerate(weights):
            self.temp_weights[i] = w
        cdef BLEU_CPP* other_cpp_instance = NULL
        if other_instance is not None:
            other_cpp_instance = other_instance.bleu_instance
        self.bleu_instance = new BLEU_CPP(temp_refs, self.temp_weights, len(weights), smoothing_function, auto_reweigh,
                                          other_cpp_instance)

    def __dealloc__(self):
        del self.bleu_instance
        PyMem_Free(self.temp_weights)

    cpdef list get_score(self, list hypotheses):
        cdef vector[vector[string]] temp_hyps = vector[vector[string]]()
        for hyp in hypotheses:
            temp_hyps.push_back(vector[string]())
            for token in hyp:
                temp_hyps.back().push_back(token.encode())
        cdef double* results = <double*> PyMem_Malloc(len(hypotheses) * sizeof(double))
        self.bleu_instance.get_score(temp_hyps, results)
        cdef list py_results = []
        for i in range(len(hypotheses)):
            py_results.append(results[i])
        PyMem_Free(results)
        return py_results
