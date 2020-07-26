def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y


def parameters_allocation_check(module):
    parameters = list(module.parameters())
    return fold(f_and, parameters, True) or not fold(f_or, parameters, False)


def handle_inputs(inputs, use_cuda):
    import torch as t
    from torch.autograd import Variable

    result = [Variable(t.from_numpy(var)) for var in inputs]
    result = [var.cuda() if use_cuda else var for var in result]

    return result


def kld_coef(i, batchloader):
    total_iterations = batchloader.TOTAL_EPOCHS * batchloader.num_batches

    import math
    return (math.tanh((i - (total_iterations)/2) / 2 * 7/2) + 1)/2
    # return (math.tanh((i - 3500)/1000) + 1)/2
