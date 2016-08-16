'''
Created on Mar 5, 2014

@author: xwang95
'''
from toolkit.num.algebra import *;
from toolkit.opt.conjugate_gradient import linSolveConjugateGradient, \
    preconditionedConjugateGradient;
import timeit;
from toolkit.opt.conjugate_gradient import nonlinearConjugateGradient
from toolkit.num.arithmetic import getExponent

def demoLinSolveConjugateGradient():
    print('[linSolveConjugateGradient]: testing');
    for i in range(1):
        n = 400;
        a = randomMat(n, n);
        a = addMatMat(a, transposeMat(a));
        for j in range(n): a[j][j] += 1000 * (random.random() + 1.0);
        x = randomVec(n);
        b = mulMatVec(a, x);
        (x, ei) = linSolveConjugateGradient(a, b, randomVec(n), tol=1e-8, ifOptimize=True);
        z = subVecVec(mulMatVec(a, x), b);
        if(not ifZero(z)): 
            print "Error";
            print(getVecNorm(z));
            printMat(z);
        print(i, ei['iterNum']);
    print('[linSolveConjugateGradient]: validated');
    return;

def demoPreconditionedConjugateGradient():
    print('[preconditionedConjugateGradient]: testing');
    for i in range(1):
        n = 400;
        a = randomMat(n, n);
        a = addMatMat(a, transposeMat(a));
        for j in range(n): a[j][j] += 1000 * (random.random() + 1.0);
        x = randomVec(n);
        b = mulMatVec(a, x);
        (x, ei) = preconditionedConjugateGradient(a, b, randomVec(n));
        z = subVecVec(mulMatVec(a, x), b);
        if(not ifZero(z)): 
            print "Error";
            print(getVecNorm(z));
            printMat(z);
        print(i, ei['iterNum']);
    print('[preconditionedConjugateGradient]: validated');
    return;

def demoNonlinearConjugateGradient():
    fFunc = lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + (x[0] + x[1]) ** 2 + (x[0] + x[1]) ** 4;
    gFunc = lambda x: [2.0 * (x[0] - 1.0) + 2.0 * (x[0] + x[1]) + 4.0 * (x[0] + x[1]) ** 3,
                       2.0 * (x[1] - 1.0) + 2.0 * (x[0] + x[1]) + 4.0 * (x[0] + x[1]) ** 3];
    (x, ei) = nonlinearConjugateGradient(fFunc, gFunc, randomVec(2), method='PR', ifPrint=False);
# #     printMat(gFunc(x));
#     print(getExponent(getVecNorm(gFunc(x))));
#     print x
#     print ei;
    return;

if __name__ == '__main__':
#     print('time={0}s'.format(timeit.timeit(setup="from __main__ import demoLinSolveConjugateGradient", 
#                                            stmt='demoLinSolveConjugateGradient()', number=1)));
#     print('time={0}s'.format(timeit.timeit(setup="from __main__ import demoPreconditionedConjugateGradient", 
#                                            stmt='demoPreconditionedConjugateGradient()', number=1)));
    print('time={0}s'.format(timeit.timeit(setup="from __main__ import demoNonlinearConjugateGradient",
                                           stmt='demoNonlinearConjugateGradient()', number=10000)));
    pass
