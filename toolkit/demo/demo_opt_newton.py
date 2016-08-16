'''
Created on Mar 9, 2014

@author: xwang95
'''
from toolkit.opt.newton import lineSearchNewtonConjugateGradient, lbfgs;
from toolkit.num.algebra import randomVec, printMat, mulMatVec, randomMat, getVecNorm;
from toolkit.num.arithmetic import _eps;
from toolkit.num.calculus import hessianFunc;
import timeit;

def demoLineSearchNewtonConjugateGradient():
    fFunc = lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + (x[0] + x[1]) ** 2 + (x[0] + x[1]) ** 4;
    gFunc = lambda x: [2.0 * (x[0] - 1.0) + 2.0 * (x[0] + x[1]) + 4.0 * (x[0] + x[1]) ** 3,
                       2.0 * (x[1] - 1.0) + 2.0 * (x[0] + x[1]) + 4.0 * (x[0] + x[1]) ** 3];
    (x, f, ei) = lineSearchNewtonConjugateGradient(fFunc, gFunc, xBegVec=randomVec(2), epsHassvec=1e-8, cgThreshold=_eps, epsDecrease=_eps, epsGrad=_eps, maxIter=50, ifPrint=False);
#     printMat(gFunc(x), 'g');
#     printMat(getVecNorm(gFunc(x)), 'gn');
    return;

def demoLbfgs():
    fFunc = lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + (x[0] + x[1]) ** 2 + (x[0] + x[1]) ** 4;
    gFunc = lambda x: [2.0 * (x[0] - 1.0) + 2.0 * (x[0] + x[1]) + 4.0 * (x[0] + x[1]) ** 3,
                       2.0 * (x[1] - 1.0) + 2.0 * (x[0] + x[1]) + 4.0 * (x[0] + x[1]) ** 3];
    (x, f, g, ei) = lbfgs(fFunc, gFunc, xBegVec=randomVec(2), cacheLen=10, epsDecrease=_eps, epsGrad=_eps, maxIter=50, ifPrint=False);
#     printMat(gFunc(x), 'g');
#     printMat(getVecNorm(gFunc(x)), 'gn');
    return;

if __name__ == '__main__':
    print('time={0}s'.format(timeit.timeit(setup="from __main__ import demoLineSearchNewtonConjugateGradient",
                                           stmt='demoLineSearchNewtonConjugateGradient()', number=10000)));
    print('time={0}s'.format(timeit.timeit(setup="from __main__ import demoLbfgs",
                                           stmt='demoLbfgs()', number=10000)));
    pass;