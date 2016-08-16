'''
Created on Mar 11, 2014

@author: xwang95
'''
from toolkit.demo.benchmark_problems import sphere, schwefel, rosenbrock
from toolkit.opt.newton import lineSearchNewtonConjugateGradient, lbfgs
from toolkit.num.algebra import randomVec, printMat, ifPositiveDefinite, \
    subVecVec, mulNumVec, getVecNorm
from toolkit.opt.conjugate_gradient import nonlinearConjugateGradient
from toolkit.num.arithmetic import _eps
from timeit import timeit
import sys

def testBed(method, problem, dim, lb, ub, repeat=1, ifPrint=False, x=None):
    if(problem.lower() == 'sphere'): (fFunc, gFunc, hFunc) = sphere(dim);
    if(problem.lower() == 'schwefel'): (fFunc, gFunc) = schwefel(dim);
    if(problem.lower() == 'rosenbrock'): (fFunc, gFunc, hFunc) = rosenbrock(dim);
    
    iterNum = 0;
    if(method.lower() == 'LineSearchNewtonConjugateGradient'.lower()):
        for i in range(repeat):
            (x, f, ei) = lineSearchNewtonConjugateGradient(fFunc, gFunc, 
                            x if x is not None else randomVec(dim, lb, ub), 
                            epsHessvec=_eps, cgThreshold=_eps, 
                            epsDecrease= -1.0, epsGrad=1e-6, 
                            maxIter=1e10, ifPrint=ifPrint);
            iterNum += ei['iterNum']; 
    elif(method.lower() == 'nonlinearConjugateGradient'.lower()):
        for i in range(repeat):
            (x, ei) = nonlinearConjugateGradient(fFunc, gFunc, 
                            x if x is not None else randomVec(dim, lb, ub), 
                            method='PR', 
                            epsDecrease= -1.0, epsGrad=1e-6, 
                            maxIter=1e10, ifPrint=ifPrint);
            iterNum += ei['iterNum'];
    elif(method.lower() == 'lbfgs'.lower()):
        for i in range(repeat):
            (x, f, g, ei) = lbfgs(fFunc, gFunc, 
                            x if x is not None else randomVec(dim, lb, ub), 
                            cacheLen=10, 
                            epsDecrease= -1.0, epsGrad=1e-6, 
                            maxIter=1e10, ifPrint=ifPrint);
            iterNum += ei['iterNum'];
    print(80*'$');
    print('iterNum = {0}'.format(ei['iterNum']));
    print(fFunc(x));
    printMat(x, 'x');
    printMat(fFunc(x), 'f', decor='e');
    printMat(gFunc(x), 'g', decor='e');
    printMat(getVecNorm(gFunc(x)), 'gn', decor='e');
    return;

if __name__ == '__main__':
    (repeat, ifPrint) = (1, True);
#     (problem, dim, lb, ub) = ('sphere', 10, -5.12, 5.12);
#     (problem, dim, lb, ub) = ('schwefel', 10, -65.536, 65.536);q
#     (problem, dim, lb, ub) = ('schwefel', 10, -65.536, 65.536);
    (problem, dim, lb, ub) = ('rosenbrock', 30, -2.048, 2.048);
    x = randomVec(dim, lb, ub);
#------------------------------------------------------------------------------ 
#     another critical point for Rosenbrock function (dim > 3)
#     x = [-9.93286e-01, 9.96651e-01, 9.98330e-01, 9.99168e-01, 9.99585e-01,
#          9.99793e-01, 9.99897e-01, 9.99949e-01, 9.99974e-01, 9.99987e-01,
#          9.99994e-01, 9.99997e-01, 9.99998e-01, 9.99999e-01, 1.00000e+00,
#          1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
#          1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
#          1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00];
    
#     print('[LineSearchNewtonConjugateGradient]');
#     def f(): testBed(method='LineSearchNewtonConjugateGradient', 
#                      problem=problem, dim=dim, lb=lb, ub=ub, repeat=repeat, 
#                      ifPrint=ifPrint, x=x);
#     print('time={0}s'.format(timeit(setup='from __main__ import f', 
#                                     stmt="f()", number=1)));
#     sys.stdin.readline();
#     print('');
#     
#     print('[nonlinearConjugateGradient]');
#     def f(): testBed(method='nonlinearConjugateGradient', 
#                      problem=problem, dim=dim, lb=lb, ub=ub, repeat=repeat, 
#                      ifPrint=ifPrint, x=x);
#     print('time={0}s'.format(timeit(setup='from __main__ import f', 
#                                     stmt="f()", number=1)));
#     sys.stdin.readline();
#     print('');
#       
    print('[lbfgs]');
    def f(): testBed(method='lbfgs', 
                     problem=problem, dim=dim, lb=lb, ub=ub, repeat=repeat, 
                     ifPrint=ifPrint, x=x);
    print('time={0}s'.format(timeit(setup='from __main__ import f', 
                                    stmt="f()", number=1)));
    print('');
    print(80*'~');
    pass
