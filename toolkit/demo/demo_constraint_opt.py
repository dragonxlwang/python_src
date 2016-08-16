'''
Created on Mar 24, 2014

@author: xwang95
'''
from toolkit.demo.benchmark_problems import rosenbrock
from toolkit.num.algebra import randomVec, printMat, randomMat, eye
from toolkit.opt.newton import lbfgsb
import sys

if __name__ == '__main__':
    dim = 30;
    (fFunc, gFunc, hFunc) = rosenbrock(dim);
    x0 = randomVec(dim, 1.0-0.0001, 2.048);
    boxConstLst = [(i, 1.0-0.0001, 2.048) for i in range(dim)];
    
    (x, f, g, ei) = lbfgsb(fFunc, gFunc, x0, boxConstLst, 
                           method='conjugate-gradient', 
                           epsDecrease=1e-20, epsGrad=1e-6, 
                           maxIter=1e10, ifPrint=True);
    printMat(f, 'f');
    printMat(x, 'x');
    printMat(g, 'g');
    printMat(ei['iterNum'], 'iterNum');
    print 80*'~';
#     sys.stdin.readline();
    
    (x, f, g, ei) = lbfgsb(fFunc, gFunc, x0, boxConstLst, 
                           method='direct-primal', 
                           epsDecrease=1e-20, epsGrad=1e-6, 
                           maxIter=1e10, ifPrint=True);
    printMat(f, 'f');
    printMat(x, 'x');
    printMat(g, 'g');
    printMat(ei['iterNum'], 'iterNum');