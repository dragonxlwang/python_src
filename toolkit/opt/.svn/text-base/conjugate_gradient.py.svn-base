'''
Created on Feb 25, 2014

@author: xwang95
'''
from toolkit.num.algebra import mulMatVec, dotVecVec, subVecVec, getVecNorm, \
    ifZeroVec, mulNumVec, addVecVec, randomMat, randomVec, printMat, addMatMat, \
    transposeMat, mulNumMat, eye, sizeMat, sizeVec, ifZero, choleskyDecomp, \
    mulMatMat, subMatMat, forwardBackwardSub, invMat, mulMatLst, rbind, zeroes, \
    linSolve
from random import randint
from toolkit.opt.line_search import cubicInterpolation, wolfeLineSearch
from toolkit.num.arithmetic import _eps
import math

def linSolveConjugateGradient(aMat, b, x0, tol=1e-6, ifOptimize=False,
                              ifShowWarning=False):
    '''[linSolveConjugateGradient]: conjugate gradient method 
    for solving (large-scale symmetric positive definite) 
    linear systems. It was developed in the 1950s by Hestenes
    and Stiefel.
    It can also used to solve symmetric linear system (not 
    positive-definite) 
    
    reference: Num. Opt. (J. N.) Algo. 5.2., P112
    '''
    warnings = set();
    flag = 0;
    #---------------------------------------------------------------- initialize
    n = sizeVec(b);
    r = subVecVec(mulMatVec(aMat, x0), b);
    rr = dotVecVec(r, r);
    p = mulNumVec(-1.0, r);
    x = x0;
    k = 0;
    while(not (math.sqrt(rr) < tol)):
        ap = mulMatVec(aMat, p);  # O(N^2)
        pap = dotVecVec(p, ap);
        #------------------------------------------------------- exit: curvature
        if(ifOptimize and pap <= 0):
            if(ifShowWarning): 
                print('[linSolveConjugateGradient]: curvature condition not ' \
                      'meet for optimization. Exit.');
            warnings.add('curvatureNotMetFlag');
            break;
        alpha = rr / pap;
        x = addVecVec(x, mulNumVec(alpha, p));
        r = addVecVec(r, mulNumVec(alpha, ap));
        rrNew = dotVecVec(r, r);
        beta = rrNew / rr;
        p = addVecVec(mulNumVec(-1.0, r), mulNumVec(beta, p));
        rr = rrNew
        k += 1;
        #------------------------------------------------ exit: max iter reached
        if(k >= n):
            if(ifShowWarning): 
                print('[linSolveConjugateGradient]: maximum iteration ' \
                      'reached. Exit.');
            warnings.add('maxIterReachedFlag'); 
            flag = 1;
            break;
    exitInfo = {'iterNum': k, 'warning': warnings, 'flag': flag};
    return (x, exitInfo);

def preconditionedConjugateGradient(aMat, b, x0, tol=1e-6, ifShowWarning=False):
    '''[preconditionedConjugateGradient]: preconditioned conjugate
    gradient for (large, sparse, symmetric positive-definite). 
    Preconditioner is computed by incomplete Cholesky decomposition.
    
    reference: Num. Opt. (J. N.) Algo. 5.3., P119
    '''
    warnings = set();
    flag = 0;
    #---------------------------------------------------------------- initialize
    n = sizeVec(b);
    r = subVecVec(mulMatVec(aMat, x0), b);
    lMat = choleskyDecomp(aMat, incompleteDecomp=True);
    uMat = transposeMat(lMat);
    y = forwardBackwardSub(lMat, r, ifForward=True, ifOverwrite=False);
    y = forwardBackwardSub(uMat, y, ifForward=False, ifOverwrite=True);
    ry = dotVecVec(r, y);
    p = mulNumVec(-1.0, y);
    x = x0;
    k = 0;
    while(not (getVecNorm(r) < tol)):
        ap = mulMatVec(aMat, p);
        alpha = ry / dotVecVec(p, ap);
        x = addVecVec(x, mulNumVec(alpha, p));
        r = addVecVec(r, mulNumVec(alpha, ap));
        y = forwardBackwardSub(lMat, r, ifForward=True, ifOverwrite=False);
        y = forwardBackwardSub(uMat, y, ifForward=False, ifOverwrite=True);
        ryNew = dotVecVec(r, y);
        beta = ryNew / ry;
        p = addVecVec(mulNumVec(-1.0, y), mulNumVec(beta, p));
        ry = ryNew;
        k += 1;
        if(k >= n):
            if(ifShowWarning): 
                print('[linSolveConjugateGradient]: maximum iteration \
                reached. Exit.')
            warnings.add('maxIterReachedFlag'); 
            flag = 1; 
            break;
    #----------------------------------------------------------------- terminate
    exitInfo = {'iterNum': k, 'warning': warnings, 'flag': flag};
    return (x, exitInfo);

def nonlinearConjugateGradient(fFunc, gFunc, x0, method='Polak-Ribiere',
                               epsDecrease=_eps, epsGrad=_eps, maxIter=100,
                               ifPrint=False):
    warnings = set();
    flag = 0;
    dr = 1.0;
    gn = 1.0;
    #---------------------------------------------------------------- initialize
    n = sizeVec(x0);
    (x, f, g) = (x0, fFunc(x0), gFunc(x0));
    gn = getVecNorm(g);
    p = mulNumVec(-1.0, g);
    (fOld, gOld) = (f, g);
    (k, t) = (0, n);
    while(not (gn < epsGrad or dr < epsDecrease)):
        (alpha, f, gLn, x, ei) = wolfeLineSearch(fFunc, gFunc, x, p,
                                                 c1=1e-4, c2=0.1, maxIter=50,
                                                 initStepLen=1.0,
                                                 ifEnforceCubic=True,
                                                 ifShowWarning=False); 
        g = gFunc(x);
        gn = getVecNorm(g);
        if(method == 'Fletcher-Reeves' or method == 'FR'): 
            beta = gn * gn / dotVecVec(gOld, gOld);
        elif(method == 'Polak-Ribiere' or method == 'PR'): 
            beta = dotVecVec(g, subVecVec(g, gOld)) / dotVecVec(gOld, gOld);
        elif(method == 'Hestenes-Stiefel' or method == 'HS'): 
            beta = dotVecVec(g, subVecVec(g, gOld)) / dotVecVec(subVecVec(g, gOld), p);
        if(t <= 0 or  # n iteration reached 
           beta < 0.2 or  # small beta (non-descent)
           dotVecVec(g, gOld) / (gn * gn) >= 0.1 or  # not orthogonal gradient
           ei['flag'] != 0):  # wolfe line search 
            (beta, t) = (0.0, n);  # restart
        p = addVecVec(mulNumVec(-1.0, g), mulNumVec(beta, p));
        dr = (fOld - f) / (_eps if f == 0.0 else f);
        (fOld, gOld) = (f, g);
        (k, t) = (k + 1, t - 1);
        if(k >= maxIter): break;
        #----------------------------------------------------------- print block
        if(ifPrint):
            print('[nonlinearConjugateGradient]: [iteration {0}] f={1:<15.6e}, ' \
                  'gLine={2:<15.6e}, gNorm={3:<15.6e}, ' \
                  'aLine={4:<15.6e}'.format(k, f, gLn, gn, alpha));
            printMat(x, 'x', decor='e');
            printMat(g, 'g', decor='e');
            if(len(ei['warning']) > 0): 
                print('       Line Search Warnings: ' \
                      '{0}'.format(','.join(ei['warning'])));
            print('         Stop Criteria Check: ' \
                  'delta_f/f={0:<15.6e} (eps={1:<15.6e}), gn={2:<15.6e} ' \
                  '(eps={3:<15.6e})'.format(dr, epsDecrease, gn, epsGrad));
    #---------------------------------------------------------------- loop exits
    if(ei['flag'] != 0): 
        for warning in ei['warning']: 
            warnings.add('lineSearch-{0}'.format(warning));
    if(k >= maxIter):
        warnings.add('maxIterReached');
    exitInfo = {'iterNum': k, 'warning': warnings, 'dr': dr, 'f': f};
    #--------------------------------------------------------------- print block
    if(ifPrint):
        print(80 * '-');
        print('[nonlinearConjugateGradient]: ' \
              'iterNum={0}, f={1:<15.6e}'.format(k, f));
        print('         Stop Criteria Check: ' \
              'delta_f/f={0:<15.6e} (eps={1:<15.6e}), gn={2:<15.6e} ' \
              '(eps={3:<15.6e})'.format(dr, epsDecrease, gn, epsGrad));
        printMat(x, 'x', decor='e');
        printMat(g, 'g', decor='e');
        if(len(exitInfo['warning']) > 0): 
            print('                    Warnings: ' \
                  '{0}'.format(','.join(exitInfo['warning'])));
    #----------------------------------------------------------------- terminate
    return (x, exitInfo);

if __name__ == '__main__':
    a = randomMat(5, 10);
    a = rbind(a, zeroes(5, 10));
    a = addMatMat(a, transposeMat(a));
    x = randomVec(10);
    printMat(x, 'x');
    b = mulMatVec(a, x);
    x = linSolve(mat=a, vec=b);
    printMat(x, 'x-ls');
    (x, ei) = linSolveConjugateGradient(a, b, randomVec(10), ifOptimize=False, 
                                        ifShowWarning=True);
    printMat(x, 'x-cg');
    pass
