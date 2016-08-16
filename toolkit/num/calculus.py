'''
Created on Mar 27, 2013

@author: xwang95
'''
import math;
import random;
from algebra import *;
from arithmetic import *;
from algebra import _eps;
# _eps = 1e-8;
_h = 1e-6;

#===============================================================================
# Univariate/Multivariate (1st/2nd) Derivative
#===============================================================================
def derivUniVarFunc(func, x, method='simple', h=_h):
    ''''method = simple:
    2-nd order error, trapezoid method (Sci. Comp. P366)
        method == 'richardson'
    4-th order error, richardson extrapolation (Sci. Comp. P369)
    '''
    if(method == 'simple'):  
        dy = func(x + h) - func(x - h);
        dx = 2.0 * h;
        return (dy / dx);
    elif(method == 'richardson'):
        dy = 8 * (func(x + h) - func(x - h)) - (func(x + 2.0 * h) - func(x - 2.0 * h));
        dx = 12.0 * h;
        return (dy / dx);
    return;

def secDerivUniVarFunc(func, x, method='simple', h=_h):
    ''''method = simple:
    2-nd order error, trapezoid method (Sci. Comp. P366)
        method == 'richardson'
    4-th order error, richardson extrapolation (Sci. Comp. P369)
    '''
    if(method == 'simple'): 
        ddy = func(x + h) - 2.0 * func(x) + func(x - h);
        dx2 = h * h;
        return (ddy / dx2);
    elif(method == 'richardson'):
        ddy = 16.0 * (func(x + h) + func(x - h)) - (func(x + 2.0 * h) + func(x - 2.0 * h)) - 30.0 * func(x);
        dx2 = 12.0 * h * h;
        return (ddy / dx2);
    return;

def gradientFunc(func, x, h=1.0):
    def uniBitVecUpdate(vec, k, val):
        newVec = [e for e in vec];
        newVec[k] = val;
        return newVec;
    uniVarF = lambda(k): (lambda(val): func(uniBitVecUpdate(x, k, val)));
    n = len(x);
    return [derivUniVarFunc(uniVarF(k), x[k], 'richardson', h) for k in range(n)];

def hessianFunc(func, x, h=1e-5):
    def mulBitVecUpdate(vec, kLst, valLst):
        newVec = [e for e in vec];
        for i in range(len(kLst)): newVec[kLst[i]] = valLst[i];
        return newVec;     
    def getHessianElement(i, j):
        if(i == j):
            ddy = (-func(mulBitVecUpdate(x, [i], [x[i] + 2.0 * h]))
                   + 16.0 * func(mulBitVecUpdate(x, [i], [x[i] + 1.0 * h]))
                   - 30.0 * func(x)
                   + 16.0 * func(mulBitVecUpdate(x, [i], [x[i] - 1.0 * h]))
                   - func(mulBitVecUpdate(x, [i], [x[i] - 2.0 * h])));
            dx2 = 12.0 * h * h;
            return (ddy / dx2);
        else:
            ddy = (func(mulBitVecUpdate(x, [i, j], [x[i] + h, x[j] + h])) 
                 - func(mulBitVecUpdate(x, [i, j], [x[i] + h, x[j] - h]))
                 + func(mulBitVecUpdate(x, [i, j], [x[i] - h, x[j] - h]))
                 - func(mulBitVecUpdate(x, [i, j], [x[i] - h, x[j] + h])));
            dx2 = 4.0 * h * h;
            return (ddy / dx2);
    n = len(x);
    hMat = zeroes(n, n);
    for i in range(n):
        for j in range(0, i + 1):
            hMat[i][j] = getHessianElement(i, j);
            hMat[j][i] = hMat[i][j];
    return hMat;

def mulHessianVecApprox(fFunc=None, gFunc=None, xVec=None, vec=None, h=1e-4):
    '''[mulHessianVecApprox]: Hessian(xVex) * vec
    '''
    if(gFunc is None): gFunc = lambda x: gradientFunc(fFunc, x);
    vNorm = getVecNorm(vec);
    normV = getNormalizedVec(vec);
    return mulNumVec(vNorm / h, subVecVec(gFunc(addVecVec(xVec, mulNumVec(h / 2.0, normV))),
                                          gFunc(subVecVec(xVec, mulNumVec(h / 2.0, normV)))));
    
if __name__ == '__main__':
    pass;
