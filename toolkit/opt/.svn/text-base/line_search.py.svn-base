'''
Created on Mar 12, 2013

@author: xwang95
'''
from toolkit.num.algebra import addVecVec, mulNumVec, dotVecVec, printMat, getVecNorm;
from toolkit.num.calculus import derivUniVarFunc; 
from random import random;
import math;
from toolkit.num.arithmetic import ifZeroNum, sgn, _eps;
import sys;

NOT_FOLD = True;
#===============================================================================
# Inexact Line Search
#===============================================================================
class WolfeLineSearch(object):
    '''[WolfeLineSearch]: Wolfe lines search algorithm for step length selection. 
    It selects a step length that satisfies the strong Wolfe conditions:
    1). Sufficient Decrease Condition (Armijo):
        f(x0 + alpha * p) <= f(x0) + c1 * alpha * p' * f'(x0)
    2). Curvature Condition: 
        |p' * f'(x0 + alpha * p)| <= c2 |p' * f'(x0)|
    where c1, c2 are two predefined constants. In general, c1 is a small positive
    value and c2 is close to 1.0.
    The search calls zoom() which maintains aLo and aHi:
    aLo has the (so far) smallest fLineFunc value while satisfying sufficient decrease condition 
    gLineFunc(aLo)' * (aHi - aLo) < 0. 
    
    args:
        xBegVec    : beginning point (vector) for line search
        dirVec     : search direction vector
        fFunc      : function evaluation
        gFunc      : gradient function evaluation (if None, numeric method used)
        aMax       : maximum step length
        c1         : constant defined for sufficient decrease condition
        c2         : constant defined for curvature condition
        ifPrint    : when set True, warnings will be print
        
    returns:
        xEndVec    : the point selected after an appropriate step length used
    
    references:
        wolfe line search algorithm: Num. Opt. (J.N.), P60 Algo. 3.5, 3.6
        initial step length: Num. Opt. (J.N.) P59
        cubic interpolation: Num. Opt. (J.N.) P59
    '''
    # input arguments
    fFunc = None;
    gFunc = None;
    xBegVec = None;
    dirVec = None;
    c1 = None;
    c2 = None;
    maxIter = None;
    initStepLen = None;
    ifPrint = None;
    ifShowWarning = None;
    # internal variables
    fLineFunc = None;
    gLineFunc = None;
    fBeg = None;
    gBeg = None;    
    iter = None;
    exitInfo = None;
    maxIterReachedFlag = None;
    tooSmallBracketFlag = None;
    ifEnforceCubic = None;
    
    def __init__(self, fFunc, gFunc, xBegVec, dirVec, c1=1e-4, c2=0.9, maxIter=50, initStepLen=1.0, ifEnforceCubic=False, ifPrint=False, ifShowWarning=False):
        self.fFunc = fFunc;
        self.gFunc = gFunc;
        self.xBegVec = xBegVec;
        self.dirVec = dirVec;
        self.c1 = c1;
        self.c2 = c2;
        self.maxIter = maxIter;
        self.initStepLen = initStepLen;
        self.ifPrint = ifPrint;
        self.ifShowWarning = ifShowWarning;
        self.fLineFunc = lambda a: self.fFunc(addVecVec(xBegVec, mulNumVec(a, dirVec)));
        if(self.gFunc is None): 
            self.gLineFunc = lambda a: derivUniVarFunc(self.fLineFunc, a, method='simple');
        else: 
            self.gLineFunc = lambda a: dotVecVec(self.gFunc(addVecVec(xBegVec, mulNumVec(a, dirVec))), dirVec);
        self.fBeg = self.fLineFunc(0.0);
        self.gBeg = self.gLineFunc(0.0);
        self.iter = 0;
        self.exitInfo = {'flag':0, 'warning':set(), 'iterNum':0};
        self.maxIterReachedFlag = False;
        self.tooSmallBracketFlag = False;
        self.ifEnforceCubic = ifEnforceCubic;  
        return;
    
    def getExitInfo(self):
        if(self.maxIterReachedFlag): self.exitInfo['warning'].add('maxIterReachedFlag');
        if(self.tooSmallBracketFlag): self.exitInfo['warning'].add('tooSmallBracketFlag');
        if(len(self.exitInfo['warning']) > 0): self.exitInfo['flag'] = 1;
        self.exitInfo['iterNum'] = self.iter;
        return self.exitInfo;
    
    def ifSufficientDecrease(self, aVal, fVal): return (fVal <= self.fBeg + self.c1 * aVal * self.gBeg);
    
    def ifCurvatureCondition(self, aVal, gVal): return (abs(gVal) <= abs(self.c2 * self.gBeg)); 
    
    def ifStrongWolfeSatisfy(self, aVal, fVal=None, gVal=None):
        if(fVal is None): fVal = self.fLineFunc(aVal);
        if(gVal is None): gVal = self.gLineFunc(aVal);
        return (self.ifSufficientDecrease(aVal, fVal) and self.ifCurvatureCondition(aVal, gVal));
        
    def getXvec(self, aVal): return addVecVec(self.xBegVec, mulNumVec(aVal, self.dirVec));
    
    def search(self):
        aPre = 0.0;
        fPre = self.fBeg;
        gPre = self.gBeg;
    
        aCur = self.initStepLen;
        fCur = None;
        gCur = None;
        
        while(True):
            fCur = self.fLineFunc(aCur);
            if(not (self.ifSufficientDecrease(aCur, fCur) and (fCur < fPre))): return self.zoom(aPre, aCur, fPre, fCur, gPre, gCur);
            gCur = self.gLineFunc(aCur);
            if(self.ifCurvatureCondition(aCur, gCur)): return (aCur, fCur, gCur, self.getXvec(aCur), self.getExitInfo());
            elif(gCur > 0.0): return self.zoom(aCur, aPre, fCur, fPre, gCur, gPre);
            else: (aPre, fPre, gPre, aCur) = (aCur, fCur, gCur, 2.0 * aCur);
            #------------------------------------------------------------------------------
            if(self.ifPrint): 
                print('[Wolfe Line Search]:[iter {0:^4}] a={1:<15.6e}, f={2:<15.6e}, g={3:<15.6e}'.format(self.iter, aCur, fCur, gCur)); 
            #------------------------------------------------------------------------------ 
            self.iter += 1;
            if(self.iter >= self.maxIter):
                self.maxIterReachedFlag = True;
                #------------------------------------------------------------------------------ 
                if(self.ifShowWarning): 
                    print('[Wolfe Line Search]: *WARNING* [iter {0:^4}] maximal iteration reached'.format(self.iter));
                #------------------------------------------------------------------------------ 
                return (aCur, fCur, gCur, self.getXvec(aCur), self.getExitInfo());
        return;
    
    def zoom(self, aLo, aHi, fLo, fHi, gLo, gHi):
        while(True):
            #------------------------------------------------------------------------------ 
            if(ifZeroNum(aLo - aHi)):
                self.tooSmallBracketFlag = True;
                if(self.ifShowWarning): 
                    print('[Wolfe Line Search]: *WARNING* [iter {0:^4}] bracket too small, [aLo={1}, aHi={2}, fLo={3}, fHi={4}]'.format(self.iter, aLo, aHi, fLo, fHi));
                return (aLo, fLo, gLo if gLo is not None else self.gLineFunc(aLo), self.getXvec(aLo), self.getExitInfo());  # exit: too small bracet
            #------------------------------------------------------------------------------ 
            if(self.ifEnforceCubic or ((gLo is not None) and (gHi is not None))):  # cubic interpolation
                if(gLo is None): gLo = self.gLineFunc(aLo);
                if(gHi is None): gHi = self.gLineFunc(aHi);
                d1 = gLo + gHi - 3.0 * (fLo - fHi) / (aLo - aHi);
                d3 = d1 * d1 - gLo * gHi;
                if(d3 >= 0):
                    d2 = sgn(aHi - aLo) * math.sqrt(d3);
                    d4 = (gHi - gLo + 2.0 * d2);
                    if(d4 != 0):
                        aMi = aHi - (aHi - aLo) * (gHi + d2 - d1) / d4;
                    else: # degrade to bisection interpolation
                        aMi = (aLo + aHi) / 2.0;
                else:  # degrade to bisection interpolation
                    aMi = (aLo + aHi) / 2.0; 
            else:  # bisection interpolation 
                aMi = (aLo + aHi) / 2.0;       
            fMi = self.fLineFunc(aMi);
            gMi = None;
            if(not (self.ifSufficientDecrease(aMi, fMi) and (fMi < fLo))):  (aHi, fHi, gHi) = (aMi, fMi, gMi);
            else:
                gMi = self.gLineFunc(aMi);
                if(self.ifCurvatureCondition(aMi, gMi)): return (aMi, fMi, gMi, self.getXvec(aMi), self.getExitInfo());  # exit: normal 
                if(gMi * (aHi - aLo) >= 0.0): (aHi, fHi, gHi) = (aLo, fLo, gLo);
                (aLo, fLo, gLo) = (aMi, fMi, gMi);
            #------------------------------------------------------------------------------ 
            if(self.ifPrint):
                print('[Wolfe Line Search]:[iter {0:^4}] zoom[aLo={1:<15.6e}, aHi={2:<15.6e}], fLo={3:<15.6e}'.format(self.iter, aLo, aHi, fLo));
            #------------------------------------------------------------------------------ 
            self.iter += 1;
            if(self.iter >= self.maxIter):
                self.maxIterReachedFlag = True;
                if(self.ifShowWarning): 
                    print('[Wolfe Line Search]: *WARNING* [iter {0:^4}] maximal iteration reached'.format(self.iter));
                print 'exit: max iter reached'
                return (aMi, fMi, gMi if gMi is not None else self.gLineFunc(aMi), self.getXvec(aMi), self.getExitInfo());  # exit: max iter reached
            #------------------------------------------------------------------------------ 
        return;

def wolfeLineSearch(fFunc, gFunc, xBegVec, dirVec, c1=1e-4, c2=0.9, maxIter=50, initStepLen=1.0, ifEnforceCubic=False, ifPrint=False, ifShowWarning=False):
    wls = WolfeLineSearch(fFunc, gFunc, xBegVec, dirVec, c1, c2, maxIter, initStepLen, ifEnforceCubic, ifPrint, ifShowWarning);
    (aVal, fVal, gVal, xEndVec, exitInfo) = wls.search();
    #------------------------------------------------------------------------------ 
    if(ifPrint):
        print('[Wolfe Line Search]: Condition Satisfied = {0}, aVal = {1:<15.6e}, fVal = {2:<15.6e}, gVal = {3:<15.6e}'.format(wls.ifStrongWolfeSatisfy(aVal, fVal, gVal), aVal, fVal, gVal));
        printMat(xEndVec, 'xEndVec', decor='e');
    #------------------------------------------------------------------------------ 
    return (aVal, fVal, gVal, xEndVec, exitInfo);

def ifStop(fOld, f, g): return (abs(fOld - f) / f < 1e-3 or abs(g) < 1e-6);

NOT_FOLD = True;
#===============================================================================
# Exact Line Search
#===============================================================================
def _goldenSectionSearch(fFunc, a, b, eps=_eps, maxIterNum=50):
    warnings = set();
    flag = 0;
    #------------------------------------------------------------------------------ 
    tau = (math.sqrt(5.0) - 1.0) / 2.0;
    x1 = a + (1.0 - tau) * (b - a);
    x2 = a + tau * (b - a);
    f1 = fFunc(x1);
    f2 = fFunc(x2);
    k = 0;
    while(b - a > eps):
        if(f1 > f2):
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + tau * (b - a);
            f2 = fFunc(x2);
        else:
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + (1.0 - tau) * (b - a);
            f1 = fFunc(x1);
        k += 1;
        if(k >= maxIterNum): break;
    #------------------------------------------------------------------------------
    if(k >= maxIterNum):
        flag = 1;
        warnings.add('maxIterReached');
    exitInfo = {'iterNum': k, 'warning': warnings, 'flag': flag}; 
    x = (a + b) / 2.0;
    return (x, exitInfo);

def goldenSectionSearch(fFunc, a, b, eps=_eps, maxIterNum=50):
    '''[goldenSectionSearch]: golden search line search
    
    reference: Sci. Comp. (M. H.) Algo. 6.1. P271
    '''
    warnings = set();
    flag = 0;
    iterNum = 0;
    #------------------------------------------------------------------------- 
    while(True):
        (x, ei) = _goldenSectionSearch(fFunc, a, b, eps, maxIterNum=50);
        if(x - a < eps):
            (a, b) = (min(a - 2.0 * (b - a), a - 2.0 * eps), max(a + 0.1 * (b - a), a + 2.0 * eps));
        elif(b - x < eps):
            (a, b) = (min(b - 0.1 * (b - a), b - 2.0 * eps), max(b + 2.0 * (b - a), b + 2.0 * eps));
        else: break;
        iterNum += 1;
        if(iterNum >= maxIterNum): break;
    #------------------------------------------------------------------------------
    if(iterNum >= maxIterNum):
        flag = 1;
        warnings.add('maxIterReached');
    exitInfo = {'iterNum': iterNum, 'warning': warnings, 'flag': flag}; 
    return (x, exitInfo);

def parabolicInterpolationSearch(fFunc, x0, x1, x2, eps=_eps, maxIterNum=50):
    '''[parabolicInterpolationSearch]: successive parabolic 
    interpolation. Convergence is not guaranteed.
    
    reference: Sci. Comp. (M. H.) P273.
    '''
    warnings = set();
    flag = 0;
    iterNum = 0;
    #------------------------------------------------------------------------- 
    f0 = fFunc(x0);
    f1 = fFunc(x1);
    f2 = fFunc(x2);
    while(True):
        p = ((x1 - x0) ** 2) * (f1 - f2) - ((x1 - x2) ** 2) * (f1 - f0);
        q = (x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0);
        if(ifZeroNum(q)):
            flag = 1;
            warnings.add('numericError');
            break;
        dx = -p / q / 2;
        (x0, f0) = (x2, f2);
        (x2, f2) = (x1, f1);
        (x1, f1) = (x1 + dx, fFunc(x1 + dx));
        if(abs(dx) < eps): break;
        iterNum += 1;
        if(iterNum >= maxIterNum):
            flag = 1;
            warnings.add('maxIterReached'); 
            break;
    #------------------------------------------------------------------------------
    exitInfo = {'iterNum': iterNum, 'warning': warnings, 'flag': flag}; 
    return (x1, exitInfo);

def _cubicInterpolation(fFunc, gFunc, x0, x1, eps=_eps, maxIterNum=50):
    '''[_cubicInterpolation]: cubic interpolation.
    requires x0 < x1 
    
    reference: Num. Opt. (J. N.) P59
    '''
    warnings = set();
    flag = 0;
    iterNum = 0;
    #------------------------------------------------------------------------- 
    if(x0 >= x1): (x0, x1) = (x1, x0);
    (f0, g0) = (fFunc(x0), gFunc(x0));
    (f1, g1) = (fFunc(x1), gFunc(x1));
    while(True):
        d1 = g0 + g1 - 3.0 * (f0 - f1) / (x0 - x1);
        d2 = math.sqrt(d1 ** 2 - g0 * g1);
        x = x1 - (x1 - x0) * (g1 + d2 - d1) / (g1 - g0 + 2.0 * d2);
        (f, g) = (fFunc(x), gFunc(x));
        if(abs(g) < eps): break;
        elif(g > 0): (x1, f1, g1) = (x, f, g);
        else: (x0, f0, g0) = (x, f, g);       
        iterNum += 1;
        if(iterNum >= maxIterNum):
            flag = 1;
            warnings.add('maxIterReached'); 
            break;
    #------------------------------------------------------------------------------
    exitInfo = {'iterNum': iterNum, 'warning': warnings, 'flag': flag, 'f':f, 'g':g}; 
    return (x, exitInfo);

def cubicInterpolation(fFunc, gFunc, a, b, eps=_eps, maxIterNum=50):
    '''[cubicInterpolation]: cubic interpolation.
    requires a < b 
    
    reference: Num. Opt. (J. N.) P59
    '''
    warnings = set();
    flag = 0;
    iterNum = 0;
    #------------------------------------------------------------------------- 
    while(True):
        (x, ei) = _cubicInterpolation(fFunc, gFunc, a, b, eps=eps, maxIterNum=50);
        if(x - a < eps):
            (a, b) = (min(a - 2.0 * (b - a), a - 2.0 * eps), max(a + 0.1 * (b - a), a + 2.0 * eps));
        elif(b - x < eps):
            (a, b) = (min(b - 0.1 * (b - a), b - 2.0 * eps), max(b + 2.0 * (b - a), b + 2.0 * eps));
        else: break;
        iterNum += 1;
        if(iterNum >= maxIterNum): break;
    #------------------------------------------------------------------------------
    if(iterNum >= maxIterNum):
        flag = 1;
        warnings.add('maxIterReached');
    exitInfo = {'iterNum': iterNum, 'warning': warnings, 'flag': flag, 'f': ei['f'], 'g': ei['g']}; 
    return (x, exitInfo);

if __name__ == '__main__':    
    pass
