'''
Created on Feb 15, 2014

@author: xwang95
'''
from toolkit.num.algebra import addVecVec, mulMatVec, getVecNorm, dotVecVec, mulNumVec, linSolve;
from toolkit.num.calculus import gradientFunc, hessianFunc;
from toolkit.num.arithmetic import ifZeroNum, _eps;
import math;
from toolkit.num.algebra import printMat, ifPositiveDefinite
from func.yaml.ypath import self_seg

class TrustRegion(object):
    '''[TrustRegion]
    
    references: Num. Opt. (J. N.) Algo. 4.1. P. 69
    '''
    eta = 0.15;
    radius = None;
    maxRadius = None;
    xBeg = None;
    x = None;
    f = None;
    g = None;
    maxIterNum = None;
    ifPrint = None;
    ifShowWarning = None;
    iterNum = 0;
    ifUpdate = None;
    exitInfo = {};
    
    def __init__(self): return;  # abstract method    
    def mSolve(self): return;  # abstract method
    def rhoFunc(self): return;  # abstract method
    def ifStop(self): return;  # abstract method
    def update(self): return;  # abstract method
    def restart(self): return;  # abstract method
        
    def iteration(self):
        pReachBoundary = self.mSolve();
        rho = self.rhoFunc();  # (self.ptCur.f - self.ptNew.f) / (self.ptCur.m - self.ptNew.m);
        if(rho < 0.25): self.radius *= 0.25;
        elif((rho > 0.75) and pReachBoundary): self.radius = min(2 * self.radius, self.maxRadius);
        if(rho > self.eta): self.update(); 
        else: self.restart();
        if(self.ifPrint):
            print('[Trust Region]:[iter {0:^4}] f={1:<15.6e}, r={2:<15.6e}, rho={3:<15.6g}, updated={4}'.format(self.iterNum, self.f, self.radius, rho, self.ifUpdate));
            printMat(self.x, 'x', decor='e')
        self.iterNum += 1;
        return;
    
    def solve(self):
        while(True):
            self.iteration();
            if(self.ifStop()): break;
        self.exitInfo['iterNum'] = self.iterNum;
        if(self.radius == self.maxRadius): self.exitInfo['warning'].add('maxRadiusReached');
        return (self.x, self.f, self.g, self.exitInfo);

class CauchyPointTrustRegion(TrustRegion):
    '''[CauchyPointTrustRegion]: It takes the point along the gradient direction
    that gets the smallest value within the radius as the next point to move, which
    is called as Cauchy Point.
    This method can work even when the model is not positive definite.
    
    references: Num. Opt. (J. N.) Algo. 4.2. P. 71 
    '''
    fFunc = None;
    gFunc = None;
    hFunc = None;
    mulHessGradFunc = None;
    
    p = None;
    pReachBoundary = None;
    
    x = None;
    f = None;
    g = None;
    gnCur = None;
    gbgCur = None;
    
    xNew = None;
    fNew = None;
    
    fDiff = None;
    mDiff = None;
    rho = None;
    def __init__(self, xBeg, fFunc, gFunc=None, hFunc=None, mulHessGradFunc=None, maxRadius=None, initRadius=None, maxIterNum=None, ifPrint=False, ifShowWarning=False):
        self.maxIterNum = maxIterNum;
        self.ifPrint = ifPrint;
        self.ifShowWarning = ifShowWarning;
        self.xBeg = xBeg;  # from base class
        self.fFunc = fFunc;
        if(gFunc is None): self.gFunc = lambda x: gradientFunc(self.fFunc, x);
        else: self.gFunc = gFunc;
        if(hFunc is None): self.hFunc = lambda x:  hessianFunc(self.fFunc, x);
        else: self.hFunc = hFunc;
        if(mulHessGradFunc is None): self.mulHessGradFunc = lambda x, g = None, h = None: mulMatVec(self.hFunc(x) if h is None else h, self.gFunc(x) if g is None else g);
        else: self.mulHessGradFunc = lambda x, g = None, h = None: mulHessGradFunc(x);
        (self.p, self.pReachBoundary) = (None, None);
        (self.x, self.f, self.g, self.gnCur, self.gbgCur) = (self.xBeg, None, None, None, None);
        (self.xNew, self.fNew) = (None, None);
        (self.fDiff, self.mDiff, self.rho) = (None, None, None);
        if(maxRadius is None):  # from base class
            g = self.gFunc(self.xBeg);
            gn = getVecNorm(g);
            self.maxRadius = 10 * gn;
            (self.g, self.gnCur) = (g, gn);
        else: self.maxRadius = maxRadius;
        if(initRadius is None): self.radius = 0.1 * self.maxRadius;  # from base class
        else: self.radius = initRadius; 
        self.exitInfo['warning'] = set();
        self.exitInfo['flag'] = 0;
        return;
    
    def mSolve(self): return self._getCauchyPoint();
    def rhoFunc(self): return self.rho;
    def ifStop(self):
        if((self.maxIterNum is not None) and (self.maxIterNum <= self.iterNum)):
            self.exitInfo['warning'].add('maxIterReachedFlag');
            self.exitInfo['flag'] = 1; 
            return True; 
        if(self.ifUpdate and (abs(self.fDiff) / max(abs(self.f), _eps) < 1e-3)): return True;
        return False;
    def update(self):
        self.ifUpdate = True;
        (self.x, self.f, self.g, self.gnCur, self.gbgCur) = (self.xNew, self.fNew, None, None, None);
        (self.xNew, self.fNew) = (None, None);
        return;
    def restart(self):
        self.ifUpdate = False;
        (self.xNew, self.fNew) = (None, None);
        return;
    
    def _getCauchyPoint(self):
        '''[getCauchyPoint]: compute the Cauchy Point in closed form.
        references: Num. Opt. (J.N.) Chp 4.1. P71
        '''
        if(self.f is None): self.f = self.fFunc(self.x);
        if(self.g is None): self.g = self.gFunc(self.x);
        if(self.gnCur is None): self.gnCur = getVecNorm(self.g);
        if(self.gbgCur is None): self.gbgCur = dotVecVec(self.g, self.mulHessGradFunc(x=self.x, g=self.g));
        
        if(self.gbgCur <= 0.0): tau = 1.0;
        else: tau = min(self.gnCur ** 3 / (self.radius * self.gbgCur), 1.0);
        beta = -tau * self.radius / self.gnCur;
        
        self.p = mulNumVec(beta, self.g);
        self.pReachBoundary = (ifZeroNum(1.0 - tau));
        self.xNew = addVecVec(self.x, self.p);
        self.fNew = self.fFunc(self.xNew);
        self.fDiff = self.fNew - self.f;
        self.mDiff = beta * (self.gnCur ** 2) + 0.5 * self.gbgCur * (beta ** 2);
        self.rho = 1.0 if(ifZeroNum(self.mDiff)) else (self.fDiff / self.mDiff);
        return self.pReachBoundary;                
    
class DoglegTrustRegion(TrustRegion):
    '''[DoglegTrustRegion]: Dogleg is the path which across two particular points: 
    the point that is along the gradient having the smallest model value and the one
    that gets the smallest model value. The constraint of the radius is then taken
    into accounts to find the farthest point along the dog let.
    
    Dogleg works only when the Hassian (or approximated Hassian) is positive definite,
    because the global minimum exists when the quadratic model is positive definite.
    For safeguard, when this is not satisfied and the grad' * Hassian * grad < 0, 
    we use the point along the negative gradient that is on the radius circle. Otherwise,
    the dogleg method fails. 
    
    references:  Num. Opt. (J. N.) P. 73
    '''
    fFunc = None;
    gFunc = None;
    hFunc = None;
    
    def __init__(self, xBeg, fFunc, gFunc=None, hFunc=None, maxRadius=None, initRadius=None, maxIterNum=None, ifPrint=False, ifShowWarning=False):
        self.maxIterNum = maxIterNum;
        self.ifPrint = ifPrint;
        self.ifShowWarning = ifShowWarning;
        self.xBeg = xBeg;  # from base class
        self.fFunc = fFunc;
        if(gFunc is None): self.gFunc = lambda x: gradientFunc(self.fFunc, x);
        else: self.gFunc = gFunc;
        if(hFunc is None): self.hFunc = lambda x:  hessianFunc(self.fFunc, x);
        else: self.hFunc = hFunc;
        
        self.radius = initRadius;
        self.maxRadius = maxRadius;
        self.xBeg = xBeg;
        self.x = xBeg;
        
        self.exitInfo['warning'] = set();
        self.exitInfo['flag'] = 0;
        return;
        
    def mSolve(self): return self._dogleg();
    def rhoFunc(self): return (1.0 if ifZeroNum(self.mDiff) else self.fDiff / self.mDiff);
    def ifStop(self):
        if((self.maxIterNum is not None) and (self.maxIterNum <= self.iterNum)):
            self.exitInfo['warning'].add('maxIterReachedFlag');
            self.exitInfo['flag'] = 1; 
            return True; 
        if(self.ifUpdate and (abs(self.fDiff) / max(abs(self.f), _eps) < 1e-3)): return True;
        return False;
    def update(self):
        self.ifUpdate = True;
        (self.x, self.f) = (self.xNew, self.fNew);
        (self.g, self.b, self.gg, self.ghg) = (None, None, None, None);
        (self.pU, self.pB, self.pUpU, self.pBpB, self.pBpU, self.p, self.pReachBoundary) = (None, None, None, None, None, None, None);
        (self.xNew, self.fNew, self.mDiff) = (None, None, None);
        return; 
    def restart(self): 
        self.ifUpdate = False;
        (self.p, self.pReachBoundary) = (None, None);
        (self.xNew, self.fNew, self.mDiff) = (None, None, None);
        return;
    
    x = None;
    f = None;
    g = None;
    h = None;
    gg = None;
    ghg = None;
    pU = None;  # unconstrained step: steepest descent direction minimizer
    pB = None;  # full step: unconstrained minimizer
    pUpU = None;
    pBpB = None;
    pBpU = None;
    p = None;
    pReachBoundary = None;
    xNew = None;
    fNew = None;
    fDiff = None;
    mDiff = None;
    def _dogleg(self):
        if(self.f is None): self.f = self.fFunc(self.x);
        if(self.g is None): self.g = self.gFunc(self.x);
        if(self.h is None): self.h = self.hFunc(self.x);
        if(self.ifShowWarning and (not ifPositiveDefinite(self.h))):
            print('[Dogleg Trust Region]:  Hassian not positive definite, run abort!');
            self.mDiff = 1.0;
            self.fNew = self.f;
            return False;
        if(self.gg is None): self.gg = dotVecVec(self.g, self.g);
        if(self.ghg is None): self.ghg = dotVecVec(self.g, mulMatVec(self.h, self.g));
        if(self.pU is None): self.pU = mulNumVec(-self.gg / self.ghg, self.g);
        if(self.pB is None): self.pB = mulNumVec(-1.0, linSolve(mat=self.h, vec=self.g));  # full step is not valid when h is not positive definite
        if(self.pUpU is None): self.pUpU = dotVecVec(self.pU, self.pU);
        if(self.pBpB is None): self.pBpB = dotVecVec(self.pB, self.pB);
        if(self.pBpU is None): self.pBpU = dotVecVec(self.pB, self.pU);
        if((self.radius ** 2) <= self.pUpU): (self.p, self.pReachBoundary) = (mulNumVec(self.radius / math.sqrt(self.pUpU), self.pU), True);
        elif((self.radius ** 2) >= self.pBpB): (self.p, self.pReachBoundary) = (self.pB, False);
        else:
            a = self.pUpU + self.pBpB - 2.0 * self.pBpU;
            b = 2.0 * (self.pBpU - self.pBpB);
            c = self.pBpB - (self.radius ** 2);
            alpha = self._solveQuadEquation(a, b, c);
            (self.p, self.pReachBoundary) = (addVecVec(mulNumVec(alpha, self.pU), mulNumVec(1 - alpha, self.pB)), True);
        self.xNew = addVecVec(self.x, self.p);
        self.fNew = self.fFunc(self.xNew);
        self.fDiff = self.fNew - self.f;
        self.mDiff = dotVecVec(self.g, self.p) + 0.5 * dotVecVec(self.p, mulMatVec(self.h, self.p));
        return self.pReachBoundary;
    
    def _solveQuadEquation(self, a, b, c):  # x2 solution
        d = math.sqrt((b ** 2) - 4.0 * a * c);
        return (-b - d) / (2.0 * a);

def trustRegion(fFunc=None, gFunc=None, hFunc=None, mulHessGradFunc=None, xBeg=None, maxRadius=None, initRadius=None, method=None, maxIterNum=None, ifPrint=False, ifShowWarning=False):
    '''[trustRegion]: trust region method
    args:
    returns:
        x
        f
        g
        ei: exitInfo
    '''
    if(method == 'dogleg'): tr = DoglegTrustRegion(xBeg, fFunc, gFunc, hFunc, maxRadius, initRadius, maxIterNum, ifPrint, ifShowWarning);
    elif(method == 'cauchy_point'): tr = CauchyPointTrustRegion(xBeg, fFunc, gFunc, hFunc, mulHessGradFunc, maxRadius, initRadius, maxIterNum, ifPrint, ifShowWarning);
    (x, f, g, ei) = tr.solve();
    if(g is None): g = gFunc(x);
    return (x, f, g, ei);

if(__name__ == '__main__'):
    pass;
