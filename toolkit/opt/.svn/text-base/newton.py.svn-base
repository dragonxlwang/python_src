'''
Created on Mar 8, 2014

@author: xwang95
'''
from toolkit.num.algebra import sizeVec, dotVecVec, getVecNorm, zeroes, \
    mulNumVec, addVecVec, printMat, mulMatVec, minusVec, subVecVec, \
    choleskyDecomp, cbind, diagMat, rbind, minusMat, transposeMat, \
    forwardBackwardSub, ifZero, addVecLst, cloneVec, mulNumMat, mulMatMat, \
    subMatMat, sizeMat, getMatCol, eye, linSolve, addMatMat, symIdfDecomp, \
    linSolveSymIdfMat, _luDecomp, ldlDecomp, linSolvePosDefMat, stdBasis, \
    extendBasis, qrDecomp, ifZeroVec, qrHouseholderDecomp, getSubMat, mulMatLst, \
    mulVecMat, luDecomp, mulPermMatVec, getInvPermIdx, invMat, det, randomMat
import math
from toolkit.num.calculus import mulHessianVecApprox, hessianFunc
from toolkit.opt.line_search import wolfeLineSearch
from toolkit.num.arithmetic import _eps
import sys

def lineSearchNewtonConjugateGradient(fFunc, gFunc, xBegVec, epsHessvec=_eps,
                                      cgThreshold=0.0, epsDecrease=_eps,
                                      epsGrad=_eps, maxIter=50, ifPrint=False):
    '''[lineSearchNewtonConjugateGradient]: Line Search Newton-CG
    
    params:
        fFunc:        function evaluation
        gFunc:        gradient evaluation
        xBegVec:      initial point x0
        epsHessvec:   small step size for Hessian-vector 
                      approximation. (P170)
        cgThreshold:  threshold for curvature criteria,
                      restart if pHp < threshold
        epsDecrease:  stop criteria for small improvement
        epsGrad:      stop criteria for small gradient
        maxIter:      max number of iteration
        ifPrint:      print additional information
    
    returns:
        x:           minimizer
        f:           function value
        exitInfo:    exit info, warning, flag, iterNum
                      
    references: Num. Opt. (J.N.) Algo. 7.1. P169. 
    '''
    warnings = set();
    flag = 0;
    dr = 1.0;
    gn = 1.0;
    #---------------------------------------------------------------- initialize
    k = 0;
    x = xBegVec;
    n = sizeVec(x);
    f = fFunc(x);
    fOld = None;
    while(True):  # newton step
        k += 1;
        #------------------------------------------- exit: max iteration reached
        if(k >= maxIter):  # exit: max iteration reached
            flag = 1;
            warnings.add('maxIterReachedFlag');
            break;
        #------------------------------------------------------------- exit: end
        g = gFunc(x);
        gg = dotVecVec(g, g);
        gn = math.sqrt(gg);
        tol = min(0.5, math.sqrt(gn)) * gn;
        
        fOld = f;
        r = g;
        rr = gg;
        p = minusVec(r);
        z = zeroes(n);
        d = p;        
        for t in range(n):  # exit CG: n iteration reached
            if(math.sqrt(rr) < tol): break;  # exit CG: normal 
            hp = mulHessianVecApprox(fFunc, gFunc, x, p, h=epsHessvec);
            php = dotVecVec(p, hp);
            if(php <= cgThreshold): break;  # exit CG: curvature not met
            alpha = rr / php;
            z = addVecVec(z, mulNumVec(alpha, p));
            r = addVecVec(r, mulNumVec(alpha, hp));
            rrNew = dotVecVec(r, r);
            beta = rrNew / rr;
            p = addVecVec(minusVec(r), mulNumVec(beta, p));
            (rr, d) = (rrNew, z);
        #----------------------------------------------------------- line search
        (a, f, gLn, x, ei) = wolfeLineSearch(fFunc, gFunc, x, d, c1=1e-4, c2=0.9,
                                             initStepLen=1.0, ifShowWarning=ifPrint);
        #-------------------------------------------------------- decrease ratio
        dr = None if fOld is None else (abs(fOld - f) / (_eps if f == 0.0 else f));
        #----------------------------------------------------------- print block
        if(ifPrint):
            print('[lineSearchNewtonConjugateGradient]: [iteration {0}] ' \
                  'f={1:<15.6e}, gLine={2:<15.6e}, gNorm={3:<15.6e}, aLine={4:<15.6e}, ' \
                  'tol={5:<15.6e}'.format(k, f, gLn, gn, a, tol));
            printMat(x, 'x', decor='e');
            printMat(g, 'g', decor='e');
            printMat(d, 'd', decor='e');
            printMat(dotVecVec(g, d) / (getVecNorm(g) * getVecNorm(d)),
                     'cos', decor='e');
            if(len(ei['warning']) > 0): 
                print('               Line Search Warnings: ' \
                      '{0}'.format(','.join(ei['warning'])));
            print('                Stop Criteria Check: ' \
                  'delta_f/f={0:<15.6e} (eps={1:<15.6e}), gn={2:<15.6e} ' \
                  '(eps={3:<15.6e})'.format(dr, epsDecrease, gn, epsGrad));
        #--------------------------------------------------------- stop criteria
        if((fOld is not None and dr < epsDecrease) or gn < epsGrad): 
            break;  # exit: too small progress
    #---------------------------------------------------------------- terminated
    exitInfo = {'iterNum': k, 'warning': warnings, 'flag': flag};
    #--------------------------------------------------------------- print block
    if(ifPrint):
        print(80 * '-');
        print('[lineSearchNewtonConjugateGradient]: iterNum={0}, '\
              'f={1:<15.6e}'.format(k, f));
        print('                Stop Criteria Check: ' \
              'delta_f/f={0:<15.6e} (eps={1:<15.6e}), gn={2:<15.6e} ' \
              '(eps={3:<15.6e})'.format(dr, epsDecrease, gn, epsGrad));
        printMat(x, 'x', decor='e');
        printMat(g, 'g', decor='e');
        if(len(exitInfo['warning']) > 0): 
            print('                           Warnings: ' \
                  '{0}'.format(','.join(exitInfo['warning'])));
    return (x, f, exitInfo);

def lbfgs(fFunc, gFunc, xBegVec, cacheLen=10, epsDecrease=_eps, epsGrad=_eps,
          maxIter=50, ifPrint=False):
    '''[lbfgs]: Limited-memory Broyden-Fletcher-Goldfarb-Shanno methods (quasi-newton)
    
    params:
        fFunc:        function evaluation
        gFunc:        gradient evaluation
        xBegVec:      initial point x0
        cacheLen:     size of cache to store (s, y) 
        epsDecrease:  stop criteria for small improvement
        epsGrad:      stop criteria for small gradient
        maxIter:      max number of iteration
        ifPrint:      print additional information
    
    returns:
        x:           minimizer
        f:           function value
        g:           gradient vector
        exitInfo:    exit info, warning, flag, iterNum
                      
    references: Num. Opt. (J.N.) Algo. 7.4-7.5. P179. 
    ''' 
    warnings = set();
    flag = 0;
    dr = 1.0;
    gn = 1.0;
    #---------------------------------------------------------------- initialize
    m = cacheLen;
    queue = [None for i in range(m)];
    alphaLst = [None for i in range(m)];
    (k, pntr) = (0, 0);
    x = xBegVec;
    while(True):
        if(k == 0):  # first iteration
            f = fFunc(x);
            g = gFunc(x);
            p = minusVec(g);
            #-------------------------------------------------- start at optimum
            if(getVecNorm(g) < epsGrad):
                flag = 1;
                warnings.add('startAtOptimumFlag');
        else:
            q = g;
            for i in range(min(m, k)):
                (s, y, rho) = queue[(pntr - i) % m];
                alpha = rho * dotVecVec(s, q);
                q = subVecVec(q, mulNumVec(alpha, y));                
                alphaLst[i] = alpha;
            (s, y, rho) = queue[pntr];
            gamma = dotVecVec(s, y) / dotVecVec(y, y);
            r = mulNumVec(gamma, q);
            for i in reversed(range(min(m, k))):
                alpha = alphaLst[i];
                (s, y, rho) = queue[(pntr - i) % m];
                beta = rho * dotVecVec(y, r);
                r = addVecVec(r, mulNumVec(alpha - beta, s));
            p = minusVec(r);
        (a, fNew, gLnNew, xNew, ei) = wolfeLineSearch(fFunc, gFunc, x, p,
                                                      c1=1e-4, c2=0.9, maxIter=50,
                                                      initStepLen=1.0,
                                                      ifEnforceCubic=True,
                                                      ifPrint=False,
                                                      ifShowWarning=ifPrint);
        gNew = gFunc(xNew);
        (k, pntr) = (k + 1, (pntr + 1) % m);
        (s, y) = (subVecVec(xNew, x), subVecVec(gNew, g));
        sy = dotVecVec(s, y);
        #--------------------------------------------------------- stop criteria
        dr = abs(fNew - f) / (_eps if fNew == 0.0 else fNew);
        gn = getVecNorm(gNew);
        if(dr < epsDecrease or gn < epsGrad): break;  # exit: too small progress
        if(k >= maxIter):
            flag = 1;
            warnings.add('maxIterReachedFlag');
            break;  # exit: max iteration reached
        if(sy == 0.0):
            flag = 1;
            warnings.add('curvatureNotMetFlag');
            break;  # exit: curvature condition not met
        #----------------------------------------------------------- update s, y
        queue[pntr] = (s, y, 1.0 / sy);
        (x, f, g) = (xNew, fNew, gNew);
        #----------------------------------------------------------- print block
        if(ifPrint):
            print('[lbfgs]: [iteration {0}] f={1:<15.6e}, gLine={2:<15.6e}, ' \
                  'gNorm={3:<15.6e}, aLine={4:<15.6e}'.format(k, f, gLnNew, gn, a));
            printMat(x, 'x', decor='e');
            printMat(g, 'g', decor='e');
            if(len(ei['warning']) > 0): 
                print('        Line Search Warnings: ' \
                      '{0}'.format(','.join(ei['warning'])));
            print('        Stop Criteria Check: '   \
                  'delta_f/f={0:<15.6e} (eps={1:<15.6e}), gn={2:<15.6e} ' \
                  '(eps={3:<15.6e})'.format(dr, epsDecrease, gn, epsGrad));
    #---------------------------------------------------------------- terminated
    (x, f, g, exitInfo) = (xNew, fNew, gNew,
                           {'iterNum': k, 'warning': warnings, 'flag': flag});
    #--------------------------------------------------------------- print block
    if(ifPrint):
        print(80 * '-');
        print('[lbfgs]: iterNum={0}, f={1:<15.6e}'.format(k, f));
        print('        Stop Criteria Check: delta_f/f={0:<15.6e} ' \
              '(eps={1:<15.6e}), gn={2:<15.6e} ' \
              '(eps={3:<15.6e})'.format(dr, epsDecrease, gn, epsGrad));
        printMat(x, 'x', decor='e');
        printMat(g, 'g', decor='e');
        if(len(exitInfo['warning']) > 0): 
            print('        Warnings: {0}'.format(','.join(exitInfo['warning'])));
    return (x, f, g, exitInfo);

class LBFGSB(object):
    m = 10;
    n = None;
    fFunc = None;
    gFunc = None;
    boxConstLst = None;
    boxConstTbl = None;
    method = None;
    epsDecrease = None;
    epsGrad = None;
    maxIter = None;
    ifPrint = None;
    #----------------------------------------------------- compact newton matrix
    queue = None;
    qPntr = None;
    qLen = None;
    #----------------------------------------------------------- M decomposition
    mMatLuDecomp = None;
    #-------------------------------------------------------------- cauchy point
    xcp = None;
    activeConstSet = None;
    c = None;
    #----------------------------------------------------- subspace minimization
    p = None;
    #----------------------------------------------------------------- iteration
    x = None;
    f = None;
    g = None;
    dr = None;
    gn = None;
    #--------------------------------------------------------------------- solve
    iterNum = None;
    exitInfo = None;
    flag = None;
        
    def compactNewtonUpdate(self, s, y):
        #--------------------------------------- delete oldest info if necessary
        if(self.qLen < self.m): self.qLen += 1;
        self.qPntr = (self.qPntr + 1) % self.m;
        self.queue[self.qPntr]['s'] = s;
        self.queue[self.qPntr]['y'] = y;
        self.queue[self.qPntr]['syLst'] = [
                dotVecVec(s, self.queue[(self.qPntr - i) % self.m]['y']) 
                for i in range(self.qLen)];
        self.queue[self.qPntr]['yyLst'] = [
                dotVecVec(y, self.queue[(self.qPntr - i) % self.m]['y'])
                for i in range(self.qLen)];
        self.queue[self.qPntr]['ssLst'] = [
                dotVecVec(s, self.queue[(self.qPntr - i) % self.m]['s'])
                for i in range(self.qLen)];
        self.queue[self.qPntr]['sy'] = self.queue[self.qPntr]['syLst'][0];
        self.queue[self.qPntr]['theta'] = (
                self.queue[self.qPntr]['yyLst'][0] / 
                self.queue[self.qPntr]['syLst'][0]);
        return;
    
    def mMatDecomp(self):
        '''[mMatDecomp]: Cholesky-like decomposition for W matrix (middle 
        matrix), cf. L-BFGS-B Eq. 3.2 and rep-q-Newton Eq. 2.25
        '''
        theta = self.queue[self.qPntr]['theta']; 
        dLst = [self.queue[(self.qPntr - i) % self.m]['sy']
                for i in range(self.qLen)];
        lt1Mat = diagMat([math.sqrt(x) for x in reversed(dLst)]);
        lt2Mat = minusMat(lt1Mat);
        rt1Mat = zeroes(self.qLen, self.qLen);
        lb2Mat = rt1Mat;
        lb1Mat = zeroes(self.qLen, self.qLen);
        for i in range(self.qLen):
            for j in range(1, self.qLen - i):
                sy1 = self.queue[(self.qPntr - i) % self.m]['syLst'][j];
                sy2 = self.queue[(self.qPntr - i - j) % self.m]['sy'];
                lb1Mat[self.qLen - 1 - i][self.qLen - 1 - i - j] = \
                    - sy1 * math.sqrt(1.0 / sy2);
        rt2Mat = minusMat(transposeMat(lb1Mat));
        jjMat = zeroes(self.qLen, self.qLen);
        for i in range(self.qLen):
            for j in range(0, self.qLen - i):
                jjMat[self.qLen - 1 - i][self.qLen - 1 - i - j] = \
                    theta * self.queue[(self.qPntr - i) % self.m]['ssLst'][j];
                jjMat[self.qLen - 1 - i - j][self.qLen - 1 - i] = \
                    jjMat[self.qLen - 1 - i][self.qLen - 1 - i - j];
        jjMat = subMatMat(jjMat, mulMatMat(lb1Mat, rt2Mat));
        jMat = choleskyDecomp(jjMat, checkSymmetry=False);
        rb1Mat = jMat;        
        rb2Mat = transposeMat(rb1Mat);
        mlMat = rbind(cbind(lt1Mat, rt1Mat), cbind(lb1Mat, rb1Mat));
        muMat = rbind(cbind(lt2Mat, rt2Mat), cbind(lb2Mat, rb2Mat));
        self.mMatLuDecomp = (mlMat, muMat);
        return;
    
    def mMatPostMulVec(self, v):
        '''[mMatPostMulVec]: it solves the equation m^(-1) * y = v, or 
        alternatively, it computes y = m * v
        
        complexity: O(m^2)
        '''
        (mlMat, muMat) = self.mMatLuDecomp;
        y = forwardBackwardSub(mlMat, v, ifForward=True, ifOverwrite=False);
        y = forwardBackwardSub(muMat, y, ifForward=False, ifOverwrite=True);
        return y;
    
    def mMatMulMat(self, mat):
        (mMat, nMat) = sizeMat(mat);
        xMat = [self.mMatPostMulVec(getMatCol(mat, c)) for c in range(nMat)];
        xMat = transposeMat(xMat);
        return xMat;
        
    def wMatPreMulVec(self, v):
        '''[wMatPreMulVec]: it computes y = v' * w
        
        complexity: O(2m * n)
        '''
        theta = self.queue[self.qPntr]['theta'];
        v1 = [dotVecVec(v, self.queue[(self.qPntr - i) % self.m]['y'])
              for i in range(self.qLen)];
        v2 = [dotVecVec(v, self.queue[(self.qPntr - i) % self.m]['s']) * theta
              for i in range(self.qLen)];
        y = v1[::-1] + v2[::-1];
        return y;
    
    def wMatRowSelect(self, r):
        '''[wMatPreMulVec]: it computes y = e_r' * w
        
        complexity: O(2m)
        '''
        theta = self.queue[self.qPntr]['theta'];
        v1 = [self.queue[(self.qPntr - i) % self.m]['y'][r]
              for i in range(self.qLen)];
        v2 = [self.queue[(self.qPntr - i) % self.m]['s'][r] * theta
              for i in range(self.qLen)];
        y = v1[::-1] + v2[::-1];
        return y;
    
    def zwMatPreMulVec(self, v, zLst):
        '''[zwMatPreMulVec]: it computes y = v' * (z' * w)
        
        complexity: O(2m * n)
        '''
        theta = self.queue[self.qPntr]['theta'];
        t = len(zLst);
        v1 = [sum([v[j] * self.queue[(self.qPntr - i) % self.m]['y'][zLst[j]] 
                   for j in range(t)])
              for i in range(self.qLen)];
        v2 = [sum([v[j] * self.queue[(self.qPntr - i) % self.m]['s'][zLst[j]]
                   for j in range(t)]) * theta
              for i in range(self.qLen)];
        y = v1[::-1] + v2[::-1];
        return y;
    
    def genCauchyPointCompute(self):
        '''[genCauchyPointCompute]: computes the cauchy point
        
        complexity: O(n) + O(m^2) * n_seg
        '''
        theta = self.queue[self.qPntr]['theta'];
        activeConstSet = set();
        piecewiseSegLst = [];
        for (i, lb, ub) in self.boxConstLst:
            if(-self.g[i] < 0 and lb is not None):
                t = (self.x[i] - lb) / self.g[i];
                if(ifZero(t)): t = 0.0;
                piecewiseSegLst.append((i, t, lb));
            elif(-self.g[i] > 0 and ub is not None):
                t = (self.x[i] - ub) / self.g[i];
                if(ifZero(t)): t = 0.0;
                piecewiseSegLst.append((i, t, ub));
        piecewiseSegLst = sorted(piecewiseSegLst, key=lambda x: x[1]);
        d = minusVec(self.g);
        j = 0;
        for (i, t, b) in piecewiseSegLst:
            if(t == 0.0): 
                (d[i], j) = (0.0, j + 1);
                activeConstSet.add(i);
            else: break;
        p = self.wMatPreMulVec(d);
        c = zeroes(2 * self.qLen);
        fp = -dotVecVec(d, d);
        fpp = -theta * fp - dotVecVec(p, self.mMatPostMulVec(p));
        tOld = 0.0;
        if(j < len(piecewiseSegLst)):
            (i, t, b) = piecewiseSegLst[j];
            dtEnd = t - tOld;
        dtMin = 0.0 if (fp == 0.0 and fpp == 0.0) else -fp / fpp;
        while(j < len(piecewiseSegLst) and dtMin > dtEnd):
            zi = b - self.x[i];
            c = addVecVec(c, mulNumVec(dtEnd, p));
            wi = self.wMatRowSelect(i);
            mwi = self.mMatPostMulVec(wi);
            gi = self.g[i];
            fp = (fp + dtEnd * fpp + (gi ** 2) + theta * zi * gi 
                  - gi * dotVecVec(mwi, c));
            fpp = (fpp - theta * (gi ** 2) - 2 * gi * dotVecVec(p, mwi)
                   - (gi ** 2) * dotVecVec(wi, mwi));
            p = addVecVec(p, mulNumVec(gi, wi));
            (d[i], j) = (0, j + 1);
            activeConstSet.add(i);
            tOld = t;
            if(j < len(piecewiseSegLst)):
                (i, t, b) = piecewiseSegLst[j];
                dtEnd = t - tOld;
            dtMin = 0.0 if (fp == 0.0 and fpp == 0.0) else -fp / fpp;
        dtMin = max(dtMin, 0.0);
        tcp = tOld + dtMin;
        xcp = cloneVec(self.x);
        for (i, t, b) in piecewiseSegLst: 
            xcp[i] = b if tcp > t else xcp[i] + tcp * d[i];
        c = addVecVec(c, mulNumVec(dtMin, p));
        (self.xcp, self.activeConstSet, self.c) = (xcp, activeConstSet, c);
        return;
    
    def directPrimalMethod(self):
        '''[directPrimalMethod]: solve the quadratic problem constrained
        by box constraints using direct primal method.
        
        complexity:  O(n) + O(m^3) + O(m^2 * t)
        '''
        theta = self.queue[self.qPntr]['theta'];
        self.p = [self.xcp[i] - self.x[i] for i in range(self.n)];
        zLst = [z for z in range(self.n) if(z not in self.activeConstSet)];
        #--------------------------------------- special case: no free variables
        if(len(zLst) == 0): return;
        mc = self.mMatPostMulVec(self.c);
        r = [self.g[zLst[i]] 
             + theta * (self.xcp[zLst[i]] - self.x[zLst[i]])
             - dotVecVec(self.wMatRowSelect(zLst[i]), mc)
             for i in range(len(zLst))];
        #----------------------------------------------------------- complete: r
        v = self.zwMatPreMulVec(r, zLst);
        v = self.mMatPostMulVec(v);
        #----------------------------------------------------------- complete: v
        zwMat = [self.wMatRowSelect(z) for z in zLst];
        nMat = mulNumMat(1.0 / theta, mulMatMat(transposeMat(zwMat), zwMat));
        nMat = subMatMat(eye(2 * self.qLen), self.mMatMulMat(nMat));
        #-------------------------------------------------------- complete: nMat
        v = linSolve(mat=nMat, vec=v);
        #----------------------------------------------------------- complete: v
        v = [dotVecVec(self.wMatRowSelect(z), v) / (theta ** 2) for z in zLst];
        d = minusVec(addVecVec(mulNumVec(1.0 / theta, r), v));
        #------------------------------------------ complete: d = - inv_Hess * r
        alpha = 1.0;  # find alpha satisfying box constraints w.r.t. zLst
        for i in range(len(zLst)):
            if zLst[i] in self.boxConstTbl:
                (z, lb, ub) = self.boxConstTbl[zLst[i]];
                if(d[i] < 0 and lb is not None):
                    alpha = min(alpha, (lb - self.xcp[z]) / d[i]);
                elif(d[i] > 0 and ub is not None):
                    alpha = min(alpha, (ub - self.xcp[z]) / d[i]);
        for i in range(len(zLst)): self.p[zLst[i]] += alpha * d[i];
        return;
    
    def conjugateGradientMethod(self):
        '''[conjugateGradientMethod]: solve the quadratic problem constrained
        by box constraints using conjugate gradient method.
        
        complexity: 
        ''' 
        theta = self.queue[self.qPntr]['theta'];
        self.p = [self.xcp[i] - self.x[i] for i in range(self.n)];
        zLst = [z for z in range(self.n) if(z not in self.activeConstSet)];
        #--------------------------------------- special case: no free variables
        if(len(zLst) == 0): return;
        mc = self.mMatPostMulVec(self.c);
        r = [self.g[zLst[i]] 
             + theta * (self.xcp[zLst[i]] - self.x[zLst[i]])
             - dotVecVec(self.wMatRowSelect(zLst[i]), mc)
             for i in range(len(zLst))];
        #----------------------------------------------------------- complete: r
        p = minusVec(r);
        rr = dotVecVec(r, r);
        rn = math.sqrt(rr);
        tol = min(0.1, math.sqrt(rn)) * rn;
        d = zeroes(len(zLst));
        for k in range(2 * self.qLen):
            if(math.sqrt(rr) < tol): break;
            alphaMax = None;
            for i in range(len(zLst)):
                if zLst[i] in self.boxConstTbl:
                    (z, lb, ub) = self.boxConstTbl[zLst[i]];
                    if(p[i] < 0 and lb is not None):
                        t = (lb - self.xcp[z] - d[i]) / p[i];
                    elif(p[i] > 0 and ub is not None):
                        t = (ub - self.xcp[z] - d[i]) / p[i];
                    else: continue;
                    alphaMax = t if alphaMax is None else min(alphaMax, t);
            #------------------------------------------------ complete: alphaMax
            zwp = self.zwMatPreMulVec(p, zLst);
            mzwp = self.mMatPostMulVec(zwp);
            zwmzwp = [dotVecVec(self.wMatRowSelect(z), mzwp) for z in zLst];  
            hp = subVecVec(mulNumVec(theta, p), zwmzwp);
            #------------------------------------------- complete: hp = Hess * p
            php = dotVecVec(p, hp);
            alpha = rr / php;
            if(alphaMax is not None and alpha > alphaMax):
                d = addVecVec(d, mulNumVec(alphaMax, p));
                break;
            else:
                d = addVecVec(d, mulNumVec(alpha, p));
                r = addVecVec(r, mulNumVec(alpha, hp));
                rrNew = dotVecVec(r, r);
                beta = rrNew / rr;
                p = addVecVec(minusVec(r), mulNumVec(beta, p));
                rr = rrNew;
        for i in range(len(zLst)): self.p[zLst[i]] += d[i];
        return;
    
    def lineSearch(self):
        '''[lineSearch]: perform line search to attempt to enforce strong Wolfe
        condition. It ensures box constraints are satisfied, but curvature
        condition might not be met.
        It requires that x + p is feasible and will be used as next point if line
        search result is not feasible and x + p is better than boundary point 
        along the direction p;
        complexity: O(n) + line saerch complexity
        '''
        alphaMax = None;
        for (i, lb, ub) in self.boxConstLst:
            if(self.p[i] < 0 and lb is not None): 
                t = (lb - self.x[i]) / self.p[i];
            elif(self.p[i] > 0 and ub is not None):
                t = (ub - self.x[i]) / self.p[i];
            else: continue;
            alphaMax = t if alphaMax is None else min(alphaMax, t);            
        (alpha, f, gLn, x, ei) = wolfeLineSearch(self.fFunc, self.gFunc, self.x,
                                                 self.p, c1=1e-4, c2=0.9,
                                                 maxIter=50, initStepLen=1.0,
                                                 ifEnforceCubic=True,
                                                 ifPrint=False,
                                                 ifShowWarning=False);
        if(alphaMax is not None and alpha > alphaMax):
            xAlphaMax = addVecVec(self.x, mulNumVec(alphaMax, self.p));
            fAlphaMax = self.fFunc(xAlphaMax);
            xSubSol = addVecVec(self.x, self.p);
            fSubSol = self.fFunc(xSubSol);
            if(fAlphaMax > fSubSol): (x, f) = (xSubSol, fSubSol);
            else: (x, f) = (xAlphaMax, fAlphaMax);
        g = self.gFunc(x);
        s = subVecVec(x, self.x);
        y = subVecVec(g, self.g);
        return (x, f, g, s, y);
    
    def iteration(self):
        if(self.iterNum == 0):  # first iteration
            self.p = minusVec(self.g);
            alpha = None;
            for (i, lb, ub) in self.boxConstLst:
                if(self.p[i] < 0 and lb is not None):
                    t = (lb - self.x[i]) / self.p[i];
                elif(self.p[i] > 0 and ub is not None):
                    t = (ub - self.x[i]) / self.p[i];
                else: continue;
                alpha = t if alpha is None else min(alpha, t);
            if(alpha is not None): self.p = mulNumVec(alpha, self.p);
        else:
            self.genCauchyPointCompute();
            if(self.method == 'direct-primal'): 
                self.directPrimalMethod();
            elif(self.method == 'conjugate-gradient'): 
                self.conjugateGradientMethod();
        (x, f, g, s, y) = self.lineSearch();
        dr = (self.f - f) / (_eps if f == 0.0 else f);
        #---------------------------------------- project gradient infinity norm
        #-------- norm is zero iff KKT satisfied, constraint optimization solved
        gn = 0.0;
        for i in range(self.n):
            xi = self.x[i] - self.g[i];
            if(i in self.boxConstTbl):
                (z, lb, ub) = self.boxConstTbl[i]; 
                if(lb is not None and xi < lb): xi = lb;
                elif(ub is not None and xi > ub): xi = ub;
            gn = max(gn, abs(xi - self.x[i]));          
        #--------------------------------------- update only if curvature is met
        if(dotVecVec(s, y) > 2.2e-16 * dotVecVec(y, y)): 
            self.compactNewtonUpdate(s, y);
        self.mMatDecomp();
        (self.x, self.f, self.g, self.dr, self.gn) = (x, f, g, dr, gn);
        return;
    
    def solve(self):
        self.iterNum = 0;
        self.flag = 0;
        warnings = set();
        while(True):
            self.iteration();
            self.iterNum += 1;
            if(self.dr < self.epsDecrease or self.gn < self.epsGrad): break;
            if(self.iterNum > self.maxIter):
                flag = 1;
                warnings.add('maxIterReached');
                break;
            #------------------------------------------------------- print block
            if(self.ifPrint):
                print('[L-BFGS-B]: [iteration {0}] ' \
                      'f={1:<15.6e},'.format(self.iterNum, self.f));
                print('            Stop Criteria Check: ' \
                      'dr={0:<15.6e} (eps={1:<15.6e}), ' \
                      'gn={2:<15.6e} (eps={3:<15.6e})'.format(self.dr,
                                    self.epsDecrease, self.gn, self.epsGrad));
                printMat(self.x, 'x', decor='e');
                printMat(self.g, 'g', decor='e');
        #----------------------------------------------------------- print block
        if(self.ifPrint):
            print(80 * '-');
            print('[L-BFGS-B]: [iteration {0}] ' \
                  'f={1:<15.6e},'.format(self.iterNum, self.f));
            print('            Stop Criteria Check: ' \
                  'dr={0:<15.6e} (eps={1:<15.6e}), ' \
                  'gn={2:<15.6e} (eps={3:<15.6e})'.format(self.dr,
                                self.epsDecrease, self.gn, self.epsGrad));
            printMat(self.x, 'x', decor='e');
            printMat(self.g, 'g', decor='e');
            if(len(warnings) > 0):
                print('            Warnings: ' \
                      '{0}'.format(','.join(warnings)));
        #------------------------------------------------------------- terminate
        self.exitInfo = {'iterNum': self.iterNum, 'warning': warnings,
                         'dr':self.dr, 'gn':self.gn};
        return (self.x, self.f, self.g, self.exitInfo);
    
    def __init__(self, fFunc, gFunc, x0, boxConstLst, method='direct-primal',
                 cacheLen=10, epsDecrease=_eps, epsGrad=1e-5, maxIter=50,
                 ifPrint=False):
        (self.m, self.n) = (cacheLen, len(x0));
        (self.fFunc, self.gFunc) = (fFunc, gFunc);
        self.boxConstLst = boxConstLst;
        self.boxConstTbl = {};
        for (i, lb, ub) in boxConstLst: self.boxConstTbl[i] = (i, lb, ub);
        self.method = method;
        (self.epsDecrease, self.epsGrad) = (epsDecrease, epsGrad);
        self.maxIter = maxIter;
        self.ifPrint = ifPrint;
        self.queue = [{} for i in range(cacheLen)];
        (self.qPntr, self.qLen) = (0, 0);
        (self.x, self.f, self.g) = (x0, fFunc(x0), gFunc(x0));
        return;

def lbfgsb(fFunc, gFunc, x0, boxConstLst, method='direct-primal',
           cacheLen=10, epsDecrease=_eps, epsGrad=1e-5, maxIter=50,
           ifPrint=False):
    inst = LBFGSB(fFunc, gFunc, x0, boxConstLst, method,
                  cacheLen, epsDecrease, epsGrad, maxIter,
                  ifPrint);
    (x, f, g, ei) = inst.solve();
    return (x, f, g, ei);

if __name__ == '__main__':
    pass
