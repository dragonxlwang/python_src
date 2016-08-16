'''
Created on Apr 20, 2014

@author: xwang95
'''
from toolkit.num.algebra import forwardBackwardSub, mulPermMatVec, dotVecVec, \
    getInvPermIdx, getSubMat, setMatCol, zeroes, randomMat, luDecomp, \
    getPermMatIdx, randomVec, subMatMat, mulMatVec, printMat, subVecVec, \
    getMatCol, cloneVec, transposeMat, getVecNorm, mulNumVec, stdBasis, ifZero, \
    sizeMat, ones, ifNonPosVec, cloneMat
import math;
from toolkit.num.arithmetic import binSgn
import sys

class Simplex(object):
    '''
        [flags]
            0: iteration continues
            1: optimal point found
            2: problem is unbounded
            3: max iter num reached
    '''
    #------------------------------------------------------------- problem scale
    n = None;
    m = None;
    bMatDecompPeriod = 10;
    #---------------------------------------------------------------- parameters
    #--------------------------------------------- constant after initialization
    cVec = None;
    bVec = None;
    aMatTranspose = None;
    initBIdxLst = None;
    #---------------------------------------------------------- primal variables
    #----------------------------------------------------- update each iteration
    xbVec = None;
    cbVec = None;
    #------------------------------------------------------------ dual variables
    #----------------------------------------------------- update each iteration
    snVec = None;
    lambdaVec = None;
    #----------------------------------------------------- steepest edge pricing
    gammaVec = None;
    #--------------------------------------------------------------------- basis
    #----------------------------------------------------- update each iteration
    bIdxLst = None;  # basis
    nIdxLst = None;  # nonbasic    
    bMatDecompL = None;
    bMatDecompU = None;
    bMatDecompPIdx = None;
    bMatDecompPInvIdx = None;
    bMatDecompLst = None;
    #--------------------------------------------------------- running parameter
    iterNum = None;
    pricingMethod = None;
    maxInitIterNum = None;
    maxIterNum = None;
    
    
    
    def __init__(self, cVec, bVec, aMat=None, aMatTranspose=None,
                 initBIdxLst=None, pricingMethod='Dantzig',
                 maxInitIterNum=100, maxIterNum=1000, bMatDecompPeriod=10):
        '''
        Constructor
        '''
        if(aMatTranspose is not None): self.aMatTranspose = aMatTranspose;
        else: self.aMatTranspose = transposeMat(aMat);
        self.cVec = cVec;
        self.bVec = bVec;
        self.initBIdxLst = initBIdxLst;
        self.pricingMethod = pricingMethod;
        self.maxInitIterNum = maxInitIterNum;
        self.maxIterNum = maxIterNum;  
        self.bMatDecompPeriod = bMatDecompPeriod;
        return;
    
    def linSolveBMatL(self, vec):
        x = mulPermMatVec(self.bMatDecompPIdx, vec);
        x = forwardBackwardSub(tMat=self.bMatDecompL, vec=x, ifForward=True,
                               transpose=False, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        for (pIdx, invPIdx, lLastRow) in self.bMatDecompLst:
            x = mulPermMatVec(pIdx, x);
            x[self.m - 1] -= dotVecVec(lLastRow[0:self.m - 1], x[0:self.m - 1]);
        return x;
    
    def linSolveBMatU(self, vec):
        x = forwardBackwardSub(tMat=self.bMatDecompU, vec=vec, ifForward=False,
                               transpose=False, rowIdx=None, colIdx=None,
                               ifOverwrite=False);
        for (pIdx, invPIdx, lLastRow) in reversed(self.bMatDecompLst):    
            x = mulPermMatVec(invPIdx, x);
        return x;
    
    def linSolveBMat(self, vec):
        x = self.linSolveBMatL(vec);
        x = self.linSolveBMatU(x);
        return x;
    
    def linSolveBMatUTranspose(self, vec):
        x = cloneVec(vec);
        for (pIdx, invPIdx, lLastRow) in self.bMatDecompLst: 
            x = mulPermMatVec(pIdx, x);
        x = forwardBackwardSub(tMat=self.bMatDecompU, vec=x, ifForward=True,
                               transpose=True, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        return x;
    
    def linSolveBMatLTranspose(self, vec):
        x = cloneVec(vec);
        for (pIdx, invPIdx, lLastRow) in reversed(self.bMatDecompLst):
            x[self.m - 1] /= lLastRow[self.m - 1];
            for i in reversed(range(self.m - 1)): 
                x[i] -= lLastRow[i] * x[self.m - 1];
            x = mulPermMatVec(invPIdx, x);
        x = forwardBackwardSub(tMat=self.bMatDecompL, vec=x, ifForward=False,
                               transpose=True, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        x = mulPermMatVec(self.bMatDecompPInvIdx, x);
        return x;
    
    def linSolveBMatTranspose(self, vec):
        x = self.linSolveBMatUTranspose(vec);
        x = self.linSolveBMatLTranspose(x);
        return x;
    
    def updateBMatDecomp(self, p, col):
        col = self.linSolveBMatL(col);
        for (pIdx, invPIdx, lLastRow) in self.bMatDecompLst: p = invPIdx[p];
        curPIdx = range(p) + range(p + 1, self.m) + [p];
        curInvPIdx = getInvPermIdx(curPIdx);
        uMat = self.bMatDecompU;
        setMatCol(uMat, p, col);
        uMat = getSubMat(uMat, rowIdx=curPIdx, colIdx=curPIdx);
        curLLastRow = zeroes(self.m);
        for j in range(self.m - 1):
            curLLastRow[j] = (uMat[self.m - 1][j] - 
                              sum([curLLastRow[i] * uMat[i][j] 
                                   for i in range(j)])) / uMat[j][j];
            uMat[self.m - 1][j] = 0.0;
        curLLastRow[self.m - 1] = 1.0;
        uMat[self.m - 1][self.m - 1] -= dotVecVec(curLLastRow[0:self.m - 1],
                            getMatCol(uMat, self.m - 1, beg=0, end=self.m - 1));
        self.bMatDecompLst.append((curPIdx, curInvPIdx, curLLastRow));
        self.bMatDecompU = uMat;
        return;
    
    def iterWithDantzigPricing(self):
        '''[iterWithDantzigPricing]:
        returns:
            0: iteration continues
            1: optimal point found
            2: problem is unbounded
        '''
        self.lambdaVec = self.linSolveBMatTranspose(self.cbVec);
        self.snVec = [self.cVec[n] - 
                      dotVecVec(self.lambdaVec, self.aMatTranspose[n]) 
                      for n in self.nIdxLst];
        #------------------------------------------ q: entering, Dantzig Pricing
        (q, s) = (None, 0.0);
        for i in range(self.n - self.m):
            if(self.snVec[i] < s): (q, s) = (i, self.snVec[i]);
        #--------------------------------------------- exit: optimal point found
        if(q is None): return 1;           
        dVec = self.linSolveBMat(self.aMatTranspose[self.nIdxLst[q]]);
        #-------------------------------------------- exit: problem is unbounded
        if(ifNonPosVec(dVec)): return 2;
        #------------------------------------------------------------ p: leaving
        (p, alpha) = (None, None);
        for i in range(self.m):
            if(dVec[i] <= 0): continue;
            beta = self.xbVec[i] / dVec[i];
            if(alpha is None or beta < alpha): (p, alpha) = (i, beta);
        #---------------------------------------------------------------- update
        self.xbVec = subVecVec(self.xbVec, mulNumVec(alpha, dVec));
        self.xbVec[p] = alpha;
        self.cbVec[p] = self.cVec[self.nIdxLst[q]];
        self.updateBMatDecomp(p, self.aMatTranspose[self.nIdxLst[q]]);
        (self.bIdxLst[p], self.nIdxLst[q]) = (self.nIdxLst[q], self.bIdxLst[p]);
        return 0;
    
    def iterWithSteepestEdgePricing(self):
        '''[iterWithSteepestEdgePricing]:
        returns:
            0: iteration continues
            1: optimal point found
            2: problem is unbounded
        '''
        self.lambdaVec = self.linSolveBMatTranspose(self.cbVec);
        self.snVec = [self.cVec[n] - 
                      dotVecVec(self.lambdaVec, self.aMatTranspose[n]) 
                      for n in self.nIdxLst];
        #------------------------------------ q: entering, Steepest Edge Pricing
        (q, s) = (None, 0.0);
        if(self.gammaVec is None):
            self.gammaVec = [getVecNorm(self.linSolveBMat(
                                        self.aMatTranspose[n])) ** 2 + 1.0 
                             for n in self.nIdxLst];
        for i in range(self.n - self.m):
            ss = self.snVec[i] / math.sqrt(self.gammaVec[i]);
            if(ss < s): (q, s) = (i, ss);
        #--------------------------------------------- exit: optimal point found
        if(q is None): return 1;
        dVec = self.linSolveBMat(self.aMatTranspose[self.nIdxLst[q]]);
        #-------------------------------------------- exit: problem is unbounded
        if(ifNonPosVec(dVec)): return 2;
        #------------------------------------------------------------ p: leaving
        (p, alpha) = (None, None);
        for i in range(self.m):
            if(dVec[i] <= 0):  continue;
            beta = self.xbVec[i] / dVec[i];
            if(alpha is None or beta < alpha): (p, alpha) = (i, beta);
        #------------------------------------------------------- pricing update1
        eVec = stdBasis(self.m, p);
        ddVec = self.linSolveBMatTranspose(dVec);
        rVec = self.linSolveBMatTranspose(eVec);
        raq = dotVecVec(rVec, self.aMatTranspose[self.nIdxLst[q]]);
        for i in range(self.n - self.m):
            if(i != q):
                aiVec = self.aMatTranspose[self.nIdxLst[i]];
                rai = dotVecVec(rVec, aiVec);
                ddai = dotVecVec(ddVec, aiVec);
                self.gammaVec[i] += (-2.0 * rai / raq * ddai
                                     + ((rai / raq) ** 2) * self.gammaVec[q]);
        #---------------------------------------------------------------- update
        self.xbVec = subVecVec(self.xbVec, mulNumVec(alpha, dVec));
        self.xbVec[p] = alpha;
        self.cbVec[p] = self.cVec[self.nIdxLst[q]];
        self.updateBMatDecomp(p, self.aMatTranspose[self.nIdxLst[q]]);
        (self.bIdxLst[p], self.nIdxLst[q]) = (self.nIdxLst[q], self.bIdxLst[p]);
        #------------------------------------------------------- pricing update2
        self.gammaVec[q] = getVecNorm(self.linSolveBMat(
                            self.aMatTranspose[self.nIdxLst[q]])) ** 2 + 1.0;
        return 0;
    
    def refreshBMat(self):
        bMat = transposeMat([self.aMatTranspose[i] for i in self.bIdxLst]);
        (self.bMatDecompL, self.bMatDecompU, pMat) = luDecomp(bMat);
        self.bMatDecompPIdx = getPermMatIdx(pMat);
        self.bMatDecompPInvIdx = getInvPermIdx(self.bMatDecompPIdx);
        self.bMatDecompLst = [];
        return;
    
    def debug(self):
        xVec = zeroes(self.n);
        for i in range(self.m): xVec[self.bIdxLst[i]] = self.xbVec[i];
        f = dotVecVec(xVec, self.cVec);
        print('basis: {0}'.format(self.bIdxLst));
        printMat(xVec, 'xVec');
        printMat(f, 'f');
        printMat(subVecVec(self.bVec,
                           mulMatVec(transposeMat(self.aMatTranspose), xVec)),
                 'z_feasibility');
        print('iter={0}'.format(self.iterNum));
        return;
    
    def simplexSolve(self, maxIterNum):
        '''[simplexSolve]:
        returns:
            0: iteration continues
            1: optimal point found
            2: problem is unbounded
            3: max iter num reached
        '''
        self.iterNum = 0;
        if(self.pricingMethod == 'Dantzig'): 
            iterMethod = self.iterWithDantzigPricing;
        elif(self.pricingMethod == 'SteepestEdge'):
            iterMethod = self.iterWithSteepestEdgePricing;
            self.gammaVec = None;
        while(True):
            self.iterNum += 1;
            if(self.iterNum % self.bMatDecompPeriod == 0): self.refreshBMat();
            flag = iterMethod();
            if(flag != 0): return flag;
            if(self.iterNum > maxIterNum): return 3;
        return;
        
    def solve(self, ifPrint=False):
        (n, m) = sizeMat(self.aMatTranspose);
        
        if(self.initBIdxLst is None):
            phase1n = n + m;
            phase1m = m
            phase1cVec = zeroes(n) + ones(m);
            phase1bVec = cloneVec(self.bVec);
            phase1aMatTranspose = cloneMat(self.aMatTranspose);
            for i in range(m): 
                vec = zeroes(m);
                vec[i] = binSgn(self.bVec[i]);
                phase1aMatTranspose.append(vec);
        phase2n = n;
        phase2m = m;        
        phase2cVec = cloneVec(self.cVec);
        phase2bVec = cloneVec(self.bVec);
        phase2aMatTranspose = cloneMat(self.aMatTranspose);
        if(self.initBIdxLst is None):
            self.n = phase1n;
            self.m = phase1m;
            self.cVec = phase1cVec;
            self.bVec = phase1bVec;
            self.aMatTranspose = phase1aMatTranspose;
            self.bIdxLst = range(phase1n - phase1m, phase1n);
            self.nIdxLst = range(phase1n - phase1m);
            self.refreshBMat();
            self.xbVec = self.linSolveBMat(self.bVec);
            self.cbVec = [self.cVec[i] for i in self.bIdxLst];
            phase1Flag = self.simplexSolve(self.maxInitIterNum);
            for i in range(self.m):
                if(self.bIdxLst[i] < self.n - self.m): continue;
                if(self.xbVec[i] != 0.0):
                    print('exit: problem infeasible');
                    return;
            if(ifPrint):            
                print('phase one finished:');
                self.debug();
                print('flag={0}'.format(phase1Flag));
        else: self.bIdxLst = self.initBIdxLst;
        self.n = phase2n;
        self.m = phase2m;
        self.cVec = phase2cVec;
        self.bVec = phase2bVec;
        self.aMatTranspose = phase2aMatTranspose;
        bSet = set([i for i in self.bIdxLst if i < self.n]);
        for i in range(self.n):
            if(len(bSet) == self.m): break;
            if(i in bSet): continue;
            vi = self.aMatTranspose[i];
            for j in bSet:
                vj = self.aMatTranspose[j];
                vi = subVecVec(vi, mulNumVec(dotVecVec(vi, vj) 
                                             / dotVecVec(vj, vj), vj));
            if(not ifZero(vi)): bSet.add(i);
        self.bIdxLst = list(bSet);
        self.nIdxLst = [i for i in range(self.n) if(i not in self.bIdxLst)];
        self.refreshBMat();
        self.xbVec = self.linSolveBMat(self.bVec);
        self.cbVec = [self.cVec[i] for i in self.bIdxLst];
        phase2Flag = self.simplexSolve(self.maxIterNum);        
        if(ifPrint):
            print('phase two finished:');
            self.debug();
            print('flag={0}'.format(phase2Flag));
        
        xVec = zeroes(self.n);
        for i in range(self.m): xVec[self.bIdxLst[i]] = self.xbVec[i];
        f = dotVecVec(xVec, self.cVec);
        exitInfo = {'iterNum' : self.iterNum, 'flag2': phase2Flag};
        if(self.initBIdxLst is None): exitInfo['flag1'] = phase1Flag;
        return (xVec, f, self.bIdxLst, exitInfo);

if(__name__ == '__main__'):
    cVec = [-2, -3, -4, 0, 0];
    bVec = [10, 15];
    aMatTranspose = [[3, 2],
                     [2, 5],
                     [1, 3],
                     [1, 0],
                     [0, 1]];
    initBIdxLst = None;
    
    cVec = [-1, -2, 1, 0, 0, 0];
    bVec = [14, 28, 30];
    aMatTranspose = [[2, 4, 2],
                     [1, 2, 5],
                     [1, 3, 5],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = [3, 4, 5];
    pricingMethod = 'SteepestEdge';
    pricingMethod = 'Dantzig';
    
    cVec = [-4, -6, 0, 0, 0];
    bVec = [11, 27, 90];
    aMatTranspose = [[-1, 1, 2],
                     [1, 1, 5],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = [2, 3, 4];
    
    cVec = [-2, 1, -2, 0, 0, 0];
    bVec = [10, 20, 5];
    aMatTranspose = [[2, 1, 0],
                     [1, 2, 1],
                     [0, -2, 2],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = [3, 4, 5];
    
    cVec = [-3, -2, -1, 0, 0];
    bVec = [30, 60, 40];
    aMatTranspose = [[4, 2, 1],
                     [1, 3, 2],
                     [1, 1, 3],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = None;
    simplex = Simplex(cVec, bVec,
                      aMatTranspose=aMatTranspose,
                      initBIdxLst=initBIdxLst,
                      pricingMethod=pricingMethod);
    (x, f, b, ei) = simplex.solve();
    printMat(x);
    print(f);
    
    cVec = [-11, -16, -15, 0, 0, 0];
    bVec = [12000, 4600, 2400];
    aMatTranspose = [[1, 2.0 / 3.0, 0.5],
                     [2, 2.0 / 3.0, 1.0 / 3.0],
                     [1.5, 1, 0.5],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = None;  # [3, 4, 5];
    
    cVec = [-100000, -40000, -18000, 0, 0, 0, 0];
    bVec = [18200, 10, 0, 0];
    aMatTranspose = [[2000, 0, -0.5, -0.9],
                     [600, 1, -0.5, 0.1],
                     [300, 0, 0.5, 0.1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]];
    initBIdxLst = None;  # [3, 4, 5];
    
