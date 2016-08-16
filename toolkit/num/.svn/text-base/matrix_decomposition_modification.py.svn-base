'''
Created on May 7, 2014

@author: xwang95
'''
from toolkit.num.algebra import mulPermMatVec, forwardBackwardSub, dotVecVec, \
    cloneVec, catVecLst, getInvPermIdx, setMatCol, getSubMat, zeroes, getMatCol, \
    sizeSquareMat, luDecomp, cloneMat, randomMat, randomVec, subVecVec, \
    mulMatVec, ifZero, transposeMat, cbind, getSubVec, getVecNorm, \
    _householderReflectorPostApply, genGivensRotationArgs, \
    _givensRotationPreApply, _givensRotationPostApply, inverseGivensRotationArgs, \
    qrHouseholderDecomp, sizeMat, sizeVec, mulVecMat, ldlDecomp, \
    linSolvePosDefMat, rbind, mulPermMatPermMat, mulInvDiagMatVec, printMat, \
    mulDiagMatVec
from random import randint
from toolkit.num.arithmetic import ifZeroNum, sgn
import sys

class LuDecompModifier(object):
    '''
    It takes a square matrix as input.
    Modifying a matrix by substitute a column vector and maintaining its LU
    decomposition. LU decomposition is used to solve linear equation of the
    matrix or its transpose.
    '''
    mat = None;  # a square matrix
    n = None;  # dim
    
    lMat = None;
    uMat = None;
    rowIdx = None;
    invRowIdx = None;
    lLst = None;
    
    refreshPeriod = 10;
    modifyIterNum = 0;
    
    def _linSolveL(self, vec):
        x = mulPermMatVec(self.rowIdx, vec);
        x = forwardBackwardSub(tMat=self.lMat, vec=x, ifForward=True,
                               transpose=False, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        for (idx, invIdx, lLastRow) in self.lLst:
            x = mulPermMatVec(idx, x);
            x[self.n - 1] -= dotVecVec(lLastRow[0:self.n - 1], x[0:self.n - 1]);
        return x;
    
    def _linSolveU(self, vec):
        x = forwardBackwardSub(tMat=self.uMat, vec=vec, ifForward=False,
                               transpose=False, rowIdx=None, colIdx=None,
                               ifOverwrite=False);
        for (idx, invIdx, lLastRow) in reversed(self.lLst):
            x = mulPermMatVec(invIdx, x);
        return x;
    
    def linSolveMat(self, vec):
        x = self._linSolveL(vec);
        x = self._linSolveU(x);
        return x;
    
    def _linSolveUTranspose(self, vec):
        x = cloneVec(vec);
        for (idx, invIdx, lLastRow) in self.lLst:
            x = mulPermMatVec(idx, x);
        x = forwardBackwardSub(tMat=self.uMat, vec=x, ifForward=True,
                               transpose=True, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        return x;
    
    def _linSolveLTranspose(self, vec):
        x = cloneVec(vec);
        for (idx, invIdx, lLastRow) in reversed(self.lLst):
            x[self.n - 1] /= lLastRow[self.n - 1];
            for i in reversed(range(self.n - 1)):
                x[i] -= lLastRow[i] * x[self.n - 1];
            x = mulPermMatVec(invIdx, x);
        x = forwardBackwardSub(tMat=self.lMat, vec=x, ifForward=False,
                               transpose=True, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        x = mulPermMatVec(self.invRowIdx, x);
        return x;
    
    def linSolveMatTranspose(self, vec):
        x = self._linSolveUTranspose(vec);
        x = self._linSolveLTranspose(x);
        return x;
    
    def refresh(self):
        (self.lMat, self.uMat, self.rowIdx) = luDecomp(mat=self.mat,
                                                                ifRowIdx=True);
        self.invRowIdx = getInvPermIdx(self.rowIdx);
        self.lLst = [];
        self.modifyIterNum = 0;
        return;
    
    def modifyColumn(self, c, vec):
        setMatCol(self.mat, c, vec);
        if(self.modifyIterNum == self.refreshPeriod):
            self.refresh();
            return;
        self.modifyIterNum += 1;
        vec = self._linSolveL(vec);
        for (idx, invIdx, lLastRow) in self.lLst: c = invIdx[c];
        curIdx = catVecLst(range(c), range(c + 1, self.n), c);
        curInvIdx = getInvPermIdx(curIdx);
        uMat = self.uMat;
        setMatCol(uMat, c, vec);
        uMat = getSubMat(uMat, rowIdx=curIdx, colIdx=curIdx);
        curLLastRow = zeroes(self.n);
        for j in range(self.n - 1):
            curLLastRow[j] = (uMat[self.n - 1][j] - 
                              sum([curLLastRow[i] * uMat[i][j] 
                                   for i in range(j)])) / uMat[j][j];
            uMat[self.n - 1][j] = 0.0;
        curLLastRow[self.n - 1] = 1.0;
        uMat[self.n - 1][self.n - 1] -= dotVecVec(curLLastRow[0:self.n - 1],
                            getMatCol(uMat, self.n - 1, beg=0, end=self.n - 1));
        self.lLst.append((curIdx, curInvIdx, curLLastRow));
        self.uMat = uMat;
        return;
        
    def __init__(self, mat, refreshPeriod=10):
        self.mat = cloneMat(mat);
        self.n = sizeSquareMat(mat);
        self.refreshPeriod = refreshPeriod;
        self.refresh();
        return;

class QrDecompModifier(object):
    mat = None;
    m = None;
    
    n = None;
    qMat = None;
    rMat = None;
    colIdx = None;
    rank = None;
    
    updatedQColIdx = None;
    refreshPeriod = 10;
    modifyIterNum = 0;
    
    def addColumn(self, vec):
        vec0 = mulVecMat(vec, self.qMat);
        vec1 = vec0[0:self.n];
        vec2 = vec0[self.n:self.m];
        alpha = -sgn(vec2[0]) * getVecNorm(vec2);
        vec2[0] -= alpha;
        if(ifZeroNum(alpha)): return False;  # column not added
        self.mat = cbind(self.mat, vec);
        if(self.modifyIterNum == self.refreshPeriod):  # refresh
            self.refresh();
            return True;
        self.n += 1;
        self.colIdx.append(self.n - 1);
        self.rank += 1;
        self.rMat = cbind(self.rMat, catVecLst(vec1, alpha,
                                               zeroes(self.m - self.n)));
        _householderReflectorPostApply(self.qMat, vec2);
        self.updatedQColIdx = range(self.n - 1, self.m);
        return True;
    
    def delColumn(self, c):
        self.mat = getSubMat(self.mat, colIdx=[j for j in range(self.n) 
                                               if j != c]);
        if(self.modifyIterNum == self.refreshPeriod):  # refresh
            self.refresh();
            return;
        rc = getInvPermIdx(self.colIdx)[c];
        self.colIdx = [j if(j < c) else (j - 1) 
                       for j in self.colIdx if(j != c)];
        self.rMat = getSubMat(self.rMat, colIdx=[j for j in range(self.n)
                                                 if(j != rc)]);
        self.n -= 1;
        self.rank -= 1;
        for j in range(rc, min(self.n, self.m - 1)):
            grArgs = genGivensRotationArgs(j, j + 1, self.rMat[j][j],
                                           self.rMat[j + 1][j]);
            if(grArgs is None): continue;
            _givensRotationPreApply(self.rMat, *grArgs,
                                    colIdx=range(j, self.n));
            _givensRotationPostApply(self.qMat,
                                     *inverseGivensRotationArgs(grArgs));
        self.updatedQColIdx = ([] if (rc == min(self.n, self.m - 1)) 
                               else range(rc, min(self.n, self.m - 1) + 1));
        return;
    
    def getYMat(self): return getSubMat(self.qMat, colIdx=range(self.n));
    
    def getUpdatedYMatColIdx(self): return [j for j in self.updatedQColIdx 
                                            if j < self.n];
                                            
    def getZMat(self, colIdx=None): 
        if(colIdx is None): colIdx = range(self.m - self.n);
        return getSubMat(self.qMat, colIdx=[self.n + i for i in colIdx]);
    
    def getUpdatedZMatColIdx(self): return [j for j in self.updatedQColIdx
                                            if j >= self.n];

    def refresh(self):
        (self.m, self.n) = sizeMat(self.mat);
        (self.qMat, self.rMat,
         self.colIdx, self.rank) = qrHouseholderDecomp(self.mat,
                                            ifColPivot=True, ifShowRank=True);
        self.updatedQColIdx = range(self.m);
        self.modifyIterNum = 0;
        return;
    
    def __init__(self, mat, refreshPeriod=10):
        self.mat = cloneMat(mat);
        self.refreshPeriod = refreshPeriod;
        self.refresh();
        return;

class LdlDecompModifier(object):
    mat = None;  # positive definite matrix 
    n = None;  # dim
    lMat = None;
    dVec = None;
    pIdx = None;
    
    refreshPeriod = 10;
    modifyIterNum = 0;
    
    def expand(self, pos, vec):
        if(len(vec) != self.n + 1): return False;
        pIdx = catVecLst([i for i in range(self.n + 1) if(i != pos)], pos);
        invPIdx = getInvPermIdx(pIdx);
        z = vec[0:self.n];
        x = vec[self.n];
        self.mat = getSubMat(rbind(cbind(self.mat, z), vec),
                             rowIdx=invPIdx, colIdx=invPIdx);
        if(self.modifyIterNum == self.refreshPeriod):
            self.refresh();
            return True;
        pz = mulPermMatVec(self.pIdx, z);        
        y = forwardBackwardSub(tMat=self.lMat, vec=pz, ifForward=True,
                               transpose=False, rowIdx=None, colIdx=None,
                               ifOverwrite=False);
        y = mulInvDiagMatVec(self.dVec, y);
        d = x - dotVecVec(y, mulDiagMatVec(self.dVec, y));
        self.lMat = rbind(cbind(self.lMat, zeroes(self.n, 1)),
                          catVecLst(y, 1.0));
        self.dVec = catVecLst(self.dVec, d);
        self.pIdx = mulPermMatPermMat(catVecLst(self.pIdx, self.n), pIdx);
        self.n += 1;
        return True;
    
    def linSolve(self, vec):
        if(len(vec) != self.n): return None;
        x = linSolvePosDefMat(ldl=(self.lMat, self.dVec),
                              vec=mulPermMatVec(self.pIdx, vec));
        x = mulPermMatVec(getInvPermIdx(self.pIdx), x);
        return x;
    
    def refresh(self):
        self.n = sizeSquareMat(self.mat);
        (self.lMat, self.dVec) = ldlDecomp(mat=self.mat, ifDiagVec=True,
                                           checkSymmetry=False);
        self.pIdx = range(self.n);
        self.modifyIterNum = 0;
        return;
    
    def __init__(self, mat, refreshPeriod=10):
        self.mat = mat;
        self.refreshPeriod = refreshPeriod;
        self.refresh();
        return;
