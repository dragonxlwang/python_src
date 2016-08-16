'''
Created on Apr 30, 2014

@author: xwang95
'''
from toolkit.num.algebra import sizeVec, dotVecVec, getVecNorm, zeroes, \
    mulNumVec, addVecVec, printMat, mulMatVec, minusVec, subVecVec, \
    choleskyDecomp, cbind, diagMat, rbind, minusMat, transposeMat, \
    forwardBackwardSub, ifZero, addVecLst, cloneVec, mulNumMat, mulMatMat, \
    subMatMat, sizeMat, getMatCol, eye, linSolve, addMatMat, symIdfDecomp, \
    linSolveSymIdfMat, _luDecomp, ldlDecomp, linSolvePosDefMat, stdBasis, \
    extendBasis, qrDecomp, ifZeroVec, qrHouseholderDecomp, getSubMat, mulMatLst, \
    mulVecMat, luDecomp, mulPermMatVec, getInvPermIdx, invMat, det, randomMat, \
    linSolveLuDecompVec, ones, catVecLst, getSubVec, \
    linSolveSymIdfMatWithIterRefine, ifPosVec, BasisConstructor, ifSquareMat, \
    sizeSquareMat, mulDiagMatVec, mulInvDiagMatVec, linSolveLeastNormSol
import math
from toolkit.num.calculus import mulHessianVecApprox, hessianFunc
from toolkit.opt.line_search import wolfeLineSearch
from toolkit.num.arithmetic import _eps, ifZeroNum, minIdx
import sys
from toolkit.opt.active_set import Simplex
from toolkit.num.matrix_decomposition_modification import LdlDecompModifier, \
    QrDecompModifier
from label_denoise.utility import getVecNorm

class EqualQuadProg(object):
    n = None;
    m = None;
    cVec = None;
    gMat = None;
    bVec = None;
    aMat = None;
    
    xVec = None;
    lambdaVec = None;
    
    zgzLdlDm = None;
    aTrQrDm = None;
    
    def __init__(self, cVec=None, gMat=None, bVec=None, aMat=None):
        '''
        EQP: if not constriant: set bVec None
        IQP-active sub EQP: always set bVec None
        '''
        self.cVec = cVec;
        self.gMat = gMat;
        self.bVec = bVec;
        self.aMat = aMat;
        if(aMat is not None): (self.m, self.n) = sizeMat(aMat);
        return;
    
    def unconstraintOpt(self):
        self.xVec = linSolvePosDefMat(mat=self.gMat, vec=minusVec(self.cVec));
        return;
    
    def directMethod(self):
        '''
        solve the x and lambda by forming the KKT matrix (n * m) * (n * m), 
        factorizing it via symmetric indefinite block decomposition. it is 
        effecitive on many problems but i may be expensive whenL is more dense
        than the original coefficient matrix.
        '''
        if(self.bVec is None):  # special case: unconstraint
            self.unconstraintOpt();
            return;
        kMat = rbind(cbind(self.gMat, transposeMat(self.aMat)),
                     cbind(self.aMat, zeroes(self.m, self.m)));
        vec = self.cVec + minusVec(self.bVec);
        sol = linSolveSymIdfMat(mat=kMat, vec=vec);
        self.xVec = minusVec(sol[0:self.n]);
        self.lambdaVec = sol[self.n: self.n + self.m];
        return;
    
    def schurComplementMethod(self):
        '''
        it is recommended to use the Schur-complement method if G is positive
        definite and AG^{-1}A^{T} can be computed relatively cheaply (because
        G is easy to invert or because m is small relative to n)
        '''
        if(self.bVec is None):  # special case: unconstraint
            self.unconstraintOpt();
            return;
        gLdl = ldlDecomp(self.gMat, ifDiagVec=True);
        vec = linSolvePosDefMat(ldl=gLdl, vec=self.cVec);
        vec = addVecVec(mulMatVec(self.aMat, vec), self.bVec);
        agMat = [linSolvePosDefMat(ldl=gLdl, vec=self.aMat[i]) 
                 for i in range(self.m)];
        agaMat = [[dotVecVec(self.aMat[i], agMat[j]) 
                   for j in range(self.m)] 
                  for i in range(self.m)];
        self.lambdaVec = linSolvePosDefMat(mat=agaMat, vec=vec);
        vec = subVecVec(mulMatVec(transposeMat(self.aMat), self.lambdaVec),
                        self.cVec);
        self.xVec = linSolvePosDefMat(ldl=gLdl, vec=vec);
        return;
    
    def nullSpaceMethod(self):
        '''
        it is required that:
            1) A matrix has full row rank;
            2) Z^{T}GZ matrix is positive definite
        and thus this method has wider applicability than the Schur-complement 
        method. When it is more expensive to compute factors of G than to 
        compute the null-space matrix Z and the factors of Z^{T}GZ, null-space
        method is more preferable.        
        '''
        if(self.bVec is None):  # special case: unconstraint
            self.unconstraintOpt();
            return;
        (qMat, rMat, colIdx, rank) = qrHouseholderDecomp(
                                                mat=transposeMat(self.aMat),
                                                ifColPivot=True,
                                                ifShowRank=True);
        yMat = getSubMat(qMat, colIdx=range(rank));
        zMat = getSubMat(qMat, colIdx=range(rank, self.n));
        ayMat = mulMatMat(self.aMat, yMat);
        (ayLMat, ayUMat, ayRowIdx) = luDecomp(ayMat, ifRowIdx=True);
        py = linSolveLuDecompVec(ayLMat, ayUMat, ayRowIdx, self.bVec,
                                 ifTranspose=False);
        ypy = mulMatVec(yMat, py);
        if(rank == self.n):  # special case: full rank
            self.xVec = ypy;
            return;
        zgMat = mulMatMat(transposeMat(zMat), self.gMat);
        zgzMat = mulMatMat(zgMat, zMat);
        pz = linSolvePosDefMat(mat=zgzMat,
                               vec=minusVec(addVecVec(mulMatVec(zgMat, ypy),
                                                mulVecMat(self.cVec, zMat))));
        zpz = mulMatVec(zMat, pz);
        self.xVec = addVecVec(ypy, zpz);        
        self.lambdaVec = linSolveLuDecompVec(ayLMat, ayUMat, ayRowIdx,
                            mulVecMat(addVecVec(self.cVec,
                                                mulMatVec(self.gMat,
                                                          self.xVec)),
                                      yMat),
                            ifTranspose=True);
        return;
    
    def projectedConjugateGradientMethod(self):
        '''
        Algo 16.2 Projected CG Method.
        Solve with preconditioner H (=I) and projection matrix 
        P = Z(Z^{T}HZ)^{-1}Z^{-T} implicitly. The projection operation requires
        solving augmented system with symmetric indefinite factorization, and
        this can also be used to identify the starting point x: A*x = b.
        Linear solving sym. idf. matrix is done with iterative refinement.  
        '''
        if(self.bVec is None):  # special case: unconstraint
            self.unconstraintOpt();
            return;
#         def solveInitPointWithSimplex():
#             cVec = catVecLst(zeroes(self.n), ones(self.m));
#             initBIdxLst = range(self.n, self.n + self.m);
#             s = Simplex(cVec, self.bVec, aMat=self.aMat,
#                         initBIdxLst=initBIdxLst);
#             (xVec, f, bIdxLst, ei) = s.solve();
#             return xVec;
        def solveInitPointWithAugmentedSystem(ldl, mat):
            xVec = zeroes(self.n);
            vec = catVecLst(zeroes(self.n), self.bVec);
            xVec = linSolveSymIdfMatWithIterRefine(ldl, mat, vec);
            xVec = xVec[0:self.n];            
            return xVec;
        def formAugmentedSystem(hMat=None):
            if(hMat is None):
                hMat = eye(self.n);
            mat = rbind(cbind(eye(self.n), transposeMat(self.aMat)),
                      cbind(self.aMat, zeroes(self.m, self.m)));
            ldl = symIdfDecomp(mat);
            return (mat, ldl);
        def projectWithAugmentedSystem(ldl, vec, mat):
            vec = catVecLst(vec, zeroes(self.m));
            xVec = linSolveSymIdfMatWithIterRefine(ldl, mat, vec);
            xVec = xVec[0:self.n];
            return xVec;
        hMat = eye(self.n);
        (mat, ldl) = formAugmentedSystem(hMat);
        self.xVec = solveInitPointWithAugmentedSystem(ldl, mat);
        rVec = addVecVec(mulMatVec(self.gMat, self.xVec), self.cVec);
        gVec = projectWithAugmentedSystem(ldl, rVec, mat);
        dVec = minusVec(gVec);
        rg = dotVecVec(rVec, gVec);
        while(not ifZeroNum(rg, eps=1e-6)):
            gd = mulMatVec(self.gMat, dVec);
            alpha = dotVecVec(rVec, gVec) / dotVecVec(dVec, gd);
            self.xVec = addVecVec(self.xVec, mulNumVec(alpha, dVec));
            rVec = addVecVec(rVec, mulNumVec(alpha, gd));
            gVec = projectWithAugmentedSystem(ldl, rVec);
            rgNew = dotVecVec(rVec, gVec);
            beta = rgNew / rg;
            dVec = subVecVec(mulNumVec(beta, dVec), gVec);
            rg = rgNew;
        return;
    
    def iqpSubEqpNullSpaceMethod(self, ifEconComp=True):
        '''
        bVec is zero and thus not needed to be set, always default to be None
        '''
        if(self.aMat is None):  # special case: unconstraint (yMat = 0)
            self.unconstraintOpt();
            return;
        if(ifSquareMat(self.aMat)):  # special case: full rank
            self.xVec = zeroes(sizeSquareMat(self.aMat));
            self.lambdaVec = linSolve(mat=transposeMat(self.aMat),
                                      vec=addVecVec(self.cVec,
                                                    mulMatVec(self.gMat,
                                                              self.xVec)));
            return;
        else:
            yMat = self.aTrQrDm.getYMat();
            zMat = self.aTrQrDm.getZMat();
            pz = self.zgzLdlDm.linSolve(minusVec(mulVecMat(self.cVec, zMat)));
            self.xVec = mulMatVec(zMat, pz);
            if(ifEconComp):  # compute lambda only when x = 0
                ayMat = mulMatMat(self.aMat, yMat);
                self.lambdaVec = linSolveLuDecompVec(
                        *luDecomp(mat=ayMat, ifRowIdx=True),
                        vec=mulVecMat(
                                addVecVec(self.cVec,
                                          mulMatVec(self.gMat, self.xVec)),
                                mat=yMat),
                        ifTranspose=True);
        return;

class InequalQuadProg(object):
    n = None;
    me = None;
    mi = None;
    cVec = None;
    gMat = None;
    beVec = None;
    biVec = None;
    aeMat = None;
    aiMat = None;
    
    _initXVec = None;
    _initActIeqLst = None;
    _initYVec = None;
    _initEtaVec = None;
    _initLambdaVec = None;
    
    xVec = None;
    lambdaVec = None;
    actIeqLst = None;
    eqp = None;
    etaVec = None;  # equality multiplier for imp
    yVec = None;
    tau = 0.9;
    
    method = None;
    iterNum = 0;
    maxIterNum = 100;
    ifPrint = False;
    
    def _asPhase1Solve(self):
        p1CVec = catVecLst(zeroes(2 * self.n), ones(self.me),
                           ones(self.mi), zeroes(self.mi));
        p1BVec = catVecLst(self.beVec, self.biVec);
        p1AMat = rbind(None if(self.beVec is None) else
                           cbind(self.aeMat,
                                 minusMat(self.aeMat),
                                 diagMat([1.0 if self.beVec[i] >= 0.0 else -1.0 
                                          for i in range(self.mi)]),
                                 zeroes(self.me, 2 * self.mi)),
                       None if(self.biVec is None) else
                           cbind(self.aiMat,
                                 minusMat(self.aiMat),
                                 zeroes(self.mi, self.me),
                                 eye(self.mi),
                                 eye(self.mi, val=-1.0)));
        p1InitBIdxLst = ([2 * self.n + i for i in range(self.me)] + 
                         [2 * self.n + self.me + i + 
                            (0 if(self.biVec[i] >= 0.0) else self.mi) 
                            for i in range(self.mi)]);
        p1Simplex = Simplex(cVec=p1CVec, bVec=p1BVec, aMat=p1AMat,
                            initBIdxLst=p1InitBIdxLst);
        (p1XVec, p1F, p1BIdxLst, p1Ei) = p1Simplex.solve(ifPrint=False);
        xVec = [p1XVec[i] - p1XVec[i + self.n] for i in range(self.n)];
        actIeqLst = [i for i in range(self.mi) 
                     if(p1XVec[2 * self.n + self.me + i] - 
                        p1XVec[2 * self.n + self.me + i + self.mi] == 0.0)];
        return (p1F, xVec, actIeqLst);
    
    def _asPreSolve(self):
        #---------------------------- check independence of equality constraints
        bc = BasisConstructor();
        depEqLst = [i for i in range(self.me) 
                    if(not bc.addVec(self.aeMat[i]))];
        if(len(depEqLst) != 0):
            print('[IQP]: program has dependent equality constraints. Exit.');
            return False;
        if(self._initXVec is None): 
            (p1F, xVec, actIeqLst) = self._asPhase1Solve();
        else:
            xVec = self._initXVec;
            if(self._initActIeqLst is None):
                actIeqLst = [i for i in range(self.mi)
                             if(ifZeroNum(dotVecVec(self.aiMat[i], xVec)
                                          - self.biVec[i]))];
            else:
                actIeqLst = self._initActIeqLst;
            if(not ((self.me == 0 or 
                     ifZeroVec(subVecVec(mulMatVec(self.aeMat, xVec),
                                         self.beVec)))
                    and 
                    (self.mi == 0 or 
                     ifPosVec(subVecVec(mulMatVec(self.aiMat, xVec),
                                        self.biVec))))):
                p1F = 1.0;
            else:
                p1F = 0.0;
        #----------------------------------------------------- check feasibility
        if(p1F != 0.0):
            print('[IQP]: program not feasible. Exit.');
            return False;
        #----------------------- select independent subset of active constraints
        self.actIeqLst = [i for i in actIeqLst if(bc.addVec(self.aiMat[i]))];
        self.xVec = xVec;
        return True;
    
    def _asBkRefreshAMat(self):
        self.eqp.cVec = addVecVec(mulMatVec(self.gMat, self.xVec), self.cVec);
        self.eqp.aMat = rbind(self.aeMat,
                              None if len(self.actIeqLst) == 0 else 
                              getSubMat(self.aiMat, rowIdx=self.actIeqLst));
        return;
    
    def _asBkRefreshAQrDm(self):
        self.eqp.aTrQrDm = QrDecompModifier(transposeMat(self.eqp.aMat));
        return;
    
    def _asBkRefreshZgzLdlDm(self):
        yMat = self.eqp.aTrQrDm.getYMat();
        zMat = self.eqp.aTrQrDm.getZMat();
        zgzMat = mulMatLst(transposeMat(zMat), self.gMat, zMat);
        self.eqp.zgzLdlDm = LdlDecompModifier(zgzMat);
        return;
    
    def _asBkClear(self):
        self.eqp.aTrQrDm = None;
        self.eqp.zgzLdlDm = None;
        return;
    
    def _asIter(self):
        '''
        [flags]:
            0: continue;
            1: optimal point found;
        '''        
        #-------------------------------------------------- solve eqp subproblem
        self.eqp.iqpSubEqpNullSpaceMethod();
        pVec = self.eqp.xVec;
        self.lambdaVec = None;
        if(ifZeroVec(pVec)):
            self.lambdaVec = zeroes(self.me + self.mi);
            if(self.eqp.aMat is None): return 1;  # exit: unconstraint
            for i in range(self.me): 
                self.lambdaVec[i] = self.eqp.lambdaVec[i];
            for i in range(len(self.actIeqLst)):
                self.lambdaVec[self.me + i] = self.eqp.lambdaVec[self.me + i];
            j = minIdx(self.eqp.lambdaVec);
            if(self.eqp.lambdaVec[j] >= 0.0): return 1;  # exit: constraint
            #------------------------------------------------- remove constraint
            self.actIeqLst = [self.actIeqLst[i] for i in 
                              range(len(self.actIeqLst)) if(i != j)];
            self._asBkRefreshAMat();
            #------------------------------------------------ bookkeeping update
            if(self.eqp.aMat is None): self._asBkClear();
            elif(self.eqp.aTrQrDm is None):
                self._asBkRefreshAQrDm();
                self._asBkRefreshZgzLdlDm();
            else:
                self.eqp.aTrQrDm.delColumn(j);  # aTrQrDm: delete column
                zVec = transposeMat(self.eqp.aTrQrDm.getZMat(
                                    colIdx=[0]))[0];
                gzVec = mulMatVec(self.gMat, zVec);
                self.eqp.zgzLdlDm.expand(# zgzLdlDm: expand
                        len(self.actIeqLst) - 1,
                        catVecLst(mulVecMat(gzVec,
                                            self.eqp.aTrQrDm.getZMat),
                                  dotVecVec(zVec, gzVec)));
        else:
            apLst = [dotVecVec(self.aiMat[i], pVec) for i in range(self.mi)];
            idxAlphaLst = [(i , (self.biVec[i] - 
                            dotVecVec(self.aiMat[i], self.xVec)) / apLst[i])
                           for i in range(self.mi) if(apLst[i] < -_eps)];
            i = minIdx([alpha for (i, alpha) in idxAlphaLst]);
            #---------------------------------------------- add block constraint
            if(i != -1 and idxAlphaLst[i][1] < 1.0):
                alpha = idxAlphaLst[i][1];
                i = idxAlphaLst[i][0];
                self.xVec = addVecVec(self.xVec, mulNumVec(alpha, pVec));
                self.actIeqLst.append(i);
                self._asBkRefreshAMat();
                #-------------------------------------------- bookkeeping update
                if(ifSquareMat(self.eqp.aMat)): self._asBkClear();
                else:
                    if(self.eqp.aTrQrDm is None): self._asBkRefreshAQrDm();
                    else: self.eqp.aTrQrDm.addColumn(self.aiMat[i]);
                    self._asBkRefreshAQrDm();
            #----------------------------------------------- no block constraint
            else:
                self.xVec = addVecVec(self.xVec, pVec);
                self._asBkRefreshAMat();                 
        return 0;
    
    def asSolve(self):
        if(not self._asPreSolve()):  return;
        self.eqp = EqualQuadProg(gMat=self.gMat);
        self._asBkRefreshAMat();
        if(self.eqp.aMat is None or ifSquareMat(self.eqp.aMat)): 
            self._asBkClear();
        else:
            self._asBkRefreshAQrDm();
            self._asBkRefreshZgzLdlDm();
        while(True):
            flag = self._asIter();
            self.iterNum += 1;
            if(self.ifPrint):
                print('[IQP] iterNum = {0}'.format(self.iterNum));
                printMat(self.xVec, 'x');
                printMat(self.lambdaVec, 'l');
                print('active inequality constraints: ', self.actIeqLst);
                print(80 * '~');
            if(flag): break;
            if(self.iterNum >= self.maxIterNum): break;
        return;

    def _impLinSolveKKT(self, mat, vec):
        xVec = linSolveSymIdfMat(mat=mat, vec=vec);
        dXVec = xVec[0:self.n];
        dEtaVec = minusVec(xVec[self.n:self.n + self.me]);
        dLambdaVec = minusVec(xVec[self.n + self.me:self.n + 
                                   self.me + self.mi]);
        dYVec = addVecVec(mulMatVec(self.aiMat, dXVec),
                          subVecVec(mulMatVec(self.aiMat, self.xVec),
                                    addVecVec(self.yVec, self.biVec)));
        return (dXVec, dEtaVec, dLambdaVec, dYVec);
    
    def _impIter(self):
        #----------------------------------------------------------- affine step
        r1Vec = addVecLst(mulMatVec(self.gMat, self.xVec),
                          self.cVec,
                          None if(self.me == 0) else \
                            mulVecMat(minusVec(self.etaVec), self.aeMat),
                          None if(self.mi == 0) else \
                            mulVecMat(minusVec(self.lambdaVec), self.aiMat));
        r2Vec = None if(self.me == 0) else \
                    subVecVec(mulMatVec(self.aeMat, self.xVec), self.beVec);
        r3Vec = None if(self.mi == 0) else \
                    subVecVec(mulMatVec(self.aiMat, self.xVec), self.biVec);
        eEval = max(getVecNorm(r1Vec),
                    0.0 if r2Vec is None else getVecNorm(r2Vec),
                    0.0 if r3Vec is None else getVecNorm(r3Vec));
        rVec = catVecLst(minusVec(r1Vec),
                         None if r2Vec is None else minusVec(r2Vec),
                         None if r3Vec is None else minusVec(r3Vec));
        mat = rbind(cbind(self.gMat,
                          None if(self.me == 0) else transposeMat(self.aeMat),
                          None if(self.mi == 0) else transposeMat(self.aiMat)),
                    cbind(self.aeMat,
                          zeroes(self.me, self.me + self.mi)),
                    cbind(self.aiMat,
                          zeroes(self.mi, self.me),
                          diagMat([-self.yVec[i] / self.lambdaVec[i] 
                                   for i in range(self.mi)])));
        (dXVec, dEtaVec, dLambdaVec, dYVec) = self._impLinSolveKKT(mat, rVec);
        alpha = 1.0
        #------------------------------------------------------------- centering
        if(self.mi != 0):
            miu = dotVecVec(self.yVec, self.etaVec) / self.mi;
            #------------------------------------------------- equal step length
            alphaAff = min(catVecLst([-self.yVec[i] / dYVec[i] 
                                      for i in range(self.mi) 
                                      if(dYVec[i] < 0.0)],
                                     [-self.lambdaVec / dLambdaVec[i]
                                      for i in range(self.mi)
                                      if(dLambdaVec[i] < 0.0)],
                                     1.0));
            miuAff = dotVecVec(addVecVec(self.yVec,
                                         mulNumVec(alphaAff, dYVec)),
                               addVecVec(self.lambdaVec,
                                         mulNumVec(alphaAff, dLambdaVec)));
            sigma = min(1.0, (miuAff / miu) ** 3);
            r3Vec = addVecLst(r3Vec,
                              [dLambdaVec[i] * dYVec[i] / self.lambdaVec[i]
                               for i in range(self.mi)],
                              mulNumVec(-sigma * miu, self.lambdaVec));
            rVec = catVecLst(minusVec(r1Vec),
                             minusVec(r2Vec),
                             minusVec(r3Vec));
            (dXVec, dEtaVec,
             dLambdaVec, dYVec) = self._impLinSolveKKT(mat, rVec);
            #------------------------------------------------- equal step length
            alpha = min(catVecLst([self.tau * self.yVec[i] / dYVec[i] 
                                   for i in range(self.mi) 
                                   if dYVec[i] < 0.0],
                                  [self.tau * self.lambdaVec[i] / dLambdaVec[i] 
                                   for i in range(self.mi) 
                                   if dLambdaVec[i] < 0.0],
                                  1.0));
        
        self.xVec = addVecVec(self.xVec, mulNumVec(alpha, dXVec));
        self.etaVec = addVecVec(self.etaVec, mulNumVec(alpha, self.etaVec));
        self.lambdaVec = addVecVec(self.lambdaVec,
                                   mulNumVec(alpha, self.lambdaVec));
        self.yVec = addVecVec(self.yVec, mulNumVec(alpha, self.yVec));
        self.tau = 0.5 * (self.tau + 1.0);
        return eEval;

    def _impPreSolve(self):
        #---------------------------- check independence of equality constraints
        bc = BasisConstructor();
        depEqLst = [i for i in range(self.me) 
                    if(not bc.addVec(self.aeMat[i]))];
        if(len(depEqLst) != 0):
            print('[IQP]: program has dependent equality constraints. Exit.');
            return False;
        if(self._initXVec is not None and 
           self._initYVec is not None and 
           self._initEtaVec is not None and 
           self._initLambdaVec is not None):
            (self.xVec, self.yVec,
             self.etaVec, self.lambdaVec) = (self._initXVec, self._initYVec,
                            self._initEtaVec, self._initLambdaVec);
        else:
            #------------------------------------------- minimal normal solution
            xVec = linSolveLeastNormSol(self.aeMat, self.beVec);
            yVec = subVecVec(mulMatVec(self.aiMat, xVec), self.biVec);
            dVec = linSolveLeastNormSol(transposeMat(rbind(self.aeMat,
                                                           self.aiMat)),
                                        addVecVec(mulMatVec(self.gMat,
                                                            self.xVec),
                                                  self.cVec));
            etaVec = dVec[0:self.me];
            lambdaVec = dVec[self.me:self.me + self.mi];
            #------------------------------------------------ enforce positivity
            deltaY = max(-1.5 * min(yVec), 0.0);
            deltaLambda = max(-1.5 * min(lambdaVec), 0.0);
            yVec = addVecVec(yVec, ones(self.mi, val=deltaY));
            lambdaVec = addVecVec(lambdaVec, ones(self.mi, val=deltaLambda));
            #------------------------------------------------ enforce similarity
            yl = dotVecVec(self.yVec, self.lambdaVec)
            deltaY2 = 0.5 * yl / sum(self.lambdaVec);
            deltaLambda2 = 0.5 * yl / sum(self.yVec);
            yVec = addVecVec(yVec, ones(self.mi, val=deltaY2));
            lambdaVec = addVecVec(lambdaVec, ones(self.mi, val=deltaLambda2));
            self.xVec = xVec;
            self.yVec = yVec;
            self.etaVec = etaVec;
            self.lambdaVec = lambdaVec;
        return True;
    
    def impSolve(self):
        if(not self._impPreSolve()): return;
        while(True): 
            eEval = self._impIter();
            self.iterNum += 1;
            if(eEval < 1e-6): return;
            if(self.ifPrint):
                print('[IQP] iterNum = {0}'.format(self.iterNum));
                printMat(self.xVec, 'x');
                printMat(self.yVec, 'y');
                printMat(self.etaVec, 'eta');
                printMat(self.lambdaVec, 'lambda');
                print(80 * '~');
            if(self.iterNum > self.maxIterNum): return;
        return;
    
    def solve(self):
        if(self.method == 'interior-point'): self.impSolve();
        elif(self.method == 'active-set'): self.asSolve();
        return;
    
    def __init__(self, cVec, gMat, beVec=None, aeMat=None, biVec=None,
                 aiMat=None, initAs=None, initImp=None, method='active-set',
                 maxIterNum=100, ifPrint=False):
        self.n = sizeVec(cVec);
        self.me = 0 if(beVec is None) else sizeVec(beVec);
        self.mi = 0 if(biVec is None) else sizeVec(biVec);
        self.cVec = cVec;
        self.gMat = gMat;
        self.beVec = beVec;
        self.aeMat = aeMat;
        self.biVec = biVec;
        self.aiMat = aiMat;
        if(initAs is None): 
            (self._initXVec, self._initActIeqLst) = (None, None);
        else:
            (self._initXVec, self._initActIeqLst) = initAs;
        if(initImp is None):
            (self._initXVec, self._initYVec, self._initEtaVec,
             self._initLambdaVec) = (None, None, None, None);
        else:
            (self._initXVec, self._initYVec, self._initEtaVec,
             self._initLambdaVec) = initImp;
        self.iterNum = 0;
        self.method = method;
        self.maxIterNum = maxIterNum;
        self.ifPrint = ifPrint;
        return;
    
if __name__ == '__main__':
#     gMat = [[6.0, 2.0, 1.0],
#             [2.0, 5.0, 2.0],
#             [1.0, 2.0, 4.0]];
#     cVec = [-8.0, -3.0, -3.0];
#     aMat = [[1.0, 0.0, 1.0],
#             [0.0, 1.0, 1.0]];
#     bVec = [3.0, 0.0];
#     eqp = EqualQuadProg(cVec, gMat, bVec, aMat);
#     eqp.directMethod();
#     printMat(eqp.lambdaVec, 'lambda');
#     printMat(eqp.xVec, 'x');
#     
#     eqp.schurComplementMethod();
#     printMat(eqp.lambdaVec, 'lambda');
#     printMat(eqp.xVec, 'x');
#     
#     eqp.nullSpaceMethod();
#     printMat(eqp.lambdaVec, 'lambda');
#     printMat(eqp.xVec, 'x');
#     
#     eqp.projectedConjugateGradientMethod();
#     printMat(eqp.lambdaVec, 'lambda');
#     printMat(eqp.xVec, 'x');
#     
#     eqp.cVec = addVecVec(eqp.cVec, mulMatVec(gMat, eqp.xVec));
#     eqp.iqpSubEqpNullSpaceMethod();
#     printMat(eqp.lambdaVec, 'lambda');
#     printMat(eqp.xVec, 'x');
    
    gMat = [[2.0, 0.0],
            [0.0, 2.0]];
    cVec = [-2.0, -5.0];
    aiMat = [[1.0, -2.0],
             [-1.0, -2.0],
             [-1.0, 2.0],
             [1.0, 0.0],
             [0.0, 1.0]];
    biVec = [-2.0, -6.0, -2.0, 0.0, 0.0];
#     iqp = InequalQuadProg(cVec, gMat, biVec=biVec, aiMat=aiMat, ifPrint=True);
    iqp = InequalQuadProg(cVec, gMat, biVec=biVec, aiMat=aiMat,
                          method='interior-point', ifPrint=True);
    iqp.solve();
    pass;
