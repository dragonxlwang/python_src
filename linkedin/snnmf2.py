'''
Created on Jul 9, 2014

@author: xwang1
'''
from toolkit.num.sparse import toSparseVec, getSvKeys, addSvSv, mulSmSv, subSvSv, \
    mulSvSm, mulNumSv, getSvElem, dotSvSv, getSmRow, getSvVals, getSmRowIdxLst, \
    transposeSm, getSvLen, getSmSize, getSmColLen, setSmElem, cloneSv, \
    getSvL1Norm, getSmL2Norm, getSmL1Norm, getSmDat, setSmRow
from toolkit.num.algebra import ones, linSolveSymIdfMat, addVecVec, mulNumVec, \
    subVecVec, randomVec
from toolkit.num.arithmetic import ifZeroNum, avg
from toolkit.utility import loadObjFromFile, writeObjToFile, mkDir
from linkedin.const import *;
import sys;
import os;
from multiprocessing import Pool, current_process
from time import time, clock

def decodeActSet(xSv, sSv, m, n, l, rl, rm, bTrSm, bIdx):
    if(sSv is None): sSv = toSparseVec(dim=l);
    aSet = set([k for k in getSvKeys(sSv) 
                if (not ifZeroNum(getSvElem(sSv, k)))]);
    bSet = set();
    for i in getSvKeys(xSv):
        for k in bIdx.get(i, []):
            bSet.add(k);
    def obj():
        rSv = subSvSv(xSv, mulSvSm(sSv, bTrSm));
        ls = dotSvSv(rSv, rSv);
        rg = 0.5 * rm * sum([x * x for x in getSvVals(sSv)]) + \
                rl * sum([x for x in getSvVals(sSv)]);
        fEval = ls + rg;
        return (fEval, ls, rg);
    def check():
        gSv = addSvSv(mulSmSv(bTrSm,
                              subSvSv(mulSvSm(sSv, bTrSm),
                                      xSv),
                              bSet),
                      addSvSv(toSparseVec(keys=list(bSet),
                                          vals=ones(m=len(bSet), val=rl),
                                          dim=l),
                              mulNumSv(rm, sSv)));
        #------------------------------------------------ check active basis
        for k in aSet:
            if(not ifZeroNum(getSvElem(gSv, k))): return "opt";
        unbound = [k for k in bSet if (k not in aSet)];
        #----------------------------------------------- check unbound basis
        if(len(unbound) > 0):
            k = min(unbound, key=lambda x: getSvElem(gSv, x));
            if(getSvElem(gSv, k) < 0.0): return k;
        #---------------------------------------------- optimiztion complete
        return "done";
    def opt():
        aLst = list(aSet);
        bGramMat = [[dotSvSv(getSmRow(bTrSm, k1),
                             getSmRow(bTrSm, k2))
                     + (rm if (k1 == k2) else 0.0)
                     for k1 in aLst] 
                    for k2 in aLst];
        vec = [dotSvSv(getSmRow(bTrSm, k), xSv) - rl for k in aLst];
        sol = linSolveSymIdfMat(mat=bGramMat, vec=vec);
        sVec = [getSvElem(sSv, k) for k in aLst];
        alpha = min([1.0 if (sVec[k] == sol[k] or sol[k] >= 0.0)
                     else sVec[k] / (sVec[k] - sol[k])
                     for k in range(len(sol))]);
        sVec = addVecVec(mulNumVec(1 - alpha, sVec),
                         mulNumVec(alpha, sol));
        return toSparseVec(keys=aLst, vals=sVec, dim=l);
    iterNum = 0;
    st = clock();
    while(True): 
        flag = check();
        iterNum += 1;
        #--------------------------------------- terminate by maximal iterations
        if(iterNum > 100 or clock() - st > 100 or flag == "done"):
            return (sSv, obj());
        else:
            if(flag != "opt"): aSet.add(flag);
            sSv = opt();
            aSet = set([k for k in getSvKeys(sSv) if 
                        (not ifZeroNum(getSvElem(sSv, k)))]);
    return;

def decodeByChunk(xTrSm, jLst, m, n, l, rlLst, rmLst, bTrSm, bIdx, sTrSm):
    retLst = [];
    cnt = 0;
    st = clock();
    r = 0.0;
    for j in jLst:
        cnt += 1;
        if(float(cnt) / len(jLst) > r):
            sys.stderr.write("[proc {0}]: {1} ({4}) vector processed. "
                             "{3} completed. " 
                             "take time = {2}s\r".format(
                                current_process().name, cnt,
                                clock() - st,
                                r, len(jLst)));
            while(float(cnt) / len(jLst) > r): r += 0.1;
            sys.stderr.flush();
        xSv = getSmRow(xTrSm, j);
        sSv = None if(sTrSm is None) else cloneSv(getSmRow(sTrSm, j));
        (sSv, objInfo) = decodeActSet(xSv=xSv,
                                      sSv=sSv,
                                      m=m,
                                      n=n,
                                      l=l,
                                      rl=rlLst[j],
                                      rm=rmLst[j],
                                      bTrSm=bTrSm,
                                      bIdx=bIdx);
        retLst.append((j, sSv, objInfo));
    return retLst;

class SNNMF2(object):
    '''
    classdocs
    '''
    bTrSm = None;
    xSm = None;
    xTrSm = None;
    sSm = None;
    m = 0;  # example number
    n = 0;  # vocabulary number
    l = 0;  # basis number
    lambdaS = 0;  # L1 regularizer
    lambdaB = 0;
    miuS = 0;  # L2 regularizer
    miuB = 0;
    bIdx = None;  # basis index
    sIdx = None;
    procNum = 10;
    startingFrom = None;
    outputDir = None;
    sTrSm = None;
    bSm = None;
    objLs = 0;
    objBRg = 0;
    objSRg = 0;
    objF = 0;
    learnBasisOptions = None;
    iterNum = None;
    maxIterNum = 200;
    
    def refreshBIdx(self):
        self.bIdx = {};
        for k in getSmRowIdxLst(self.bTrSm):
            bSv = getSmRow(self.bTrSm, k);
            for i in getSvKeys(bSv):
                if(i not in self.bIdx): self.bIdx[i] = [];
                self.bIdx[i].append(k);
        return; 
    
    def refreshSIdx(self):
        self.sIdx = {};
        for k in getSmRowIdxLst(self.sSm):
            srSv = getSmRow(self.sSm, k);
            for j in getSvKeys(srSv):
                if(j not in self.sIdx): self.sIdx[j] = [];
                self.sIdx[j].append(k);
        return;
    
    def decode(self):
        print("\n[iter {0}]: start decoding, submitting {1} " \
              "processes".format(self.iterNum, self.procNum));
        self.refreshBIdx();
        pool = Pool(processes=self.procNum);
        blockSize = int(float(self.m) / self.procNum) + 1;
        results = [];
        rlLst = [self.lambdaS for j in range(self.m)];
        rmLst = [self.miuS for j in range(self.m)];
        for p in range(self.procNum):
            jLst = range((p * blockSize),
                         min(((p + 1) * blockSize), self.m));
            results.append(pool.apply_async(func=decodeByChunk,
                                            args=(self.xTrSm, jLst,
                                                  self.m, self.n, self.l,
                                                  rlLst, rmLst,
                                                  self.bTrSm, self.bIdx,
                                                  self.sTrSm)));
        pool.close();
        pool.join();
        print("");
        print("[iter {0}]: end decoding".format(self.iterNum));
        self.sSm = ({}, (self.l, self.m));
        (self.objLs, self.objSRg,
                                self.objF) = (0.0, 0.0, 0.0);
        for r in results:
            retLst = r.get();
            for (j, sSv, objInfo) in retLst:
                (objF, objLs, objRg) = objInfo;
                self.objLs += objLs;
                self.objSRg += objRg;
                for k in getSvKeys(sSv):
                    v = getSvElem(sSv, k);
                    setSmElem(self.sSm, k, j, v);
#         self.objBRg = 0.5 * self.miuB * getSmL2Norm(self.bTrSm) + \
#                         self.lambdaB * getSmL1Norm(self.bTrSm);
        self.objF = self.objLs + self.objSRg + self.objBRg;
        #--------------------------------------------------------- transpose
        self.sTrSm = transposeSm(self.sSm);
        return;
    
    def learnBasis(self, l1ReweightMethod=None,
                   l2ReweightMethod=None,
                   addInfo=None):
        print("\n[iter {0}]: start learning basis, submitting {1} " \
              "processes".format(self.iterNum, self.procNum));
        self.refreshSIdx();
        pool = Pool(processes=self.procNum);
        blockSize = int(float(self.n) / self.procNum) + 1;
        results = [];
        #--------------------------------------------------------- l1 reweighted
        if(l1ReweightMethod == None):
            rlLst = [self.lambdaB for j in range(self.n)];
        elif(l1ReweightMethod == "reverse_idf"):
            idfLst = [len(getSvKeys(getSmRow(self.xSm, i))) + 1.0 
                  for i in range(self.n)];
            avgIdf = avg(idfLst);
            rlLst = [avgIdf * self.lambdaB / idfLst[j] for j in range(self.n)];
        elif(l1ReweightMethod == "concave_log"):
            l1Lst = [getSvL1Norm(getSmRow(self.bSm, j))
                     for j in range(self.n)];
            rlLst = [self.lambdaB / (l1Lst[j] + 1e-2) for j in range(self.n)];
        #--------------------------------------------------------- l2 reweighted
        if(l2ReweightMethod == None):
            rmLst = [self.miuB for j in range(self.n)];
        else:
            pass;    
        for p in range(self.procNum):
            jLst = [j for j in range(self.n) if (j % self.procNum == p)]; 
            results.append(pool.apply_async(func=decodeByChunk,
                                            args=(self.xSm, jLst,
                                                  self.n, self.m, self.l,
                                                  rlLst, rmLst,
                                                  self.sSm, self.sIdx,
                                                  self.bSm)));
        pool.close();
        pool.join();
        print("");
        print("[iter {0}]: end learning basis".format(self.iterNum));
        self.bTrSm = ({}, (self.l, self.n));
        (self.objLs, self.objBRg,
                                self.objF) = (0.0, 0.0, 0.0);
        for r in results:
            retLst = r.get();
            for (i, bTrSv, objInfo) in retLst:
                (objF, objLs, objRg) = objInfo;
                self.objLs += objLs;
                self.objBRg += objRg;
                for k in getSvKeys(bTrSv):
                    v = getSvElem(bTrSv, k);
                    setSmElem(self.bTrSm, k, i, v);
        #=======================================================================
        # supplement diminished vectors
        #=======================================================================
        #=======================================================================
        # for k in range(self.l):
        #     if(k not in getSmDat(self.bTrSm)):
        #         setSmRow(self.bTrSm, k, toSparseVec(vec=randomVec(self.n)));
        #=======================================================================
#         self.objSRg = 0.5 * self.miuS * getSmL2Norm(self.sSm) + \
#                         self.lambdaS * getSmL1Norm(self.sSm);
        self.objF = self.objLs + self.objSRg + self.objBRg;
        #--------------------------------------------------------- transpose
        self.bSm = transposeSm(self.bTrSm);
        return;
    
    def debug(self):
        self.refreshBIdx();
        for j in range(self.m):
            decodeActSet(xSv=getSmRow(self.xTrSm, j),
                         sSv=None if (self.sTrSm is None)  
                                else cloneSv(getSmRow(self.sTrSm, j)),
                         m=self.m,
                         n=self.n,
                         l=self.l,
                         rl=self.lambdaS,
                         rm=self.miuS,
                         bTrSm=self.bTrSm,
                         bIdx=self.bIdx);
                                 
    def work(self):
        if(self.learnBasisOptions == None):
            (lbL1ReweightMethod, lbL2ReweightMethod,
                        lbAddInfo) = (None, None, None);
        else: (lbL1ReweightMethod, lbL2ReweightMethod,
                        lbAddInfo) = self.learnBasisOptions;
        self.iterNum = 1;
        if(self.startingFrom is not None):
            (task, self.iterNum) = self.startingFrom;
            if(task == "decode"):
                self.sSm = loadObjFromFile(os.path.join(self.outputDir,
                                        "{0}.sSm".format(self.iterNum)));
                self.sTrSm = transposeSm(self.sSm);
                #===============================================================
                # learn basis
                #===============================================================
                self.learnBasis(l1ReweightMethod=lbL1ReweightMethod,
                                l2ReweightMethod=lbL2ReweightMethod,
                                addInfo=lbAddInfo);
                #--------------------------------------------------- write bTrSm
                writeObjToFile(os.path.join(self.outputDir,
                                        "{0}.bTrSm".format(self.iterNum)),
                               self.bTrSm);
                #----------------------------------------------------- write obj
                writeObjToFile(os.path.join(self.outputDir,
                                        "{0}.learn.obj".format(self.iterNum)),
                            (self.objF, self.objLs, self.objBRg, self.objSRg));
                #----------------------------------------------------- print obj
                print("objective: {0}".format((self.objF, self.objLs,
                                               self.objBRg, self.objSRg)));
            elif(task == "learnBasis"):
                self.bTrSm = loadObjFromFile(os.path.join(self.outputDir,
                                        "{0}.bTrSm".format(self.iterNum)));
                self.bSm = transposeSm(self.bTrSm);
            self.iterNum += 1;
        while(True):
            #===================================================================
            # decode
            #===================================================================
            self.decode();
            #--------------------------------------------------------- write sSm
            writeObjToFile(os.path.join(self.outputDir,
                                        "{0}.sSm".format(self.iterNum)),
                           self.sSm);
            #--------------------------------------------------------- write obj
            writeObjToFile(os.path.join(self.outputDir,
                                        "{0}.decode.obj".format(self.iterNum)),
                            (self.objF, self.objLs, self.objBRg, self.objSRg));
            #--------------------------------------------------------- print obj
            print("objective: F={0}, Ls={1}, BRg={2}, SRg={3}".format(
                    self.objF, self.objLs, self.objBRg, self.objSRg));                                   
            #===================================================================
            # learn basis
            #===================================================================
            self.learnBasis(l1ReweightMethod=lbL1ReweightMethod,
                                l2ReweightMethod=lbL2ReweightMethod,
                                addInfo=lbAddInfo);
            #------------------------------------------------------- write bTrSm
            writeObjToFile(os.path.join(self.outputDir,
                                        "{0}.bTrSm".format(self.iterNum)),
                           self.bTrSm);
            #--------------------------------------------------------- write obj
            writeObjToFile(os.path.join(self.outputDir,
                                        "{0}.learn.obj".format(self.iterNum)),
                            (self.objF, self.objLs, self.objBRg, self.objSRg));
            #--------------------------------------------------------- print obj
            print("objective: F={0}, Ls={1}, BRg={2}, SRg={3}".format(
                    self.objF, self.objLs, self.objBRg, self.objSRg));
            print("active phrase number: {0}".format(len(
                                                    getSmDat(self.bTrSm))));
            self.iterNum += 1;
            if(self.iterNum > self.maxIterNum):
                print("maximum number of iterations reached"); 
                break;
        return;
    
    def __init__(self,
                 xSmFilePath=globalXSmFilePath,
                 bTrSmFilePath=globalInitBTrSmFilePath,
                 lambdaS=1.0, lambdaB=1.0, miuS=0.001, miuB=1e-5, procNum=10,
                 outputDir=None, startingFrom=None, learnBasisOptions=None,
                 initBTrSm=None):
        '''
        Constructor
        '''        
        self.xSm = loadObjFromFile(xSmFilePath);
        self.xTrSm = transposeSm(self.xSm);
        if(initBTrSm is None):
            self.bTrSm = loadObjFromFile(bTrSmFilePath);
        else:
            self.bTrSm = initBTrSm;
        self.bSm = transposeSm(self.bTrSm);
        (self.n, self.m) = getSmSize(self.xSm);
        self.l = getSmColLen(self.bTrSm);
        (self.lambdaS, self.lambdaB, self.miuS, self.miuB) = \
            (lambdaS, lambdaB, miuS, miuB);
        self.procNum = procNum;
        self.startingFrom = startingFrom;
        if(outputDir is None):
            self.outputDir = os.path.join(dataDir,
                            "snnmf-{0}-{1}-{2}-{3}".format(lambdaS, lambdaB,
                                                            miuS, miuB));
        else: self.outputDir = outputDir;
        mkDir(self.outputDir);
        self.learnBasisOptions = learnBasisOptions;
        return;
    
if __name__ == '__main__':
    pass;
