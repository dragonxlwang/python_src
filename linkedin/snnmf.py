'''
Created on Jun 23, 2014

@author: xwang1
'''
from toolkit.num.arithmetic import maxIdx, ifZeroNum, avg
from linkedin.const import vocabularySize, dataDir, initialPhraseFilePath, \
    vocabularyFilePath
import os;
import sys;
from toolkit.num.algebra import linSolve, linSolveSymIdfMat, addVecVec, \
    mulNumVec, subVecVec, printMat, getMatRank, ones, mulMatVec
from toolkit.num.sparse import dotSvSv, getSv, updateAddSv, getSvDat, getSmRow, \
    subSvSv, mulSvSm, mulSmSv, toDenseVec, getSvElem, toSparseVec, \
    getSmDat, getSmRowIdxLst, getSmSize, addSvSv, mulNumSv, getSvVals, setSmElem, \
    getSvKeys
from timeit import timeit
from time import time, sleep
import math
from numpy.lib.utils import source
from multiprocessing import Pool
from multiprocessing import current_process

class SNNMF(object):
    bTrSm = None;  # basis transpose sparse matrix
    xTrSm = None; 
    sSm = None;
    bIdx = {};
    m = 0;
    n = 0;
    l = 0;
    lambdaS = 0;
    lambdaB = 0;
    miuS = 0;
    miuB = 0;
    word2id = None;
    id2word = None;
    task = None;
    xFilePath = None;
    epochNum = None;
    recordLst = None;
    def _seqToSv(self, seq):
        d = {};
        for x in seq: d[x] = d.get(x, 0.0) + 1.0;
        return (d, self.n);
    
    #===========================================================================
    # obsolete
    #===========================================================================
    def _decodeGreedy(self, seq):
        sSv = getSv({}, self.l);
        t = 0;
        while(t < len(seq)):
            p = t;
            while(t < len(seq) and 
                  len(self.bIdx.get(self._genIdx(seq[p:t + 1]), [])) != 0):
                t += 1;
            if(p == t): t += 1;
            else:
                idx = self._genIdx(seq[p:t]);
                xSv = self._seqToSv(seq[p:t]);
                k = max(self.bIdx[idx],
                        key=lambda k: dotSvSv(getSmRow(self.bTrSm, k), xSv));
                updateAddSv(sSv, ({k:dotSvSv(getSmRow(self.bTrSm, k), xSv)},
                                  self.l));
        return sSv;
    
    def _decodeActSet(self, xSv=None, sSv=None):
        bSet = set();
        for i in getSvIdxLst(xSv):
            for k in self.bIdx.get(str(self._genIdx([i])), []):
                bSet.add(k);
        def obj():
            rSv = subSvSv(xSv, mulSvSm(sSv, self.bTrSm));
            return (dotSvSv(rSv, rSv) + 
                    0.5 * self.miuS * sum([x * x 
                                           for x in getSvVals(sSv)]) + 
                    self.lambdaS * sum([x for x in getSvVals(sSv)]));
        def opt(asLst):
            bGramMat = [[dotSvSv(getSmRow(self.bTrSm, k1),
                                 getSmRow(self.bTrSm, k2))
                         + (self.miuS if k1 == k2 else 0.0) 
                         for k1 in asLst] for k2 in asLst]; 
            vec = [dotSvSv(getSmRow(self.bTrSm, k1), xSv) 
                   - self.lambdaS for k1 in asLst];
            s2Vec = linSolveSymIdfMat(mat=bGramMat, vec=vec);
            sVec = [getSvElem(sSv, k) for k in asLst];
            alpha = min([1.0 if(sVec[k] == s2Vec[k] or s2Vec[k] >= 0.0) 
                         else sVec[k] / (sVec[k] - s2Vec[k])
                         for k in range(len(s2Vec))]);
            s3Vec = addVecVec(sVec, mulNumVec(alpha, subVecVec(s2Vec, sVec)));
            return toSparseVec(keys=asLst, vals=s3Vec, dim=self.l);
        def check():
            gSv = addSvSv(mulSmSv(self.bTrSm,
                                  subSvSv(mulSvSm(sSv, self.bTrSm),
                                          xSv),
                                  bSet),
                          addSvSv(toSparseVec(keys=list(bSet),
                                              vals=ones(len(bSet),
                                                        val=self.lambdaS),
                                              dim=self.l),
                                  mulNumSv(self.miuS, sSv))
                          );
            #--------------------------- current active set optimal not achieved
            for k in getSvDat(sSv):
                if(not ifZeroNum(getSvElem(gSv, k))):
                    return "opt";
            #------------------------------------------------- include new basis
            idxLst = [k for k in bSet if (k not in getSvDat(sSv))];
            if(len(idxLst) > 0):
                k = min(idxLst, key=lambda x: getSvElem(gSv, x));
                if(getSvElem(gSv, k) < 0.0): return k;
            #--------------------------------------------- optimization complete
            return "done";
        iterNum = 0;
        stTime = time();
        while(True):
            flag = check();
            if(flag == "opt"): sSv = opt(getSvKeys(sSv));
            elif(flag == "done"):
                #===============================================================
                # DEBUG
                #===============================================================
#                 print "iterNum={0}, time={1}, obj={2}".format(iterNum,
#                                                               time() - stTime,
#                                                               obj());
#                 print "xSv=" + str(xSv);
#                 print "==>" + ' '.join(["{0}:{1}".format(k, self.id2word[k]) 
#                                         for k in getSvDat(xSv)]);
#                 print "sSv=" + str(sSv);
#                 for k in sorted(getSvDat(sSv)):
#                     print "\t" + str(k) + ": " \
#                             + str(getSvElem(sSv, k)) + ":: "\
#                             + str(getSmRow(self.bTrSm, k)) + "\t" \
#                             + ' '.join([self.id2word[s] 
#                                 for s in getSvDat(getSmRow(self.bTrSm, k))]);
#                 sys.stdin.readline();
                #===============================================================
                # DEBUGEND
                #=============================================================== 
                return sSv;
            else: sSv = opt(getSvIdxLst(sSv) + [flag]);
            iterNum += 1;
        return;
        
    def decode(self):
        cnt = 0;
        stTime = time();
        sSvLst = [];
        
        
        for record in self.recordLst:
            cnt += 1;
            if(cnt % 1000 == 0):
                sys.stdout.write('{0} records processed,' \
                                 ' take time {1} s\r'.format(cnt,
                                                             time() - stTime));
                sys.stdout.flush();
            (fv, tag, comment) = record;
            #===================================================================
            # print comment
            #===================================================================
            for seq in fv:
                #-------------------------------------- option 1: for warm start
#                 sSv = self._decodeGreedy(seq);
                #--------------------------------------- option 2: start at zero
                sSv = toSparseVec(keys=[], vals=[], dim=self.l);
                #------------------------------------------- active set decoding
                sSv = self._decodeActSet(seq, sSv);
                sSvLst.append(sSv);
        self.m = len(sSvLst);
        self.sSm = ({}, (self.l, self.m));
        for colIdx in range(self.m):
            for rowIdx in getSvIdxLst(sSvLst[colIdx]):
                setSmElem(self.sSm, rowIdx, colIdx,
                          getSvElem(sSvLst[colIdx], rowIdx));
        sFilePath = self.xFilePath + ".{0}.{1}".format(self.epochNum,
                                                       self.task);
        sFile = open(sFilePath, "w");
        sFile.write(str(self.sSm));
        sFile.close();
        return;
    
    def dictionary(self):
        
        return;
    
    def work(self):
        if(self.task == "coding"): self.decode();
        elif(self.task == "dictionary"):
            (self.m, self.n) = (self.n, self.m);
            self.bTrSm = self.sSm;
            (self.lambdaS, self.miuS) = (self.lambdaB, self.miuB);
            
        return;
    
    def _genIdx(self, lst):
        sortedLst = sorted(lst);
        return ' '.join([str(x) for x in sortedLst]);
    
    def _genPowSet(self, sortedLst):
        if(len(sortedLst) == 0): return [""];
        if(len(sortedLst) == 1): return [str(sortedLst[0]), ""];
        else:
            idxLst = self._genPowSet(sortedLst[1:]);
            return [' '.join([str(sortedLst[0]), idx]) 
                    for idx in idxLst] + idxLst;
    
    def _bIdxInit(self):
        self.bIdx = {};
        for k in getSmRowIdxLst(self.bTrSm):
            idxLst = [str(w) for w in getSvKeys(getSmRow(self.bTrSm, k))];
#             idxLst = self._genPowSet(sorted(
#                                     getSvIdxLst(getSmRow(self.bTrSm, k))));
            for idx in idxLst:
                if(idx not in self.bIdx): self.bIdx[idx] = [];
                self.bIdx[idx].append(k);
            p = ' '.join(sorted(idxLst));
            if(p not in self.bIdx): self.bIdx[p] = [];
            self.bIdx[p].append(k);
        return;
    
    def loadXTr(self, filePath):
        file = open(filePath);
        self.xTrSm = eval(file.readline());   
        file.close();
        return;
    
    def loadBTr(self, filePath=None):
        if(filePath is None):
            self.bTrSm = ({}, (self.l, self.n));
            #------------------------------------ initialize phrase list by word
            for i in range(self.n): getSmDat(self.bTrSm)[i] = ({i:1.0}, self.n);
            #---------------------------------- load phrase list sorted by count
            ipFile = open(initialPhraseFilePath);
            i = self.n;
            bUniStrSet = set();
            for ln in ipFile:
                (ph, c) = ln.split('\t');
                tokIdLst = [self.word2id[x] for x in [w.lower() 
                                                      for w in ph.split()] 
                            if x in self.word2id];
                bSv = self._seqToSv(tokIdLst);
                if(len(getSvDat(bSv)) <= 1): continue;
                s = math.sqrt(sum([x * x for x in getSvDat(bSv).values()]));
                for k in getSvDat(bSv): getSvDat(bSv)[k] /= s;
                uniStr = '\xaa'.join(['{0}:{1}'.format(k, getSvElem(bSv, k)) 
                                      for k in sorted(getSvDat(bSv))]);
                if(uniStr in bUniStrSet): continue;
                else: bUniStrSet.add(uniStr);
                getSmDat(self.bTrSm)[i] = bSv;
                i += 1;
                if(i == self.l): break;
            ipFile.close(); 
        else:
            file = open(filePath);
            self.bTrSm = eval(file.readline());
            file.close();
        self._bIdxInit(); 
        return;
        
    def loadS(self, filePath):
        file = open(filePath);
        self.sSm = eval(file.readline());
        file.close();
        return;
    
    def loadVocMap(self):
        #-------------------------------------------------- load vocabulary file
        vocFile = open(vocabularyFilePath);
        vocLst = eval(vocFile.readline());
        vocDict = {};
        for i in range(len(vocLst)): vocDict[vocLst[i]] = i;
        vocFile.close();
        self.word2id = vocDict;
        self.id2word = {};
        for w in self.word2id: self.id2word[self.word2id[w]] = w;
        return;
        
    def __init__(self, vocabularySize, ratioLtoN,
                 sSmFilePath=None, bTrSmFilePath=None, xFilePath=None,
                 lambdaS=2.0, lambdaB=2.0, miuS=0.001, miuB=0.001,
                 task=None, epochNum=None):
        curTime = time();
        self.n = vocabularySize;
        self.l = int(vocabularySize * ratioLtoN);
        self.task = task;
        self.xFilePath = xFilePath;
        self.epochNum = epochNum;
        #--------------------------------------------------------------- loading
        self.loadVocMap();
#         self.loadXTr(xFilePath);
        if(self.task == "dictionary"): 
            self.loadS(sSmFilePath);
        elif(self.task == "coding"): 
            self.loadBTr(bTrSmFilePath);
#         print('[snnmf]: loading finishes, take time {0} s '  
#               'loading {1} records'.format(time() - curTime,
#                                            len(self.recordLst)));
        self.lambdaS = lambdaS;
        self.lambdaB = lambdaB;
        self.miuS = miuS;
        self.miuB = miuB;
        return;

lst = [];
def func(x):
    sleep(2);
    print x;
    lst.append(x);
    return;

def func2(x, y):
    print current_process().name, x, y;
    return x * y;

def test():
    print "pool"
    pool = Pool(processes=10);
    for i in range(10):
        pool.apply_async(func2, args=[i, i + 1], callback=func);
    pool.close();
    pool.join();

#     snnmf = SNNMF(vocabularySize, 1.5,
#                   xFilePath=os.path.join(dataDir,
#                             "global.fv"),
#                   task="coding", epochNum=1);
#     print getSmSize(snnmf.bTrSm);
#     snnmf.decode();
    return;

if __name__ == '__main__':
    test();
#     print('time={0}s'.format(timeit(stmt='test()', 
#                                     setup='from __main__ import test', 
#                                     number=1)));
    pass
