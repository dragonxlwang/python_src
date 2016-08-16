'''
Created on Jul 29, 2014

@author: xwang1
'''
from toolkit.num.probability import poissonSampling, multinomialSampling
from toolkit.num.sparse import toSparseVec, toSparseMat, getSmDat, getSvKeys, \
    getSvElem, getSvL2Norm
from toolkit.num.algebra import ones, zeroes, mulMatMat, transposeMat, randomMat
from random import randint
from toolkit.utility import mkDir, writeObjToFile, loadObjFromFile
import os;
from linkedin.const import dataDir
import math
from scipy.stats.mstats_basic import threshold
from linkedin.snnmf import lst

#===============================================================================
# Settings
#===============================================================================

paraN = 50;
paraM = 1000;
paraL = 250;


class ToyDataGenerator(object):
    '''
    classdocs
    '''
    n = None;  # vocabulary size
    m = None;  # sentence size
    l = None;  # phrase size
    
    unigramDist = None;  # zipf's law enforced
    poiPhLenParam = None;  # poisson distributed phrase length
    poiSeLenParam = None;  # poisson distributed sentence length of phrases
    bMat = None;
    sMat = None;
    xMat = None;
    
    dumpDir = None;
    
    def construct(self):
        #---------------------------------------- construct unigram distribution
        s = sum([1.0 / (i + 1) for i in range(self.n)]);
        self.unigramDist = [1.0 / ((i + 1) * s) for i in range(self.n)];
        #------------------------------------- phrase poisson length distributed
        plLst = [x + 1 for x in 
                 poissonSampling(lmbda=self.poiPhLenParam, size=self.l)];
        #----------------------------------------------------------- construct B
        self.bMat = zeroes(self.n, self.l);
        for k in range(self.l):
            wlst = [multinomialSampling(self.unigramDist) 
                    for i in range(plLst[k])];
            for i in wlst: self.bMat[i][k] += 1.0;
        #----------------------------------------------------------- construct S
        self.sMat = zeroes(self.l, self.m);
        slLst = [x + 1 for x in 
                 poissonSampling(lmbda=self.poiSeLenParam, size=self.m)];
        for j in range(self.m):
            pLst = [randint(0, self.l - 1) for k in range(slLst[j])];
            for k in pLst: self.sMat[k][j] += 1.0;
        #----------------------------------------------------------- construct X
        self.xMat = mulMatMat(self.bMat, self.sMat);
        return;
    
    def dump(self):
        mkDir(self.dumpDir);
        print("write bMat");
        writeObjToFile(os.path.join(self.dumpDir, "bMat"), self.bMat);
        print("write sMat");
        writeObjToFile(os.path.join(self.dumpDir, "sMat"), self.sMat);
        print("write xMat");
        writeObjToFile(os.path.join(self.dumpDir, "xMat"), self.xMat);
        xSm = toSparseMat(self.xMat);
        print("write xSm");
        writeObjToFile(os.path.join(self.dumpDir, "xSm"), xSm);
        bTrSm = toSparseMat(transposeMat(self.bMat));
        print("write bTrSm");
        writeObjToFile(os.path.join(self.dumpDir, "bTrSm"), bTrSm);
        initBTrSmRandomDense = toSparseMat(randomMat(self.l, self.n, 0.0, 1.0));
        print("write initBTrSmRandomDense");
        writeObjToFile(os.path.join(self.dumpDir, "initBTrSmRandomDense"),
                       initBTrSmRandomDense);
        return;
    
    def __init__(self, n=paraN, m=paraM, l=paraL, poiPhLenParam=1.8,
                 poiSeLenParam=3.0, dumpDir=None):
        '''
        Constructor
        '''
        (self.n, self.m, self.l) = (n, m, l);
        self.poiPhLenParam = poiPhLenParam;
        self.poiSeLenParam = poiSeLenParam;
        if(dumpDir is None):
            decor = "n_{0}_m_{1}_l_{2}_poiPhLen_{3}_poiSeLen_{4}".format(n, m,
                    l, poiPhLenParam, poiSeLenParam);
            dumpDir = os.path.join(dataDir, "snnmf", "toy", decor);
        self.dumpDir = dumpDir;
        return;

class ToyDataEvaluator(object):
    n = None;  # vocabulary size
    m = None;  # sentence size
    l = None;  # phrase size
    
    unigramDist = None;  # zipf's law enforced
    poiPhLenParam = None;  # poisson distributed phrase length
    poiSeLenParam = None;  # poisson distributed sentence length of phrases
    bMat = None;
    sMat = None;
    xMat = None;
    
    dumpDir = None;
    learningParams = None;
    bTrSmLearned = None;
    iterNum = None;
    
    def _hashBSv(self, bSv, threshold):
        s = math.sqrt(getSvL2Norm(bSv));
        lst = sorted([x for x in getSvKeys(bSv) 
                      if(getSvElem(bSv, x) / s >= threshold)]);
        hashStr = str(lst);
        return hashStr;
    
    def basisEval(self):
        learningParamDecor = "toy_" + '_'.join([str(x) 
                                                for x in self.learningParams]);
        toyDataExpDir = os.path.join(self.dumpDir, learningParamDecor);
        self.bTrSmLearned = loadObjFromFile(os.path.join(self.dumpDir,
                                            toyDataExpDir,
                                            "{0}.bTrSm".format(self.iterNum)));
        #=======================================================================
        # load finished
        #=======================================================================
        hit = 0;
        bStrSet = set();
        hitBSvSet = set();
        uniBSvSet = set();
        for k in range(self.l):
            bStrSet.add(str([i for i in range(self.n) 
                             if(self.bMat[i][k] != 0)]));
        for bSv in getSmDat(self.bTrSmLearned).values():
            hashStr = self._hashBSv(bSv, threshold=0.1);
            uniBSvSet.add(hashStr);
            if(hashStr in bStrSet):
                hit += 1;
                hitBSvSet.add(hashStr);
#         print hit, self.l, len(getSmDat(self.bTrSmLearned)), len(hitBSvSet);
#         for bSv in getSmDat(self.bTrSmLearned).values():
#             for 
        return (hit, self.l, len(uniBSvSet), len(hitBSvSet));
    
    def load(self):
        self.bMat = loadObjFromFile(os.path.join(self.dumpDir, "bMat"));
        self.sMat = loadObjFromFile(os.path.join(self.dumpDir, "sMat"));
        self.xMat = loadObjFromFile(os.path.join(self.dumpDir, "xMat"));
        return;
        
    def __init__(self, n=paraN, m=paraM, l=paraL, poiPhLenParam=1.8,
                 poiSeLenParam=3.0,
                 learningParams=["initBTrSmRandomDense", 1e-3, 1e-3,
                                 1e-5, 1e-5, "concave_log", None, None],
                 iterNum=100, dumpDir=None):
#       "reverse_idf"  "concave_log",
#                                                     None, 
        (self.n, self.m, self.l) = (n, m, l);
        self.poiPhLenParam = poiPhLenParam;
        self.poiSeLenParam = poiSeLenParam;
        if(dumpDir is None):
            decor = "n_{0}_m_{1}_l_{2}_poiPhLen_{3}_poiSeLen_{4}".format(n, m,
                    l, poiPhLenParam, poiSeLenParam);
            dumpDir = os.path.join(dataDir, "snnmf", "toy2", decor);
        self.dumpDir = dumpDir;
        self.learningParams = learningParams;
        self.iterNum = iterNum;
        return;
    
def initBTrSmRandomDenseGen(n, l, dir):
    initBTrSmRandomDense = toSparseMat(randomMat(l, n, 0.0, 1.0));
    print("write initBTrSmRandomDense");
    writeObjToFile(os.path.join(dir,
                                "initBTrSmRandomDense_{0}_{1}".format(l, n)),
                                initBTrSmRandomDense);
    return;
        
def convertBTrSmToReadableTxt(filePath, bTrSmFilePath):
    threshold = 0.1;
    bTrSmLearned = loadObjFromFile(filePath);
    bTrSm = loadObjFromFile(bTrSmFilePath);
    bSvSet = set();
    for bSv in getSmDat(bTrSm).values():
        s = math.sqrt(getSvL2Norm(bSv));
        lst = sorted([x for x in getSvKeys(bSv) 
                          if(getSvElem(bSv, x) / s >= threshold)]);
        bSvSet.add(str(lst));    
    txtFile = open(filePath + ".txt", "w");
    for bSv in getSmDat(bTrSmLearned).values():
        s = math.sqrt(getSvL2Norm(bSv));
        lst = sorted([x for x in getSvKeys(bSv) 
                          if(getSvElem(bSv, x) / s >= threshold)]);
        txtFile.write(str(lst) + (" correct" if str(lst) in bSvSet else "")
                       + '\n');
    txtFile.close();
    return;

if(__name__ == "__main__"):
#     tdg = ToyDataGenerator();
#     tdg.construct();
#     tdg.dump();

    tde = ToyDataEvaluator();
    tde.load();
    print tde.dumpDir;
    print tde.learningParams;
    for tde.iterNum in range(1, 201):
        ret = tde.basisEval();
        print("[{0}]: {1}".format(tde.iterNum, " ".join([str(x) 
                                                         for x in ret])));
        
#         convertBTrSmToReadableTxt("/home/xwang1/data/snnmf/toy/"
#                                   "n_50_m_1000_l_250_poiPhLen_1.8_poiSeLen_3.0/"
#                                 "toy_initBTrSmRandomDense_0.0001_0.0001_1e-05_1e-05_None_None_None/"
#                                 "200.bTrSm",
#                                 "/home/xwang1/data/snnmf/toy/"
#                                   "n_50_m_1000_l_250_poiPhLen_1.8_poiSeLen_3.0/"
#                                 "bTrSm",);

#     initBTrSmRandomDenseGen(n=paraN, l=int(paraL * 2.0),
#                             dir="/home/xwang1/data/snnmf/toy/" \
#                             "n_50_m_1000_l_250_poiPhLen_1.8_poiSeLen_3.0");
