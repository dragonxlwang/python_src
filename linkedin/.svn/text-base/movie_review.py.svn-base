'''
Created on Aug 4, 2014

@author: xwang1
'''
import os;
import string;
import sys;
# from deep_nlp.text import stemWithSnowballStemmer, stopwordsSet
from toolkit.utility import writeObjToFile, loadObjFromFile, writeMatrix, \
    readMatrix, splitCrossValidation
from toolkit.num.sparse import toSparseMat, toSparseVec, transposeSm, getSmDat, \
    getSvKeys, getSvElem, getSmRow, getSvVals, getSmSize, getSmRowIdxLst, \
    getSvL1Norm, getSvL2Norm
from toolkit.num.algebra import ones, randomMat
from linkedin.snnmf3 import SNNMF2
from random import randint
import math

class MovieReviewData(object):
    puncSet = set(string.punctuation);
    
    dataDir = "/home/xwang1/data/txt_sentoken";
    negDataDir = "/home/xwang1/data/txt_sentoken/neg";
    posDataDir = "/home/xwang1/data/txt_sentoken/pos";
    
    word2Id = None;
    id2Word = None;
    vocabularySize = 3000;
    
    def _ifPunc(self, word):
        for c in word:
            if(c not in self.puncSet): 
                return False;
        return True;
    
    def clauseSeg(self, ln):
        tokLst = ln.split();
        clauseLst = [];
        i = 0;
        while(i < len(tokLst)):
            j = i;
            while(j < len(tokLst)):
                if(self._ifPunc(tokLst[j])): break;
                else: j += 1;
            clauseLst.append(tokLst[i:j]);
            i = j;
            while(i < len(tokLst) and self._ifPunc(tokLst[i])): i += 1;
        return clauseLst;
    
    def buildVocabulary(self):
#         filePathLst = [os.path.join(self.negDataDir, f) 
#                        for f in os.listdir(self.negDataDir)] + \
#                        [os.path.join(self.posDataDir, f)
#                         for f in os.listdir(self.posDataDir)];
#         wordFreqTable = {};
#         for filePath in filePathLst:
#             file = open(filePath);
#             for ln in file:
#                 tokLst = [w for w in ln.split() if(not self._ifPunc(w))];
#                 tokLst = [stemWithSnowballStemmer(w) for w in tokLst];
#                 tokLst = [w for w in tokLst if(w not in stopwordsSet)];
#                 for tok in tokLst:
#                     wordFreqTable[tok] = wordFreqTable.get(tok, 0) + 1;
#             file.close();
#         wLst = [w for w in 
#                 sorted(wordFreqTable, key=lambda x:-wordFreqTable[x])];
#         wLst = wLst[0:self.vocabularySize];
#         self.word2Id = {};
#         self.id2Word = {};
#         for i in range(self.vocabularySize):
#             self.word2Id[wLst[i]] = i;
#             self.id2Word[i] = wLst[i];            
        return;
    
    def construct(self):
        filePathLst = [os.path.join(self.negDataDir, f) 
                       for f in os.listdir(self.negDataDir)] + \
                       [os.path.join(self.posDataDir, f)
                        for f in os.listdir(self.posDataDir)];
        cid = 0;
        dat = {};
        for filePath in filePathLst:
            file = open(filePath);
            for ln in file:
                cLst = self.clauseSeg(ln);
                for cTokLst in cLst:
                    wVec = [self.word2Id[w] for w in cTokLst
                                           if w in self.word2Id];
                    sv = toSparseVec(keys=wVec,
                                     vals=ones(len(wVec)),
                                     dim=self.vocabularySize);
                    dat[cid] = sv;
                    cid += 1;
            file.close();
        xTrSm = (dat, (cid, self.vocabularySize));                    
        return xTrSm;
    
    def __init__(self):
        if(not os.path.exists(os.path.join(self.dataDir, "word2Id"))):
            self.buildVocabulary();
            writeObjToFile(os.path.join(self.dataDir, "word2Id"), self.word2Id);
            writeObjToFile(os.path.join(self.dataDir, "id2Word"), self.id2Word);
        else:
            self.word2Id = loadObjFromFile(os.path.join(self.dataDir,
                                                        "word2Id"));
            self.id2Word = loadObjFromFile(os.path.join(self.dataDir,
                                                        "id2Word"));
        if(not os.path.exists(os.path.join(self.dataDir, "xSm"))):
            xTrSm = self.construct();
            xSm = transposeSm(xTrSm);
            writeObjToFile(os.path.join(self.dataDir, "xSm"), xSm);
            writeObjToFile(os.path.join(self.dataDir, "xTrSm"), xTrSm); 
        return;
    
    def initBTrMatRandomDenseGen(self, l):
        fp = os.path.join(self.dataDir, "initBTrMatRandomDense_{0}".format(l));
        if(not os.path.exists(fp)):
            initBTr = randomMat(l, self.vocabularySize);
            with open(fp, "w") as writer: writeMatrix(initBTr, writer);
        return;
    
    def initBTrSmRandomSparseGen(self, l, k):
        fp = os.path.join(self.dataDir,
                          "initBTrSmRandomSparse_{0}_{1}".format(l, k));
        if(not os.path.exists(fp)):
            mat = {};
            for j in range(l):
                lst = [randint(0, self.vocabularySize - 1) for i in range(k)];
                sv = toSparseVec(keys=lst, vals=ones(k),
                                 dim=self.vocabularySize);
                mat[j] = sv;
            bTrSm = (mat, (l, self.vocabularySize));
            writeObjToFile(fp, bTrSm);   
            return;
        
    def snnmfTraining(self,
                      learningParams):
        (initBTrSm, lambdaS, lambdaB, miuS, miuB,
         lbL1ReweightMethod, lbL2ReweightMethod, lbAddInfo) = learningParams;
        print("learningParams: {0}".format(learningParams));
        if(initBTrSm.startswith("initBTrMatRandomDense_")):
            l = int(initBTrSm.replace("initBTrMatRandomDense_", ""));
            self.initBTrMatRandomDenseGen(l);
            with open(os.path.join(self.dataDir, initBTrSm)) as reader: 
                (bTrMat, eof) = readMatrix(reader);
        elif(initBTrSm.startswith("initBTrSmRandomSparse_")):
            (l, k) = [int(x) for x in 
                initBTrSm.replace("initBTrSmRandomSparse_", "").split("_")];
            self.initBTrSmRandomSparseGen(l, k);
            
        learningParamsDecor = '_'.join([str(x) for x in learningParams]);
        outputDir = os.path.join(self.dataDir, learningParamsDecor);
        snnmf = SNNMF2(
                    xSmFilePath=os.path.join(self.dataDir, "xSm"),
                    bTrSmFilePath=os.path.join(self.dataDir, initBTrSm),
                    lambdaS=lambdaS, lambdaB=lambdaB, miuS=miuS, miuB=miuB,
                    procNum=10,
                    outputDir=outputDir,
                    startingFrom=None,  # ("learnBasis", 20),
                    learnBasisOptions=(lbL1ReweightMethod,
                                       lbL2ReweightMethod,
                                       lbAddInfo));
        snnmf.work();
        return;
    
    def analyzeBTrSm(self, learningParams, iterNum):
        learningParamsDecor = '_'.join([str(x) for x in learningParams]);
        outputDir = os.path.join(self.dataDir, learningParamsDecor);
        bTrSm = loadObjFromFile(os.path.join(outputDir,
                                             "{0}.bTrSm".format(iterNum)));
        basisTxtFilePath = os.path.join(outputDir,
                                        "{0}.basisTxt".format(iterNum));
        basisTxtFile = open(basisTxtFilePath, "w");
        for k in getSmRowIdxLst(bTrSm):
            bSv = getSmRow(bTrSm, k);
            iLst = sorted(getSvKeys(bSv), key=lambda x:-getSvElem(bSv, x));
            ln = str(k) + ": " + "\t".join(["{0}:{1}".format(self.id2Word[i],
                                             getSvElem(bSv, i)) 
                            for i in iLst]);
            basisTxtFile.write(ln + "\n");
            print ln;
        basisTxtFile.close();
                
    def unigramFeatureExtract(self):
#         def feaExt(fp, label):
#             with open(fp) as file:
#                 fv = {};
#                 for ln in file:
#                     stemLst = [stemWithSnowballStemmer(w) for w in ln.split()];
#                     vec = [self.word2Id[x] for x in stemLst
#                            if x in self.word2Id];
#                     for i in vec: fv[i] = fv.get(i, 0.0) + 1.0;
#                 nm = math.sqrt(sum([x * x for x in fv.values()]));
#                 fvLn = "{0}\t{1}".format(label,
#                                         " ".join(["{0}:{1}".format(x + 1,
#                                                         fv[x] / nm)
#                                                   for x in sorted(fv)]));
#             return fvLn;
#         negFilePathLst = [os.path.join(self.negDataDir, f) 
#                        for f in os.listdir(self.negDataDir)]
#         posFilePathLst = [os.path.join(self.posDataDir, f)
#                         for f in os.listdir(self.posDataDir)];
#         fvLnLst = [];
#         for fp in negFilePathLst: fvLnLst.append(feaExt(fp, "-1"));
#         for fp in posFilePathLst: fvLnLst.append(feaExt(fp, "+1"));
#         splitCrossValidation(fvLnLst, foldNum=5,
#                              outputDir=os.path.join(self.dataDir, "unigram"));
        return;
    
    def phraseFeatureExtract(self, learningParams, iterNum, presence=False):
        learningParamsDecor = '_'.join([str(x) for x in learningParams]);
        outputDir = os.path.join(self.dataDir, learningParamsDecor);
        bTrSm = loadObjFromFile(os.path.join(outputDir,
                                             "{0}.bTrSm".format(iterNum)));
        bNorm = {};
        for k in getSmRowIdxLst(bTrSm):
            bNorm[k] = getSvL1Norm(getSmRow(bTrSm, k));
#             bNorm[k] = math.sqrt(getSvL2Norm(getSmRow(bTrSm, k)));
            
        sSm = loadObjFromFile(os.path.join(outputDir,
                                           "{0}.sSm".format(iterNum)));
        sTrSm = transposeSm(sSm);
        filePathLst = [(os.path.join(self.negDataDir, f), "-1") 
                       for f in os.listdir(self.negDataDir)] + \
                       [(os.path.join(self.posDataDir, f), "+1")
                        for f in os.listdir(self.posDataDir)];
        cid = 0;
        fvLnLst = [];
        for (filePath, label) in filePathLst:
            with open(filePath) as file:
                fv = {};
                for ln in file:
                    cLst = self.clauseSeg(ln);
                    for cTokLst in cLst:
                        sSv = getSmRow(sTrSm, cid);
                        for x in getSvKeys(sSv):
                            if(not presence):
                                fv[x] = fv.get(x, 0.0) + \
                                    getSvElem(sSv, x) * bNorm[x];
                            else:
                                if(x not in fv): fv[x] = 1.0;                                        
                        cid += 1;
                nm = math.sqrt(sum([x * x for x in fv.values()]));
                fvLn = "{0}\t{1}".format(label,
                                         " ".join(["{0}:{1}".format(x + 1,
                                                                    fv[x] / nm)
                                                   for x in sorted(fv)]));
                fvLnLst.append(fvLn);
        print cid;
        print getSmSize(sTrSm);
        splitCrossValidation(fvLnLst, foldNum=5,
            outputDir=os.path.join(self.dataDir,
                "phrase_" + learningParamsDecor + "_{0}".format(iterNum) + \
                ("_presence" if presence else "")));
        return;
    
if __name__ == '__main__':
    mrd = MovieReviewData();
    
    learningParams = ("initBTrSmRandomSparse_9000_3",
                                      1e-2, 1e-2, 1e-6, 1e-6, "concave_log",
                                      None, None);
#     (initBTrSm, lambdaS, lambdaB, miuS, miuB,
#          lbL1ReweightMethod, lbL2ReweightMethod, lbAddInfo) = learningParams;

    #------------------------------------------------------------- analyze bTrSm
#     mrd.analyzeBTrSm(learningParams, iterNum=5);
#     mrd.analyzeBTrSm(learningParams, iterNum=10);
#     mrd.analyzeBTrSm(learningParams, iterNum=15);
#     mrd.analyzeBTrSm(learningParams, iterNum=20);
#     mrd.analyzeBTrSm(learningParams, iterNum=30);
#     mrd.analyzeBTrSm(learningParams, iterNum=40);
#     mrd.analyzeBTrSm(learningParams, iterNum=50);
#     
    
    #------------------------------------------------ unigram feature extraction
    #===========================================================================
    # mrd.unigramFeatureExtract();
    #===========================================================================
    
    #------------------------------------------------- phrase feature extraction
    #===========================================================================
    # mrd.phraseFeatureExtract(learningParams, iterNum=200, presence=True);
    #===========================================================================
    
    lmbdaS = float(sys.argv[1]);
    lmbdaB = float(sys.argv[2]);
    reweightMethod = sys.argv[3];
    if(reweightMethod == "None"): reweightMethod = None;
    learningParams = ("initBTrSmRandomSparse_9000_3",
                                      lmbdaS, lmbdaB, 1e-6, 1e-6, reweightMethod,
                                      None, None);
    mrd.snnmfTraining(learningParams=learningParams);
    pass;
