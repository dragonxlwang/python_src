'''
Created on Jul 9, 2014

@author: xwang1
'''
from linkedin.const import *
from toolkit.num.sparse import getSmDat, setSmRow, getSvKeys, getSvVals, \
    getSvDat, getSvElem, getSmSize, getSmRow, getSmRowIdxLst, transposeSm, \
    getSmColLen, getSmL1Norm, getSmL2Norm, mulSmSm, subSmSm
from toolkit.utility import loadObjFromFile, writeObjToFile
from time import time
import math
import sys;
import os;
from toolkit.num.algebra import subMatMat, mulMatMat
import argparse
from argparse import ArgumentParser

def _seqToSv(seq, dim):
    d = {};
    for x in seq: d[x] = d.get(x, 0.0) + 1.0;
    return (d, dim);

def loadVocMap():
    #------------------------------------------------------ load vocabulary file
    st = time();
    print('start loading vocabulary')
    vocLst = loadObjFromFile(globalVocFilteredLstPath);    
    vocDict = {};
    for i in range(len(vocLst)): vocDict[vocLst[i]] = i;
    word2id = vocDict;
    id2word = {};
    for w in word2id: id2word[word2id[w]] = w;
    print('\t takes time {0}, load {1} words'.format(time() - st, len(vocLst)));
    return (word2id, id2word);
        
        
def genInitBTrSm(ratioLtoN):
    st = time();
    print('start initializing bTrSm');
    (word2id, id2word) = loadVocMap();
    n = vocabularySize;
    l = int(ratioLtoN * n);
    bTrSm = ({}, (l, n));
    #-------------------------------------------- initialize phrase list by word
    for i in range(n): setSmRow(bTrSm, i, ({i:1.0}, n));
    #---------------------------------- load phrase list sorted by count
    ipFile = open(initialPhraseFilePath);
    k = n;
    bUniStrSet = set();
    for ln in ipFile:
        (ph, c) = ln.split('\t');
        tokIdLst = [word2id[x] for x in [w.lower() for w in ph.split()] 
                    if x in word2id];
        bSv = _seqToSv(tokIdLst, n);
        if(len(getSvKeys(bSv)) <= 1): continue;
        s = math.sqrt(sum([x * x for x in getSvVals(bSv)]));
        for i in getSvKeys(bSv): getSvDat(bSv)[i] /= s;
        uniStr = '\xaa'.join(['{0}:{1}'.format(i, getSvElem(bSv, i)) 
                              for i in sorted(getSvKeys(bSv))]);
        if(uniStr in bUniStrSet): continue;
        else: bUniStrSet.add(uniStr);
        setSmRow(bTrSm, k, bSv);
        k += 1;        
        if(k == l): break;
    ipFile.close(); 
    print('\t takes time {0}'.format(time() - st));
    return bTrSm;

def analyzeBTrSm(dumpDir , iterNum, filtered=True):     
    bTrSm = loadObjFromFile(os.path.join(dumpDir,
                                        "{0}.bTrSm".format(iterNum)));
    (word2id, id2word) = loadVocMap();
    file = open(os.path.join(dumpDir,
                             "{0}.basisTxt{1}".format(iterNum,
                                                       ".filtered" if filtered 
                                                       else "")), "w");
    
    unigramIdxLst = [];
    print getSmSize(bTrSm);
    print 'phrase no: {0}'.format(len(getSmRowIdxLst(bTrSm)));
    for k in sorted(getSmRowIdxLst(bTrSm),
                    key=lambda k:-len(getSvKeys(getSmRow(bTrSm, k)))):
        if(len(getSvKeys(getSmRow(bTrSm, k))) == 1):
            unigramIdxLst.append(k);
            continue;
        bSv = getSmRow(bTrSm, k);
        #=======================================================================
        # 
        #=======================================================================
#         skipFlag = False;
#         for v in getSvVals(bSv):
#             if(v < 1.0): 
#                 skipFlag = True;
#                 break;
#         if(skipFlag): continue;
        #=======================================================================
        # 
        #=======================================================================
        if(filtered):
            vec = ["{0}:{1}".format(id2word[k], getSvElem(bSv, k))
                            for k in sorted(getSvKeys(bSv),
                                            key=lambda x: getSvElem(bSv, x))
                                  if(getSvElem(bSv, k) > 0.1)
                                  ];
        else: 
            vec = ["{0}:{1}".format(id2word[k], getSvElem(bSv, k))
                            for k in sorted(getSvKeys(bSv),
                                            key=lambda x: getSvElem(bSv, x))];
        if(len(vec) > 1): file.write('\t'.join(vec) + "\n");
    unigramSet = set();
    print 'unigram no: {0}'.format(len(unigramIdxLst));
    for k in sorted(unigramIdxLst,
                    key=lambda k: id2word[getSvKeys(getSmRow(bTrSm, k))[0]]):
        bSv = getSmRow(bTrSm, k);
        flag = False;
        if(id2word[getSvKeys(getSmRow(bTrSm, k))[0]] in unigramSet):
            print flag;
            flag = True;
        unigramSet.add(id2word[getSvKeys(getSmRow(bTrSm, k))[0]]);
        file.write('\t'.join(["{0}:{1}".format(id2word[k], getSvElem(bSv, k))
                        for k in sorted(getSvKeys(bSv),
                                        key=lambda x: getSvElem(bSv, x))]) 
                   + "\t\t" + ("DUPDUPDUP" if flag else "") + "\n");
    file.close();
    return;
        
def basisToString(bSv, id2word, multiplier):
    vec = ["{0}:{1}".format(id2word[i], getSvElem(bSv, i) * multiplier)
                            for i in sorted(getSvKeys(bSv),
                                            key=lambda x:-getSvElem(bSv, x))];
    bString = '\t'.join(vec);
    return bString;
  
def analyzeSSm(settingId, iterNum):
    if(settingId == 3):
        (settings, settingDir) = (setting3, globalSetting3Dir);
    elif(settingId == 4):
        (settings, settingDir) = (setting4, globalSetting4Dir);
    elif(settingId == 5):
        (settings, settingDir) = (setting5, globalSetting5Dir);
    elif(settingId == 6):
        (settings, settingDir) = (setting6, globalSetting6Dir); 
    sSm = loadObjFromFile(os.path.join(settingDir,
                                       "{0}.sSm".format(iterNum)));
    bTrSm = loadObjFromFile(os.path.join(settingDir,
                                         "{0}.bTrSm".format(iterNum)));
    xSm = loadObjFromFile(globalXSmFilePath);
    (word2id, id2word) = loadVocMap();
    xTrSm = transposeSm(xSm);
    sTrSm = transposeSm(sSm);
    (n, m) = getSmSize(xSm);
    l = getSmColLen(bTrSm);
    
    recLst = loadObjFromFile(globalFvFilePath);
    
    j = 0;
    for record in recLst:
        (fv, tag, comment) = record;
        print comment;
        for jj in range(len(fv)):
            xSv = getSmRow(xTrSm, j + jj);
            print('==>:' + '\t'.join([id2word[i] for i in getSvKeys(xSv)]));
            sSv = getSmRow(sTrSm, j + jj);
            for k in getSvKeys(sSv):
                print('\t[{0}:{1}]: {2}'.format(k, getSvElem(sSv, k),
                                basisToString(getSmRow(bTrSm, k), id2word,
                                              getSvElem(sSv, k))));
            print('');
        print 80 * '~';
        j += len(fv);
    return;

def modelObjective(settingId, iterNum):
    if(settingId == 3):
        (settings, settingDir) = (setting3, globalSetting3Dir);
    elif(settingId == 4):
        (settings, settingDir) = (setting4, globalSetting4Dir);
    elif(settingId == 5):
        (settings, settingDir) = (setting5, globalSetting5Dir);
    elif(settingId == 6):
        (settings, settingDir) = (setting6, globalSetting6Dir);
    (lambdaS, lambdaB, miuS, miuB, lbL1ReweightMethod, lbL2ReweightMethod,
                        lbAddInfo) = settings; 
    sSm = loadObjFromFile(os.path.join(settingDir,
                                       "{0}.sSm".format(iterNum)));
    bTrSm = loadObjFromFile(os.path.join(settingDir,
                                         "{0}.bTrSm".format(iterNum)));
    xSm = loadObjFromFile(globalXSmFilePath);
    xTrSm = transposeSm(xSm);
    sTrSm = transposeSm(sSm);
    (n, m) = getSmSize(xSm);
    l = getSmColLen(bTrSm);
    objBRg = lambdaB * getSmL1Norm(bTrSm) + 0.5 * miuB * getSmL2Norm(bTrSm);
    objSRg = lambdaS * getSmL1Norm(sTrSm) + 0.5 * miuS * getSmL2Norm(sTrSm);
    objLs = getSmL2Norm(subSmSm(xTrSm, mulSmSm(sTrSm, bTrSm,
                                               procNum=5),
                                procNum=5));
    objF = objLs + objBRg + objSRg;
    return (objF, objLs, objBRg, objSRg);
    
if __name__ == '__main__':
    dumpDir = "/home/xwang1/data/"\
              "feedback/global.bTrSm.init_0.2_0.001_1e-06_1e-06_concave_log_None_None";
    for i in range(1, 15, 1):
        analyzeBTrSm(dumpDir, iterNum=i, filtered=True);
#     for i in range(100, 201, 20):
#         analyzeBTrSm(dumpDir, iterNum=i, filtered=True);
    pass
