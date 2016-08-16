'''
Created on May 4, 2013

@author: xwang95
'''

import deep_nlp.sentiment;
import toolkit.variables;
import os;
import random;
import math;
import time;
import toolkit.utility;

class SentiGibbsSampler(object):
    '''
    classdocs
    '''
    ''' Input '''
    refLstTable = None;
    constrLstTable = None;
    citationTxtTable = None;
    wordIndexer = None;
    sentimentAnalyzer = None;
    ''' Variables '''
    posLex = None;
    negLex = None;
    posPriorLangDict = None;
    negPriorLangDict = None;
    citationTxtWordIdTable = None;
    citationTxtWordCntTable = None;
    ''' Output '''
    sentiLabelLstTable = None;
    posteriorLabelTable = None;
    violatedConstraintWeightVector = None;
    ''' Paramemter '''
    sentiConcentrationPrior = None;
    sentiLangPrior = None;
    uniformLangPrior = None;
    ''' Bookkeeping variables '''
    bkSentiCntPerRefIdTable = None;
    bkSentiRefLangDictTable = None;
    
    def __init__(self, refLstTable, constrLstTable, citationTxtTable, wordIndexer, sentimentAnalyzer, sentiConcentrationPrior, sentiLangPrior, uniformLangPrior):
        '''
        Constructor
        '''
        print('[senti_gibbs_sampler]: pass paramemter');
        self.refLstTable = refLstTable;
        self.constrLstTable = constrLstTable;
        self.citationTxtTable = citationTxtTable;
        self.wordIndexer = wordIndexer;
        self.sentimentAnalyzer = sentimentAnalyzer;
        ''' paramemter '''
        self.sentiConcentrationPrior = sentiConcentrationPrior;
        self.sentiLangPrior = sentiLangPrior;
        self.uniformLangPrior = uniformLangPrior;
        ''' data sentiment lex '''
        print('[senti_gibbs_sampler]: load sentiment lexicon');
        self.negLex = set();
        self.posLex = set();        
        self.negPriorLangDict = {};
        self.posPriorLangDict = {};
        for token in self.wordIndexer.wordToIdTable:
            if(token in self.sentimentAnalyzer.negLex): self.negLex.add(token);
            if(token in self.sentimentAnalyzer.posLex): self.posLex.add(token);
        for lex in self.negLex: self.negPriorLangDict[wordIndexer.getWordIndex(lex)] = self.sentiLangPrior;
        for lex in self.posLex: self.posPriorLangDict[wordIndexer.getWordIndex(lex)] = self.sentiLangPrior;
        
        ''' observation '''
        print('[senti_gibbs_sampler]: load observation');
        self.citationTxtWordIdTable = {};  # pmid - ref - wordId : value
        self.citationTxtWordCntTable = {};  # pmid - ref : word count
        for pmid in self.citationTxtTable:
            if(pmid not in self.citationTxtWordIdTable): 
                self.citationTxtWordIdTable[pmid] = {};
                self.citationTxtWordCntTable[pmid] = {};
            for refId in self.citationTxtTable[pmid]:
                if(refId not in self.citationTxtWordIdTable[pmid]): 
                    self.citationTxtWordIdTable[pmid][refId] = {};
                    self.citationTxtWordCntTable[pmid][refId] = 0;
                for wordId in self.wordIndexer.getTokenLstIndexLst(self.citationTxtTable[pmid][refId]):
                    self.citationTxtWordIdTable[pmid][refId][wordId] = self.citationTxtWordIdTable[pmid][refId].get(wordId, 0.0) + 1.0;
                self.citationTxtWordCntTable[pmid][refId] += len(self.citationTxtTable[pmid][refId]); 
        
        ''' sentiLabelLstTable (initialization) '''
        print('[senti_gibbs_sampler]: randomize sentiment labels');
        self.sentiLabelLstTable = {};
        for pmid in self.refLstTable: 
            self.sentiLabelLstTable[pmid] = {};
            for refId in self.refLstTable[pmid]: self.sentiLabelLstTable[pmid][refId] = self.randomBinaryVal();
        
        ''' output '''
        print('[senti_gibbs_sampler]: initialize posterior label table');
        self.posteriorLabelTable = {};
        for pmid in self.refLstTable:
            self.posteriorLabelTable[pmid] = {};
            for refId in self.refLstTable[pmid]: self.posteriorLabelTable[pmid][refId] = [0.0, 0.0];
        
        ''' book-keeping variables '''
        print('[senti_gibbs_sampler]: book-keeping initialization');
        self.bkSentiCntPerRefIdTable = {};  # sentiment proportion prior
        self.bkSentiRefLangDictTable = {};  # language model
        for pmid in self.sentiLabelLstTable:
            for refId in self.sentiLabelLstTable[pmid]:
                ''' bkSentiCntPerRefIdTable '''
                if(refId not in self.bkSentiCntPerRefIdTable): self.bkSentiCntPerRefIdTable[refId] = [0.0, 0.0];
                if(self.sentiLabelLstTable[pmid][refId] == 1): self.bkSentiCntPerRefIdTable[refId][1] += 1.0;
                else: self.bkSentiCntPerRefIdTable[refId][0] += 1.0;
                ''' bkSentiRefLangDictTable '''
                if(refId not in self.bkSentiRefLangDictTable): self.bkSentiRefLangDictTable[refId] = [{}, {}];
                if(self.sentiLabelLstTable[pmid][refId] == 1):
                    for wordId in self.wordIndexer.getTokenLstIndexLst(self.citationTxtTable[pmid][refId]):
                        self.bkSentiRefLangDictTable[refId][1][wordId] = self.bkSentiRefLangDictTable[refId][1].get(wordId, 0.0) + 1.0;
                else:
                    for wordId in self.wordIndexer.getTokenLstIndexLst(self.citationTxtTable[pmid][refId]):
                        self.bkSentiRefLangDictTable[refId][0][wordId] = self.bkSentiRefLangDictTable[refId][0].get(wordId, 0.0) + 1.0;
        return;
    
    def getViolatedConstraintWeight(self, pmid, refId):  # not used
        violatedWeight = 0.0;
        for coCitedId in self.constrLstTable[pmid][refId]:
            if(self.sentiLabelLstTable[pmid][refId] != self.sentiLabelLstTable[pmid][coCitedId]): violatedWeight += self.constrLstTable[pmid][refId][coCitedId];
        return violatedWeight;
    
    def getViolatedConstraintWeightVector(self, pmid, refId):
        violatedWeightVector = [0.0, 0.0];
        for coCitedId in self.constrLstTable[pmid][refId]:
            if(self.sentiLabelLstTable[pmid][coCitedId] == 1): 
                violatedWeightVector[0] += self.constrLstTable[pmid][refId][coCitedId];
            else: violatedWeightVector[1] += self.constrLstTable[pmid][refId][coCitedId];
        return violatedWeightVector;
                
    def getPriorProp(self, pmid, refId):
        violatedWeightVector = self.getViolatedConstraintWeightVector(pmid, refId);
        preferenceVector = [self.sentiConcentrationPrior + self.bkSentiCntPerRefIdTable[refId][0],
                            self.sentiConcentrationPrior + self.bkSentiCntPerRefIdTable[refId][1]];
        priorPropVector = [preferenceVector[0] * math.exp(-violatedWeightVector[0]),
                           preferenceVector[1] * math.exp(-violatedWeightVector[1])];
        return priorPropVector;
    
    def getLikelihood(self, pmid, refId):
        ''' prior: sentiment lexicon + uniform prior
            likelihood: sentiRefLangDictTable '''
        negNormalizerBase = self.uniformLangPrior * self.wordIndexer.vocabularySize + self.sentiLangPrior * len(self.negLex);
        posNormalizerBase = self.uniformLangPrior * self.wordIndexer.vocabularySize + self.sentiLangPrior * len(self.posLex);
        
        ratio = 1;
        for (wordId, wordCnt) in self.citationTxtWordIdTable[pmid][refId].items():
            negFactorBase = self.uniformLangPrior + self.negPriorLangDict.get(wordId, 0.0) + self.bkSentiRefLangDictTable[refId][0].get(wordId, 0.0);
            posFactorBase = self.uniformLangPrior + self.posPriorLangDict.get(wordId, 0.0) + self.bkSentiRefLangDictTable[refId][1].get(wordId, 0.0);
            for i in range(int(wordCnt)): ratio *= (negFactorBase + i) / (posFactorBase + i);
        for i in range(self.citationTxtWordCntTable[pmid][refId]): ratio /= (negNormalizerBase + i) / (posNormalizerBase + i);
        return ratio;
             
    def samplePerPmidRefId(self, pmid, refId):
        oldLabel = self.sentiLabelLstTable[pmid][refId];
        for (wordId, wordCnt) in self.citationTxtWordIdTable[pmid][refId].items():
            self.bkSentiRefLangDictTable[refId][oldLabel][wordId] -= wordCnt;
            if(abs(self.bkSentiRefLangDictTable[refId][oldLabel][wordId]) <= 0.5): del self.bkSentiRefLangDictTable[refId][oldLabel][wordId];
        self.bkSentiCntPerRefIdTable[refId][oldLabel] -= 1.0;
        
        [negPriorProp, posPriorProp] = self.getPriorProp(pmid, refId);
        negProp = negPriorProp * self.getLikelihood(pmid, refId);
        posProp = posPriorProp;
        p = negProp / (negProp + posProp);
        sampledLabel = self.randomBinaryVal(p);
        ''' book-keeping '''
        for (wordId, wordCnt) in self.citationTxtWordIdTable[pmid][refId].items():
            self.bkSentiRefLangDictTable[refId][sampledLabel][wordId] = self.bkSentiRefLangDictTable[refId][sampledLabel].get(wordId, 0.0) + wordCnt;
        self.bkSentiCntPerRefIdTable[refId][sampledLabel] += 1.0;
        self.sentiLabelLstTable[pmid][refId] = sampledLabel;
        return;
    
    def sampleIteration(self, posteriorLabelTable=None):
        for pmid in self.refLstTable:
            for refId in self.refLstTable[pmid]:
                self.samplePerPmidRefId(pmid, refId);
                if(posteriorLabelTable is not None): posteriorLabelTable[pmid][refId][self.sentiLabelLstTable[pmid][refId]] += 1.0;
        return;            
    
    def monitorViolatedConstraintWeight(self):
        return sum([sum([self.getViolatedConstraintWeight(pmid, refId) for refId in self.constrLstTable[pmid]]) for pmid in self.constrLstTable]);
        
    def randomBinaryVal(self, p=None):
        if(p is None): p = 0.5;
        if(random.random() > p): return 1;
        else: return 0;

    def run(self, burnInHr, sampliHr):
        violatedConstraintWeightLst = [];
        violatedConstraintWeight = self.monitorViolatedConstraintWeight();
        violatedConstraintWeightLst.append(violatedConstraintWeight);
        print('[senti_gibbs_sampler]: burn-in hour = {0} hr'.format(burnInHr));
        print('                       Progress:');
        (prog, step) = (0.0, 0.05);
        timeStart = time.clock();
        iterNum = 0;
        while(True):
            timeNow = time.clock();
            prog = ((timeNow - timeStart) / 3600.0) / burnInHr;
            toolkit.utility.printProgressBar(prog, step, 'iter = {0}, elapsed_time = {1} sec, violate = {2}'.format(iterNum, timeNow - timeStart, violatedConstraintWeight));
            if(prog >= 1.0): break;
            self.sampleIteration();
            violatedConstraintWeight = self.monitorViolatedConstraintWeight();
            violatedConstraintWeightLst.append(violatedConstraintWeight);
            iterNum += 1;
        print('');
        print('[senti_gibbs_sampler]: samplin hour = {0} hr'.format(sampliHr));
        print('                       Progress:');
        (prog, step) = (0.0, 0.05);
        timeStart = time.clock();
        iterNum = 0;
        while(True):
            timeNow = time.clock();
            prog = ((timeNow - timeStart) / 3600.0) / burnInHr;
            toolkit.utility.printProgressBar(prog, step, 'iter = {0}, elapsed_time = {1} sec, violate = {2}'.format(iterNum, timeNow - timeStart, violatedConstraintWeight));
            if(prog >= 1.0): break;
            self.sampleIteration(self.posteriorLabelTable);
            violatedConstraintWeight = self.monitorViolatedConstraintWeight();
            violatedConstraintWeightLst.append(violatedConstraintWeight);
            iterNum += 1;
        print('');
        self.violatedConstraintWeightVector = violatedConstraintWeightLst;
        return; 
    
    def dumpPosteriorLabelTableFile(self, dumpFilePath):
        dumpPosteriorLabelTableFile(dumpFilePath, self.posteriorLabelTable);
    
    def dumpViolatedConstraintWeightFile(self, dumpFilePath):
        dumpFile = open(dumpFilePath, 'w');
        for x in self.violatedConstraintWeightVector: dumpFile.write('{0}\n'.format(x));
        dumpFile.close();
        return;
    
def dumpPosteriorLabelTableFile(dumpFilePath, posteriorLabelTable):
    dumpFile = open(dumpFilePath, 'w');
    for pmid in posteriorLabelTable:
        for refId in posteriorLabelTable[pmid]:
            dumpFile.write('{0}<=:|:=>{1}<=:|:=>{2}<=:|:=>{3}\n'.format(pmid, refId, posteriorLabelTable[pmid][refId][0], posteriorLabelTable[pmid][refId][1]));
    dumpFile.close();
    return;

def loadPosteriorLabelTableFile(dumpFilePath):
    dumpFile = open(dumpFilePath, 'r');
    posteriorLabelTable = {};
    lineLSt = dumpFile.readlines();
    lineLSt = [ln.strip() for ln in lineLSt if(ln.strip())];
    dumpFile.close();
    for ln in lineLSt: 
        [pmid, refId, negCnt, posCnt] = [toolkit.utility.parseNumVal(x) for x in ln.split('<=:|:=>')];
        if(pmid not in posteriorLabelTable): posteriorLabelTable[pmid] = {};
        if(refId not in posteriorLabelTable[pmid]): posteriorLabelTable[pmid][refId] = [negCnt, posCnt];
    return posteriorLabelTable;
