'''
Created on Jul 16, 2013

@author: wangxl
'''
import optparse;
import sys;
import bcolor;
sys.path.append('.');
import random;
import utility;
import itertools;
import math;
import os;
import re;

class Inferer(object):
    '''
    classdocs
    '''
    layoutPriorDist = None;
    expertModel = None;
    vSlots = None;
    layoutToIndex = None;
    indexToLayout = None;
    layoutNum = None;
    vNameLst = [];
    vNameToIndex = None;
    indexToVName = None;
    vNum = None;
    ediNum = None;
    mtNum = None;
    slotDiffMap = None;
    marginLayoutIdx = None;
    data = None;
    vSetSet = None;
    
    def __init__(self, ediNum, mtNum):
        '''
        Constructor
        '''
        # model
        self.layoutPriorDist = [];  # initialized by initialPriorModel
        self.expertModel = {};  # initialized by initialExpertModel
        self.ediNum = ediNum;  # parameter
        self.mtNum = mtNum;  # parameter
        # constants 
        self.vSlots = {0: [1, 2, 3, 4, 11, None],
                       1: [1, 2, 3, 11, None],
                       2: [1, 2, 3, 4, 5, None],
                       3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, None]};
        self.layoutToIndex = {};  # initialized by buildLayoutIndex
        self.indexToLayout = {};  # initialized by buildLayoutIndex
        self.layoutNum = {};  # initialized by buildLayoutIndex
        self.vNameLst = ['vl', 'vm', 'vn', 'vs'];
        self.vNameToIndex = {'vl': 0, 'vm': 1, 'vn': 2, 'vs': 3};
        self.indexToVName = {0: 'vl', 1: 'vm', 2: 'vn', 3: 'vs'};
        self.vNum = len(self.indexToVName);
        self.slotDiffMap = [[0 for i in range(15)] for j in range(15)];  # slot difference, initialized by buidSlotDiff
        self.marginLayoutIdx = {};  # initialized by buildLayoutIndex
        self.vSetSet = set();  # initialized by buildLayoutIndex
        # data
        self.data = {};  # (nquery, weight, annotations, vSet)
        # initialization
        self.buidSlotDiff();  # cache slot diff map
        self.buildLayoutIndex();  # cache (marginal) layout index
        self.initialExpertModel();
        self.initialPriorModel(self.data);
        return;
    
    def cacheModel(self, filePath):
        file = open(filePath, 'w');
        file.write(str(self.layoutPriorDist) + '\n');
        file.write(str(self.expertModel) + '\n');
        file.write(str(self.ediNum) + '\n');
        file.write(str(self.mtNum) + '\n');
        file.close();
        return;
    
    def loadModel(self, filePath):
        file = open(filePath, 'r');
        self.layoutPriorDist = eval(file.readline().strip());
        self.expertModel = eval(file.readline().strip());
        self.ediNum = eval(file.readline().strip());
        self.mtNum = eval(file.readline().strip());
        file.close();
        return;
        
    NOT_FOLD = True;
    #===========================================================================
    # Initialization & Cache
    #===========================================================================
    def buidSlotDiff(self):
        for optPos in range(1, 12):
            for ediPos in range(1, 12):
                if(optPos <= 6):
                    if(ediPos < optPos): self.slotDiffMap[optPos][ediPos] = ((optPos - ediPos) * 2 - 1);
                    elif(ediPos <= 2 * optPos - 1): self.slotDiffMap[optPos][ediPos] = ((ediPos - optPos) * 2); 
                    else: self.slotDiffMap[optPos][ediPos] = (ediPos - 1);
                else:
                    if(ediPos >= optPos): self.slotDiffMap[optPos][ediPos] = ((ediPos - optPos) * 2);
                    elif(ediPos >= 2 * optPos - 11):  self.slotDiffMap[optPos][ediPos] = ((optPos - ediPos) * 2 - 1);
                    else: self.slotDiffMap[optPos][ediPos] = (11 - ediPos);
        return;
    
    def buildLayoutIndex(self):
        self.indexToLayout = {};
        self.layoutToIndex = {};
        self.layoutNum = {};
        self.vSetSet = set();
        self.marginLayoutIdx = {};
        for vSetLst in itertools.product([0, -1], [1, -1], [2, -1], [3, -1]):
            vSet = tuple(vSetLst);  # vSet is a tuple, so as to be hashable
            self.vSetSet.add(vSet);
            self.indexToLayout[vSet] = {};
            self.layoutToIndex[vSet] = {};
            self.layoutNum[vSet] = 0;
            self.marginLayoutIdx[vSet] = {};
            for j in self.__availableVLst(vSet): self.marginLayoutIdx[vSet][j] = [[], []];
            for layoutIntLst in itertools.product(*[self.vSlots[x] if(x != -1) else [-1] for x in vSet]):
                self.layoutToIndex[vSet][str(layoutIntLst)] = self.layoutNum[vSet];
                self.indexToLayout[vSet][self.layoutNum[vSet]] = layoutIntLst;
                for j in self.__availableVLst(vSet): self.marginLayoutIdx[vSet][j][1 if(layoutIntLst[j] is not None) else 0].append(self.layoutNum[vSet]);
                self.layoutNum[vSet] += 1;
        return;
    
    def initialPriorModel(self, data):
        alpha = 0.85;
        self.layoutPriorDist = {};
        for vSet in self.vSetSet:
            self.layoutPriorDist[vSet] = [];
            for i in range(self.layoutNum[vSet]): self.layoutPriorDist[vSet].append(0.0);
        for (nquery, weight, annotations, vSet) in data:
            for ediIdx in annotations['editor']:
                self.layoutPriorDist[vSet][self.layoutToIndex[vSet][str(annotations['editor'][ediIdx])]] += weight;
        for vSet in self.vSetSet:
            nt = sum(self.layoutPriorDist[vSet]);
            self.layoutPriorDist[vSet] = [((((x / nt) * alpha) if(nt != 0) else (alpha / self.layoutNum[vSet])) + 
                                            ((1.0 - alpha) / self.layoutNum[vSet])) for x in self.layoutPriorDist[vSet]];
        return;
         
    def initialExpertModel(self):
        c1 = 0.8;
        self.expertModel['editor'] = [];
        self.expertModel['mt'] = [];
        self.expertModel['user'] = {};
        for idx in range(self.ediNum):  # initialize editor expertise model: l:variance, p:sensitivity, q:specificity
            self.expertModel['editor'].append({0: {'l': c1, 'p': c1, 'q': c1},
                                               1: {'l': c1, 'p': c1, 'q': c1},
                                               2: {'l': c1, 'p': c1, 'q': c1},
                                               3: {'l': c1, 'p': c1, 'q': c1}});
        for idx in range(self.mtNum):  # initialize mt expertise model
            self.expertModel['mt'].append({0: {'p': c1, 'q': c1},
                                           1: {'p': c1, 'q': c1},
                                           2: {'p': c1, 'q': c1},
                                           3: {'p': c1, 'q': c1}});
        self.expertModel['user'] = {0: {'a1':c1, 'a2':c1, 'b1':(-c1), 'b2':c1, 'p':{-1:0.2, 0:0.7, 1:0.1}},
                                    1: {'a1':c1, 'a2':c1, 'b1':(-c1), 'b2':c1, 'p':{-1:0.2, 0:0.7, 1:0.1}},
                                    2: {'a1':c1, 'a2':c1, 'b1':(-c1), 'b2':c1, 'p':{-1:0.2, 0:0.7, 1:0.1}},
                                    3: {'a1':c1, 'a2':c1, 'b1':(-c1), 'b2':c1, 'p':{-1:0.2, 0:0.7, 1:0.1}}};  # initialize user model
        return;            
    
    NOT_FOLD = True;
    #===========================================================================
    # Utility
    #===========================================================================
    def __poisson(self, lmbda, k): return (math.pow(lmbda, k) * math.exp(-lmbda) / math.factorial(k));
    
    def __sigm(self, x): 
        if(abs(x) <= 5e2): return (1.0 / (1.0 + math.exp(-x)));
        return (1.0 if(x > 0) else 0.0);        
    
    def __logSigm(self, x):
        if(abs(x) <= 5e2): return math.log(1.0 / (1.0 + math.exp(-x)));
        if(x > 0): return 0.0;
        if(x < 0): return x;
        
    def __logCosigm(self, x): return self.__logSigm(-x);
           
    def __optLayoutToOptChoices(self, optLayout):
        optChoices = set();
        for j in range(self.vNum):
            if((optLayout[j] is not None) and (optLayout[j] != -1)): optChoices.add(j);  # not not shown and not unavailable
        return optChoices;
    
    def __availableVLst(self, vSet): return [vIdx for vIdx in vSet if(vIdx != -1)];
    
    def __printExpertModel(self):
        # print user model
        for editorIdx in range(self.ediNum):
            print('editor: {0}'.format(editorIdx));
            for j in range(self.vNum): print('p={0}, q={1}, l={2}'.format(self.expertModel['editor'][editorIdx][j]['p'],
                                                                          self.expertModel['editor'][editorIdx][j]['q'],
                                                                          self.expertModel['editor'][editorIdx][j]['l']));
        for mtIdx in range(self.mtNum):
            print('mt: {0}'.format(mtIdx));
            for j in range(self.vNum): print('p={0}, q={1}'.format(self.expertModel['mt'][mtIdx][j]['p'],
                                                                   self.expertModel['mt'][mtIdx][j]['q']));
        print('user:');
        for j in range(self.vNum): print('a1={0}, b1={1}, a2={2}, b2={3}, p={4}'.format(self.expertModel['user'][j]['a1'],
                                                                                        self.expertModel['user'][j]['b1'],
                                                                                        self.expertModel['user'][j]['a2'],
                                                                                        self.expertModel['user'][j]['b2'],
                                                                                        self.expertModel['user'][j]['p']));
    
    NOT_FOLD = True;
    #===========================================================================
    # Likelihood
    #===========================================================================
    def getAnnotationLikelihoodForEditor(self, editorIdx, editorLayout, optLayout, vSet):
        prob = 1.0;
        for j in self.__availableVLst(vSet): 
            ediModel = self.expertModel['editor'][editorIdx][j];
            ediPos = editorLayout[j];
            optPos = optLayout[j];
            if((optPos is None) and (ediPos is None)): prob *= ediModel['q'];
            elif((optPos is None) and (ediPos is not None)): prob *= (1.0 - ediModel['q']) / len(self.vSlots[j]);
            elif((optPos is not None) and (ediPos is None)): prob *= (1.0 - ediModel['p']);
            elif((optPos is not None) and (ediPos is not None)): prob *= ediModel['q'] * self.__poisson(ediModel['l'], self.slotDiffMap[optPos][ediPos]);
        return prob;
    
    def __getAnnotationLogLikelihoodForEditor(self, editorIdx, editorLayout, optLayout, vSet):
        logprob = 0.0;
        for j in self.__availableVLst(vSet): 
            ediModel = self.expertModel['editor'][editorIdx][j];
            ediPos = editorLayout[j];
            optPos = optLayout[j];
            if((optPos is None) and (ediPos is None)): logprob += math.log(ediModel['q']);
            elif((optPos is None) and (ediPos is not None)): logprob += math.log(1.0 - ediModel['q']) - math.log(len(self.vSlots[j]));
            elif((optPos is not None) and (ediPos is None)): logprob += math.log(1.0 - ediModel['p']);
            elif((optPos is not None) and (ediPos is not None)): logprob += math.log(ediModel['q']) + math.log(self.__poisson(ediModel['l'], self.slotDiffMap[optPos][ediPos]));
        return logprob;
    
    def getAnnotationLikelihoodForMT(self, mtIdx, mtChoices, optChoices, vSet):
        prob = 1.0;
        for j in self.__availableVLst(vSet):
            mtModel = self.expertModel['mt'][mtIdx][j];
            if((j not in optChoices) and (j not in mtChoices)): prob *= mtModel['q'];
            elif((j not in optChoices) and (j in mtChoices)): prob *= (1 - mtModel['q']);
            elif((j in optChoices) and (j not in mtChoices)): prob *= (1 - mtModel['p']);
            elif((j in optChoices) and (j in mtChoices)): prob *= mtModel['p'];
        return prob;
    
    def __getAnnotationLogLikelihoodForMT(self, mtIdx, mtChoices, optChoices, vSet):
        logprob = 0.0;
        for j in self.__availableVLst(vSet):
            mtModel = self.expertModel['mt'][mtIdx][j];
            if((j not in optChoices) and (j not in mtChoices)): logprob += math.log(mtModel['q']);
            elif((j not in optChoices) and (j in mtChoices)): logprob += math.log(1 - mtModel['q']);
            elif((j in optChoices) and (j not in mtChoices)): logprob += math.log(1 - mtModel['p']);
            elif((j in optChoices) and (j in mtChoices)): logprob += math.log(mtModel['p']);
        return logprob;
    
    def getAnnotationLikelihoodForUser(self, vIdxPosRewardLst, optLayout, vSet):
        prob = 1.0;        
        for (vIdx, showedPos, reward) in vIdxPosRewardLst:
            if(vSet[vIdx] == -1): print('bugs here');
            userModel = self.expertModel['user'][vIdx];
            optPos = optLayout[vIdx];
            if(optPos is not None):
                r = self.slotDiffMap[optPos][showedPos];
                x1 = userModel['a1'] + userModel['b1'] * r;
                x2 = userModel['a2'] + userModel['b2'] * r;
                if(reward == 1): prob *= self.__sigm(x1);                    
                elif(reward == 0): prob *= self.__sigm(-x1) * self.__sigm(-x2);
                elif(reward == -1): prob *= self.__sigm(-x1) * self.__sigm(x2);
            else: prob *= userModel['p'][int(reward)];
        return prob;

    def __getAnnotationLogLikelihoodForUser(self, vIdxPosRewardLst, optLayout, vSet):
        logprob = 0.0;        
        for (vIdx, showedPos, reward) in vIdxPosRewardLst:
            if(vSet[vIdx] == -1): print('bugs here');
            userModel = self.expertModel['user'][vIdx];
            optPos = optLayout[vIdx];
            if(optPos is not None):
                r = self.slotDiffMap[optPos][showedPos];
                x1 = userModel['a1'] + userModel['b1'] * r;
                x2 = userModel['a2'] + userModel['b2'] * r;
                if(reward == 1): logprob += self.__logSigm(x1);
                elif(reward == 0): logprob += self.__logSigm(-x1) + self.__logSigm(-x2);
                elif(reward == -1): logprob += self.__logSigm(-x1) + self.__logSigm(x2);
            else: logprob += math.log(userModel['p'][int(reward)]);
        return logprob;
       
    def getAnnotationLikelihood(self, optLayout, annotations, vSet):
        prob = 1;
        if('editor' in annotations):
            for editorIdx in annotations['editor']: 
                x = self.getAnnotationLikelihoodForEditor(editorIdx, annotations['editor'][editorIdx], optLayout, vSet);
                prob *= x;
        if('user' in annotations): 
            for mtIdx in annotations['mt']: 
                x = self.getAnnotationLikelihoodForMT(mtIdx, annotations['mt'][mtIdx], self.__optLayoutToOptChoices(optLayout), vSet);
                prob *= x;
        if('user' in annotations):
                x = self.getAnnotationLikelihoodForUser(annotations['user'], optLayout, vSet);
                prob *= x;
        if(prob <= 0.0): print('[error]: probability <= 0.0');
        return prob;
    
    def __getAnnotationLogLikelihood(self, optLayout, annotations, vSet):
        logprob = 0.0;
        if('editor' in annotations):
            for editorIdx in annotations['editor']: 
                logprob += self.__getAnnotationLogLikelihoodForEditor(editorIdx, annotations['editor'][editorIdx], optLayout, vSet);
        if('user' in annotations): 
            for mtIdx in annotations['mt']: 
                logprob += self.__getAnnotationLogLikelihoodForMT(mtIdx, annotations['mt'][mtIdx], self.__optLayoutToOptChoices(optLayout), vSet);
        if('user' in annotations):
            logprob += self.__getAnnotationLogLikelihoodForUser(annotations['user'], optLayout, vSet);
        return logprob;
    
    def getQueryPosteriorLayoutDist(self, annotations, vSet):
        jpd = [self.getAnnotationLikelihood(self.indexToLayout[vSet][i], annotations, vSet) * self.layoutPriorDist[vSet][i] for i in range(self.layoutNum[vSet])];
        pf = sum(jpd);
        epsilon = 1e-8;
        #=======================================================================
        # smooth
        #=======================================================================
        pd = [(x / pf) * (1 - epsilon) + epsilon * (1.0 / len(jpd)) for x in jpd];
        marginPd = {};       
        for i in range(self.layoutNum[vSet]):
            for j in self.__availableVLst(vSet): 
                if(j not in marginPd): marginPd[j] = [0.0, 0.0];
                marginPd[j][0 if(self.indexToLayout[vSet][i][j] is None) else 1] += pd[i];
        return (pd, marginPd);
    
    NOT_FOLD = True;
    #===========================================================================
    # Run
    #===========================================================================
    def queryIterUpdate(self, annotations, vSet, nquery, weight, expertModelBkkp, newLayoutPriorDist):
        # posterior distribution
        (pd, marginPd) = self.getQueryPosteriorLayoutDist(annotations, vSet);
        # update layout prior
        for i in range(self.layoutNum[vSet]): newLayoutPriorDist[vSet][i] += weight * pd[i];
        # update editor model
        # 1 - denominator; 0 - numerator;
        for editorIdx in annotations['editor']:
            layout = annotations['editor'][editorIdx];
            for j in self.__availableVLst(vSet):
                # update 'p'
                if(layout[j] is not None): expertModelBkkp['editor'][editorIdx][j]['p'][0] += weight * marginPd[j][1];
                expertModelBkkp['editor'][editorIdx][j]['p'][1] += weight * marginPd[j][1];
                # update 'q'
                if(layout[j] is None): expertModelBkkp['editor'][editorIdx][j]['q'][0] += weight * marginPd[j][0];
                expertModelBkkp['editor'][editorIdx][j]['q'][1] += weight * marginPd[j][0];
                # update 'l'
                if(layout[j] is not None):
                    expertModelBkkp['editor'][editorIdx][j]['l'][0] += weight * sum([pd[optLayoutIdx] * self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][j]][layout[j]] 
                                                                    for optLayoutIdx in self.marginLayoutIdx[vSet][j][1]]);
                    expertModelBkkp['editor'][editorIdx][j]['l'][1] += weight * marginPd[j][1];
        # update mt model
        for mtIdx in annotations['mt']:
            choices = annotations['mt'][mtIdx];
            for j in self.__availableVLst(vSet):
                # update 'p'
                if(j in choices): expertModelBkkp['mt'][mtIdx][j]['p'][0] += weight * marginPd[j][1];
                expertModelBkkp['mt'][mtIdx][j]['p'][1] += weight * marginPd[j][1];
                # update 'q'
                if(j not in choices): expertModelBkkp['mt'][mtIdx][j]['q'][0] += weight * marginPd[j][0];
                expertModelBkkp['mt'][mtIdx][j]['q'][1] += weight * marginPd[j][0];
        # update user model
        for (vIdx, showedPos, reward) in annotations['user']: 
            # update 'a1'
            term1 = weight * marginPd[vIdx][1] if(reward == 1) else 0.0;
            term2 = weight * sum([pd[optLayoutIdx] * self.__sigm(self.expertModel['user'][vIdx]['a1'] + 
                                                                 self.expertModel['user'][vIdx]['b1'] * 
                                                                 self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][vIdx]][showedPos])
                                  for optLayoutIdx in self.marginLayoutIdx[vSet][vIdx][1]]);
            expertModelBkkp['user'][vIdx]['a1'] += term1 - term2;
            # update 'b1'
            term3 = weight * sum([pd[optLayoutIdx] * self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][vIdx]][showedPos]
                                  for optLayoutIdx in self.marginLayoutIdx[vSet][vIdx][1]]) if(reward == 1) else 0.0;
            term4 = weight * sum([pd[optLayoutIdx] * self.__sigm(self.expertModel['user'][vIdx]['a1'] + 
                                                                 self.expertModel['user'][vIdx]['b1'] * 
                                                                 self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][vIdx]][showedPos])
                                                   * self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][vIdx]][showedPos]
                                  for optLayoutIdx in self.marginLayoutIdx[vSet][vIdx][1]]);
            expertModelBkkp['user'][vIdx]['b1'] += term3 - term4;
            # update 'a2'
            term5 = weight * marginPd[vIdx][1] if(reward == -1) else 0.0;
            term6 = weight * sum([pd[optLayoutIdx] * self.__sigm(self.expertModel['user'][vIdx]['a2'] + 
                                                                 self.expertModel['user'][vIdx]['b2'] * 
                                                                 self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][vIdx]][showedPos])
                                  for optLayoutIdx in self.marginLayoutIdx[vSet][vIdx][1]]) if(reward <= 0) else 0.0;
            expertModelBkkp['user'][vIdx]['a2'] += term5 - term6;
            # update 'b2'
            term7 = weight * sum([pd[optLayoutIdx] * self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][vIdx]][showedPos]
                                  for optLayoutIdx in self.marginLayoutIdx[vSet][vIdx][1]]) if(reward == -1) else 0.0;
            term8 = weight * sum([pd[optLayoutIdx] * self.__sigm(self.expertModel['user'][vIdx]['a2'] + 
                                                                 self.expertModel['user'][vIdx]['b2'] * 
                                                                 self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][vIdx]][showedPos])
                                                    * self.slotDiffMap[self.indexToLayout[vSet][optLayoutIdx][vIdx]][showedPos]
                                  for optLayoutIdx in self.marginLayoutIdx[vSet][vIdx][1]]) if(reward <= 0) else 0.0;
            expertModelBkkp['user'][vIdx]['b2'] += term7 - term8;
            # update 'p0', 'p1', 'p2'
            expertModelBkkp['user'][vIdx]['p'][reward] += weight * marginPd[vIdx][0];
        return;            
    
    def beforeQueryIterUpdate(self):
        expertModelBkkp = {'editor': {}, 'mt': {}, 'user': {}};
        for ediIdx in range(self.ediNum):
            expertModelBkkp['editor'][ediIdx] = {0: {'l': [0.0, 0.0], 'p': [0.0, 0.0], 'q': [0.0, 0.0]},
                                                 1: {'l': [0.0, 0.0], 'p': [0.0, 0.0], 'q': [0.0, 0.0]},
                                                 2: {'l': [0.0, 0.0], 'p': [0.0, 0.0], 'q': [0.0, 0.0]},
                                                 3: {'l': [0.0, 0.0], 'p': [0.0, 0.0], 'q': [0.0, 0.0]}};
        for mtIdx in range(self.mtNum):
            expertModelBkkp['mt'][mtIdx] = {0: {'p': [0.0, 0.0], 'q': [0.0, 0.0]},
                                            1: {'p': [0.0, 0.0], 'q': [0.0, 0.0]},
                                            2: {'p': [0.0, 0.0], 'q': [0.0, 0.0]},
                                            3: {'p': [0.0, 0.0], 'q': [0.0, 0.0]}};
        expertModelBkkp['user'] = {0: {'a1':0.0, 'b1':0.0, 'a2':0.0, 'b2':0.0, 'p':{-1:0.0, 0:0.0, 1:0.0}},
                                   1: {'a1':0.0, 'b1':0.0, 'a2':0.0, 'b2':0.0, 'p':{-1:0.0, 0:0.0, 1:0.0}},
                                   2: {'a1':0.0, 'b1':0.0, 'a2':0.0, 'b2':0.0, 'p':{-1:0.0, 0:0.0, 1:0.0}},
                                   3: {'a1':0.0, 'b1':0.0, 'a2':0.0, 'b2':0.0, 'p':{-1:0.0, 0:0.0, 1:0.0}}};
        newLayoutPriorDist = {};
        for vSet in self.vSetSet: newLayoutPriorDist[vSet] = [0.0 for x in range(self.layoutNum[vSet])];
        return (newLayoutPriorDist, expertModelBkkp);
        
    def afterQueryIterUpdate(self, newLayoutPriorDist, expertModelBkkp, step):
        # normalize layout prior dist
        for vSet in self.vSetSet:
            pf = sum(newLayoutPriorDist[vSet]);
            if(pf != 0): self.layoutPriorDist[vSet] = [(x / pf) for x in newLayoutPriorDist[vSet]];
        # editor model normalization
        for editorIdx in range(self.ediNum):
            for j in range(self.vNum):
                self.expertModel['editor'][editorIdx][j]['p'] = expertModelBkkp['editor'][editorIdx][j]['p'][0] / expertModelBkkp['editor'][editorIdx][j]['p'][1];
                self.expertModel['editor'][editorIdx][j]['q'] = expertModelBkkp['editor'][editorIdx][j]['q'][0] / expertModelBkkp['editor'][editorIdx][j]['q'][1];
                self.expertModel['editor'][editorIdx][j]['l'] = expertModelBkkp['editor'][editorIdx][j]['l'][0] / expertModelBkkp['editor'][editorIdx][j]['l'][1];
        # mt model normalization
        for mtIdx in range(self.mtNum):
            for j in range(self.vNum):
                self.expertModel['mt'][mtIdx][j]['p'] = expertModelBkkp['mt'][mtIdx][j]['p'][0] / expertModelBkkp['mt'][mtIdx][j]['p'][1];
                self.expertModel['mt'][mtIdx][j]['q'] = expertModelBkkp['mt'][mtIdx][j]['q'][0] / expertModelBkkp['mt'][mtIdx][j]['q'][1];
        # user model normalization
        for j in range(self.vNum):
            self.expertModel['user'][j]['a1'] += expertModelBkkp['user'][j]['a1'] * step;  # coordinate descent
            self.expertModel['user'][j]['b1'] += expertModelBkkp['user'][j]['b1'] * step;
            self.expertModel['user'][j]['a2'] += expertModelBkkp['user'][j]['a2'] * step;
            self.expertModel['user'][j]['b2'] += expertModelBkkp['user'][j]['b2'] * step;
            for reward in self.expertModel['user'][j]['p']:  # normalization
                self.expertModel['user'][j]['p'][reward] = expertModelBkkp['user'][j]['p'][reward] / sum(expertModelBkkp['user'][j]['p'].values());
        return;          

    def iter(self, data, step):
        bcolor.cPrintln('\t\t[Iter]: Create Bookkeeping Variables ...', 'header');
        (newLayoutPriorDist, expertModelBkkp) = self.beforeQueryIterUpdate();
        bcolor.cPrintln('\t\t[Iter]: Looping Through Queries ...', 'header');
        total = len(data);
        processed = 0;
        for (nquery, weight, annotations, vSet) in data: 
            self.queryIterUpdate(annotations, vSet, nquery, weight, expertModelBkkp, newLayoutPriorDist);
            processed += 1;
            if(processed % 1000 == 0): utility.printProgressBar(float(processed) / total, 0.05, str(processed));
        print('');
        bcolor.cPrintln('\t\t[Iter]: Update Model ...', 'header');
        self.afterQueryIterUpdate(newLayoutPriorDist, expertModelBkkp, step);
        bcolor.cPrintln('\t\t[Iter]: Iter finishes', 'header');        
        return;
    
    def computePosteriorLayout(self, data, layoutFilePath):
        layoutFile = open(layoutFilePath, 'w');
        total = len(data);
        processed = 0;
#         s = 0.0;
#         ss = 0.0;
#         sss = 0.0;
#         ssss = 0.0;
#         sssss = 0.0;
        for (nquery, weight, annotations, vSet) in data:
            jpd = [self.getAnnotationLikelihood(self.indexToLayout[vSet][i], annotations, vSet) * self.layoutPriorDist[vSet][i] for i in range(self.layoutNum[vSet])];
            pf = sum(jpd);
            epsilon = 1e-8;
            #=======================================================================
            # smooth
            #=======================================================================
            pd = [(x / pf) * (1 - epsilon) + epsilon * (1.0 / len(jpd)) for x in jpd];
            pdMap = {};
            for i in range(self.layoutNum[vSet]): pdMap[str(self.indexToLayout[vSet][i])] = pd[i];
            topLayoutStrLst = [layoutStr for layoutStr in sorted(pdMap, key=lambda x:pdMap[x], reverse=True)];
            topLayoutStrLst = topLayoutStrLst[:10];
            #===================================================================
            # 
            #===================================================================
#             s += pdMap[topLayoutStrLst[0]];
#             ss += pdMap[topLayoutStrLst[0]] + pdMap[topLayoutStrLst[1]];
#             sss += pdMap[topLayoutStrLst[0]] + pdMap[topLayoutStrLst[1]] + pdMap[topLayoutStrLst[2]];
#             ssss += pdMap[topLayoutStrLst[0]] + pdMap[topLayoutStrLst[1]] + pdMap[topLayoutStrLst[2]] + pdMap[topLayoutStrLst[3]];
#             sssss += pdMap[topLayoutStrLst[0]] + pdMap[topLayoutStrLst[1]] + pdMap[topLayoutStrLst[2]] + pdMap[topLayoutStrLst[3]] + pdMap[topLayoutStrLst[4]];
            #===================================================================
            # 
            #===================================================================
            retLst = [(nquery, l, (pdMap[l] * weight)) for l in topLayoutStrLst if(pdMap[l] > 0.05)];

            for t in retLst: layoutFile.write(str(t) + '\n');  
            processed += 1;
            if(processed % 1000 == 0): utility.printProgressBar(float(processed) / total, 0.05, str(processed)); 
        print('');        
#         print(s / len(data));     
#         print(ss / len(data));     
#         print(sss / len(data));     
#         print(ssss / len(data));     
#         print(sssss / len(data));     
        layoutFile.close();
        return;        
            
    def assessELBO(self, data):
        elbo = 0.0;
        for (nquery, weight, annotations, vSet) in data:
            for i in range(self.layoutNum[vSet]):
                if(self.layoutPriorDist[vSet][i] == 0.0): 
                    print('error');
                    print(vSet);
                    print(self.indexToLayout[vSet][i]);
            loglikelihoodLst = [self.__getAnnotationLogLikelihood(self.indexToLayout[vSet][i], annotations, vSet) for i in range(self.layoutNum[vSet])];
            joinloglikLst = [loglikelihoodLst[i] + math.log(self.layoutPriorDist[vSet][i]) for i in range(self.layoutNum[vSet])];
            joinLikeliLst = [(math.exp(loglikelihoodLst[i]) * self.layoutPriorDist[vSet][i]) for i in range(self.layoutNum[vSet])];
            pf = sum(joinLikeliLst);
            pd = [((x / pf) if(pf != 0) else 0.0) for x in joinLikeliLst];
            elbo += weight * sum([pd[i] * joinloglikLst[i] for i in range(self.layoutNum[vSet])]);
        return elbo;
    
    def loadData(self, filePath=os.path.join(os.path.expanduser('~'), 'exp/data_1edi_1mt_user')):
        self.data = [];
        infile = open(filePath);
        for ln in infile:
            (nquery, weight, annotations, vSet) = ln.strip().split('\x01');
            weight = eval(weight);
#             weight = 1.0;
            annotations = eval(annotations);
            netReward = {};
            cntReward = {};
            for (vIdx, showedPos, reward) in annotations['user']:
                netReward[(vIdx, showedPos)] = netReward.get((vIdx, showedPos), 0.0) + reward;
                cntReward[(vIdx, showedPos)] = cntReward.get((vIdx, showedPos), 0.0) + 1.0;
                #===============================================================
                # remove duplicated entries
                #===============================================================
            annotations['user'] = [];
            for (vIdx, showedPos) in netReward:
                r = float(netReward[(vIdx, showedPos)]) / cntReward[(vIdx, showedPos)];
                if(r > 0.1):
                    annotations['user'].append((vIdx, showedPos, 1));
                elif(r < -0.1):
                    annotations['user'].append((vIdx, showedPos, -1));
                else:
                    annotations['user'].append((vIdx, showedPos, 0));
            
            vSet = eval(vSet);
            self.data.append((nquery, weight, annotations, vSet));
        # slice the data
        self.data = self.data[:];
        infile.close();
    
    def run(self, cacheModelFilePath=None):
        step = 5e-4;
        print('loading data ...');
        self.loadData();
        iterNum = 0;
        oldElbo = -1e9;
        #=======================================================================
        # cache model
        #=======================================================================
        if(cacheModelFilePath is not None):
            reg = re.compile(r'.*?(\d*)_iter');
            iterNum = eval(reg.match(cacheModelFilePath).group(1));
            print('reading from iter = {0}'.format(iterNum));
            self.loadModel(cacheModelFilePath);
            self.__printExpertModel();
            iterNum += 1;     
            elbo = self.assessELBO(self.data);
            print('computing elbo ...');
            bcolor.cPrintln('elbo = {0}'.format(elbo), 'warning');   
        while(True):
            print('iter ...');
            self.iter(self.data, step);
            filePath = os.path.join(os.path.expanduser('~'), 'exp/cache_model_1edi_1mt_user_{0}_iter'.format(iterNum));
            print('cache model at {0}'.format(filePath));
            self.cacheModel(filePath);
            if(iterNum % 3 == 0):
                elbo = self.assessELBO(self.data);
                print('computing elbo ...');
                bcolor.cPrintln('elbo = {0}'.format(elbo), 'warning');
                if(oldElbo >= elbo * 1.05): break;
                oldElbo = elbo;
            iterNum += 1;
        return;
    
    def posteriorEstimate(self, cacheModelFilePath=None):
        print('loading data ...');
        self.loadData();
        #=======================================================================
        # cache model
        #=======================================================================
        if(cacheModelFilePath is not None):
            reg = re.compile(r'.*?(\d*)_iter');
            iterNum = eval(reg.match(cacheModelFilePath).group(1));
            print('reading from iter = {0}'.format(iterNum));
            self.loadModel(cacheModelFilePath);
        else:
            print('need model to be load');
            return;
        self.computePosteriorLayout(self.data, cacheModelFilePath + "_post_estimate");
        return;

def convertFileToPigTable(infilePath, outfilePath):
    infile = open(infilePath);
    outfile = open(outfilePath, 'w');
    for ln in infile:
        (nquery, l, w) = eval(ln.strip());
        outfile.write('\x01'.join([str(x) for x in [nquery, l, w]]) + '\n');
    infile.close();
    outfile.close();
    return;
    
if(__name__ == '__main__'):
    iterNum = 9;
    filePath = os.path.join(os.path.expanduser('~'), 'exp/cache_model_1edi_1mt_user_{0}_iter'.format(iterNum))

#     inferer = Inferer(1, 1);
#     inferer.run(filePath);
#     inferer.posteriorEstimate(filePath);
    postestmateFilePath = filePath + "_post_estimate";
    pigfilePath = postestmateFilePath + '_pig';
    convertFileToPigTable(postestmateFilePath, pigfilePath);
