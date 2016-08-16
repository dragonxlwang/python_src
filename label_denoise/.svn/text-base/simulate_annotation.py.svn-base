'''
Created on Jul 24, 2013

@author: wangxl
'''
import optparse;
import sys;
sys.path.append('.');
import os;
import random;
import vw_feature_extract;
import utility;
import zlib;
import base64;
import bcolor;
import math;

def mtAnnotationMergePerQuery(args):
    outputDict = {};
    for ln in sys.stdin:
        (ctr, nquery, v) = eval(ln.strip());
        if(nquery not in outputDict): outputDict[nquery] = {};
        outputDict[nquery][v] = ctr;
    for nquery in outputDict: print(str((nquery, outputDict[nquery])));
    return;

def extractNqueryVertical(args):
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        for v in [e['v'] for e in eventLst if(vw_feature_extract.checkIfSlotIsVertical(e['v']))]:
            print('{0}\x01{1}'.format(nquery, v));
    return;

def accumulateNqueryVertical(args):
    vSet = set();
    curQuery = None;
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (nquery, v) = ln.strip().split('\x01');
        if(curQuery is None): curQuery = nquery;
        if(curQuery != nquery): 
            print('{0}\x01{1}'.format(nquery, vSet));
            curQuery = nquery;
            vSet = set();
        vSet.add(v);
    print('{0}\x01{1}'.format(nquery, vSet));
    return;       

def getEditorUniqNquery(args):
    qSet = set();
    for ln in sys.stdin:
        (c, q, v) = eval(ln.strip());
        qSet.add(q);
    for q in qSet: print q;

def getTransWithVertical(args):
    for ln in sys.stdin:
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(len([e for e in eventLst if(vw_feature_extract.checkIfSlotIsVertical(e['v']))]) != 0):
            print nquery;

def ruleBasedFormingEdiMtAnnotation(args):
    editorFile = open(args[0], 'r');
    print("editor: " + args[0]);
    mtFile = open(args[1], 'r');    
    print("mt " + args[1]);
    outFile = open(args[2], 'w');
    print("output " + args[2]);
    layout = {};
    binpre = {};
    getVerticalName = vw_feature_extract.getVerticalName;
    getSlotName = vw_feature_extract.getSlotName;
    # editor annotation
    lncnt = 0;
    print('processing editor file');
    for ln in editorFile:
        lncnt += 1;
        if(lncnt % 10000 == 0):
            print('{0}: processed {1} lines'.format(utility.getTimeStr(), lncnt));
        (cnt, nquery, l) = eval(ln.strip());
        if(nquery not in layout): layout[nquery] = {};
        vLst = l.split('$');
        for v in vLst:
            name = getVerticalName(v);
            slot = getSlotName(v);
            if(name not in layout[nquery]): layout[nquery][name] = (v, cnt);
            elif(layout[nquery][name][1] < cnt): layout[nquery][name] = (v, cnt);
            elif((layout[nquery][name][1] == cnt) and (getSlotName(layout[nquery][name][0]) > slot)):
                layout[nquery][name] = (v, cnt);        
    # mt annotation
    lncnt = 0;
    print('processing mt file');
    for ln in mtFile:
        lncnt += 1;
        if(lncnt % 10000 == 0):
            print('{0}: processed {1} lines'.format(utility.getTimeStr(), lncnt));
        (ctr, nquery, vName) = eval(ln.strip());
        if(ctr > 1): 
            if(nquery not in binpre): binpre[nquery] = set();
            binpre[nquery].add(vName);  # inlucde high-confident vertical presence
        if((ctr < 0.5) and (nquery in layout) and (vName in layout[nquery])): 
            del layout[nquery][vName];  # over-rule unconfident annotator 
    editorFile.close();
    mtFile.close();
    nquerySet = set();
    for nquery in layout: nquerySet.add(nquery);
    for nquery in binpre: nquerySet.add(nquery);
    lncnt = 0;
    print('total queries: {0}'.format(len(nquerySet)));
    for nquery in nquerySet:
        lncnt += 1;
        if(lncnt % 10000 == 0):
            print('{0}: processed {1} queryies'.format(utility.getTimeStr(), lncnt));
        l = layout.get(nquery, {});
        b = binpre.get(nquery, set());
        outFile.write('{0}\x01{1}\x01{2}\n'.format(nquery, l, b));
    outFile.close();     
    print('finished');
    return;  

def pigStreamCollectClickAfterSort(args):
    curNquery = None;
    curClickDict = {};
    for ln in sys.stdin:
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(curNquery is None): curNquery = nquery;
        elif(curNquery != nquery):
            print('{0}\x01{1}'.format(nquery, curClickDict));
            curNquery = nquery;
            curClickDict = {};
        for e in eventLst:
            if(not vw_feature_extract.checkIfSlotIsVertical(e['v'])): continue;
            v = vw_feature_extract.getFullSlotName(e['v'], e['s']);
            r = e['r'];
            if(v not in curClickDict): curClickDict[v] = {};
            curClickDict[v][r] = curClickDict[v].get(r, 0) + 1;
    print('{0}\x01{1}'.format(nquery, curClickDict));                
    return;

def pigStreamGetPossibleVerticalPerQuery(args):
    curNquery = None;
    vSet = set();
    for ln in sys.stdin:
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(curNquery is None): curNquery = nquery;
        if(curNquery != nquery):
            print('{0}\x01{1}'.format(curNquery, vSet));                                                                       
            curNquery = nquery;           
            vSet = set();
        for e in eventLst: 
            if(vw_feature_extract.checkIfSlotIsVertical(e['v'])): vSet.add(e['v']);
    print('{0}\x01{1}'.format(curNquery, vSet));                
    return;

def prepDataForInfer(args):
    outputDict = {};
    editor_mt_anno_file = open(os.path.join(os.path.expanduser('~'), 'exp/editor_mt_anno.out'));
    cnt = 0;
    for ln in editor_mt_anno_file:
        (nquery, edi_anno_str, mt_anno_str) = ln.strip().split('\x01');
        outputDict[nquery] = {'nquery': nquery, 'edi': edi_anno_str, 'mt': mt_anno_str};
        cnt += 1;
        if(cnt % 10000 == 0):
            bcolor.cPrintln('{0}:\t{1}: {2} lines processed'.format(utility.getTimeStr(), 'exp/editor_mt_anno.out', cnt));
    editor_mt_anno_file.close();
    
    click_anno_file = open(os.path.join(os.path.expanduser('~'), 'exp/click_nquery_vidxposrewardlst.out.gz'));
    cnt = 0;
    for ln in click_anno_file:
        (nquery, vidx_pos_reward_lst_str) = ln.strip().split('\x01');
        if(nquery not in outputDict): outputDict[nquery] = {};
        outputDict[nquery]['click'] = vidx_pos_reward_lst_str;
        cnt += 1;
        if(cnt % 100000 == 0):
            bcolor.cPrintln('{0}:\t{1}: {2} lines processed'.format(utility.getTimeStr(), 'exp/click_nquery_vidxposrewardlst.out.gz', cnt));
    click_anno_file.close();    
    
    cntln = 0;
    nquery_cnt_file = open(os.path.join(os.path.expanduser('~'), 'exp/nquery_cnt_rel.out.gz'));
    for ln in nquery_cnt_file:
        (nquery, cnt) = ln.strip().split('\x01');
        if(nquery in outputDict): outputDict[nquery]['cnt'] = cnt;
        cntln += 1;
        if(cntln % 100000 == 0):
            bcolor.cPrintln('{0}:\t{1}: {2} lines processed'.format(utility.getTimeStr(), 'exp/nquery_cnt_rel.out.gz', cntln));
    nquery_cnt_file.close();
    
    nquery_vset_rel_file = open(os.path.join(os.path.expanduser('~'), 'exp/nquery_vset_rel.out.gz'));
#     outputFile = open(os.path.join(os.path.expanduser('~'), 'exp/nquery_edi_mt_click_vset_rel_small.out'), 'w');
    outputFile = open(os.path.join(os.path.expanduser('~'), 'exp/nquery_edi_mt_click_vset_rel.out'), 'w');
    cnt = 0;
    for ln in nquery_vset_rel_file:
        (nquery, vset_str) = ln.strip().split('\x01');
        # remove queryies does not have any vertical backends
        vset = eval(vset_str);
        if(len(vset) == 0): continue;
        if(nquery in outputDict):
#             if(('edi' in outputDict[nquery]) and ('mt' in outputDict[nquery]) and ('click' in outputDict[nquery])):
#                 outputFile.write('\x01'.join([nquery,
#                                               outputDict[nquery]['edi'],
#                                               outputDict[nquery]['mt'],
#                                               outputDict[nquery]['click'],
#                                               vset_str,
#                                               outputDict[nquery]['cnt']]) + '\n');
            outputFile.write('\x01'.join([nquery,
                                          outputDict[nquery]['edi'] if('edi' in outputDict[nquery]) else str(None),
                                          outputDict[nquery]['mt'] if('mt' in outputDict[nquery]) else str(None),
                                          outputDict[nquery]['click'] if('click' in outputDict[nquery]) else str(None),
                                          vset_str,
                                          outputDict[nquery]['cnt']]) + '\n');
            
        cnt += 1;
        if(cnt % 100000 == 0):
            bcolor.cPrintln('{0}:\t{1}: {2} lines processed'.format(utility.getTimeStr(), 'exp/nquery_vset_rel.out.gz', cnt));
    nquery_vset_rel_file.close();
    outputFile.close();

def pigStreamGatherClickDataAfterGrouped(args):
    curNquery = None;
    vIdxPosRewardLst = [];
    for ln in sys.stdin:
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(curNquery is None): curNquery = nquery;
        if(curNquery != nquery): 
            print('{0}\x01{1}'.format(curNquery, vIdxPosRewardLst));
            curNquery = nquery;
            vIdxPosRewardLst = [];
        for e in eventLst:
            if(vw_feature_extract.checkIfSlotIsVertical(e['v'])):
                vIdxPosRewardLst.append((e['v'], vw_feature_extract.getSlotPos(e['s']), e['r']));
    print('{0}\x01{1}'.format(curNquery, vIdxPosRewardLst));
    return;

def statNqueryEdiMtClickVset(args):
    ediNum = 0;
    mtNum = 0;
    clickNum = 0;
    vsetNum = 0;
    cnt = 0;
    cnt2 = 0;
    infile = open(os.path.join(os.path.expanduser('~'), 'exp/nquery_edi_mt_click_vset_rel.out'));
    for ln in infile:
        (nquery, edi, mt, click, vset) = ln.strip().split('\x01');
        edi = eval(edi);
        mt = eval(mt);
        click = eval(click);
        vset = eval(vset);
        if((edi is not None)): ediNum += 1;
        if((mt is not None)): mtNum += 1;
        if((click is not None)): clickNum += 1;
        if((vset is not None) and len(vset) > 0): vsetNum += 1;
        if((edi is None) and (mt is None) and (click is not None)): cnt += 1;
        if((edi is not None) and (mt is not None) and (click is None)): cnt2 += 1;
    print('ediNum = {0}'.format(ediNum));
    print('mtNum = {0}'.format(mtNum));
    print('clickNum = {0}'.format(clickNum));
    print('vset = {0}'.format(vsetNum));
    print(cnt);
    print(cnt2);
    return;

NOT_FOLD = True;
#===============================================================================
# Generate Data (Adding Noise)
#===============================================================================
class Simulator(object):
    expertModel = None;
    vNameLst = ['vl', 'vm', 'vn', 'vs'];
    vNameToIndex = {'vl': 0, 'vm': 1, 'vn': 2, 'vs': 3};
    indexToVName = {0: 'vl', 1: 'vm', 2: 'vn', 3: 'vs'};
    vNum = 4;
    slotDiffMap = {};
    vSlots = {0: [1, 2, 3, 4, 11],
              1: [1, 2, 3, 11],
              2: [1, 2, 3, 4, 5],
              3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]};
    ediNum = None;
    mtNum = None;
                   
    def buidSlotDiff(self):
        self.slotDiffMap = [[0 for i in range(15)] for j in range(15)]; 
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
       
    def initExpertModel(self):
        c1 = 0.8;
        self.expertModel = {};
        self.expertModel['editor'] = [];
        self.expertModel['mt'] = [];
        self.expertModel['user'] = {};
        self.expertModel['editor'].append({0: {'l': 0.0, 'p': 1.0, 'q': 1.0},
                                           1: {'l': 0.0, 'p': 1.0, 'q': 1.0},
                                           2: {'l': 0.0, 'p': 1.0, 'q': 1.0},
                                           3: {'l': 0.0, 'p': 1.0, 'q': 1.0}});
        self.expertModel['mt'].append({0: {'p': 1.0, 'q': 1.0},
                                       1: {'p': 1.0, 'q': 1.0},
                                       2: {'p': 1.0, 'q': 1.0},
                                       3: {'p': 1.0, 'q': 1.0}});
        self.ediNum = len(self.expertModel['editor']);
        self.mtNum = len(self.expertModel['mt']);
        return;
    
    def __init__(self):
        self.buidSlotDiff();
        self.initExpertModel();
        return;
    
    def __binarySample(self, b, p, q):
        if(b):
            if(random.random() <= p): return True;
            else: return False;
        else:
            if(random.random() <= q): return False;
            else: return True;
    
    def __poisson(self, lmbda, k): return (math.pow(lmbda, k) * math.exp(-lmbda) / math.factorial(k));
    
    def __multinomialSampling(self, pdf):
        x = random.random();
        i = 0;
        cdf = 0.0;
        while(True): 
            cdf += pdf[i];
            if(cdf >= x): return i;
            i += 1;
            if(i > len(pdf)): return len(pdf) - 1;
        return;

    def __uniformSample(self, slots): return slots[random.randint(0, len(slots) - 1)];
        
    def __poissonSample(self, k, l, slots):
        p = {};
        for i in slots: p[i] = self.__poisson(l, self.slotDiffMap[k][i]);
        pf = sum(p.values());
        pdf = [];
        kLst = [];
        for (k, v) in p.iteritems():
            kLst.append(k);
            pdf.append(v / pf);
        return kLst[self.__multinomialSampling(pdf)];
            
    def genEditorAnno(self, optLayout, vSet, ediIdx):
        layout = [-1, -1, -1, -1];
        for vIdx in range(len(optLayout)):
            if(optLayout[vIdx] == -1): layout[vIdx] = -1;
            elif(optLayout[vIdx] == None): 
                b = self.__binarySample(False, self.expertModel['editor'][ediIdx][vIdx]['p'], self.expertModel['editor'][ediIdx][vIdx]['q']);
                if(not b): layout[vIdx] = None;
                else: layout[vIdx] = self.__uniformSample(self.vSlots[vIdx]);
            else:
                b = self.__binarySample(True, self.expertModel['editor'][ediIdx][vIdx]['p'], self.expertModel['editor'][ediIdx][vIdx]['q']);
                if(not b): layout[vIdx] = None;
                else: layout[vIdx] = self.__poissonSample(optLayout[vIdx], self.expertModel['editor'][ediIdx][vIdx]['l'], self.vSlots[vIdx]);                
        return layout;
    
    def genMtAnno(self, optChoices, vSet, mtIdx):
        choices = set();
        for vIdx in vSet:
            if(vIdx == -1): continue;  # not available
            if(self.__binarySample((True if(vIdx in optChoices) else False), 
                                   self.expertModel['mt'][mtIdx][vIdx]['p'], 
                                   self.expertModel['mt'][mtIdx][vIdx]['q'])): choices.add(vIdx);
        return choices;
    
    def generateData(self):
        self.initExpertModel();
                
        nQueryEdiMtClickVsetFile = open(os.path.join(os.path.expanduser('~'), 'exp/nquery_edi_mt_click_vset_cnt_rel_small.out'));
        dataFile = open(os.path.join(os.path.expanduser('~'), 'exp/data_1edi_1mt_user'), 'w');
        for ln in nQueryEdiMtClickVsetFile:
            (nquery, edi, mt, click, vset, cnt) = ln.strip().split('\x01');
            edi = eval(edi);  # dict
            mt = eval(mt);  # set
            click = eval(click);  # lst
            vset = eval(vset);  # set
            cnt = eval(cnt);
            
            optLayout = [-1, -1, -1, -1];
            choices = set();
            vIdxPosRewardLst = [];
            vSet = (0 if(self.indexToVName[0] in vset) else -1,
                    1 if(self.indexToVName[1] in vset) else -1,
                    2 if(self.indexToVName[2] in vset) else -1,
                    3 if(self.indexToVName[3] in vset) else -1);
            weight = cnt;
            for vName in self.vNameToIndex:
                if(vName in edi):  # showed vertical
                    optLayout[self.vNameToIndex[vName]] = vw_feature_extract.getSlotName(edi[vName][0]);
                elif(vName in vset):  # unshowed vertical
                    optLayout[self.vNameToIndex[vName]] = None;
                else:  # unavailable vertical
                    optLayout[self.vNameToIndex[vName]] = -1; 
            for vName in self.vNameToIndex:
                if(vName in mt): choices.add(self.vNameToIndex[vName]);
            for (vName, pos, reward) in click:
                vIdxPosRewardLst.append((self.vNameToIndex[vName], pos, reward));
            annotations = {'editor': {}, 'mt': {}, 'user': None};
            for ediIdx in range(self.ediNum): annotations['editor'][ediIdx] = self.genEditorAnno(optLayout, vSet, ediIdx);
            for mtIdx in range(self.mtNum): annotations['mt'][mtIdx] = self.genMtAnno(choices, vSet, mtIdx);
            annotations['user'] = vIdxPosRewardLst
            dataFile.write('{0}\x01{1}\x01{2}\x01{3}\n'.format(nquery, weight, annotations, vSet));
        nQueryEdiMtClickVsetFile.close();
        dataFile.close();
    
if __name__ == '__main__':
    simulator = Simulator();
    simulator.generateData();
    pass
