'''
Created on Jun 21, 2013

@author: wangxl
'''
import sys;
sys.path.append('.');
import zlib;
import base64;
import json;
import itertools;
import utility;
import math;
import vowpal_wabbit;
import os;

NOT_FOLD = True;
#===============================================================================
# Utility
#===============================================================================
def getGrams(s):
    unigram = {};
    bigram = {};
    coOccur = {};
    if(not s): return;
    tokLst = s.split();
    for i in range(len(tokLst)): unigram[tokLst[i]] = 1.0;
    for i in range(len(tokLst) - 1): bigram['<=bi=>'.join([tokLst[i], tokLst[i + 1]])] = 1.0;
    tokLst.sort();
    coOccur = dict([('<=co=>'.join(list(x)), 1.0) for x in itertools.combinations(tokLst, 2)]);
    return (unigram, bigram, coOccur);

def getRawFea(ln):
    obj = eval(ln.strip().split('\x01')[-1]);
    fv = eval(zlib.decompress(base64.b64decode(obj['fv'])));
    #=======================================================================
    # query unibram & bigram
    #=======================================================================
    if('q' not in fv): fv['q'] = {};
    nquery = obj['nquery'];
    nquery = nquery.replace(':', '$COLON$');
    nquery = nquery.replace('|', '$PIPE$');
    (unigram, bigram, coOccur) = getGrams(nquery);
    fv['q']['unigram'] = unigram;
    fv['q']['bigram'] = bigram;
    fv['q']['coOccur'] = coOccur;
    #=======================================================================
    # global service management 
    #=======================================================================
    if('gsm_backend_calls' in obj):
        fv['q']['gsm_backend_calls'] = {};
        gsm_backend_calls = json.loads(zlib.decompress(base64.b64decode(obj['gsm_backend_calls'])));
        for k in gsm_backend_calls.keys(): fv['q']['gsm_backend_calls']['gsm_' + k] = 1.0;  # ???
    #=======================================================================
    # query profile
    #=======================================================================
    if('qp_backend_calls' in obj):
        fv['q']['qp_backend_calls'] = {};
        qp_backend_calls = json.loads(zlib.decompress(base64.b64decode(obj['qp_backend_calls'])));
        for k in qp_backend_calls.keys(): fv['q']['qp_backend_calls']['qp_' + k] = 1.0;  # ???
    #===========================================================================
    # return (fv, eventLst, id, ts)
    #===========================================================================
    eventLst = obj['e'];
    id = obj['id'];
    ts = obj['ts'];
    return (fv, eventLst, id, ts, nquery);

def _getFeatureName(*args):
    return '##'.join([str(arg) for arg in args]);

def getActionFeaVec(fv, eName):
    feaVec = {};
    feaVec[_getFeatureName('a', 'name', eName)] = 1.0;  # action feature
    if(eName.startswith('w') and ('w' in fv) and ('a' in fv['w'])):  # action feature from web
        for (k, v) in fv['w']['a'].iteritems(): feaVec[_getFeatureName('a', 'w', k)] = v;
    if((not eName.startswith('w')) and (eName in fv) and ('a' in fv[eName])):  # action feature from vertical
        for (k, v) in fv[eName]['a'].iteritems(): feaVec[_getFeatureName('a', eName, k)] = v;
    return feaVec;

def getResultSetFeaVec(fv, eName):
    feaVec = {};
    if((not eName.startswith('w')) and (eName in fv) and ('rs' in fv[eName])):  # rs:: vertical
        for (k, v) in fv[eName]['rs'].iteritems(): feaVec[_getFeatureName('rs', eName, k)] = v;
    if(('global' in fv) and ('rs' in fv['global'])):  # rs:: 
        for (k, v) in fv['global']['rs'].iteritems(): feaVec[_getFeatureName('rs', 'global', k)] = v;
    return feaVec;

def getQueryFeaVec(fv):
    feaVec = {};
    vertLst = ['vn', 'vl', 'vs', 'vm', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w'];
    queryCacheFieldLst = ['at', 'navqdom', 'qp', 'gsm_backend_calls', 'qp_backend_calls', 'unigram'];
    fieldLst = queryCacheFieldLst + vertLst;
    for field in fieldLst: 
        if(field in fv['q']):
            for (k, v) in fv['q'][field].iteritems(): feaVec[_getFeatureName('q', field, k)] = v;
    return feaVec;

def getLexicalFeaVec(fv):
    feaVec = {};
    for field in ['unigram', 'bigram', 'coOccur']:
        if(field in fv['q']):
            for (k, v) in fv['q'][field].iteritems(): feaVec[_getFeatureName('l', field, k)] = v;
    return feaVec;

def getClickFeaVec(reward):
    feaVec = {};
    feaVec[_getFeatureName('r', 'click')] = reward;
    return feaVec;

def getAugmentEventLst(eventLst):
    wLst = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10'];
    rLst = [e['r'] for e in eventLst];
    augEventLst = [];
    wIdx = 0;
    for e in sorted(eventLst, key=lambda e: utility.parseNumVal(e['s'].replace('s', '').strip())):
        if(e['v'].startswith('w')):
            while(e['v'].strip() != wLst[wIdx]):
                augEventLst.append({'r':0.0, 'v':wLst[wIdx], 's':"<font color=\"red\">{0}</font>".format('aug')});
                wIdx += 1;
            wIdx += 1;
        augEventLst.append(e);
    return augEventLst;

def _highlightVerticalName(name):
    if(name.startswith('v')):
        return """<a style = "color: yellow; background-color: black">{0}</a>""".format(name);
    else: return name;

def _highlightClick(s):
    if(utility.parseNumVal(str(s)) > 0):
        return """<a style = "color: red; background-color: blue">{0}</a>""".format(str(s));
    else: return str(s);

def _htmlHeader():
    str = '\n'.join(["<HTML>", "<PRE>"]);
    return str;

def _htmlTailer():
    str = '\n'.join(["</PRE>", "</HTML>"]);
    return str;

def getSlotPos(s): return int(utility.parseNumVal(s.replace('s', '').strip()));

def getFullSlotName(v, s):
    if(v.startswith('w')): return;
    if(isinstance(s, str)): s = getSlotPos(s);
    return v + str(s);

def getVerticalName(v): return ''.join([x for x in v if(not x.isdigit())]);

def getSlotName(v): return eval(''.join([x for x in v if(x.isdigit())]));
    
def checkIfSlotIsVertical(v):
    if(v.startswith('w')): return False;
    return True;

def getVerticalCompactRep(eventLst):
    return '$'.join(sorted([getFullSlotName(e['v'], e['s']) for e in eventLst if checkIfSlotIsVertical(e['v'])]));

def getVerticalCompactRepFromVNameLst(vNameLst): return '$'.join(sorted([x for x in vNameLst]));

def packHadoopLine(k, v): return '{0}\t{1}'.format(str(k).replace('\t', '$TAB$'), str(v));

def unpackHadoopLine(ln): return [x.replace('$TAB$', '\t') for x in ln.strip().split('\t', 1)];

    
NOT_FOLD = True;
#===============================================================================
# 
#===============================================================================
def featureExtract():
    print(_htmlHeader());
    i = 0;
    qFreqTable = {};
    for line in sys.stdin:
        (fv, eventLst, id, ts, nquery) = getRawFea(line);
        #=======================================================================
        # faeture extraction
        #=======================================================================
        #=======================================================================
        # page shared features
        #=======================================================================
        qFeaVec = getQueryFeaVec(fv);
        lFeaVec = getLexicalFeaVec(fv);
        #=======================================================================
        # event private features
        #=======================================================================
        augEventLst = getAugmentEventLst(eventLst);
        eventRelatedFeaVecLst = [(getActionFeaVec(fv, e['v']), getResultSetFeaVec(fv, e['v']), getClickFeaVec(e['r']), e) for e in augEventLst];
        
        for (aFeaVec, rsFeaVec, clickFeaVec, e) in eventRelatedFeaVecLst:
            #===================================================================
            # pointwise regression example generation
            #===================================================================
            tag = ';'.join(str(x) for x in [e['v'], e['r'], ts]);
            label = e['r'];
            vwEx = vowpal_wabbit.VWExample(label=label, weight=1.0, tag=tag);
            vwEx.insertFeatureDict('q', qFeaVec);  # query
            vwEx.insertFeatureDict('l', lFeaVec);  # lexical query
            vwEx.insertFeatureDict('a', aFeaVec);  # action
            vwEx.insertFeatureDict('rs', rsFeaVec);  # rs
            print('{0}\t{1}'.format(ts, str(vwEx)));
    print(_htmlTailer());
    return;

def generateAnnotation(options):
    if(options.map):
        clickDict = {};
        rewardDict = {};
        occurDict = {};
        for line in sys.stdin:
            (fv, eventLst, id, ts, nquery) = getRawFea(line);
            if nquery not in clickDict: 
                clickDict[nquery] = {};
                rewardDict[nquery] = {};
                occurDict[nquery] = {};
            for e in eventLst:
                v = e['v'];
                s = int(utility.parseNumVal(e['s'].replace('s', '').strip()));
                r = e['r'];
                if(v.startswith('v')): v += str(s);
                rewardDict[nquery][v] = rewardDict[nquery].get(v, 0.0) + r;  # update reward
                if(r == 1.0): clickDict[nquery][v] = clickDict[nquery].get(v, 0.0) + 1.0;  # update click
                occurDict[nquery][v] = occurDict[nquery].get(v, 0.0) + 1.0;  # update occur
        for nquery in occurDict:
            ln = str((occurDict, clickDict, rewardDict));
            print("{0}\t{1}".format(nquery, ln));
    elif(options.reduce):  # reducer = 1
        occurDict = {};
        clickDict = {};
        rewardDict = {};
        posGaugeDict = {};
        
        for ln in sys.stdin:
            (nquery, v) = ln.strip().split('\t', 1);
            (oDict, cDict, rDict) = eval(v);
            if(nquery not in occurDict):
                occurDict[nquery] = {};
                clickDict[nquery] = {};
                rewardDict[nquery] = {};
            for (k, v) in oDict.iteritems(): occurDict[nquery][k] = occurDict[nquery].get(k) + v;
            for (k, v) in cDict.iteritmes(): clickDict[nquery][k] = clickDict[nquery].get(k) + v;
            for (k, v) in rDict.iteritems(): rewardDict[nquery][k] = rewardDict[nquery].get(k) + v;
            
    return;

def debugPrintDict(varName, var):
    print(varName);
    for k in sorted(var):
        print('    {0} : {1}'.format(k, var[k]));
    print('');
    return;

vNameLst = ['vl', 'vm', 'vn', 'vs'];
vNameToIndex = {'vl': 0, 'vm': 1, 'vn': 2, 'vs': 3};
indexToVName = {0: 'vl', 1: 'vm', 2: 'vn', 3: 'vs'};

def crossDict(f1, f2): return dict([('{0}<=cat=>{1}'.format(k1, k2), v1 * v2) for (k1, v1) in f1.items() for (k2, v2) in f2.items()]);

def extractPointFeature(fv, eName):
    af = getActionFeaVec(fv, eName);
    qf = getQueryFeaVec(fv);  # not joined
    lf = getLexicalFeaVec(fv);  # not joined
    rf = getActionFeaVec(fv, eName);
    aqf = crossDict(af, qf);
    alf = crossDict(af, lf);
    return {'af': af, 'aqf': aqf, 'alf': alf, 'rf': rf};

def substractPointFeature(feaVec1, feaVec2):
    feaVec = {};
    for n in feaVec1:
        if(n not in feaVec): feaVec[n] = {};
        for k in feaVec1[n]: feaVec[n][k] = feaVec[n].get(k, 0.0) + feaVec1[n][k];
    for n in feaVec2:
        if(n not in feaVec): feaVec[n] = {};
        for k in feaVec2[n]: feaVec[n][k] = feaVec[n].get(k, 0.0) - feaVec2[n][k];
    return feaVec;

def dumpPairFeature(feaVec, outfile, weight):
    vwEx = vowpal_wabbit.VWExample(label= +1, weight=weight);
    for n in feaVec: vwEx.insertFeatureDict(n, feaVec[n]);
    outfile.write('{0}\n'.format(str(vwEx)));
    return;

def extractPairFeature(fv, eName1, eName2):  # eName1 > eName2
    feaVec1 = extractPointFeature(fv, eName1);
    feaVec2 = extractPointFeature(fv, eName2);
    return substractPointFeature(feaVec1, feaVec2);

def svmRankFeatureExtraction(args):
    infilePath = os.path.join(os.path.expanduser('~'), 'exp/nquery_layout_weight_avgfv_1000.out');
    outfilePath = os.path.join(os.path.expanduser('~'), 'exp/svmrank_feavec.out');
    infile = open(infilePath);
    outfile = open(outfilePath, 'w');
    for ln in infile:
        (nquery, layout, weight, avgfv) = ln.strip().split('\x01');
        layout = eval(layout);
        weight = eval(weight);
        avgfv = eval(zlib.decompress(base64.b64decode(avgfv)));
        for j in range(len(indexToVName)):
            if(layout[j] is -1): continue;  # vertical not available
            # compare to verticals
            for vIdx in range(len(indexToVName)):
                if(layout[vIdx] is -1): continue;
                if((layout[j] is not None) and ((layout[vIdx] is None) or (layout[j] < layout[vIdx]))):
                    dumpPairFeature(extractPairFeature(avgfv, indexToVName[j], indexToVName[vIdx]), outfile, weight);
            # compare to web slot
            if(layout[j] is not None):
                for wIdx in range(1, 11):
                    wName = 'w' + str(wIdx);
                    if(wIdx >= vIdx): dumpPairFeature(extractPairFeature(avgfv, indexToVName[j], wName), outfile, weight);
                    else: dumpPairFeature(extractPairFeature(avgfv, wName, indexToVName[j]), outfile, weight);
    infile.close();
    outfile.close();
    return;

if __name__ == '__main__':
    svmRankFeatureExtraction(None);
    pass
