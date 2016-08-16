'''
Created on Jul 2, 2013

@author: wangxl
'''
import optparse;
import sys;
sys.path.append('.');
import random;
import vw_feature_extract;
import utility;
import zlib;
import base64;

def attachTags(args):
    for line in sys.stdin:
        k = random.random();
        if(k < 0.5): tag = "data_for_click";
        else: tag = "data_for_annotation";
        print('{0}\t{1}'.format(tag, line));
    return;

def sliceClickData(args):
    for line in sys.stdin:
        (k, v) = line.split('\t', 1);
        if(k == 'data_for_click'): print(v);
    return;

def sliceAnnotationData(args):
    for line in sys.stdin:
        (k, v) = line.split('\t', 1);
        if(k == 'data_for_annotation'): print(v);
    return;

def checkIfGoodVerticalPage(eventLst):
    hasVertical = False;
    verticalAllClicked = True;
    for e in eventLst:
        if(vw_feature_extract.checkIfSlotIsVertical(e['v'])): hasVertical = True;
        if(vw_feature_extract.checkIfSlotIsVertical(e['v']) and e['r'] <= 0): verticalAllClicked = False;
    return (hasVertical and verticalAllClicked);        
    
def collectEditorialAnnotationMap(args):
    outputDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(checkIfGoodVerticalPage(eventLst)):
            verLstCompRep = vw_feature_extract.getVerticalCompactRep(eventLst);
            key = str((nquery, verLstCompRep));
            outputDict[key] = outputDict.get(key, 0) + 1;
    for (k, v) in outputDict.iteritems(): print(vw_feature_extract.packHadoopLine(k, v));
    
def collectEditorialAnnotationReduce(args):
    outputDict = {};
    for ln in sys.stdin:
        (k, v) = vw_feature_extract.unpackHadoopLine(ln);
        v = utility.parseNumVal(v);
        (nquery, verLstCompRep) = eval(k);
        if nquery not in outputDict: outputDict[nquery] = {};
        outputDict[nquery][verLstCompRep] = outputDict[nquery].get(verLstCompRep, 0) + v;
    for nquery in outputDict:
        for verLstCompRep in sorted(outputDict[nquery], key=lambda x: outputDict[nquery][x], reverse=True):
            print(str((outputDict[nquery][verLstCompRep], nquery, verLstCompRep)));    
    
def collectMTAnnotationMap(args):
    outputDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        for e in eventLst:
            if(vw_feature_extract.checkIfSlotIsVertical(e['v'])):
                s = vw_feature_extract.getSlotPos(e['s']);
                v = vw_feature_extract.getFullSlotName(e['v'], e['s']);
                r = e['r'];
                bgKey = str(('=|bk|=', v));
                qrKey = str((nquery, v));
                if(bgKey not in outputDict): outputDict[bgKey] = [0.0, 0.0];
                if(qrKey not in outputDict): outputDict[qrKey] = [0.0, 0.0];
                if(r == 1): 
                    outputDict[bgKey][1] += 1.0;
                    outputDict[qrKey][1] += 1.0;
                else: 
                    outputDict[bgKey][0] += 1.0;
                    outputDict[qrKey][0] += 1.0;
    for (key, val) in outputDict.iteritems(): print(vw_feature_extract.packHadoopLine(key, val));
    return;

def collectMTAnnotationReduce(args):
    inputDict = {};
    tmpDict = {};
    outputDict = {};
    for ln in sys.stdin:
        (key, val) = vw_feature_extract.unpackHadoopLine(ln);
        val = eval(val);
        if(key not in inputDict): inputDict[key] = [0.0, 0.0];
        inputDict[key][0] += val[0];
        inputDict[key][1] += val[1];
    for (key, val) in inputDict.iteritems():
        (nquery, v) = eval(key);
        if(nquery == '=|bk|='): continue;
        bgVal = inputDict[str(('=|bk|=', v))];
        ctr = val[1] / (val[0] + val[1]);
        bgCtr = bgVal[1] / (bgVal[0] + bgVal[1]);
        ctrAgainstBg = (ctr / bgCtr) if(bgCtr != 0) else 0;  # avoid 0/0
        tmpKey = str((nquery, vw_feature_extract.getVerticalName(v)));
        if(tmpKey not in tmpDict): tmpDict[tmpKey] = [];
        tmpDict[tmpKey].append(ctrAgainstBg);
    for tmpKey in tmpDict: outputDict[tmpKey] = float(sum(tmpDict[tmpKey])) / len(tmpDict[tmpKey]);
    for key in sorted(outputDict, key=lambda x:outputDict[x], reverse=True): 
        (nquery, v) = eval(key);
        avgCtr = outputDict[key];
        print(str((avgCtr, nquery, v)));
    return;
                
def sortMTResult(args):
    dict = {};
    for ln in sys.stdin:
        (nquery, v, ctr) = eval(ln.strip());
        dict[ln] = ctr;
    for ln in sorted(dict, key=lambda x: dict[x], reverse=True): print(ln);
    return;

def verDistMap(args):
    outDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(nquery not in outDict): outDict[nquery] = {};
        for e in eventLst:
            if(vw_feature_extract.checkIfSlotIsVertical(e['v'])): 
                vsName = vw_feature_extract.getFullSlotName(e['v'], e['s']);
                outDict[nquery][vsName] = outDict[nquery].get(vsName, 0) + 1;
    for nquery in outDict: print(vw_feature_extract.packHadoopLine(nquery, outDict[nquery]));
    return;

def verDistReduce(args):
    outDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (key, val) = vw_feature_extract.unpackHadoopLine(ln);
        val = eval(val);
        if(key not in outDict): outDict[key] = {};
        for k, v in val.iteritems(): outDict[key][k] = outDict[key].get(k, 0) + v;
    for (key, val) in outDict.iteritems(): print(vw_feature_extract.packHadoopLine(key, val));
    return;

def verDistDebug(args):
    verQueryNumDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (nquery, dict) = vw_feature_extract.unpackHadoopLine(ln);
        dict = eval(dict);
        verQueryNumDict[len(dict)] = verQueryNumDict.get(len(dict), 0) + 1;
    for k in sorted(verQueryNumDict): print(k, verQueryNumDict[k]);

def getNqueryNumMap(args):
    outputDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        outputDict[nquery] = outputDict.get(nquery, 0) + 1;
    for (k, v) in outputDict.iteritems(): print(vw_feature_extract.packHadoopLine(k, v));

def getNqueryNumReduce(args):
    outputDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (nquery, v) = vw_feature_extract.unpackHadoopLine(ln);
        v = eval(v);
        outputDict[nquery] = outputDict.get(nquery, 0) + v;
    for nquery in sorted(outputDict, key=lambda x:outputDict[x], reverse=True):
        print(vw_feature_extract.packHadoopLine(nquery, outputDict[nquery]));


def getQueryFeaLstMap(args):
    q = "facebook";
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(nquery == q): print(vw_feature_extract.packHadoopLine(q, str((fv, eventLst))));

def checkFsStructureMap(args):
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        for k1 in fv:
            if(not isinstance(fv[k1], dict)): 
                print('false');
                return;
            for k2 in fv[k1]:
                if(not isinstance(fv[k1][k2], dict)):
                    print('false');
                    return;
                for k3 in fv[k1][k2]:
                    if(not isinstance(fv[k1][k2][k3], (int, long, float))):
                        print('false');
                        return

def _mergeFv(fv, newFv, multiplier=1.0):
    for k1 in newFv:
        if(k1 not in fv): fv[k1] = {};
        for k2 in newFv[k1]:
            if(k2 not in fv[k1]): fv[k1][k2] = {};
            for (k3, v3) in newFv[k1][k2].iteritems():
                fv[k1][k2][k3] = fv[k1][k2].get(k3, 0.0) + v3 * multiplier;
    return;

def _scaleFv(fv, cnt):
    for k1 in fv:
        for k2 in fv[k1]:
            for (k3, v3) in fv[k1][k2].iteritems():
                fv[k1][k2][k3] /= float(cnt);
    return fv;

def avgFvMap(args):
    outputDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(nquery not in outputDict): outputDict[nquery] = {'cnt': 0, 'fv': {}};
        outputDict[nquery]['cnt'] += 1;
        _mergeFv(outputDict[nquery]['fv'], fv);
    for nquery in outputDict:
        print(vw_feature_extract.packHadoopLine(nquery, (outputDict[nquery]['cnt'], outputDict[nquery]['fv'])));
    return;

def avgFvReduce(args):
    outputDict = {};
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (nquery, tuple) = vw_feature_extract.unpackHadoopLine(ln);
        (cnt, fv) = eval(tuple);
        if(nquery not in outputDict): outputDict[nquery] = {'cnt': 0, 'fv': {}};
        outputDict[nquery]['cnt'] += cnt;
        _mergeFv(outputDict[nquery]['fv'], fv, cnt);
    for nquery in outputDict:
        _scaleFv(outputDict[nquery]['fv'], outputDict[nquery]['cnt']);
    for nquery in sorted(outputDict, key=lambda x:outputDict[x]['cnt'], reverse=True):
        print(vw_feature_extract.packHadoopLine(nquery, (outputDict[nquery]['cnt'], outputDict[nquery]['fv'])));
    return;
                    
def avgFvPigStream(args):
    cnt = 0;
    avgFv = {};
    curNquery = None;    
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        if(curNquery is None): curNquery = nquery;
        if(curNquery != nquery): 
            outputLn = base64.b64encode(zlib.compress(str(_scaleFv(avgFv, cnt))));
            print('{0}\x01{1}'.format(curNquery, outputLn));
            cnt = 0;
            avgFv = {};
            curNquery = nquery;
        cnt += 1;
        _mergeFv(avgFv, fv);
    outputLn = base64.b64encode(zlib.compress(str(_scaleFv(avgFv, cnt))));
    print('{0}\x01{1}'.format(curNquery, outputLn));
    return;

def getNquery(args):
    for ln in sys.stdin:
        if(not ln.strip()): continue;
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        print('{0}\x01{1}'.format(nquery, ln.strip()));

    
if __name__ == '__main__':
    optParser = optparse.OptionParser("");
    optParser.add_option('-a', '--attach_tags', action='store_true', dest='attachTags', default=False, help='attach tags for click / annotation');
    optParser.add_option('-o', '--output_data', dest='portion', help='select portion of data for output');
    (options, args) = optParser.parse_args();
    if(options.attachTags): attachTags(args);
    elif(options.portion == 'click'): sliceClickData(args);
    elif(options.portion == 'annotation'): sliceAnnotationData(args);        
    pass
