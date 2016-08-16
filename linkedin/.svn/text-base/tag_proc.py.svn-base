'''
Created on Jun 3, 2014

@author: xwang1
'''
from linkedin.const import cleanedTextFilePath, dataDir, workingFilePath, \
    stemFilePath, vocabularyFilePath, vocabularySize, globalTxtFilePath, \
    globalVocTableFilePath, globalVocFilteredLstPath, globalFvFilePath,\
    globalXSmFilePath
import os.path;
import re;
import sys;
import re;
from linkedin.const import cleanedTextFilePath, cleanedTextEngFilePath
from deep_nlp.text import preprocText, postag, nerChunk, langIden, \
    postagCoarseTag, stopwordsSet
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from toolkit.num.sparse import toSparseVec, setSmElem, getSmElem
from toolkit.utility import writeObjToFile, loadObjFromFile

tranRePatt = re.compile(r'^(.*?):::(.*?)$');


def printCatDist():
    catTbl = {};
    file = open(cleanedTextFilePath);
    for ln in file:
        if(len(ln.strip()) == 0): continue;
        cat = tranRePatt.search(ln).group(1);
        catTbl[cat] = catTbl.get(cat, 0) + 1;
    catDistFile = open(os.path.join(dataDir, "cat_dist"), "w");
    for cat in sorted(catTbl, key=lambda x:-catTbl[x]):
        catDistFile.write("{0}, {1}\n".format(cat, catTbl[cat]));
    file.close();
    catDistFile.close();
    return;

def func():
    catTbl = {};
    file = open(cleanedTextFilePath);
    cnt = 0;
    outFile = open(os.path.join(dataDir, "test"), "w");
    for ln in file:
        if(len(ln.strip()) == 0): continue;
        (cat, com) = tranRePatt.search(ln).group(1, 2);
        if(cat == 'Global' and 'recommendation' in com.lower()):
            cnt += 1;
            outFile.write(com + '\n');
    print cnt;
    file.close();
    outFile.close();
    return;

def preproc(filename):
    file = open(filename);
    outfile = open(filename + ".eng", 'w');
    regTag = re.compile('^(.*?):::(.*?)$');
    engData = {}; 
    i = 0;
    j = 0;
    for ln in file:
        if(len(ln.strip()) == 0): continue;
        i += 1;
        if(i % 1000 == 0): 
            sys.stdout.write('{0} processed, {1} collected\n'.format(i, j));
            sys.stdout.flush();
        (tag, comment) = regTag.match(ln).group(1, 2);        
        toks = preprocText(comment.lower(), stemmingOption=False,
                           rmStopwordsOption=False);
        lang = langIden([tok for senToks in toks for tok in senToks]);
        if(lang != 'english'): continue;
        j += 1;
        if(tag not in engData): engData[tag] = [];
        engData[tag].append(comment);
    for tag in engData: 
        for comment in engData[tag]:
            outfile.write('{0}:::{1}\n'.format(tag, comment));
    outfile.close();
    file.close();
    return;

def getTag():
    file = open(cleanedTextEngFilePath);
    outfile = open(cleanedTextEngFilePath + ".tag", "w");
    tagDict = {};
    i = 0;
    for ln in file:
        (tag, comment) = tranRePatt.match(ln).group(1, 2);
        tagDict[tag] = tagDict.get(tag, 0) + 1;
        i += 1;
        if(i % 10000 == 0): 
            sys.stdout.write("{0} processed \r".format(i));
            sys.stdout.flush();
    for tag in sorted(tagDict, key=lambda x:-tagDict[x]):
        outfile.write("{0}:::{1}\n".format(tag, tagDict[tag]));
    file.close();
    outfile.close();
    return;

def filterByTag():
    file = open(cleanedTextEngFilePath);
    tagfile = open(cleanedTextEngFilePath + ".tag");
    outfile = open(cleanedTextEngFilePath + ".filtered_by_tag", "w");
    filteredTagSet = set();
    for ln in tagfile:
        (tag, cnt) = tranRePatt.match(ln).group(1, 2);
        if(len(filteredTagSet) < 15): filteredTagSet.add(tag.strip());
    for ln in file:
        (tag, comment) = tranRePatt.match(ln).group(1, 2);
        if(tag.strip() in filteredTagSet): outfile.write(ln);
    file.close();
    tagfile.close();
    outfile.close();


         
#===============================================================================
# extract only those with tag "global"
#===============================================================================
def _getAlphabetSubStr(s):
    return ''.join([x for x in s.lower() 
                    if(ord(x) <= ord('z') and ord(x) >= ord('a'))]);

def workingFileFilterByTag():
    selectedTag = "global";
    file = open(workingFilePath);
    outFile = open(workingFilePath + ".global", "w");
    i = 0;
    outputSet = set();
    for ln in file:
        i += 1;
        if(i % 10000 == 0):
            sys.stderr.write("{0} processed\r".format(i));
            sys.stderr.flush();
        (tag, comment) = tranRePatt.match(ln).group(1, 2);
        if(tag.lower() == selectedTag): 
            outputSet.add(ln);
            if(len(outputSet) >= 50000): break;
    for ln in outputSet: outFile.write(ln);
    file.close();
    outFile.close();
    return;

def stem():
    file = open(globalTxtFilePath);
    stemTbl = {};
    i = 0;
    for ln in file:
        i += 1;
        if(i % 10000 == 0):
            sys.stdout.write("{0} processed \r".format(i));
            sys.stdout.flush();
        (tag, comment) = tranRePatt.match(ln).group(1, 2);
        try:
            txt = preprocText(comment.lower(), stemmingOption=True,
                              rmStopwordsOption=True);
        except:
            continue;
        for sent in txt:
            for tok in sent:
                if(tok in stopwordsSet):
                    print tok;
                    sys.stdin.readline();
                if(tok == "on"):
                    print tok;
                    sys.stdin.readline();
                stemTbl[tok] = stemTbl.get(tok, 0) + 1;
    writeObjToFile(globalVocTableFilePath, stemTbl);
    file.close();
    return;

def constructVoc():
    vocTbl = loadObjFromFile(globalVocTableFilePath);
    vocLst = sorted(vocTbl, key=lambda x:-vocTbl[x]);
    vocLst = [x for x in vocLst if(len(_getAlphabetSubStr(x)) > 0)];
    vocLst = vocLst[0:15000];
    writeObjToFile(globalVocFilteredLstPath, vocLst);
    return;

def featureExtract():
    file = open(globalTxtFilePath);
    vocLst = loadObjFromFile(globalVocFilteredLstPath);
    vocDict = {};
    for i in range(len(vocLst)): vocDict[vocLst[i]] = i;
    i = 0;
    outLst = [];
    for ln in file:
        i += 1;
        if(i % 10000 == 0):
            sys.stderr.write("{0} processed \r".format(i));
            sys.stderr.flush();
        (tag, comment) = tranRePatt.match(ln).group(1, 2);
        try:
            txt = preprocText(comment.lower(), stemmingOption=True,
                              rmStopwordsOption=True);
        except:
            continue;
        feaVec = [[vocDict[tok] for tok in sent if tok in vocDict] 
                  for sent in txt];
        feaVec = [sf for sf in feaVec if(len(sf) != 0)];
        if(len(feaVec) == 0): continue;
        outLst.append([feaVec, tag, comment]);
    writeObjToFile(globalFvFilePath, outLst);
    file.close();
    return;

def extractXsm():
    recLst = loadObjFromFile(globalFvFilePath);    
    sentNum = 0;
    for record in recLst:
        (fv, tag, comment) = record;
        sentNum += len(fv);
    xSm = ({}, (vocabularySize, sentNum));
    sentIdx = 0;
    for record in recLst:
        (fv, tag, comment) = record;
        for seq in fv:
            for wordIdx in seq:
                setSmElem(xSm, wordIdx, sentIdx,
                          getSmElem(xSm, wordIdx, sentIdx) + 1.0);
            sentIdx += 1;
    writeObjToFile(globalXSmFilePath, xSm);
    return;
    
if __name__ == '__main__':
#     printCatDist();
#     func();
#     preproc(sys.argv[1]);
#     getTag();
#     filterByTag();
#     stem();
#     featureExtract(sys.argv[1]);
#     workingFileFilterByTag();
#     extractXsm();
#     stem();
#     constructVoc();
#     featureExtract();
    pass
