'''
Created on May 3, 2013

@author: xwang95
'''
import corpus.pubmed;
import toolkit.utility;
import toolkit.variables;
import os;
import deep_nlp.word_index;
import sys;
import re;
import senti_gibbs_sampler;
from toolkit import utility

def procData(outCitFilePath):
    citCtxtLst = [];
    cutoffVal = 5;
    pmd = corpus.pubmed.getPubMedCorpus();
        
    for citingPmid in pmd.docs:
        if('citLst' in pmd.docs[citingPmid]): citCtxtLst.extend(pmd.docs[citingPmid]['citLst']);
    print('total citation conunt: {0}'.format(len(citCtxtLst)));
    
    paperMaxCocitedNum = {};
    paperCitCnt = {};
    for citCtxt in citCtxtLst:
        paperCitCnt[citCtxt['citedDocPmid']] = paperCitCnt.get(citCtxt['citedDocPmid'], 0) + 1;
        paperMaxCocitedNum[citCtxt['citedDocPmid']] = max(paperMaxCocitedNum.get(citCtxt['citedDocPmid'], 0), len(citCtxt['coCitedDocPmidLst']) - 1);
    
    ''' get histogram of citation count,  co-citation number '''
    print('paper max cocited histogram:');
    toolkit.utility.printFreqTable(toolkit.utility.getFreqTable(paperMaxCocitedNum));
    print('paper citation count histogram:');
    toolkit.utility.printFreqTable(toolkit.utility.getFreqTable(paperCitCnt));
    
    filteredPmidSet = set([pmid for pmid in paperCitCnt if(paperCitCnt[pmid] >= cutoffVal)]);
    paperMaxCocitedNum = {};
    paperCitCnt = {};
    for citCtxt in citCtxtLst:
        if(citCtxt['citedDocPmid'] not in filteredPmidSet): continue;
        paperCitCnt[citCtxt['citedDocPmid']] = paperCitCnt.get(citCtxt['citedDocPmid'], 0) + 1;
        paperMaxCocitedNum[citCtxt['citedDocPmid']] = max(paperMaxCocitedNum.get(citCtxt['citedDocPmid'], 0),
                                                          len([x for x in citCtxt['coCitedDocPmidLst'] if((citCtxt['citedDocPmid'] != x) and (x in filteredPmidSet))]));
    
    ''' after filtering '''
    ''' get histogram of citation count,  co-citation number '''
    print('paper max cocited histogram:');
    toolkit.utility.printFreqTable(toolkit.utility.getFreqTable(paperMaxCocitedNum));
    print('paper citation count histogram:');
    toolkit.utility.printFreqTable(toolkit.utility.getFreqTable(paperCitCnt));
    
    ''' dump filtered file '''
    citFile = open(outCitFilePath, 'w');
    for citCtxt in citCtxtLst:
        if(citCtxt['citedDocPmid'] not in filteredPmidSet): continue;
        citCtxt['coCitedDocPmidLst'] = [x for x in citCtxt['coCitedDocPmidLst'] if((citCtxt['citedDocPmid'] != x) and (x in filteredPmidSet))];
        citFile.write('{0}\n'.format(citCtxt['citingDocPmid']));
        citFile.write('{0}\n'.format(citCtxt['citedDocPmid']));
        citFile.write('{0}\n'.format(citCtxt['coCitedDocPmidLst']));
        citFile.write('{0}\n'.format(citCtxt['txt']));
        citFile.write('\n');
    citFile.close();
    return;

def cleanData(inCitFilePath, outCitFilePath):
    (citMetaGraph, citDict) = corpus.pubmed.readCitationFile(inCitFilePath);
    refReg = re.compile('--ref_pmid=.*?--');
    parenthesesReg = re.compile('[\(\)\[\]]');
    outCitFile = open(outCitFilePath, 'w');

    for pmid in citDict:
        for citCtxt in citDict[pmid]:
            txt = citCtxt['txt'];
            txt = refReg.sub('', txt);
            txt = parenthesesReg.sub('', txt);
            tokenLst = [];
            for lst in deep_nlp.text.preprocText(txt, False, True): tokenLst.extend(lst);
            tokenLst = [token.lower() for token in tokenLst if(len(token) >= 3)];
            citCtxt['txt'] = ' '.join(tokenLst);
            
            outCitFile.write('{0}\n'.format(citCtxt['citingDocPmid']));
            outCitFile.write('{0}\n'.format(citCtxt['citedDocPmid']));
            outCitFile.write('{0}\n'.format(citCtxt['coCitedDocPmidLst']));
            outCitFile.write('{0}\n'.format(citCtxt['txt']));
            outCitFile.write('\n');    
    outCitFile.close();
    return;

def buildIndex(citFilePath, wordIndexFilePath, cutoff, reserveWordSet):
    (citMetaGraph, citDict) = corpus.pubmed.readCitationFile(citFilePath);
    citDict.pop(0, None);  # temporally fix a bug
    
    wordIndexer = deep_nlp.word_index.WordIndexer();
    tfTable = {};
    j = 0;
    for pmid in citDict:
        for entry in citDict[pmid]:
            for token in entry['txt'].split():
                tfTable[token] = tfTable.get(token, 0) + 1;
    for (k, v) in tfTable.items(): 
        if(v >= cutoff): wordIndexer.updateIndex(k);
        elif(v in reserveWordSet): 
            wordIndexer.updateIndex(k);
            j += 1;
    wordIndexer.dumpWordIndexer(wordIndexFilePath);
    print('vocabulary size: {0}'.format(wordIndexer.vocabularySize));
    print('reserved word size: {0} ({1})'.format(len(reserveWordSet), j));
    return;
    
def getData(citFilePath, wordIndexFilePath):
    (citMetaGraph, citDict) = corpus.pubmed.readCitationFile(citFilePath);
    citDict.pop(0, None);  # temporally fix a bug
    dSet = set();
    for pmid in citDict:
        for entry in citDict[pmid]: dSet.add(entry['citedDocPmid']);
    
    ''' generate reference list '''
    refLstTable = {};
    for pmid in citDict: 
        if(pmid not in refLstTable): refLstTable[pmid] = set();
        for entry in citDict[pmid]: refLstTable[pmid].add(entry['citedDocPmid']);
    ''' generate constraint list '''
    constrLstTable = {};
    for pmid in citDict:
        constrLstTable[pmid] = {};
        for entry in citDict[pmid]:
            citedPmid = entry['citedDocPmid'];
            if(citedPmid not in constrLstTable[pmid]): constrLstTable[pmid][citedPmid] = {};
            for coCitedPmid in entry['coCitedDocPmidLst']:
                base = toolkit.utility.factorial(len(entry['coCitedDocPmidLst']) + 1) / 2;
                if(coCitedPmid not in refLstTable[pmid]): continue;  # temporaly fix bug
                constrLstTable[pmid][citedPmid][coCitedPmid] = constrLstTable[pmid][citedPmid].get(coCitedPmid, 0.0) + 1.0 / base;
    ''' generate citation context '''
    citationTxtTable = {};
    wordIndexer = deep_nlp.word_index.WordIndexer(filePath=wordIndexFilePath);
    for pmid in citDict:
        if(pmid not in citationTxtTable): citationTxtTable[pmid] = {};
        for entry in citDict[pmid]:
            citedPmid = entry['citedDocPmid'];
            txt = entry['txt'];
            if(citedPmid not in citationTxtTable[pmid]): citationTxtTable[pmid][citedPmid] = [];
            citationTxtTable[pmid][citedPmid].extend(wordIndexer.getTokenLstIndexLst([token for token in txt.split() if(wordIndexer.ifWordIndexed(token))]));
    
    sentimentAnalyzer = deep_nlp.sentiment.SentimentAnalyzer(os.path.join(toolkit.variables.RESOURCE_DIR, 'lex/clues.lex'));
    
#     i = 0;
#     for pmid in constrLstTable:
#         for refId in constrLstTable[pmid]:
#             for coCitedId in constrLstTable[pmid][refId]:
#                 if(coCitedId not in refLstTable[pmid]):
#                     print('{3} wrong! {0}, {1}, {2}'.format(pmid, refId, coCitedId, i));
#                     i += 1;
#     print('finished')
#     print j;
#     sys.stdin.read();
                    
    return (refLstTable, constrLstTable, citationTxtTable, wordIndexer, sentimentAnalyzer);

def runAlgo(citFilePath, wordIndexFilePath, burnInHr, sampliHr, dumpFilePath, violatedConstraintWeightFilePath):
    (refLstTable, constrLstTable, citationTxtTable, wordIndexer, sentimentAnalyzer) = getData(citFilePath, wordIndexFilePath);
#     sentiGibbsSampler = senti_gibbs_sampler.SentiGibbsSampler(refLstTable, constrLstTable, citationTxtTable, wordIndexer, sentimentAnalyzer, 1e-2, 1e-1, 1e-4);
    sentiGibbsSampler = senti_gibbs_sampler.SentiGibbsSampler(refLstTable, constrLstTable, citationTxtTable, wordIndexer, sentimentAnalyzer, 1.0, 1e-1, 1e-4);
    sentiGibbsSampler.run(burnInHr, sampliHr);
    sentiGibbsSampler.dumpPosteriorLabelTableFile(dumpFilePath);
    sentiGibbsSampler.dumpViolatedConstraintWeightFile(violatedConstraintWeightFilePath);
    return;    

def postAnalysis(citFilePath, wordIndexFilePath, labelFilePath, goldenStandardFilePath):
    (refLstTable, constrLstTable, citationTxtTable, wordIndexer, sentimentAnalyzer) = getData(citFilePath, wordIndexFilePath);
    labelLst = [];
    for ln in open(labelFilePath, 'r'): labelLst.append([toolkit.utility.parseNumVal(x) for x in ln.strip().split('<=:|:=>')]);
    sentiScorePerRefId = {};
    sentiPmidRefIdTable = {};
    for (pmid, refId, pos, neg) in labelLst: 
        if(refId not in sentiScorePerRefId): sentiScorePerRefId[refId] = [0.0, 0.0];
        sentiScorePerRefId[refId] = [sentiScorePerRefId[refId][0] + neg / 50.0, sentiScorePerRefId[refId][1] + pos / 50.0];
        if(pmid not in sentiPmidRefIdTable): sentiPmidRefIdTable[pmid] = {};
        sentiPmidRefIdTable[pmid][refId] = [neg, pos];
    
    # debug
#     while(True):
#         pmid = toolkit.utility.parseNumVal(raw_input('pmid:'));
#         refId = toolkit.utility.parseNumVal(raw_input('refId:'));
#         print('{0}:{1}'.format(*sentiPmidRefIdTable[pmid][refId]));
    # end of debug
    
    goldenStandardLst = [];
    correctCnt = 0;
    totalCnt = 0;
    for ln in open(goldenStandardFilePath): goldenStandardLst.append([toolkit.utility.parseNumVal(x) for x in ln.split() if x.strip()]);
    for (pmid, refId, gsLabel) in goldenStandardLst:
        negPosCntVec = sentiPmidRefIdTable[pmid][refId];
        if(negPosCntVec[gsLabel] > negPosCntVec[1 - gsLabel]): correct = True;
        else: correct = False;
        if(correct): correctCnt += 1;
        totalCnt += 1;
        print('{0} \t {1} \t {2} \t {3} \t {4} \t {5}'.format(pmid, refId, gsLabel, negPosCntVec[0], negPosCntVec[1], correct));
    print('{0} \t {1}'.format(correctCnt, totalCnt));
    return;

if __name__ == '__main__':
    outCitFilePath = os.path.join(toolkit.variables.CIT_SENTI_RES_DIR, 'pubmed_citation_filtered.txt');
    outCleanedCitFilePath = os.path.join(toolkit.variables.CIT_SENTI_RES_DIR, 'pubmed_citation_filtered.cleaned.txt');
    wordIndexFilePath = os.path.join(toolkit.variables.CIT_SENTI_RES_DIR, 'pubmed_citation_filtered.wordindex');
    posteriorLabelTableFilePath = os.path.join(toolkit.variables.CIT_SENTI_RES_DIR, 'pubmed_citation_filtered3.label');
    violatedConstraintWeightFilePath = os.path.join(toolkit.variables.CIT_SENTI_RES_DIR, 'pubmed_citation_filtered3.constraint');
    goldenStandardFilePath = os.path.join(toolkit.variables.CIT_SENTI_RES_DIR, 'pubmed_citation_golden.txt');
#     procData(outCitFilePath);
#     cleanData(outCitFilePath, outCleanedCitFilePath);
#     buildIndex(outCleanedCitFilePath, wordIndexFilePath, 10, set(deep_nlp.sentiment.SentimentAnalyzer(os.path.join(toolkit.variables.RESOURCE_DIR, 'lex/clues.lex')).lex.keys()));
#     (refLstTable, constrLstTable, citationTxtTable, wordIndexer, sentimentAnalyzer) = getData(outCleanedCitFilePath, wordIndexFilePath);
    runAlgo(outCleanedCitFilePath, wordIndexFilePath, 1, 1, posteriorLabelTableFilePath, violatedConstraintWeightFilePath);
#     postAnalysis(outCleanedCitFilePath, wordIndexFilePath, posteriorLabelTableFilePath, goldenStandardFilePath);
    pass
