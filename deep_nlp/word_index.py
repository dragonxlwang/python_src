'''
Created on May 4, 2013

@author: xwang95
'''
import re;
import toolkit.utility;

class WordIndexer(object):
    '''
    classdocs
    '''
    wordToIdTable = {};
    idToWordTable = {};
    vocabularySize = 0;
    wordCountMode = False;
    wordCountTable = {};
        
    def __init__(self, wordIndexerFilePath=None, wordCountMode=False):
        '''
        Constructor
        '''
        self.wordToIdTable = {};
        self.idToWordTable = {};
        self.vocabularySize = 0;
        if(wordCountMode): self.turnonWordCountMode();
        else: self.turnoffWordCountMode();
        if(wordIndexerFilePath is not None):
            self.loadWordIndexer(wordIndexerFilePath);
        return;    
    
    def turnonWordCountMode(self):
        self.wordCountMode = True;
        self.wordCountTable = {};
        return;
    
    def turnoffWordCountMode(self):
        self.wordCountMode = False;
        self.wordCountTable = None;
        return;
    
    def _updateIndex(self, word):
        if(word not in self.wordToIdTable):
            self.wordToIdTable[word] = self.vocabularySize;
            self.idToWordTable[self.vocabularySize] = word;
            self.vocabularySize += 1;
        return;
    
    def ifWordIndexed(self, word): return (word in self.wordToIdTable);
    
    def getWordIndex(self, word, autoInsert=True):
        if(autoInsert): self._updateIndex(word); 
        return self.wordToIdTable.get(word, -1);
        
    def getTokenLstIndexLst(self, tokenLst, autoInsert=True, removeNone=True): 
        idLst = [self.getWordIndex(token, autoInsert) for token in tokenLst];
        if(self.wordCountMode):
            for x in idLst: 
                self.wordCountTable[x] = self.wordCountTable.get(x, 0) + 1.0;
        if(removeNone): idLst = [x for x in idLst if(x != -1)];
        return idLst;
        
    def getTokenLstIndexBow(self, tokenLst, autoInsert=True, removeNone=True):
        bow = {};
        for id in self.getTokenLstIndexLst(tokenLst, autoInsert, removeNone):
            bow[id] = bow.get(id, 0) + 1.0;
        return bow;
    
    def dumpWordIndexer(self, filePath):
        toolkit.utility.writeDictFile(filePath, self.idToWordTable);
        return;
        
    def loadWordIndexer(self, filePath, append=False):
        self.idToWordTable = toolkit.utility.readDictFile(filePath,
                                            ifKeyNum=True, ifValNum=False);
        if(append):
            for (id, word) in self.idToWordTable.items():
                self._updateIndex(word);
        else:
            self.idToWordTable = {};
            for (id, word) in self.idToWordTable.items():
                self.wordToIdTable[word] = id;
                self.vocabularySize = len(self.idToWordTable);
        return;
    
    def dumpWordCounts(self, filePath):
        wordToCountTable = {};
        for k in self.wordCountTable: 
            wordToCountTable[self.idToWordTable[k]] = self.wordCountTable[k]; 
        toolkit.utility.writeDictFile(filePath, wordToCountTable,
                                      keylst=sorted(wordToCountTable,
                                        key=lambda x:-wordToCountTable[x]));
        return;
    
    def cutoffLowFreqWords(self, threshold):
        idToWordTable = {};
        wordToIdTable = {};
        vocabularySize = 0;
        # rebuild index
        for id in self.wordCountTable:
            cnt = self.wordCountTable[id];
            if(cnt >= threshold):
                word = self.idToWordTable[id];
                wordToIdTable[word] = vocabularySize;
                idToWordTable[vocabularySize] = word; 
                vocabularySize += 1;
        self.wordToIdTable = wordToIdTable;
        self.idToWordTable = idToWordTable;
        self.vocabularySize = vocabularySize;
        self.turnoffWordCountMode();
        return;
        

if(__name__ == '__main__'):
    str = '''in 1999, cgattii emerged on the east coast of vancouver island 
            (vi), british columbia (bc), [canada] ( --ref_pmid=-1-- ), and is 
            now considered endemic in the environment ( --ref_pmid=-1-- , 
            --ref_pmid=-1-- ), affecting human ( --ref_pmid=17370514-- ) and 
            animal populations ( --ref_pmid=-1-- ) travel histories of patients 
            have been used to monitor fungal spread ( --ref_pmid=17370514-- ) 
            and to estimate the incubation period of this disease ( 
            --ref_pmid=17370544-- , --ref_pmid=-1-- )''';
    print(str);
    wordIndexer = WordIndexer();
    print wordIndexer.getTokenLstIndexLst(str);
    print wordIndexer.idToWordTable[4];
    
