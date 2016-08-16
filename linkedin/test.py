'''
Created on Jun 10, 2014

@author: xwang1
'''
import sys;
import re;
from linkedin.const import cleanedTextFilePath
from deep_nlp.text import preprocText, postag, nerChunk, langIden, \
    postagCoarseTag
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords

def func():
    file = open(cleanedTextFilePath);
    for ln in file:
        if(len(ln.strip()) == 0): continue;        
        toks = preprocText(ln, stemmingOption=False, rmStopwordsOption=False);
        for senToks in toks:
            print(senToks);
            print(postag(senToks));
            print(nerChunk(senToks));
        sys.stdin.readline();
    file.close();
    return;

def entityExtract():
    file = open(cleanedTextFilePath);
    entityDict = {};
    regTag = re.compile('^(.*?):::(.*?)$'); 
    for ln in file:
        if(len(ln.strip()) == 0): continue;
        (tag, comment) = regTag.match(ln).group(1, 2);        
        toks = preprocText(comment.lower(), stemmingOption=False, 
                           rmStopwordsOption=False);
        lang = langIden([tok for senToks in toks for tok in senToks]);
        if(lang != 'english'): continue;        
        postags = [postagCoarseTag(senToks) for senToks in toks];
        for s in range(len(toks)):
            i = 0;
            str = '';
            while(i < len(toks[s])):
                j = i;
                flag = False;
                while(j < len(toks[s]) and postags[s][j] in 
                      ['NN', 'JJ', 'VB', 'RB']):
                    if(postags[s][j] in ['NN', 'VB']): flag = True;
                    j += 1;
                if(i != j and flag):
                    str = ' '.join(toks[s][i:j]);
                    pos = ' '.join(postags[s][i:j]);
                    entityDict[str] = entityDict.get(str, 0) + 1;
#                     print(tag);
#                     print(i, j);
#                     print(str);
#                     print(pos);
#                     print(toks[s]);
#                     print(postags[s]);
#                     sys.stdin.readline();
                i = j + 1;
    file.close();
    for ent in sorted(entityDict, key=lambda x: -entityDict[x]):
        print(ent);
        sys.stdin.readline();
    return;

if __name__ == '__main__':
    entityExtract();
#     func();
    pass
