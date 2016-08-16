'''
Created on Feb 1, 2013

@author: xwang95
'''
import nltk;
import nltk.tokenize;
import nltk.stem;
import nltk.chunk;
from nltk.corpus import stopwords;


stemmer = nltk.stem.PorterStemmer();
snowballStemmer = nltk.stem.SnowballStemmer('english');
stopwordsLst = nltk.corpus.stopwords.words('english');
stopwordsSet = set(stopwordsLst);

#===============================================================================
# Get: stopwords set
#===============================================================================
def getStopwordsSet(): return stopwordsSet;

#===============================================================================
# Get: porter stemmer 
#===============================================================================
def getPorterStemmer(): return stemmer;
    
#===============================================================================
# stem (using porter stemmer)
#===============================================================================
def stemWithPorterStemmer(tok): return stemmer.stem(tok);

#===============================================================================
# stem (using snowball stemmer)
#===============================================================================
def stemWithSnowballStemmer(tok): return str(snowballStemmer.stem(tok));

#===============================================================================
# sentence boundary detection (doc -> sent)
#===============================================================================
def sentTokenize(docStr): return nltk.sent_tokenize(docStr); 

#===============================================================================
# tokenize (sent -> token)
#===============================================================================
def wordTokenize(sentStr, stemmingOption=False, rmStopwordsOption=False):
    tokenLst = nltk.word_tokenize(sentStr);
    if(stemmingOption): tokenLst = [stemWithSnowballStemmer(tok) for tok 
                                    in tokenLst];
    if(rmStopwordsOption): tokenLst = [tok for tok in tokenLst 
                                       if(tok not in getStopwordsSet())];        
    return tokenLst;

#===============================================================================
# sentence boundary detection + tokenize (revome stopwords,  stemming)
#===============================================================================
def preprocText(txt, stemmingOption, rmStopwordsOption): 
    return [wordTokenize(sent, stemmingOption, rmStopwordsOption) for sent 
            in sentTokenize(txt)];

#===============================================================================
# postag (get list of pos-tag)
#===============================================================================
def postag(tokenLst): return [pair[1] for pair in nltk.pos_tag(tokenLst)];

#===============================================================================
# postag (get list of pairs [word, tag])
#===============================================================================
def postagWordTagPair(tokenLst): return nltk.pos_tag(tokenLst);

#===============================================================================
# convert fine postag to coarse tag
#===============================================================================
def convertPostagFineToCoarse(tag):
    if(len(tag) > 2): return tag[0:2];
    else: return tag;
    
#===============================================================================
# postag (get list of COARSE postag)
#===============================================================================
def postagCoarseTag(tokenLst): 
    return [convertPostagFineToCoarse(tag) for tag 
            in postag(tokenLst)];

#===============================================================================
# get the NER chunks represented by trees
#===============================================================================
def nerChunk(tokenLst, postagLst=None):
    if(not postagLst): postagLst = postag(tokenLst);
    tagged = [(tokenLst[i], postagLst[i]) for i in range(len(tokenLst))];
    entities = nltk.chunk.ne_chunk(tagged);
    return entities;

def langIden(tokens, candLang=None, lower=True):
    ws = set([tok.lower() for tok in tokens]);
    langScores = {};
    if(candLang is None): candLang = stopwords.fileids();
    for lang in candLang: 
        langScores[lang] = len(ws.intersection(stopwords.words(lang)));
    return max(langScores, key=lambda x:langScores[x]);
        
if __name__ == '__main__':
#     txt = '''John Herbert Dillinger, Jr. (June 22, 1903 2013 July 22, 1934) was 
#     an American bank robber in the Depression-era United States. His gang robbed 
#     two dozen banks and four police stations. Dillinger escaped from jail twice. 
#     Dillinger was also charged with, but never convicted of, the murder of an 
#     East Chicago, Indiana, police officer during a shoot-out, Dillinger's only 
#     homicide charge.''';
#     docToks = preprocText(txt, stemmingOption=False, rmStopwordsOption=False);
#     for sentTok in docToks:
#         print sentTok;
#         print postag(sentTok);
#         print postagCoarseTag(sentTok); 
#         print langIden(sentTok, ['spanish', 'english']);
# #     sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good.";
# #     tokenLst = wordTokenize(sentence);
# #     entites = nerChunk(tokenLst);
# #     print entites.node
# 
#     txt = '''
#     That may result in more season ticket sales, which is good news for Swetoha. 
#     The team lost 6-foot-8 Elizabeth Cambage of Australia, who declined to return 
#     to the Shock following her rookie season. But they've gained forward Nicole 
#     Powell in a trade with the New York Liberty and Wiggins from Minnesota 
#     this season. (Diggins') social media experience is well-documented and 
#     becoming the first female athlete to sign with (rapper turned sports agent) 
#     Jay Z,'' Swetoha said. ''But basketball is first. That's why we drafted her.'' 
#           '''
#     print(preprocText(txt, False, False));        
    print getStopwordsSet();
    pass;
