'''
Created on Mar 15, 2015

@author: xwang95
'''
from os import path
import os, sys
import string
from deep_nlp import word_index
from deep_nlp.word_index import WordIndexer

sent_seg = """!,-.:;?""";

class NipsDataProcess():
    root_dir = '/home/xwang95/data/nips/nipstxt';
    proc_txt_dir = '/home/xwang95/data/nips/processed/txt';
    proc_num_dir = '/home/xwang95/data/nips/processed/num';
    def get_yeardir(self, yr):
        if(yr < 0 or yr > 12):
            print('error: nips data year ranging from 00 to 12');
            return;
        yr_str = str(yr);
        if(len(yr_str) < 2): yr_str = '0' + yr_str;
        year_dir = path.join(self.root_dir, 'nips' + yr_str);
        return year_dir;
    
    def get_yeardocs(self, yr):  # yr = 0 ... 12
        year_dir = self.get_yeardir(yr);
        for fn in os.listdir(year_dir):
            fp = path.join(year_dir, fn);
            with open(fp) as fin:
                line_lst = fin.readlines();
                st = 0;
                while(st < len(line_lst)):  # find abstract start
                    if(line_lst[st].strip().lower() == 'abstract'): break;
                    st += 1;
                end = st + 1;
                while(end < len(line_lst)):  # find abstract end
                    if(len(line_lst[end].strip().split()) == 1 or
                       (len(line_lst[end].strip().split()) == 2 and
                        line_lst[end].strip().split()[0].startswith('1')) or
                       (len(line_lst[end].strip().split()) == 2 and
                        line_lst[end].strip().split()[0].startswith('I')) or
                       end > st + 100): break;
                    end += 1;
                abs_lnlst = line_lst[st + 1:end];
                abs_sentlst = self._seg_to_sentence_lst(abs_lnlst);
                filefullname = '_'.join(fp.split('/')[-2:]);
                print('{0} processing ...'.format(filefullname));
                with open(path.join(self.proc_txt_dir, 
                                    filefullname), 'w') as fout:
                    fout.write('\n'.join([' '.join(sent) 
                                          for sent in abs_sentlst]));                    
        return;
    
    def index(self):
        wi = WordIndexer(wordCountMode=True);
        for fn in os.listdir(self.proc_txt_dir):
            fp = path.join(self.proc_txt_dir, fn);
            with open(fp) as fin:
                sentlst = [x.split() for x in fin.readlines()];
                for sent in sentlst:
                    wi.getTokenLstIndexLst(sent, autoInsert=True,
                                           removeNone=True);
        wi.dumpWordCounts(path.join(self.proc_num_dir, 
                                    'nips.wordcount.py.txt'));
        wi.cutoffLowFreqWords(3);   # cut off infreq words
        data = {};
        for fn in os.listdir(self.proc_txt_dir):
            fp = path.join(self.proc_txt_dir, fn);
            with open(fp) as fin:
                sentlst = [x.split() for x in fin.readlines()];
                sentlst = [wi.getTokenLstIndexBow(sent, autoInsert=False,
                                                  removeNone=True)
                           for sent in sentlst];
                # remove sentence with freq word < 5
                sentlst = [sent for sent in sentlst if(len(sent) >= 5)];
                data[fn] = sentlst;
        fp = path.join(self.proc_num_dir, 'nips.data.py.num');
        with open(fp, 'w') as fout:
            fout.write(str(data));
        fp = path.join(self.proc_num_dir, 'nips.index.py.txt');
        wi.dumpWordIndexer(fp);            
        print('finished indexing ...');
        return;
        
    def _seg_to_sentence_lst(self, lnlst):
        abs = '-';
        for i in range(len(lnlst)):
            if(abs[-1] == '-'):
                abs = abs[0:-1] + lnlst[i].strip();
            else:
                abs = abs + ' ' + lnlst[i].strip();
                string.split
        toklst = abs.split();
        st = 0;
        end = st + 1;
        sentlst = [];
        while(True):
            if((toklst[end - 1][-1] in sent_seg and end - st >= 3) or
                end == len(toklst)):
                sentlst.append([_rm_lt_punc_chars(w).lower() 
                                for w in toklst[st:end]]);
                st = end;
            end += 1;
            if(end > len(toklst)): break;
#         # filter out sentence with length < 5
#         sentlst = [sent for sent in sentlst if(len(sent) >= 5)];
        # select at most first 30 sentences
        if(len(sentlst) > 30): sentlst = sentlst[0:30];
        return sentlst;

def _rm_lt_punc_chars(w):  # remove leading and trailing punctuation chars
    st = 0;
    end = len(w);
    while(st < end and w[st] in string.punctuation): st += 1;
    while(end > st and w[end - 1] in string.punctuation): end -= 1;
    return w[st:end] if(st < end) else '';  

def nips_proc():
    ndp = NipsDataProcess();
    for i in range(13):
        ndp.get_yeardocs(i);
    ndp.index();
    return;

if __name__ == '__main__':
    nips_proc();
    
