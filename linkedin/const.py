'''
Created on Jun 3, 2014

@author: xwang1
'''
import os.path;
import re;
from os.path import expanduser
import sys

homePath = expanduser("~");
dataDir = os.path.join(homePath, "data");
cleanedTextFilePath = os.path.join(homePath, "data", "cleaned_text_5_12.txt");
cleanedTextEngFilePath = os.path.join(homePath, "data",
                                      "data/cleaned_text_eng.txt");
workingFilePath = os.path.join(homePath, "data",
                               "cleaned_text_eng.txt.filtered_by_tag");
initialPhraseFilePath = "/home/xwang1/data/sentiment_stuff_recent/cleaned_ngrams.txt";
#------------------------------------------------------------------------------ 
stemFilePath = os.path.join(dataDir, "stem.table");
vocabularyFilePath = os.path.join(dataDir, "stem.filtered.list");
#------------------------------------------------------------------------------ 
vocabularySize = 15000;
globalTxtFilePath = os.path.join(dataDir,
                                 "cleaned_text_eng.txt.filtered_by_tag.global");
globalVocTableFilePath = os.path.join(dataDir, "global.voc.table");
globalVocFilteredLstPath = os.path.join(dataDir, "global.voc.lst");
globalFvFilePath = os.path.join(dataDir, "global.fv");
globalXSmFilePath = os.path.join(homePath, "data", "global.xSm");
globalInitBTrSmFilePath = os.path.join(homePath, "data", "global.bTrSm.init");
global400BTrSmFilePath = os.path.join(homePath, "data", "global", "400.bTrSm");
global400BasisTxtFilePath = os.path.join(homePath, "data", "global",
                                         "400.basis.txt");
#------------------------------------------------------------------------------ 
setting1 = (0.5, 0.1, 0.001, 0.001);
globalSetting1Dir = os.path.join(dataDir, "global-" + 
                                 "-".join([str(x) for x in setting1]));
global1000BTrSmSetting1FilePath = os.path.join(globalSetting1Dir, "1000.bTrSm"); 
global1000BasisTxtSetting1FilePath = os.path.join(globalSetting1Dir,
                                                  "1000.basis.txt");
global300BTrSmSetting1FilePath = os.path.join(globalSetting1Dir, "300.bTrSm"); 
global300BasisTxtSetting1FilePath = os.path.join(globalSetting1Dir,
                                                  "300.basis.txt");

#------------------------------------------------------------------------------ 
setting2 = (0.2, 0.2, 0.001, 0.001);
globalSetting2Dir = os.path.join(dataDir, "global-" + 
                                 "-".join([str(x) for x in setting2]));
global300BTrSmSetting2FilePath = os.path.join(globalSetting2Dir, "300.bTrSm"); 
global300BasisTxtSetting2FilePath = os.path.join(globalSetting2Dir,
                                                  "300.basis.txt");
global270BTrSmSetting2FilePath = os.path.join(globalSetting2Dir, "270.bTrSm"); 
global270BasisTxtSetting2FilePath = os.path.join(globalSetting2Dir,
                                                  "270.basis.txt");
                                                  
#------------------------------------------------------------------------------ 
setting3 = (0.2, 0.02, 0.001, 0.001, None, None, None);
globalSetting3Dir = os.path.join(dataDir, "global-" + 
                                 "-".join([str(x) for x in setting3]));
#------------------------------------------------------------------------------ 
setting4 = (0.2, 0.2, 0.001, 0.001, "reverse_idf", None, None);
globalSetting4Dir = os.path.join(dataDir, "global-" + 
                                 "-".join([str(x) for x in setting4]));
#------------------------------------------------------------------------------ 
setting5 = (0.2, 0.2, 0.001, 0.001, "concave_log", None, None);
globalSetting5Dir = os.path.join(dataDir, "global-" + 
                                 "-".join([str(x) for x in setting5]));
#------------------------------------------------------------------------------ 
setting6 = (0.5, 0.1, 0.001, 0.001, None, None, None);
globalSetting6Dir = os.path.join(dataDir, "global-" + 
                                 "-".join([str(x) for x in setting6]));                                  
if __name__ == '__main__':
    pass
