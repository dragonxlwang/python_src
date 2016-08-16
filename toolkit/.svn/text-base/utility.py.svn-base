'''
Created on Feb 1, 2013

@author: xwang95
'''
import random;
import os;
import os.path;
import sys;
import shutil;
import math;
import re;

NOT_FOLD = True;
#===============================================================================
# Section: Topic Modeling
#===============================================================================
def getRankedIdxList(lst, reverseOption=False):
    return [idx for idx in sorted(range(len(lst)), key=lambda x: lst[x], reverse=reverseOption)];

def getRankedIdxMatrix(mtx, reverseOption=False):
    return [getRankedIdxList(lst, reverseOption) for lst in mtx];

def getDictRank(dict, key, reverse=False):
    rankToKey = {};
    keyToRank = {};
    for k in sorted(dict, key=key, reverse=reverse):
        r = len(rankToKey);
        rankToKey[r] = k;
        keyToRank[k] = r;
    return (rankToKey, keyToRank);

NOT_FOLD = True;    
#===============================================================================
# parse
#===============================================================================
def parseNumVal(s, defaultVal=None):
    s = s.strip();
    try:
        v = int(s);
    except:
        try:
            v = float(s);
        except:
            if(defaultVal is not None): v = defaultVal;
            else: v = float('nan');
    return v;

def parseYear(s):
    yearLst = [];
    cnt = 0;
    for i in range(len(s)):
        if('0' <= s[i] and s[i] <= '9'): cnt += 1;
        else: cnt = 0;
        if(cnt == 4): yearLst.append(parseNumVal(s[(i - 3):(i + 1)]));
    return yearLst;

NOT_FOLD = True;
#===============================================================================
# I/O
#===============================================================================
def readLines(num, reader):
    eof = False;
    lineLst = [reader.readline() for i in range(num)];
    if(not lineLst[num - 1]): eof = True;
    lineLst = [line.strip() for line in lineLst];
    return (lineLst, eof);

def readUntil(checkFunc, reader):
    eof = False;
    lineLst = [];
    while(True):
        lineLst.append(reader.readline());
        if(not lineLst[-1]):
            eof = True;
            break;
        if(checkFunc(lineLst[-1].strip())): break;
    lineLst = [line.strip() for line in lineLst];
    if(eof): lineLst = lineLst[:-1];  # no end of file line
    return (lineLst, eof);

def readChunk(reader): return readUntil(lambda x: (x == ""), reader);

def readMatrix(reader, row=None):
    lines = [];
    if(row is not None): (lines, eof) = readLines(row, reader);
    else: (lines, eof) = readUntil(lambda x: (x == ''), reader);
    return ([[parseNumVal(s) for s in line.split()] for line in lines], eof);

def writeMatrix(mat, writer):
    for vec in mat: writer.write(' '.join([str(x) for x in vec]) + '\n');
    return;

def readVector(reader):
    (lines, eof) = readLines(1, reader);
    if(not eof): return ([parseNumVal(s) for s in lines[0].split()], eof);
    else: return([], eof);

def printProgressBar(prog, step=0.04, addStr=""):
    prct = int(math.ceil(100 * prog));
    totlBlk = int(math.ceil(1.0 / step));
    progBlk = max(min(totlBlk, int(math.ceil(prog / step))) - 1, 0);
    futrBlk = max(totlBlk - progBlk - 1, 0);
    bar = '[' + ''.join(['=' for i in range(progBlk)]) + '>' + ''.join([' ' for i in range(futrBlk)]) + ']' + '({0})'.format(addStr);
    s = '\r[{0:>3}%]: {1}'.format(prct, bar);
    sys.stdout.write(s);
    sys.stdout.flush();
    return s;

def rFillSpaces(s, lineWidth=300):
    return str(s) + ''.join([' ' for i in range(lineWidth - len(s))]); 

def readDictFile(filePath, ifKeyNum=False, ifValNum=False):
    t = {};
    reader = open(filePath, 'r');
    lineLSt = reader.readlines();
    lineLSt = [ln.strip() for ln in lineLSt if(ln.strip())];
    reader.close();
    for ln in lineLSt: 
        [k, v] = ln.split('<=:|:=>');
        if(ifKeyNum): k = parseNumVal(k);
        if(ifValNum): v = parseNumVal(v);
        t[k] = v;
    return t;

def writeDictFile(filePath, t, keylst=None):
    writer = open(filePath, 'w');
    if(keylst==None):
        for (k, v) in t.items(): 
            writer.write('{0}<=:|:=>{1}\n'.format(k, v));
    else:
        for k in keylst:
            v = t[k];
            writer.write('{0}<=:|:=>{1}\n'.format(k, v));
    writer.close();

def loadObjFromFile(filePath):
    file = open(filePath);
    obj = eval(file.readline());
    file.close();
    return obj;

def writeObjToFile(filePath, obj):
    file = open(filePath, "w");
    file.write(str(obj));
    file.close();
    return;

def splitCrossValidation(lnLst, foldNum, outputDir):
    mkDir(outputDir);
    writersTrain = [open(os.path.join(outputDir, "train_{0}".format(j)), "w") 
                    for j in range(foldNum)];
    writersTest = [open(os.path.join(outputDir, "test_{0}".format(j)), "w") 
                    for j in range(foldNum)];
    for i in range(len(lnLst)):
        for j in range(foldNum):
            if(i % foldNum == j): writersTest[j].write(lnLst[i] + "\n");
            else: writersTrain[j].write(lnLst[i] + "\n");
    for j in range(foldNum): 
        writersTrain[j].close();
        writersTest[j].close();
    return;
        
NOT_FOLD = True;
#===============================================================================
# string
#===============================================================================
def rmLeadingStr(s, srm):
    if(s.startswith(srm)): return s[len(srm):];
    return s;

def rmTrailingStr(s, srm):
    if(s.endswith(srm)): return s[:-len(srm)];

NOT_FOLD = True;
#===============================================================================
# file management
#===============================================================================
def removePath(path):
    if(not os.path.exists(path)): print('[remove_path@utility]: path not exist: {0}, Doing Nothing'.format(path));
    if(os.path.isdir(path)):
        print('[remove_path@utility]: removing directory: {0}'.format(path));
        shutil.rmtree(path);
    if(os.path.isfile(path)):
        print('removing files: {0}'.format(path));
        os.remove(path);
    return;

def mkDir(path):
    if(os.path.isdir(path)): print('[make_dir@utility]: path {0} already exists'
                                   ' as DIRECTORY! Doing Nothing'.format(path));
    elif(os.path.isfile(path)): print('[make_dir@utility]: path {0} already '
                                'exist as FILE! Doing Nothing'.format(path));
    else:
        print('[make_dir@utility]: make directory {0}'.format(path));
        os.makedirs(path);
    return;

NOT_FOLD = True;
#===============================================================================
# computing 
#===============================================================================
def factorial(n): 
    if(n == 0): return 1;
    else: return reduce(lambda x, y:x * y, range(1, int(n) + 1));

def getVecNorm(vec, order): return math.pow(sum([math.pow(x, order) for x in vec]), 1.0 / order);

def normalizeVector(vec, order=None):
    if(order is not None): norm = getVecNorm(vec, order); 
    else: norm = float(sum(vec));
    return [x / norm for x in vec];    

def getDistExpectation(dist): return sum([k * dist[k] for k in dist]);

def getDistVariation(dist, expt=None):
    if(expt is None): expt = getDistExpectation(dist);
    return sum([((k - expt) ** 2) * dist[k] for k in dist]);

def getDistStd(dist, expt=None): return math.sqrt(getDistVariation(dist, expt));

def getMatrixVecMultiply(m, v): return [sum([vec[i] * v[i] for i in range(len(v))]) for vec in m];

def getTransposeSquareMatrix(m): return [[m[j][i] for j in range(len(m))] for i in range(len(m))];

def getVecSubstract(v1, v2): return [v1[i] - v2[i] for i in range(len(v1))];

def getFreqTable(t):
    freqTable = {};
    for (k, v) in t.items(): freqTable[v] = freqTable.get(v, 0) + 1;
    return freqTable; 

def printFreqTable(ft):
    for k in sorted(ft): print('{0}\t{1}'.format(k, ft[k]));
    return; 

def regexFindCreditCardNumber(str):
    '''match credit card number, including  optionally delimited 15-digit 
    American Express numbers and 16-digit VISA, MasterCard, Discover, and Japan
    Credit Bureau card numbers. China UnionPay and Maestro are not included.
    
    reference: http://www.richardsramblings.com/regex/credit-card-numbers/
               http://www.regular-expressions.info/creditcard.html
    '''
    patt = re.compile(r'\b(?<!\-|\.)(?:(?:(?:4\d|5[1-5]|65)(\d\d)(?!\1{3})|'
                      r'35(?:2[89]|[3-8]\d)|'
                      r'6(?:011|4[4-9]\d|22(?:1(?!1\d|2[1-5])|[2-8]'
                      r'|9(?=1\d|2[1-5]))))([\ \-]?)(?<!\d\ \d{4}\ )'
                      r'(?!(\d)\3{3})(\d{4})\2(?!\4|(\d)\5{3}|'
                      r'1234|2345|3456|5678|7890)(\d{4})(?!\ \d{4}\ \d)\2(?!\6|'
                      r'(\d)\7{3}|1234|3456)|3[47]\d{2}([\ \-]?)(?<!\d\ \d{4}\ )'
                      r'(?!(\d)\9{5}|123456|234567|345678)\d{6}(?!\ \d{5}\ \d)'
                      r'\8(?!(\d)\10{4}|12345|56789|67890)\d)\d{4}(?!\-)'
                      r'(?!\.\d)\b');
    for m in patt.finditer(str):
        print (m.span(), m.group());
    return;

def regexFindPhoneNumber(str):
    patt = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}'
                      r'[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})');
    for m in patt.finditer(str):
        print (m.span(), m.group());
    return;

def ifEqualLst(lst1, lst2):
    if(len(lst1) != len(lst2)): return False;
    for i in range(len(lst1)):
        if(lst1[i] != lst2[i]): return False;
    return True;

if __name__ == '__main__':
    regexFindCreditCardNumber('credit number is: 6011-0160-1101-6011');
    regexFindCreditCardNumber('credit number is: 6011016011016011');
    regexFindCreditCardNumber('several credit numbers: 3711-078176-01234, '
                              'or 4123 5123 6123 7123, and 5123412361237123');
    regexFindCreditCardNumber('4123 5123 6123 7123');
    regexFindPhoneNumber('my phone number: 217-418-4097');
    regexFindPhoneNumber('my phone number: 2174184097');
    regexFindPhoneNumber('my phone number: (000)000.0000');

