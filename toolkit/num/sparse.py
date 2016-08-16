'''
Created on Jun 25, 2014

@author: xwang1
'''
from toolkit.num.arithmetic import ifZeroNum
from multiprocessing import Pool
from random import shuffle
import sys;
from toolkit.num.algebra import cloneVec, sizeMat
from time import time, clock

def toSparseVec(vec=None, keys=[], vals=[], dim=None):
    if(vec is not None):
        l = len(vec);
        d = {};
        for i in range(len(vec)):
            if(not ifZeroNum(vec[i])): d[i] = vec[i];
    else:
        l = dim;
        d = {};
        for i in range(len(keys)): 
            if(not ifZeroNum(vals[i])): 
                d[keys[i]] = d.get(keys[i], 0.0) + vals[i];        
    return (d, l);

def toDenseVec(sv):
    (d, l) = sv;
    return [d.get(k, 0.0) for k in range(l)];

def getSv(d, l): return (d, l);

def getSvDat(sv): return sv[0];

def getSvLen(sv): return sv[1];

def getSvVals(sv): return sv[0].values();

def getSvKeys(sv): return sv[0].keys();

def getSvElem(sv, idx): return getSvDat(sv).get(idx, 0.0);

def setSvElem(sv, idx, val): getSvDat(sv)[idx] = val;

def getSvL1Norm(sv): return sum([abs(x) for x in getSvVals(sv)]);

def getSvL2Norm(sv): return sum([x ** 2 for x in getSvVals(sv)]);

def cloneSv(sv):  
    (d, l) = sv;
    (d2, l2) = ({}, l);
    for k in d: 
        if(d[k] != 0):
            d2[k] = d[k];
    return (d2, l2);
    
def updateAddSv(sv1, sv2):
    if(getSvLen(sv1) != getSvLen(sv2)): 
        print("[sparse add vec vec]: inconsistent dimension");
        return None;
    d1 = getSvDat(sv1);
    d2 = getSvDat(sv2);
    for k in d2: d1[k] = d1.get(k, 0.0) + d2[k];
    return sv1;

def updateSubSv(sv1, sv2):
    if(getSvLen(sv1) != getSvLen(sv2)): 
        print("[sparse sub vec vec]: inconsistent dimension");
        return None;
    d1 = getSvDat(sv1);
    d2 = getSvDat(sv2);
    for k in d2: d1[k] = d1.get(k, 0.0) - d2[k];
    return sv1;

def updateMulNum(sv, a):
    (d, l) = sv;
    for k in d: d[k] = a * d[k];
    return sv;
    
def addSvSv(sv1, sv2):
    sv = cloneSv(sv1);
    return cloneSv(updateAddSv(sv, sv2));

def subSvSv(sv1, sv2):
    sv = cloneSv(sv1);
    return cloneSv(updateSubSv(sv, sv2));

def mulNumSv(a, sv):
    sv = cloneSv(sv);
    return cloneSv(updateMulNum(sv, a));

def getSmRow(sm, r): return sm[0].get(r, ({}, getSmRowLen(sm)));

def getSmSize(sm): return sm[1];

def getSmDat(sm): return sm[0];

def getSmRowIdxLst(sm): return sm[0].keys();

def getSmRowLen(sm): return sm[1][1];  # column number

def getSmColLen(sm): return sm[1][0];  # row number

def setSmElem(sm, rowIdx, colIdx, val):
    (mat, (rn, cn)) = sm;
    if(rowIdx not in mat): mat[rowIdx] = getSv({}, cn);
    setSvElem(mat[rowIdx], colIdx, val);
    return;

def setSmRow(sm, rowIdx, sv):
    getSmDat(sm)[rowIdx] = sv;
    return;

def getSmElem(sm, rowIdx, colIdx):
    return getSvElem(getSmRow(sm, rowIdx), colIdx);
    
def mulSvSm(sv, sm):
    if(getSvLen(sv) != getSmColLen(sm)):
        print("[sparse mul vec mat]: inconsistent dimension");
        return None;
    retSv = ({}, getSmRowLen(sm));
    dSv = getSvDat(sv);
    for k in dSv: updateAddSv(retSv, mulNumSv(dSv[k], getSmRow(sm, k)));
    return cloneSv(retSv);

def mulSmSv(sm, sv, idxSet=None):
    if(getSmRowLen(sm) != getSvLen(sv)):
        print("[sparse mul mat vec]: inconsistent dimension");
        return None;
    if(idxSet is None): idxSet = getSmRowIdxLst(sm);
    dRetSv = {};
    retSv = (dRetSv, getSmColLen(sm));
    for r in idxSet: 
        v = dotSvSv(sv, getSmRow(sm, r));
        if(v != 0.0): dRetSv[r] = v;
    return retSv;

def dotSvSv(sv1, sv2):
    if(getSvLen(sv1) != getSvLen(sv2)): 
        print("[sparse dot vec vec]: inconsistent dimension");
        return None;
    (d1, l1) = sv1;
    (d2, l2) = sv2;
    if(len(d1) > len(d2)): (d1, d2) = (d2, d1);
    return sum([d1[k] * d2[k] for k in d1 if(k in d2)]);

def transposeSm(sm):
    (mat, (rn, cn)) = sm;
    smTr = ({}, (cn, rn));
    for r in getSmRowIdxLst(sm):
        rSv = getSmRow(sm, r);
        for c in getSvKeys(rSv):
            setSmElem(smTr, c, r, getSmElem(sm, r, c));
    return smTr;

def cloneSm(sm):
    (mat, (rn, cn)) = sm;
    mat2 = {};
    for r in getSmRowIdxLst(sm): mat2[r] = cloneSv(getSmRow(sm, r));
    return (mat2, (rn, cn));

def _paraFunc1AddSmSm(args):
    (r, sv1, sv2) = args;
    return (r, addSvSv(sv1, sv2));
        
def addSmSm(sm1, sm2, procNum=1):
    if(getSmSize(sm1) != getSmSize(sm2)):
        print("[add sm sm]: inconsistent dimension");
        return None;
    rIdxSet = set(getSmRowIdxLst(sm1));
    for r in getSmRowIdxLst(sm2): rIdxSet.add(r);
    mat = {};
    if(procNum == 1):
        for r in rIdxSet:
            mat[r] = addSvSv(getSmRow(sm1, r), getSmRow(sm2, r));
    else:
        pool = Pool(processes=procNum);
        result = pool.map_async(func=_paraFunc1AddSmSm,
                                iterable=[(r, getSmRow(sm1, r),
                                           getSmRow(sm2, r)) for r in rIdxSet]);
        result.wait();
        rLst = result.get();
        for (r, sv) in rLst: mat[r] = sv;
    return (mat, getSmSize(sm1));

def _paraFunc1SubSmSm(args):
    (r, sv1, sv2) = args;
    return (r, subSvSv(sv1, sv2));

def subSmSm(sm1, sm2, procNum=1):
    if(getSmSize(sm1) != getSmSize(sm2)):
        print("[sub sm sm]: inconsistent dimension");
        return None;
    rIdxSet = set(getSmRowIdxLst(sm1));
    for r in getSmRowIdxLst(sm2): rIdxSet.add(r);
    mat = {};
    if(procNum == 1):
        for r in rIdxSet:
            mat[r] = subSvSv(getSmRow(sm1, r), getSmRow(sm2, r));
    else:
        pool = Pool(processes=procNum);
        result = pool.map_async(func=_paraFunc1SubSmSm,
                                iterable=[(r, getSmRow(sm1, r),
                                           getSmRow(sm2, r)) for r in rIdxSet]);
        result.wait();
        rLst = result.get();
        for (r, sv) in rLst: mat[r] = sv;
    return (mat, getSmSize(sm1));

def _paraFunc1MulSmSm(args):
    (r, sv, sm2) = args;
    return (r, mulSvSm(sv, sm2));

def mulSmSm(sm1, sm2, procNum=1):
    (mat1, (rn1, cn1)) = sm1;
    (mat2, (rn2, cn2)) = sm2;
    if(cn1 != rn2):
        print("[mul sm sm]: inconsistent dimension");
        return None;
    mat = {};
    if(procNum == 1):
        for r in getSmRowIdxLst(sm1): mat[r] = mulSvSm(getSmRow(sm1, r), sm2);
    else:
        pool = Pool(processes=procNum);
        result = pool.map_async(func=_paraFunc1MulSmSm,
                                iterable=[(r, getSmRow(sm1, r), sm2) for r in 
                                          getSmRowIdxLst(sm1)]);
        result.wait();
        rLst = result.get();
        for (r, sv) in rLst: mat[r] = sv;
    return (mat, (rn1, cn2));

def toSparseMat(mat, eps=0.0):
    (m, n) = sizeMat(mat);
    sm = ({}, (m, n));
    for i in range(m):
        for j in range(n):
            if(not ifZeroNum(mat[i][j], eps=eps)): 
                setSmElem(sm, i, j, mat[i][j]);
    return sm;

def getSmL1Norm(sm, procNum=1):
    if(procNum == 1):
        return sum([getSvL1Norm(getSmRow(sm, r)) for r in getSmRowIdxLst(sm)]);
    else:
        pool = Pool(processes=procNum);
        result = pool.map_async(func=getSvL1Norm, iterable=[getSmRow(sm, r) 
                                                for r in getSmRowIdxLst(sm)]);
        result.wait();
        rLst = result.get();
        return sum(rLst);

def getSmL2Norm(sm, procNum=1):
    if(procNum == 1):
        return sum([getSvL2Norm(getSmRow(sm, r)) 
                    for r in getSmRowIdxLst(sm)]);
    else:
        pool = Pool(processes=procNum);
        result = pool.map_async(func=getSvL2Norm, iterable=[getSmRow(sm, r)
                                                for r in getSmRowIdxLst(sm)]);
        result.wait();
        rLst = result.get();
        return sum(rLst);

if __name__ == '__main__':
#     sv1 = ({1:1.0, 2:2.0}, 5);
#     sv2 = ({3:1.0, 1:1.0}, 5);
#     keys = [1, 2, 4];
#     vals = [1.0, 3.0, 5.0];
#     dim = 5;
#     print toSparseVec(keys=keys, vals=vals, dim=dim);
#     
    sm = ({0:({1:2}, 5), 2:({4:3}, 5)}, (3, 5));
    print transposeSm(sm);
    pass
