'''
Created on Mar 13, 2013

@author: xwang95
'''
import time;
import random;
import types;
import math;
import arithmetic;
from arithmetic import sgn, ifZeroNum, _eps;
import sys;

# _eps = 1e-8;

#===============================================================================
# Matrix/Vector Operation
#===============================================================================
def addVecVec(v1, v2):
    if(len(v1) != len(v2)):
        print("[add vec vec]: not consistent dimension");
        return None;
    return [v1[i] + v2[i] for i in range(len(v1))];

def addVecLst(v1, v2, *args):
    x = addVecVec(v1, v2);
    for arg in args: x = addVecVec(x, arg);
    return x;

def subVecVec(v1, v2):
    if(len(v1) != len(v2)):
        print("[sub vec vec]: not consistent dimension");
        return None;
    return [v1[i] - v2[i] for i in range(len(v1))];

def mulNumVec(a, vec): return [a * vec[i] for i in range(len(vec))];

def divNumVec(a, vec): return [vec[i] / a for i in range(len(vec))];

def getVecNorm(vec, order=2): 
    return math.pow(sum([math.pow(x, order) for x in vec]), 1.0 / order);

def getVecInfNorm(vec): return max([abs(x) for x in vec]);

def getVecZeroNorm(vec, eps=_eps): 
    return sum([1.0 for x in vec if(not ifZeroNum(x, eps))]);

def dotVecVec(v1, v2): return sum([v1[i] * v2[i] for i in range(len(v1))]);

def mulNumMat(a, mat): return [[a * x for x in vec] for vec in mat];

def divNumMat(a, mat): return [[x / a for x in vec] for vec in mat];
 
def mulMatVec(mat, vec): return [dotVecVec(vec, v) for v in mat];

def mulVecMat(vec, mat): 
    return [sum([vec[i] * mat[i][j] for i in range(len(mat))]) 
            for j in range(len(mat[0]))];

def mulMatMat(mat1, mat2):
    (m1, n1) = sizeMat(mat1);
    (m2, n2) = sizeMat(mat2);
    if(n1 != m2):
        print('[mul mat mat]: not consistent dimension, ' \
              'n1={0}, m2={1}'.format(n1, m2));
        return None;
    return [[sum([mat1[i][k] * mat2[k][j] for k in range(n1)]) 
             for j in range(n2)] 
            for i in range(m1)];
# def mulMatMat2(mat1, mat2):
#    (m1, n1) = sizeMat(mat1);
#    (m2, n2) = sizeMat(mat2);
#    if(n1 != m2):
#        print("[mul mat mat]: not consistent dimension");
#        return None;
#    mat = zeroes(n1, n1);
#    for i in range(m1):
#        for j in range(n2):
#            for k in range(n1):
#                mat[i][j] += mat1[i][k] * mat2[k][j];
#    return mat;

def mulMatLst(mat1, mat2, *args):
    x = mulMatMat(mat1, mat2);
    for arg in args: x = mulMatMat(x, arg);
    return x;

def subMatMat(mat1, mat2):
    (m1, n1) = sizeMat(mat1);
    (m2, n2) = sizeMat(mat2);
    if(not (m1 == m2 and n1 == n2)):
        print("[sub mat mat]: not consistent dimension");
        return None;
    return [[mat1[i][j] - mat2[i][j] for j in range(n1)] for i in range(m1)];

def addMatMat(mat1, mat2):
    (m1, n1) = sizeMat(mat1);
    (m2, n2) = sizeMat(mat2);
    if(not (m1 == m2 and n1 == n2)):
        print("[mul mat mat]: not consistent dimension");
        return None;
    return [[mat1[i][j] + mat2[i][j] for j in range(n1)] for i in range(m1)];

def transposeMat(mat):
    (m, n) = sizeMat(mat);
    return [[mat[i][j] for i in range(m)] for j in range(n)];

def getMatCol(mat, c, beg=0, end=None):
    if(end is None): end = len(mat); 
    return [mat[i][c] for i in range(beg, end)];

def getMatRow(mat, r, beg=0, end=None):
    if(end is None): end = len(mat[0]);
    return [mat[r][i] for i in range(beg, end)];

def setMatCol(mat, c, vec, beg=0, end=None):
    if(end is None): end = len(mat);
    if(len(vec) != end - beg):
        print("[set mat col]: not consistent dimension");
        return None;
    for i in range(beg, end): mat[i][c] = vec[i - beg];
    return;

def setMatRow(mat, r, vec, beg=0, end=None):
    if(end is None): end = len(mat[0]);
    if(len(vec) != end - beg):
        print("[set mat row]: not consistent dimension");
        return None; 
    for i in range(beg, end): mat[r][i] = vec[i - beg];
    return;

def cloneMat(mat): return [[float(x) for x in vec] for vec in mat];

def cloneVec(vec): return [float(x) for x in vec];

def getSubMat(mat, rowIdx=None, colIdx=None):
    (m, n) = sizeMat(mat);
    if(rowIdx is None): rowIdx = range(m);
    if(colIdx is None): colIdx = range(n);
    return [[mat[i][j] for j in colIdx] for i in rowIdx];

def getSubVec(vec, idx): return [vec[i] for i in idx];
    
def cbind(*args, **kwargs):
    args = [arg for arg in args if(arg is not None)];
    if(len(args) == 0): return None;
    m = len(args[0]);
    for arg in args:
        if(len(arg) != m):
            print('[cbind]: not consistent dimension');
            return None;
    mat = [[] for i in range(m)];
    for arg in args:
        if(ifMatNotVec(arg)):
            for i in range(m): mat[i].extend(arg[i]);
        else:
            for i in range(m): mat[i].append(arg[i]);
    return mat;

def rbind(*args, **kwargs):
    args = [arg for arg in args if(arg is not None)];
    if(len(args) == 0): return None;
    if(ifMatNotVec(args[0])): m = len(args[0][0]);
    else: m = len(args[0]);
    for arg in args:
        if(ifMatNotVec(arg)):
            if(len(arg[0]) != m):
                print('[cbind]: not consistent dimension');
                return None;
        else:
            if(len(arg) != m):
                print('[cbind]: not consistent dimension');
                return None;
    mat = [];
    for arg in args:
        if(ifMatNotVec(arg)): mat.extend(cloneMat(arg));
        else: mat.append(cloneVec(arg));
    return mat;

def catVecLst(*args):
    args = [arg for arg in args if(arg is not None)];
    if(len(args) == 0): return None;
    vec = [];
    for arg in args:
        if(ifVecNotNum(arg)): vec.extend(arg);
        else: vec.append(arg);
    return vec;

def minusVec(vec): return mulNumVec(-1.0, vec);

def minusMat(mat): return mulNumMat(-1.0, mat);

def mulPermMatVec(idx, vec): return [vec[i] for i in idx];

def mulPermMatPermMat(idx1, idx2): return [idx2[x] for x in idx1];

def mulDiagMatVec(dVec, vec):
    if(len(dVec) != len(vec)):
        print('[mul diagmat vec]: not consistent dimension');
        return None;
    return [dVec[i] * vec[i] for i in range(len(dVec))];

def mulInvDiagMatVec(dVec, vec):
    if(len(dVec) != len(vec)):
        print('[mul inv diagmat vec]: not consistent dimension');
        return None;
    return [vec[i] / dVec[i] for i in range(len(dVec))];

def mulDiagMatMat(dVec, mat):
    (m, n) = sizeMat(mat);
    if(m != len(dVec)):
        print('[mul diagmat mat]: not consistent dimension');
        return None;
    return [[dVec[i] * mat[i][j] for j in range(n)] for i in range(m)];

def mulMatDiagMat(mat, dVec):
    (m, n) = sizeMat(mat);
    if(n != len(dVec)):
        print('[mul mat diagmat]: not consistent dimension');
        return None;
    return [[mat[i][j] * dVec[j] for j in range(n)] for i in range(m)];

NOT_FOLD = True;
#===============================================================================
# Special Matrix/Vector
#===============================================================================
def randomMat(m, n, lb=0.0, ub=1.0): 
    return [[lb + random.random() * (ub - lb) for j in range(n)] for i in range(m)];

def randomSymmetricMat(m):
    mat = randomMat(m, m);
    return addMatMat(mat, transposeMat(mat));

def randomPosDefMat(m):
    lMat = getLowerTriangularMat(randomSymmetricMat(m));
    return mulMatMat(lMat, transposeMat(lMat));

def randomVec(m, lb=0.0, up=1.0): 
    return randomMat(1, m, lb, up)[0];

def zeroes(m, n=None): return ones(m, n, val=0.0);
#     if(n is not None): return [[0.0 for j in range(n)] for i in range(m)];
#     else: return [0.0 for i in range(m)];

def ones(m, n=None, val=1.0):
    if(n is not None): return [[val for j in range(n)] for i in range(m)];
    else: return [val for i in range(m)];
    
def eye(m, n=None, val=1.0):
    if(n is None): n = m;
    mat = zeroes(m, n);
    for i in range(min(n, m)): mat[i][i] = val;
    return mat;

def getLowerTriangularMat(mat, reduced=False):
    (m, n) = sizeMat(mat);
    if(not reduced): x = zeroes(m, n);
    else: x = zeroes(m, min(m, n));
    for i in range(m):
        for j in range(0, min(i + 1, n)):
            x[i][j] = mat[i][j];
    return x;

def getUpperTriangularMat(mat, reduced=False):
    (m, n) = sizeMat(mat);
    if(not reduced): x = zeroes(m, n);
    else: x = zeroes(min(m, n), n);
    for i in range(m):
        for j in range(min(i, n), n):
            x[i][j] = mat[i][j];
    return x;

def getLowerHessMat(mat):
    if(not ifSquareMat(mat)):
        print('[getLowerHessMat]: not a square matrix');
        return None;
    n = sizeSquareMat(mat);
    x = zeroes(n, n);
    for i in range(n):
        for j in range(0, min(i + 2, n)):
            x[i][j] = mat[i][j];
    return x;

def getUpperHessMat(mat):
    if(not ifSquareMat(mat)):
        print('[getUpperHessMat]: not a square matrix');
        return None;
    n = sizeSquareMat(mat);
    x = zeroes(n, n);
    for i in range(n):
        for j in range(min(max(i - 1, 0), n), n):
            x[i][j] = mat[i][j];
    return x;

def stdBasis(dim, k):
    '''k-th Euclidean basis of dim dimension space'''
    if(k >= dim):
        print('[standard basis]: dimension {0} <= k {1}'.format(dim, k));
        return None;
    vec = [0.0 for i in range(dim)];
    vec[k] = 1.0;
    return vec;

def diagMat(vec):
    mat = zeroes(len(vec), len(vec));
    for i in range(len(vec)): mat[i][i] = vec[i];
    return mat;

def getNormalizedVec(vec):
    l = math.sqrt(dotVecVec(vec, vec));
    if(abs(l) <= _eps): return zeroes(len(vec));
    return [x / l for x in vec];
    
NOT_FOLD = True;
#===============================================================================
# Matrix Utility
#===============================================================================
def _numToString(x, prec=(15, 5), shortZero=False, eps=_eps, decor=''):
    if(shortZero): x = 0 if(ifZeroNum(x, eps=eps)) else x;
    if(x != 0): return ("{0:" + str(int(prec[0])) + "." + str(int(prec[1])) + decor + "}").format(float(x));  # exact 0
#    if(abs(x) > _eps): return ("{0:" + str(int(prec[0])) + "." + str(int(prec[1])) + "f}").format(x);
    else: return ('{0:' + str(int(prec[0])) + '}').format(0);

def numToString(x, prec=(15, 5), shortZero=False, decor=''): 
    return _numToString(x, prec, shortZero, decor=decor);

def vecToString(vec, prec=(15, 5), shortZero=False, decor='', abbrev=(9, 1)):
    if(abbrev is not None):
        (f, l) = abbrev;
        if(len(vec) > f + l):
            fv = vec[:f];
            lv = vec[-l:]; 
            return ('   [' + 
                    ' '.join([numToString(x, prec, shortZero, decor) 
                              for x in fv]) + 
                    ' ' + ('.' * prec[0]) + ' ' + 
                    ' '.join([numToString(x, prec, shortZero, decor) 
                              for x in lv]) + 
                    ']');
    return "   [" + " ".join([numToString(x, prec, shortZero, decor) 
                              for x in vec]) + "]";

def matToString(mat, prec=(15, 5), shortZero=False, decor='',
                abbrev=(9, 1, 9, 1)):
    if(abbrev is not None):
        (rf, rl, cf, cl) = abbrev;
        abbrev = (cf, cl);
        (m, n) = sizeMat(mat);
        if(m > rf + rl):
            fm = mat[:rf];
            lm = mat[-rl:];
            return ('\n'.join([vecToString(vec, prec, shortZero, decor, abbrev) 
                               for vec in fm]) + 
                    '\n' + '    ' + ('.' * prec[0]) + 
                    ' ' + ('.' * prec[0]) + '\n' + 
                    '\n'.join([vecToString(vec, prec, shortZero, decor, abbrev) 
                               for vec in lm]));
    return '\n'.join([vecToString(vec, prec, shortZero, decor, abbrev) 
                      for vec in mat]);

def printMat(mat, strMatName=None, prec=(15, 5), shortZero=False, decor='',
             abbrev=(9, 1, 9, 1)):
    if(mat is None): 
        if(strMatName is None): strMatName = 'm';
        print("{0} = ".format(strMatName));
        print("   NONE");
        return;
    if(getType(mat) == 'mat'):
        if(strMatName is None): strMatName = 'm';
        print("{0} = ".format(strMatName));
        print(matToString(mat, prec, shortZero, decor, abbrev));
    elif(getType(mat) == 'vec'):
        if(strMatName is None): strMatName = 'v';
        print("{0} = ".format(strMatName));
        print(vecToString(mat, prec, shortZero, decor,
                          abbrev[-2:] if abbrev is not None else abbrev));
    elif(getType(mat) == 'num'):
        if(strMatName is None): strMatName = 'x';
        print("{0} = {1}".format(strMatName, _numToString(mat, prec, shortZero, decor=decor)))
    return;

def sizeMat(mat): return (len(mat), len(mat[0]));

def sizeSquareMat(mat): return len(mat);

def sizeVec(vec): return len(vec);

def size(obj):
    try:
        isMatrix = True;
        (m, n) = sizeMat(obj);
    except:
        isMatrix = False;
    if(isMatrix): return sizeMat(obj);
    else: return sizeVec(obj);

def ifLowerTriangularMat(mat, eps=_eps):
    (m, n) = sizeMat(mat);
    for i in range(min(m, n)):
        for j in range(i + 1, n):
            if(not ifZeroNum(mat[i][j], eps=eps)): return False;
    return True;

def ifUpperTriangularMat(mat, eps=_eps):
    (m, n) = sizeMat(mat);
    for i in range(m):
        for j in range(0, min(i, n)):
            if(not ifZeroNum(mat[i][j], eps=eps)): return False;
    return True;    

def ifTriangularMat(mat, eps=_eps): return (ifUpperTriangularMat(mat, eps=eps) or ifLowerTriangularMat(mat, eps=eps));

def ifLowerHessMat(mat, eps=_eps):
    if(not ifSquareMat(mat)): return False;
    (m, n) = sizeMat(mat);
    for i in range(min(m, n)):
        for j in range(i + 2, n):
            if(not ifZeroNum(mat[i][j], eps=eps)): return False;
    return True;

def ifUpperHessMat(mat, eps=_eps):
    if(not ifSquareMat(mat)): return False;
    (m, n) = sizeMat(mat);
    for i in range(m):
        for j in range(0, min(i - 1, n)):
            if(not ifZeroNum(mat[i][j], eps=eps)): return False;
    return True;

def ifHessMat(mat, eps=_eps):
    '''In linear algebra, a Hessenberg matrix is a special kind of square matrix, 
    one that is "almost" triangular. To be exact, an upper Hessenberg matrix has 
    zero entries below the first subdiagonal, and a lower Hessenberg matrix has 
    zero entries above the first superdiagonal. They are named after Karl Hessenberg.'''
    return (ifUpperHessMat(mat, eps=eps) or ifLowerHessMat(mat, eps=eps));

def ifTridiagonalMat(mat, eps=_eps):
    '''In linear algebra, a tridiagonal matrix is a matrix that has 
    nonzero elements only on the main diagonal, the first diagonal 
    below this, and the first diagonal above the main diagonal.'''
    return (ifUpperHessMat(mat, eps=eps) and ifLowerHessMat(mat, eps=eps));

def ifSymmetryMat(mat, eps=_eps):
    if(not ifSquareMat(mat)): return False;
    n = sizeSquareMat(mat);
    for i in range(n):
        for j in range(i + 1, n): 
            if(not ifZeroNum(mat[i][j] - mat[j][i], eps=eps)): return False;
    return True;

def ifSquareMat(mat):
    (m, n) = sizeMat(mat);
    return (m == n);

def ifEqualMat(mat1, mat2, eps=_eps): return ifZeroMat(subMatMat(mat1, mat2), eps=eps);

def ifEyeMat(mat, eps=_eps): return ifEqualMat(eye(*sizeMat(mat)), mat, eps=eps);

def ifUnitaryMat(mat, eps=_eps):
    if(not ifSquareMat(mat)): return False;
    return ifEyeMat(mulMatMat(transposeMat(mat), mat), eps=eps);
    
def ifSingularMat(mat=None, lu=None):
    ''' test if matrix is singular using LU decomposition '''
    if(((mat is not None) and (not ifSquareMat(mat))) or ((lu is not None) and (not ifSquareMat(lu[0])))): 
        print('[ifSingularMat]: not a square matrix');
        return None;
    if(lu is None): lu = _luDecomp(mat);
    return _luIfFullRank(lu);

def ifPositiveDefinite(mat, ifCheckSymmetry=True):
    (flag, lMat) = _choleskyDecomp(mat=mat, checkSymmetry=ifCheckSymmetry);
    return (flag == 0);

def ifMatNotVec(obj):
    '''
    [] is a vec, [[], [], ..., []] is a matrix
    '''
    if(len(obj) > 0 and isinstance(obj[0], list)): return True;
    return False;
#     try:
#         m = obj[0][0];
#         return True;
#     except:
#         return False;

def ifVecNotNum(obj):
    if(isinstance(obj, list)): return True;
    return False;

def getType(obj):
    if(not isinstance(obj, list)): return 'num';
    if(not isinstance(obj[0], list)): return 'vec';
    return 'mat';        
            
def ifZeroMat(mat, eps=_eps):
    ''' test if all elements in a matrix are zero '''
    for vec in mat:
        if(not ifZeroVec(vec, eps=eps)): return False;
    return True;

def ifZeroVec(vec, eps=_eps):
    ''' test if all elements in a vec are zero '''
    for x in vec:
        if(not ifZeroNum(x, eps=eps)): return False;
    return True;

def ifZero(obj, eps=_eps):
    t = getType(obj);
    if(t == 'num'): return ifZeroNum(obj, eps=eps);
    if(t == 'vec'): return ifZeroVec(obj, eps=eps);
    if(t == 'mat'): return ifZeroMat(obj, eps=eps);
    return '[ifZero]: error type';    

def ifNonPosVec(vec):
    for x in vec:
        if(x > 0): return False;
    return True;

def ifNegVec(vec):
    for x in vec:
        if(x >= 0): return False;
    return True;

def ifNonNegVec(vec):
    for x in vec:
        if(x < 0): return False;
    return True;

def ifPosVec(vec):
    for x in vec:
        if(x <= 0): return False;
    return True;        

def vecToMat(vec):
    ''' convert a l-length vec to a l*1 matrix ''' 
    return [[x] for x in vec];

def matToVec(mat):
    (m, n) = sizeMat(mat);
    if(n != 1):
        print('[convert matrix to vector]: need m * 1 matrix');
        return None;
    return [vec[0] for vec in mat];

def getMatRank(mat):  
    ''' compute rank using Gram-Schmidt Orthogonalization Method '''
    (q, r, colIdx, rank) = _gramschmidtDecomp(mat);
    return rank;

def getMatTrace(mat):
    if(not ifSquareMat(mat)):
        print('[get matrix trace]: not square matrix');
        return None;
    return sum([mat[i][i] for i in range(len(mat))]); 

def getMatDiag(mat):
    ''' get matrix diagonal elements in a vector ''' 
    return [mat[i][i] for i in range(len(mat))];

def getPivotRowMat(rowIdx):
    n = len(rowIdx);
    pMat = zeroes(n, n);
    for i in range(n): pMat[i][rowIdx[i]] = 1.0;
    return pMat; 

def getPivotColMat(colIdx):
    n = len(colIdx);
    pMat = zeroes(n, n);
    for i in range(n): pMat[colIdx[i]][i] = 1.0;
    return pMat;

def getInvPermIdx(idx):
    invIdx = zeroes(len(idx));
    for i in range(len(idx)): invIdx[idx[i]] = i;
    return invIdx;    

def getInvDiagVec(dVec): return [1.0 / d for d in dVec];

def getPermMatIdx(pMat): 
    idx = [];
    for vec in pMat:
        for i in range(len(vec)):
            if(vec[i] == 1.0):
                idx.append(i);
                continue;
    return idx;
    
def det(mat=None, lu=None):
    '''[det]: compute matrix determinant by LU decomposition
    mat -> determinant
    lu  -> determinant
    
    args: mat, lu
    returns: determinant
    '''
    if(((mat is not None) and (not ifSquareMat(mat))) or ((lu is not None) and (not ifSquareMat(lu[0])))): 
        print('[determinant]: not a square matrix');
        return None;
    if(lu is None): lu = _luDecomp(mat);
    return reduce(lambda x, y: x * y, [lu[0][lu[1][i]][i] for i in range(len(lu[1]))]);

NOT_FOLD = True;
#===============================================================================
# LU Decomposition, Linear System, 
#===============================================================================
def _luDecomp(mat):  
    #     pivoted LU decomposition (rowIdx), 
    #     L = m's lower triangular, U = m's upper triangular
    luMat = cloneMat(mat);
    (m, n) = sizeMat(luMat);
    rowIdx = range(m);
    for k in range(min(m, n)):
        pivot = k;
        for p in range(k, m):  # select pivot
            if(abs(luMat[rowIdx[p]][k]) > abs(luMat[rowIdx[pivot]][k])): 
                pivot = p;
        swap = rowIdx[pivot];
        rowIdx[pivot] = rowIdx[k];
        rowIdx[k] = swap;
        if(abs(luMat[swap][k]) <= _eps): continue;  # rank deficiency 
        for p in range(k + 1, m):
            r = luMat[rowIdx[p]][k] / luMat[rowIdx[k]][k];
            luMat[rowIdx[p]][k] = r;
            for j in range(k + 1, n): 
                luMat[rowIdx[p]][j] -= r * luMat[rowIdx[k]][j];
    return (luMat, rowIdx);

def luDecomp(mat=None, lu=None, ifRowIdx=False):
    '''[luDecomp]: pivoted LU decomposition for matrix M:
    mat -> lMat, uMat, pMat: P * M = L * U
    
    args: mat
    returns: lMat, uMat, pMat
    '''
    if(lu is None): lu = _luDecomp(mat);
    (luMat, rowIdx) = lu;
    (m, n) = sizeMat(luMat);    
    luMat = getSubMat(luMat, rowIdx=rowIdx);
    lMat = getLowerTriangularMat(luMat, reduced=True);  # lower-triangular
    for i in range(min(m, n)): lMat[i][i] = 1.0;
    uMat = getUpperTriangularMat(luMat, reduced=True);  # upper-triangular
    if(not ifRowIdx):
        pMat = zeroes(m, m);
        for i in range(m): pMat[i][rowIdx[i]] = 1.0;  # permutation matrix
        return (lMat, uMat, pMat);
    else: return (lMat, uMat, rowIdx);

def _luIfFullRank(lu):  # check if a matrix is singular
    (luMat, rowIdx) = lu;
    (m, n) = sizeMat(luMat);
    for i in range(min(m, n)):
        if(ifZeroNum(luMat[rowIdx[i]][i])): return True;
    return False;
    
def forwardBackwardSub(tMat, vec, ifForward=True, transpose=False,
                       rowIdx=None, colIdx=None, ifOverwrite=False):
    '''[forwardBackwardSub]: forward/backward substitution 
    for a squared lower/upper-triangular matrix.
    '''
    if(ifOverwrite): x = vec;
    else: x = cloneVec(vec);
    n = sizeSquareMat(tMat);
    if(rowIdx is None): rowIdx = range(n);
    if(colIdx is None): colIdx = range(n);
    if(not transpose):
        if(ifForward):
            for i in range(n):
                x[i] = (x[i] - sum([tMat[rowIdx[i]][colIdx[j]] * x[j] 
                        for j in range(i)])) / tMat[rowIdx[i]][colIdx[i]];
        else:
            for i in reversed(range(n)):
                x[i] = (x[i] - sum([tMat[rowIdx[i]][colIdx[j]] * x[j] 
                        for j in range(i + 1, n)])) / tMat[rowIdx[i]][colIdx[i]];
    else:
        if(ifForward):
            for i in range(n):
                x[i] = (x[i] - sum([tMat[colIdx[j]][rowIdx[i]] * x[j] 
                        for j in range(i)])) / tMat[colIdx[i]][rowIdx[i]];
        else:
            for i in reversed(range(n)):
                x[i] = (x[i] - sum([tMat[colIdx[j]][rowIdx[i]] * x[j] 
                        for j in range(i + 1, n)])) / tMat[colIdx[i]][rowIdx[i]];
    return x;

def _linSolveLuVec(lu, vec):  
    # solve a (full-column rank) linear system with LU decomposition
    (luMat, rowIdx) = lu;
    (m, n) = sizeMat(luMat);        
    if(n > m):
        print('[Linear System Solve]: not full-column rank matrix');
        return None;
    x = cloneVec(vec);
    for i in range(m):  # forward substitution
        x[rowIdx[i]] -= sum([luMat[rowIdx[i]][j] * x[rowIdx[j]] 
                             for j in range(0, min(i, n))]);  
    for i in range(n + 1, m):
        if(not ifZeroNum(x[rowIdx[i]])):
            print('[Linear System Solve]: over-determined inconsistent matrix');
            return None;
    for i in reversed(range(n)): 
        if(ifZeroNum(luMat[rowIdx[i]][i])):
            print('[Linear System Solve]: not full-column rank matrix');
            return None;
        x[rowIdx[i]] = (x[rowIdx[i]] - sum([luMat[rowIdx[i]][j] * x[rowIdx[j]] 
                            for j in range(i + 1, n)])) / luMat[rowIdx[i]][i];
    return [x[i] for i in rowIdx[0:n]];

def _linSolveMatVec(mat, vec): return _linSolveLuVec(_luDecomp(mat), vec);

def _linSolveLuMat(lu, mat2): 
    if(_luIfFullRank(lu)):
        print('[Linear System Solve]: not full rank matrix');
        return None;
    else: return transposeMat([_linSolveLuVec(lu, getMatCol(mat2, i)) for i in range(sizeMat(mat2)[1])]);

def _linSolveMatMat(mat1, mat2): return _linSolveLuMat(_luDecomp(mat1), mat2);

def linSolveLuDecompVec(lMat, uMat, rowIdx, vec, ifTranspose=False):
    x = cloneVec(vec);
    if(not ifTranspose):
        x = mulPermMatVec(rowIdx, x);
        x = forwardBackwardSub(tMat=lMat, vec=x, ifForward=True,
                               transpose=False, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        x = forwardBackwardSub(tMat=uMat, vec=x, ifForward=False,
                               transpose=False, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
    else:
        x = forwardBackwardSub(tMat=uMat, vec=x, ifForward=True,
                               transpose=True, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        x = forwardBackwardSub(tMat=lMat, vec=x, ifForward=False,
                               transpose=True, rowIdx=None, colIdx=None,
                               ifOverwrite=True);
        x = mulPermMatVec(getInvPermIdx(rowIdx), x);
    return x;
    
def linSolve(mat=None, vec=None, lu=None, mat1=None, mat2=None):
    '''[linSolve]: linear system solver (1. full-column rank, 2. consistent):
        mat,   vec         -> x:    mat  * x    = vec
        lu,    vec         -> x:    lu   * x    = vec
        lu,    mat2        -> mat:  lu   * mat  = mat2
        mat1,  mat2        -> mat:  mat1 * mat  = mat2
        
        args: mat, vec, lu, mat1, mat2
        returns: x
    '''
    if((mat is not None) and (vec is not None)): 
        return _linSolveMatVec(mat, vec);
    elif((lu is not None) and (vec is not None)): 
        return _linSolveLuVec(lu, vec);
    elif((lu is not None) and (mat2 is not None)): 
        return _linSolveLuMat(lu, mat2);
    elif((mat1 is not None) and (mat2 is not None)): 
        return _linSolveMatMat(mat1, mat2);
    return;

def invMat(mat):
    '''[invMat]: invert a matrix:
    mat -> invMat: mat * invMat = I
    
    args: mat
    returns: invMat
    '''
    if(not ifSquareMat(mat)):
        print('[Inverse Matrix]: not square matrix');
        return None;
    return _linSolveMatMat(mat, eye(sizeMat(mat)[0]));

def invMat2Dim(mat):
    det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if(det == 0): return None;
    return [[mat[1][1] / det, -mat[0][1] / det],
            [-mat[1][0] / det, mat[0][0] / det]];
        
NOT_FOLD = True;
#===============================================================================
# Symmetric matrix decomposition: Cholesky Decomposition, LDL Decomposition
#===============================================================================
def _choleskyDecomp(mat, incompleteDecomp=False, checkSymmetry=True):
    '''returns: -2 not symmetry
                -1 not positive definite
                 0 positive definite
    '''
    if(not incompleteDecomp):
        if(checkSymmetry and (not ifSymmetryMat(mat))): return (-2, None);
        lMat = getLowerTriangularMat(mat);
        d = len(lMat);
        for i in range(d):
            if(lMat[i][i] <= 0): return (-1, None);
            lMat[i][i] = math.sqrt(lMat[i][i]);
            for j in range(i + 1, d): 
                lMat[j][i] /= lMat[i][i];
                for k in range(i + 1, j + 1): lMat[j][k] -= lMat[j][i] * lMat[k][i];
    else:
        if(checkSymmetry and (not ifSymmetryMat(mat))): return (-2, None);
        lMat = getLowerTriangularMat(mat);
        d = len(lMat);
        for i in range(d):
            if(lMat[i][i] <= 0): return (-1, None);
            lMat[i][i] = math.sqrt(lMat[i][i]);
            for j in range(i + 1, d): 
                if(ifZeroNum(lMat[j][i])): lMat[j][i] = 0.0;
                else: lMat[j][i] /= lMat[i][i];
                for k in range(i + 1, j + 1):
                    if(ifZeroNum(lMat[j][k])): lMat[j][k] = 0.0;
                    else: lMat[j][k] -= lMat[j][i] * lMat[k][i];
    return (0, lMat);

def choleskyDecomp(mat, incompleteDecomp=False, checkSymmetry=True):
    '''[choleskyDecomp]: Cholesky Decomposition for Hermitian, positive-definite matrix (PD):
    mat -> lMat: lMat * lMat_transpose = mat
    
    args: mat, a positive definite matrix
    returns: lMat, a lower triangular matrix
    '''
    (flag, lMat) = _choleskyDecomp(mat=mat, incompleteDecomp=incompleteDecomp, checkSymmetry=checkSymmetry);
    if(checkSymmetry and flag == -2): 
        print('[Cholesky Decomposition]: not symmetric matrix');
        return None;
    elif(flag == -1): 
        print('[Cholesky Decomposition]: not positive-definite matrix');
        return None;
    else:
        return lMat;

def ldlDecomp(mat, ifDiagVec=False, checkSymmetry=True):
    '''[ldlDecomp]: LDL Decomposition for symmetric matrix:
    mat -> lMat, dMat: lMat * dMat * lMat_transpose
    
    args: mat, a symmetric matrix
    returns:
        lMat: a lower triangular matrix
        dMat: a diagonal matrix
    '''
    if(checkSymmetry):
        if(not ifSymmetryMat(mat)):
            print('[LDL Decomposition]: not symmetric matrix');
            return None;
    lMat = getLowerTriangularMat(mat);
    d = len(lMat);
    dVec = zeroes(d);
    
    for i in range(d):
        dVec[i] = lMat[i][i];
        lMat[i][i] = 1.0;
        for j in range(i + 1, d):
            lMat[j][i] /= dVec[i];
            for k in range(i + 1, j + 1): 
                lMat[j][k] -= lMat[j][i] * lMat[k][i] * dVec[i];
    if(ifDiagVec): return (lMat, dVec);
    else: return (lMat, diagMat(dVec));

def linSolvePosDefMat(mat=None, ldl=None, vec=None):
    if(ldl is None): ldl = ldlDecomp(mat, ifDiagVec=True);
    (lMat, dVec) = ldl;
    n = sizeVec(dVec);
    x = forwardBackwardSub(tMat=lMat, vec=vec, ifForward=True, transpose=False,
                           rowIdx=None, colIdx=None, ifOverwrite=False);
    x = [x[i] / dVec[i] for i in range(n)];
    x = forwardBackwardSub(tMat=lMat, vec=x, ifForward=False, transpose=True,
                           rowIdx=None, colIdx=None, ifOverwrite=True);
    return x;    

def symIdfDecomp(mat, pivotingMethod='Bunch-Parlett', fullResult=False):
    '''[symIdfDecomp]: pivoted LDL decomposition for symmetric, nonsingular
    (singular case can be handelled by Bunch-Parlett pivoting) matrix (possibly
    indefinite. P * A * P' = L * D * L'
    '''
    def pivotingBunchParlett(p, i, d, m):
        (maxDiag, diagIdx) = (None, None);
        (maxOffD, offDIdx) = (None, None);
        for j in range(i, d):
            if(maxDiag is None or maxDiag < abs(m[p[j]][p[j]])):
                (maxDiag, diagIdx) = (abs(m[p[j]][p[j]]), j);
        for j in range(i, d):
            for k in range(i, j + 1):
                if(maxOffD is None or maxOffD < abs(m[p[j]][p[k]])):
                    (maxOffD, offDIdx) = (abs(m[p[j]][p[k]]), (j, k));
        if(maxOffD is not None):
            if(maxOffD < _eps and maxDiag < _eps): s = 3;
            elif(maxOffD == 0.0 or maxDiag / maxOffD > 0.640388): s = 1;
            else: s = 2;
        else:
            if(maxDiag < _eps): s = 3;
            else: s = 1;
        if(s == 1): return (1, m[p[diagIdx]][p[diagIdx]], diagIdx);
        elif(s == 2): 
            blk = [[m[p[offDIdx[0]]][p[offDIdx[0]]],
                    m[p[offDIdx[0]]][p[offDIdx[1]]]],
                   [m[p[offDIdx[1]]][p[offDIdx[0]]],
                    m[p[offDIdx[1]]][p[offDIdx[1]]]]];
            return (2, blk, offDIdx);
        elif(s == 3): return (3, None, None);
    def pivotingNone(p, i, d, m):
        return (1, m[p[i]][p[i]], i);
    if(pivotingMethod == 'Bunch-Parlett'): method = pivotingBunchParlett;
    d = sizeSquareMat(mat);
    pIdx = [i for i in range(d)];
    i = 0;
    dBlockLst = [];
    mat = cloneMat(mat);
    while(i < d):
        (s, blk, idx) = method(pIdx, i, d, mat);
        if(s == 1):
            (pIdx[i], pIdx[idx]) = (pIdx[idx], pIdx[i]);
            dBlockLst.append((1, blk));
            for j in range(i, d): mat[pIdx[j]][pIdx[i]] /= blk;
            for j in range(i + 1, d):
                for k in range(i + 1, j + 1):
                    mat[pIdx[j]][pIdx[k]] -= mat[pIdx[j]][pIdx[i]] \
                                           * mat[pIdx[k]][pIdx[i]] \
                                           * blk;
                    mat[pIdx[k]][pIdx[j]] = mat[pIdx[j]][pIdx[k]]; 
            i += 1;
        elif(s == 2):
            ii = i + 1;
            (idxi, idxii) = idx;
            (pIdx[i], pIdx[idxi]) = (pIdx[idxi], pIdx[i]);
            if(idxii == i): idxii = idxi;  # special case: swap
            (pIdx[ii], pIdx[idxii]) = (pIdx[idxii], pIdx[ii]);
            dBlockLst.append((2, blk));
            det = blk[0][0] * blk[1][1] - blk[0][1] * blk[1][0];
            iBlk = [[blk[1][1] / det, -blk[0][1] / det],
                    [-blk[1][0] / det, blk[0][0] / det]];
            for j in range(i, d):
                (mat[pIdx[j]][pIdx[i]], mat[pIdx[j]][pIdx[ii]]) = (
                    (mat[pIdx[j]][pIdx[i]] * iBlk[0][0] + 
                     mat[pIdx[j]][pIdx[ii]] * iBlk[1][0]),
                    (mat[pIdx[j]][pIdx[i]] * iBlk[0][1] + 
                     mat[pIdx[j]][pIdx[ii]] * iBlk[1][1]));
            for j in range(i + 2, d):
                for k in range(i + 2, j + 1):
                    mat[pIdx[j]][pIdx[k]] -= (
                mat[pIdx[j]][pIdx[i]] * mat[pIdx[k]][pIdx[i]] * blk[0][0] 
              + mat[pIdx[j]][pIdx[i]] * mat[pIdx[k]][pIdx[ii]] * blk[0][1]
              + mat[pIdx[j]][pIdx[ii]] * mat[pIdx[k]][pIdx[i]] * blk[1][0]
              + mat[pIdx[j]][pIdx[ii]] * mat[pIdx[k]][pIdx[ii]] * blk[1][1]);
                    mat[pIdx[k]][pIdx[j]] = mat[pIdx[j]][pIdx[k]];
            i += 2;
        else:  # no nonsigular block
            break;
    if(fullResult):
        pMat = getPivotRowMat(pIdx);
        lMat = zeroes(d, d);
        for i in range(d):
            for j in range(i + 1):
                lMat[i][j] = mat[pIdx[i]][pIdx[j]];
        dMat = zeroes(d, d);
        i = 0;
        for (s, blk) in dBlockLst:
            if(s == 1):
                dMat[i][i] = blk;
                i += 1;
            else:
                dMat[i][i] = blk[0][0];
                dMat[i][i + 1] = blk[0][1];
                dMat[i + 1][i] = blk[1][0];
                dMat[i + 1][i + 1] = blk[1][1];
                i += 2;
        return (pIdx, mat, dBlockLst, pMat, lMat, dMat);
    else:
        return (pIdx, mat, dBlockLst);

def linSolveSymIdfMat(mat=None, ldl=None, vec=None):
    if(ldl is None): ldl = symIdfDecomp(mat);
    (pIdx, mat, dBlockLst) = ldl;
    x = [vec[i] for i in pIdx];
    x = forwardBackwardSub(tMat=mat, vec=x, ifForward=True, transpose=False,
                           rowIdx=pIdx, colIdx=pIdx, ifOverwrite=True);   
    i = 0;
    for (s, blk) in dBlockLst:
        if(s == 1): x[i] = x[i] / blk;
        else: (x[i], x[i + 1]) = mulMatVec(invMat2Dim(blk), x[i:i + 2]);
        i += s;
    y = forwardBackwardSub(tMat=mat, vec=x, ifForward=False, transpose=True,
                           rowIdx=pIdx, colIdx=pIdx, ifOverwrite=True);
    x = zeroes(len(y));
    for i in range(len(pIdx)): x[pIdx[i]] = y[i];
    return x;

def linSolveSymIdfMatWithIterRefine(ldl, mat, vec, tol=_eps, maxIterNum=100):
    xVec = zeroes(sizeVec(vec));
    iterNum = 0;
    while(iterNum < maxIterNum):
        rVec = subVecVec(vec, mulMatVec(mat, xVec));
        if(dotVecVec(rVec, rVec) <= tol): return xVec;
        dxVec = linSolveSymIdfMat(ldl=ldl, vec=rVec);
        xVec = addVecVec(xVec, dxVec);
        iterNum += 1;
    return;

def linSolveLeastNormSol(mat, vec):
    tMat = transposeMat(mat);
    return mulMatVec(tMat, 
                     linSolvePosDefMat(mat=mulMatMat(mat, tMat), vec=vec));
            
NOT_FOLD = True;
#===============================================================================
# QR Decomposition, Householder, Gram-Schmidt, Givens-Hessenberg Rotation 
#===============================================================================
def _householderReflectorPreApply(mat, v):  
    # householder reflection: mat -> H(v) * mat
    n = len(mat);
    r = n - len(v);
    beta = dotVecVec(v, v);
    if(ifZeroNum(beta)): return;
    for col in range(n):
        gamma = dotVecVec(getMatCol(mat, col, r, n), v);
        for i in range(r, n): mat[i][col] -= (2 * gamma / beta) * v[i - r];
    return;

def _householderReflectorPostApply(mat, v):  
    # householder reflection: mat -> mat * H(v)
    n = sizeSquareMat(mat);
    r = n - len(v);
    beta = dotVecVec(v, v);
    if(ifZeroNum(beta)): return;  
    for row in range(n):
        gamma = dotVecVec(getMatRow(mat, row, r, n), v);
        for j in range(r, n): mat[row][j] -= (2 * gamma / beta) * v[j - r];
    return;        

def hess(mat, checkSquare=True):
    '''[hess]: Reduce a matrix to its Hessenberg form by a sequence of Householder
    reflections, returns the Hessenberg form and a unitary matrix:
    mat -> hMat, pMat: mat = pMat * hMat * pMat_transpose
    
    args: mat
    returns: 
        hMat, Upper Hessenberg matrix (Tridiagonal matrix when mat is symmetric)
        pMat, unitary matrix
    
    complexity: O(N^3)
    reference: Sci Comp, (M.H), P123, P187 
    '''  
    if(not ifSquareMat(mat)):
        print('[hess]: not square matrix');
        return None;
    n = sizeSquareMat(mat);
    hMat = cloneMat(mat);
    pMat = eye(n);
    for k in range(n - 2):
        v = getMatCol(hMat, k, k + 1, n);
        alpha = -sgn(v[0]) * getVecNorm(v);
        v[0] -= alpha; 
        _householderReflectorPreApply(hMat, v);
        _householderReflectorPostApply(hMat, v);
        _householderReflectorPostApply(pMat, v);
    return (pMat, hMat);

def _householderDecomp(mat, ifColPivot=False):  
    # compressed Householder QR decomposition
    (m, n) = sizeMat(mat);
    aMat = rbind(mat, zeroes(n));  
    #     (m+1) * n: Q (upper-triangular) and 
    #     Householder reflectors (lower-traingular)
    kk = 0;
    kkLst = ones(n, val=m - 1);
    colIdx = range(n);
    rank = 0;
    for k in range(n):
        if(ifColPivot):
            (col, coln) = (None, None);
            for j in range(k, n):
                jn = getVecNorm(getMatCol(aMat, colIdx[j], kk, m));
                if(col is None or jn > coln): (col, coln) = (j, jn);
            (colIdx[k], colIdx[col]) = (colIdx[col], colIdx[k]);
        kkLst[colIdx[k]] = kk;
        v = getMatCol(aMat, colIdx[k], kk, m);  # householder reflector
        alpha = -sgn(aMat[kk][colIdx[k]]) * getVecNorm(v);
        if(ifZeroNum(alpha)): continue;  # rank not updated, dependent column
        else: rank = max(rank, kk + 1);
        v[0] -= alpha;
        beta = dotVecVec(v, v);
        #-------------------------------------- this cannot happen if alpha != 0
        if(ifZeroNum(beta)): continue;  # lower parts already eliminated,
        aMat[kk][colIdx[k]] = alpha;
        setMatCol(aMat, colIdx[k], v, kk + 1, m + 1);
        for j in range(k + 1, n):
            gamma = dotVecVec(v, getMatCol(aMat, colIdx[j], kk, m));
            for i in range(kk, m): 
                aMat[i][colIdx[j]] -= (2 * gamma / beta) * v[i - kk];
        kk += 1;
        if(kk == m): break;  # full row rank
    return (aMat, kkLst, colIdx, rank);

def qrHouseholderDecomp(mat, ifColPivot=False, ifShowRank=False):
    '''[qrHouseholderDecomp]: Householder QR Decomposition
    mat -> qMat, rMat: mat = qMat * rMat
    
    args: 
        mat: matrix to be QR decomposed
    returns:
        qMat: orthonormal matrix, full basis
        rMat: upper-triangular matrix
    
    complexity: O(N^3)
    '''
    (m, n) = sizeMat(mat);
    (aMat, kkLst, colIdx, rank) = _householderDecomp(mat, ifColPivot);
    rMat = zeroes(m, n);
    for k in range(n):
        setMatCol(rMat, k, getMatCol(aMat, colIdx[k], 0, kkLst[colIdx[k]] + 1),
                  0, kkLst[colIdx[k]] + 1);
    qMat = eye(m);
    for k in reversed(range(n)):
        v = getMatCol(aMat, colIdx[k], kkLst[colIdx[k]] + 1, m + 1);
        beta = dotVecVec(v, v);
        if(ifZeroNum(beta)): continue;  # no Householder reflector employed
        for j in range(m):  # apply k-th reflector to j-th column in Q
            gamma = dotVecVec(v, getMatCol(qMat, j, kkLst[colIdx[k]], m));
            for i in range(kkLst[colIdx[k]], m):
                qMat[i][j] -= (2 * gamma / beta) * v[i - kkLst[colIdx[k]]];
    addRetLst = [];
    if(ifColPivot): addRetLst.append(colIdx);
    if(ifShowRank): addRetLst.append(rank);
    return ([qMat, rMat] + addRetLst);

def _gramschmidtDecomp(mat):
    (m, n) = size(mat);
    qMat = cloneMat(mat);  # Q: m * n
    rMat = zeroes(n, n);  # R: n * n
    colIdx = range(n);
    rank = 0;
    for k in range(n):        
        (p, normP) = (k, getVecNorm(getMatCol(qMat, colIdx[k])));  # pivot column selection
        for j in range(k + 1, n):
            (px, normPx) = (j, getVecNorm(getMatCol(qMat, colIdx[j])));
            if(normPx > normP): (p, normP) = (px, normPx);
        if(ifZeroNum(normP)): break;
        rank += 1;
        swap = colIdx[k]; 
        colIdx[k] = colIdx[p];
        colIdx[p] = swap;
        for i in range(m): qMat[i][colIdx[k]] /= normP;  # basis
        rMat[colIdx[k]][colIdx[k]] = normP;
        vec = getMatCol(qMat, colIdx[k]);
        for j in range(k + 1, n):
            gamma = dotVecVec(getMatCol(qMat, colIdx[j]), vec);
            rMat[colIdx[k]][colIdx[j]] = gamma;
            for i in range(m): qMat[i][colIdx[j]] -= gamma * qMat[i][colIdx[k]];
    return (qMat, rMat, colIdx, rank);

def qrGramschmidtDecomp(mat): 
    '''[qrGramschmidtDecomp]: Gram-Schmidt QR Decomposition: 
    mat -> (qMat, rMat, pMat): mat * pMat = qMat * rMat
    
    args:
        mat: matrix to be QR decomposed
    returns:
        qMat: a matrix whose columns are basis, (not a full basis)
        rMat: representation of mat * pMat w.r.t. qMat
        pMat: column-pivoted matrix
    
    complexity: O(N^3)
    note: the rank of mat is the number of columns of qMat
    '''
    (m, n) = sizeMat(mat);
    (q, r, colIdx, rank) = _gramschmidtDecomp(mat);
    qMat = getSubMat(q, colIdx=colIdx[0:min(n, rank)]);  # compact Q (under-determined)
    rMat = getSubMat(r, rowIdx=colIdx[0:min(n, rank)], colIdx=colIdx);  # compact R (under-determined)
    pMat = getPivotColMat(colIdx);
    return (qMat, rMat, pMat);

def _givensRotationPreApply(mat, i, j, c, s, colIdx=None):
    if(colIdx is None):
        (m, n) = sizeMat(mat);
        colIdx = range(n);
    for col in colIdx: 
        (mat[i][col], mat[j][col]) = (c * mat[i][col] + s * mat[j][col],
                                      - s * mat[i][col] + c * mat[j][col]);
    return;

def _givensRotationPostApply(mat, i, j, c, s, rowIdx=None):
    if(rowIdx is None):
        (m, n) = sizeMat(mat);
        rowIdx = range(m);
    for row in rowIdx: 
        (mat[row][i], mat[row][j]) = (c * mat[row][i] - s * mat[row][j],
                                      s * mat[row][i] + c * mat[row][j]);
    return;

def inverseGivensRotationArgs(givensRotationArgs):
    if(givensRotationArgs is None): return None;
    (i, j, c, s) = givensRotationArgs;
    return (i, j, c, -s);

def genGivensRotationArgs(ka, kb, a, b):
    if(a == 0 and b == 0): return None;
    elif(abs(a) > abs(b)):  
        # numerical consideration to avoid overflow/underflow
        tan = b / a;
        c = 1.0 / math.sqrt(1.0 + tan * tan);
        s = c * tan;
    else:
        ctg = a / b;
        s = 1.0 / math.sqrt(1.0 + ctg * ctg);
        c = s * ctg;
    givensRotationArgs = (ka, kb, c, s);
    return givensRotationArgs;

def _qrDecompHessMat(hMat):  
    # QR decomposition for Hessenberg Matrix by Givens Rotation
    n = len(hMat);
    rMat = cloneMat(hMat);
    givensLst = [];
    for k in range(n - 1):
        givensRotationArgs = genGivensRotationArgs(k, k + 1,
                                                   rMat[k][k], rMat[k + 1][k]);
        if(givensRotationArgs is None): continue;
        givensLst.append(givensRotationArgs);
        _givensRotationPreApply(rMat, *givensRotationArgs);
    return (rMat, givensLst);  # ... G3 G2 G1 H = R

def qrDecompHessMat(hMat):  
    '''[qrDecompHessMat]: QR decomposition for Hessenberg Matrix
    hMat -> qMat, rMat: hMat = qMat * rMat
    
    args: 
        hMat: Hessenberg Matrix
    returns:
        qMat: orthonormal matrix
        rMat: upper-triangular matrix
    '''
    (rMat, givensLst) = _qrDecompHessMat(hMat);
    n = len(hMat);
    qMat = eye(n);
    for givensRotationArgs in reversed(givensLst): 
        _givensRotationPreApply(qMat,
                                *inverseGivensRotationArgs(givensRotationArgs));
    return (qMat, rMat);

def qrDecomp(mat, method='Householder'):
    '''[qrDecomp]: QR Decomposition of a matrix
    mat  -> qMat, rMat:         mat = qMat * rMat         (Householder)
    mat  -> qMat, rMat, pMat:   mat * pMat = qMat * rMat  (Gramschmidt)
    mat  -> qMat, rMat:         mat = qMat * rMat         (Givens-Hess)
    args: 
        mat: matrix to be QR decomposed
             Hessenberg matrix to be QR decomposed        (Givens-Hess)
        method: 
            Householder, qMat is a full basis matrix
            Gramschmidt, qMat, rMat is reduced to the number of rank
            Givens-Hess, mat input is a Hessenberg matrix
    returns:
        qMat: orthonormal matrix
        rMat: upper-triangular matrix
        pMat: pivoting matrix (Gram-Schmidt methods)
    
    complexity: 
        O(N^3) for Householder, Gramschmidt;
        O(N^2) for Givens-Hess; 
    reference: 
        Householder: Sci Comp, (M.H), P124 Algo 3.1
        Gramschmidt: Sci Comp, (M.H), P131 Algo 3.3
        Givens-Hess: Sci Comp, (M,H), P127, P187        
    '''    
    if(method == 'Householder'): return qrHouseholderDecomp(mat);
    if(method == 'Gramschmidt'): return qrGramschmidtDecomp(mat);
    if(method == 'Givens-Hess'):
        if(not ifUpperHessMat(mat)):
            print('[qrDecomp-Givens-Hess]: not a Hessenberg matrix'); 
            return None;
        return qrDecompHessMat(mat);
    return;

NOT_FOLD = True;
#===============================================================================
# Orthogonalization Methods (Except QR)
#===============================================================================
def extendBasis(mat):  
    '''[extendBasis]: extend basis in mat by guessing
    
    args: mat, a matrix each column is a basis
    returns: bMat, a square basis matrix that extends mat
    '''
    (m, n) = size(mat);
    bLst = [];
    for c in range(n): bLst.append(getMatCol(mat, c));
    while(True):
        vec = randomVec(m);
        for bVec in bLst: 
            vec = subVecVec(vec, mulNumVec(dotVecVec(vec, bVec), bVec));
        vec = getNormalizedVec(vec);
        if(dotVecVec(vec, vec) > _eps): bLst.append(vec);
        if(len(bLst) == m): break; 
    return transposeMat(bLst);

class BasisConstructor(object):
    bLst = [];
    dim = 0;
    
    def __init__(self, vecLst=[]):
        self.bLst = [];
        self.dim = 0;
        for vec in vecLst: self.addVec(vec);
        return;
    
    def addVec(self, vec):
        vec = cloneVec(vec);
        for bVec in self.bLst:
            vec = subVecVec(vec, mulNumVec(dotVecVec(vec, bVec), bVec));
        if(ifZeroVec(vec)):
            return False;
        else: 
            self.bLst.append(getNormalizedVec(vec));
            self.dim += 1;
            return True;
    
NOT_FOLD = True;
#===============================================================================
# Eigen Methods: Power Iteration, Schur decomposition,
#                QR with Shift with Hessenberg reduction
#===============================================================================
def _inverseIteration(mat, shft, maxIter=1e4):  # inverse iteration with shift (Sci. Comp. Algo. 4.3 P177)
    if(not ifSquareMat(mat)):
        print('[Inverse Iteration]: not square matrix');
        return None;
    n = sizeSquareMat(mat);
    mat = subMatMat(mat, mulNumMat(shft, eye(n)));
    lu = _luDecomp(mat);
    if(ifSingularMat(lu=lu)):
        print('[Inverse Iteration]: singular matrix');
        return None;
    x = randomVec(n);
    l = 0.0;
    iter = 0;
    while(True):
        iter += 1;
        y = linSolve(lu=lu, vec=x);
        rq = dotVecVec(x, y) / dotVecVec(y, y);
        if(ifZeroNum(rq - l)): break;
        x = getNormalizedVec(y);
        l = rq;
        if(iter > maxIter):
            print('[Inverse Iteration]: no unqiue eigenvalue close to shift');
            return None;
    l += shft; 
    return (x, l);

def _rayleighQuotientIteration(mat, maxIter=1e4):  # inverse iteration shifted by Rayleigh Quotient (Sci. Comp. Algo. 4.4 P178)
    if(not ifSquareMat(mat)):
        print('[Rayleigh Quotient Iteration]: not square matrix');
        return None;
    n = sizeSquareMat(mat);
    x = randomVec(n);
    l = dotVecVec(x, mulVecMat(x, mat)) / dotVecVec(x, x);
    iter = 0;
    while(True):        
        iter += 1;
        y = linSolve(mat=subMatMat(mat, mulNumMat(l, eye(n))), vec=x);
        if(y is None):
            print('[Rayleigh Quotient Iteration]: almost singular, exit becuase unstability') 
            break;
        rq = dotVecVec(x, y) / dotVecVec(y, y);
        if(abs(rq) < 1e-3): break;
        x = getNormalizedVec(y);
        l += rq;
        if(iter > maxIter):
            print('[Rayleigh Quotient Iteration]: exceed max iter');
            return None;        
    return (x, l);   

def _schurDecompBasicQrIteration(mat):
    '''[schurDecompBasicQrIteration]: compute an upper triangular matrix uMat and a unitary 
    matrix vMat such that mat = uMat * tMat * uMat* is the Schur decomposition of mat. It 
    requires mat has distinct real eigenvalues. If mat is Hermitian, then tMat is a diagonal
    matrix:
    mat -> (uMat, tMat): mat = uMat * tMat * uMat*
    
    args: mat, a square matrix
    returns:
        uMat: unitary matrix
        tMat: upper-triangular matrix, or diagonal matrix if mat is Hermitian/Symmetric
    
    reference: Sol Large-scale Eigenvalue Problems, Chp 3, Algo 3.1, P 52
    '''
    if(not ifSquareMat(mat)):
        print('[schurDecompBasicQrIteration]: not a square matrix');
        return None;
    uMat = eye(sizeSquareMat(mat));
    tMat = cloneMat(mat);
    oldEigDiagVec = zeroes(sizeSquareMat(mat));
    newEigDiagVec = zeroes(sizeSquareMat(mat)); 
    while(True):
        (qMat, rMat) = qrDecomp(tMat);
        uMat = mulMatMat(uMat, qMat);
        tMat = mulMatMat(rMat, qMat);
        newEigDiagVec = getMatDiag(tMat);
        if(ifZeroVec(subVecVec(oldEigDiagVec, newEigDiagVec))): break;
        oldEigDiagVec = newEigDiagVec;
    return (uMat, tMat);

def _schurDecompShiftQrIteration(mat):
    '''[schurDecompShiftQrIteration]: compute an upper triangular matrix uMat and a unitary 
    matrix vMat such that mat = uMat * tMat * uMat* is the Schur decomposition of mat. It 
    requires mat has distinct real eigenvalues. If mat is Hermitian, then tMat is a diagonal
    matrix. Convergence is speeded up by adding shift (bottom-right eigenvalue approximation):
    mat -> (uMat, tMat): mat = uMat * tMat * uMat*
    
    args: mat, a square matrix
    returns:
        uMat: unitary matrix
        tMat: upper-triangular matrix, or diagonal matrix if mat is Hermitian/Symmetric
    '''
    if(not ifSquareMat(mat)):
        print('[schurDecompShiftQrIteration]: not a square matrix');
        return None;
    n = sizeSquareMat(mat);
    uMat = eye(n);
    tMat = cloneMat(mat);
    oldEigDiagVec = zeroes(n);
    newEigDiagVec = zeroes(n); 
    while(True):
        shft = tMat[n - 1][n - 1];
        for i in range(n): tMat[i][i] -= shft;
        (qMat, rMat) = qrDecomp(tMat);
        uMat = mulMatMat(uMat, qMat);
        tMat = mulMatMat(rMat, qMat);
        for i in range(n): tMat[i][i] += shft;
        newEigDiagVec = getMatDiag(tMat);
        if(ifZeroVec(subVecVec(oldEigDiagVec, newEigDiagVec))): break;
        oldEigDiagVec = newEigDiagVec;
    return (uMat, tMat);

def _schurDecompHessRqShiftDeflationQrIteration(hMat):  
    '''[schurDecompHessRqShiftDeflationQrIteration]: compute the schur normal form for a upper Hessenberg matrix.
    QR iteration is performed with Rayleigh Quotient shift. If perfect shift is used, we can deflate, i.e.,
    proceed the algorithm with a smaller matrix. The Hessenberg matrix hMat is overwritten, decomposed it as:
    hMat -> (uMat, tMat): hMat = uMat * tMat * uMat*
    
    args: hMat, a Hessenberg square matrix
    returns:
        uMat: unitary matrix
        tMat: upper-triangular matrix, or diagonal matrix if hMat is Hermitian/Symmetric (tridiagonal)
    
    reference: Sol Large-scale Eigenvalue Problems, Chp 3, Algo 3.4, P 56
    '''
    n = sizeSquareMat(hMat);
    totalGivensLst = [];
    while(True):
        shift = hMat[n - 1][n - 1];
        for k in range(n): hMat[k][k] -= shift;
        (hMat, givensLst) = _qrDecompHessMat(hMat);
        totalGivensLst.extend(givensLst);
        for grArgs in givensLst: 
            _givensRotationPostApply(hMat, *inverseGivensRotationArgs(grArgs));
        for k in range(n): hMat[k][k] += shift;
        if(ifZeroNum(abs(hMat[k][k - 1]))): break;
    if(n == 2):
        tMat = hMat;
        uMat = eye(n);
        for grArgs in reversed(totalGivensLst): 
            _givensRotationPreApply(uMat, *inverseGivensRotationArgs(grArgs));
    else:
        (subU, subT) = _schurDecompHessRqShiftDeflationQrIteration(
                            getSubMat(hMat, range(0, n - 1), range(0, n - 1)));
        b = getMatCol(hMat, n - 1, end=n - 1);
        lbd = hMat[n - 1][n - 1];
        tMat = cbind(subT, mulMatVec(transposeMat(subU), b));  # t
        tMat.append(zeroes(n));
        tMat[n - 1][n - 1] = lbd;
        uMat = cbind(subU, zeroes(n - 1));  # u
        uMat.append(zeroes(n));
        uMat[n - 1][n - 1] = 1;
        for grArgs in reversed(totalGivensLst): 
            _givensRotationPreApply(uMat, *inverseGivensRotationArgs(grArgs));
    return  (uMat, tMat);

def schurDecompHessRqShiftDeflationQrIteration(mat):
    '''[schurDecompHessRqShiftDeflationQrIteration]: compute the schur normal form for a matrix through Hessenberg
    matrix. QR iteration is performed with Rayleigh Quotient shift. If perfect shift is used, we can deflate, i.e.,
    proceed the algorithm with a smaller matrix. The Hessenberg matrix is computed by hess(), and then schur
    decomposed by _schurDecompHessRqShiftDeflationQrIteration(). The overall matrix decomposition is:
    mat -> (uMat, tMat): mat = uMat * tMat * uMat*
    
    args: Mat, a square matrix
    returns:
        uMat: unitary matrix
        tMat: upper-triangular matrix, or diagonal matrix if hMat is Hermitian/Symmetric (tridiagonal)
    
    reference: Sol Large-scale Eigenvalue Problems, Chp 3, Algo 3.4, P 56
    '''
    (pMat, hMat) = hess(mat);
    (uMat, tMat) = _schurDecompHessRqShiftDeflationQrIteration(cloneMat(hMat));
    uMat = mulMatMat(pMat, uMat);
    return (uMat, tMat);

def _svd(mat):
    (n, p) = sizeMat(mat);
    (vMat, dMat) = schurDecompHessRqShiftDeflationQrIteration(mulMatMat(transposeMat(mat), mat));
    colIdx = [k for k in sorted(range(p), key=lambda x:abs(dMat[x][x]), reverse=True) if(abs(dMat[k][k]) >= _eps)];
    vMat = getSubMat(vMat, colIdx=colIdx);
    sMat = diagMat([math.sqrt(x) for x in getSubVec(getMatDiag(dMat), colIdx)]);
    uMat = mulMatMat(mulMatMat(mat, vMat), diagMat([1.0 / x for x in getMatDiag(sMat)]));
    return (uMat, sMat, vMat);

def svd(mat, econ=False):
    '''[svd]: singular-value decomposition. compute the Schur decomposition for mat' * mat (eigen-decomposition):
    It produces a diagonal matrix sMat of the same dimension as mat, with nonnegative diagonal elements in 
    decreasing order, and unitary matrices uMat and vMat so that mat = uMat * sMat * vMat'.
    If 'econ = True', then it produces the "economy size" decomposition:  If X is m-by-n with m > n, then svd 
    computes only the first r columns of U and S is r-by-r; for m < n, only the first r columns of V are computed
    and S is r-by-r where r is the rank.
    mat -> uMat, sMat, vMat: mat = uMat * sMat * vMat'
    '''  
    if(econ): return _svd(mat);
    else:
        (uMat, dMat, vMat) = _svd(mat);
        (uMat, sMat, vMat) = (extendBasis(uMat), zeroes(*sizeMat(mat)), extendBasis(vMat));
        for r in range(len(dMat)): sMat[r][r] = dMat[r][r];
        return (uMat, sMat, vMat);

NOT_FOLD = True;
#===============================================================================
# Documentations
#===============================================================================
def documentation():
    print('documentations:');
    print('==> orthogonalization method');
    print(hess.__doc__);
    print(qrDecomp.__doc__);
    print(extendBasis.__doc__);
    print('==> linear system method');
    print(luDecomp.__doc__);
    print(linSolve.__doc__);
    print(invMat.__doc__);
    print(choleskyDecomp.__doc__);
    print(ldlDecomp.__doc__);
    print(det.__doc__);
    
NOT_FOLD = True;
#===============================================================================
# Test
#===============================================================================
if __name__ == '__main__':
    documentation();
    pass;
