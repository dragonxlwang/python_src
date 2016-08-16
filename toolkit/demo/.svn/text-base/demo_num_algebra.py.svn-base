'''
Created on Mar 16, 2013

@author: xwang95
'''
from toolkit.num.algebra import *;
from toolkit.num import algebra;
import sys;
from toolkit.num.algebra import _luDecomp;
import random;
import math;
import timeit;
from array import array;
from toolkit.num.arithmetic import ifInvalidNum

sqMat = [[0.421761282626275, 0.035711678574190, 0.743132468124916, 0.031832846377421, 0.694828622975817],
         [0.915735525189067, 0.849129305868777, 0.392227019534168, 0.276922984960890, 0.317099480060861],
         [0.792207329559554, 0.933993247757551, 0.655477890177557, 0.046171390631154, 0.950222048838355],
         [0.959492426392903, 0.678735154857773, 0.171186687811562, 0.097131781235848, 0.034446080502909],
         [0.655740699156587, 0.757740130578333, 0.706046088019609, 0.823457828327293, 0.438744359656398]];
b = [0.381558457093008, 0.765516788149002, 0.795199901137063, 0.186872604554379, 0.489764395788231];

def orthogonalizationMethod(ifValidate=True, ifDemo=True):
    if(ifValidate):
        for i in range(100):
            if(i < 50):
                mat = randomMat(10, 10);
                (p, h) = hess(mat);
                z = subMatMat(mat, mulMatLst(p, h, transposeMat(p)));
                if(not (ifZero(z) and ifUnitaryMat(p) and ifUpperHessMat(h))): break;
            if(i < 100):
                mat = randomPosDefMat(10);
                (p, h) = hess(mat);
                z = subMatMat(mat, mulMatLst(p, h, transposeMat(p)));
                if(not (ifZero(z) and ifUnitaryMat(p) and ifTridiagonalMat(h) and ifSymmetryMat(h))): break;
        if(i == 99): print('==> hess validated');
        else: print('==> hess error!');
        for i in range(100):
            if(i < 10):
                mat = randomMat(10, 10);
                (qMat, rMat) = qrDecomp(mat, method='Householder');
                z = subMatMat(mat, mulMatMat(qMat, rMat));
                if(not (ifZero(z) and ifUnitaryMat(qMat) and ifUpperTriangularMat(rMat))): break;
            if(i < 20):
                mat = randomMat(10, 5);
                mat = cbind(mat, mat);
                (qMat, rMat) = qrDecomp(mat, method='Householder');
                z = subMatMat(mat, mulMatMat(qMat, rMat));
                if(not (ifZero(z) and ifUnitaryMat(qMat) and ifUpperTriangularMat(rMat))): break;
            if(i < 30):
                mat = randomMat(5, 10);
                mat = rbind(mat, mat);
                (qMat, rMat) = qrDecomp(mat, method='Householder');
                z = subMatMat(mat, mulMatMat(qMat, rMat));
                if(not (ifZero(z) and ifUnitaryMat(qMat) and ifUpperTriangularMat(rMat))): break;
            if(i < 40):
                mat = randomMat(10, 10);
                (qMat, rMat, pMat) = qrDecomp(mat, method='Gramschmidt');
                z = subMatMat(mulMatMat(mat, pMat), mulMatMat(qMat, rMat));
                if(not (ifZero(z) and ifEyeMat(mulMatMat(transposeMat(qMat), qMat)) and ifUpperTriangularMat(rMat))): break;
            if(i < 50):
                mat = randomMat(10, 5);
                mat = cbind(mat, mat);
                (qMat, rMat, pMat) = qrDecomp(mat, method='Gramschmidt');
                z = subMatMat(mulMatMat(mat, pMat), mulMatMat(qMat, rMat));
                if(not (ifZero(z) and ifEyeMat(mulMatMat(transposeMat(qMat), qMat)) and ifUpperTriangularMat(rMat))): break;
            if(i < 60):
                mat = randomMat(5, 10);
                mat = rbind(mat, mat);
                (qMat, rMat, pMat) = qrDecomp(mat, method='Gramschmidt');
                z = subMatMat(mulMatMat(mat, pMat), mulMatMat(qMat, rMat));
                if(not (ifZero(z) and ifEyeMat(mulMatMat(transposeMat(qMat), qMat)) and ifUpperTriangularMat(rMat))): break;
            if(i < 70):
                mat = getUpperHessMat(randomMat(10, 10));
                (qMat, rMat) = qrDecomp(mat, method='Givens-Hess');
                z = subMatMat(mat, mulMatMat(qMat, rMat));
                if(not (ifZero(z) and ifUnitaryMat(qMat) and ifUpperTriangularMat(rMat))): break;
            if(i < 100):
                mat = getUpperHessMat(randomSymmetricMat(10));
                (qMat, rMat) = qrDecomp(mat, method='Givens-Hess');
                z = subMatMat(mat, mulMatMat(qMat, rMat));
                if(not (ifZero(z) and ifUnitaryMat(qMat) and ifUpperTriangularMat(rMat))): break;
            if(i < 90):
                mat = randomMat(5, 10);
                mat = rbind(mat, mat);
                (qMat, rMat, pMat) = qrDecomp(mat, method='Gramschmidt');
                qMat = extendBasis(qMat);
                if(not ifUnitaryMat(qMat)): break;
        if(i == 99): print('==> QR validated');
        else: print('==> QR error!');
        for i in range(100):
            if(i < 100):
                mat = randomMat(5, 10);
                mat = rbind(mat, mat);
                (qMat, rMat, pMat) = qrDecomp(mat, method='Gramschmidt');
                qMat = extendBasis(qMat);
                if(not ifUnitaryMat(qMat)): break;
        if(i == 99): print('==> extendBasis validated');
        else: print('==> extendBasis error!');
    
    print('');
    print('documentations:');
    print(hess.__doc__);
    print(qrDecomp.__doc__);
    print(extendBasis.__doc__);
#     return;
    if(ifDemo):
        #=======================================================================
        # Hess
        #=======================================================================
        print('==> Hessenberg Decomposition for random matrix')
        mat = randomMat(10, 10);
        (p, h) = hess(mat);
        printMat(p, 'p');
        printMat(h, 'h');
        z = subMatMat(mat, mulMatLst(p, h, transposeMat(p)));
        print('[{0}]: if M = P * H * P_transpose'.format(ifZero(z)));
        print('[{0}]: if P unitary matrix'.format(ifUnitaryMat(p)));
        print('[{0}]: if H upper Hessenberg matrix'.format(ifUpperHessMat(h)));
        print('');
        
        print('==> Hessenberg Decomposition for positive definite matrix')
        mat = randomPosDefMat(10);
        (p, h) = hess(mat);
        printMat(p, 'p');
        printMat(h, 'h');
        z = subMatMat(mat, mulMatLst(p, h, transposeMat(p)));
        print('[{0}]: if M = P * H * P_transpose'.format(ifZero(z)));
        print('[{0}]: if P unitary matrix'.format(ifUnitaryMat(p)));
        print('[{0}]: if H tridiagonal matrix'.format(ifTridiagonalMat(h)));
        print('[{0}]: if H symmetric matrix'.format(ifSymmetryMat(h)));
        print('');
        #=======================================================================
        # QR
        #=======================================================================
        print('==> QR decomposition with Householder method')
        mat = randomMat(10, 10);
        (qMat, rMat) = qrDecomp(mat, method='Householder');
        printMat(qMat, 'qMat');
        printMat(rMat, 'rMat');
        z = subMatMat(mat, mulMatMat(qMat, rMat));
        print('[{0}]: if M = Q * R'.format(ifZero(z)));
        print('[{0}]: if Q unitary matrix'.format(ifUnitaryMat(qMat)));
        print('[{0}]: if R upper-triangular matrix'.format(ifUpperTriangularMat(rMat)));
        print('');
        
        print('==> QR decomposition with Gram-Schmidt method')
        mat = randomMat(5, 10);
        mat = rbind(mat, mat);
        (qMat, rMat, pMat) = qrDecomp(mat, method='Gramschmidt');
        printMat(qMat, 'qMat');
        printMat(rMat, 'rMat');
        printMat(pMat, 'pMat');
        z = subMatMat(mulMatMat(mat, pMat), mulMatMat(qMat, rMat));
        print('[{0}]: if M * P = Q * R'.format(ifZero(z)));
        print('[{0}]: if Q\'s columns are basis'.format(ifEyeMat(mulMatMat(transposeMat(qMat), qMat))));
        print('[{0}]: if R upper-triangular matrix'.format(ifUpperTriangularMat(rMat)));
        print('');
        
        print('==> QR decomposition with Givens-Hess method');
        mat = getUpperHessMat(randomMat(10, 10));
        (qMat, rMat) = qrDecomp(mat, method='Givens-Hess');
        printMat(qMat, 'qMat');
        printMat(rMat, 'rMat');
        z = subMatMat(mat, mulMatMat(qMat, rMat));
        print('[{0}]: if M = Q * R'.format(ifZero(z)));
        print('[{0}]: if Q unitary matrix'.format(ifUnitaryMat(qMat)));
        print('[{0}]: if R upper-triangular matrix'.format(ifUpperTriangularMat(rMat)));
        mat = getUpperHessMat(randomSymmetricMat(10));
        (qMat, rMat) = qrDecomp(mat, method='Givens-Hess');
        printMat(qMat, 'qMat');
        printMat(rMat, 'rMat');
        z = subMatMat(mat, mulMatMat(qMat, rMat));
        print('[{0}]: if M = Q * R'.format(ifZero(z)));
        print('[{0}]: if Q unitary matrix'.format(ifUnitaryMat(qMat)));
        print('[{0}]: if R upper-triangular matrix'.format(ifUpperTriangularMat(rMat)));
    return;

def linearSystemMethod(ifValidate=True, ifDemo=True):
    if(ifValidate):
        for i in range(100):
            m = randomMat(5, 5);
            [l, u, p] = luDecomp(m);
            z = subMatMat(mulMatMat(p, m), mulMatMat(l, u));
            if(not ifZero(z)):
                print('==> luDecomp error!');
                break;
        if(i == 99): print('==> luDecomp validated');
        for i in range(100):
            if(i < 20):  # full-rank matrix, vec
                m = addMatMat(randomMat(5, 5), mulNumMat(10, eye(5)));
                x = randomVec(5);
                z = subVecVec(mulMatVec(m, linSolve(mat=m, vec=x)), x);
            elif(i < 40):  # full-column rank, matrix, vec (over-determinent)
                m = addMatMat(randomMat(8, 4), mulNumMat(10, eye(8, 4)));
                x = mulMatVec(m, randomVec(4));
                z = subVecVec(mulMatVec(m, linSolve(mat=m, vec=x)), x);
            elif(i < 60):  # full-column rank, matrix, matrix (over-determinent)
                mat1 = randomMat(8, 4);
                mat2 = cbind(*[mulMatVec(mat1, randomVec(4)) for c in range(5)]);
                x = linSolve(mat1=mat1, mat2=mat2);
                z = subMatMat(mulMatMat(mat1, x), mat2);
            elif(i < 100):  # invMat
                mat = randomMat(5, 5);
                iMat = invMat(mat);
                z = subMatMat(eye(5), mulMatMat(mat, iMat));
            if(not ifZero(z)):
                print('==> linSolve error!');
                break;
        if(i == 99): print('==> linSolve validated');
        for i in range(100):
            if(i < 50):  # cholesky decomposition
                mat = addMatMat(randomMat(5, 5), mulNumMat(10, eye(5)));
                mat = addMatMat(mat, transposeMat(mat));
                lMat = choleskyDecomp(mat, checkSymmetry=True);
                if(not ifLowerTriangularMat(lMat)): break;
                z = subMatMat(mat, mulMatMat(lMat, transposeMat(lMat)));
            elif(i < 100):  # LDL decomposition
                mat = randomMat(5, 5);
                mat = addMatMat(mat, transposeMat(mat));
#                 mat = subMatMat(mat, mulNumMat(100, eye(5)));
                (lMat, dMat) = ldlDecomp(mat, checkSymmetry=True);
                z = subMatMat(mat, mulMatLst(lMat, dMat, transposeMat(lMat)));
            if(not ifZero(z)):
                print('==> cholesky error!');
                break;
        if(i == 99): print('==> choleskyDecomp validated');
    print(luDecomp.__doc__);
    print(linSolve.__doc__);
    print(invMat.__doc__);
    print(choleskyDecomp.__doc__);
    print(ldlDecomp.__doc__);
    print(det.__doc__);
    
    a = sqMat;
    if(ifDemo):
        #===========================================================================
        # linear equation system
        #===========================================================================
        [l, u, p] = luDecomp(a);
        x = linSolve(mat=a, vec=b);
        printMat(a, 'a');
        printMat(b, 'b');
        printMat(l, 'l');
        printMat(u, 'u');
        printMat(p, 'p');
        printMat(mulMatMat(l, u), 'l*u');
        printMat(mulMatMat(p, a), 'p*a');
        printMat(subVecVec(b, mulMatVec(a, x)), 'residual');
        
        m = randomMat(8, 4);  # singular matrix
        y = mulMatVec(m, randomVec(4));
#         y = randomVec(8);
        x = linSolve(mat=m, vec=y);
        [l, u, p] = luDecomp(m);
        printMat(l, 'l');
        printMat(u, 'u');
        printMat(p, 'p');
        printMat(mulMatMat(l, u), 'l*u');
        printMat(mulMatMat(p, m), 'p*m');
        printMat(x);
#         printMat(subVecVec(y, mulMatVec(m, x)), 'residual');

        mat1 = randomMat(8, 4)
        mat2 = cbind(*[mulMatVec(mat1, randomVec(4)) for i in range(5)]);
        x = linSolve(mat1=mat1, mat2=mat2);
        z = subMatMat(mulMatMat(mat1, x), mat2);
    
        #===========================================================================
        # matrix inverse
        #===========================================================================
        a = randomMat(7, 7);  # non-singular matrix
        x = invMat(a);
        printMat(a, 'a');
        printMat(x, 'inv(a)');
        printMat(mulMatMat(a, x), 'a * inv(a)');
        
        a = randomMat(8, 4);  # singular matrix
        a = cbind(a, a);
        printMat(a, 'a')
        printMat(invMat(a), 'inv(a)');
    
        #===========================================================================
        # Choleskey Decomposition
        #===========================================================================
        a = randomMat(10, 10);
        d = addMatMat(addMatMat(a, transposeMat(a)), mulNumMat(5, eye(10)));
        printMat(d, 'M');
        x = choleskyDecomp(d);
        y = transposeMat(x);
        printMat(x, 'L');
        printMat(y, 'L\'');
        printMat(subMatMat(mulMatMat(x, y), d), 'LL\' - M');
    return;

def eigMethod(ifValidate=True, ifDemo=True):
    if(ifValidate):
        for i in range(100):
            if(i < 20):
                dMat = diagMat(randomVec(5));
                pMat = randomMat(5, 5);
                mat = mulMatLst(pMat, dMat, invMat(pMat));
                (uMat, tMat) = algebra._schurDecompBasicQrIteration(mat);
                z = subMatMat(mat, mulMatLst(uMat, tMat, transposeMat(uMat)));
                if(not (ifZeroMat(z) and ifUnitaryMat(uMat) and ifUpperTriangularMat(tMat, eps=1e-2))): break;
            elif(i < 40):
                dMat = diagMat(randomVec(5));
                pMat = randomMat(5, 5);
                mat = mulMatLst(pMat, dMat, invMat(pMat));
                (uMat, tMat) = algebra._schurDecompShiftQrIteration(mat);
                z = subMatMat(mat, mulMatLst(uMat, tMat, transposeMat(uMat)));
                if(not (ifZeroMat(z) and ifUnitaryMat(uMat) and ifUpperTriangularMat(tMat, eps=1e-2))): break;
            elif(i < 60):
                dMat = diagMat(randomVec(5));
                pMat = randomMat(5, 5);
                mat = mulMatLst(pMat, dMat, invMat(pMat));
                (uMat, tMat) = schurDecompHessRqShiftDeflationQrIteration(mat);
                z = subMatMat(mat, mulMatLst(uMat, tMat, transposeMat(uMat)));
                if(not (ifZeroMat(z) and ifUnitaryMat(uMat) and ifUpperTriangularMat(tMat, eps=1e-2))): break;
            else:
                mat = randomMat(5, 5);
                mat = addMatMat(mat, transposeMat(mat));
                (uMat, tMat) = schurDecompHessRqShiftDeflationQrIteration(mat);
                z = subMatMat(mat, mulMatLst(uMat, tMat, transposeMat(uMat)));
                if(not (ifZeroMat(z) and ifUnitaryMat(uMat) and ifUpperTriangularMat(tMat, eps=1e-2))): break;
        if(i == 99):
            print('''validated: [schurDecompBasicQrIteration],
           [schurDecompShiftQrIteration],
           [schurDecompHessRqShiftDeflationQrIteration]''');
        else:
            print('''error    : [schurDecompBasicQrIteration],
           [schurDecompShiftQrIteration],
           [schurDecompHessRqShiftDeflationQrIteration]''');
    print('');
    
    print('documentations');
    print(algebra._schurDecompBasicQrIteration.__doc__);
    print(algebra._schurDecompShiftQrIteration.__doc__);
    print(schurDecompHessRqShiftDeflationQrIteration.__doc__);
    print(svd.__doc__);
    print('');
    
    if(ifDemo):
        print('==> [schurDecompBasicQrIteration]:')
        dMat = diagMat(randomVec(5));
        pMat = randomMat(5, 5);
        mat = mulMatLst(pMat, dMat, invMat(pMat));
        (uMat, tMat) = algebra._schurDecompBasicQrIteration(mat);
        z = subMatMat(mat, mulMatLst(uMat, tMat, transposeMat(uMat)));
        print('[{0}]: if mat = uMat * tMat * uMat*'.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifUnitaryMat(uMat)));
        print('[{0}]: if tMat upper-triangular matrix'.format(ifUpperTriangularMat(tMat, eps=1e-2)));
        print('');
        
        print('==> [schurDecompShiftQrIteration]:')
        dMat = diagMat(randomVec(5));
        pMat = randomMat(5, 5);
        mat = mulMatLst(pMat, dMat, invMat(pMat));
        (uMat, tMat) = algebra._schurDecompShiftQrIteration(mat);
        z = subMatMat(mat, mulMatLst(uMat, tMat, transposeMat(uMat)));
        print('[{0}]: if mat = uMat * tMat * uMat*'.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifUnitaryMat(uMat)));
        print('[{0}]: if tMat upper-triangular matrix'.format(ifUpperTriangularMat(tMat, eps=1e-2)));
        print('');
        
        print('==> [schurDecompHessRqShiftDeflationQrIteration]:')
        dMat = diagMat(randomVec(5));
        pMat = randomMat(5, 5);
        mat = mulMatLst(pMat, dMat, invMat(pMat));
        (uMat, tMat) = schurDecompHessRqShiftDeflationQrIteration(mat);
        z = subMatMat(mat, mulMatLst(uMat, tMat, transposeMat(uMat)));
        print('[{0}]: if mat = uMat * tMat * uMat*'.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifUnitaryMat(uMat)));
        print('[{0}]: if tMat upper-triangular matrix'.format(ifUpperTriangularMat(tMat, eps=1e-2)));
        print('');
        
        print('==> [svd]')
        mat = randomMat(5, 10);
        (uMat, sMat, vMat) = svd(mat);
        z = subMatMat(mat, mulMatLst(uMat, sMat, transposeMat(vMat)));
        print('[{0}]: if mat = uMat * sMat * vMat\''.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifUnitaryMat(uMat)));
        print('[{0}]: if vMat unitary matrix'.format(ifUnitaryMat(vMat)));
        
        mat = randomMat(10, 5);
        (uMat, sMat, vMat) = svd(mat);
        z = subMatMat(mat, mulMatLst(uMat, sMat, transposeMat(vMat)));
        print('[{0}]: if mat = uMat * sMat * vMat\''.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifUnitaryMat(uMat)));
        print('[{0}]: if vMat unitary matrix'.format(ifUnitaryMat(vMat)));
        
        mat = randomMat(10, 10);
        (uMat, sMat, vMat) = svd(mat);
        z = subMatMat(mat, mulMatLst(uMat, sMat, transposeMat(vMat)));
        print('[{0}]: if mat = uMat * sMat * vMat\''.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifUnitaryMat(uMat)));
        print('[{0}]: if vMat unitary matrix'.format(ifUnitaryMat(vMat)));    
        
        print('==> [svdEcon]')
        mat = randomMat(5, 10);
        (uMat, sMat, vMat) = svd(mat, econ=True);
        z = subMatMat(mat, mulMatLst(uMat, sMat, transposeMat(vMat)));
        print('[{0}]: if mat = uMat * sMat * vMat\''.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifEyeMat(mulMatMat(transposeMat(uMat), uMat))));
        print('[{0}]: if vMat unitary matrix'.format(ifEyeMat(mulMatMat(transposeMat(vMat), vMat))));
        
        mat = randomMat(10, 5);
        (uMat, sMat, vMat) = svd(mat, econ=True);
        z = subMatMat(mat, mulMatLst(uMat, sMat, transposeMat(vMat)));
        print('[{0}]: if mat = uMat * sMat * vMat\''.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifEyeMat(mulMatMat(transposeMat(uMat), uMat))));
        print('[{0}]: if vMat unitary matrix'.format(ifEyeMat(mulMatMat(transposeMat(vMat), vMat))));
        
        mat = randomMat(10, 10);
        (uMat, sMat, vMat) = svd(mat, econ=True);
        z = subMatMat(mat, mulMatLst(uMat, sMat, transposeMat(vMat)));
        print('[{0}]: if mat = uMat * sMat * vMat\''.format(ifZeroMat(z)));
        print('[{0}]: if uMat unitary matrix'.format(ifEyeMat(mulMatMat(transposeMat(uMat), uMat))));
        print('[{0}]: if vMat unitary matrix'.format(ifEyeMat(mulMatMat(transposeMat(vMat), vMat))));    
    return;


if __name__ == '__main__':
#     print('time={0}s'.format(timeit.timeit(setup="from __main__ import linearSystemMethod", stmt='linearSystemMethod()', number=1))); 
    print('time={0}s'.format(timeit.timeit(setup="from __main__ import orthogonalizationMethod", stmt='orthogonalizationMethod()', number=1)));
#     print('time={0}s'.format(timeit.timeit(setup="from __main__ import eigMethod", stmt='eigMethod()', number=1)));
    
#     a = randomSymmetricMat(20);
#     for i in range(10):
#         a = randomSymmetricMat(20);
#         vec = randomVec(20);
# #         a = cbind(a, a);
# #         a = rbind(a, a);
#         (pIdx, mat, dBlockLst, pMat, lMat, dMat) = symIdfDecomp(a, fullResult=True);
# #         z = subMatMat(mulMatLst(pMat, a, transposeMat(pMat)),
# #                   mulMatLst(lMat, dMat, transposeMat(lMat)));
# #         printMat(dMat, 'd');
# #         print(ifZero(z));
# #         printMat(lMat, 'lmat');
# #         printMat(dMat, 'dmat');

        
#         print linSolve(mat=mulMatMat(transposeMat(lMat), pMat), vec=vec);
#         x = linSolveSymIdfMat(mat=a, vec=vec);
#         z = subVecVec(mulMatVec(a, x), vec);
#         print ifZero(z);
#         x = linSolveSymIdfMat(ldl=(pIdx, mat, dBlockLst), vec=vec);
#         z = subVecVec(mulMatVec(a, x), vec);
#         print ifZero(z);
#         

#     printMat(z, 'z');
#     print pIdx
    
#     for blk in dBlockLst: print(blk);
#     if(not ifZero(z)):
#         printMat(z);
    
#     amat = randomMat(2, 10);
#     bmat = randomMat(2, 10);
#     cmat = randomMat(2, 10);
#     dmat = randomMat(5, 10);
#     mat = transposeMat(rbind(amat, bmat, amat, dmat, amat, amat,
#                              dmat, cmat, dmat, amat, bmat, dmat, cmat));
#     mat = transposeMat(rbind(amat, bmat, amat, bmat, cmat, cmat, cmat, 
#                              cmat, cmat, cmat));
#      
#     (qMat, rMat, colIdx, rank) = qrHouseholderDecomp(mat, ifColPivot=True,
#                                                      ifShowRank=True);
#     print rank;
# #     printMat(qMat, 'qMat', shortZero=True, abbrev=None);
# #     printMat(rMat, 'rMat', shortZero=True, abbrev=None);
#       
#     z = subMatMat(getSubMat(mat, colIdx=colIdx), mulMatMat(qMat, rMat));
#     if(not (ifZero(z) and ifUnitaryMat(qMat) and ifUpperTriangularMat(rMat))):
#         print 'error';
#     else: print 'correct';
#       
#     (qMat, rMat, rank) = qrHouseholderDecomp(mat, ifColPivot=False,
#                                                      ifShowRank=True);
#     print rank;
#     printMat(qMat, 'qMat', shortZero=True, abbrev=None);
#     printMat(rMat, 'rMat', shortZero=True, abbrev=None);
#       
#     z = subMatMat(mat, mulMatMat(qMat, rMat));
#     if(not (ifZero(z) and ifUnitaryMat(qMat) and ifUpperTriangularMat(rMat))):
#         print 'error';
# #         printMat(z, 'z', shortZero=True, abbrev=None)
#     else: print 'correct';
    
#     a = randomMat(10, 10);
#     lu = _luDecomp(a);
#     (l, u, r) = luDecomp(lu=lu, ifRowIdx=True);
#     printMat(subMatMat(getSubMat(a, rowIdx=r), mulMatMat(l, u)), shortZero=True);

#     a = randomMat(5, 10);
#     (l, u, p) = luDecomp(a, ifRowIdx=True);
#     z = subMatMat(mulMatMat(l, u), getSubMat(a, rowIdx=p));
#     print(ifZero(z));

#     printMat(catVecLst(range(5), range(5)));
#     mat = randomMat(5, 5);
#     (lMat, uMat, rowIdx) = luDecomp(mat, ifRowIdx=True);
#     zMat = subMatMat(mulMatMat(lMat, uMat), getSubMat(mat, rowIdx=rowIdx));
#     printMat(zMat, 'zmat', shortZero=True);

#     mat = randomMat(5, 5);
#     mat = addMatMat(mat, transposeMat(mat));
# #                 mat = subMatMat(mat, mulNumMat(100, eye(5)));
#     (lMat, dMat) = ldlDecomp(mat, ifDiagVec=True, checkSymmetry=True);
#     z = subMatMat(mat, mulMatLst(lMat, diagMat(dMat), transposeMat(lMat)));
#     printMat(z)
    
#     for i in range(10):
#         v = getMatCol(qMat, i);
#         print i;
#         printMat(mulMatVec(transposeMat(mat), v), 'm*v', shortZero=True);
    pass;
