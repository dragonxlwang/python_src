'''
Created on May 7, 2014

@author: xwang95
'''
from toolkit.num.algebra import mulPermMatVec, forwardBackwardSub, dotVecVec, \
    cloneVec, catVecLst, getInvPermIdx, setMatCol, getSubMat, zeroes, getMatCol, \
    sizeSquareMat, luDecomp, cloneMat, randomMat, randomVec, subVecVec, \
    mulMatVec, ifZero, transposeMat, qrHouseholderDecomp, sizeMat, cbind, \
    subMatMat, mulMatMat, ifUnitaryMat, ifUpperTriangularMat, printMat, eye, \
    ifZeroVec, ifEqualMat, randomPosDefMat, ifLowerTriangularMat, mulDiagMatMat
from random import randint, random
from toolkit.num.matrix_decomposition_modification import LuDecompModifier, \
    QrDecompModifier, LdlDecompModifier
import sys
from toolkit.utility import ifEqualLst

def demoLuDecompModifier():
    m = 10;
    mat = randomMat(m, m);
    ludm = LuDecompModifier(mat);
    for i in range(1000):
        c = randint(0, m - 1);
        vec = randomVec(m);
        setMatCol(mat, c, vec);
        ludm.modifyColumn(c, vec);
        y = randomVec(m);
        x = ludm.linSolveMat(y);
        z = subVecVec(y, mulMatVec(mat, x));
        if(not ifZero(z)): print('error');
        y = randomVec(m);
        x = ludm.linSolveMatTranspose(y);
        z = subVecVec(y, mulMatVec(transposeMat(mat), x));
        if(not ifZero(z)): 
            print('error');
            break;
    print('validated!');
    return;

def demoQrDecompModifier():
    (m, n) = (20, 3);
    mat = randomMat(m, n);
    qrdm = QrDecompModifier(mat);
    qMat = cloneMat(qrdm.qMat);
    for i in range(1000):
        x = randint(0, 1);
        (m, n) = sizeMat(mat);
        if(n == m): x = 0;
        if(n == 0): x = 1;
        if(x == 0):
            c = randint(0, n - 1);
            mat = getSubMat(mat, colIdx=[j for j in range(n) if j != c]);
            qrdm.delColumn(c);
#             print 'delete column', c, 'n={0}'.format(qrdm.n);
        else:
            vec = randomVec(m);
            mat = cbind(mat, vec);
            qrdm.addColumn(vec);
#             print 'add column', 'n={0}'.format(qrdm.n);
        z = subMatMat(getSubMat(mat, colIdx=qrdm.colIdx),
                      mulMatMat(qrdm.qMat, qrdm.rMat));   
        if(not (ifZero(z) and ifUnitaryMat(qrdm.qMat) and 
                ifUpperTriangularMat(qrdm.rMat))):
            print('error');
            break;
        d = subMatMat(qrdm.qMat, qMat);
        qcIdx = [j for j in range(m) if(not ifZeroVec(getMatCol(d, j)))];
        if(not ifEqualLst(qcIdx, qrdm.updatedQColIdx)):
            print('error');
            break;
        qMat = cloneMat(qrdm.qMat);
#         printMat(qrdm.qMat);
#         print(qrdm.updatedQColIdx);
#         sys.stdin.readline();
    print('validated!');
    return;

def demoLdlDecompModifier():
    for i in range(10):
        n = 3;
        mat = randomPosDefMat(n);
        ldldm = LdlDecompModifier(mat, refreshPeriod=2);
        for j in range(30):
            pos = randint(0, ldldm.n);
            vec = catVecLst(randomVec(ldldm.n), random() * 100);
            ldldm.expand(pos, vec);
            mat = mulMatMat(ldldm.lMat,
                            mulDiagMatMat(ldldm.dVec,
                                          transposeMat(ldldm.lMat)));
            mat = getSubMat(mat, rowIdx=getInvPermIdx(ldldm.pIdx),
                            colIdx=getInvPermIdx(ldldm.pIdx));
            z1 = subMatMat(mat, ldldm.mat);
            y = randomVec(ldldm.n);
            x = ldldm.linSolve(y);
            z = subVecVec(y, mulMatVec(ldldm.mat, x));
            if(not (ifZero(z) or ifZero(z1) or 
                    ifLowerTriangularMat(ldldm.lMat))):
                print('error!');
                break;
            print('pos=', pos);
            printMat(vec, 'vec');
            printMat(mat, 'mat');
            sys.stdin.readline();
    print('validate!');
    return;

if(__name__ == '__main__'):
#     demoLuDecompModifier();
#     demoQrDecompModifier();
#     demoLdlDecompModifier();

    qrdm = QrDecompModifier(mat=randomMat(4, 0));
    printMat(qrdm.qMat);
    printMat(qrdm.rMat);

#     qrdm.delColumn(0);
#     printMat(qrdm.qMat);
#     printMat(qrdm.rMat);