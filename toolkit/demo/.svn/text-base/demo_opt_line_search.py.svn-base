'''
Created on Feb 15, 2014

@author: xwang95
'''

from toolkit.opt.line_search import wolfeLineSearch, goldenSectionSearch, \
    parabolicInterpolationSearch, cubicInterpolation, ifStop
from toolkit.num.algebra import mulNumVec, mulMatVec, invMat;
import random;
import sys;
from toolkit.num.arithmetic import avg;
import math
from toolkit.num.calculus import derivUniVarFunc
from toolkit.num.arithmetic import ifZeroNum

def demoWolfeLineSearch(ifValidate=True, ifDemo=True):
    if(ifValidate):
        newtonIterNumLst = [];
        linSeaIterNumLst = [];    
        for i in range(50):
            newtonIterNum = 0;
            linSeaIterNum = 0;
            fFunc = lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + (x[0] + x[1]) ** 2;
            gFunc = lambda x: [2.0 * (x[0] - 1.0) + 2.0 * (x[0] + x[1]),
                               2.0 * (x[1] - 1.0) + 2.0 * (x[0] + x[1])];
            invHFunc = lambda x: invMat([[4.0, 2.0],  # newton methods
                                    [2.0, 4.0]]);
#             invHFunc = lambda x: [[1.0, 0.0],  # gradient descent
#                                [0.0, 1.0]];
            x = [(random.random() - 0.5) * 1e3 for i in range(2)];
            fOld = fFunc(x);
            while(True):
                dirVec = mulNumVec(-1, mulMatVec(invHFunc(x), gFunc(x)));
                (a, f, g, x, ei) = wolfeLineSearch(fFunc, gFunc, x, dirVec, initStepLen=1.0, c1=1e-4, c2=0.1, ifEnforceCubic=True);
                linSeaIterNum += ei['iterNum'];
                newtonIterNum += 1;
                if(ifStop(fOld, f, g)): break;
                fOld = f;
            print('f={0:<15.6e}, g={1}, newton={2}, linesearch={3}'.format(f, '{0:<15.6e}'.format(g) if g is not None else '{0:<15}'.format(g), newtonIterNum, linSeaIterNum));
            newtonIterNumLst.append(newtonIterNum);
            linSeaIterNumLst.append(linSeaIterNum);
        print(avg(newtonIterNumLst), avg(linSeaIterNumLst));
    if(ifDemo):
        fFunc = lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + (x[0] + x[1]) ** 2;
        gFunc = lambda x: [2.0 * (x[0] - 1.0) + 2.0 * (x[0] + x[1]),
                           2.0 * (x[1] - 1.0) + 2.0 * (x[0] + x[1])];
        invHFunc = lambda x: [[4.0, 2.0],
                           [2.0, 4.0]];
#         x = [1e3, 1e3];
#         x = [0.0, 0.0];
        x = [(random.random() - 0.5) * 1e3 for i in range(2)];
        fOld = fFunc(x);
        while(True):
            dirVec = mulNumVec(-1, mulMatVec(invHFunc(x), gFunc(x)));
            (a, f, g, x, ei) = wolfeLineSearch(fFunc, gFunc, x, dirVec);  # , ifPrint=True, ifShowWarning=True);
            print('a={0:<15.6e}, f={1:<15.6e}, g={2}, ei={3}'.format(a, f, '{0:<15.6e}'.format(g) if g is not None else g, ei));
            if(abs(fOld - f) / f < 1e-3): break;
            fOld = f;
    return;

def demoGoldenSectionSearch():
    fFunc = lambda x: 0.5 - x * math.exp(-x ** 2);
    (x, ei) = goldenSectionSearch(fFunc, 0, 0.01);
    print x, ei;

def demoParabolicInterpolationSearch():
    fFunc = lambda x: 0.5 - x * math.exp(-x ** 2);
    (x, ei) = parabolicInterpolationSearch(fFunc, 0, 1.2, 0.6);
    print x, ei;
    
def demoCubicInterpolation():
    fFunc = lambda x: 0.5 - x * math.exp(-x ** 2);
    gFunc = lambda x: derivUniVarFunc(fFunc, x, method='richardson');
    (x, ei) = cubicInterpolation(fFunc, gFunc, 0, 1);
    print x, ei;
    
if __name__ == '__main__':
    demoWolfeLineSearch();
#     demoGoldenSectionSearch();
#     demoParabolicInterpolationSearch();
#     demoCubicInterpolation();
    pass;
