'''
Created on Feb 24, 2014

@author: xwang95
'''
from toolkit.opt.trust_region import trustRegion;
from toolkit.num.algebra import mulNumVec, mulMatVec, invMat;
import random;
import sys;
from toolkit.num.arithmetic import avg;
from toolkit.num.algebra import getVecNorm, printMat
import math;

fFunc = lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + (x[0] + x[1]) ** 2;
gFunc = lambda x: [2.0 * (x[0] - 1.0) + 2.0 * (x[0] + x[1]),
                               2.0 * (x[1] - 1.0) + 2.0 * (x[0] + x[1])];
hFunc = lambda x: [[4.0, 2.0],  # newton methods
                   [2.0, 4.0]];
#             hFunc = lambda x: [[1.0, 0.0],  # gradient descent
#                                [0.0, 1.0]];

def demoTrustRegion(ifValidate=True, ifDemo=True):
    if(ifValidate):
        for t in range(100):
            x = [(random.random() - 0.5) * 1e3 for i in range(2)];
            g = gFunc(x);
            gn = getVecNorm(g);
            (x, f, g, ei) = trustRegion(fFunc=fFunc,
                                   gFunc=gFunc,
                                   hFunc=hFunc,
                                   mulHessGradFunc=None,
                                   xBeg=x,
                                   maxRadius=10.0 * gn,
                                   initRadius=1.0 * gn,
                                   method='cauchy_point',
#                                    method='dogleg',
                                   maxIterNum=20,
                                   ifPrint=False,
                                   ifShowWarning=False);
            print('test[{0}]: f = {1:<15.6e}, exitInfo={2}'.format(t, f, ei));
        
    if(ifDemo):
        x = [(random.random() - 0.5) * 1e3 for i in range(2)];
        g = gFunc(x);
        gn = getVecNorm(g);
    
        (x, f, g, ei) = trustRegion(fFunc=fFunc,
                                   gFunc=gFunc,
                                   hFunc=hFunc,
                                   mulHessGradFunc=None,
                                   xBeg=x,
                                   maxRadius=10.0 * gn,
                                   initRadius=0.001 * gn,
#                                    method='cauchy_point',
                                    method='dogleg',
                                   maxIterNum=20,
                                   ifPrint=True,
                                   ifShowWarning=True);
        print('****************************')
        printMat(x, 'x', decor='e');
        print('f = {0:<15.6e}'.format(f));
        printMat(g, 'g', decor='e');
        print(ei);
    return;

if __name__ == '__main__':
    demoTrustRegion();
    
