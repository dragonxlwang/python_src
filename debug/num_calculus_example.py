
'''
Created on Mar 28, 2013

@author: xwang95
'''

from toolkit.num.algebra import *;
from toolkit.num.arithmetic import *;
from toolkit.num.calculus import *;

if __name__ == '__main__':
    func = lambda(x): 3 * x[0] ** 2 + 6 * x[0] * x[2] + x[1] ** 2 - 4 * x[1] * x[2] + 8 * x[2] ** 2;
    printMat(hessianFunc(func, [1, 1, 1]));
    pass;
