'''
Created on Mar 15, 2013

@author: xwang95
'''
import math;
import random;
from toolkit.utility import parseNumVal

_eps = 1e-8;
_h = 1e-6;

def sgn(x):
    '''tri-value''' 
    if(x > _eps): return 1.0;
    elif(x < -_eps): return -1.0;
    else: return 0.0;

def binSgn(x): return 1.0 if x >= 0.0 else -1.0;  # binary value +/-1

def avg(x): return float(sum(x)) / len(x);

def var(x): 
    a = avg(x);
    return sum([float(((y - a) ** 2)) for y in x]) / len(x);

def std(x): return math.sqrt(var(x));

def sqrt(x): 
    if(abs(x) <= _eps): return 0.0;
    return math.sqrt(x);

def getRandomSubsetIdx(n, m= -1):
    if(m == -1): m = n;
    v = range(n);
    random.shuffle(v);
    return v[0:m];

def getQuotientReminder(x, p):
    q = x / p;
    r = x - x * p;
    return (q, r);

def ifInvalidNum(x):
    if(math.isinf(x) or math.isnan(x)): return True;
    return False;

def ifZeroNum(x, eps=_eps): return (abs(x) <= eps);

def getExponent(x): return parseNumVal('{0:e}'.format(x).split('e')[1]);
    
def minIdx(vec):
    if(len(vec) == 0): return -1;
    else:
        x = 0;
        for i in range(1, len(vec)):
            if(vec[i] < vec[x]):
                x = i;
        return x;

def maxIdx(vec):
    if(len(vec) == 0): return -1;
    else:
        x = 0;
        for i in range(1, len(vec)):
            if(vec[i] > vec[x]):
                x = i;
        return x;

if __name__ == '__main__':
    print getExponent(12334.2);
    pass
