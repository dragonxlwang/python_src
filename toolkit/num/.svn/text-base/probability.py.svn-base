'''
Created on Jan 13, 2014

@author: xwang95
'''
import math;
import random;
import scipy.stats;
from toolkit.num.arithmetic import ifInvalidNum;

def logNormPdf(x, mean, var): return -math.log((2 * math.pi * var) ** 0.5) - ((x - mean) ** 2) / (2.0 * var);

def normPdf(x, mean, var): return math.exp(logNormPdf(x, mean, var));

def multinomialSampling(pmf):
    x = random.random();
    i = 0;
    cmf = 0.0;
    while(True): 
        cmf += pmf[i];
        if(cmf >= x): return i;
        i += 1;
        if(i >= len(pmf)): return len(pmf) - 1;
    return;

def poissonSampling(lmbda, size):
    pmf = [];
    def _poissonProb(k):
        return (math.pow(lmbda, k) * math.exp(-lmbda) / math.factorial(k));
    def sample():
        x = random.random();
        i = 0;
        cmf = 0.0;
        while(True):
            while(i >= len(pmf)): pmf.append(_poissonProb(len(pmf)));
            cmf += pmf[i];
            if(cmf >= x): return i;
            i += 1; 
    return [sample() for k in range(size)];

def normPhi(x): return scipy.stats.norm.cdf(x);

def normQfunc(x):
    return 1 - normPhi(x);

def logNormQfunc(x):
    if(x > 8): return -math.log(12) - (x ** 2) / 2;  # convergence
    return math.log(normQfunc(x));

def normPdfQfuncRatio(x):
    if(x > 8): return 12.0 / math.sqrt(2 * math.pi);  # convergence
    else: return scipy.stats.norm.pdf(x) / normQfunc(x);
    
if __name__ == '__main__':
#     print logNormPdf(0.1, 0, 1);
#     print math.log(scipy.stats.norm.pdf(0.1));
    a = poissonSampling(lmbda=3.0, size=10000);
    b = {};
    for x in a: b[x] = b.get(x, 0) + 1;
    for x in sorted(b): print x, b[x];