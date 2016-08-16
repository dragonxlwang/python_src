'''
Created on Aug 18, 2014

@author: xwang1
'''
import numpy as np
import scipy as sp
import sys;
from scipy.sparse.csr import csr_matrix
from scipy import linalg
from scipy.sparse.csc import csc_matrix

def proj_linf

def update_b1(x, s, b, lambda_b1, rho, miu_b):
    '''
    x, s, b, lambda_b1: csc matrix
    rho, miu_b 
    '''
    sst = (s * s.T).todense();
    sst_cond = sst + (rho + miu_b) * sp.eye(*sst.shape);
    r = (x * s.T) - lambda_b1 + rho * b;
    b1 = csc_matrix(linalg.solve(sst_cond.T, r.T, sym_pos=True));
    return b1;

# def projectLinfUnitNormBall(xSv):
#     '''
#     '''
#     uSv = cloneSv(xSv);
#     idxLst = [k for k in getSvKeys(uSv) if(abs(getSvElem(uSv, k)) > 1.0)];
#     if(len(idxLst) == 0):
#         k = max(getSvKeys(uSv), key=lambda k: abs(getSvElem(uSv, k)));
#         setSvElem(uSv, k, 1.0 if(getSvElem(uSv, k) > 0.0) else -1.0);
#     else:
#         for k in idxLst:
#             setSvElem(uSv, k, 1.0 if(getSvElem(uSv, k) > 0.0) else -1.0);
#     return uSv;
# 
# def proximityL1Operator(xSv, t):
#     '''
#     argmin_u  (t||u||_1 + 1/2 * ||u-x||^2)
#     '''
#     uSv = toSparseVec(dim=getSvLen(xSv));
#     for k in getSvKeys(xSv):
#         u = getSvElem(uSv, k);
#         if(u > t / 2.0): setSvElem(uSv, k, u - t / 2.0);
#         elif(u < -t / 2.0): setSvElem(uSv, k, u + t / 2.0);
#     return uSv;
# 
# def 

if __name__ == '__main__':
#     update_b1(x, s, b, lambda_b1, rho, miu_b);
    pass;
