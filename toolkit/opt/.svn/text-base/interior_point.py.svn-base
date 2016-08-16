'''
Created on Apr 23, 2014

@author: xwang95
'''
from toolkit.num.algebra import getMatRow, getMatCol, subVecVec, addVecLst, \
    mulMatLst, mulMatVec, minusVec, addVecVec, linSolveSymIdfMat, mulMatMat, \
    _luDecomp, linSolve, dotVecVec, mulNumVec, transposeMat, sizeMat, \
    getVecInfNorm, printMat
from toolkit.num.arithmetic import _eps

class IpmLp(object):
    '''
    classdocs
    '''
    #------------------------------------------------------------- problem scale
    n = None;
    m = None;
    #---------------------------------------------------------------- parameters
    #--------------------------------------------- constant after initialization
    cVec = None;
    bVec = None;
    aMat = None;
    aMatTranspose = None;
    
    xVec = None;
    lambdaVec = None;
    sVec = None;
    
    eta = None;
    #-------------------------------------------------------- running parameters
    iterNum = None;
    
    def __init__(self, cVec, bVec, aMat=None, aMatTranspose=None,
                 initStartingPoint=None):
        self.eta = 0.9;
        if(aMat is not None): self.aMat = aMat;
        if(aMatTranspose is not None): self.aMatTranspose = aMatTranspose;
        if(self.aMat is None): self.aMat = transposeMat(self.aMatTranspose);
        if(self.aMatTranspose is None):
            self.aMatTranspose = transposeMat(self.aMat);
        self.cVec = cVec;
        self.bVec = bVec;
        (self.n, self.m) = sizeMat(self.aMatTranspose);
        if(initStartingPoint is not None):
            (self.xVec, self.lambdaVec, self.sVec) = initStartingPoint;
        else: self.initialization();
        return;
    
    def initialization(self):
        aaMat = mulMatMat(self.aMat, self.aMatTranspose);
        lu = _luDecomp(aaMat);
        xVec = mulMatVec(self.aMatTranspose, linSolve(vec=self.bVec, lu=lu));
        lambdaVec = linSolve(vec=mulMatVec(self.aMat, self.cVec), lu=lu);
        sVec = subVecVec(self.cVec, mulMatVec(self.aMatTranspose, lambdaVec));
        deltaX = max(-1.5 * min(xVec), 0.0);
        deltaS = max(-1.5 * min(sVec), 0.0);
        for i in range(self.n): xVec[i] += deltaX;
        for i in range(self.n): sVec[i] += deltaS;
        deltaX = dotVecVec(xVec, sVec) / (2 * sum(sVec));
        deltaS = dotVecVec(xVec, sVec) / (2 * sum(xVec));
        for i in range(self.n): xVec[i] += deltaX;
        for i in range(self.n): sVec[i] += deltaS;
        self.xVec = xVec;
        self.lambdaVec = lambdaVec;
        self.sVec = sVec;
        return;
    
    def linSolveKKT(self, rcVec, rbVec, rxsVec):
        xDsVec = [self.xVec[i] / self.sVec[i] for i in range(self.n)];
        adaMat = [[sum([self.aMat[i][k] * self.aMat[j][k] * xDsVec[k] 
                        for k in range(self.n)])
                   for j in range(self.m)] 
                  for i in range(self.m)];
        dlambdaVec = linSolveSymIdfMat(mat=adaMat,
                                       vec=subVecVec(mulMatVec(self.aMat,
                                                [rxsVec[i] / self.sVec[i] 
                                                 - rcVec[i] * xDsVec[i] 
                                                 for i in range(self.n)]),
                                                     rbVec));
        dsVec = minusVec(addVecVec(rcVec,
                                   mulMatVec(self.aMatTranspose, dlambdaVec)));
        dxVec = [-rxsVec[i] / self.sVec[i] - xDsVec[i] * dsVec[i] 
                 for i in range(self.n)];
        return (dxVec, dlambdaVec, dsVec);
    
    def iter(self, eps=1e-6):
        #----------------------------------------------------------- affine step
        rcVec = addVecVec(mulMatVec(self.aMatTranspose, self.lambdaVec),
                          subVecVec(self.sVec, self.cVec));
        rbVec = subVecVec(mulMatVec(self.aMat, self.xVec), self.bVec);
        rxsVec = [self.xVec[i] * self.sVec[i] for i in range(self.n)];
        ef = max(getVecInfNorm(rcVec),
                 getVecInfNorm(rbVec),
                 getVecInfNorm(rxsVec));
        if(ef <= eps): return 1;
        (dxVec, dlambdaVec, dsVec) = self.linSolveKKT(rcVec, rbVec, rxsVec);
        alphaAffPri = min(1.0, min([-self.xVec[i] / dxVec[i] 
                           for i in range(self.n) if(dxVec[i] < 0.0)]));
        alphaAffDual = min(1.0, min([-self.sVec[i] / dsVec[i]
                            for i in range(self.n) if(dsVec[i] < 0.0)]));
        miuAff = dotVecVec(addVecVec(self.xVec,
                                     mulNumVec(alphaAffPri, dxVec)),
                           addVecVec(self.sVec,
                                     mulNumVec(alphaAffDual, dsVec))) / self.n;
        miu = dotVecVec(self.xVec, self.sVec) / self.n;
        sigma = min(1.0, (miuAff / miu) ** 3);
        #----------------------------------------------- corrector and centering
        rxsVec = [self.xVec[i] * self.sVec[i] 
                  + dxVec[i] * dsVec[i] 
                  - sigma * miu 
                  for i in range(self.n)];
        (dxVec, dlambdaVec, dsVec) = self.linSolveKKT(rcVec, rbVec, rxsVec);
        alphaPri = min(1.0, self.eta * min([-self.xVec[i] / dxVec[i] 
                        for i in range(self.n) if(dxVec[i] < 0.0)]));
        alphaDual = min(1.0, self.eta * min([-self.sVec[i] / dsVec[i]
                        for i in range(self.n) if(dsVec[i] < 0.0)]));
        #---------------------------------------------------------------- update
        self.eta = 0.5 * (self.eta + 1.0);
        self.xVec = addVecVec(self.xVec,
                              mulNumVec(alphaPri, dxVec));
        self.lambdaVec = addVecVec(self.lambdaVec,
                                   mulNumVec(alphaDual, dlambdaVec));
        self.sVec = addVecVec(self.sVec,
                              mulNumVec(alphaDual, dsVec));
        return 0;
    
    def solve(self):
        self.iterNum = 0;
        while(True):
            self.iterNum += 1;
            flag = self.iter(eps=1e-6);
            if(flag == 1): break;
        print self.iterNum;
        self.debug();
        return;
    
    def debug(self):
        f = dotVecVec(self.xVec, self.cVec);
        printMat(self.xVec, 'xVec');
        printMat(f, 'f');
        printMat(subVecVec(self.bVec,
                           mulMatVec(self.aMat, self.xVec)),
                 'z_feasibility');
        print('iter={0}'.format(self.iterNum));
        return;
    
if __name__ == '__main__':
    cVec = [-2, -3, -4, 0, 0];
    bVec = [10, 15];
    aMatTranspose = [[3, 2],
                     [2, 5],
                     [1, 3],
                     [1, 0],
                     [0, 1]];
    initBIdxLst = None;
    
    cVec = [-1, -2, 1, 0, 0, 0];
    bVec = [14, 28, 30];
    aMatTranspose = [[2, 4, 2],
                     [1, 2, 5],
                     [1, 3, 5],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = [3, 4, 5];
    
    cVec = [-4, -6, 0, 0, 0];
    bVec = [11, 27, 90];
    aMatTranspose = [[-1, 1, 2],
                     [1, 1, 5],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = [2, 3, 4];
    
    cVec = [-2, 1, -2, 0, 0, 0];
    bVec = [10, 20, 5];
    aMatTranspose = [[2, 1, 0],
                     [1, 2, 1],
                     [0, -2, 2],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = [3, 4, 5];
    
    cVec = [-3, -2, -1, 0, 0];
    bVec = [30, 60, 40];
    aMatTranspose = [[4, 2, 1],
                     [1, 3, 2],
                     [1, 1, 3],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = None;
    ipmlp = IpmLp(cVec, bVec, aMatTranspose=aMatTranspose,
                 initStartingPoint=None);
    ipmlp.solve();
    
    cVec = [-11, -16, -15, 0, 0, 0];
    bVec = [12000, 4600, 2400];
    aMatTranspose = [[1, 2.0 / 3.0, 0.5],
                     [2, 2.0 / 3.0, 1.0 / 3.0],
                     [1.5, 1, 0.5],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]];
    initBIdxLst = None;  # [3, 4, 5];
    
    cVec = [-100000, -40000, -18000, 0, 0, 0, 0];
    bVec = [18200, 10, 0, 0];
    aMatTranspose = [[2000, 0, -0.5, -0.9],
                     [600, 1, -0.5, 0.1],
                     [300, 0, 0.5, 0.1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]];
    initBIdxLst = None;  # [3, 4, 5];
    
    pass
