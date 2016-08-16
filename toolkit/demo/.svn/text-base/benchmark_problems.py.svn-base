'''
Created on Mar 10, 2014

@author: xwang95
'''
from toolkit.num.algebra import eye, mulNumMat, randomVec, printMat, subVecVec, \
    getVecNorm, zeroes, mulMatVec, subMatMat
from toolkit.num.calculus import gradientFunc, mulHessianVecApprox, hessianFunc


def sphere(n):
    def fFunc(xVec): return sum([x ** 2 for x in xVec]);
    def gFunc(xVec): return [2.0 * x for x in xVec];
    def hFunc(xVec): return mulNumMat(2.0, eye(n));
    return (fFunc, gFunc, hFunc);

def schwefel(n):
    def fFunc(xVec): return sum([sum([xVec[j] for j in range(i + 1)]) ** 2 
                                 for i in range(n)]);
    def gFunc(xVec): return gradientFunc(fFunc, xVec);        
    return (fFunc, gFunc);

def rosenbrock(n):  # n>=3
    def fFunc(xVec): return sum([100.0 * ((xVec[i + 1] - xVec[i] ** 2) ** 2) 
                                 + (xVec[i] - 1) ** 2 for i in range(n - 1)]);
    def gFunc(xVec): 
        g = [0.0 for i in range(n)];
        for i in range(n):
            if(i == 0):
                g[0] = 200.0 * (-2.0 * xVec[1] * xVec[0] + 
                                2.0 * (xVec[0] ** 3)) + 2.0 * (xVec[0] - 1.0);
            elif(i == n - 1):
                g[n - 1] = 200.0 * (xVec[n - 1] - (xVec[n - 2] ** 2));
            else:
                g[i] = 200.0 * (-2.0 * xVec[i + 1] * xVec[i] + 
                                2.0 * (xVec[i] ** 3) + xVec[i] - 
                                (xVec[i - 1] ** 2)) + 2.0 * (xVec[i] - 1.0);
        return g;
    def hFunc(xVec):
        h = zeroes(n, n); 
        for i in range(n):
            for j in range(i, n):
                if(j == i):
                    if(i == n - 1): h[n - 1][n - 1] = 200.0;
                    elif(i == 0): h[i][i] = 200.0 * (-2.0 * xVec[i + 1] 
                                                     + 6 * (xVec[i] ** 2)) + 2.0;
                    else: h[i][i] = 200.0 * (-2.0 * xVec[i + 1] 
                                             + 6 * (xVec[i] ** 2)) + 202.0;
                elif(j == i + 1): 
                    h[i][j] = 200.0 * (-2.0 * xVec[i]);
                    h[j][i] = h[i][j];
                elif(j == i - 1): 
                    h[i][j] = 200.0 * (-2.0 * xVec[i - 1]);
                    h[j][i] = h[i][j];
        return h;
    return (fFunc, gFunc, hFunc);

if __name__ == '__main__':
    n = 300;
    (f, g, h) = rosenbrock(n);
    x = randomVec(n, -2.048, 2.048);
    y = randomVec(n, -3.0, 3.0);
    g1 = g(x);
    g2 = gradientFunc(f, x, h=1.0);
    z = subVecVec(g1, g2);
#     printMat(z, decor='e');
#     printMat(getVecNorm(z));
#     printMat(subMatMat(h(x), hessianFunc(f, x, h=1e-6)), 'h');
#     hy1 = mulMatVec(h(x), y);
#     hy2 = mulHessianVecApprox(gFunc=g, xVec=x, vec=y, h=1e-4);
#     hy3 = mulMatVec(hessianFunc(f, x, h=1e-6), y);
#     z = subVecVec(hy1, hy2);
#     printMat(z, decor='e');
#     printMat(getVecNorm(z));    
#     z = subVecVec(hy1, hy3);
#     printMat(z, decor='e');
#     printMat(getVecNorm(z));    
    pass;
