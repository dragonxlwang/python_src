'''
Created on Mar 10, 2013

@author: xwang95
'''
import math;
import random;
import math;
import sys;

def psi(x):
    p = 0;
    while(x <= 7):
        p -= 1.0 / x;
        x += 1.0;
    xInv = 1 / x;
    p = p + math.log(x);
    p -= 0.5 * xInv;
    xxInv = xInv * xInv;
    p -= 1.0 / 12 * xxInv;
    xxxInv = xxInv * xxInv;
    p += 1.0 / 120 * xxxInv;
    xxxInv = xxxInv * xxInv;
    p -= 1.0 / 252 * xxxInv;
    xxxInv = xxxInv * xxInv;
    p += 1.0 / 240 * xxxInv;
    xxxInv = xxxInv * xxInv;
    p -= 1.0 / 132 * xxxInv;
    xxxInv = xxxInv * xxInv;
    p += 691.0 / 32760 * xxxInv;
    xxxInv = xxxInv * xxInv;
    p -= 1.0 / 12 * xxxInv;  
    return p ;

def lnGamma(x):
    cof =  [76.18009172947146, -86.50532032941677,
            24.01409824083091, -1.231739572450155, 
            0.1208650973866179e-2,-0.5395239384953e-5];
    tmp = x + 5.5 - (x + 0.5) * math.log(x + 5.5);
    ser = 1.000000000190015;
    for j in range(6): ser += cof[j] / (x + j);
    return (math.log(2.5066282746310005 * ser / x) - tmp);

def lnGamma2(x):
    x = x+6.0;
    z = 1.0/(x*x);
    z = ( ( (-0.000595238095238*z+0.000793650793651) *z -0.002777777777778) *z +0.083333333333333)/x;
    z = (x-0.5)*math.log(x) - x + 0.918938533204673 + z -math.log(x-1) - math.log(x-2) - math.log(x-3) - math.log(x-4) -\
                math.log(x-5) - math.log(x-6);
    return z;

def gamma(z):
    z = float(z);
    t = z + 6.5;
    x = 0.99999999999980993 \
        + 676.5203681218851 / z \
        - 1259.1392167224028 / (z + 1) \
        + 771.32342877765313 / (z + 2) \
        - 176.61502916214059 / (z + 3) \
        + 12.507343278686905 / (z + 4) \
        - 0.13857109526572012 / (z + 5) \
        + 9.9843695780195716e-6 / (z + 6) \
        + 1.5056327351493116e-7 / (z + 7);
    return math.sqrt(2) * math.sqrt(math.pi) * math.pow(t, z-0.5) * math.exp(-t) * x;


if __name__ == '__main__':
    for i in range(1, 100):
        x = random.random()*100;
        a = lnGamma(x);
        if(sys.version_info[0] + sys.version_info[0] * 0.1 >= 3.2):
            b = math.log(math.gamma(x));
            d = abs(a-b);
            print('{0}\t{1}\t{2}\t{3}\t{4}'.format(x, a, b, d, d/a));\

#     print psi(1);
#     print scipy.special.psi(2.3)
#     print scipy.special.polygamma(0, 2.3);
    pass
