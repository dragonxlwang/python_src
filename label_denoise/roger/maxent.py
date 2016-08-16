import sys, os
from hashlib import md5

class maxEntExample:
# Hash the training sample (possibly with quadratic features) 
# and output it in the format of http://svn.corp.yahoo.com/view/yahoo/adsciences/users/ocetin/me-trainer/trunk/doc/README
    def __init__(self, target, wt=1.0, tag="", nb=20, quad=[]):
        self.label = int(target == 1)
        self.wt = wt
        self.tag = tag
        self.namespaces = {}
        self.nb = nb
        self.quad = quad

    def hashed_token(self,ns,token):
        m = md5(ns+token.encode('utf8'))
        return (int(m.hexdigest()[:(self.nb+3)/4],16) >> (-self.nb%4)) + 1

    def output_format(self,f,v):
        ret = None
        if v==0: 
            ret = ''
        elif v==1: 
            ret = str(f)
        else: 
            ret = '%d~%0.6g' % (f,v)
        #print ret
        return str(ret)

    def __str__(self):
        try:
            out = self.myset()
            return out
        except Exception, e:
            print >> sys.stderr, "ERROR in MaxEnt __Str__ :", e
            raise e

    def myset(self):
        out = [self.output_format(0,1)]
        for ns1, ns2 in  [(x,None) for x in self.namespaces.keys()] + self.quad:
           assert(ns1>ns2), "Failed %s(ns1) > %s(ns2) maxent.py" % (ns1, ns2)
           for f1,v1 in self.namespaces[ns1].iteritems():
                if ns2:
                    for f2,v2 in self.namespaces[ns2].iteritems():
                        try:
                            out += [self.output_format(self.hashed_token(ns1+ns2,f1+f2),v1*v2)]
                        except Exception, e1:
                            print >> sys.stderr, "ERROR in quad:", e1
                            raise e1
                else:
                    try:
                        out += [self.output_format(self.hashed_token(ns1,f1),v1)]
                    except Exception, e2:
                        print >> sys.stderr, "ERROR in non-quad:", e2
                        raise e2
                    
        strEx = ' '.join(out)
        #print >> sys.stderr, strEx
        return (';'.join([str(self.tag), str(self.label), strEx, str(self.wt)])).encode('utf8')
