'''
Created on Jun 21, 2013

@author: wangxl
'''

class VWExample:
    label = None;
    weight = None;
    tag = None;
    feaMap = None;  # dict: key->namespace, value->feature/value
    
    def __init__(self, label, weight=1.0, tag=""):
        self.label = label;
        self.weight = weight;
        self.tag = tag;
        self.feaMap = {};
        return;
    
    def insertFeature(self, n="", f=None, v=1.0):  # add namespace, feature, value
        if(n not in self.feaMap): self.feaMap[n] = {};
        self.feaMap[n][f] = v;
        return;
    
    def insertFeatureLst(self, feaLst):
        for (n, f, v) in feaLst: self.insertFeature(n, f, v);
        return;
    
    def insertFeatureDict(self, namespace, feaDict):
        for (f, v) in feaDict.iteritems(): self.insertFeature(namespace, f, v);
        return;
    
    def __getKeyValStr(self, k, v):
        if(v == 1.0): return "{0}".format(str(k));
        else: return "{0}:{1:.6f}".format(k, v);        
        
    def __str__(self):
        head = ' '.join(str(x) for x in [self.label, self.weight, self.tag]);
        fStr = ' '.join(['|{0} '.format(n) + 
                         ' '.join([self.__getKeyValStr(f, self.feaMap[n][f]) for f in self.feaMap[n]])
                         for n in self.feaMap]);
        return (head + fStr);
            
if __name__ == '__main__':
    vwEx = VWExample(1.0, 1.0, 'tag');
    n = "Says";
    for f in ["black", "with", "white", "stripes"]: vwEx.insertFeature(n, f);
    n = "OtherFeatures";
    for (f, v) in {"NumberOfLegs":4.0, "HasStripes":1.0}.items(): vwEx.insertFeature(n, f, v);
    print(vwEx);
    pass
