'''
Created on Aug 10, 2014

@author: xwang1
'''
from argparse import ArgumentParser
from linkedin.snnmf2 import SNNMF2
import os;

def parse():
    parser = ArgumentParser();
    parser.add_argument("--task", action="store", dest="task");
    parser.add_argument("--data", action="store", dest="data");
    parser.add_argument("--bTrSm", action="store", dest="bTrSm", default=None);
    parser.add_argument("--lambdaS", action="store", dest="lambdaS",
                        default=1e-4, type=float);
    parser.add_argument("--lambdaB", action="store", dest="lambdaB",
                        default=1e-4, type=float);
    parser.add_argument("--miuS", action="store", dest="miuS",
                        default=1e-6, type=float);
    parser.add_argument("--miuB", action="store", dest="miuB",
                        default=1e-6, type=float);
    parser.add_argument("--lbL1ReweightMethod", action="store",
                        dest="lbL1ReweightMethod", default=None);
    parser.add_argument("--lbL2ReweightMethod", action="store",
                        dest="lbL2ReweightMethod", default=None);
    parser.add_argument("--lbAddInfo", action="store",
                        dest="lbAddInfo", default=None);
    parser.add_argument("--affix", action="store", dest="affix", default=None);
    parser.add_argument("--startingFrom", action="store", dest="startingFrom",
                    default=None, help="(task, iterNum) decode, learnBasis");
    args = parser.parse_args();
    
    print("task                   = {0}".format(args.task));
    print("data                   = {0}".format(args.data));
    print("bTrSm                  = {0}".format(args.bTrSm));
    print("lambdaS                = {0}".format(args.lambdaS));
    print("lambdaB                = {0}".format(args.lambdaB));
    print("miuS                   = {0}".format(args.miuS));
    print("miuB                   = {0}".format(args.miuB));
    print("lbL1ReweightMethod     = {0}".format(args.lbL1ReweightMethod));
    print("lbL2ReweightMethod     = {0}".format(args.lbL2ReweightMethod));
    print("lbAddInfo              = {0}".format(args.lbAddInfo));
    print("affix                  = {0}".format(args.affix));
    print("startingFrom (t, num)  = {0}".format(args.startingFrom));
    
    if(args.data == "toy"):
        dataDir = "/home/xwang1/data/snnmf/toy/" \
                  "n_50_m_1000_l_250_poiPhLen_1.8_poiSeLen_3.0";
        xSmFilePath = os.path.join(dataDir, "xSm");
        if(args.bTrSm is None):
            bTrSm = "initBTrSmRandomDense";
        elif(args.bTrSm == "initBTrSmRandomDense" or
             args.bTrSm == "initBTrSmRandomDense_500_50" or
             args.bTrSm == "bTrSm"):
            bTrSm = args.bTrSm;
        else:
            print("bTrSm can only be valued as:");
            print("      initBTrSmRandomDense");
            print("      initBTrSmRandomDense_500_50");
            print("      bTrSm");
            return;
        bTrSmFilePath = os.path.join(dataDir, bTrSm);            
    elif(args.data == "movie_review"):
        dataDir = "/home/xwang1/data/txt_sentoken";
        xSmFilePath = os.path.join(dataDir, "xSm");
        if(args.bTrSm is None):
            bTrSm = "initBTrSmRandomSparse_9000_3";
        else:
            bTrSm = args.bTrSm;
        bTrSmFilePath = os.path.join(dataDir, bTrSm);
    elif(args.data == "feedback"):
        dataDir = "/home/xwang1/data/feedback";
        xSmFilePath = "/home/xwang1/data/global.xSm";
        bTrSm = "global.bTrSm.init";
        bTrSmFilePath = "/home/xwang1/data/global.bTrSm.init";
    if(args.startingFrom is not None): startingFrom = eval(args.startingFrom);
    else: startingFrom = None;    
    learningParams = (bTrSm, args.lambdaS, args.lambdaB, args.miuS, args.miuB,
                      args.lbL1ReweightMethod, args.lbL2ReweightMethod,
                      args.lbAddInfo);
    learningParamsDecor = '_'.join([str(x) for x in learningParams]);
    dumpDir = os.path.join(dataDir, learningParamsDecor + \
                    ("_" + args.affix if(args.affix is not None) else ""));
    if(args.task == "learn"):
        print("dumpDir: {0}".format(dumpDir));
        snnmf = SNNMF2(xSmFilePath=xSmFilePath,
                       bTrSmFilePath=bTrSmFilePath,
                       lambdaS=args.lambdaS, lambdaB=args.lambdaB,
                       miuS=args.miuS, miuB=args.miuB, procNum=10,
                       outputDir=dumpDir, startingFrom=startingFrom,
                       learnBasisOptions=(args.lbL1ReweightMethod,
                                          args.lbL2ReweightMethod,
                                          args.lbAddInfo));
        snnmf.work();              
    return;

if __name__ == '__main__':
    parse();
    pass
