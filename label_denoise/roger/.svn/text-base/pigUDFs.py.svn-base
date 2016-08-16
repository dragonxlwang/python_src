#! /grid/0/gs/hod-comps/current/bin/python
# -*- coding: UTF-8 -*-

###
# Streaming UDFs for agile fedsci
# Copyright (c) 2010 Yahoo! Inc.  All rights reserved.
#
# $Id: pigUDFs.py 858 2011-08-30 18:00:22Z lamkhede $
###

import sys, os, urlparse, zlib, pickle, base64
from optparse import OptionParser

###
# DDINFO is a string that is a ^B delimited variable-length list and looks like:
#
#   ddinfo: =  <shortcut>^B<shortcut>^B<shortcut>
#
#each <shortcut> field is in turn a string that is | delimited and looks like:
#
# <shortcut> := <it>|<global_position>|<info>
#
#Where:
#
#    * <it> is the shortcut type. The type of the shortcut is currently logged at the link-level in the viewinfo#’it’ parameter.
#    * <position> is the global position of the shortcut within the search results page. Its value is currently logged at the link-level in viewinfo#’gpos’ parameter. For cases where we want to log additional <info> for a shortcut that is not shown, the <global_position> is empty. In future, should we do 2D tiling <position> can be split out to be a tuple with another delimiter (^E?).
#    * <info> is any additional info related to the given shortcut. For example, this filed may contain the outcome of the randomization slotting procedure as well as if content is retrieved from the dedicated backend. 
#
###
# Format as of May 2011 (for VIP065 bucket) :shortcut_news_dd|11|upd=20110323175551;show=true;modpos=8;src=northstar;score=-0.806449;mpos=8;;loc=slot,8,1
def parseDDInfo(ddinfo):
    parsed = []
    shortcuts = ddinfo.strip().split(ur'\u0002') # split on ^B
    for sc in shortcuts:
        fields = sc.split('|')
        itStr, gposStr, infoStr = "", "", ""
        itStr = fields[0].strip()
        if len(fields) == 2:
            infoStr = fields[1].strip()
        elif len(fields) == 3:
            gposStr = fields[1].strip()
            infoStr = fields[2].strip()
        else:
            print >> sys.stderr, "ERROR. Invalid ddinfo format:", ddinfo
            return []
        if len(gposStr) > 3: # to correct cases like sc_bz_def_i|loc=slot,4,1|phighconf=0;woeid=2375029;lhighconfs=[1,1];mlrscores=[-2.7493262076312996,-3.05743892088642];eltags=[0,0];cat=[1,96925687,business to business];cat=[2,96930312,printing & publishing];cat=[3,96931877,newspapers];count=2;radius=50;
            infoStr += gposStr + ';'
            gposStr = ""
        info = {}
        for f in infoStr.strip().split(';'): #conf=0.2000;trial=1;audit=1;content=1;pos=4;res=1
            ff = f.split('=')
            if len(ff) == 2:
                info[ff[0]] = ff[1]
        parsed += [dict([('it', itStr), ('gpos', gposStr), ('info', info)])]
    return parsed
###
#To log the post-retrieval features using ULT, a new page-level parameter similar to the DDINFO parameter would ideally need to be created. Let's call it DDPRINFO.
#
#The DDPRINFO would be a string that is a ^BB delimited variable-length list and looks like:
#
#   ddprinfo: =  <shortcut>^BB<shortcut>^BB<shortcut>
#
#where each <shortcut> field is in turn a string that is ^BE delimited and looks like:
#
# <shortcut> := <it>^BE<hits>^BE<info>
#
#Where:
#
#    * <it> is the shortcut type. The type of the shortcut is currently logged at the link-level in the viewinfo#’it’ parameter.
#    * <hits> is the number of matched documents in the dedicated backend.
#    * <info> is in turn a ^BF delimited variable-length list and looks like: 
#
#   info: =  <doc>^BF<doc>^BF<doc>
#
#where each <doc> field is in turn a string that is ^BG delimited and looks like:
#
# <doc> := <id>^BG<score>^BG<title>^BG<abstract>^BG<feat>
#
#Where:
#
#    * <id> is the document ID.
#    * <score> is the document score.
#    * <title> is the title of the document.
#    * <abstract> is the smart summary of the document.
#    * <log> is the rank logs of the document.
#    * <feat> is the summary features of the document. 
###
def parseDDPrInfo(ddprinfo, full):
    parsed = [] # list of {it, hits, info=[{id, score, title, abstract, feats={}}] }
    shortcuts = ddprinfo.strip().split(ur'\u0002B')
    #print >> sys.stderr, "Total shortcuts = ", len(shortcuts)
    for sc in shortcuts:
        fields = sc.split(ur'\u0002E')
        info = [] #info has multiple docs
        #print >> sys.stderr, "Total fields = ", len(fields)
        if len(fields) != 3: #there must be 3 fields it, hits and info
            print >> sys.stderr, 'ERROR: Could not parse, invalid format: shortcut [' + sc + '] ddprinfo [' +  ddprinfo + ']'
            return []
        for doc in fields[2].split(ur'\u0002F'):
            docFields = doc.split(ur'\u0002G')
            #print >> sys.stderr, "Total info fields", len(docFields)
            if len(docFields) >= 5:
                featDict = {}
                docFields[4] = docFields[4].strip().strip('{').strip('}')
                for feat in docFields[4].split(','):
                    name_value = feat.split(':')
                    if len(name_value) != 2:
                        print >> sys.stderr, "ERROR in parsing feature:", feat, "in", docFields[4].encode("UTF-8")
                        continue
                    featDict[name_value[0].strip('"')] = name_value[1]
                if full: # include title and abstract
                    info += [dict([('id', docFields[0]), ('score', docFields[1]), ('title', docFields[2]), ('abstract', docFields[3]), ('feat', featDict)])]
                else:
                    info += [dict([('id', docFields[0]), ('score', docFields[1]), ('feat', featDict) ])]
        parsed += [dict([ ('it', fields[0]), ('hits', fields[1]), ('info', info) ])]
    return parsed

###
# a simple function to demostrate
#
# Define the UDF in Pig as
# define simpleUDF `fedsci/pigUDFs.py -f simpleUDF REC#` input(stdin using PigStorage('\t')) output(stdout using PigStorage('\t'));
# 
# and run as
# y = stream x through simpleUDF;
#
# This should give a list of records like (REC#1,...original fields...). And y can be further used as any other Pig relation.
###

def simpleUDF(args):
    prefix = "record#"
    if len(args) > 0:
        prefix = args[0]
    lineNum = 1
    for line in sys.stdin:
        print >> sys.stdout, prefix + str(lineNum) + '\t' + line.strip()
        lineNum += 1
    return

# define getDDData `fedsci/pigUDFs.py -f getDDDataUDF` input(stdin using PigStorage('\t')) output(stdout using PigStorage('\t'));
def getDDDataUDF(args):
    for line in sys.stdin:
        fields = line.decode("UTF-8").strip('\n').split('\t')
        ddprinfo = []
        ddinfo = []
        if len(fields) != 3:
            print >> sys.stderr, "ERROR: Incorrect number of columns in data"
            print >> sys.stdout,  'ERROR' + '\t' + fields[0] 
            return
        full = True
        if len(args) > 0 and args[0] == 'False':
            full = False
        if not full:
            ddinfo = str(parseDDInfo(fields[1])) 
            ddprinfo = str(parseDDPrInfo(fields[2]), full)
            output = '\t'.join([fields[0].encode("UTF-8"), ddinfo.encode("UTF-8"), ddprinfo.encode("UTF-8")])
        else:
            ddprinfo = base64.b64encode(zlib.compress(pickle.dumps(parseDDPrInfo(fields[2], full)))) #.decode("UTF-8")
            ddinfo = base64.b64encode(zlib.compress(pickle.dumps(parseDDInfo(fields[1])))) #.decode("UTF-8")
            output = '\t'.join([fields[0].encode("UTF-8"), ddinfo.encode("UTF-8"), ddprinfo.encode("UTF-8")])
        print >> sys.stdout, output
    return

# define getDDPrInfo `fedsci/pigUDFs.py -f getDDPrInfoUDF n pairs` input(stdin using PigStorage('\t')) output(stdout using PigStorage('\t'));
def getDDPrInfoUDF(args):
    if len(args) < 2:
        print >> sys.stderr, "getDDPrInfoUDF() requires at least 2 arguments, ddprinfo column number and output format"
        sys.exit(1)
    col = int(args[0]) - 1 # given column is 1-based. 
    outputFormat = args[1]
    print >> sys.stderr, "Applying getDDPrInfoUDF to the data, col =", col, "output =", args[1]
    for line in sys.stdin:
        fields = line.decode('UTF-8').strip('\n').split('\t')
        ddprinfo = []
        if len(fields) > col and len(fields[col].strip()) > 1:
            ddprinfo = parseDDPrInfo(fields[col])
        if outputFormat == "all":
            print >> sys.stdout, line.strip('\n').encode('UTF-8') + '\t' +  str(ddprinfo).encode('UTF-8')
        elif outputFormat == "pair":
            print >> sys.stdout, fields[col].encode('UTF-8') + '\t' + str(ddprinfo).encode('UTF-8')
#        elif outputFormat == "pair" and len(fields) <= col:
#            print >> sys.stdout, "" + '\t' + str(ddprinfo).encode('UTF-8')
        else:
            print >> sys.stdout, fields[0] + '\t' + str(ddprinfo)
    
def getDDInfoUDF(args):
    if len(args) < 2:
        print >> sys.stderr, "getDDInfoUDF() requires at least 2 arguments, ddinfo column number and output format"
        sys.exit(1)
    col = int(args[0]) - 1 # given column is 1-based. 
    outputFormat = args[1]
    print >> sys.stderr, "Applying getDDInfoUDF to the data, col =", col, "output =", args[1]
    for line in sys.stdin:
        fields = line.strip('\n').split('\t')
        ddinfo = []
        if len(fields) > col:
            ddinfo = parseDDInfo(fields[col])
        if outputFormat == "all":
            print >> sys.stdout, line.strip() + '\t' +  str(ddinfo)
        elif outputFormat == "pair" and len(fields) > col:
            print >> sys.stdout, fields[col] + '\t' + str(ddprinfo).encode('UTF-8')
        elif outputFormat == "pair" and len(fields) <= col:
            print >> sys.stdout, "" + '\t' + str(ddinfo)
        else:
            print >> sys.stdout, str(ddinfo)

def getHostUDF(args):
    """
    define getHostUDF `fedsci/pigUDFs.py -f getHostUDF` input(stdin using PigStorage('\t')) output(stdout using PigStorage('\t'));
    x = load '/grid/0/tmp/lamkhede/simple.test' using PigStorage('\t');
    y = foreach x generate $6 as url;                                                                              
    z = stream y through getHostUDF; 
    """
    print >> sys.stderr, "PYTHON VERSION:", sys.version
    for line in sys.stdin:
        url = line.strip()
        if len(url) > 0:
            if  url.find("://") == -1 or url.find("://") > 5:
                url = "http://" + url
            parsedURL = urlparse.urlsplit(url); 
            print >> sys.stdout, '\t'.join([url, str(parsedURL.hostname)]) # requires python 2.5
    return

if __name__ == '__main__':
    parser = OptionParser(usage="%prog -f func [options]", version="%prog 1.0")
    parser.add_option("-f", "--function", dest="func", metavar="FUNC", help="Function to invoke", default=None)

    (options, args) = parser.parse_args()
    if options.func not in ['simpleUDF', 'getDDInfoUDF','getDDPrInfoUDF', 'getHostUDF', 'getDDDataUDF']:
        print >> sys.stderr, "Function does not exist:", options.func
        sys.exit(1)
    else:
        eval(options.func + '(args)') # we also pass all args that may have been passed for the specific UDF
