import os, sys, zlib, base64, json, codecs
import logging, logging.config
from optparse import OptionParser
from maxent import maxEntExample
from itertools import combinations
from urlparse import parse_qs
from pigUDFs import parseDDInfo
from flattenPyDict import flatten
import cProfile

options = None
myLogger = None
# utf8_encode , utf8_decode , utf8_reader , utf8_writer = codecs.lookup("utf-8")

class VWExample:
    def __init__(self, target, wt=1.0, tag=""):
        self.label = target
        self.wt = wt
        self.tag = tag
        self.namespaces = {}

    def change(self, lb, wf):
        self.label = lb
        self.wt = self.wt * wf

    def __str__(self):
        head = str(self.tag)
        if self.label == self.label:
            head = ' '.join([str(self.label), str(self.wt), str(self.tag)])
        exampleParts = []
        for ns, fv in self.namespaces.iteritems():
            nf = [str(ns)]
            for f, v in fv.iteritems(): 
                if v == 0.0:
                    continue
                elif v == 1.0:
                    nf += [f]  # [f.encode("utf8")]
                else:
                    try:
                        nf += [f + ':' + "%0.6f" % v]  # [f.encode("utf8") + ':' + "%0.6f" % v]
                    except TypeError, te:
                        # myLogger.error("Could not convert to string: feature:%s, value:%s" % (f.encode("utf8"), v))
                        myLogger.error("Could not convert to string: feature:%s, value:%s" % (f, v))
                        raise te
            strNF = ' '.join(nf)
            exampleParts += [strNF]
        return (head + '|' + '   |'.join(exampleParts)).encode('utf8')  # needed here as we are writing query terms directly


# computed using src/r/orthonormalization.R
gramSchmidt = {'w1':{'d1':2.227093, 'd2':0.0, 'd3':0.0, 'w1':1.0}, 'vn':{'d1':1.987446, 'd2':1.360473, 'd3':0.0, 'vn':1.0}, 'vm':{'d1':2.315117, 'd2':0.571343, 'd3':1.831717, 'vm':1.0}}

def dictupdate(fvdict, newfv, featset=None):
    if not featset:
        fvdict.update(newfv)
    else:
        for key in newfv:
            if key in featset:
                fvdict[key] = newfv[key]

def regressorExample(id, event, fv, action_feat_dict=None, featset=None, query=None):
    pr = -1.0
    if 'probs' in event:
        pr = event['probs'][event['v']]
    else:
        pr = event['p']
    myLogger.debug("Entering regressorExample(id=%s, event=%s)" % (id, event))
    if pr <= 0.0:
        myLogger.debug("Invalid probability. Rejecting: id=%s, event=%s" % (id, event))
        return None
    else:
        myLogger.debug("Accepted with prob id=%s, event=%s" % (id, event))

    # if event['v'][0] == 'w' and pr == 1.0: # these examples are less interesting.
    #    return None

    # # Change the target if it's not NaN : Click = 1.0, 0.0 otherwise
    # if options.CTRTarget and event['r'] == event['r'] and event['r'] != 1:
    #    event['r'] = 0.0

    tags = [event['v'], str(event['r']), str(1.0 / pr), str(id)]

    vertical_weight = 1.0
    if not options.testExamples:
        vertical_weight = options.pos_vertical_weight

    if options.maxent:
        myLogger.warn("Constructing MaxEnt example")
        ex = maxEntExample(event['r'], 1.0 / pr, ','.join(tags), options.maxent, [('q', 'a')])  # quadratic features [('q', 'a')]
    else:
        myLogger.debug("Constructing VW example")
        # if user click on any vertical DD, we think it is important and
        # hence give higher weight
        # ex = VWExample(event['r'], 1.0 / pr, ';'.join(tags))
        if event['r'] == 1 and event['v'][0] == 'v':
            ex = VWExample(event['r'], vertical_weight / pr, ';'.join(tags))
        else:
            ex = VWExample(event['r'], 1.0 / pr, ';'.join(tags))
    q = {}
    l = {}
    a = {event['v']:1.0}
    rs = {}
    u = {}

    # query namespace
    if 'q' in fv:
        if 'bow' in fv['q']:
            # we do not have sequence info. so these are just co-occurrences
            qterms = fv['q']['bow'].keys()
            qterms.sort()
            for i in xrange(len(qterms)):
                qterms[i] = qterms[i].replace(':', '$COLON$')
                qterms[i] = qterms[i].replace('|', '$PIPE$')

            for ng in xrange(1, options.ngrams + 1):
                ngrams = dict([('__'.join(list(x)), 1.0) for x in combinations(qterms, ng)])
                if len(ngrams) > 0:
                    # print ngrams
                    q.update(ngrams)
                    if ng == 1:
                        l.update(ngrams)

            if options.bigram and len(query) >= 2 and len(query) <= 10:
                bigrams = {}
                for i in xrange(1, len(query)):
                    bigrams['(' + '__'.join(query[i - 1:i + 1]) + ')'] = 1.0
                l.update(bigrams)

            if options.co_occurance and len(qterms) >= 2 and len(qterms) <= 10:
                cooccur = dict([('[' + '__'.join(list(x)) + ']', 1.0) for x in combinations(qterms, 2)])
                l.update(cooccur)
        else:
            myLogger.warn("'q::bow' namespace does not exist for PV %s" % (id))
        if 'at' in fv['q']: #############
            # q.update(fv['q']['at'])
            dictupdate(q, fv['q']['at'], featset)
        else:
            myLogger.debug("'q::at' namespace does not exist for PV %s" % (id))
        if 'navqdom' in fv['q']: #############
            # q.update(fv['q']['navqdom'])
            dictupdate(q, fv['q']['navqdom'], featset)
        else:
            myLogger.debug("'q::navqdom' namespace does not exist for PV %s" % (id))
        if 'qp' in fv['q']: #############
            # q.update(fv['q']['qp'])
            dictupdate(q, fv['q']['qp'], featset)
        else:
            myLogger.debug("'q::qp' namespace does not exist for PV %s" % (id))
        # web pre-retrieval features to web event
        if 'w' in fv['q'] and event['v'][0] == 'w':#????????????????
            # q.update(fv['q']['w'])
            dictupdate(q, fv['q']['w'], featset)
        else:
            myLogger.debug("'q::w' namespace does not exist for PV %s" % (id))

        if event['v'] in fv['q']: #????????????????
            # q.update(fv['q'][event['v']])
            dictupdate(q, fv['q'][event['v']], featset)
        else:
            myLogger.debug("%s namespace does not exist in 'q' namespace for PV %s" % (event['v'], id))
        # expose local pre-retrieval features to all verticals
        if 'vl' in fv['q']:#############
            # q.update(fv['q']['vl'])
            dictupdate(q, fv['q']['vl'], featset)

        if options.gsmbackends and 'gsm_backend_calls' in fv['q']:
            q.update(fv['q']['gsm_backend_calls'])
        if options.qpbackends and 'qp_backend_calls' in fv['q']:
            q.update(fv['q']['qp_backend_calls'])

    else:
        myLogger.warn("No query features found for PV %s" % id)

    if action_feat_dict:
        vname = event['v']
        if vname in action_feat_dict:
            v_act_f = action_feat_dict[vname]["a"]
            a.update(v_act_f)

    if event['v'][0] == 'w':  # special handing for web
        if 'w' in fv:
            if 'a' in fv['w']:
                # a.update(fv['w']['a'])
                dictupdate(a, fv['w']['a'], featset)
        else:
            myLogger.debug("No web features found for PV %s" % id)
    else:
        if event['v'] in fv:
            if 'a' in fv[event['v']]:
                # a.update(fv[event['v']]['a'])
                dictupdate(a, fv[event['v']]['a'], featset)
            if 'rs' in fv[event['v']]:
                # rs.update(fv[event['v']]['rs'])
                dictupdate(rs, fv[event['v']]['rs'], featset)
        else:
            myLogger.warn("No result-set features of %s found for PV %s" % (event['v'], id))

    if 'global' in fv and 'rs' in fv['global']:
        # rs.update(fv['global']['rs'])
        dictupdate(rs, fv['global']['rs'], featset)

    if options.demog_features and 'u' in fv:
        # u.update(fv['u'])
        dictupdate(u, fv['u'], featset)

    ex.namespaces['a'] = a
    ex.namespaces['q'] = q
    ex.namespaces['l'] = l
    if rs != {}:
        ex.namespaces['rs'] = rs
    # ex.namespaces['u'] = u

    return ex

def generateTrainingExamples(id, events, fv, ts, action_feat_dict, featset=None, query=None):
    nEx = 0
    for event in events:
        if options.slots and event['s'] not in options.slots:
            myLogger.debug("PV %s: %s not in slots being considered: %s" % (id, event['s'], options.slots))
            continue
        if event['r'] == 0.0:
            if not options.considerAbandoned:  # don't generate examples for abandonments
                continue
        if options.verticals and event['v'] not in options.verticals:
            continue
        # if event['v'][0] != 'w' and event['p'] == 1.0:
        #  myLogger.warn("PV %s has p=1.0, this seems unlikely now. event = %s" % (id, event))

        ex = regressorExample(id, event, fv, action_feat_dict, featset, query)
        if ex:
            if options.considerAbandoned:
                # Abandonments are are 2 types: (i) no click on the page anywhere.  (ii) click on the links above.
                # All (ii) should be ignored. while in (i) we assign r = -1.0 and scale the
                # importance weights by x (= 0.2 to begin with)
                if event['r'] == 0.0:  # and 'cf' not in pvObj: # no click anywhere
                    myLogger.warn("changing label to -1.0 from 0.0")
                    ex.change(-1.0, float(options.abandonedWeight))
            try:
                # time-stamp needed for sorting by it
                print >> sys.stdout, str(ts), '\t', str(ex)
                nEx += 1
            except Exception, e:
                myLogger.error("Error while writing example: %s for PV %s" % (str(e), id))
    return nEx

def generateProdTestPredictions(id, events, fv, prodDDInfoStr):
    # need to restrict only to NS specific IT types for comparison
    itTypes = {'vm':['sc_wmov', 'sc_wmovo'], 'vn':['shortcut_news_dd']}
    default = {'s':'NONE', 'r':float('nan'), 'p':float('nan'), 'v' : ""}
    prodScores = {'vl':0.0, 'vm':0.0, 'vn':0.0, 'w1':0.0}
    ddinfoObj = parseDDInfo(zlib.decompress(base64.b64decode(prodDDInfoStr)))
    for i in ddinfoObj:
        if i['it'] in itTypes['vm'] and i['info']['loc'] == 'top,0,1':
            prodScores['vm'] = 1.0
        elif i['it'] in itTypes['vn'] and 'show' in i['info'] and i['info']['show'] == 'true' and i['info']['modpos'] == '0':
            prodScores['vn'] = 1.0
    if sum(prodScores.values()) > 1:
        myLogger.warn("Ignoring PV %s. Looks like > 1 DD to be slotted at s1.1. ProdScores: %s" % (id, prodScores))
        return 0
    elif sum(prodScores.values()) == 0:
        prodScores['w1'] = 1.0  # consider 'w1' at s1.1 if no DD is slotted at there
    explorationEvent = None
    for e in events:
        if e['s'] == 's1.1':
            explorationEvent = e
            break
    if not explorationEvent:
        myLogger.warn("Ignoring PV %s. No s1.1 in events %s" % (events, id))
        return 0
    nEx = 0
    outLines = []
    for vert in ['vn', 'vl', 'vm', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10']:
        if options.verticals and vert not in options.verticals:
            myLogger.debug("PV %s: %s not in verticals being considered: %s" % (id, vert, options.verticals))
            continue
        if vert[0] != 'w' and (vert not in fv or 'rs' not in fv[vert]):  # if there are no resultset features; no need to generate test example
            myLogger.warn("PV %s: %s does not have resultset features" % (id, vert))
            continue
        event = default
        event['v'] = vert
        if vert == explorationEvent['v']:
            event = explorationEvent
        pr = event['probs'][event['v']] if 'probs' in event else event['p']
        if pr <= 0.0:
            continue
        tags = [event['v'], str(event['r']), str(1.0 / pr), str(id)]
        outLines += [' '.join([str(prodScores[vert]), ';'.join(tags)])]
    print >> sys.stdout, '\n'.join(outLines)
    return len(outLines)

def generateProbTestExamples(id, events, fv, ts, action_feat_dict, featset=None, query=None):
    default = {'s':'NONE', 'r':float('nan'), 'p':float('nan'), 'v' : ""}
    nEx = 0
    event = None

    for event in events:

        if options.slots and event['s'] not in options.slots:
            myLogger.debug("PV %s: %s not in slots being considered: %s" % (id, event['s'], options.slots))
            continue

        for vert in ['vn', 'vl', 'vs', 'vm', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10']:
            if options.verticals and vert not in options.verticals:
                myLogger.debug("PV %s: %s not in verticals being considered: %s" % (id, vert, options.verticals))
                continue
            if vert[0] != 'w' and (vert not in fv or 'rs' not in fv[vert]):  # if there are no resultset features; no need to generate test example
                myLogger.debug("PV %s: %s does not have resultset features" % (id, vert))
                continue

            if not vert in event['probs']:
                continue

            this_event = event.copy()

            this_event['p'] = event['probs'][vert] if event['probs'][vert] > 0 else float('nan')
            this_event['r'] = event['r'] if event['v'] == vert else float('nan')
            this_event['v'] = vert

            myLogger.debug("Generating example for %s for PV %s" % (vert, id))
            ex = regressorExample(id, this_event, fv, action_feat_dict, featset=None, query=query)
            if ex:
                try:
                    print >> sys.stdout, str(ts), '\t', str(ex)
                    nEx += 1
                except Exception, e:
                    myLogger.error("Error while writing example: %s for PV %s" % (str(e), id))
    return nEx

def addFeature(dct, names):
    n = names.pop(0)
    if len(names) > 1:
        dct[n] = addFeature(dct.get(n, {}), names)
    elif len(names) == 1:
        dct[n] = dct.get(n, []) + [names.pop(0)]
    return dct

# if a feature namespace is not specified all features from it are accepted
# otherwise only the features in acceptFV are kept.
def filterFeatures(fv, acceptFV):
    for k, v in fv.iteritems():
        if k in acceptFV and type(acceptFV[k]) == type({}) and type(v) == type({}):
            fv[k] = filterFeatures(v, acceptFV[k])
        elif k in acceptFV and type(acceptFV[k]) == type([]):
            allf = fv[k]
            fv[k] = dict([(i, allf[i]) for i in acceptFV[k] if i in allf])
    return fv

def getQueryBoWVIP065(queryStr):
    bow, bigrams = {}, {}
    # nsinfo = json.loads(zlib.decompress(base64.b64decode(nsinfoStr)))
    # qpQuery = None
    # for i in nsinfo['backendcalls']:
    #   if i["source"] == "QP":
    #        qpQuery = i["query"]
    #        break
    if queryStr:
        # qpArgs = parse_qs(qpQuery)
        # query = qpArgs['rq'][0] if 'rq' in qpArgs else qpArgs['uq'][0]
        # qterms = utf8_decode(query)[0].split()
        qterms = queryStr.split()
        for j in qterms:
            bow[j] = 1.0
        if len(qterms) > 1:
            for k in xrange(1, len(qterms)):
                bigrams['__'.join([qterms[k - 1], qterms[k]])] = 1.0
    return (bow, bigrams)

def run():
    acceptFV = None
    nPV = 0
    nEx = 0

    itTypes = {'vm':['sc_wmov', 'sc_wmovo'], 'vn':['shortcut_news_dd']}
    itList = itTypes['vm'] + itTypes['vn']

    gsm_backend_mapper = {'NewsDD': 'gsm_news', 'MovieDetails':'gsm_movie', 'Local': 'gsm_local'}

    if options.verticals:  # "w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,vm,vn"
        options.verticals = options.verticals.split(',')
    if options.slots:  # None
        options.slots = options.slots.split(',')
    if options.CTRTarget:  # False
        myLogger.info("Setting click=1.0 and everything else to 0.0")
    if options.fwhite:  # None
        acceptFV = {}
        with open(options.fwhite, 'rb') as fin:
            for line in fin:
                addFeature(acceptFV, line.strip().split('::'))
        myLogger.info("Accept list:%s", str(acceptFV))
    if options.action_fname:  # "action-features.normalized.20120925.pruned.json.txt"
        action_f = open(options.action_fname)
        action_feat_dict = {}
        for line in action_f:
            action_feat_dict.update(json.loads(line))
    else:
        action_feat_dict = None

    featset = None
    if options.fstats != None:  # None
        featset = set()
        fin = open(options.fstats, 'rb')
        for line in fin:
            featobj = json.loads(line)
            if featobj["status"] != "prod":
                continue
            fname = featobj["name"].split("::")[-1]
            featset.add(fname)

    for line in sys.stdin:
        try:
            pvObj = eval(line.strip().split('\x01')[-1])
            #===================================================================
            # for k, v in pvObj.iteritems():
            #     print('{0} <==> {1}'.format(k, v));
            # return;
            #===================================================================
        
        except SyntaxError:
            myLogger.error("Wrong input format:%s" % (line))
            continue

        nPV += 1

        if options.prodPreds:
            nEx += generateProdTestPredictions(pvObj['id'], pvObj['e'], json.loads(zlib.decompress(base64.b64decode(pvObj['fv']))), pvObj['nsddo'])
        else:
            strFV = zlib.decompress(base64.b64decode(pvObj['fv']))  # .decode("utf8")
            myLogger.debug("Processing PV %s with FV %s" % (pvObj['id'], pvObj['fv']))
            NaN = float('nan')
            fv = eval(strFV)    #fv: feature vector in dictionary
            #===================================================================
            # for k, v in fv.iteritems():
            #     print('{0} <==> {1}'.format(k, v));
            # return;
            #===================================================================
        
            if acceptFV: #None
                fv = filterFeatures(fv, acceptFV)
                myLogger.debug("Filtered FV: %s", str(fv))
            if 'q' in fv and 'bow' not in fv['q']:
                bow, bigrams = getQueryBoWVIP065(pvObj['nquery'])   #get BagOfWords and bigrams
                fv['q']['bow'] = bow
                if options.ngrams == 2: #default 1
                    fv['q']['bigrams'] = bigrams

            if options.gsmbackends and 'gsm_backend_calls' in pvObj:    # global service management 
                fv['q']['gsm_backend_calls'] = {}
                gsm_backend_calls = json.loads(zlib.decompress(base64.b64decode(pvObj['gsm_backend_calls'])))
                for key in gsm_backend_calls.keys():
                    if key in gsm_backend_mapper:
                        fv['q']['gsm_backend_calls'].update({gsm_backend_mapper[key.strip()]:1.0})
                    else:
                        fv['q']['gsm_backend_calls'].update({'gsm_' + key.strip():1.0})

            if options.qpbackends and 'qp_backend_calls' in pvObj:  # query profile
                fv['q']['qp_backend_calls'] = {}
                qp_backend_calls = json.loads(zlib.decompress(base64.b64decode(pvObj['qp_backend_calls'])))

                for k, v in qp_backend_calls.iteritems():
                    fv['q']['qp_backend_calls'].update({'qp_' + k:1.0})

            query = pvObj['q'].lower().rstrip().split() # normalize query
            for i in xrange(len(query)):
                query[i] = unicode(query[i].replace(':', '$COLON$'))
                query[i] = unicode(query[i].replace('|', '$PIPE$'))

            if not options.testExamples:    #False
                nEx += generateTrainingExamples(pvObj['id'], pvObj['e'], fv, pvObj['ts'], action_feat_dict, featset, query)
            elif options.jsonOut:
                flatEx = flatten(fv, {}, "")
                print >> sys.stdout, '\t'.join([pvObj['id'], json.dumps(flatEx, separators=",:")])
                nEx += 1
            else:
                nEx += generateProbTestExamples(pvObj['id'], pvObj['e'], fv, pvObj['ts'], action_feat_dict, featset, query)
        if nPV % 1000 == 0:
            myLogger.info("Processed %d page views, generated %d examples" % (nPV, nEx))
    myLogger.info("Processed %d page views, generated %d examples" % (nPV, nEx))

if __name__ == '__main__':
    # sys.setdefaultencoding('utf8') can't do this as it is removed from sys namespace after its initial use by site module
    logging.config.fileConfig("pylogging.cfg")
    logging.info(' '.join(sys.argv))
    myLogger = logging.getLogger("vwExampleGenerator")

    parser = OptionParser("")
    parser.add_option("-0", "--abandoned", help="consider abandoned results", dest="considerAbandoned" , action="store_true", default=False)
    parser.add_option("-w", "--abandoned_weight", help="if considering abandoned results during training, which multiplicative factor (between 0 and 1) to use for their importance weight", dest="abandonedWeight", default=1.0)
    parser.add_option("-t", "--test", help="Generate test examples", dest="testExamples", action="store_true", default=False)
    parser.add_option("-m", "--maxent", help="Number of bits for hashed maxent output (Default = 0: use VW)", dest="maxent", type="int", default=0)
    parser.add_option("-p", "--namespaces", help="Namespaces to consider", dest="namespaces", default="a,q,rs,u")
    parser.add_option("-v", "--verticals", help="Verticals to consider", dest="verticals", default="w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,vm,vn")
    parser.add_option("-s", "--slots", help="List of slots to consider (restrict to s1.1 if using the IPS policy evaluator)", dest="slots", default=None)
    parser.add_option("--ctr-target", help="Use click=1 and set other labels to 0", dest="CTRTarget", action="store_true", default=False)
    parser.add_option("-n", "--ngrams", dest="ngrams", help="Generate ngrams upto n=N", metavar="N", default=1, type="int")
    parser.add_option("--bigram", dest="bigram", help="Generate bigram features", action="store_true", default=True)
    parser.add_option("--co-occurance", dest="co_occurance", help="Generate co_occurance features", action="store_true", default=True)
    parser.add_option("--keep-features", dest="fwhite", help="Files containing whitelist of features", default=None)
    parser.add_option("--use-qp-backends", dest="qpbackends", help="Use backend call features (QP) to insert into the q namespace", action="store_true", default=False)
    parser.add_option("--use-gsm-backends", dest="gsmbackends", help="Use backend call features (GSM) to insert into the q namespace", action="store_true", default=False)
    parser.add_option("--demog-features", dest="demog_features", help="Use demographics", action="store_true", default=False)
    parser.add_option("--prod-preds", dest="prodPreds", help="Generate VW style predictions from production actions. Requires a production DDInfo field ('nsddo')", default=False, action="store_true")
    parser.add_option("--action-features", dest="action_fname", help="action features file (normalized)", default="action-features.normalized.20120925.pruned.json.txt")
    parser.add_option("--json-out", dest="jsonOut", help="", default=False, action="store_true")
    parser.add_option("--pos-vertical-weight", dest="pos_vertical_weight", help="Give a higher importance weight on vertical which receives click", default=2.0)
    parser.add_option("--fstats", dest="fstats", help="Feature meta data file (if given, use only features in prod status)", default=None)
    options, args = parser.parse_args()
    run()
    # cProfile.run("run()")

