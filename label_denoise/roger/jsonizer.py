#!/usr/bin/env python

import os, sys, re, math
import json, zlib, base64
from processNSInfo import processNSInfo
from normalizeFeatures import serializeFV
from parseqp import parseInputQP
from parseyfed import getReturningProviders

last_id = None
id2Features = None
record_set = []
nPVs = 0
nOutPVs = 0
noNSInfo = 0

n_total = 0
n_misses = 0

def qnormalize(oquery,spquery):
    if spquery.strip()!="":
        query = spquery.lower()
    else:
        query = oquery.lower()
    query = re.sub('\s+',' ', query.strip())
    return query

def process_recordset(record_set):
    global nPVs
    global nOutPVs
    global noNSInfo
    global n_total
    global n_misses
    query, qlas, qpddi, ddi, yfed, nsinfo, nsddo, gender, birth, spquery, webfeat = None, None, None, None, None, None, None, None, None, None, None
    events = []
    jobj = {}
    nPVs += 1
    if len(record_set) == 0 :
        return None

    for r in record_set:
            #fields
            #1:srcpvid, 2:seed, 3:bc, 4:ts, 5:action, 6:reward, 7:weight, 8:query, 9:qlas, 10:qpddi, 11:ddi, 12:yfed, 13:nsinfo, 14:nsddo, 15:gender, 16:birth_year, 17:sp, 18:webfeat
            try:
                srcpvid, seed, bc, ts, action, reward, weight, query, qlas,\
                qpddi, ddi, yfed, nsinfo, nsddo, gender, birth_year, spquery, webfeat = r
            except ValueError:
                print r
                # malformed example
                return
            vertical, slot = action.split('_',1)
            slotnum = float(slot)

            # ignore all events beyond slot 11, so we are going to look at only events
            # below it.
            if slotnum >=12 :
                continue

            s = 's%s' % slot
            v = None
            if vertical.startswith('news') :
                v = 'vn'
            elif vertical.startswith('movie') :
                v = 'vm'
            elif vertical.startswith('local') :
                v = 'vl'
            elif vertical.startswith('product') :
                v = 'vs'
            elif vertical.startswith('web') :
                # we don't want slots 1.1, so we are going to see if the
                # if its 1.1 and truncate it to 1
                if slotnum > 11 :
                    continue

                v = 'w%g' % int(slotnum)

            reward = float(reward)
            if reward > 1.0:
                reward = 1.0
            rw = {
                's' : s,
                'v' : v,
                'p' : 0.0,
                'r' : reward
            }

            events.append((slotnum, rw))

    jobj['id'] = srcpvid
    jobj['sd'] = seed[:8]
    jobj['bc'] = bc
    jobj['ts'] = long(ts)
    jobj['q'] = query if query else ''
    jobj['qa'] = base64.b64encode(zlib.compress(qlas))
    jobj['yf'] = base64.b64encode(zlib.compress(yfed))
    jobj['qpddi'] = qpddi if qpddi else ''
    jobj['ddi'] = base64.b64encode(zlib.compress(ddi))

    nsInfoObj= None
    if len(nsinfo) > 0:
        nsInfoObj = processNSInfo(nsinfo, id2Features)
        #print nsInfoObj['outputprocessqp'], nsInfoObj['outputslotting'], ddi, query
    else:
        noNSInfo += 1
        #return # throw away the pages not having nsinfo, these anyway didn't have any choices to explore.
    jobj['fv'] = ''
    jobj['nsinfo'] = ''
    jobj['qp_backend_calls'] = ''
    jobj['gsm_backend_calls'] = ''

    if yfed:
        gsm_dict = getReturningProviders(yfed)
        jobj['gsm_backend_calls'] = base64.b64encode(zlib.compress(json.dumps(gsm_dict, separators=",:")))

    normalized_query = qnormalize(query, spquery)
    jobj['spquery'] = spquery
    jobj['nquery'] = normalized_query

    if nsInfoObj:
        if birth_year != '':
            nsInfoObj['features']['u::birth_year'] = birth_year
        if gender != '':
            if gender == 'f':
                gender = 1
            else:
                gender = 0
            nsInfoObj['features']['u::gender'] = gender

        #print "passing", repr(nsInfoObj['features'])
        if len(webfeat) > 0:
            nsInfoObj['features'].update(eval(webfeat.replace('__','_')))
        jobj['fv'] = serializeFV(None, nsInfoObj['features'])
        del nsInfoObj['features']
        jobj['nsinfo'] = base64.b64encode(zlib.compress(json.dumps(nsInfoObj, separators=",:")))
        if 'inputqp' in nsInfoObj:
            cat_dict = parseInputQP(nsInfoObj['inputqp'])
            jobj['qp_backend_calls'] = base64.b64encode(zlib.compress(json.dumps(cat_dict, separators=",:")))
    else:
        jobj['fv'] = {}
        jobj['fv']['q'] = {}
        jobj['fv'] = base64.b64encode(zlib.compress(json.dumps(jobj['fv'], separators=",:")))

    jobj['nsddo'] = base64.b64encode(zlib.compress(nsddo))
    jobj['e'] = [e[1] for e in sorted(events)]
    if not nsInfoObj:
        newe = []
        for e in jobj['e']:
            e['p'] = 1.0
            newe.append(e)
        jobj['e'] = newe

    print '\01'.join([jobj['id'], json.dumps(jobj, separators=",:")])
    nOutPVs += 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as fin:
            id2Features = {}
            for line in fin:
                feat = json.loads(line)
                id2Features[int(feat["id"])] = feat["name"]

    for line in sys.stdin:
        parts = line.strip().split('\01')
        if last_id != parts[0]:
            process_recordset(record_set)
            record_set = [parts]
            last_id = parts[0]

        else:
            record_set.append(parts)

    if record_set:
        process_recordset(record_set)

    print >> sys.stderr, "Processed %d page-views. Wrote %d page-views" % (nPVs, nOutPVs)
    print >> sys.stderr, "No NSInfo for %d page-views (%f %%)" % (noNSInfo, float(noNSInfo)*100.0/nPVs)
    print >> sys.stderr, "DONE!"


