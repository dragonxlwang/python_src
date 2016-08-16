import sys, os, json

def addKeys(d, keys):
    if len(keys) > 0:
        if keys[0] not in d:
            d[keys[0]] = {}
        return addKeys(d[keys[0]], keys[1:])
    else:
        return d

def createNestedDict(pydict, newdict, level=1, sep='::'):
    seenNameSpaces = []
    for k,v in pydict.iteritems():
        levels = k.split(sep)
        if tuple(levels[:-1]) not in seenNameSpaces:
            addKeys(newdict, levels[:-1])
            seenNameSpaces += [tuple(levels[:-1])]
        x = newdict
        for l in levels:
            if l in x :
                x = x[l]
            else:
                x[l] = v
                break
    return newdict

def flatten(pydict, newdict, prefix, sep='::'):
    #print >> sys.stderr, "Got invalid input", pydict, prefix
    for k,v in pydict.iteritems():
        k = k.encode('utf8')
        tmp = sep.join([prefix, str(k)]) if prefix else str(k)
        if type(v) == type({}): #this is a dict
            newdict.update(flatten(v, newdict, tmp))
        else:
            newdict[tmp] = v
    return newdict
    

if __name__ == '__main__':
    print createNestedDict({'x':0, 'a::b': 1, 'a::c': 2, 'n::b::c': 3, 'n::b::b': 4}, {}, 1)
    sys.exit()
    for line in sys.stdin:
        fields = line.strip().split('\t')
        #print '\t'.join([fields[0], json.dumps(flatten(json.loads(fields[1]), None), separators=",:")])
        for k,v in flatten(json.loads(fields[0]), {}, sys.argv[1]).iteritems():
            print '\t'.join([str(k).replace('::', '\t'), str(v)])


