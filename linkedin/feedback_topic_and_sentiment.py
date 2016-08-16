#! /usr/bin/python

import string
import re
import codecs
import sys
import os
## date 5/12
class FeedbackProcessor():


    def __init__(self):
        with open('feedback_speech_acts_and_topics1.txt', 'r') as fd:
            self.data = fd.read().split("\n")
        with open('cleaned_ngrams.txt', 'r') as ff:
	       self.ngrams = ff.read().split('\n')    
        with open('cleaned_words.txt', 'r') as fc:
            self.words = fc.read().split('\n')
        with open('prepositions.txt', 'r' ) as fp:
            self.p = fp.read().split('\n')
        with open('sentiment_words.txt', 'r') as fs:
            self.s = fs.read().split('\n')
        with open('feedback_verbs_reviewed.txt', 'r') as fv:
            self.verbs_reviewed = fv.read().split('\n')
        self.word_map = {} #high rankig labeled words
        self.ngram_map  = {} #high ranking ngmrams only
        self.speech_act_map = {}
        self.topic_map = {}
        self.selected = {}
        self.cores_only = {}
        self.cores_count = {}
        self.cores_list = []
        self.prep = set()
        self.sentiment = {}
        self.core_sentiment = {}
        self.s_by_top = []
        self.topic_map = {}#map of normed phrases and their core topics
        self.final_added = set()
        self.nouns =  set()
        self.verbs = set()
        self.vp = {}
        self.verb_map = {}
        self.verbs_reviewed_map = {}
        self.category_map = {}
        for i in self.s:
            if len(i.split('\t')) > 2:
                word = i.split('\t')[0]
                count = i.split('\t')[1]
                sentiment = i.split('\t')[2]
                self.sentiment[word] = sentiment
        self.sentiment_set = set(list(self.sentiment.keys()))
        print 'current sentiment words', self.sentiment_set
        for v in self.verbs_reviewed:
            if len(v.split('\t')) > 1:
                verb= v.split('\t')[0]
                count = v.split('\t')[1]
                self.verbs_reviewed_map[verb] = int(count)

        for w in self.words:
            if len(w.split('\t')) > 1:
                word = w.split('\t')[0]
                count = w.split('\t')[1]
                count = int(count)
                self.word_map[word] = count

        for i in self.p:
            self.prep.add(i)

        for i in self.ngrams:
            if len(i.split('\t')) > 1:
                ngram = i.split('\t')[0]
                count = i.split('\t')[1]
                count = int(count)
                if count > 20 :
                    self.ngram_map[ngram] = count
        for line in self.data[0:50000]: ##senior quality analyst   3302    4865:senior quality analyst:4
            if 'LONG' in line or 'SHORT' in line or 'MEDIUM' in line:
                fields = line.split('\t')
                #print fields
                if len(fields) >=5:
                    raw = fields[0]
                    category = fields[1]
                    speech_act = fields[2]
                    normed = fields[4]
                    
                    self.speech_act_map[raw] = speech_act
                    self.topic_map[raw] = normed
                    clause_ngrams = self.get_ngrams(normed)#returns list of ngrams
                    matched_ordered = self.process_normed(clause_ngrams)
                    matched = matched_ordered[0]
                    ordered = matched_ordered[1]
                    #if 'but' in raw.split(): print 'contrast', raw
                    if matched!= {} and len(normed.split()) >3:
                        final = self.get_core(matched, ordered)
                        #if len(final) == 2: print final, normed
                        self.selected[raw] = (final, matched, normed, ordered, speech_act, category)
                        final_string = self.construct_final_topic(final)
                        self.cores_list.append(final_string)
                        if final_string not in self.category_map:
                            self.category_map[final_string] = [category]
                        else:
                            if category not in self.category_map[final_string]: self.category_map[final_string].append(category)
                        if final_string not in self.topic_map:
                            self.topic_map[final_string] = [normed]
                        else:

                            if  type(self.topic_map[final_string]) is list: self.topic_map[final_string].append(normed)
                        if speech_act not in self.cores_only: 
                            self.cores_only[speech_act] = [final_string]
                        else:
                            self.cores_only[speech_act].append(final_string)
                        
                    elif matched!= {} and len(normed.split()) <=3:
                        final = [normed]
                        self.selected[raw] = (final, matched, normed, ordered, speech_act, category)
                        final_string = self.construct_final_topic(final)
                        self.cores_list.append(final_string)
                        if final_string not in self.category_map:
                            self.category_map[final_string] = [category]
                        else:
                            if category not in self.category_map[final_string]: self.category_map[final_string].append(category)
                        if final_string not in self.topic_map:
                            self.topic_map[final_string] = [normed]
                        else:

                            if  type(self.topic_map[final_string]) is list: self.topic_map[final_string].append(normed)
                        if speech_act not in self.cores_only:  ## speech acts and topics contained in them
                            self.cores_only[speech_act] = [final_string]
                        else:
                            self.cores_only[speech_act].append(final_string)
                        
                    else:
                        self.selected[raw] = 'NA'
        
        for i in self.cores_list:
            count = self.cores_list.count(i)
            #print i, count
            words = i.split()
            if words[-1] not in self.prep and words[0] not in self.prep and 'NEG' not in words[-1] and len(words) > 2 and count > 3:
                if 'of' == words[-1] or words[0] == 'of':print i
                self.cores_count[i] = count
        
        for i in self.cores_count:
            words = i.split()
            for word in words:
                if word in self.sentiment and i not in self.core_sentiment:
                    sent = self.sentiment[word]
                    if sent!= 'NA':
                        self.core_sentiment[i] = sent

        for s in self.cores_only:
            topics = self.cores_only[s]##list of topics assosiated with speech act
            
            for topic in topics:
                num = topics.count(topic)
                if num > 3 and len(topic.split()) >2 and topic in self.core_sentiment and topic in self.category_map:
                    category = self.category_map[topic]
                    if (s,topic,num, sent,category) not in self.s_by_top:
                        self.s_by_top.append((s,topic,num,sent,category))


    def infer_pos(self):
        """take ngram list and infer parts of speech of top ranking
        words via their distribution"""

        for ngram in self.ngram_map:
            words =ngram.split()
            count = self.ngram_map[ngram]
            count = int(count)
            if len(words) > 1:
                if words[0] == 'please' or words[0] == 'pls':
                    if words[1] != 'NEG' and words[1] not in self.prep :
                        if len(words[1]) > 2 and 'ly' != words[1][-2:]:
                            self.verbs.add(words[1])
                            #if words[1] == 'go': print 'go', words[1], len(words[1])
                    elif words[1] == 'NEG' and len(words) >2 :
                        if len(words[2]) > 2 and 'ly' != words[2][-2:] and words[2] not in self.prep:
                            self.verbs.add(words[2])
                            #if words[2] == 'go': print 'go', words[2], len(words[2])
        for v in self.verbs:
            if v in self.word_map:
                count = self.word_map[v]
                print v, count
                self.verb_map[v] = count


    def infer_vp(self):
        """step through raw phrases and self.verbs and infer vps"""
        for raw in self.topic_map:
            words = raw.split()
            if len(words) > 2:
                for v in self.verbs_reviewed_map:
                    if v in words and v!= words[-1]:
                        ind = words.index(v)
                        if len(words[ind:]) > 3:
                            lenn = ind+3
                            vp = words[ind:lenn]

                            vp = ' '.join(words[ind:lenn])
                            self.vp[vp] = self.verbs_reviewed_map[v]

                        else:
                            vp = words[ind:]
                            vp = ' '.join(words[ind:])
                            #print 'short vp', v, 'vp', vp
                            self.vp[vp] = self.verbs_reviewed_map[v]
        for vp in self.vp:
            if vp in self.ngram_map:
                print vp, self.ngram_map[vp]

        


    def get_more_bad(self):
        """ gets more negative phrases"""
        self.more_bad = {}
        self.selected_bad = {}
        for topic in self.topic_map:
            if topic in self.core_sentiment and len(topic.split()) >2:
                sent = self.core_sentiment[topic]
                normed_clauses = self.topic_map[topic]
                for clause in normed_clauses:
                    if len(clause.split()) >= 6 and 'but' not in clause:
                        rest = clause.replace(topic, '')##remove topic phrase and keep the rest
                        #for ngram in self.ngram_map:
                        #    if ngram in rest and ngram!= topic:
                        if topic not in self.more_bad:
                            self.more_bad[topic] = [rest]
                        else:
                            if rest not in self.more_bad[topic]:self.more_bad[topic].append(rest)

                    elif len(clause.split()) >=6 and 'but' in clause.split():
                        rest = clause.replace(topic, '')
                        print 'CONTRAST', rest, 'TOPIC',topic
        for i in self.more_bad:

            if 'NEG' not in i or len(i.split()) >=1: 
                for j in self.more_bad[i]:
                    counts = []
                    selected = {}
                    for ngram in self.ngram_map:
                        count  = int(self.ngram_map[ngram])
                        ngram_words = ngram.split()
                        ngram_word_set = set(ngram_words)
                        if ngram in j and len(ngram.split()) > 2 and ngram!=i and ngram.split()[-1] not in self.prep\
                        and ngram_word_set.intersection(self.sentiment_set) == set() and count > 30:## if none of the new words are in known sent
                            #print i, 'ngram', ngram
                            counts.append(count)
                            selected[count] = ngram
                    if counts!=[]:
                        maxx = max(counts)
                        chosen = selected[maxx]
                    else:
                        chosen = 'NONE'
                    if chosen!= 'NONE': self.selected_bad[i] = chosen
                    #print i, 'CHOSEN REST', chosen
        for i in self.selected_bad:
            added = self.selected_bad[i]
            #print i, 'ADDED', added, self.ngram_map[added]
            self.final_added.add(added)

        for i in self.final_added:
            print i
            


        pass

    def construct_final_topic(self, final):
        """take final ngrams and constuct topic string"""
        
        if len(final) == 1:
            if final[-1] in self.prep:
                print 'ENDS IN PREP', final
                short = final[0:-1]
                final_string = ' '.join(short)
            else:
                final_string = ' '.join(final)
            #print final_string
            return final_string

        elif len(final) >1:
            words = []

            for i in final:
                tokens = i.split()
                for t in tokens:
                    if t not in words and len(words) <4:## dont create topics longer than 4 words
                        words.append(t)
            if words[-1] in self.prep: 
                #print words
                short= words[0:-1]
                final_string = ' '.join(short)##strip phrase-final preposition
                #print final_string
            else:
                final_string = ' '.join(words)
            return final_string
        else:
            print 'I CANT STRING THIS', final



    def process_normed(self,clause_ngrams):
        """ get topic based on high ranking ngrams """
        ##5222:control coordinator:3,10850:access coordinator:3
        matched = {}
        ordered = []
       
        for i in clause_ngrams:

            if i in self.ngram_map:
                ordered.append(i)
                matched[i] = int(self.ngram_map[i])
        return (matched, ordered)


    def get_core(self, matched, ordered):
        """from selected ngrams get the core ngrams and construct core topic phrase"""
        core = []
        indeces = []
        final = []
        ranked = sorted(matched.values())
        top_ranked = max(ranked)
        #print ranked, top_ranked
        high_enough = ranked[-3:]
        #print top_ranked
        for i in matched:
            words = i.split()
            if matched[i] == top_ranked  and words[0] not in self.prep and words[1] not in self.prep :
                #final.append(i)
                core.append(i)
                ind = ordered.index(i)
                prev = ind -1
                nxt = ind+1
                indeces.append(prev)
                indeces.append(ind) 


            elif 'NEG' in i and i not in core:
                core.append(i)
                ind = ordered.index(i)
                prev = ind -1
                nxt = ind+1
                indeces.append(prev)
                indeces.append(ind)
                indeces.append(nxt)

            elif 'NEG_ADV' in i and i not in core:
                core.append(i)
                ind = ordered.index(i)
                prev = ind -1
                nxt = ind+1
                indeces.append(prev)
                indeces.append(ind)
                indeces.append(nxt)


            elif 'NEG' not in i and 'NEG_ADV' not in i and words[-1] not in self.prep and i not in core:##doesn't end in prep
                core.append(i)
                ind = ordered.index(i)
                prev = ind -1
                nxt = ind+1
            elif matched[i] == top_ranked and matched[i] in high_enough and words[-1] not in self.prep and i not in core:
                core.append(i)
                ind = ordered.index(i)
                prev = ind -1
                nxt = ind+1
            else:
                if core == [] and matched[i] == top_ranked:
                    core.append(i)
                    #print matched, 'CORE', core

            
        for i in indeces:
            if i>=0 and i < len(ordered):
                if ordered[i] not in final: 
                    final.append(ordered[i])
        if final == []:
            final = core        
        #if core == []: print 'STILL EMPTY', matched, ordered
        return final



    def get_ngrams(self, clause):
        """get ngrams from clause """
        i = 0; j = 0
        words = clause.split()
        cleaned = []
        bigrams = []
        trigrams = []
        ngrams = []
    
        if len(words) >=1:
            while i < len(words)-1:
                bi = words[i] + ' ' + words[i+1]
                i+=1
                bigrams.append(bi)
             
        if len(words) >=2:
           while j < len(cleaned)-2:
               tri = words[j]+ ' ' + words[j+1] +' '  +words[j+2]
               j+=1
               trigrams.append(tri)
        ngrams.extend(bigrams)
        ngrams.extend(trigrams)
        #print ngrams
        return ngrams   
        #print clause, bigrams, trigrams
        ## ADD BIGRAMS AND TRIGRAMS INTO A MAP
        #if bigrams!=[]:
        """for bi in bigrams:
            if bi not in self.clause_ngrams:
                self.clause_ngrams[bi] = 1
            else:
                self.clause_ngrams[bi] = self.clause_ngrams[bi]+1
        #if trigrams != []:
        for tri in trigrams:
            if tri not in self.clause_ngrams:
                self.clause_ngrams[tri] = 1
	    else:
                self.clause_ngrams[tri] = self.clause_ngrams[tri]+1      """ 

        
   


    def write_stuff(self):
        """record things for inspection"""
        """with open('titles_fixed.txt', 'w') as fw:
            for tup in reversed(sorted(self.clause_count.items(), key = lambda x:x[1])):
                word = tup[0]
                count = tup[1]
                fw.write(word)
                fw.write('\t')
                fw.write(str(count))
                fw.write('\n')  """
        with open('feedback_topics_by_speech_act.txt', 'w') as fw1:
            for i in self.s_by_top:
                fw1.write(i[0])
                fw1.write('\t')
                fw1.write(i[1])
                fw1.write('\t')
                fw1.write(str(i[2]))
                fw1.write('\t')
                fw1.write(str(i[3]))
                fw1.write('\t')
                fw1.write(str(i[4]))
                fw1.write('\n')

        with open('feedback_topic_count.txt' , 'w') as fw:
            for i in reversed(sorted(self.cores_count.items(), key = lambda x:x[1])):
                word = i[0]
                count = i[1]
                if word in self.category_map:
                    category = self.category_map[word]
                    print word, category
                else:
                    category = 'NONE'
                if word in self.core_sentiment:
                    sent = self.core_sentiment[word]
                
                    if sent != 'NA':
                        fw.write(word)
                        fw.write('\t')
                        fw.write(str(count))
                        fw.write('\t')
                        fw.write(sent)
                    
                        fw.write('\t')
                        fw.write(str(category))
                        fw.write('\n')

                """else:
                    sent = 'NA'
                fw.write(word)
                fw.write('\t')
                fw.write(str(count))
                fw.write('\t')
                fw.write(sent)
                fw.write('\n') """

        with open('feedback_topics.txt', 'w') as fw:  
            #for tup in reversed(sorted(self.elaboration.items(), key = lambda x:x[1])):
            for i in self.selected:
                
                topics = self.selected[i]
                if topics!= 'NA':
                    fw.write(i)
                    fw.write('\t')

                    fw.write(str(topics[0]))
                    fw.write('\t')

                    fw.write(str(topics[1]))
                    fw.write('\t')
                    fw.write(str(topics[2]))
                    fw.write('\t')
                    fw.write(str(topics[3]))
                    fw.write('\t')
                    fw.write(str(topics[4]))
                    fw.write('\t')
                    fw.write(str(topics[5]))
                    fw.write('\n')
        with open('feedback_added_sentiment_phrases.txt', 'w') as fw:
            for i in self.final_added:
                fw.write(i)
                fw.write('\n')
        with open('feedback_verbs.txt' , 'w') as fw:
            for i in reversed(sorted(self.verb_map.items(), key = lambda x:x[1])):
                fw.write(i[0])
                fw.write('\t')
                fw.write(str(i[1]))
                fw.write('\n')
        with open('feedback_vp_sorted.txt', 'w') as fw:
            for i in reversed(sorted(self.vp.items(), key = lambda x:x[1])):
                fw.write(i[0])
                fw.write('\t')
                fw.write(str(i[1]))
                fw.write('\n')




if __name__ == '__main__':
    fp = FeedbackProcessor()
    
    #fp.get_more_bad()
    fp.infer_pos()
    fp.infer_vp()
    fp.write_stuff()

