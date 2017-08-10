# -*- coding: utf-8 -*-
"""
Author: Adam Ek
Contact: adek2204@student.su.se / adam.leif.ek@gmail.com
Year: 2016
Comments:
"""
import sys
import math
import numpy as np
#import re
from os import listdir
from collections import defaultdict
from collections import OrderedDict
from read_succore import get_eval_keys
from colloc_stagger import collocations


class TextFormatter:
    def __init__(self, file_regex = False):
        self.tagged_text = [[]]

        self.collocation_patterns = []
        self.bigrams = []
        self.nnvbbigrams = []
        self.tokens = 0

        self.w_counts = defaultdict(int)

        if not file_regex:
            self.file_regex = '.*.conll'
        else:
            self.file_regex = file_regex

    def read_text_stagger(self, path):
        with open(path) as textfile:
            text = textfile.read()
            text = text.split('\n')
            # iterate over words
            for i, s in enumerate(text):
                s = s.split('\t')
                if len(s) > 2:
                    # format word and extract information, then save it
#                    self.tagged_text[-1].append([s[0], s[2], s[3],
#                                        s[4].split('|'), s[6] ,s[7]])
                    self.tagged_text[-1].append('\t'.join(s))
                else:
                    # add new sentence
                    self.tagged_text.append([])

        self.create_coll(self.tagged_text)
        return self.tagged_text

    def read_text_succore(self, file):
        with open(file) as textfile:
            text = textfile.read()
            text = text.split('\n')
            # iterate over words
            for i, s in enumerate(text):
                s = s.split('\t')
                if len(s) > 2:
                    # format word and extract information
                    self.tagged_text[-1].append([s[2], s[4], s[5],
                                        s[7].split('|'), s[8] ,s[9], s[10], s[11]])
                    self.w_counts[s[4]] += 1
                else:
                    self.tagged_text.append([])

        self.create_coll(self.tagged_text)

        return self.tagged_text

    def create_coll(self, text):
        return collocations(text)

class AnaphoraResolutor:
    def __init__(self, collocation_patterns, graph = False):
        self.text = 'default'
        self.pp = set(['du', 'han', 'hon',
                       'vi', 'ni', 'de',
                       'mig', 'dig', 
                       'honom', 'henne',
                       'oss', 'er', 'dem',
                       'hans', 'hennes',
                       'min', 'din', 
                       'mitt', 'ditt', 
                       'jag'])

        self.graph = graph
        self.collocation_patterns = collocation_patterns
        self.resolved_antecedents = defaultdict(dict)
        self.last_antecedent = defaultdict(dict)

        # pronoun information collection
        self.current_anaphor = ''  # current pronoun
        self.anaphora_graph = ''
        self.nps = set()
        
        self.rest1 = 0
        self.skip_candidate = ''

    def network_anaphora(self, context):
#        for x in context:
#            for y in x:
#                print(y)
        self.current_anaphor = context[0][-1]
        # self.parse_dependency(main_sentence)
        
        ### SKIP NN/PM in [NN/PM VB PRN] construction
#        if len([x[2] for x in context[0][-3:]]) == 3:
#            if [x[2] for x in context[0][-3:]][1] == 'VB':
#                if [x[2] for x in context[0][-3:]][0] in ['PM', 'NN']:
##                    print([x[1] for x in context[0][-3:]])
##                    print([x[2] for x in context[0][-3:]])
##                    print('')
#                    self.skip_candidate = context[0][-3]
        

        if self.current_anaphor[1] not in self.resolved_antecedents:
            self.resolved_antecedents[self.current_anaphor[1]] = defaultdict(int)
            self.last_antecedent[self.current_anaphor[1]]['e'] = ''
            self.last_antecedent[self.current_anaphor[1]]['s'] = ''
            
        if np.random.choice(['a','b'], p=[0.05,0.95]) == 'a':
            return OrderedDict()

        return self.candidates(context)

    def candidates(self, context):
        candidates = []

        c_scores = defaultdict(int)
        sent_count = defaultdict(int)

        # find NOUNS and PROPERNAMES
        gennum = []
        if 'NEU' in self.current_anaphor[3]:
            gennum.append('NEU')
        elif 'UTR' in self.current_anaphor[3]:
            gennum.append('UTR')
        elif 'UTR/NEU' in self.current_anaphor[3]:
            gennum.append('UTR')
            gennum.append('NEU')
        else:
            gennum.append('MAN')

        if 'SIN' in self.current_anaphor[3]:
            gennum.append('SIN')
        elif 'SIN/PLU' in self.current_anaphor[3]:
            gennum.append('SIN')
            gennum.append('PLU')
        else:
            gennum.append('PLU')

        for n, sent in enumerate(context):
            candidates.append([])  # candidates for current sentence
            for z, w in enumerate(sent):
#                print(w)
#                if self.current_anaphor[1] == 'jag':

#                if w[2] == 'NN':
#                    self.nps.add(w[1])
#                    if len([x for x in w[3] if x in gennum]) >= 2: # Gender and numerus agreement
#                        candidates[n].append(w)
#                        sent_count[w[1]] += 1
#                    else:
#                        pass
                    
                if w[2] == 'PM':   
#                if w[3] == 'PROPN':
                    self.nps.add(w[1])
#                    print(w)
#                    if 'SIN' in gennum:    
                    sent_count[w[1]] += 1
                    candidates[n].append(w)
#                    else:
#                        continue

        if self.last_antecedent[self.current_anaphor[1]]['e'] != '':
            if self.last_antecedent[self.current_anaphor[1]]['e'][1] not in [y[1] for x in candidates for y in x]:
                candidates.append([self.last_antecedent[self.current_anaphor[1]]['e']])
#                print(candidates)

        # set initial scores dependent on context
        for m, cand_sent in enumerate(candidates):
            for n, cand in enumerate(cand_sent):
                dfn, aktv, pp, dist, rep = 0, 0, 0, 0, 0

                # how to integrate with added candidates?
                # indicative verb, active verbs are considered indicadive.
                if (m+1) == len(candidates):
                    if not self.last_antecedent[self.current_anaphor[1]]['s'] == '':
                        pp = self.last_antecedent[self.current_anaphor[1]]['s'][0]
#                        pp = 0
                        dfn = self.last_antecedent[self.current_anaphor[1]]['s'][1]
                        aktv = self.last_antecedent[self.current_anaphor[1]]['s'][2]
#                        pp = 0
                else:
                    if (n+1) != len(cand_sent):
                        if context[m][(n+1)][2] == 'VB':
#                        if context[m][(n+1)][3] == 'VERB':
                            if 'AKT' in context[m][n+1][3]:
#                                print(1)
                                aktv = 1

#                     check if prep phrase
                        cand_index = int(cand[0]) - 1
#                        print(context[m])
                        if context[m][(n+1)][2] == 'PP':
#                        if context[m][(n+1)][3] == 'ADP':
#                            print(2)
                            pp = -1
                        else:                        
                            if cand_index > 3:
                                for w in [x for x in context[m][cand_index-4:cand_index]]:
                                    if w[-4] == cand[-4]:
                                        if w[2] == 'PP':
#                                        if w[2] == 'ADP':
#                                            print(3)
                                            pp = -1
                            else:
                                for w in [x for x in context[m][:cand_index]]:
                                    if w[-4] == cand[-4]:
                                        if w[2] == 'PP':
#                                        if w[2] == 'ADP':
#                                            print(4)
                                            pp = -1

                    # definiteness, DEF or genetiv
                    if [x for x in context[m] if x[2] == 'DT' and x[-4] == cand[0] and 'DEF' in x[3]]:
#                    if [x for x in context[m] if x[3] == 'DET' and x[-4] == cand[0] and 'DEF' in x[4]]:
                        dfn = 1
                    elif 'GEN' in cand[4] or 'DEF' in cand[4]:
                        dfn = 1
                    else:
                        dfn = 0

                # check distance
                if m == 0:
                    dist = 3
                elif m == 1:
                    dist = 2
                elif m == 2:
                    dist = 1
                else:
                    dist = 0
                    
#                # +1/-1                    
#                if self.current_anaphor[1] in ['hans', 'hennes']:
#                    if int(cand[0]) == int(self.current_anaphor[0])-2:
#                        dist += 1

                # check for repeated candidates
                if sent_count[cand[1]] >= 3:
                    rep = 2
                elif sent_count[cand[1]] == 2:
                    rep = 1
                else:
                    rep = 0
                    
#                print(pp)
                if cand[1] not in c_scores.keys():
                    c_scores[cand[1]] = self.score(cand, dist, pp, rep, dfn, aktv)
                else:
                    score2 = self.score(cand, dist, pp, rep, dfn, aktv)
                    if sum(c_scores[cand[1]]) < sum(score2):
                        c_scores[cand[1]] = score2

        # dist, pp, dfn, akt_vb, theme, semrole, propern, colloc
        # sort the dictionary
        top = OrderedDict(sorted(c_scores.items(), key=lambda x: sum(x[1]),
                                 reverse=True))
                             
        cheat = ''
        for cd in candidates:
            for i, w in enumerate(cd):
                if i == 0:
                    cheat = w[1]
                    
        return self.sort_candidates(top, [y for x in candidates for y in x], cheat)

    # do the remaining preferences
    def score(self, w, dist, ppn, repeated, dfn, indvb):
        
        score = [dist, repeated, dfn]
#        score = [dist, ppn, indvb, repeated, dfn]
        
        if w[0] == '1':
            score.append(0)
        else:
            score.append(0)

        # semantic role agreement
        if w[-3] == self.current_anaphor[-3]:
            score.append(1)
        else:
            score.append(0)

        #collocation pattern
        wpattern = []
        apattern = []
        for w1, w2 in self.collocation_patterns:
            if w1 == w[1]:
                wpattern.append(w2)
            elif w1 == self.current_anaphor[1]:
                apattern.append(w2)

        if [x for x in wpattern if x in apattern]:
            score.append(2)#2
        else:
            score.append(0)

        return score

    def sort_candidates(self, top, candidates, cheat = True):
#        ## BASELINE
#        if cheat:
#            top[cheat][0] += 100
#            
        #resolve equality
        highest = []
        for num, c in enumerate(top):
            if num == 0:
                highest.append(c)
            else:
                if sum(top[c]) == sum(top[highest[0]]):
                    highest.append(c)

        #collocation preference
        if len(highest) > 1:
#            print('1>', highest)
            new_high = []
            for name in highest:
                if top[name][-1] > 1:
                    top[name][-1] += 1
                    new_high.append(name)
                    
            if not new_high:
                new_high = highest

            #distance preference
            if len(new_high) > 1:
                ordering = [x[1] for x in candidates if x[1] in new_high]
                top[ordering[0]][1] += 1
                    

        ntop = OrderedDict(sorted(top.items(), key=lambda x: sum(x[1]), reverse=True))
        
        for ll, c in enumerate(ntop):
            if ll == 0:
                self.resolved_antecedents[self.current_anaphor[1]][c] += 1
                self.last_antecedent[self.current_anaphor[1]]['e'] = [x for x in candidates if x[1] == c][0]
                self.last_antecedent[self.current_anaphor[1]]['s'] = [ntop[c][1], ntop[c][3], ntop[c][4]]
#            print(c, sum(top[c]), top[c], '!!!')

        return ntop

def get_k(tdict, i):
    for x in tdict:
        if tdict[x] == i:
            return x

if __name__ == '__main__':
    A = False
    pnl = 4

    printstuff = True
    pron_freq = defaultdict(int)
    pron_ante = defaultdict(int)
    
    features = {'t':[0,0,0,0,0,0,0,0,0,0],'f':[0,0,0,0,0,0,0,0,0,0]}
    g = [0,0]
    TP_rocaucstat = []
    FP_rocaucstat = []
    FN_rocaucstat =[]
    HA = [0,0]
    
    def do_prn(k, lol=False):
        
        pp = set(['han', 'hon', 'honom', 'henne'])
        context_l = k
        
        AS = []
        NS = []    
        
        score = defaultdict(list)
        for i in range(0,30):
            score[i] = [0,0]
            score[-i] = [0,0]
    
        dirr = '/home/adam/data/SUCCORE/'
    #    dirr = '/home/xubuntu/git/socialnetworkextraction/data/succore/'
        
        oo = []
        for c, text_file in enumerate(listdir(dirr)):
            tf = TextFormatter()
            if not text_file.endswith('.conll'):
                continue
#            print(text_file)
            text = tf.read_text_succore(dirr+text_file)
            
    
            ar = AnaphoraResolutor(tf.collocation_patterns)
    
            oo.append(text_file)
    
            # Etc variables
            fnch = False
            re = [0,0,0,0,0,0]
            ne = [0,0,0,0,0,0]
            
            eval_keys = get_eval_keys(text)
    #        print(eval_keys)
            for i, sentence in enumerate(text):
                if sentence:
#                    q = False
                    for n, word in enumerate(sentence):
                        fnch = False
                        if word[-2] == '(PRO)':
                            if word[1] in pp:
                                pron_freq[word[1]] += 1
                                # get referent of pronoun
                                ref = word[-1].replace('(','').replace(')','')
                                if ref not in eval_keys.keys():
                                    continue
                                pron_ante[word[1]] += 1
                                # get context around pronoun
                                context = [text[i][:n+1]]
                                for ti in range(1,context_l+1):
                                    context.append(text[i-ti])
                                    
#                                if q != 1:
#                                    if word[-1] == '(1)' and text_file == 'kk44.conll' and word[0] == '1':
#                                        q = 1
#                                        print('>>>', word)
#                                        for ss in context:
#                                            for ww in ss:
#                                                print(ww)
#                                        print('-----------')
#                                print(len(context))
                                # correct antecedent in candidates?
                                if any(y[1] for x in context for y in x if y[1] in eval_keys[ref]):
                                    g[0] += 1
                                    fnch = True
                                else:

                                    g[1] += 1
                                    
                                res = ar.network_anaphora(context)
#                                if fnch:
#                                    print([y[1] for x in context for y in x])
#                                    print(res.keys())
#                                    print(eval_keys[ref], '\n')
                                if res != OrderedDict():
                                    top_score = '-'
                                    for topc, antecedent in enumerate(res):
                                        if topc == 0:
                                            top_score = antecedent
                                            text[i][n][1] = antecedent
                                            text[i][n][2] = 'PM'
                                        else:
                                            pass
    
                                    if eval_keys[ref]:
                                        # antecedent among candidates?
                                        if any(x for x in res.keys() if x in eval_keys[ref]):
                                            ne[0] += 1 #tpcs
                                            if top_score in eval_keys[ref]:
#                                                TP_rocaucstat.append(sum(res[top_score])/sum([sum(x) for y, x in res.items()]))
                                                re[0] += 1
                                                continue
                                            else:
#                                                FN_rocaucstat.append(sum(res[top_score])/sum([sum(x) for y, x in res.items()]))
                                                re[2] += 1
#                                                HA[1] += 1
                                                continue
                                        else:
                                            if fnch == True:
                                                print(' '.join([y[1] for x in context for y in x]))
                                                print([x for x in res.keys()], eval_keys[ref])
                                                print('')
                                                ne[2] += 1 #FN
                                                re[1] += 1
#                                                HA[1] += 1
                                                continue
#                                                fnch = False
                                            else:
#                                                if sum([sum(x) for y, x in res.items()]) == 0:
#                                                    FP_rocaucstat.append(1)
#                                                else:
#                                                    FP_rocaucstat.append(sum(res[top_score])/sum([sum(x) for y, x in res.items()]))
                                                ne[1] += 1 #FP
                                                re[1] += 1
#                                                HA[1] += 1
                                                continue
                                    else:
                                        continue # TODO
                                else:
                                    if fnch == True:
                                        re[1] += 1
                                        ne[1] += 1
                                    else:
                                        re[3] += 1
                                        ne[3] += 1
                                    # nothing returned
#                                    FP_rocaucstat.append(1)
#                                    re[1] += 1
#                                    ne[1] += 1
#                                    HA[1] += 1
                                    continue
            
            NS.append(ne)
            AS.append(re)
    
        genre1_score = [0,0,0,0,0,0]
        genre2_score = [0,0,0,0,0,0]
        genre1_avg = [0,0,0,0,0,0]
        genre2_avg = [0,0,0,0,0,0]
    
        show_doc = False
        doc_count = defaultdict(int)
        if printstuff:
            avg_scoreAS = [0,0,0,0]
            avg_scoreNS = [0,0,0,0]
            a_tp = 0
            a_fp = 0
            a_fn = 0
            b_tp = 0
            b_fp = 0
            b_fn = 0
            for i, doc in enumerate(AS):
                doc_count[oo[i]] += sum(doc)
                if show_doc:
                    print('-----', oo[i])
                if oo[i][:2] in ['kk', 'kl', 'kn']:
                    genre1_score[0] += doc[0]
                    genre1_score[1] += doc[1]
                    genre1_score[2] += doc[2]
                    genre1_score[3] += NS[i][0]
                    genre1_score[4] += NS[i][1]
                    genre1_score[5] += NS[i][2]
                else:
                    genre2_score[0] += doc[0]
                    genre2_score[1] += doc[1]
                    genre2_score[2] += doc[2]
                    genre2_score[3] += NS[i][0]
                    genre2_score[4] += NS[i][1]
                    genre2_score[5] += NS[i][2]
                
                #AS
                tp1 = doc[0]
                fp1 = doc[1]
                fn1 = doc[2]
                a_tp += doc[0]
                a_fp += doc[1]
                a_fn += doc[2]
                #NS
                tp2 = NS[i][0]
                fp2 = NS[i][1]
                fn2 = NS[i][2]
                b_tp += NS[i][0]
                b_fp += NS[i][1]
                b_fn += NS[i][2]
                if show_doc:
                    print(tp1, fp1, fn1, sum([tp1, fp1, fn1]))
                    print(tp2, fp2, fn2, sum([tp2, fp2, fn2]))
#                print(genre1_score, sum(genre1_score))
#                print([a_tp, a_fp, a_fn, b_tp, b_fp, b_fn], sum([a_tp, a_fp, a_fn, b_tp, b_fp, b_fn]))
                
                #AS
                try:
                    f1 = 0
                    pr1 = tp1 / (fp1+tp1+fn1)
                    re1 = tp1 / (fn1+tp1)
                    if pr1+re1 != 0:
                        f1 = (2*pr1*re1) / (pr1+re1)
    
#                    print([pr1, re1, f1])
                    avg_scoreAS[0] += pr1 
                    avg_scoreAS[1] += re1 
                    avg_scoreAS[2] += f1 
                    avg_scoreAS[3] += 1
                    
                    if oo[i][:2] in ['kk', 'kl', 'kn']:
                        genre1_avg[0] += pr1
                        genre1_avg[1] += re1
                        genre1_avg[2] += f1
                    else:
                        genre2_avg[0] += pr1
                        genre2_avg[1] += re1
                        genre2_avg[2] += f1
                    if show_doc:
                        print(pr1, re1, f1)
                        
                except Exception as e:
#                    print(tp1, (tp1+fp1))
#                    print(tp1, (tp1+fn1))
#                    print(e)
                    pass
                #NS
                try:
                    f2 = 0
                    pr2 = tp2 / (fp2+tp2)
                    re2 = tp2 / (fn2+tp2)    
                    if pr2+re2 != 0:
                        f2 = (2*pr2*re2) / (pr2+re2)         
    
#                    print([pr2, re2, f2])
                    avg_scoreNS[0] += pr2
                    avg_scoreNS[1] += re2
                    avg_scoreNS[2] += f2 
                    avg_scoreNS[3] += 1
                    
                    if oo[i][:2] in ['kk', 'kl', 'kn']:
                        genre1_avg[3] += pr2
                        genre1_avg[4] += re2
                        genre1_avg[5] += f2
                    else:
                        genre2_avg[3] += pr2
                        genre2_avg[4] += re2
                        genre2_avg[5] += f2
                    if show_doc:
                        print(pr2, re2, f2)
                        
                except Exception as e:
#                    print(tp2, (tp2+fp2))
#                    print(tp2, (tp2+fn2))
#                    print(e)
                    pass
#                print('')
        print('AS->',a_tp, a_fp, a_fn, sum([a_tp,a_fp,a_fn]), '\nNS->',b_tp, b_fp, b_fn, sum([b_tp, b_fp,b_fn]))
        try:
            
            a_pr = a_tp / (a_fp+a_tp+a_fn)
            a_re = a_tp / (a_fn+a_tp)
            a_f = (2*a_pr*a_re) / (a_pr+a_re)
            print('TOT_AS:', a_pr, a_re, a_f)
        except Exception as e:
            print(e)        
            pass
        
        try:
            b_pr = b_tp / (b_fp+b_tp)
            b_re = b_tp / (b_fn+b_tp)
            b_f = (2*b_pr*b_re) / (b_pr+b_re)  
            print('TOT_NS', b_pr, b_re, b_f)
        except Exception as e:
            print(e)
            pass
        
        _AS = [0,0,0,0]
        _NS = [0,0,0,0]
#        print('avg', avg_scoreNS, avg_scoreAS)
        for i, avg in enumerate(avg_scoreAS):
            _AS[i] = avg_scoreAS[i]/10
            _NS[i] = avg_scoreNS[i]/10
            
#        print('AS_AVG', _AS[:3])
#        print('NS_AVG', _NS[:3])
#        print(doc_count, sum(doc_count.values()))
#        print(a_tp, a_fp, a_fn, b_tp, b_fp, b_fn)
        
        try:
            g1 = genre1_score[0]/(genre1_score[0]+genre1_score[1])
            g2 = genre1_score[0]/(genre1_score[0]+genre1_score[2])
            g3 = genre1_score[3]/(genre1_score[3]+genre1_score[4])
            g4 = genre1_score[3]/(genre1_score[3]+genre1_score[5])
            print('genre1_AStot_pr', g1, 're', g2, 'fsc', (2*g2*g1)/(g1+g2))
            print('genre1_NStot_pr', g3, 're', g4, 'fsc', (2*g3*g4)/(g3+g4))
        except:
            pass

        try:
            g1 = genre2_score[0]/(genre2_score[0]+genre2_score[1])
            g2 = genre2_score[0]/(genre2_score[0]+genre2_score[2])
            g3 = genre2_score[3]/(genre2_score[3]+genre2_score[4])
            g4 = genre2_score[3]/(genre2_score[3]+genre2_score[5])
            print('genre2_AStot_pr', g1, 're', g2, 'fsc', (2*g2*g1)/(g1+g2))
            print('genre2_NStot_pr', g3, 're', g4, 'fsc', (2*g3*g4)/(g3+g4))
        except:
            pass


#        print('genre1_ASavg', [x/10 for x in genre1_avg[:3]])
#        print('genre1_NSavg', [x/10 for x in genre1_avg[3:]])
#        print('genre2_ASavg:', [x/10 for x in genre2_avg][:3])
#        print('genre2_NSavg:', [x/10 for x in genre2_avg][3:])
#        y_true = [1]*len(TP_rocaucstat) + [0]*len(FP_rocaucstat)+ [1]*len(FN_rocaucstat)
#        y_scor = [1]*len(TP_rocaucstat) + [0]*len(FP_rocaucstat)+ [0]*len(FN_rocaucstat)
#        print('AS_ROC', roc_auc_score(y_true, y_scor))
#        
#        y_true = [1]*b_tp + [0]*b_fp+ [1]*b_fn
#        y_scor = [1]*b_tp + [0]*b_fp+ [0]*b_fn
#        print('CS_ROC', roc_auc_score(y_true, y_scor))
#        
##        print(features)
#        print(len(TP_rocaucstat) + len(FP_rocaucstat) + len(FN_rocaucstat))
#        print(HA)
#    
#for w in range(1,10):
#    print('\nRANGE:', w)
#    do_prn(w)

do_prn(4)
#print(pron_ante, pron_freq)
#            
#    print('Totals:')
#    print('if sum(correct_antecedent) >= 10: successrate =', check[0]/sum(check))
#    print('avg_score for correct_antecedent =', sum(avgcorr)/len(avgcorr))
##    print('avg_candidates per anaphora =', totc/sum([y for x in pronounset for y in pronounset[x] for y in pronounset[x][y]]))
#    print(totc)
#    print(tr.rest1)