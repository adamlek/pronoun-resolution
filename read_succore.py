#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:48:26 2017

@author: adam
"""
from collections import defaultdict

ok = ['PM']

def get_eval_keys(text):
    sing_ents = ['(NAM)', '(NOM)']#, 'NOM)', 'NAM)']
    startplur_ents = ['(NAM', '(NOM']
    endplur_ents = ['NAM)', 'NOM)']
        
    names = defaultdict(set)
    cap = False
    for i, sentence in enumerate(text):
        if sentence:
            for n, entry in enumerate(sentence):
                idn, name = entry[-1].replace('(','').replace(')',''), entry[1]
#                print(idn, names.keys())
                if idn in names.keys() and entry[2] in ok:
                    names[idn].add(name)
                else:
                    if cap:
                        if entry[-2] in endplur_ents:
                            if entry[2] in ok:
                                names[idn].add(name)
                            cap = False
                    # singles
                    if entry[-2] in sing_ents[0]:
                        if entry[2] in ok:
                            names[idn].add(name)
                    
                    elif entry[-2] in startplur_ents[0]:
                        if entry[2] in ok:
                            names[idn].add(name)
                        cap = True
                    else:
                        pass
    return names

if __name__ == '__main__':
    tagged_text = []
    with open('/home/adam/data/SUCCORE/aa05.conll') as textfile:
        text = textfile.read()
        text = text.split('\n')
        
            # iterate over words
        for i, s in enumerate(text):
            s = s.split('\t')
            if len(s) > 2:
                # format word and extract information
                tagged_text[-1].append([s[2], s[4], s[5], s[7].split('|'), s[8] ,s[9], s[10], s[11]])
            else:
                tagged_text.append([])
