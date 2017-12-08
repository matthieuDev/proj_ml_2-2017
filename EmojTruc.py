import re
import pickle

prog = re.compile("\s((?:[^\s][^\w\n]+)(?:[^\s](?:[^\w\n]+|$))+)")
emojiiTable = {}
for fn in ['train_pos.txt', 'train_neg.txt','test_data.txt']:
    with open(fn) as f:
        for line in f.readlines():
            for result in prog.finditer(line):
                g1 = result.group(1).strip()
                if not g1 in emojiiTable:
                    emojiiTable[g1] = (1,line)
                else:
                    emojiiTable[g1] = (emojiiTable[g1][0] + 1,emojiiTable[g1][1])
theList =  [(v[0], v[1], k) for k, v in emojiiTable.items()]
theListSorted = sorted(theList, key=lambda t: t[0])
theListSorted[-150:]

t2 = {}
for _, sample, moji in theListSorted[-150:]:
    print('"'+moji+'" in "'+sample)
    t2[moji] = input()
    
with open('emojtable.pkl', 'wb') as f:
        pickle.dump(t2, f, pickle.HIGHEST_PROTOCOL)