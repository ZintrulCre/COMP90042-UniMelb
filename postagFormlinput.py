import pickle
import nltk
from method import getRelation

mlinput=pickle.load(open("mlinput.txt",'rb'))
relation=pickle.load(open("relation.txt",'rb'))

i=0
for entry in mlinput:
    print(round(i/len(mlinput),2))
    if entry['label']!='NOT ENOUGH INFO':
        evidence_verbs=entry['verbs'][len(entry['verbs'])-1]
        evidence_verbs_tagger=nltk.pos_tag(evidence_verbs)
        evidence_relation=getRelation(evidence_verbs_tagger)
        entry['verbs']=(entry['verbs'][0],evidence_relation)
        i+=1
    # if entry['label']=='NOT ENOUGH INFO':
    #     print(entry)
    #     break

pickle.dump(mlinput,open("mlinput_revised.txt",'wb'))