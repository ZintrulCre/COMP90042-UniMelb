import os
import json
from tqdm import tqdm
import pickle
from time import time
from collections import defaultdict
t1=time()
train_json = json.load(open("Source/train.json",'r+'))
relation=pickle.load(open('relation.txt','rb'))

wiki_json = "wiki.json"
wiki_dict = pickle.load(open(wiki_json, 'rb'))
print(len(wiki_dict))

ml_input=[]
for train_entry in train_json:
    evidence_dict = {}
    if train_json[train_entry]['label']=='NOT ENOUGH INFO':
        evidence_dict['verbs']=tuple(relation[i])
        evidence_dict['label']='NOT ENOUGH INFO'
        ml_input.append(evidence_dict)
    else:
        evidence_label=train_json[train_entry]['evidence']
        relation_tuple=relation[i]
        for evidence_entry in evidence_label:
            evidence_dict['verbs'] = (relation[i], wiki_dict[evidence_entry[0]][str(evidence_entry[1])])
            evidence_dict['label'] = train_json[train_entry]['label']
            ml_input.append(evidence_dict)
    print(evidence_dict)

pickle.dump(ml_input,open("mlinput.txt",'wb'))
