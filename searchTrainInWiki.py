import os
import json
from tqdm import tqdm
import pickle
from time import time
t1=time()
train_json = json.load(open("Source/train.json",'r+'))
# relation=pickle.load(open('relation.txt','rb'))

json_dicts = []
path = "wiki-pages-json"
files = os.listdir(path)
for file in tqdm(files):
    if not os.path.isdir(file):
        json_dicts.append(json.load(open(path + '/' + file)))
print(json_dicts)

mlinput=[]
for train_entry in train_json:
    evidence_dict = {}
    if train_json[train_entry]['label']=='NOT ENOUGH INFO':
        # evidence_dict['verbs']=tuple(relation[i])
        evidence_dict['label']='NOT ENOUGH INFO'
        mlinput.append(evidence_dict)
    else:
        evidence_label=train_json[train_entry]['evidence']
        # relation_tuple=tuple(relation[i])
        for evidence_entry in evidence_label:
            for json_dict in json_dicts:
                if evidence_entry[0] in json_dict:
                    # evidence_dict['verbs']=(relation_tuple,part1[evidence_train_entry[0]][str(evidence_train_entry[1])])
                    evidence_dict['label']=train_json[train_entry]['label']
                    mlinput.append(evidence_dict)
    print(evidence_dict)

pickle.dump(mlinput,open("mlinput.txt",'wb'))
