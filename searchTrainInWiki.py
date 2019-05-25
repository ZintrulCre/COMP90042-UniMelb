import json
import pickle
from time import time
t1=time()
actual = json.load(open("train.json",'r+'))
relation=pickle.load(open('relation.txt','rb'))

print("start loading")
part1=json.load(open('wiki-001.json','r+'))
part2=json.load(open('wiki-002.json', 'r+'))
part3=json.load(open('wiki-003.json', 'r+'))
part4=json.load(open('wiki-004.json', 'r+'))
print("finish loading")

mlinput=[]
evidence_dict={}
i=0
for entry in actual:
    print('processing',i)
    if actual[entry]['label']=='NOT ENOUGH INFO':
        evidence_dict['verbs']=tuple(relation[i])
        evidence_dict['label']='NOT ENOUGH INFO'
        mlinput.append(evidence_dict)
        evidence_dict={}
    else:
        evidence_label=actual[entry]['evidence']
        relation_tuple=tuple(relation[i])
        for evidence_entry in evidence_label:
            if part1.__contains__(evidence_entry[0]):
                evidence_dict['verbs']=(relation_tuple,part1[evidence_entry[0]][str(evidence_entry[1])])
                evidence_dict['label']=actual[entry]['label']
                mlinput.append(evidence_dict)
                evidence_dict={}
            elif part2.__contains__(evidence_entry[0]):
                evidence_dict['verbs']=(relation_tuple,part2[evidence_entry[0]][str(evidence_entry[1])])
                evidence_dict['label']=actual[entry]['label']
                mlinput.append(evidence_dict)
                evidence_dict={}
            elif part3.__contains__(evidence_entry[0]):
                evidence_dict['verbs']=(relation_tuple,part3[evidence_entry[0]][str(evidence_entry[1])])
                evidence_dict['label']=actual[entry]['label']
                mlinput.append(evidence_dict)
                evidence_dict={}
            elif  part4.__contains__(evidence_entry[0]):
                evidence_dict['verbs']=(relation_tuple,part4[evidence_entry[0]][str(evidence_entry[1])])
                evidence_dict['label']=actual[entry]['label']
                mlinput.append(evidence_dict)
                evidence_dict={}
    i+=1

pickle.dump(mlinput,open("mlinput.txt",'wb'))

#

#
# for entry in actual:
#     entry['evidence']

