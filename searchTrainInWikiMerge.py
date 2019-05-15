import json
import pickle
from time import time
t1=time()
actual = json.load(open("train.json",'r+'))

print("start loading")
part1=json.load(open('wiki-001.json','r+'))
# part1=[]
# part3=[]
# part4=[]
part2=json.load(open('wiki-002.json', 'r+'))
part3=json.load(open('wiki-003.json', 'r+'))
part4=json.load(open('wiki-004.json', 'r+'))
print("finish loading",time()-t1)

mlinput_merge=[]
evidence_dict={}
i=0
for entry in actual:
    print("processing",i)
    i+=1
    if actual[entry]['label']=="NOT ENOUGH INFO":
        evidence_dict['claim']=actual[entry]['claim']
        evidence_dict['label']=actual[entry]['label']
        evidence_dict['evidence']="NOT ENOUGH INFO"
        mlinput_merge.append(evidence_dict)
        evidence_dict={}
    else:
        evidence=actual[entry]['evidence']
        evidence_dict['claim']=actual[entry]['claim']
        evidence_dict['label']=actual[entry]['label']
        evidence_sentence=[]
        for evidenve_entry in evidence:
            if part1.__contains__(evidenve_entry[0]):
                evidence_sentence.append(part1[evidenve_entry[0]][str(evidenve_entry[1])])
            elif part2.__contains__(evidenve_entry[0]):
                evidence_sentence.append(part2[evidenve_entry[0]][str(evidenve_entry[1])])
            elif part3.__contains__(evidenve_entry[0]):
                evidence_sentence.append(part3[evidenve_entry[0]][str(evidenve_entry[1])])
            elif part4.__contains__(evidenve_entry[0]):
                evidence_sentence.append(part4[evidenve_entry[0]][str(evidenve_entry[1])])
        evidence_dict['evidence']=evidence_sentence
        evidence_sentence=[]
        mlinput_merge.append(evidence_dict)
        evidence_dict={}

pickle.dump(mlinput_merge,open("mlinput_merge.txt",'wb'))
print(len(mlinput_merge))
print(mlinput_merge)