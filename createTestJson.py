import json
import pickle
test_claim=json.load(open('test-unlabelled.json','r+'))
text_result=json.load(open('text_result.json','r+'))
mlinput=pickle.load(open('mlinput_merge.txt','rb'))
#print(mlinput[:10])

testinput=[]
for id in text_result:
    output = {}
    output['claim']=test_claim[id]['claim']
    output['evidence']=[]
    for evid_dict in text_result[id]:
        for evid_sent in evid_dict.values():
            output['evidence'].append(evid_sent.split(' '))
    testinput.append(output)

print(len(testinput))

with open('testinput.json','w+') as f :
    json.dump(testinput,f)

pickle.dump(testinput,open("testinput.txt",'wb'))
