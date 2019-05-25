import pickle
from nltk import pos_tag
from method import getEntity
mlinput_merge=pickle.load(open("mlinput_merge.txt",'rb'))

i=0
for entry in mlinput_merge:
    print(i)
    i+=1
    entry_split=entry['claim'].split(' ')
    if '' in entry_split:
        entry_split.remove('')
    try:
        entity=getEntity(pos_tag(entry_split))
    except:
        exit(0)
    if entry['label']=='NOT ENOUGH INFO':
        entity.append('NOT ENOUGH INFO')
        entry['evidence']=entity
    else:
        entry['evidence'].append(entity)

pickle.dump(mlinput_merge,open('mlinput_merge_with_entity.txt','wb'))