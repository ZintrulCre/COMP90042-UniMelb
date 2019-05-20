import json

def getEntity(word_pos):
    entity_name=[]
    entity=[]
    i=0
    while(i<len(word_pos)):
        '''extract the entity'''
        if word_pos[i][1][:2]=='NN':
            entity_name.append(word_pos[i][0])
            if i == len(word_pos)-1:
                entity.append(word_pos[i][0])
                return entity
            j=i+1
            try:
                while(word_pos[j][1][:2]=='NN' and j!=len(word_pos)-1):
                    entity_name.append(word_pos[j][0])
                    j+=1
            except: print(word_pos,word_pos[j-1])
            i=j
        i+=1
        if len(entity_name)!=0:
            entity.append('_'.join(entity_name))
            entity_name=[]
    return entity

def getRelation(word_pos):
    relation_name = []
    relation = []
    i = 0
    while (i < len(word_pos)):
        '''extract the entity'''
        if word_pos[i][1][:2] == 'VB':
            relation_name.append(word_pos[i][0])
            if i == len(word_pos) - 1:
                return word_pos[i][0]
            j = i + 1
            while ((word_pos[j][1][:2] == 'VB' or word_pos[j][1][:2] =='RB' or word_pos[j][1][:2] =='IN')and j!=len(word_pos)-1):
                relation_name.append(word_pos[j][0])
                j += 1
            i = j
        i += 1
        if len(relation_name) != 0:
            relation.append('_'.join(relation_name))
            relation_name = []
    return relation

def getFromWiki(entity,label):
    filePath="wiki-pages-json/wiki-"
    relevantWiki=[]
    for i in range(1,110):
        print("searching for the wiki"+str(i))
        if i < 10:
            fileRealPath=filePath+'00'+str(i)+".json"
        elif i < 100:
            fileRealPath=filePath+'0'+str(i)+".json"
        else:
            fileRealPath=filePath+str(i)+'.json'
        f=json.load(open(fileRealPath, 'r+'))
        if f.__contains__(entity):
            return f[entity][label]

            # wiki_entry = f.readlines()
            # for entry in wiki_entry:
            #     for eachEntity in entity:
            #         if entry.split(' ')[0] == eachEntity:
            #             print("I find one!")
            #             relevantWiki.append(entry)
            # f.close()
    return relevantWiki

