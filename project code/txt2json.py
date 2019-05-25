import json
inDirectPath='wiki-pages-text/wiki-'
outDirectPath="wiki-pages-json/wiki-"
wiki1 = {}

for i in range(1, 110):
    # print("searching for the wiki"+str(i))
    if i < 10:
        inRealPath = inDirectPath + '00' + str(i) + ".txt"
        outRealPath= outDirectPath +'00'+ str(i)+".json"
    elif i < 100:
        inRealPath = inDirectPath+ '0' + str(i) + ".txt"
        outRealPath= outDirectPath +'0'+ str(i)+".json"
    else:
        inRealPath = inDirectPath + str(i) + '.txt'
        outRealPath= outDirectPath +str(i)+".json"

    with open(inRealPath, 'r+') as fin:
        print('adding file ',i)
        wiki_entry = fin.readlines()
        title = ''
        for entry in wiki_entry:
            title = entry.split(' ')[0]
            label = entry.split(' ')[1]
            sent = entry.split(' ')[2:]
            if not wiki1.__contains__(title):
                wiki1[title] = {label: sent}
            else:
                wiki1[title][label] = sent

    if i==27:
        with open('wiki-001.json', 'w+') as fout:
            print("dumping json file ",i)
            json.dump(wiki1, fout)
            wiki1={}

    elif i==54:
        with open('wiki-002.json', 'w+') as fout:
            print("dumping json file ",i)
            json.dump(wiki1, fout)
            wiki1={}

    elif i==81:
        with open('wiki-003.json', 'w+') as fout:
            print("dumping json file ",i)
            json.dump(wiki1, fout)
            wiki1={}

    elif i==109:
        with open('wiki-004.json', 'w+') as fout:
            print("dumping json file ",i)
            json.dump(wiki1, fout)
