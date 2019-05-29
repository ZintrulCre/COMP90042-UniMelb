import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
text_result=json.load(open('text_result.json','r+'))

vectorizer = TfidfVectorizer()


def takeSecond(elem):
    return elem[1]

for ID in text_result:
    entry=text_result[ID]
    if len(entry['evidence'])>3:
        claim = entry['claim']
        evidence = []
        evidence.append(claim)
        for sent in entry['evidence']:
            evidence.append(sent[2])
        print(len(evidence))
        X = vectorizer.fit_transform(evidence)
        query = X.toarray()[0]
        corpus = X.toarray()
        result = []
        for i in range(1, len(corpus)):
            result.append((i - 1, np.dot(query, corpus[i].T)))
        result = sorted(result, key=takeSecond, reverse=True)
        evidence_revised = []
        i = 0
        while i < 3:
            evidence_revised.append(entry['evidence'][result[i][0]])
            i += 1
        entry['evidence'] = evidence_revised

with open('text_result.json','w+') as f:
    json.dump(text_result,f)

