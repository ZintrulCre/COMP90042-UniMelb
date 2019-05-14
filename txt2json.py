import os
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

input_path = 'Source/wiki-pages-text/'
wiki_dict = defaultdict(dict)

print('Reading files from ' + input_path)
for i in tqdm(range(1, 110)):
    file_name = str(i).zfill(3)
    input_file = input_path + "wiki-" + file_name + '.txt'

    with open(input_file, 'r+') as wiki_file:
        for entry in wiki_file:
            entry = entry.split(' ')
            title = entry[0]
            label = entry[1]
            text = " ".join(entry[2:])
            wiki_dict[title][label] = text


output_file = "wiki.json"
with open(output_file, 'wb') as o:
    print("Dumping results to file " + output_file)
    pickle.dump(wiki_dict, o)
