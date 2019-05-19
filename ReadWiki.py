import os
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

input_path = 'Source/wiki-pages-text/'
wiki_dict = {}
output_files = []
output_file_name = "Wiki/wiki-"

j = 1
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
            if title not in wiki_dict:
                wiki_dict[title] = {}
            wiki_dict[title][label] = text
    
    if i % 2 == 0:
        with open(output_file_name + str(j) + '.json', 'wb') as o:
            pickle.dump(wiki_dict, o)
            wiki_dict = {}
            j += 1