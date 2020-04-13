import json
from collections import defaultdict

import pygtrie

char_to_name = defaultdict(list)
idx_to_name = {}
name_to_idx = {}


char_to_name = pygtrie.CharTrie()

with open("imagenet1000_clsidx_to_labels.txt", "r") as img_classes:
    for line in img_classes:
        line = line.strip("\n")
        class_index, class_label = line.split(":")
        class_index = int(class_index)
        # for now on no capital letters
        class_label = class_label.lower()
        class_label_list = class_label.split(",")
        idx_to_name[class_index] = class_label_list
        for name in class_label_list:
            char_to_name[name] = class_index


# create pickle objects
json.dump(idx_to_name, open("dic_idx_to_label.json", "w"), indent=4)
json.dump(char_to_name.items(), open("trie_char_to_name_to_idx.json", "w"), indent=4)
