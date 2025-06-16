from io import open
import glob
import os
import unicodedata
import string


all_letters = string.ascii_letters + ".,;'-"
n_letters = len(all_letters) + 1 # +1 for EOS token

def find_files(path): return glob.glob(path)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# read file and split into lines
def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip()) for line in f]
    
category_lines = {}
all_categories = []

for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError("Data not found, make sure the data is downloaded")

# print('# categories:', n_categories, all_categories)
# print(unicode_to_ascii("O'Néàl"))