"""
This script aims to statistic the result of translation in different languages

usage:
    "python result_statistics.py $checkpoint_path diverse"
    or
    "python result_statistics.py $checkpoint_path related"
"""

import sys
import os
import re
from collections import OrderedDict

result_path = sys.argv[1]
# dataset = sys.argv[2]
dataset = "related"

files = os.listdir(result_path)

# 读取所有结果
result_files = [i for i in files if 'test_' in i]
result_files = sorted(result_files)

bleus = OrderedDict()

for file in result_files:
    # 根据文件名解析出language
    lang = re.match(r"test_(.*)_eng.txt", file).group(1)

    # 读取文件内容
    f = open(result_path + '/' + file, encoding="utf-8")
    data = f.readlines()
    f.close()
    last_line = data[-1]

    # 解析出BLEU值
    bleu = re.match(r"(.*)BLEU = (.+?) (.*)", last_line).group(2)
    bleus[lang] = bleu

# lang_orders = "aze	bel	glg	slk	tur	rus	por	ces".split()

# language list
if dataset == "diverse":
    lang_orders = "bos	mar	hin	mkd	ell	bul	fra	kor".split()
elif dataset == "related":
    lang_orders = "aze	bel	glg	slk	tur	rus	por	ces".split()
else:
    print("could not deal with the result of dataset", dataset)
    exit(0)

for lang in lang_orders:
    print("%10s" % lang, end='')

print()

# print("\\t".join([bleus[lang + 'eng'] for lang in lang_orders]))
print(("%10s"*len(lang_orders)) % tuple([bleus[lang] for lang in lang_orders]))

# print("%10s\t" % bleus[lang + 'eng'], end='')
print()

print("avg:", sum([float(bleus[lang]) for lang in lang_orders]) / len(bleus))