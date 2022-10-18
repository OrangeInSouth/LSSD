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

files = os.listdir(result_path)

result_file_mode = r"test_(.{2})_en.txt"
# 读取所有结果
result_files = [i for i in files if re.match(result_file_mode, i)]
result_files = sorted(result_files)

bleus = OrderedDict()

for file in result_files:
    # 根据文件名解析出language
    lang = re.match(result_file_mode, file).group(1)

    # 读取文件内容
    f = open(result_path + '/' + file, encoding="utf-8")
    data = f.readlines()
    f.close()
    last_line = data[-1]

    # 解析出BLEU值
    bleu = re.match(r"(.*)BLEU = (.+?) (.*)", last_line).group(2)
    bleus[lang] = bleu


# language list
lang_orders = "fr de zh et ro tr".split()

for lang in lang_orders:
    print("%10s" % lang, end='')

print()

print(("%10s"*len(lang_orders)) % tuple([bleus[lang] for lang in lang_orders]))

print()

print("avg:", sum([float(bleus[lang]) for lang in lang_orders]) / len(bleus))