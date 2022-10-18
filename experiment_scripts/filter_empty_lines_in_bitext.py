"""
This script aims to filter empty lines in a parallel corpus.

run example:
python filter_empty_lines_in_bitext.py source_file target_file
"""

import sys

source_file_name = sys.argv[1]
target_file_name = sys.argv[2]

f = open(source_file_name)
source_file = f.readlines()
f.close()

f = open(target_file_name)
target_file = f.readlines()
f.close()

assert len(source_file) == len(target_file), "size mismatch: source file and target file"

filtered_source_file_name = source_file_name + '.filtered'
f_source = open(filtered_source_file_name, 'w+')
filtered_target_file_name = target_file_name + '.filtered'
f_target = open(filtered_target_file_name, 'w+')

empty_lines_count = 0
for source_sentence, target_sentence in zip(source_file, target_file):
    if len(source_sentence.strip()) > 0 and len(target_sentence.strip()) > 0:
        f_source.write(source_sentence)
        f_target.write(target_sentence)
    else:
        empty_lines_count += 1

f_source.close()
f_target.close()

print(f"filtered files are written into: {filtered_source_file_name}, {filtered_target_file_name}")
print(f"{empty_lines_count} empty lines are filtered")