"""
This script aims to print language-specific best checkpoints according to LS_valid_loss_history.json
"""

import json
import sys

checkpoint_path = sys.argv[1]

print(f"checkpoint path: {checkpoint_path}")

f = open(checkpoint_path + '/LS_valid_loss_history.json')
LS_valid_loss_history = json.load(f)
f.close()

print(f"{'language pair':30}|{'epoch':10}")
for lang_pair, loss_history in LS_valid_loss_history.items():
    if '-' in lang_pair:
        print(f"{lang_pair:30}|{loss_history.index(min(loss_history)) + 1:10}")

average_loss = LS_valid_loss_history['all']
print(f"{'Average loss':30}|{loss_history.index(min(average_loss)) + 1:10}")